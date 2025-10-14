"""
Seraph MCP — Redis Cache Backend

Asynchronous Redis cache implementation with:
- JSON serialization for values
- Per-key TTL support
- Namespace prefixing for safe multi-tenant usage
- Batch operations using Redis pipelines (mget, set/delete in batches)

Requires: redis>=4.2 with asyncio support

Example:
    cache = RedisCacheBackend(redis_url="redis://localhost:6379", namespace="seraph", default_ttl=3600)
    await cache.set("greeting", {"msg": "hello"}, ttl=60)
    val = await cache.get("greeting")
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from ..interface import CacheInterface

logger = logging.getLogger(__name__)

try:
    # redis-py asyncio client (v4+)
    from redis.asyncio import Redis
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "Redis async client is required but not installed. "
        "Install with: pip install 'redis>=5.0.0' or add 'redis' to your dependencies."
    ) from e


class RedisCacheBackend(CacheInterface):
    """
    Redis cache backend with JSON serialization and TTL.

    Notes:
    - Keys are prefixed with the configured namespace to avoid collisions.
    - Values are stored as UTF-8 JSON strings.
    - TTL is applied via Redis EX seconds (None -> default_ttl, 0 -> no expiry).
    - Batch operations use pipelining to reduce round-trips.
    """

    def __init__(
        self,
        redis_url: str,
        namespace: str = "seraph",
        default_ttl: int = 3600,
        max_connections: int = 10,
        socket_timeout: int = 5,
        decode_responses: bool = True,
    ) -> None:
        """
        Initialize Redis cache backend.

        Args:
            redis_url: Connection URL, e.g., redis://localhost:6379/0 or rediss:// for TLS
            namespace: Prefix for all keys (e.g., "seraph")
            default_ttl: Default TTL in seconds (0 => no expiry)
            max_connections: Connection pool size
            socket_timeout: Socket timeout in seconds
            decode_responses: If True, values returned as str, not bytes
        """
        if not redis_url:
            raise ValueError("redis_url is required")

        self.namespace = namespace.strip() or "seraph"
        self.default_ttl = max(0, int(default_ttl))
        self._hits = 0
        self._misses = 0
        self._sets = 0
        self._deletes = 0

        # Create Redis client (lazy connection; connects on first command)
        # Type annotation accounts for decode_responses setting (Union type from from_url)
        self._client = Redis.from_url(  # type: ignore[call-overload]
            url=redis_url,
            decode_responses=decode_responses,
            max_connections=max_connections,
            socket_timeout=socket_timeout,
        )

        # Lock for any client-protected sequences if needed
        self._lock = asyncio.Lock()

    # ------------ Helpers ------------

    def _make_key(self, key: str) -> str:
        """Create namespaced key."""
        return f"{self.namespace}:{key}"

    @staticmethod
    def _to_json(value: Any) -> str:
        """Serialize value to JSON string."""
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _from_json(data: str | bytes | None) -> Any | None:
        """Deserialize JSON string to Python object. Returns None if data is None."""
        if data is None:
            return None
        # Handle both str and bytes for flexibility
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        try:
            return json.loads(data)
        except (ValueError, UnicodeDecodeError) as e:
            # If not valid JSON, log warning and return raw data as-is
            logger.warning(
                f"Failed to decode JSON from cache, returning raw data: {e}",
                extra={"data_preview": data[:100] if len(data) > 100 else data, "error": str(e)},
            )
            return data
        except Exception as e:
            logger.error(f"Unexpected error deserializing cache data: {e}", exc_info=True)
            return data

    def _ttl_seconds(self, ttl: int | None) -> int | None:
        """
        Normalize TTL:
        - None -> default_ttl
        - 0 or negative -> no expiry (return None)
        - positive -> provided ttl
        """
        if ttl is None:
            ttl = self.default_ttl
        ttl = int(ttl)
        return ttl if ttl > 0 else None

    # ------------ Core Interface ------------

    async def get(self, key: str) -> Any | None:
        """Retrieve a value by key."""
        try:
            ns_key = self._make_key(key)
            data = await self._client.get(ns_key)
            if data is None:
                self._misses += 1
                return None

            self._hits += 1
            return self._from_json(data)
        except Exception as e:
            logger.error(
                f"Failed to get key '{key}' from Redis: {e}",
                extra={"key": key, "namespace": self.namespace, "error": str(e)},
                exc_info=True,
            )
            self._misses += 1
            return None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Store a value with optional TTL."""
        try:
            ns_key = self._make_key(key)
            ex = self._ttl_seconds(ttl)
            payload = self._to_json(value)
            # redis-py returns True or 'OK' depending on decode_responses
            res = await self._client.set(name=ns_key, value=payload, ex=ex)
            success = bool(res)  # True or 'OK'
            if success:
                self._sets += 1
            return success
        except (TypeError, ValueError) as e:
            logger.error(
                f"Failed to serialize value for key '{key}': {e}",
                extra={"key": key, "value_type": type(value).__name__, "error": str(e)},
                exc_info=True,
            )
            return False
        except Exception as e:
            logger.error(
                f"Failed to set key '{key}' in Redis: {e}",
                extra={"key": key, "namespace": self.namespace, "ttl": ttl, "error": str(e)},
                exc_info=True,
            )
            return False

    async def delete(self, key: str) -> bool:
        """Delete a single key."""
        try:
            ns_key = self._make_key(key)
            deleted = await self._client.delete(ns_key)
            if deleted:
                self._deletes += 1
            return bool(deleted)
        except Exception as e:
            logger.error(
                f"Failed to delete key '{key}' from Redis: {e}",
                extra={"key": key, "namespace": self.namespace, "error": str(e)},
                exc_info=True,
            )
            return False

    async def exists(self, key: str) -> bool:
        """Check if a key exists."""
        try:
            ns_key = self._make_key(key)
            return bool(await self._client.exists(ns_key))
        except Exception as e:
            logger.error(
                f"Failed to check existence of key '{key}' in Redis: {e}",
                extra={"key": key, "namespace": self.namespace, "error": str(e)},
                exc_info=True,
            )
            return False

    async def clear(self) -> bool:
        """
        Clear all entries under the namespace.

        Implementation: SCAN match "<namespace>:*" and DEL in batches.
        """
        try:
            pattern = f"{self.namespace}:*"
            cursor = 0
            total_deleted = 0
            # Use batches to avoid large single DEL calls
            batch_size = 1000

            while True:
                cursor, keys = await self._client.scan(cursor=cursor, match=pattern, count=batch_size)
                if keys:
                    # DEL supports multiple keys, but keep batches reasonable
                    # Note: delete returns number of keys removed
                    total_deleted += await self._client.delete(*keys)
                if cursor == 0:
                    break

            self._deletes += total_deleted
            logger.info(f"Cleared {total_deleted} keys from namespace '{self.namespace}'")
            return True
        except Exception as e:
            logger.error(
                f"Failed to clear cache for namespace '{self.namespace}': {e}",
                extra={"namespace": self.namespace, "error": str(e)},
                exc_info=True,
            )
            return False

    async def get_stats(self) -> dict[str, Any]:
        """Return cache statistics and basic Redis info."""
        stats: dict[str, Any] = {
            "backend": "redis",
            "namespace": self.namespace,
            "default_ttl": self.default_ttl,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": 0.0,
            "sets": self._sets,
            "deletes": self._deletes,
            "connected": False,
        }

        total_requests = self._hits + self._misses
        stats["hit_rate"] = round((self._hits / total_requests) * 100, 2) if total_requests else 0.0

        try:
            # PING to check connectivity
            pong = await self._client.ping()
            stats["connected"] = bool(pong)

            # Fetch minimal INFO for insight (server + keyspace)
            info = await self._client.info(section="server")
            keyspace = await self._client.info(section="keyspace")

            stats["redis_version"] = info.get("redis_version")
            stats["redis_mode"] = info.get("redis_mode")
            stats["os"] = info.get("os")
            # Keyspace counters (approx)
            # DB-specific stats are under 'db0', 'db1', etc.; not filtered by namespace.
            # Provide raw info so callers can inspect if needed.
            stats["keyspace"] = keyspace
        except Exception as e:
            # If INFO is restricted or fails, keep minimal stats
            logger.warning(f"Failed to get Redis INFO (restricted or unavailable): {e}", extra={"error": str(e)})

        return stats

    async def close(self) -> None:
        """Close the Redis client and release resources."""
        try:
            await self._client.aclose()
            logger.info(f"Closed Redis cache backend for namespace '{self.namespace}'")
        except Exception as e:
            logger.error(
                f"Error closing Redis client: {e}", extra={"namespace": self.namespace, "error": str(e)}, exc_info=True
            )
        finally:
            # Ensure pool disconnect
            try:
                await self._client.connection_pool.disconnect()
            except Exception as e:
                logger.warning(f"Error disconnecting Redis connection pool: {e}", extra={"error": str(e)})

    # ------------ Batch operations (pipeline) ------------

    async def get_many(self, keys: list[str]) -> dict[str, Any]:
        """
        Retrieve multiple values in one round-trip using MGET.
        Missing keys are omitted from the result.
        """
        if not keys:
            return {}

        try:
            ns_keys = [self._make_key(k) for k in keys]
            values = await self._client.mget(ns_keys)

            result: dict[str, Any] = {}
            # mget preserves order
            for k, raw in zip(keys, values, strict=False):
                if raw is None:
                    self._misses += 1
                    continue
                self._hits += 1
                result[k] = self._from_json(raw)

            return result
        except Exception as e:
            logger.error(
                f"Failed to get multiple keys from Redis: {e}",
                extra={"key_count": len(keys), "namespace": self.namespace, "error": str(e)},
                exc_info=True,
            )
            return {}

    async def set_many(self, items: dict[str, Any], ttl: int | None = None) -> int:
        """
        Store multiple values using a pipeline. Applies the same TTL to all items.
        Returns number of items successfully stored.
        """
        if not items:
            return 0

        try:
            ex = self._ttl_seconds(ttl)
            pipe = self._client.pipeline(transaction=False)

            for key, value in items.items():
                ns_key = self._make_key(key)
                payload = self._to_json(value)
                pipe.set(ns_key, payload, ex=ex)

            results = await pipe.execute()
            # Results are ["OK" | True | 1 | None...] depending on server/config
            success_count = sum(1 for r in results if r in (True, "OK", b"OK"))
            self._sets += success_count
            return success_count
        except Exception as e:
            logger.error(
                f"Failed to set multiple keys in Redis: {e}",
                extra={"key_count": len(items), "namespace": self.namespace, "ttl": ttl, "error": str(e)},
                exc_info=True,
            )
            return 0

    async def delete_many(self, keys: list[str]) -> int:
        """
        Delete multiple keys using a pipeline.
        Returns number of keys successfully deleted.
        """
        if not keys:
            return 0

        try:
            ns_keys = [self._make_key(k) for k in keys]
            # Redis DEL variadic returns count; pipeline may return counts per op if separate
            # Prefer a single DEL for all keys when reasonable; if too large, chunk.
            deleted_total = 0
            chunk_size = 1000

            for i in range(0, len(ns_keys), chunk_size):
                chunk = ns_keys[i : i + chunk_size]
                # A single DEL for the chunk
                deleted_total += int(await self._client.delete(*chunk))

            self._deletes += deleted_total
            return deleted_total
        except Exception as e:
            logger.error(
                f"Failed to delete multiple keys from Redis: {e}",
                extra={"key_count": len(keys), "namespace": self.namespace, "error": str(e)},
                exc_info=True,
            )
            return 0
