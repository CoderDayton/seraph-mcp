"""
Seraph MCP — Memory Cache Backend

In-memory cache implementation with LRU eviction and TTL support.
Thread-safe and suitable for single-process deployments.
"""

import asyncio
import logging
import time
from collections import OrderedDict
from typing import Any

from ..interface import CacheInterface

logger = logging.getLogger(__name__)


class MemoryCacheBackend(CacheInterface):
    """
    In-memory cache backend with LRU eviction.

    Features:
    - LRU eviction when max_size is reached
    - Per-key TTL support
    - Thread-safe operations
    - O(1) get/set/delete operations
    """

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: int = 3600,
        namespace: str = "seraph",
    ):
        """
        Initialize memory cache backend.

        Args:
            max_size: Maximum number of entries (LRU eviction when exceeded)
            default_ttl: Default TTL in seconds (0 = no expiry)
            namespace: Cache key namespace/prefix
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.namespace = namespace

        # Cache storage: key -> (value, expiry_time)
        self._cache: OrderedDict[str, tuple[Any, float | None]] = OrderedDict()

        # Stats
        self._hits = 0
        self._misses = 0
        self._sets = 0
        self._deletes = 0
        self._evictions = 0

        # Lock for thread safety
        self._lock = asyncio.Lock()

    def _make_key(self, key: str) -> str:
        """Create namespaced cache key."""
        return f"{self.namespace}:{key}"

    def _is_expired(self, expiry: float | None) -> bool:
        """Check if entry is expired."""
        if expiry is None:
            return False
        return time.time() > expiry

    async def get(self, key: str) -> Any | None:
        """Retrieve value from cache."""
        if not key:
            logger.warning("Attempted to get cache value with empty key")
            return None

        try:
            async with self._lock:
                cache_key = self._make_key(key)

                if cache_key not in self._cache:
                    self._misses += 1
                    return None

                value, expiry = self._cache[cache_key]

                # Check expiry
                if self._is_expired(expiry):
                    # Remove expired entry
                    del self._cache[cache_key]
                    self._misses += 1
                    return None

                # Move to end (mark as recently used)
                self._cache.move_to_end(cache_key)
                self._hits += 1

                return value
        except Exception as e:
            logger.error(
                f"Unexpected error getting key '{key}' from memory cache: {e}",
                extra={"key": key, "namespace": self.namespace, "error": str(e)},
                exc_info=True,
            )
            self._misses += 1
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
    ) -> bool:
        """Store value in cache."""
        if not key:
            logger.warning("Attempted to set cache value with empty key")
            return False

        try:
            async with self._lock:
                cache_key = self._make_key(key)

                # Calculate expiry time
                if ttl is None:
                    ttl = self.default_ttl

                if ttl > 0:
                    expiry = time.time() + ttl
                else:
                    expiry = None  # No expiry

                # Evict if at capacity and key is new
                if cache_key not in self._cache and len(self._cache) >= self.max_size:
                    # Remove oldest entry (LRU)
                    evicted_key, _ = self._cache.popitem(last=False)
                    self._evictions += 1
                    logger.debug(f"Evicted key from memory cache: {evicted_key}")

                # Store value
                self._cache[cache_key] = (value, expiry)
                self._cache.move_to_end(cache_key)
                self._sets += 1

                return True
        except Exception as e:
            logger.error(
                f"Unexpected error setting key '{key}' in memory cache: {e}",
                extra={"key": key, "namespace": self.namespace, "ttl": ttl, "error": str(e)},
                exc_info=True,
            )
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if not key:
            logger.warning("Attempted to delete cache value with empty key")
            return False

        try:
            async with self._lock:
                cache_key = self._make_key(key)

                if cache_key in self._cache:
                    del self._cache[cache_key]
                    self._deletes += 1
                    return True

                return False
        except Exception as e:
            logger.error(
                f"Unexpected error deleting key '{key}' from memory cache: {e}",
                extra={"key": key, "namespace": self.namespace, "error": str(e)},
                exc_info=True,
            )
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        if not key:
            return False

        try:
            async with self._lock:
                cache_key = self._make_key(key)

                if cache_key not in self._cache:
                    return False

                _, expiry = self._cache[cache_key]

                if self._is_expired(expiry):
                    # Remove expired entry
                    del self._cache[cache_key]
                    return False

                return True
        except Exception as e:
            logger.error(
                f"Unexpected error checking existence of key '{key}' in memory cache: {e}",
                extra={"key": key, "namespace": self.namespace, "error": str(e)},
                exc_info=True,
            )
            return False

    async def clear(self) -> bool:
        """Clear all entries from cache."""
        try:
            async with self._lock:
                size = len(self._cache)
                self._cache.clear()
                logger.info(f"Cleared {size} entries from memory cache namespace '{self.namespace}'")
                return True
        except Exception as e:
            logger.error(
                f"Unexpected error clearing memory cache: {e}",
                extra={"namespace": self.namespace, "error": str(e)},
                exc_info=True,
            )
            return False

    async def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        async with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0.0

            return {
                "backend": "memory",
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(hit_rate, 2),
                "sets": self._sets,
                "deletes": self._deletes,
                "evictions": self._evictions,
                "namespace": self.namespace,
            }

    async def close(self) -> None:
        """Close cache and release resources."""
        # Memory backend doesn't need cleanup - data persists in-process
        logger.debug(f"Memory cache backend closed for namespace '{self.namespace}'")

    async def get_many(self, keys: list[str]) -> dict[str, Any]:
        """Retrieve multiple values efficiently."""
        if not keys:
            return {}

        try:
            async with self._lock:
                result = {}

                for key in keys:
                    if not key:
                        continue

                    cache_key = self._make_key(key)

                    if cache_key in self._cache:
                        value, expiry = self._cache[cache_key]

                        if not self._is_expired(expiry):
                            result[key] = value
                            self._cache.move_to_end(cache_key)
                            self._hits += 1
                        else:
                            del self._cache[cache_key]
                            self._misses += 1
                    else:
                        self._misses += 1

                return result
        except Exception as e:
            logger.error(
                f"Unexpected error getting multiple keys from memory cache: {e}",
                extra={"key_count": len(keys), "namespace": self.namespace, "error": str(e)},
                exc_info=True,
            )
            return {}

    async def set_many(
        self,
        items: dict[str, Any],
        ttl: int | None = None,
    ) -> int:
        """Store multiple values efficiently."""
        if not items:
            return 0

        try:
            async with self._lock:
                count = 0

                # Calculate expiry once
                if ttl is None:
                    ttl = self.default_ttl

                expiry = time.time() + ttl if ttl > 0 else None

                for key, value in items.items():
                    if not key:
                        continue

                    cache_key = self._make_key(key)

                    # Evict if needed
                    if cache_key not in self._cache and len(self._cache) >= self.max_size:
                        self._cache.popitem(last=False)
                        self._evictions += 1

                    self._cache[cache_key] = (value, expiry)
                    self._cache.move_to_end(cache_key)
                    self._sets += 1
                    count += 1

                return count
        except Exception as e:
            logger.error(
                f"Unexpected error setting multiple keys in memory cache: {e}",
                extra={"key_count": len(items), "namespace": self.namespace, "ttl": ttl, "error": str(e)},
                exc_info=True,
            )
            return 0

    async def delete_many(self, keys: list[str]) -> int:
        """Delete multiple keys efficiently."""
        if not keys:
            return 0

        try:
            async with self._lock:
                count = 0

                for key in keys:
                    if not key:
                        continue

                    cache_key = self._make_key(key)

                    if cache_key in self._cache:
                        del self._cache[cache_key]
                        self._deletes += 1
                        count += 1

                return count
        except Exception as e:
            logger.error(
                f"Unexpected error deleting multiple keys from memory cache: {e}",
                extra={"key_count": len(keys), "namespace": self.namespace, "error": str(e)},
                exc_info=True,
            )
            return 0
