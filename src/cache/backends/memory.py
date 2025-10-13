"""
Seraph MCP — Memory Cache Backend

In-memory cache implementation with LRU eviction and TTL support.
Thread-safe and suitable for single-process deployments.
"""

import asyncio
import time
from collections import OrderedDict
from typing import Any

from ..interface import CacheInterface


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
        self.default_ttl = max(0, int(default_ttl))  # Convert negative to 0
        self.namespace = namespace

        # Cache storage: key -> {"value": ..., "expires_at": ...}
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()

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

    def _is_expired(self, entry: dict[str, Any]) -> bool:
        """Check if entry is expired."""
        expiry = entry.get("expires_at")
        if expiry is None:
            return False
        return time.time() > expiry

    async def get(self, key: str) -> Any | None:
        """Retrieve value from cache."""
        async with self._lock:
            cache_key = self._make_key(key)

            if cache_key not in self._cache:
                self._misses += 1
                return None

            entry = self._cache[cache_key]

            # Check expiry
            if self._is_expired(entry):
                # Remove expired entry
                del self._cache[cache_key]
                self._misses += 1
                return None

            # Move to end (mark as recently used)
            self._cache.move_to_end(cache_key)
            self._hits += 1

            return entry["value"]

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
    ) -> bool:
        """Store value in cache."""
        async with self._lock:
            cache_key = self._make_key(key)

            # Calculate expiry time
            ttl_seconds = self._ttl_seconds(ttl)
            
            if ttl_seconds is not None and ttl_seconds > 0:
                expiry = time.time() + ttl_seconds
            else:
                expiry = None  # No expiry

            # Evict if at capacity and key is new
            if cache_key not in self._cache and len(self._cache) >= self.max_size:
                # Remove oldest entry (LRU)
                self._cache.popitem(last=False)
                self._evictions += 1

            # Store value
            self._cache[cache_key] = {"value": value, "expires_at": expiry}
            self._cache.move_to_end(cache_key)
            self._sets += 1

            return True

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        async with self._lock:
            cache_key = self._make_key(key)

            if cache_key in self._cache:
                del self._cache[cache_key]
                self._deletes += 1
                return True

            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        async with self._lock:
            cache_key = self._make_key(key)

            if cache_key not in self._cache:
                return False

            entry = self._cache[cache_key]

            if self._is_expired(entry):
                # Remove expired entry
                del self._cache[cache_key]
                return False

            return True

    async def clear(self) -> bool:
        """Clear all entries from cache."""
        async with self._lock:
            self._cache.clear()
            return True

    async def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        async with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0.0

            return {
                "backend": "memory",
                "size": len(self._cache),
                "max_size": self.max_size,
                "default_ttl": self.default_ttl,
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
        async with self._lock:
            self._cache.clear()

    async def get_many(self, keys: list[str]) -> dict[str, Any]:
        """Retrieve multiple values efficiently."""
        async with self._lock:
            result = {}

            for key in keys:
                cache_key = self._make_key(key)

                if cache_key in self._cache:
                    entry = self._cache[cache_key]

                    if not self._is_expired(entry):
                        result[key] = entry["value"]
                        self._cache.move_to_end(cache_key)
                        self._hits += 1
                    else:
                        del self._cache[cache_key]
                        self._misses += 1
                else:
                    self._misses += 1

            return result

    async def set_many(
        self,
        items: dict[str, Any],
        ttl: int | None = None,
    ) -> int:
        """Store multiple values efficiently."""
        async with self._lock:
            count = 0

            # Calculate expiry once
            ttl_seconds = self._ttl_seconds(ttl)
            expiry = time.time() + ttl_seconds if ttl_seconds and ttl_seconds > 0 else None

            for key, value in items.items():
                cache_key = self._make_key(key)

                # Evict if needed
                if cache_key not in self._cache and len(self._cache) >= self.max_size:
                    self._cache.popitem(last=False)
                    self._evictions += 1

                self._cache[cache_key] = {"value": value, "expires_at": expiry}
                self._cache.move_to_end(cache_key)
                self._sets += 1
                count += 1

            return count

    async def delete_many(self, keys: list[str]) -> int:
        """Delete multiple keys efficiently."""
        async with self._lock:
            count = 0

            for key in keys:
                cache_key = self._make_key(key)

                if cache_key in self._cache:
                    del self._cache[cache_key]
                    self._deletes += 1
                    count += 1

            return count
