"""
Seraph MCP — Cache Interface

Defines the abstract interface that all cache backends must implement.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional


class CacheInterface(ABC):
    """
    Abstract base class for cache backends.

    All cache implementations must implement this interface to ensure
    consistent behavior across different backends (memory, Redis, etc.).
    """

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a value from the cache.

        Args:
            key: Cache key

        Returns:
            Cached value if found and not expired, None otherwise
        """
        pass

    @abstractmethod
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Store a value in the cache.

        Args:
            key: Cache key
            value: Value to cache (must be serializable)
            ttl: Time-to-live in seconds (None = use default, 0 = no expiry)

        Returns:
            True if stored successfully, False otherwise
        """
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """
        Delete a key from the cache.

        Args:
            key: Cache key to delete

        Returns:
            True if key was deleted, False if key didn't exist
        """
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """
        Check if a key exists in the cache.

        Args:
            key: Cache key to check

        Returns:
            True if key exists and is not expired, False otherwise
        """
        pass

    @abstractmethod
    async def clear(self) -> bool:
        """
        Clear all entries from the cache.

        Returns:
            True if cache was cleared successfully
        """
        pass

    @abstractmethod
    async def get_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics (hits, misses, size, etc.)
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """
        Close the cache backend and release resources.

        Should be called during graceful shutdown.
        """
        pass

    async def get_many(self, keys: list[str]) -> dict[str, Any]:
        """
        Retrieve multiple values from the cache.

        Default implementation calls get() for each key.
        Backends can override for better performance.

        Args:
            keys: List of cache keys

        Returns:
            Dictionary mapping keys to values (missing keys are omitted)
        """
        result = {}
        for key in keys:
            value = await self.get(key)
            if value is not None:
                result[key] = value
        return result

    async def set_many(
        self,
        items: dict[str, Any],
        ttl: Optional[int] = None,
    ) -> int:
        """
        Store multiple values in the cache.

        Default implementation calls set() for each item.
        Backends can override for better performance.

        Args:
            items: Dictionary mapping keys to values
            ttl: Time-to-live in seconds (applies to all items)

        Returns:
            Number of items successfully stored
        """
        count = 0
        for key, value in items.items():
            if await self.set(key, value, ttl):
                count += 1
        return count

    async def delete_many(self, keys: list[str]) -> int:
        """
        Delete multiple keys from the cache.

        Default implementation calls delete() for each key.
        Backends can override for better performance.

        Args:
            keys: List of cache keys to delete

        Returns:
            Number of keys successfully deleted
        """
        count = 0
        for key in keys:
            if await self.delete(key):
                count += 1
        return count
