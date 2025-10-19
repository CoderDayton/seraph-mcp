"""
Seraph MCP — Memory Cache Backend Tests

Comprehensive test suite for the in-memory cache backend.
Tests LRU eviction, TTL support, thread safety, and all interface methods.

Python 3.12+ with modern async patterns and type hints.
"""

import asyncio
from typing import Any

import pytest

from src.cache.backends.memory import MemoryCacheBackend


class TestMemoryCacheBackend:
    """Test suite for MemoryCacheBackend."""

    @pytest.fixture
    async def cache(self) -> MemoryCacheBackend:
        """Create a fresh memory cache instance for each test."""
        return MemoryCacheBackend(
            max_size=100,
            default_ttl=3600,
            namespace="test",
        )

    async def test_initialization(self) -> None:
        """Test cache initialization with custom parameters."""
        cache = MemoryCacheBackend(
            max_size=100,
            default_ttl=1800,
            namespace="custom",
        )
        assert cache.max_size == 100
        assert cache.default_ttl == 1800
        assert cache.namespace == "custom"

        stats = await cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["size"] == 0

    async def test_set_and_get(self, cache: MemoryCacheBackend) -> None:
        """Test basic set and get operations."""
        # Set a value
        result = await cache.set("key1", "value1")
        assert result is True

        # Get the value
        value = await cache.get("key1")
        assert value == "value1"

        # Stats should reflect the operations
        stats = await cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 0
        assert stats["sets"] == 1

    async def test_get_nonexistent_key(self, cache: MemoryCacheBackend) -> None:
        """Test getting a key that doesn't exist."""
        value = await cache.get("nonexistent")
        assert value is None

        stats = await cache.get_stats()
        assert stats["misses"] == 1
        assert stats["hits"] == 0

    async def test_set_with_various_types(self, cache: MemoryCacheBackend) -> None:
        """Test storing different data types."""
        test_data: dict[str, Any] = {
            "string": "hello",
            "int": 42,
            "float": 3.14,
            "bool": True,
            "none": None,
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
        }

        for key, value in test_data.items():
            await cache.set(key, value)

        for key, expected_value in test_data.items():
            actual_value = await cache.get(key)
            assert actual_value == expected_value

    async def test_delete(self, cache: MemoryCacheBackend) -> None:
        """Test deleting keys."""
        # Set a value
        await cache.set("key1", "value1")
        assert await cache.exists("key1") is True

        # Delete it
        result = await cache.delete("key1")
        assert result is True

        # Verify it's gone
        assert await cache.exists("key1") is False
        assert await cache.get("key1") is None

        # Try to delete non-existent key
        result = await cache.delete("key1")
        assert result is False

    async def test_exists(self, cache: MemoryCacheBackend) -> None:
        """Test checking key existence."""
        # Key doesn't exist initially
        assert await cache.exists("key1") is False

        # Set the key
        await cache.set("key1", "value1")
        assert await cache.exists("key1") is True

        # Delete the key
        await cache.delete("key1")
        assert await cache.exists("key1") is False

    async def test_clear(self, cache: MemoryCacheBackend) -> None:
        """Test clearing all cache entries."""
        # Add multiple entries
        for i in range(5):
            await cache.set(f"key{i}", f"value{i}")

        stats = await cache.get_stats()
        assert stats["size"] == 5

        # Clear the cache
        result = await cache.clear()
        assert result is True

        # Verify all entries are gone
        stats = await cache.get_stats()
        assert stats["size"] == 0

        for i in range(5):
            assert await cache.exists(f"key{i}") is False

    async def test_ttl_expiration(self, cache: MemoryCacheBackend) -> None:
        """Test that entries expire after TTL."""
        # Set a value with 1 second TTL
        await cache.set("key1", "value1", ttl=1)

        # Should exist immediately
        assert await cache.exists("key1") is True
        assert await cache.get("key1") == "value1"

        # Wait for expiration (2.0s to ensure TTL=1s is fully expired)
        await asyncio.sleep(2.0)

        # Should be expired now
        assert await cache.exists("key1") is False
        assert await cache.get("key1") is None

    async def test_ttl_zero_no_expiry(self, cache: MemoryCacheBackend) -> None:
        """Test that TTL=0 means no expiration."""
        await cache.set("key1", "value1", ttl=0)

        # Should exist immediately
        assert await cache.get("key1") == "value1"

        # Should still exist after default TTL would have expired
        await asyncio.sleep(0.1)
        assert await cache.get("key1") == "value1"

    async def test_ttl_none_uses_default(self, cache: MemoryCacheBackend) -> None:
        """Test that TTL=None uses the default TTL."""
        # Create cache with short default TTL
        short_ttl_cache = MemoryCacheBackend(
            max_size=10,
            default_ttl=1,
            namespace="test",
        )

        await short_ttl_cache.set("key1", "value1", ttl=None)
        assert await short_ttl_cache.get("key1") == "value1"

        # Wait for default TTL to expire (2.0s to ensure TTL=1s is fully expired)
        await asyncio.sleep(2.0)
        assert await short_ttl_cache.get("key1") is None

    async def test_lru_eviction(self) -> None:
        """Test LRU eviction when max_size is reached."""
        # Create cache with small max_size for this test
        cache = MemoryCacheBackend(max_size=10, default_ttl=3600, namespace="test")

        # Fill the cache to max_size (10)
        for i in range(10):
            await cache.set(f"key{i}", f"value{i}")

        stats = await cache.get_stats()
        assert stats["size"] == 10
        assert stats["evictions"] == 0

        # Access key0 to make it recently used
        await cache.get("key0")

        # Add one more item (should evict key1, the least recently used)
        await cache.set("key10", "value10")

        stats = await cache.get_stats()
        assert stats["size"] == 10
        assert stats["evictions"] == 1

        # key1 should be evicted, key0 should still exist
        assert await cache.exists("key1") is False
        assert await cache.exists("key0") is True
        assert await cache.exists("key10") is True

    async def test_get_many(self, cache: MemoryCacheBackend) -> None:
        """Test batch get operation."""
        # Set multiple values
        for i in range(5):
            await cache.set(f"key{i}", f"value{i}")

        # Get multiple keys
        result = await cache.get_many(["key0", "key2", "key4", "nonexistent"])

        assert result == {
            "key0": "value0",
            "key2": "value2",
            "key4": "value4",
        }
        assert "nonexistent" not in result

    async def test_set_many(self, cache: MemoryCacheBackend) -> None:
        """Test batch set operation."""
        items = {
            "key1": "value1",
            "key2": "value2",
            "key3": "value3",
        }

        count = await cache.set_many(items)
        assert count == 3

        # Verify all items were set
        for key, value in items.items():
            assert await cache.get(key) == value

    async def test_set_many_with_ttl(self, cache: MemoryCacheBackend) -> None:
        """Test batch set with TTL."""
        items = {
            "key1": "value1",
            "key2": "value2",
        }

        await cache.set_many(items, ttl=1)

        # Should exist immediately
        assert await cache.get("key1") == "value1"
        assert await cache.get("key2") == "value2"

        # Wait for expiration (2.0s to ensure TTL=1s is fully expired)
        await asyncio.sleep(2.0)

        # Should be expired
        assert await cache.get("key1") is None
        assert await cache.get("key2") is None

    async def test_delete_many(self, cache: MemoryCacheBackend) -> None:
        """Test batch delete operation."""
        # Set multiple values
        for i in range(5):
            await cache.set(f"key{i}", f"value{i}")

        # Delete multiple keys
        count = await cache.delete_many(["key0", "key2", "key4", "nonexistent"])

        # Should delete 3 keys (nonexistent doesn't count)
        assert count == 3

        # Verify correct keys were deleted
        assert await cache.exists("key0") is False
        assert await cache.exists("key1") is True
        assert await cache.exists("key2") is False
        assert await cache.exists("key3") is True
        assert await cache.exists("key4") is False

    async def test_get_stats(self, cache: MemoryCacheBackend) -> None:
        """Test statistics tracking."""
        # Initial stats
        stats = await cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["sets"] == 0
        assert stats["deletes"] == 0
        assert stats["evictions"] == 0
        assert stats["size"] == 0

        # Perform various operations
        await cache.set("key1", "value1")
        await cache.get("key1")  # hit
        await cache.get("key2")  # miss
        await cache.delete("key1")

        stats = await cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["sets"] == 1
        assert stats["deletes"] == 1
        assert stats["size"] == 0

    async def test_namespace_isolation(self) -> None:
        """Test that different namespaces are isolated."""
        cache1 = MemoryCacheBackend(namespace="ns1")
        cache2 = MemoryCacheBackend(namespace="ns2")

        await cache1.set("key1", "value1")
        await cache2.set("key1", "value2")

        # Each cache should have its own value
        assert await cache1.get("key1") == "value1"
        assert await cache2.get("key1") == "value2"

    async def test_close(self, cache: MemoryCacheBackend) -> None:
        """Test cache close operation."""
        await cache.set("key1", "value1")

        # Close should succeed
        await cache.close()

        # Cache should still work after close (memory backend doesn't need cleanup)
        assert await cache.get("key1") == "value1"

    async def test_concurrent_operations(self, cache: MemoryCacheBackend) -> None:
        """Test thread safety with concurrent operations."""

        async def set_values(start: int, end: int) -> None:
            for i in range(start, end):
                await cache.set(f"key{i}", f"value{i}")

        # Run concurrent set operations
        await asyncio.gather(
            set_values(0, 5),
            set_values(5, 10),
            set_values(10, 15),
        )

        # Verify all values were set correctly
        for i in range(15):
            value = await cache.get(f"key{i}")
            assert value == f"value{i}"

    async def test_overwrite_existing_key(self, cache: MemoryCacheBackend) -> None:
        """Test that setting an existing key overwrites the value."""
        await cache.set("key1", "value1")
        assert await cache.get("key1") == "value1"

        await cache.set("key1", "value2")
        assert await cache.get("key1") == "value2"

        # Should only count as 2 sets, not 3
        stats = await cache.get_stats()
        assert stats["sets"] == 2

    async def test_empty_cache_operations(self, cache: MemoryCacheBackend) -> None:
        """Test operations on empty cache."""
        # Clear empty cache
        assert await cache.clear() is True

        # Get from empty cache
        assert await cache.get("key1") is None

        # Delete from empty cache
        assert await cache.delete("key1") is False

        # Exists on empty cache
        assert await cache.exists("key1") is False

        # Get many from empty cache
        assert await cache.get_many(["key1", "key2"]) == {}

        # Delete many from empty cache
        assert await cache.delete_many(["key1", "key2"]) == 0
