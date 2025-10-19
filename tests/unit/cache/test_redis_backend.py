"""
Seraph MCP — Redis Cache Backend Tests

Comprehensive test suite for the Redis cache backend.
Tests all interface methods, TTL handling, namespace isolation, and error conditions.

Python 3.12+ with modern async patterns and type hints.
Requires Redis server running on localhost:6379 (or TEST_REDIS_URL env var).
"""

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

import pytest

from src.cache.backends.redis import RedisCacheBackend

# Check if Redis is available
try:
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    redis_available = sock.connect_ex(("localhost", 6379)) == 0
    sock.close()
except Exception:
    redis_available = False

pytestmark = pytest.mark.skipif(not redis_available, reason="Redis server not available")


class TestRedisCacheBackend:
    """Test suite for RedisCacheBackend."""

    @pytest.fixture
    async def cache(self, test_redis_url: str) -> AsyncGenerator[RedisCacheBackend, None]:
        """Create a fresh Redis cache instance for each test."""
        cache = RedisCacheBackend(
            redis_url=test_redis_url,
            namespace="test",
            default_ttl=3600,
            max_connections=5,
            socket_timeout=2,
        )
        # Clear any existing data
        await cache.clear()
        yield cache
        # Cleanup after test
        await cache.clear()
        await cache.close()

    async def test_initialization(self, test_redis_url: str) -> None:
        """Test cache initialization with custom parameters."""
        cache = RedisCacheBackend(
            redis_url=test_redis_url,
            namespace="custom",
            default_ttl=1800,
            max_connections=10,
            socket_timeout=5,
        )

        assert cache.namespace == "custom"
        assert cache.default_ttl == 1800

        # Verify connection works
        stats = await cache.get_stats()
        assert stats["backend"] == "redis"
        assert stats["namespace"] == "custom"

        await cache.close()

    async def test_set_and_get(self, cache: RedisCacheBackend) -> None:
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

    async def test_get_nonexistent_key(self, cache: RedisCacheBackend) -> None:
        """Test getting a key that doesn't exist."""
        value = await cache.get("nonexistent")
        assert value is None

        stats = await cache.get_stats()
        assert stats["misses"] == 1
        assert stats["hits"] == 0

    async def test_set_with_various_types(self, cache: RedisCacheBackend) -> None:
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

    async def test_delete(self, cache: RedisCacheBackend) -> None:
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

    async def test_exists(self, cache: RedisCacheBackend) -> None:
        """Test checking key existence."""
        # Key doesn't exist initially
        assert await cache.exists("key1") is False

        # Set the key
        await cache.set("key1", "value1")
        assert await cache.exists("key1") is True

        # Delete the key
        await cache.delete("key1")
        assert await cache.exists("key1") is False

    async def test_clear(self, cache: RedisCacheBackend) -> None:
        """Test clearing all cache entries in namespace."""
        # Add multiple entries
        for i in range(5):
            await cache.set(f"key{i}", f"value{i}")

        # Verify entries exist
        for i in range(5):
            assert await cache.exists(f"key{i}") is True

        # Clear the cache
        result = await cache.clear()
        assert result is True

        # Verify all entries are gone
        for i in range(5):
            assert await cache.exists(f"key{i}") is False

    async def test_ttl_expiration(self, cache: RedisCacheBackend) -> None:
        """Test that entries expire after TTL."""
        # Set a value with 1 second TTL
        await cache.set("key1", "value1", ttl=1)

        # Should exist immediately
        assert await cache.exists("key1") is True
        assert await cache.get("key1") == "value1"

        # Wait for expiration
        await asyncio.sleep(1.1)

        # Should be expired now
        assert await cache.exists("key1") is False
        assert await cache.get("key1") is None

    async def test_ttl_zero_no_expiry(self, cache: RedisCacheBackend) -> None:
        """Test that TTL=0 means no expiration."""
        await cache.set("key1", "value1", ttl=0)

        # Should exist immediately
        assert await cache.get("key1") == "value1"

        # Should still exist after some time
        await asyncio.sleep(0.2)
        assert await cache.get("key1") == "value1"

    async def test_ttl_none_uses_default(self, cache: RedisCacheBackend, test_redis_url: str) -> None:
        """Test that TTL=None uses the default TTL."""
        # Create cache with short default TTL
        short_ttl_cache = RedisCacheBackend(
            redis_url=test_redis_url,
            namespace="test_ttl",
            default_ttl=1,
        )

        await short_ttl_cache.set("key1", "value1", ttl=None)
        assert await short_ttl_cache.get("key1") == "value1"

        # Wait for default TTL to expire
        await asyncio.sleep(1.1)
        assert await short_ttl_cache.get("key1") is None

        await short_ttl_cache.close()

    async def test_get_many(self, cache: RedisCacheBackend) -> None:
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

    async def test_set_many(self, cache: RedisCacheBackend) -> None:
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

    async def test_set_many_with_ttl(self, cache: RedisCacheBackend) -> None:
        """Test batch set with TTL."""
        items = {
            "key1": "value1",
            "key2": "value2",
        }

        await cache.set_many(items, ttl=1)

        # Should exist immediately
        assert await cache.get("key1") == "value1"
        assert await cache.get("key2") == "value2"

        # Wait for expiration
        await asyncio.sleep(1.1)

        # Should be expired
        assert await cache.get("key1") is None
        assert await cache.get("key2") is None

    async def test_delete_many(self, cache: RedisCacheBackend) -> None:
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

    async def test_get_stats(self, cache: RedisCacheBackend) -> None:
        """Test statistics tracking."""
        # Initial stats
        stats = await cache.get_stats()
        assert stats["backend"] == "redis"
        assert stats["namespace"] == "test"
        assert stats["hits"] == 0
        assert stats["misses"] == 0

        # Perform various operations
        await cache.set("key1", "value1")
        await cache.get("key1")  # hit
        await cache.get("key2")  # miss

        stats = await cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1

    async def test_namespace_isolation(self, test_redis_url: str) -> None:
        """Test that different namespaces are isolated."""
        cache1 = RedisCacheBackend(redis_url=test_redis_url, namespace="ns1")
        cache2 = RedisCacheBackend(redis_url=test_redis_url, namespace="ns2")

        await cache1.set("key1", "value1")
        await cache2.set("key1", "value2")

        # Each cache should have its own value
        assert await cache1.get("key1") == "value1"
        assert await cache2.get("key1") == "value2"

        # Clear one namespace shouldn't affect the other
        await cache1.clear()
        assert await cache1.get("key1") is None
        assert await cache2.get("key1") == "value2"

        await cache1.close()
        await cache2.close()

    async def test_close(self, cache: RedisCacheBackend) -> None:
        """Test cache close operation."""
        await cache.set("key1", "value1")

        # Close should succeed
        await cache.close()

        # After close, operations should fail gracefully or reconnect
        # (depends on implementation)

    async def test_concurrent_operations(self, cache: RedisCacheBackend) -> None:
        """Test concurrent operations."""

        async def set_values(start: int, end: int) -> None:
            for i in range(start, end):
                await cache.set(f"key{i}", f"value{i}")

        # Run concurrent set operations
        await asyncio.gather(
            set_values(0, 10),
            set_values(10, 20),
            set_values(20, 30),
        )

        # Verify all values were set correctly
        for i in range(30):
            value = await cache.get(f"key{i}")
            assert value == f"value{i}"

    async def test_overwrite_existing_key(self, cache: RedisCacheBackend) -> None:
        """Test that setting an existing key overwrites the value."""
        await cache.set("key1", "value1")
        assert await cache.get("key1") == "value1"

        await cache.set("key1", "value2")
        assert await cache.get("key1") == "value2"

    async def test_complex_data_structures(self, cache: RedisCacheBackend) -> None:
        """Test storing complex nested data structures."""
        complex_data = {
            "user": {
                "id": 123,
                "name": "Alice",
                "tags": ["python", "redis", "async"],
            },
            "metadata": {
                "created": "2025-01-01",
                "active": True,
                "score": 95.5,
            },
            "items": [
                {"id": 1, "name": "Item 1"},
                {"id": 2, "name": "Item 2"},
            ],
        }

        await cache.set("complex", complex_data)
        result = await cache.get("complex")
        assert result == complex_data

    async def test_unicode_data(self, cache: RedisCacheBackend) -> None:
        """Test handling of Unicode data."""
        unicode_data = {
            "greeting": "Hello 世界 🌍",
            "emoji": "🎉🎊🎈",
            "special": "Ñoño café",
        }

        await cache.set("unicode", unicode_data)
        result = await cache.get("unicode")
        assert result == unicode_data

    async def test_empty_cache_operations(self, cache: RedisCacheBackend) -> None:
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

    async def test_connection_error_handling(self, test_redis_url: str) -> None:
        """Test handling of connection errors."""
        # Create cache with invalid URL
        cache = RedisCacheBackend(
            redis_url="redis://invalid-host:9999/0",
            namespace="test",
            socket_timeout=1,
        )

        # Operations should handle errors gracefully
        with pytest.raises(Exception):  # noqa: B017, PT011
            await cache.set("key1", "value1")

    async def test_large_values(self, cache: RedisCacheBackend) -> None:
        """Test storing and retrieving large values."""
        # Create a large value (~100KB of data)
        large_value = {"data": "x" * 100000}

        await cache.set("large", large_value)
        result = await cache.get("large")
        assert result == large_value

    async def test_many_keys(self, cache: RedisCacheBackend) -> None:
        """Test handling many keys."""
        # Set 100 keys
        for i in range(100):
            await cache.set(f"key{i}", f"value{i}")

        # Verify all keys exist
        for i in range(100):
            assert await cache.exists(f"key{i}") is True

        # Get all keys
        keys = [f"key{i}" for i in range(100)]
        result = await cache.get_many(keys)
        assert len(result) == 100

        # Clear all
        await cache.clear()
        for i in range(100):
            assert await cache.exists(f"key{i}") is False

    async def test_special_characters_in_keys(self, cache: RedisCacheBackend) -> None:
        """Test keys with special characters."""
        special_keys = [
            "key:with:colons",
            "key-with-dashes",
            "key_with_underscores",
            "key.with.dots",
            "key/with/slashes",
        ]

        for key in special_keys:
            await cache.set(key, f"value-{key}")

        for key in special_keys:
            result = await cache.get(key)
            assert result == f"value-{key}"
