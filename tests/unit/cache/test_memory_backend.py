"""
Seraph MCP — Memory Backend Unit Tests

Comprehensive unit tests for the in-memory cache backend, covering:
- Basic operations (get, set, delete, exists)
- TTL handling and expiration
- LRU eviction when at capacity
- Namespace prefixing
- Clear operation
- Statistics tracking
- Thread safety and concurrent operations
"""

import asyncio
import time
from typing import Any

import pytest

from src.cache.backends.memory import MemoryCacheBackend


class TestMemoryCacheBackendInitialization:
    """Test memory backend initialization and configuration."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        backend = MemoryCacheBackend()

        assert backend.namespace == "seraph"
        assert backend.default_ttl == 3600
        assert backend.max_size == 1000
        assert backend._hits == 0
        assert backend._misses == 0
        assert backend._sets == 0
        assert backend._deletes == 0

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        backend = MemoryCacheBackend(
            namespace="custom",
            default_ttl=7200,
            max_size=500,
        )

        assert backend.namespace == "custom"
        assert backend.default_ttl == 7200
        assert backend.max_size == 500

    def test_init_with_zero_max_size(self):
        """Test that zero max_size is handled."""
        backend = MemoryCacheBackend(max_size=0)
        assert backend.max_size == 0

    def test_init_negative_ttl_converted_to_zero(self):
        """Test that negative TTL is converted to 0."""
        backend = MemoryCacheBackend(default_ttl=-100)
        assert backend.default_ttl == 0


class TestMemoryCacheBackendHelpers:
    """Test helper methods."""

    def test_make_key_with_namespace(self):
        """Test key generation with namespace prefix."""
        backend = MemoryCacheBackend(namespace="myapp")

        assert backend._make_key("user:123") == "myapp:user:123"
        assert backend._make_key("session") == "myapp:session"

    def test_ttl_seconds_normalization(self):
        """Test TTL normalization logic."""
        backend = MemoryCacheBackend(default_ttl=3600)

        # None -> default_ttl
        assert backend._ttl_seconds(None) == 3600

        # 0 or negative -> None (no expiry)
        assert backend._ttl_seconds(0) is None
        assert backend._ttl_seconds(-100) is None

        # Positive -> as provided
        assert backend._ttl_seconds(60) == 60
        assert backend._ttl_seconds(7200) == 7200

    def test_is_expired_no_expiry(self):
        """Test that items with no expiry never expire."""
        backend = MemoryCacheBackend()
        entry = {"value": "test", "expires_at": None}

        assert backend._is_expired(entry) is False

    def test_is_expired_not_yet_expired(self):
        """Test that items are not expired before expiry time."""
        backend = MemoryCacheBackend()
        future_time = time.time() + 3600
        entry = {"value": "test", "expires_at": future_time}

        assert backend._is_expired(entry) is False

    def test_is_expired_already_expired(self):
        """Test that items are expired after expiry time."""
        backend = MemoryCacheBackend()
        past_time = time.time() - 1
        entry = {"value": "test", "expires_at": past_time}

        assert backend._is_expired(entry) is True


@pytest.mark.asyncio
class TestMemoryCacheBackendBasicOperations:
    """Test basic cache operations."""

    async def test_get_existing_key(self):
        """Test retrieving an existing key."""
        backend = MemoryCacheBackend()

        await backend.set("mykey", "myvalue")
        result = await backend.get("mykey")

        assert result == "myvalue"
        assert backend._hits == 1
        assert backend._misses == 0

    async def test_get_missing_key(self):
        """Test retrieving a non-existent key."""
        backend = MemoryCacheBackend()

        result = await backend.get("nonexistent")

        assert result is None
        assert backend._hits == 0
        assert backend._misses == 1

    async def test_get_expired_key(self):
        """Test that expired keys return None."""
        backend = MemoryCacheBackend()

        # Set with very short TTL
        await backend.set("expires", "value", ttl=1)

        # Should exist immediately
        result = await backend.get("expires")
        assert result == "value"

        # Wait for expiration
        await asyncio.sleep(1.5)

        # Should be expired and return None
        result = await backend.get("expires")
        assert result is None
        assert backend._misses == 1

    async def test_set_with_default_ttl(self):
        """Test setting a value with default TTL."""
        backend = MemoryCacheBackend(default_ttl=60)

        success = await backend.set("mykey", "myvalue")

        assert success is True
        assert backend._sets == 1

        # Verify value is stored
        result = await backend.get("mykey")
        assert result == "myvalue"

    async def test_set_with_custom_ttl(self):
        """Test setting a value with custom TTL."""
        backend = MemoryCacheBackend()

        success = await backend.set("mykey", "myvalue", ttl=30)
        assert success is True

        # Value should exist
        result = await backend.get("mykey")
        assert result == "myvalue"

    async def test_set_with_zero_ttl_no_expiry(self):
        """Test setting a value with TTL=0 (no expiry)."""
        backend = MemoryCacheBackend()

        await backend.set("mykey", "myvalue", ttl=0)

        # Check internal storage to verify no expiry
        ns_key = backend._make_key("mykey")
        assert ns_key in backend._cache
        assert backend._cache[ns_key]["expires_at"] is None

    async def test_set_overwrites_existing(self):
        """Test that setting an existing key updates the value."""
        backend = MemoryCacheBackend()

        await backend.set("mykey", "value1")
        await backend.set("mykey", "value2")

        result = await backend.get("mykey")
        assert result == "value2"
        assert backend._sets == 2

    async def test_set_complex_data(self):
        """Test setting complex data structures."""
        backend = MemoryCacheBackend()

        complex_data = {
            "user": {"id": 123, "name": "Alice"},
            "tags": ["python", "memory", "async"],
            "metadata": {"created": "2025-01-01", "active": True},
        }

        await backend.set("complex", complex_data)
        result = await backend.get("complex")
        assert result == complex_data

    async def test_delete_existing_key(self):
        """Test deleting an existing key."""
        backend = MemoryCacheBackend()

        await backend.set("mykey", "myvalue")
        deleted = await backend.delete("mykey")

        assert deleted is True
        assert backend._deletes == 1

        # Verify key is gone
        result = await backend.get("mykey")
        assert result is None

    async def test_delete_missing_key(self):
        """Test deleting a non-existent key."""
        backend = MemoryCacheBackend()

        deleted = await backend.delete("nonexistent")
        assert deleted is False

    async def test_exists_for_existing_key(self):
        """Test checking existence of an existing key."""
        backend = MemoryCacheBackend()

        await backend.set("mykey", "myvalue")
        exists = await backend.exists("mykey")

        assert exists is True

    async def test_exists_for_missing_key(self):
        """Test checking existence of a missing key."""
        backend = MemoryCacheBackend()

        exists = await backend.exists("nonexistent")
        assert exists is False

    async def test_exists_for_expired_key(self):
        """Test that expired keys are reported as not existing."""
        backend = MemoryCacheBackend()

        await backend.set("expires", "value", ttl=1)
        await asyncio.sleep(1.5)

        exists = await backend.exists("expires")
        assert exists is False


@pytest.mark.asyncio
class TestMemoryCacheBackendLRUEviction:
    """Test LRU eviction when cache reaches max size."""

    async def test_eviction_when_at_capacity(self):
        """Test that oldest items are evicted when at capacity."""
        backend = MemoryCacheBackend(max_size=3)

        # Fill cache to capacity
        await backend.set("key1", "value1")
        await backend.set("key2", "value2")
        await backend.set("key3", "value3")

        # Verify all exist
        assert await backend.exists("key1") is True
        assert await backend.exists("key2") is True
        assert await backend.exists("key3") is True

        # Add one more item, should evict key1 (oldest)
        await backend.set("key4", "value4")

        # key1 should be evicted
        assert await backend.exists("key1") is False
        assert await backend.exists("key2") is True
        assert await backend.exists("key3") is True
        assert await backend.exists("key4") is True

    async def test_get_updates_access_order(self):
        """Test that get operations update LRU order."""
        backend = MemoryCacheBackend(max_size=3)

        await backend.set("key1", "value1")
        await backend.set("key2", "value2")
        await backend.set("key3", "value3")

        # Access key1 to make it most recently used
        await backend.get("key1")

        # Add key4, should evict key2 (now oldest)
        await backend.set("key4", "value4")

        assert await backend.exists("key1") is True  # Accessed recently
        assert await backend.exists("key2") is False  # Evicted
        assert await backend.exists("key3") is True
        assert await backend.exists("key4") is True

    async def test_no_eviction_when_under_capacity(self):
        """Test that no eviction occurs when under capacity."""
        backend = MemoryCacheBackend(max_size=10)

        for i in range(5):
            await backend.set(f"key{i}", f"value{i}")

        # All items should still exist
        for i in range(5):
            assert await backend.exists(f"key{i}") is True


@pytest.mark.asyncio
class TestMemoryCacheBackendClear:
    """Test clear operation with namespace isolation."""

    async def test_clear_removes_all_namespace_keys(self):
        """Test that clear removes all keys in the namespace."""
        backend = MemoryCacheBackend(namespace="app1")

        await backend.set("key1", "value1")
        await backend.set("key2", "value2")
        await backend.set("key3", "value3")

        success = await backend.clear()
        assert success is True

        # Verify all keys are gone
        assert await backend.exists("key1") is False
        assert await backend.exists("key2") is False
        assert await backend.exists("key3") is False

    async def test_clear_empty_cache(self):
        """Test clearing when cache is empty."""
        backend = MemoryCacheBackend()

        success = await backend.clear()
        assert success is True

    async def test_clear_namespace_isolation(self):
        """Test that clear only affects the specific namespace."""
        backend1 = MemoryCacheBackend(namespace="app1")
        backend2 = MemoryCacheBackend(namespace="app2")

        # Add keys to both backends (they share the same internal dict in this test)
        await backend1.set("key1", "value1")
        await backend2.set("key1", "value1")

        # Clear backend1
        await backend1.clear()

        # Backend1 keys should be gone
        assert await backend1.exists("key1") is False

        # Backend2 keys should still exist
        assert await backend2.exists("key1") is True


@pytest.mark.asyncio
class TestMemoryCacheBackendStats:
    """Test statistics and monitoring."""

    async def test_get_stats_basic(self):
        """Test basic statistics collection."""
        backend = MemoryCacheBackend(
            namespace="test",
            default_ttl=3600,
            max_size=100,
        )

        # Perform some operations
        await backend.set("key1", "value1")
        await backend.get("key1")  # Hit
        await backend.get("key2")  # Miss
        await backend.delete("key1")

        stats = await backend.get_stats()

        assert stats["backend"] == "memory"
        assert stats["namespace"] == "test"
        assert stats["default_ttl"] == 3600
        assert stats["max_size"] == 100
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["sets"] == 1
        assert stats["deletes"] == 1
        assert stats["hit_rate"] == 50.0  # 1 hit out of 2 requests
        assert "size" in stats

    async def test_get_stats_no_requests_zero_hit_rate(self):
        """Test hit rate calculation when no requests."""
        backend = MemoryCacheBackend()

        stats = await backend.get_stats()
        assert stats["hit_rate"] == 0.0

    async def test_hit_miss_tracking(self):
        """Test accurate hit/miss tracking."""
        backend = MemoryCacheBackend()

        await backend.set("exists", "value")

        # 3 hits
        await backend.get("exists")
        await backend.get("exists")
        await backend.get("exists")

        # 2 misses
        await backend.get("miss1")
        await backend.get("miss2")

        stats = await backend.get_stats()
        assert stats["hits"] == 3
        assert stats["misses"] == 2
        assert stats["hit_rate"] == 60.0  # 3 hits out of 5 total

    async def test_stats_size_reflects_current_entries(self):
        """Test that size in stats reflects current number of entries."""
        backend = MemoryCacheBackend()

        # Add 5 entries
        for i in range(5):
            await backend.set(f"key{i}", f"value{i}")

        stats = await backend.get_stats()
        assert stats["size"] == 5

        # Delete 2 entries
        await backend.delete("key0")
        await backend.delete("key1")

        stats = await backend.get_stats()
        assert stats["size"] == 3


@pytest.mark.asyncio
class TestMemoryCacheBackendResourceManagement:
    """Test resource cleanup and lifecycle."""

    async def test_close_clears_cache(self):
        """Test that close clears all entries."""
        backend = MemoryCacheBackend()

        await backend.set("key1", "value1")
        await backend.set("key2", "value2")

        await backend.close()

        # Cache should be empty after close
        stats = await backend.get_stats()
        assert stats["size"] == 0

    async def test_close_idempotent(self):
        """Test that close can be called multiple times safely."""
        backend = MemoryCacheBackend()

        await backend.close()
        await backend.close()  # Should not raise


@pytest.mark.asyncio
class TestMemoryCacheBackendEdgeCases:
    """Test edge cases and special scenarios."""

    async def test_unicode_keys_and_values(self):
        """Test handling of Unicode keys and values."""
        backend = MemoryCacheBackend()

        unicode_data = {
            "greeting": "Hello 世界 🌍",
            "emoji": "🎉🎊🎈",
            "special": "Ñoño café",
        }

        await backend.set("unicode", unicode_data)
        result = await backend.get("unicode")
        assert result == unicode_data

    async def test_none_as_value(self):
        """Test storing None as a value."""
        backend = MemoryCacheBackend()

        await backend.set("null_key", None)
        result = await backend.get("null_key")
        assert result is None

        # Key should still exist
        exists = await backend.exists("null_key")
        assert exists is True

    async def test_large_values(self):
        """Test storing and retrieving large values."""
        backend = MemoryCacheBackend()

        # Create a large value (~1MB of data)
        large_value = {"data": "x" * 1000000}

        await backend.set("large", large_value)
        result = await backend.get("large")
        assert result == large_value

    async def test_special_characters_in_keys(self):
        """Test keys with special characters."""
        backend = MemoryCacheBackend()

        special_keys = [
            "key:with:colons",
            "key-with-dashes",
            "key_with_underscores",
            "key.with.dots",
            "key/with/slashes",
        ]

        for key in special_keys:
            await backend.set(key, f"value-{key}")

        for key in special_keys:
            result = await backend.get(key)
            assert result == f"value-{key}"

    async def test_concurrent_operations(self):
        """Test concurrent get/set operations."""
        backend = MemoryCacheBackend()

        async def set_value(key: str, value: str):
            await backend.set(key, value)

        async def get_value(key: str):
            return await backend.get(key)

        # Set 100 keys concurrently
        await asyncio.gather(*[
            set_value(f"key{i}", f"value{i}")
            for i in range(100)
        ])

        # Get all keys concurrently
        results = await asyncio.gather(*[
            get_value(f"key{i}")
            for i in range(100)
        ])

        # Verify all values
        for i, result in enumerate(results):
            assert result == f"value{i}"

    async def test_empty_string_key(self):
        """Test handling of empty string as key."""
        backend = MemoryCacheBackend()

        await backend.set("", "empty_key_value")
        result = await backend.get("")
        assert result == "empty_key_value"

    async def test_very_long_key(self):
        """Test handling of very long keys."""
        backend = MemoryCacheBackend()

        long_key = "k" * 10000
        await backend.set(long_key, "value")
        result = await backend.get(long_key)
        assert result == "value"
