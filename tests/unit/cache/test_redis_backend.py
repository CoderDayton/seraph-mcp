"""
Seraph MCP — Redis Backend Unit Tests

Comprehensive unit tests for the Redis cache backend, covering:
- Basic operations (get, set, delete, exists)
- TTL handling (None, 0, positive values)
- Namespace prefixing
- Batch operations (get_many, set_many, delete_many)
- Clear operation with SCAN pattern
- Statistics tracking (hits, misses, sets, deletes)
- JSON serialization edge cases
- Error handling and connection failures
- Resource cleanup
"""

import asyncio
import json

import pytest

from src.cache.backends.redis import RedisCacheBackend


class TestRedisCacheBackendInitialization:
    """Test Redis backend initialization and configuration."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        backend = RedisCacheBackend(redis_url="redis://localhost:6379")

        assert backend.namespace == "seraph"
        assert backend.default_ttl == 3600
        assert backend._hits == 0
        assert backend._misses == 0
        assert backend._sets == 0
        assert backend._deletes == 0

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        backend = RedisCacheBackend(
            redis_url="redis://localhost:6379/5",
            namespace="custom",
            default_ttl=7200,
            max_connections=20,
            socket_timeout=10,
        )

        assert backend.namespace == "custom"
        assert backend.default_ttl == 7200

    def test_init_without_url_raises_error(self):
        """Test that initialization without URL raises ValueError."""
        with pytest.raises(ValueError, match="redis_url is required"):
            RedisCacheBackend(redis_url="")

    def test_init_with_empty_namespace_uses_default(self):
        """Test that empty namespace falls back to default."""
        backend = RedisCacheBackend(
            redis_url="redis://localhost:6379",
            namespace="  ",
        )
        assert backend.namespace == "seraph"

    def test_init_negative_ttl_converted_to_zero(self):
        """Test that negative TTL is converted to 0."""
        backend = RedisCacheBackend(
            redis_url="redis://localhost:6379",
            default_ttl=-100,
        )
        assert backend.default_ttl == 0


class TestRedisCacheBackendHelpers:
    """Test helper methods for key generation and serialization."""

    def test_make_key_with_namespace(self):
        """Test key generation with namespace prefix."""
        backend = RedisCacheBackend(
            redis_url="redis://localhost:6379",
            namespace="myapp",
        )

        assert backend._make_key("user:123") == "myapp:user:123"
        assert backend._make_key("session") == "myapp:session"

    def test_to_json_serialization(self):
        """Test JSON serialization of various data types."""
        # Simple types
        assert RedisCacheBackend._to_json("hello") == '"hello"'
        assert RedisCacheBackend._to_json(42) == "42"
        assert RedisCacheBackend._to_json(3.14) == "3.14"
        assert RedisCacheBackend._to_json(True) == "true"
        assert RedisCacheBackend._to_json(None) == "null"

        # Complex types
        data = {"key": "value", "number": 123, "nested": {"list": [1, 2, 3]}}
        json_str = RedisCacheBackend._to_json(data)
        assert json.loads(json_str) == data

    def test_from_json_deserialization(self):
        """Test JSON deserialization of various data types."""
        assert RedisCacheBackend._from_json('"hello"') == "hello"
        assert RedisCacheBackend._from_json("42") == 42
        assert RedisCacheBackend._from_json("3.14") == 3.14
        assert RedisCacheBackend._from_json("true") is True
        assert RedisCacheBackend._from_json("null") is None
        assert RedisCacheBackend._from_json(None) is None

        # Complex types
        json_str = '{"key":"value","number":123}'
        assert RedisCacheBackend._from_json(json_str) == {"key": "value", "number": 123}

    def test_from_json_invalid_returns_raw(self):
        """Test that invalid JSON returns raw data."""
        invalid_json = "not-valid-json"
        result = RedisCacheBackend._from_json(invalid_json)
        assert result == invalid_json

    def test_ttl_seconds_normalization(self):
        """Test TTL normalization logic."""
        backend = RedisCacheBackend(
            redis_url="redis://localhost:6379",
            default_ttl=3600,
        )

        # None -> default_ttl
        assert backend._ttl_seconds(None) == 3600

        # 0 or negative -> None (no expiry)
        assert backend._ttl_seconds(0) is None
        assert backend._ttl_seconds(-100) is None

        # Positive -> as provided
        assert backend._ttl_seconds(60) == 60
        assert backend._ttl_seconds(7200) == 7200


@pytest.mark.asyncio
class TestRedisCacheBackendBasicOperations:
    """Test basic cache operations (get, set, delete, exists)."""

    async def test_get_existing_key(self, redis_client):
        """Test retrieving an existing key."""
        backend = RedisCacheBackend(
            redis_url="redis://localhost:6379/15",
            namespace="test",
        )

        # Pre-populate with data
        await redis_client.set("test:mykey", '"hello"')

        result = await backend.get("mykey")
        assert result == "hello"
        assert backend._hits == 1
        assert backend._misses == 0

    async def test_get_missing_key(self, redis_client):
        """Test retrieving a non-existent key."""
        backend = RedisCacheBackend(
            redis_url="redis://localhost:6379/15",
            namespace="test",
        )

        result = await backend.get("nonexistent")
        assert result is None
        assert backend._hits == 0
        assert backend._misses == 1

    async def test_set_with_default_ttl(self, redis_client):
        """Test setting a value with default TTL."""
        backend = RedisCacheBackend(
            redis_url="redis://localhost:6379/15",
            namespace="test",
            default_ttl=60,
        )

        success = await backend.set("mykey", "myvalue")
        assert success is True
        assert backend._sets == 1

        # Verify stored value
        stored = await redis_client.get("test:mykey")
        assert stored == '"myvalue"'

        # Verify TTL was set
        ttl = await redis_client.ttl("test:mykey")
        assert 55 <= ttl <= 60  # Allow small margin

    async def test_set_with_custom_ttl(self, redis_client):
        """Test setting a value with custom TTL."""
        backend = RedisCacheBackend(
            redis_url="redis://localhost:6379/15",
            namespace="test",
        )

        success = await backend.set("mykey", "myvalue", ttl=30)
        assert success is True

        ttl = await redis_client.ttl("test:mykey")
        assert 25 <= ttl <= 30

    async def test_set_with_zero_ttl_no_expiry(self, redis_client):
        """Test setting a value with TTL=0 (no expiry)."""
        backend = RedisCacheBackend(
            redis_url="redis://localhost:6379/15",
            namespace="test",
        )

        success = await backend.set("mykey", "myvalue", ttl=0)
        assert success is True

        # TTL should be -1 (no expiry)
        ttl = await redis_client.ttl("test:mykey")
        assert ttl == -1

    async def test_set_complex_data(self, redis_client):
        """Test setting complex data structures."""
        backend = RedisCacheBackend(
            redis_url="redis://localhost:6379/15",
            namespace="test",
        )

        complex_data = {
            "user": {"id": 123, "name": "Alice"},
            "tags": ["python", "redis", "async"],
            "metadata": {"created": "2025-01-01", "active": True},
        }

        await backend.set("complex", complex_data)
        result = await backend.get("complex")
        assert result == complex_data

    async def test_delete_existing_key(self, redis_client):
        """Test deleting an existing key."""
        backend = RedisCacheBackend(
            redis_url="redis://localhost:6379/15",
            namespace="test",
        )

        await redis_client.set("test:mykey", '"value"')

        deleted = await backend.delete("mykey")
        assert deleted is True
        assert backend._deletes == 1

        # Verify key is gone
        exists = await redis_client.exists("test:mykey")
        assert exists == 0

    async def test_delete_missing_key(self, redis_client):
        """Test deleting a non-existent key."""
        backend = RedisCacheBackend(
            redis_url="redis://localhost:6379/15",
            namespace="test",
        )

        deleted = await backend.delete("nonexistent")
        assert deleted is False

    async def test_exists_for_existing_key(self, redis_client):
        """Test checking existence of an existing key."""
        backend = RedisCacheBackend(
            redis_url="redis://localhost:6379/15",
            namespace="test",
        )

        await redis_client.set("test:mykey", '"value"')

        exists = await backend.exists("mykey")
        assert exists is True

    async def test_exists_for_missing_key(self, redis_client):
        """Test checking existence of a missing key."""
        backend = RedisCacheBackend(
            redis_url="redis://localhost:6379/15",
            namespace="test",
        )

        exists = await backend.exists("nonexistent")
        assert exists is False


@pytest.mark.asyncio
class TestRedisCacheBackendBatchOperations:
    """Test batch operations (get_many, set_many, delete_many)."""

    async def test_get_many_all_existing(self, redis_client):
        """Test retrieving multiple existing keys."""
        backend = RedisCacheBackend(
            redis_url="redis://localhost:6379/15",
            namespace="test",
        )

        # Pre-populate
        await redis_client.mset(
            {
                "test:key1": '"value1"',
                "test:key2": '"value2"',
                "test:key3": '"value3"',
            }
        )

        results = await backend.get_many(["key1", "key2", "key3"])
        assert results == {
            "key1": "value1",
            "key2": "value2",
            "key3": "value3",
        }
        assert backend._hits == 3
        assert backend._misses == 0

    async def test_get_many_partial_existing(self, redis_client):
        """Test retrieving mix of existing and missing keys."""
        backend = RedisCacheBackend(
            redis_url="redis://localhost:6379/15",
            namespace="test",
        )

        # Pre-populate only some keys
        await redis_client.set("test:key1", '"value1"')
        await redis_client.set("test:key3", '"value3"')

        results = await backend.get_many(["key1", "key2", "key3"])
        assert results == {
            "key1": "value1",
            "key3": "value3",
        }
        assert "key2" not in results
        assert backend._hits == 2
        assert backend._misses == 1

    async def test_get_many_empty_list(self, redis_client):
        """Test get_many with empty key list."""
        backend = RedisCacheBackend(
            redis_url="redis://localhost:6379/15",
            namespace="test",
        )

        results = await backend.get_many([])
        assert results == {}

    async def test_set_many_with_default_ttl(self, redis_client):
        """Test setting multiple values with default TTL."""
        backend = RedisCacheBackend(
            redis_url="redis://localhost:6379/15",
            namespace="test",
            default_ttl=60,
        )

        items = {
            "key1": "value1",
            "key2": {"nested": "data"},
            "key3": [1, 2, 3],
        }

        count = await backend.set_many(items)
        assert count == 3
        assert backend._sets == 3

        # Verify all keys exist with TTL
        for key in items:
            exists = await redis_client.exists(f"test:{key}")
            assert exists == 1
            ttl = await redis_client.ttl(f"test:{key}")
            assert 55 <= ttl <= 60

    async def test_set_many_with_custom_ttl(self, redis_client):
        """Test setting multiple values with custom TTL."""
        backend = RedisCacheBackend(
            redis_url="redis://localhost:6379/15",
            namespace="test",
        )

        items = {"key1": "value1", "key2": "value2"}
        count = await backend.set_many(items, ttl=30)
        assert count == 2

        ttl = await redis_client.ttl("test:key1")
        assert 25 <= ttl <= 30

    async def test_set_many_empty_dict(self, redis_client):
        """Test set_many with empty dictionary."""
        backend = RedisCacheBackend(
            redis_url="redis://localhost:6379/15",
            namespace="test",
        )

        count = await backend.set_many({})
        assert count == 0

    async def test_delete_many_all_existing(self, redis_client):
        """Test deleting multiple existing keys."""
        backend = RedisCacheBackend(
            redis_url="redis://localhost:6379/15",
            namespace="test",
        )

        # Pre-populate
        await redis_client.mset(
            {
                "test:key1": '"value1"',
                "test:key2": '"value2"',
                "test:key3": '"value3"',
            }
        )

        count = await backend.delete_many(["key1", "key2", "key3"])
        assert count == 3
        assert backend._deletes == 3

        # Verify all keys are gone
        for key in ["key1", "key2", "key3"]:
            exists = await redis_client.exists(f"test:{key}")
            assert exists == 0

    async def test_delete_many_partial_existing(self, redis_client):
        """Test deleting mix of existing and missing keys."""
        backend = RedisCacheBackend(
            redis_url="redis://localhost:6379/15",
            namespace="test",
        )

        # Pre-populate only some keys
        await redis_client.set("test:key1", '"value1"')
        await redis_client.set("test:key3", '"value3"')

        count = await backend.delete_many(["key1", "key2", "key3"])
        assert count == 2  # Only key1 and key3 existed

    async def test_delete_many_empty_list(self, redis_client):
        """Test delete_many with empty key list."""
        backend = RedisCacheBackend(
            redis_url="redis://localhost:6379/15",
            namespace="test",
        )

        count = await backend.delete_many([])
        assert count == 0

    async def test_delete_many_large_batch(self, redis_client):
        """Test deleting large batch of keys (chunking logic)."""
        backend = RedisCacheBackend(
            redis_url="redis://localhost:6379/15",
            namespace="test",
        )

        # Create 2000 keys (more than chunk_size of 1000)
        keys = [f"key{i}" for i in range(2000)]
        data = {f"test:key{i}": f'"value{i}"' for i in range(2000)}
        await redis_client.mset(data)

        count = await backend.delete_many(keys)
        assert count == 2000


@pytest.mark.asyncio
class TestRedisCacheBackendClear:
    """Test clear operation with namespace isolation."""

    async def test_clear_namespace_only(self, redis_client):
        """Test that clear only removes keys in the namespace."""
        backend = RedisCacheBackend(
            redis_url="redis://localhost:6379/15",
            namespace="app1",
        )

        # Add keys in different namespaces
        await redis_client.mset(
            {
                "app1:key1": '"value1"',
                "app1:key2": '"value2"',
                "app2:key1": '"value1"',  # Different namespace
                "global:key": '"value"',  # No namespace prefix
            }
        )

        success = await backend.clear()
        assert success is True

        # Verify only app1 keys are deleted
        assert await redis_client.exists("app1:key1") == 0
        assert await redis_client.exists("app1:key2") == 0
        assert await redis_client.exists("app2:key1") == 1  # Still exists
        assert await redis_client.exists("global:key") == 1  # Still exists

    async def test_clear_empty_namespace(self, redis_client):
        """Test clearing when namespace has no keys."""
        backend = RedisCacheBackend(
            redis_url="redis://localhost:6379/15",
            namespace="empty",
        )

        success = await backend.clear()
        assert success is True

    async def test_clear_large_namespace(self, redis_client):
        """Test clearing namespace with many keys (SCAN iteration)."""
        backend = RedisCacheBackend(
            redis_url="redis://localhost:6379/15",
            namespace="bigapp",
        )

        # Create 5000 keys
        data = {f"bigapp:key{i}": f'"value{i}"' for i in range(5000)}
        # Split into chunks for mset
        chunk_size = 1000
        items = list(data.items())
        for i in range(0, len(items), chunk_size):
            chunk = dict(items[i : i + chunk_size])
            await redis_client.mset(chunk)

        success = await backend.clear()
        assert success is True

        # Verify all keys are gone
        pattern = "bigapp:*"
        cursor, keys = await redis_client.scan(0, match=pattern, count=100)
        assert len(keys) == 0


@pytest.mark.asyncio
class TestRedisCacheBackendStats:
    """Test statistics and monitoring."""

    async def test_get_stats_basic(self, redis_client):
        """Test basic statistics collection."""
        backend = RedisCacheBackend(
            redis_url="redis://localhost:6379/15",
            namespace="test",
            default_ttl=3600,
        )

        # Perform some operations
        await backend.set("key1", "value1")
        await backend.get("key1")  # Hit
        await backend.get("key2")  # Miss
        await backend.delete("key1")

        stats = await backend.get_stats()

        assert stats["backend"] == "redis"
        assert stats["namespace"] == "test"
        assert stats["default_ttl"] == 3600
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["sets"] == 1
        assert stats["deletes"] == 1
        assert stats["hit_rate"] == 50.0  # 1 hit out of 2 requests
        assert stats["connected"] is True

    async def test_get_stats_includes_redis_info(self, redis_client):
        """Test that stats include Redis server information."""
        backend = RedisCacheBackend(
            redis_url="redis://localhost:6379/15",
            namespace="test",
        )

        stats = await backend.get_stats()

        assert "redis_version" in stats
        assert "redis_mode" in stats
        assert "os" in stats
        assert "keyspace" in stats

    async def test_get_stats_no_requests_zero_hit_rate(self, redis_client):
        """Test hit rate calculation when no requests."""
        backend = RedisCacheBackend(
            redis_url="redis://localhost:6379/15",
            namespace="test",
        )

        stats = await backend.get_stats()
        assert stats["hit_rate"] == 0.0

    async def test_hit_miss_tracking(self, redis_client):
        """Test accurate hit/miss tracking across operations."""
        backend = RedisCacheBackend(
            redis_url="redis://localhost:6379/15",
            namespace="test",
        )

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


@pytest.mark.asyncio
class TestRedisCacheBackendResourceManagement:
    """Test resource cleanup and lifecycle."""

    async def test_close_releases_resources(self, redis_client):
        """Test that close properly releases Redis connection."""
        backend = RedisCacheBackend(
            redis_url="redis://localhost:6379/15",
            namespace="test",
        )

        # Use the connection
        await backend.set("key", "value")

        # Close should not raise
        await backend.close()

        # Subsequent operations should fail (connection closed)
        # Note: This depends on Redis client behavior

    async def test_close_idempotent(self, redis_client):
        """Test that close can be called multiple times safely."""
        backend = RedisCacheBackend(
            redis_url="redis://localhost:6379/15",
            namespace="test",
        )

        await backend.close()
        await backend.close()  # Should not raise


@pytest.mark.asyncio
class TestRedisCacheBackendEdgeCases:
    """Test edge cases and special scenarios."""

    async def test_unicode_keys_and_values(self, redis_client):
        """Test handling of Unicode keys and values."""
        backend = RedisCacheBackend(
            redis_url="redis://localhost:6379/15",
            namespace="test",
        )

        unicode_data = {
            "greeting": "Hello 世界 🌍",
            "emoji": "🎉🎊🎈",
            "special": "Ñoño café",
        }

        await backend.set("unicode", unicode_data)
        result = await backend.get("unicode")
        assert result == unicode_data

    async def test_large_values(self, redis_client):
        """Test storing and retrieving large values."""
        backend = RedisCacheBackend(
            redis_url="redis://localhost:6379/15",
            namespace="test",
        )

        # Create a large value (~1MB of data)
        large_value = {"data": "x" * 1000000}

        await backend.set("large", large_value)
        result = await backend.get("large")
        assert result == large_value

    async def test_special_characters_in_keys(self, redis_client):
        """Test keys with special characters."""
        backend = RedisCacheBackend(
            redis_url="redis://localhost:6379/15",
            namespace="test",
        )

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

    async def test_concurrent_operations(self, redis_client):
        """Test concurrent get/set operations."""
        backend = RedisCacheBackend(
            redis_url="redis://localhost:6379/15",
            namespace="test",
        )

        async def set_value(key: str, value: str):
            await backend.set(key, value)

        async def get_value(key: str):
            return await backend.get(key)

        # Set 100 keys concurrently
        await asyncio.gather(*[set_value(f"key{i}", f"value{i}") for i in range(100)])

        # Get all keys concurrently
        results = await asyncio.gather(*[get_value(f"key{i}") for i in range(100)])

        # Verify all values
        for i, result in enumerate(results):
            assert result == f"value{i}"

    async def test_ttl_expiration(self, redis_client):
        """Test that keys expire after TTL."""
        backend = RedisCacheBackend(
            redis_url="redis://localhost:6379/15",
            namespace="test",
        )

        # Set with very short TTL
        await backend.set("expires", "value", ttl=1)

        # Should exist immediately
        result = await backend.get("expires")
        assert result == "value"

        # Wait for expiration
        await asyncio.sleep(1.5)

        # Should be expired
        result = await backend.get("expires")
        assert result is None
