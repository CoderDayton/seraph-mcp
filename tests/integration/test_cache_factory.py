"""
Seraph MCP — Cache Factory Integration Tests

Tests for the cache factory that creates and manages cache instances.
Tests factory pattern, singleton behavior, configuration, and lifecycle management.

Python 3.12+ with modern async patterns and type hints.
"""

from collections.abc import AsyncGenerator

import pytest  # type: ignore[import-untyped]

from src.cache.factory import (
    clear_cache_registry,
    close_all_caches,
    create_cache,
    get_cache,
    list_cache_instances,
    reset_cache_factory,
)
from src.cache.interface import CacheInterface
from src.config import CacheBackend, CacheConfig

# Check if Redis is available
try:
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    redis_available = sock.connect_ex(("localhost", 6379)) == 0
    sock.close()
except Exception:
    redis_available = False


class TestCacheFactory:
    """Test suite for cache factory functionality."""

    @pytest.fixture(autouse=True)  # type: ignore[misc]
    async def cleanup(self) -> AsyncGenerator[None, None]:  # type: ignore[misc]
        """Clean up cache instances after each test."""
        yield
        await close_all_caches()
        reset_cache_factory()

    async def test_create_memory_cache_default(self) -> None:
        """Test creating a memory cache with default configuration."""
        cache = create_cache()

        assert cache is not None
        assert isinstance(cache, CacheInterface)

        # Should be able to use it
        await cache.set("test_key", "test_value")
        value = await cache.get("test_key")
        assert value == "test_value"

    async def test_create_memory_cache_explicit_config(self) -> None:
        """Test creating a memory cache with explicit configuration."""
        config = CacheConfig(
            backend=CacheBackend.MEMORY,
            namespace="test_ns",
            max_size=50,
            ttl_seconds=1800,
        )

        cache = create_cache(config=config, name="custom")

        assert cache is not None
        await cache.set("key1", "value1")
        assert await cache.get("key1") == "value1"

    @pytest.mark.skipif(not redis_available, reason="Redis server not available")
    async def test_create_redis_cache_with_config(self, test_redis_url: str) -> None:
        """Test creating a Redis cache with configuration."""
        config = CacheConfig(
            backend=CacheBackend.REDIS,
            redis_url=test_redis_url,
            namespace="test_redis",
            ttl_seconds=3600,
        )

        cache = create_cache(config=config, name="redis_test")

        assert cache is not None
        await cache.set("key1", "value1")
        assert await cache.get("key1") == "value1"

    async def test_singleton_behavior(self) -> None:
        """Test that factory returns the same instance for the same name."""
        cache1 = create_cache(name="singleton_test")
        cache2 = create_cache(name="singleton_test")

        # Should be the exact same instance
        assert cache1 is cache2

    async def test_multiple_named_instances(self) -> None:
        """Test creating multiple named cache instances."""
        cache1 = create_cache(name="cache1")
        cache2 = create_cache(name="cache2")

        # Should be different instances
        assert cache1 is not cache2

        # Should maintain separate data
        await cache1.set("key", "value1")
        await cache2.set("key", "value2")

        assert await cache1.get("key") == "value1"
        assert await cache2.get("key") == "value2"

    async def test_get_cache_creates_if_not_exists(self) -> None:
        """Test that get_cache creates instance if it doesn't exist."""
        cache = get_cache("new_instance")

        assert cache is not None
        await cache.set("key", "value")
        assert await cache.get("key") == "value"

    async def test_get_cache_returns_existing(self) -> None:
        """Test that get_cache returns existing instance."""
        cache1 = create_cache(name="existing")
        await cache1.set("key", "value")

        cache2 = get_cache("existing")

        assert cache1 is cache2
        assert await cache2.get("key") == "value"

    async def test_list_cache_instances(self) -> None:
        """Test listing all cache instances."""
        # Initially empty
        instances = list_cache_instances()
        assert len(instances) == 0

        # Create some instances
        create_cache(name="cache1")
        create_cache(name="cache2")
        create_cache(name="cache3")

        instances = list_cache_instances()
        assert len(instances) == 3
        assert "cache1" in instances
        assert "cache2" in instances
        assert "cache3" in instances

    async def test_close_all_caches(self) -> None:
        """Test closing all cache instances."""
        # Create multiple caches
        cache1 = create_cache(name="cache1")
        cache2 = create_cache(name="cache2")

        await cache1.set("key", "value")
        await cache2.set("key", "value")

        # Close all
        await close_all_caches()

        # Registry should be cleared
        instances = list_cache_instances()
        assert len(instances) == 0

    async def test_reset_cache_factory(self) -> None:
        """Test resetting the cache factory."""
        # Create some instances
        create_cache(name="cache1")
        create_cache(name="cache2")

        assert len(list_cache_instances()) == 2

        # Reset factory
        reset_cache_factory()

        # Registry should be cleared
        assert len(list_cache_instances()) == 0

    async def test_clear_cache_registry_alias(self) -> None:
        """Test that clear_cache_registry is an alias for reset_cache_factory."""
        create_cache(name="cache1")
        assert len(list_cache_instances()) == 1

        clear_cache_registry()

        assert len(list_cache_instances()) == 0

    async def test_namespace_isolation(self) -> None:
        """Test that different namespaces are properly isolated."""
        config1 = CacheConfig(
            backend=CacheBackend.MEMORY,
            namespace="ns1",
        )
        config2 = CacheConfig(
            backend=CacheBackend.MEMORY,
            namespace="ns2",
        )

        cache1 = create_cache(config=config1, name="cache_ns1")
        cache2 = create_cache(config=config2, name="cache_ns2")

        # Same key in different namespaces
        await cache1.set("shared_key", "value1")
        await cache2.set("shared_key", "value2")

        # Should maintain separate values
        assert await cache1.get("shared_key") == "value1"
        assert await cache2.get("shared_key") == "value2"

    @pytest.mark.skipif(not redis_available, reason="Redis server not available")
    async def test_different_backends(self, test_redis_url: str) -> None:
        """Test creating caches with different backends."""
        mem_config = CacheConfig(
            backend=CacheBackend.MEMORY,
            namespace="mem",
        )
        redis_config = CacheConfig(
            backend=CacheBackend.REDIS,
            redis_url=test_redis_url,
            namespace="redis",
        )

        mem_cache = create_cache(config=mem_config, name="memory")
        redis_cache = create_cache(config=redis_config, name="redis")

        # Both should work independently
        await mem_cache.set("key", "mem_value")
        await redis_cache.set("key", "redis_value")

        assert await mem_cache.get("key") == "mem_value"
        assert await redis_cache.get("key") == "redis_value"

    async def test_concurrent_factory_calls(self) -> None:
        """Test that factory is thread-safe with concurrent calls."""
        import asyncio

        async def create_and_use_cache(name: str) -> str:
            cache = create_cache(name=name)
            await cache.set("key", name)
            value = await cache.get("key")
            return str(value)

        # Create multiple caches concurrently
        results = await asyncio.gather(
            create_and_use_cache("cache1"),
            create_and_use_cache("cache2"),
            create_and_use_cache("cache3"),
            create_and_use_cache("cache1"),  # Duplicate
        )

        assert results[0] == "cache1"
        assert results[1] == "cache2"
        assert results[2] == "cache3"
        assert results[3] == "cache1"

        # Should have 3 unique instances
        assert len(list_cache_instances()) == 3

    async def test_cache_stats_per_instance(self) -> None:
        """Test that cache statistics are tracked per instance."""
        cache1 = create_cache(name="stats1")
        cache2 = create_cache(name="stats2")

        # Perform operations on cache1
        await cache1.set("key1", "value1")
        await cache1.get("key1")

        # Perform operations on cache2
        await cache2.set("key2", "value2")
        await cache2.get("key2")
        await cache2.get("nonexistent")

        # Check stats are separate
        stats1 = await cache1.get_stats()
        stats2 = await cache2.get_stats()

        assert stats1["hits"] == 1
        assert stats1["misses"] == 0
        assert stats2["hits"] == 1
        assert stats2["misses"] == 1

    async def test_default_cache_name(self) -> None:
        """Test that 'default' is the default cache name."""
        cache1 = create_cache()
        cache2 = get_cache()

        # Should return the same instance
        assert cache1 is cache2

        instances = list_cache_instances()
        assert "default" in instances

    async def test_cache_persistence_across_gets(self) -> None:
        """Test that cache data persists across get_cache calls."""
        # Create and populate cache
        cache1 = create_cache(name="persistent")
        await cache1.set("key1", "value1")
        await cache1.set("key2", "value2")

        # Get the same cache later
        cache2 = get_cache("persistent")

        # Data should still be there
        assert await cache2.get("key1") == "value1"
        assert await cache2.get("key2") == "value2"

    @pytest.mark.skipif(not redis_available, reason="Redis server not available")
    async def test_mixed_backend_operations(self, test_redis_url: str) -> None:
        """Test operations with mixed memory and Redis backends."""
        mem_cache = create_cache(
            config=CacheConfig(backend=CacheBackend.MEMORY),
            name="mem",
        )
        redis_cache = create_cache(
            config=CacheConfig(
                backend=CacheBackend.REDIS,
                redis_url=test_redis_url,
            ),
            name="redis",
        )

        # Set data in both
        await mem_cache.set("shared", "from_memory")
        await redis_cache.set("shared", "from_redis")

        # Each should maintain its own value
        assert await mem_cache.get("shared") == "from_memory"
        assert await redis_cache.get("shared") == "from_redis"

        # Clear one shouldn't affect the other
        await mem_cache.clear()
        assert await mem_cache.get("shared") is None
        assert await redis_cache.get("shared") == "from_redis"

    async def test_factory_with_invalid_redis_url(self) -> None:
        """Test factory behavior with invalid Redis configuration."""
        config = CacheConfig(
            backend=CacheBackend.REDIS,
            redis_url="redis://invalid-host:9999/0",
            namespace="test",
        )

        # Should create the backend but may fail on first operation
        cache = create_cache(config=config, name="invalid_redis")
        assert cache is not None

        # Actual connection error occurs on first operation
        # (behavior depends on implementation)

    async def test_ttl_configuration(self) -> None:
        """Test that TTL configuration is properly applied."""
        config = CacheConfig(
            backend=CacheBackend.MEMORY,
            ttl_seconds=1,
        )

        cache = create_cache(config=config, name="ttl_test")

        # Set a value (should use default TTL of 1 second)
        await cache.set("key", "value", ttl=None)
        assert await cache.get("key") == "value"

        # Wait for expiration
        import asyncio

        await asyncio.sleep(1.1)

        # Should be expired
        assert await cache.get("key") is None

    async def test_max_size_configuration(self) -> None:
        """Test that max_size configuration is properly applied."""
        config = CacheConfig(
            backend=CacheBackend.MEMORY,
            max_size=3,
        )

        cache = create_cache(config=config, name="maxsize_test")

        # Fill cache to max
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.set("key3", "value3")

        # Add one more (should trigger eviction)
        await cache.set("key4", "value4")

        # Check that LRU eviction occurred
        stats = await cache.get_stats()
        assert stats["evictions"] >= 1
