"""
Seraph MCP — Cache Factory Integration Tests

Integration tests for the cache factory, covering:
- Backend selection via configuration
- Switching between memory and Redis backends
- Factory singleton behavior
- Configuration validation
- Real Redis integration (when available)
- Error handling for missing Redis configuration
"""

import os
from typing import Any

import pytest

from src.cache import close_all_caches, create_cache, reset_cache_factory
from src.cache.backends.memory import MemoryCacheBackend
from src.cache.backends.redis import RedisCacheBackend


@pytest.fixture(autouse=True)
async def cleanup_factory():
    """Clean up cache factory after each test."""
    yield
    await close_all_caches()
    reset_cache_factory()


@pytest.mark.asyncio
class TestCacheFactoryMemoryBackend:
    """Test cache factory with memory backend."""

    async def test_create_memory_backend_by_default(self, mock_env_memory):
        """Test that memory backend is created by default."""
        cache = create_cache()

        assert isinstance(cache, MemoryCacheBackend)
        assert cache.namespace == "test"

        stats = await cache.get_stats()
        assert stats["backend"] == "memory"

    async def test_create_memory_backend_explicit(self, mock_env_memory):
        """Test explicitly creating memory backend."""
        cache = create_cache()

        # Test basic operations
        await cache.set("key1", "value1")
        result = await cache.get("key1")
        assert result == "value1"

    async def test_memory_backend_respects_config(self, monkeypatch):
        """Test that memory backend respects configuration."""
        monkeypatch.setenv("CACHE_BACKEND", "memory")
        monkeypatch.setenv("CACHE_MAX_SIZE", "50")
        monkeypatch.setenv("CACHE_TTL_SECONDS", "7200")
        monkeypatch.setenv("CACHE_NAMESPACE", "testapp")

        cache = create_cache()

        assert cache.namespace == "testapp"
        assert cache.default_ttl == 7200
        assert cache.max_size == 50


@pytest.mark.asyncio
class TestCacheFactoryRedisBackend:
    """Test cache factory with Redis backend."""

    async def test_create_redis_backend(self, mock_env_redis, redis_client):
        """Test creating Redis backend via factory."""
        cache = create_cache()

        assert isinstance(cache, RedisCacheBackend)
        assert cache.namespace == "test"

        stats = await cache.get_stats()
        assert stats["backend"] == "redis"
        assert stats["connected"] is True

    async def test_redis_backend_basic_operations(self, mock_env_redis, redis_client):
        """Test basic operations with Redis backend."""
        cache = create_cache()

        # Set and get
        await cache.set("integration_key", {"data": "integration_value"})
        result = await cache.get("integration_key")
        assert result == {"data": "integration_value"}

        # Delete
        deleted = await cache.delete("integration_key")
        assert deleted is True

        result = await cache.get("integration_key")
        assert result is None

    async def test_redis_backend_respects_config(self, monkeypatch, test_redis_url):
        """Test that Redis backend respects configuration."""
        monkeypatch.setenv("CACHE_BACKEND", "redis")
        monkeypatch.setenv("REDIS_URL", test_redis_url)
        monkeypatch.setenv("REDIS_MAX_CONNECTIONS", "15")
        monkeypatch.setenv("CACHE_TTL_SECONDS", "1800")
        monkeypatch.setenv("CACHE_NAMESPACE", "prodapp")

        cache = create_cache()

        assert cache.namespace == "prodapp"
        assert cache.default_ttl == 1800

    async def test_redis_backend_without_url_raises_error(self, monkeypatch):
        """Test that Redis backend without URL raises error."""
        monkeypatch.setenv("CACHE_BACKEND", "redis")
        monkeypatch.delenv("REDIS_URL", raising=False)

        with pytest.raises((ValueError, RuntimeError)):
            create_cache()


@pytest.mark.asyncio
class TestCacheFactorySingleton:
    """Test cache factory singleton behavior."""

    async def test_factory_returns_same_instance(self, mock_env_memory):
        """Test that factory returns the same cache instance."""
        cache1 = create_cache()
        cache2 = create_cache()

        assert cache1 is cache2

    async def test_factory_state_persists(self, mock_env_memory):
        """Test that cache state persists across factory calls."""
        cache1 = create_cache()
        await cache1.set("persistent_key", "persistent_value")

        cache2 = create_cache()
        result = await cache2.get("persistent_key")

        assert result == "persistent_value"

    async def test_reset_factory_creates_new_instance(self, mock_env_memory):
        """Test that reset_cache_factory creates a new instance."""
        cache1 = create_cache()
        await cache1.set("key", "value")

        # Reset factory
        await close_all_caches()
        reset_cache_factory()

        cache2 = create_cache()

        # Should be a new instance
        assert cache1 is not cache2

        # Old data should be gone
        result = await cache2.get("key")
        assert result is None


@pytest.mark.asyncio
class TestCacheFactoryBackendSwitching:
    """Test switching between backends."""

    async def test_switch_from_memory_to_redis(
        self,
        monkeypatch,
        test_redis_url,
        redis_client
    ):
        """Test switching from memory to Redis backend."""
        # Start with memory
        monkeypatch.setenv("CACHE_BACKEND", "memory")
        monkeypatch.setenv("CACHE_NAMESPACE", "switch_test")

        cache1 = create_cache()
        assert isinstance(cache1, MemoryCacheBackend)

        await cache1.set("mem_key", "mem_value")

        # Switch to Redis
        await close_all_caches()
        reset_cache_factory()

        monkeypatch.setenv("CACHE_BACKEND", "redis")
        monkeypatch.setenv("REDIS_URL", test_redis_url)
        monkeypatch.setenv("CACHE_NAMESPACE", "switch_test")

        cache2 = create_cache()
        assert isinstance(cache2, RedisCacheBackend)

        # Old memory data is not available in Redis
        result = await cache2.get("mem_key")
        assert result is None

        # Can set new data in Redis
        await cache2.set("redis_key", "redis_value")
        result = await cache2.get("redis_key")
        assert result == "redis_value"

    async def test_switch_from_redis_to_memory(
        self,
        monkeypatch,
        test_redis_url,
        redis_client
    ):
        """Test switching from Redis to memory backend."""
        # Start with Redis
        monkeypatch.setenv("CACHE_BACKEND", "redis")
        monkeypatch.setenv("REDIS_URL", test_redis_url)
        monkeypatch.setenv("CACHE_NAMESPACE", "switch_test")

        cache1 = create_cache()
        assert isinstance(cache1, RedisCacheBackend)

        await cache1.set("redis_key", "redis_value")

        # Switch to memory
        await close_all_caches()
        reset_cache_factory()

        monkeypatch.setenv("CACHE_BACKEND", "memory")
        monkeypatch.setenv("CACHE_NAMESPACE", "switch_test")

        cache2 = create_cache()
        assert isinstance(cache2, MemoryCacheBackend)

        # Old Redis data is not available in memory
        result = await cache2.get("redis_key")
        assert result is None

        # Can set new data in memory
        await cache2.set("mem_key", "mem_value")
        result = await cache2.get("mem_key")
        assert result == "mem_value"


@pytest.mark.asyncio
class TestCacheFactoryNamespaceIsolation:
    """Test namespace isolation across different cache instances."""

    async def test_memory_namespace_isolation(self, monkeypatch):
        """Test that different namespaces are isolated in memory."""
        # Create first cache with namespace 'app1'
        monkeypatch.setenv("CACHE_BACKEND", "memory")
        monkeypatch.setenv("CACHE_NAMESPACE", "app1")

        cache1 = create_cache()
        await cache1.set("shared_key", "app1_value")

        # Reset and create second cache with namespace 'app2'
        await close_all_caches()
        reset_cache_factory()

        monkeypatch.setenv("CACHE_NAMESPACE", "app2")
        cache2 = create_cache()

        # Different namespace, so key shouldn't exist
        result = await cache2.get("shared_key")
        assert result is None

        # Set in app2
        await cache2.set("shared_key", "app2_value")
        result = await cache2.get("shared_key")
        assert result == "app2_value"

    async def test_redis_namespace_isolation(
        self,
        monkeypatch,
        test_redis_url,
        redis_client
    ):
        """Test that different namespaces are isolated in Redis."""
        # Create first cache with namespace 'service1'
        monkeypatch.setenv("CACHE_BACKEND", "redis")
        monkeypatch.setenv("REDIS_URL", test_redis_url)
        monkeypatch.setenv("CACHE_NAMESPACE", "service1")

        cache1 = create_cache()
        await cache1.set("shared_key", "service1_value")

        # Reset and create second cache with namespace 'service2'
        await close_all_caches()
        reset_cache_factory()

        monkeypatch.setenv("CACHE_NAMESPACE", "service2")
        cache2 = create_cache()

        # Different namespace, so key shouldn't exist
        result = await cache2.get("shared_key")
        assert result is None

        # Set in service2
        await cache2.set("shared_key", "service2_value")
        result = await cache2.get("shared_key")
        assert result == "service2_value"

        # Verify service1 data is still intact
        await close_all_caches()
        reset_cache_factory()

        monkeypatch.setenv("CACHE_NAMESPACE", "service1")
        cache1_new = create_cache()
        result = await cache1_new.get("shared_key")
        assert result == "service1_value"


@pytest.mark.asyncio
class TestCacheFactoryCleanup:
    """Test resource cleanup via factory."""

    async def test_close_all_caches_memory(self, mock_env_memory):
        """Test closing all memory cache instances."""
        cache = create_cache()
        await cache.set("key", "value")

        await close_all_caches()

        # After close, cache should be cleared
        stats = await cache.get_stats()
        assert stats["size"] == 0

    async def test_close_all_caches_redis(self, mock_env_redis, redis_client):
        """Test closing all Redis cache instances."""
        cache = create_cache()
        await cache.set("key", "value")

        # Should not raise
        await close_all_caches()

    async def test_close_all_caches_idempotent(self, mock_env_memory):
        """Test that close_all_caches can be called multiple times."""
        create_cache()

        await close_all_caches()
        await close_all_caches()  # Should not raise


@pytest.mark.asyncio
class TestCacheFactoryErrorHandling:
    """Test error handling in cache factory."""

    async def test_invalid_backend_uses_default(self, monkeypatch):
        """Test that invalid backend falls back to default."""
        monkeypatch.setenv("CACHE_BACKEND", "invalid_backend")
        monkeypatch.setenv("CACHE_NAMESPACE", "test")

        # Should fall back to memory backend
        cache = create_cache()
        assert isinstance(cache, MemoryCacheBackend)

    async def test_redis_connection_failure_handling(self, monkeypatch):
        """Test handling of Redis connection failures."""
        monkeypatch.setenv("CACHE_BACKEND", "redis")
        monkeypatch.setenv("REDIS_URL", "redis://nonexistent:9999/0")
        monkeypatch.setenv("REDIS_SOCKET_TIMEOUT", "1")
        monkeypatch.setenv("CACHE_NAMESPACE", "test")

        # Factory should create backend, but operations will fail
        cache = create_cache()
        assert isinstance(cache, RedisCacheBackend)

        # Operations should handle connection errors gracefully
        # (depending on implementation, may raise or return None)


@pytest.mark.asyncio
class TestCacheFactoryConfiguration:
    """Test configuration handling in cache factory."""

    async def test_config_from_environment(self, monkeypatch):
        """Test that factory reads configuration from environment."""
        monkeypatch.setenv("CACHE_BACKEND", "memory")
        monkeypatch.setenv("CACHE_TTL_SECONDS", "5400")
        monkeypatch.setenv("CACHE_MAX_SIZE", "250")
        monkeypatch.setenv("CACHE_NAMESPACE", "envtest")

        cache = create_cache()

        assert cache.namespace == "envtest"
        assert cache.default_ttl == 5400
        assert cache.max_size == 250

    async def test_default_values_when_not_configured(self, monkeypatch):
        """Test default values when environment not set."""
        # Clear relevant env vars
        for key in ["CACHE_BACKEND", "CACHE_TTL_SECONDS", "CACHE_MAX_SIZE", "CACHE_NAMESPACE"]:
            monkeypatch.delenv(key, raising=False)

        cache = create_cache()

        # Should use defaults
        assert isinstance(cache, MemoryCacheBackend)
        assert cache.namespace == "seraph"
