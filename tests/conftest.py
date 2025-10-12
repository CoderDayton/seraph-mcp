"""
Seraph MCP — Test Configuration and Shared Fixtures

Provides pytest configuration and shared fixtures for unit and integration tests.
"""

import asyncio
import os
from typing import AsyncGenerator, Generator

import pytest
from redis.asyncio import Redis

# Set test environment
os.environ["ENVIRONMENT"] = "test"
os.environ["LOG_LEVEL"] = "DEBUG"


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_redis_url() -> str:
    """Get Redis URL for testing."""
    return os.environ.get("TEST_REDIS_URL", "redis://localhost:6379/15")


@pytest.fixture
async def redis_client(test_redis_url: str) -> AsyncGenerator[Redis, None]:
    """Create a Redis client for testing."""
    client = Redis.from_url(test_redis_url, decode_responses=True)

    # Ensure we can connect
    try:
        await client.ping()
    except Exception as e:
        pytest.skip(f"Redis not available for testing: {e}")

    # Clear test database before test
    await client.flushdb()

    yield client

    # Cleanup after test
    await client.flushdb()
    await client.close()
    await client.connection_pool.disconnect()


@pytest.fixture
def mock_env_memory(monkeypatch):
    """Set environment variables for memory cache backend."""
    monkeypatch.setenv("CACHE_BACKEND", "memory")
    monkeypatch.setenv("CACHE_MAX_SIZE", "100")
    monkeypatch.setenv("CACHE_TTL_SECONDS", "3600")
    monkeypatch.setenv("CACHE_NAMESPACE", "test")


@pytest.fixture
def mock_env_redis(monkeypatch, test_redis_url):
    """Set environment variables for Redis cache backend."""
    monkeypatch.setenv("CACHE_BACKEND", "redis")
    monkeypatch.setenv("REDIS_URL", test_redis_url)
    monkeypatch.setenv("REDIS_MAX_CONNECTIONS", "5")
    monkeypatch.setenv("REDIS_SOCKET_TIMEOUT", "2")
    monkeypatch.setenv("CACHE_TTL_SECONDS", "3600")
    monkeypatch.setenv("CACHE_NAMESPACE", "test")


@pytest.fixture
def sample_cache_data():
    """Sample data for cache testing."""
    return {
        "simple_string": "hello",
        "simple_int": 42,
        "simple_float": 3.14,
        "simple_bool": True,
        "simple_none": None,
        "complex_dict": {
            "nested": {
                "key": "value",
                "number": 123,
                "list": [1, 2, 3],
            }
        },
        "complex_list": [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
        ],
    }
