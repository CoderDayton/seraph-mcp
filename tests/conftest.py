"""
Seraph MCP — Test Configuration and Shared Fixtures

Provides pytest configuration and shared fixtures for unit and integration tests.
Python 3.12+ with modern type hints and async patterns.
"""

import asyncio
import os
from collections.abc import AsyncGenerator, Generator
from typing import Any

import pytest
import pytest_asyncio
from redis.asyncio import Redis

# Set test environment
os.environ["ENVIRONMENT"] = "test"
os.environ["LOG_LEVEL"] = "DEBUG"


# Redis availability checker
def is_redis_available() -> bool:
    """Check if Redis server is available for testing."""
    import socket

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(("localhost", 6379))
        sock.close()
        return result == 0
    except Exception:
        return False


# Skip marker for Redis tests
redis_available = pytest.mark.skipif(not is_redis_available(), reason="Redis server not available")


@pytest.fixture(scope="session")
def event_loop_policy() -> asyncio.AbstractEventLoopPolicy:
    """Get the event loop policy for the test session."""
    return asyncio.get_event_loop_policy()


@pytest.fixture(scope="function")
def event_loop(event_loop_policy: asyncio.AbstractEventLoopPolicy) -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for each test function."""
    loop = event_loop_policy.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_redis_url() -> str:
    """Get Redis URL for testing (database 15 for isolation)."""
    return os.environ.get("TEST_REDIS_URL", "redis://localhost:6379/15")


@pytest_asyncio.fixture
async def redis_client(test_redis_url: str) -> AsyncGenerator[Redis, None]:  # type: ignore[misc]
    """
    Create a Redis client for testing.

    Automatically skips tests if Redis is not available.
    Clears the test database before and after each test.
    """
    client: Redis = Redis.from_url(test_redis_url, decode_responses=False)  # type: ignore[call-arg]

    # Ensure we can connect
    try:
        await client.ping()
    except Exception as e:
        await client.close()
        pytest.skip(f"Redis not available for testing: {e}")

    # Clear test database before test
    await client.flushdb()

    yield client

    # Cleanup after test
    try:
        await client.flushdb()
    finally:
        await client.close()


@pytest.fixture  # type: ignore[misc]
def mock_env_memory(monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[type-arg]
    """Set environment variables for memory cache backend."""
    monkeypatch.setenv("CACHE_BACKEND", "memory")
    monkeypatch.setenv("CACHE_MAX_SIZE", "100")
    monkeypatch.setenv("CACHE_TTL_SECONDS", "3600")
    monkeypatch.setenv("CACHE_NAMESPACE", "test")


@pytest.fixture  # type: ignore[misc]
def mock_env_redis(monkeypatch: pytest.MonkeyPatch, test_redis_url: str) -> None:  # type: ignore[type-arg]
    """Set environment variables for Redis cache backend."""
    if not is_redis_available():
        pytest.skip("Redis not available")
    monkeypatch.setenv("CACHE_BACKEND", "redis")
    monkeypatch.setenv("REDIS_URL", test_redis_url)
    monkeypatch.setenv("REDIS_MAX_CONNECTIONS", "5")
    monkeypatch.setenv("REDIS_SOCKET_TIMEOUT", "2")
    monkeypatch.setenv("CACHE_TTL_SECONDS", "3600")
    monkeypatch.setenv("CACHE_NAMESPACE", "test")


@pytest.fixture  # type: ignore[misc]
def sample_cache_data() -> dict[str, Any]:
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


@pytest.fixture  # type: ignore[misc]
def sample_text_short() -> str:
    """Short text sample for context optimization tests."""
    return """
    Machine learning is a subset of artificial intelligence that focuses on
    developing systems that can learn from data. These systems improve their
    performance over time without being explicitly programmed.
    """


@pytest.fixture  # type: ignore[misc]
def sample_text_long() -> str:
    """Long text sample for context optimization tests."""
    return (
        """
    Artificial intelligence (AI) has revolutionized numerous industries and continues
    to shape the future of technology. Machine learning, a crucial subset of AI,
    enables computers to learn from data and make decisions without explicit programming.

    Deep learning, a specialized branch of machine learning, uses neural networks with
    multiple layers to process complex patterns in large datasets. These networks can
    recognize images, understand natural language, and even generate creative content.

    Natural language processing (NLP) is another critical area of AI that focuses on
    the interaction between computers and human language. NLP powers applications like
    chatbots, translation services, and sentiment analysis tools.

    The ethical implications of AI development are increasingly important. Issues like
    bias in algorithms, privacy concerns, and the potential impact on employment require
    careful consideration. Responsible AI development prioritizes transparency, fairness,
    and accountability.

    As AI continues to advance, the collaboration between humans and machines becomes
    more sophisticated. The goal is not to replace human intelligence but to augment
    it, enabling us to solve complex problems more effectively.
    """
        * 3
    )  # Repeat to make it longer


@pytest.fixture  # type: ignore[misc]
def sample_compression_config() -> dict[str, Any]:
    """Sample configuration for compression tests."""
    return {
        "enabled": True,
        "method": "auto",
        "quality_threshold": 0.85,
        "max_tokens": 4000,
        "cache_enabled": True,
    }


@pytest.fixture  # type: ignore[misc]
def mock_ai_provider() -> Any:
    """Mock AI provider for testing context optimization."""
    from unittest.mock import AsyncMock, MagicMock

    mock = MagicMock()
    mock.generate = AsyncMock(return_value="This is compressed text that preserves meaning.")
    mock.count_tokens = MagicMock(return_value=50)
    return mock


@pytest.fixture(autouse=True)  # type: ignore[misc]
def reset_cache_factory() -> Generator[None, None, None]:
    """Reset cache factory after each test to prevent state leakage."""
    yield
    # Import here to avoid circular imports
    try:
        from src.cache.factory import reset_cache_factory

        reset_cache_factory()
    except ImportError:
        pass


@pytest.fixture  # type: ignore[misc]
def temp_cache_dir(tmp_path: Any) -> str:
    """Create a temporary directory for cache testing."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return str(cache_dir)
