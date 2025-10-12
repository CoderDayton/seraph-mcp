# Seraph MCP — Test Suite

Comprehensive test suite for the Seraph MCP platform, ensuring quality, reliability, and SDD compliance.

---

## Overview

The test suite covers:
- **Unit tests**: Isolated testing of individual components
- **Integration tests**: Testing component interactions
- **Smoke tests**: Quick validation of critical paths
- **Performance tests**: Optional performance benchmarking

---

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── unit/                    # Unit tests (isolated components)
│   ├── cache/
│   │   ├── test_memory_backend.py
│   │   └── test_redis_backend.py
│   ├── config/
│   ├── observability/
│   └── test_errors.py
├── integration/             # Integration tests (component interaction)
│   ├── test_cache_factory.py
│   └── test_server_lifecycle.py
└── README.md               # This file
```

---

## Running Tests

### All Tests

```bash
# Run all tests with coverage
uv run pytest --cov=src --cov-report=term --cov-report=html

# Run with verbose output
uv run pytest -v

# Run with debug output
uv run pytest -vv -s
```

### Unit Tests Only

```bash
uv run pytest tests/unit/
```

### Integration Tests Only

```bash
uv run pytest tests/integration/
```

### Specific Test File

```bash
uv run pytest tests/unit/cache/test_redis_backend.py
```

### Specific Test Function

```bash
uv run pytest tests/unit/cache/test_redis_backend.py::TestRedisCacheBackendBasicOperations::test_get_existing_key
```

### With Coverage Report

```bash
# Terminal report
uv run pytest --cov=src --cov-report=term-missing

# HTML report (opens in browser)
uv run pytest --cov=src --cov-report=html
open htmlcov/index.html
```

---

## Test Categories

### Unit Tests

Test individual components in isolation:

```python
@pytest.mark.asyncio
async def test_cache_set_and_get():
    """Test basic cache operations."""
    backend = MemoryCacheBackend()
    await backend.set("key", "value")
    result = await backend.get("key")
    assert result == "value"
```

**Coverage:**
- Cache backends (memory and Redis)
- Configuration loading and validation
- Error types and handling
- Observability adapter
- Utility functions

### Integration Tests

Test component interactions:

```python
@pytest.mark.asyncio
async def test_cache_factory_with_redis(mock_env_redis, redis_client):
    """Test cache factory creates Redis backend correctly."""
    cache = create_cache()
    assert isinstance(cache, RedisCacheBackend)
    
    await cache.set("key", "value")
    result = await cache.get("key")
    assert result == "value"
```

**Coverage:**
- Cache factory with different backends
- Server lifecycle (startup/shutdown)
- MCP tool invocation
- Configuration switching
- Namespace isolation

### Smoke Tests

Quick validation of critical paths:

```bash
# Test server startup
CACHE_BACKEND=memory uv run python -c "
import asyncio
from src.server import initialize_server, cleanup_server

async def test():
    await initialize_server()
    print('✓ Server initialized')
    await cleanup_server()
    print('✓ Server cleanup complete')

asyncio.run(test())
"
```

---

## Prerequisites

### Required Services

**Redis** (for Redis backend tests):
```bash
# Using Docker
docker run -d --name test-redis -p 6379:6379 redis:7-alpine

# Or use docker-compose
docker-compose up -d redis
```

### Environment Variables

Set test-specific environment variables:

```bash
# Test environment
export ENVIRONMENT=test
export LOG_LEVEL=DEBUG

# Redis connection for tests
export TEST_REDIS_URL=redis://localhost:6379/15

# Cache configuration
export CACHE_BACKEND=redis
export CACHE_NAMESPACE=test
```

---

## Fixtures

### Shared Fixtures (conftest.py)

**`event_loop`**: Session-scoped event loop
```python
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
```

**`redis_client`**: Redis client for testing
```python
@pytest.fixture
async def redis_client(test_redis_url):
    """Create Redis client for testing."""
    client = Redis.from_url(test_redis_url, decode_responses=True)
    await client.ping()
    await client.flushdb()  # Clear before test
    yield client
    await client.flushdb()  # Clear after test
    await client.close()
```

**`mock_env_memory`**: Memory cache environment
```python
@pytest.fixture
def mock_env_memory(monkeypatch):
    """Set environment for memory cache backend."""
    monkeypatch.setenv("CACHE_BACKEND", "memory")
    monkeypatch.setenv("CACHE_MAX_SIZE", "100")
    monkeypatch.setenv("CACHE_TTL_SECONDS", "3600")
```

**`mock_env_redis`**: Redis cache environment
```python
@pytest.fixture
def mock_env_redis(monkeypatch, test_redis_url):
    """Set environment for Redis cache backend."""
    monkeypatch.setenv("CACHE_BACKEND", "redis")
    monkeypatch.setenv("REDIS_URL", test_redis_url)
```

---

## Writing New Tests

### Test Naming Convention

- **File names**: `test_<module>.py`
- **Class names**: `Test<Feature>`
- **Function names**: `test_<behavior>`

```python
# tests/unit/cache/test_new_feature.py

@pytest.mark.asyncio
class TestNewFeature:
    """Test new feature functionality."""
    
    async def test_feature_basic_case(self):
        """Test basic case for new feature."""
        pass
    
    async def test_feature_edge_case(self):
        """Test edge case for new feature."""
        pass
    
    async def test_feature_error_handling(self):
        """Test error handling for new feature."""
        pass
```

### Test Structure (AAA Pattern)

```python
async def test_example():
    """Test description."""
    # Arrange: Set up test data and conditions
    backend = MemoryCacheBackend()
    test_data = {"key": "value"}
    
    # Act: Perform the operation
    await backend.set("test_key", test_data)
    result = await backend.get("test_key")
    
    # Assert: Verify the outcome
    assert result == test_data
```

### Async Tests

Always use `@pytest.mark.asyncio` for async tests:

```python
@pytest.mark.asyncio
async def test_async_operation():
    """Test async operation."""
    result = await async_function()
    assert result is not None
```

### Mocking

Use pytest fixtures and monkeypatch for mocking:

```python
@pytest.fixture
def mock_redis_error(monkeypatch):
    """Mock Redis connection error."""
    def mock_ping():
        raise ConnectionError("Redis unavailable")
    
    monkeypatch.setattr("redis.asyncio.Redis.ping", mock_ping)
```

---

## Coverage Requirements

Per SDD.md, core code must maintain **≥85% test coverage**.

### Check Coverage

```bash
# Run with coverage
uv run pytest --cov=src --cov-report=term-missing

# Generate HTML report
uv run pytest --cov=src --cov-report=html
open htmlcov/index.html

# Fail if below threshold
uv run pytest --cov=src --cov-fail-under=85
```

### Coverage by Module

```bash
uv run coverage report --include="src/cache/*"
uv run coverage report --include="src/config/*"
uv run coverage report --include="src/observability/*"
```

---

## Continuous Integration

Tests run automatically in CI/CD pipeline (`.github/workflows/ci.yml`):

1. **Lint & Format**: Ruff checks
2. **Type Check**: MyPy validation
3. **Unit Tests**: All unit tests with coverage
4. **Integration Tests**: Component interaction tests
5. **Security Scan**: Bandit, Safety, secret detection
6. **Smoke Tests**: Critical path validation
7. **Coverage Gate**: Fails if coverage < 85%

---

## Performance Testing

Optional performance tests (not run by default):

```bash
# Run performance tests
uv run pytest tests/performance/ --benchmark-only

# Save benchmark results
uv run pytest tests/performance/ --benchmark-autosave

# Compare with baseline
uv run pytest tests/performance/ --benchmark-compare
```

---

## Troubleshooting

### Redis Connection Failures

**Problem**: Tests fail with Redis connection errors.

**Solution**:
```bash
# Check Redis is running
redis-cli ping

# Start Redis if not running
docker run -d -p 6379:6379 redis:7-alpine

# Set correct Redis URL
export TEST_REDIS_URL=redis://localhost:6379/15
```

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'src'`

**Solution**:
```bash
# Install package in editable mode
uv pip install -e .

# Or set PYTHONPATH
export PYTHONPATH=$(pwd):$PYTHONPATH
```

### Async Warnings

**Problem**: `RuntimeWarning: coroutine was never awaited`

**Solution**:
```python
# Use @pytest.mark.asyncio decorator
@pytest.mark.asyncio
async def test_async_function():
    result = await async_function()  # Don't forget await!
```

### Fixture Cleanup

**Problem**: Tests interfere with each other.

**Solution**:
```python
@pytest.fixture(autouse=True)
async def cleanup():
    """Clean up after each test."""
    yield
    await close_all_caches()
    reset_cache_factory()
```

---

## Best Practices

### 1. Test Independence

Each test should be independent and not rely on other tests:

```python
# ✅ Good: Independent test
async def test_cache_set():
    cache = MemoryCacheBackend()
    await cache.set("key", "value")
    assert await cache.exists("key")

# ❌ Bad: Depends on previous test state
async def test_cache_get():
    # Assumes "key" exists from previous test
    result = await cache.get("key")
```

### 2. Clear Test Names

Use descriptive names that explain what is being tested:

```python
# ✅ Good
async def test_cache_returns_none_for_expired_keys()

# ❌ Bad
async def test_cache_1()
```

### 3. One Assertion Per Test (when possible)

Focus each test on a single behavior:

```python
# ✅ Good
async def test_set_stores_value():
    cache = MemoryCacheBackend()
    success = await cache.set("key", "value")
    assert success is True

async def test_get_retrieves_stored_value():
    cache = MemoryCacheBackend()
    await cache.set("key", "value")
    result = await cache.get("key")
    assert result == "value"

# ❌ Less ideal
async def test_set_and_get():
    cache = MemoryCacheBackend()
    assert await cache.set("key", "value")
    assert await cache.get("key") == "value"
    assert await cache.exists("key")
    # Testing too many things at once
```

### 4. Use Fixtures for Common Setup

```python
@pytest.fixture
async def cache_with_data():
    """Cache pre-populated with test data."""
    cache = MemoryCacheBackend()
    await cache.set("key1", "value1")
    await cache.set("key2", "value2")
    return cache

async def test_something(cache_with_data):
    result = await cache_with_data.get("key1")
    assert result == "value1"
```

### 5. Test Error Cases

Don't just test the happy path:

```python
async def test_get_nonexistent_key_returns_none():
    cache = MemoryCacheBackend()
    result = await cache.get("nonexistent")
    assert result is None

async def test_invalid_ttl_raises_error():
    cache = MemoryCacheBackend()
    with pytest.raises(ValueError):
        await cache.set("key", "value", ttl="invalid")
```

---

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-asyncio Documentation](https://pytest-asyncio.readthedocs.io/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [SDD.md](../docs/SDD.md) — System design document
- [PLUGIN_GUIDE.md](../docs/PLUGIN_GUIDE.md) — Plugin testing guidelines

---

## Contributing

When adding new features:

1. Write tests **before** implementation (TDD)
2. Ensure tests pass locally
3. Maintain ≥85% coverage for core modules
4. Add integration tests for component interactions
5. Update this README if adding new test patterns

---

**Happy Testing! 🧪**