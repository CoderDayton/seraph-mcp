# Seraph MCP Test Suite

Comprehensive test suite for the Seraph MCP project, covering all major components with modern Python 3.12+ async patterns and type hints.

## Overview

The test suite is organized into **unit tests** and **integration tests**, providing thorough coverage of:
- Cache backends (Memory and Redis)
- Cache factory and lifecycle management
- Context optimization components
- Seraph compression system
- Configuration and error handling

## Test Structure

```
tests/
├── conftest.py                              # Shared fixtures and configuration
├── unit/                                    # Unit tests (isolated component testing)
│   ├── cache/
│   │   ├── __init__.py
│   │   ├── test_memory_backend.py          # Memory cache backend tests
│   │   └── test_redis_backend.py           # Redis cache backend tests
│   └── context_optimization/
│       ├── __init__.py
│       └── test_seraph_compression.py      # Seraph compression tests
└── integration/                             # Integration tests (component interaction)
    └── test_cache_factory.py               # Cache factory integration tests
```

## Running Tests

### Run All Tests
```bash
pytest tests/
```

### Run Specific Test Categories
```bash
# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# Cache tests only
pytest tests/unit/cache/

# Context optimization tests only
pytest tests/unit/context_optimization/
```

### Run Specific Test Files
```bash
pytest tests/unit/cache/test_memory_backend.py
pytest tests/unit/cache/test_redis_backend.py
pytest tests/integration/test_cache_factory.py
```

### Run with Coverage
```bash
pytest tests/ --cov=src --cov-report=html
```

### Run with Verbose Output
```bash
pytest tests/ -v
```

## Test Requirements

### Core Dependencies
- `pytest` - Testing framework
- `pytest-asyncio` - Async test support
- `pytest-cov` - Coverage reporting

### Optional Dependencies
- `redis` - Required for Redis backend tests
- Redis server running on `localhost:6379` (or set `TEST_REDIS_URL` env var)

### Environment Variables
- `TEST_REDIS_URL` - Redis connection URL for tests (default: `redis://localhost:6379/15`)
- `ENVIRONMENT` - Set to `test` automatically by conftest
- `LOG_LEVEL` - Set to `DEBUG` automatically by conftest

## Test Coverage

### Cache System
- **Memory Backend** (`test_memory_backend.py`)
  - Basic CRUD operations (get, set, delete, exists)
  - TTL and expiration handling
  - LRU eviction when max_size reached
  - Batch operations (get_many, set_many, delete_many)
  - Namespace isolation
  - Statistics tracking
  - Concurrent operations
  - Edge cases (empty values, Unicode, special characters)

- **Redis Backend** (`test_redis_backend.py`)
  - All Memory Backend tests plus:
  - Redis-specific connection handling
  - Redis persistence across restarts
  - Redis namespace isolation with SCAN
  - Error handling for connection failures
  - Large batch operations

- **Cache Factory** (`test_cache_factory.py`)
  - Factory pattern and singleton behavior
  - Multiple named cache instances
  - Configuration handling
  - Backend selection (Memory vs Redis)
  - Lifecycle management (create, get, close)
  - Namespace isolation across instances
  - Concurrent factory calls
  - Mixed backend operations

### Context Optimization
- **Seraph Compression** (`test_seraph_compression.py`)
  - Utility functions (token counting, hashing, simhash)
  - BM25 ranking algorithm
  - Multi-layer compression (L1/L2/L3)
  - Deterministic compression
  - Compression caching
  - Quality preservation
  - Various text types (short, long, Unicode, code blocks)
  - Edge cases (empty text, whitespace, special characters)

## Fixtures

### Shared Fixtures (conftest.py)
- `event_loop_policy` - Event loop policy for async tests
- `event_loop` - Event loop for each test function
- `test_redis_url` - Redis URL for testing (database 15)
- `redis_client` - Configured Redis client with auto-cleanup
- `mock_env_memory` - Environment variables for memory cache
- `mock_env_redis` - Environment variables for Redis cache
- `sample_cache_data` - Sample data for cache tests
- `sample_text_short` - Short text for compression tests
- `sample_text_long` - Long text for compression tests
- `sample_compression_config` - Configuration for compression tests
- `mock_ai_provider` - Mock AI provider for optimization tests
- `reset_cache_factory` - Auto-reset cache factory after each test
- `temp_cache_dir` - Temporary directory for cache tests

## Best Practices

### Test Organization
- One test class per component/feature
- Descriptive test names: `test_<action>_<condition>`
- Use fixtures for setup/teardown
- Group related tests in classes

### Async Testing
- All async tests use `async def` and `await`
- Use `pytest.mark.asyncio` (handled by pytest-asyncio)
- Proper cleanup in fixtures with `yield`

### Assertions
- Use specific assertions (`assert x == y`, not `assert x`)
- Test both success and failure cases
- Verify stats/metadata after operations

### Mocking
- Mock external dependencies (AI providers, network calls)
- Use `pytest.MonkeyPatch` for environment variables
- Create minimal, focused mocks

### Coverage Goals
- Aim for >80% code coverage
- Focus on critical paths and edge cases
- Don't sacrifice readability for coverage

## Troubleshooting

### Redis Connection Issues
If Redis tests fail with connection errors:
1. Ensure Redis is running: `redis-cli ping`
2. Check connection: `redis-cli -h localhost -p 6379`
3. Set custom URL: `export TEST_REDIS_URL=redis://your-host:6379/15`
4. Tests auto-skip if Redis unavailable

### Import Errors
If you see import errors:
1. Install test dependencies: `pip install -e ".[dev]"`
2. Ensure you're in the project root
3. Activate virtual environment: `source .venv/bin/activate`

### Async Warnings
If you see async warnings or errors:
1. Ensure `pytest-asyncio` is installed
2. Check that async tests use `async def`
3. Verify fixtures use proper async/await patterns

### Type Checker Warnings
Type checker warnings about pytest decorators are expected and can be ignored:
- `Untyped function decorator obscures type of function`
- These are normal pytest fixture behavior

## Contributing

When adding new tests:
1. Follow existing patterns and naming conventions
2. Add fixtures to `conftest.py` if reusable
3. Include docstrings explaining what's being tested
4. Test both happy paths and error conditions
5. Run full test suite before committing
6. Update this README if adding new test categories

## Continuous Integration

Tests run automatically on:
- Pull request creation/updates
- Pushes to main branch
- Manual workflow dispatch

CI configuration: `.github/workflows/test.yml`

## Performance

Test suite performance targets:
- Full suite: < 30 seconds
- Unit tests: < 10 seconds
- Integration tests: < 20 seconds
- Individual test: < 1 second (except long-running integration tests)

Slow tests are marked with `@pytest.mark.slow` for optional exclusion:
```bash
pytest tests/ -m "not slow"
```

## Additional Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-asyncio documentation](https://pytest-asyncio.readthedocs.io/)
- [Seraph MCP Contributing Guide](../CONTRIBUTING.md)
- [Seraph MCP System Design Document](../docs/SDD.md)
