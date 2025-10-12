# Seraph MCP — Testing Quick Start Guide

**Get up and running with tests in 5 minutes** ⚡

---

## Prerequisites

- Python 3.10+
- Redis (for integration tests)

---

## Quick Start

### 1. Install Dependencies

```bash
# Using uv (recommended)
uv sync --all-extras

# Or using pip
pip install -e ".[dev]"
```

### 2. Start Redis (for integration tests)

```bash
# Option 1: Docker (recommended)
docker run -d --name test-redis -p 6379:6379 redis:7-alpine

# Option 2: Docker Compose
docker-compose up -d redis

# Option 3: Local Redis
redis-server --daemonize yes
```

### 3. Set Environment Variables

```bash
export ENVIRONMENT=test
export TEST_REDIS_URL=redis://localhost:6379/15
```

### 4. Run Tests

```bash
# Run all tests with coverage
pytest --cov=src --cov-report=term-missing

# Expected output:
# ✓ 100+ tests pass
# ✓ Coverage ≥85%
```

---

## Common Commands

### Run Specific Test Types

```bash
# Unit tests only (fast, no Redis needed)
pytest tests/unit/

# Integration tests only (requires Redis)
pytest tests/integration/

# Specific test file
pytest tests/unit/cache/test_redis_backend.py

# Specific test function
pytest tests/unit/cache/test_redis_backend.py::TestRedisCacheBackendBasicOperations::test_get_existing_key
```

### Coverage Reports

```bash
# Terminal report with missing lines
pytest --cov=src --cov-report=term-missing

# HTML report (opens in browser)
pytest --cov=src --cov-report=html
open htmlcov/index.html

# XML report (for CI)
pytest --cov=src --cov-report=xml
```

### Verbose Output

```bash
# Verbose
pytest -v

# Very verbose with print statements
pytest -vv -s

# Show locals on failure
pytest -l
```

---

## Code Quality Checks

### Linting

```bash
# Check code style
ruff check .

# Auto-fix issues
ruff check --fix .
```

### Formatting

```bash
# Check formatting
ruff format --check .

# Auto-format
ruff format .
```

### Type Checking

```bash
# Check types
mypy src/
```

### Security Scanning

```bash
# Python security issues
bandit -r src/

# Dependency vulnerabilities
safety check

# Secrets detection
detect-secrets scan
```

---

## Pre-commit Hooks

### Install Hooks

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Now hooks run automatically on git commit
```

### Run Manually

```bash
# Run on all files
pre-commit run --all-files

# Run specific hook
pre-commit run ruff --all-files
```

---

## Troubleshooting

### Redis Connection Failed

**Error:** `ConnectionRefusedError: [Errno 111] Connection refused`

**Solution:**
```bash
# Check Redis is running
redis-cli ping
# Expected: PONG

# If not running, start it
docker run -d -p 6379:6379 redis:7-alpine
```

### Import Errors

**Error:** `ModuleNotFoundError: No module named 'src'`

**Solution:**
```bash
# Install package in editable mode
pip install -e .

# Or set PYTHONPATH
export PYTHONPATH=$(pwd):$PYTHONPATH
```

### Coverage Too Low

**Error:** `FAIL Required test coverage of 85% not reached`

**Solution:**
```bash
# See which files need tests
pytest --cov=src --cov-report=term-missing

# Add tests for files with low coverage
# Files should be in tests/unit/ or tests/integration/
```

### Tests Hanging

**Issue:** Tests never complete

**Solution:**
```bash
# Add timeout
pytest --timeout=30

# Check for infinite loops or missing await
# Look for: async def without await calls
```

---

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures
├── unit/                    # Fast, isolated tests
│   ├── cache/
│   │   ├── test_memory_backend.py    # Memory cache tests
│   │   └── test_redis_backend.py     # Redis cache tests
│   ├── config/              # Configuration tests
│   └── observability/       # Observability tests
└── integration/             # Slower, integrated tests
    └── test_cache_factory.py         # Factory integration tests
```

---

## Test Markers

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"
```

---

## CI/CD

Tests run automatically in GitHub Actions on:
- Every push to main/develop
- Every pull request
- Manual workflow dispatch

**Quality Gates:**
- ✅ Linting (ruff)
- ✅ Formatting (ruff)
- ✅ Type checking (mypy)
- ✅ Unit tests (≥85% coverage)
- ✅ Integration tests
- ✅ Security scanning
- ✅ Secret detection
- ✅ SDD compliance

---

## Example Test

```python
# tests/unit/cache/test_example.py
import pytest
from src.cache.backends.memory import MemoryCacheBackend

@pytest.mark.asyncio
async def test_cache_basic_operations():
    """Test basic cache set and get."""
    # Arrange
    cache = MemoryCacheBackend()
    
    # Act
    await cache.set("test_key", "test_value")
    result = await cache.get("test_key")
    
    # Assert
    assert result == "test_value"
```

Run it:
```bash
pytest tests/unit/cache/test_example.py -v
```

---

## Performance Tips

### Faster Test Runs

```bash
# Run tests in parallel (requires pytest-xdist)
pytest -n auto

# Stop after first failure
pytest -x

# Run only failed tests from last run
pytest --lf

# Run failed tests first, then others
pytest --ff
```

### Skip Slow Tests

```python
# Mark test as slow
@pytest.mark.slow
async def test_expensive_operation():
    pass
```

```bash
# Skip slow tests
pytest -m "not slow"
```

---

## Resources

- **Full Test Documentation:** `tests/README.md`
- **SDD Compliance:** `docs/SDD.md`
- **Plugin Testing:** `docs/PLUGIN_GUIDE.md`
- **pytest Docs:** https://docs.pytest.org/

---

## Success Criteria

Your test setup is working correctly when:

- ✅ `pytest` runs without errors
- ✅ All 100+ tests pass
- ✅ Coverage is ≥85%
- ✅ `ruff check .` shows no issues
- ✅ `mypy src/` passes type checking
- ✅ Redis integration tests work

---

## Getting Help

If you're stuck:

1. Check `tests/README.md` for detailed documentation
2. Review existing tests for patterns
3. Check CI logs for detailed error messages
4. Open an issue with:
   - Command you ran
   - Full error output
   - Environment details (OS, Python version)

---

**Ready to test? Run this:**

```bash
# Complete test suite with coverage
pytest --cov=src --cov-report=term-missing -v

# If all tests pass, you're ready to go! 🎉
```

---

**Happy Testing! 🧪**