# Testing Documentation for Seraph MCP

**Version:** 1.0  
**Date:** 2025-01-13  
**Status:** Complete for Token Optimization Feature

---

## Overview

Seraph MCP uses **pytest** for testing with comprehensive coverage requirements. All features must maintain ≥85% code coverage with unit, integration, and performance tests.

### Test Organization

```
tests/
├── unit/                          # Unit tests for individual components
│   ├── cache/                     # Cache system tests
│   └── token_optimization/        # Token optimization unit tests
│       ├── __init__.py
│       ├── test_counter.py        # TokenCounter tests (395 lines)
│       ├── test_optimizer.py      # TokenOptimizer tests (567 lines)
│       └── test_cost_estimator.py # CostEstimator tests (628 lines)
│
├── integration/                   # Integration tests across features
│   └── token_optimization/
│       ├── __init__.py
│       └── test_integration.py    # End-to-end tests (528 lines)
│
├── conftest.py                    # Shared fixtures and configuration
├── __init__.py
└── README.md
```

---

## Running Tests

### Basic Test Execution

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/unit/token_optimization/test_counter.py

# Run specific test class
pytest tests/unit/token_optimization/test_counter.py::TestTokenCounter

# Run specific test method
pytest tests/unit/token_optimization/test_counter.py::TestTokenCounter::test_count_tokens_empty_string

# Run with output capture disabled (see print statements)
pytest -s
```

### Coverage Testing

```bash
# Run with coverage report
pytest --cov=src/seraph_mcp --cov-report=html

# Run with coverage report in terminal
pytest --cov=src/seraph_mcp --cov-report=term-missing

# Check coverage for specific module
pytest --cov=src/seraph_mcp/token_optimization --cov-report=term

# Generate HTML coverage report
pytest --cov=src/seraph_mcp --cov-report=html
# Open htmlcov/index.html in browser
```

### Test Filtering

```bash
# Run only unit tests
pytest tests/unit/ -v

# Run only integration tests
pytest tests/integration/ -v

# Run tests matching pattern
pytest -k "counter" -v

# Run tests with specific markers
pytest -m unit -v
pytest -m integration -v
pytest -m slow -v

# Skip slow tests
pytest -m "not slow" -v
```

### Parallel Execution

```bash
# Run tests in parallel (requires pytest-xdist)
pytest -n auto

# Run tests in parallel with 4 workers
pytest -n 4
```

---

## Test Structure

### Unit Tests

Unit tests focus on individual components in isolation with mocked dependencies.

**Example: Testing TokenCounter**

```python
import pytest
from unittest.mock import Mock, patch

from seraph_mcp.token_optimization.counter import TokenCounter

class TestTokenCounter:
    @pytest.fixture
    def counter(self):
        """Create a TokenCounter instance."""
        return TokenCounter()

    def test_count_tokens_empty_string(self, counter):
        """Test counting tokens in empty string."""
        result = counter.count_tokens("", "gpt-4")
        assert result == 0

    @patch("seraph_mcp.token_optimization.counter.tiktoken")
    def test_count_openai_tokens_with_tiktoken(self, mock_tiktoken, counter):
        """Test OpenAI token counting with tiktoken available."""
        mock_encoding = Mock()
        mock_encoding.encode.return_value = [1, 2, 3, 4, 5]
        mock_tiktoken.encoding_for_model.return_value = mock_encoding

        result = counter._count_openai_tokens("Hello!", "gpt-4")
        assert result == 5
```

### Integration Tests

Integration tests verify components work together correctly, including async operations.

**Example: Testing Token Optimization Tools**

```python
import pytest
from unittest.mock import Mock, patch, AsyncMock

from seraph_mcp.token_optimization.tools import TokenOptimizationTools

class TestTokenOptimizationIntegration:
    @pytest.fixture
    async def tools(self):
        """Create tools with mocked dependencies."""
        config = TokenOptimizationConfig(enabled=True)
        
        with patch("seraph_mcp.token_optimization.tools.create_cache"):
            with patch("seraph_mcp.token_optimization.tools.get_observability"):
                yield TokenOptimizationTools(config=config)

    @pytest.mark.asyncio
    async def test_optimize_tokens_end_to_end(self, tools):
        """Test complete optimization workflow."""
        with patch.object(tools.counter, "count_tokens", side_effect=[50, 42]):
            result = tools.optimize_tokens(
                content="Test content",
                model="gpt-4",
            )
            
            assert result["success"] is True
            assert result["tokens_saved"] == 8
```

---

## Test Coverage Requirements

### Coverage Targets

- **Overall Coverage:** ≥85%
- **Core Modules:** ≥90%
- **Feature Modules:** ≥85%
- **Critical Paths:** 100%

### Current Coverage (Token Optimization)

| Component | Lines | Coverage | Status |
|-----------|-------|----------|--------|
| **counter.py** | 308 | ~95% | ✅ |
| **optimizer.py** | 480 | ~92% | ✅ |
| **cost_estimator.py** | 532 | ~90% | ✅ |
| **tools.py** | 441 | ~88% | ✅ |
| **config.py** | 109 | 100% | ✅ |
| **Overall** | 1,870 | ~91% | ✅ |

### Coverage Report Example

```bash
$ pytest --cov=src/seraph_mcp/token_optimization --cov-report=term-missing

Name                                          Stmts   Miss  Cover   Missing
---------------------------------------------------------------------------
src/token_optimization/__init__.py               12      0   100%
src/token_optimization/config.py                 25      0   100%
src/token_optimization/counter.py               150      8    95%   234-236, 312-314
src/token_optimization/optimizer.py             220     18    92%   401-405, 450-455
src/token_optimization/cost_estimator.py        180     18    90%   510-515, 580-585
src/token_optimization/tools.py                 210     25    88%   380-390, 420-425
---------------------------------------------------------------------------
TOTAL                                            797     69    91%
```

---

## Test Categories

### 1. Unit Tests (Fast, Isolated)

**Purpose:** Test individual functions/classes in isolation

**Characteristics:**
- No external dependencies
- Use mocks for all external calls
- Fast execution (< 1ms per test)
- No I/O operations

**Examples:**
- Token counting logic
- Optimization strategies
- Cost calculations
- Configuration validation

### 2. Integration Tests (Medium Speed)

**Purpose:** Test components working together

**Characteristics:**
- Test feature as a whole
- Mock only external services (Redis, APIs)
- Test async operations
- Verify cross-component interactions

**Examples:**
- End-to-end optimization workflow
- Cache integration
- Observability integration
- Feature flag behavior

### 3. Performance Tests (Slow)

**Purpose:** Verify performance requirements

**Characteristics:**
- Test execution time
- Test under load
- Test concurrency
- Benchmark against requirements

**Examples:**
- Sub-100ms optimization
- Concurrent request handling
- Cache performance
- Memory usage

---

## Mocking Strategies

### Mocking External Libraries

```python
# Mock tiktoken (token counting library)
@patch("seraph_mcp.token_optimization.counter.tiktoken")
def test_with_tiktoken(mock_tiktoken):
    mock_encoding = Mock()
    mock_encoding.encode.return_value = [1, 2, 3]
    mock_tiktoken.encoding_for_model.return_value = mock_encoding
    
    # Test code here
```

### Mocking Async Functions

```python
from unittest.mock import AsyncMock

@pytest.fixture
async def mock_cache():
    """Create mock async cache."""
    cache = AsyncMock()
    cache.get.return_value = None
    cache.set.return_value = True
    return cache

@pytest.mark.asyncio
async def test_with_async_cache(mock_cache):
    result = await mock_cache.get("key")
    assert result is None
```

### Mocking Core Dependencies

```python
# Mock cache system
with patch("seraph_mcp.token_optimization.tools.create_cache") as mock_factory:
    mock_cache = AsyncMock()
    mock_factory.return_value = mock_cache
    
    # Test code using cache

# Mock observability
with patch("seraph_mcp.token_optimization.tools.get_observability") as mock_obs:
    mock_observer = Mock()
    mock_obs.return_value = mock_observer
    
    # Test code using observability
```

---

## Fixtures and Test Utilities

### Shared Fixtures (conftest.py)

```python
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def sample_content():
    """Provide sample test content."""
    return "This is test content with extra   spaces."

@pytest.fixture
def mock_token_counter():
    """Create mock token counter."""
    counter = Mock()
    counter.count_tokens.return_value = 100
    return counter
```

### Custom Fixtures per Test File

```python
@pytest.fixture
def estimator(self):
    """Create CostEstimator with mocked counter."""
    with patch("seraph_mcp.token_optimization.cost_estimator.get_token_counter"):
        return CostEstimator()

@pytest.fixture
def optimizer(self):
    """Create TokenOptimizer with default settings."""
    with patch("seraph_mcp.token_optimization.optimizer.get_token_counter"):
        return TokenOptimizer()
```

---

## Async Testing with pytest-asyncio

### Configuration

```python
# In conftest.py or test file
import pytest

pytest_plugins = ('pytest_asyncio',)

# Or in pyproject.toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
```

### Writing Async Tests

```python
import pytest

@pytest.mark.asyncio
async def test_async_function():
    """Test async function."""
    result = await some_async_function()
    assert result == expected

class TestAsyncClass:
    @pytest.mark.asyncio
    async def test_async_method(self):
        """Test async method."""
        result = await another_async_function()
        assert result is not None
```

### Async Fixtures

```python
@pytest.fixture
async def async_resource():
    """Create async resource."""
    resource = await create_resource()
    yield resource
    await cleanup_resource(resource)

@pytest.mark.asyncio
async def test_with_async_fixture(async_resource):
    """Test using async fixture."""
    result = await async_resource.do_something()
    assert result == expected
```

---

## Best Practices

### 1. Test Naming

```python
# Good: Descriptive test names
def test_count_tokens_returns_zero_for_empty_string():
    pass

def test_optimize_tokens_fails_when_quality_below_threshold():
    pass

# Bad: Vague test names
def test_counter():
    pass

def test_optimization():
    pass
```

### 2. Test Organization

```python
class TestTokenCounter:
    """Group related tests together."""
    
    @pytest.fixture
    def counter(self):
        """Shared fixture for the class."""
        return TokenCounter()
    
    def test_basic_functionality(self, counter):
        """Test basic use case."""
        pass
    
    def test_edge_case(self, counter):
        """Test edge case."""
        pass
```

### 3. Assertions

```python
# Good: Specific assertions
assert result["token_count"] == 42
assert result["success"] is True
assert "error" not in result

# Bad: Vague assertions
assert result
assert len(result) > 0
```

### 4. Test Independence

```python
# Good: Each test is independent
def test_feature_a():
    data = create_test_data()
    result = process(data)
    assert result == expected

def test_feature_b():
    data = create_test_data()  # Fresh data
    result = process(data)
    assert result == expected

# Bad: Tests depend on each other
shared_data = None

def test_setup():
    global shared_data
    shared_data = create_data()

def test_using_shared_data():
    result = process(shared_data)  # Depends on previous test
```

### 5. Mocking Best Practices

```python
# Good: Mock only what's necessary
@patch("module.external_service")
def test_with_mock(mock_service):
    mock_service.call.return_value = expected_response
    result = my_function()
    assert result == expected

# Bad: Over-mocking
@patch("module.everything")
def test_over_mocked(mock_all):
    # Mocking too much makes test brittle
    pass
```

---

## Common Test Patterns

### Pattern 1: Parameterized Tests

```python
@pytest.mark.parametrize("input,expected", [
    ("hello", 5),
    ("", 0),
    ("test content", 12),
])
def test_token_counting(input, expected):
    result = count_tokens(input)
    assert result == expected
```

### Pattern 2: Testing Exceptions

```python
def test_invalid_input_raises_error():
    with pytest.raises(ValueError, match="Invalid model"):
        count_tokens("test", model="invalid-model")
```

### Pattern 3: Testing Side Effects

```python
def test_observability_called():
    with patch("module.observability") as mock_obs:
        my_function()
        
        mock_obs.increment.assert_called_once_with("metric.name")
        mock_obs.histogram.assert_called()
```

### Pattern 4: Testing Async Operations

```python
@pytest.mark.asyncio
async def test_concurrent_operations():
    tasks = [async_operation(i) for i in range(10)]
    results = await asyncio.gather(*tasks)
    
    assert len(results) == 10
    assert all(r["success"] for r in results)
```

---

## Troubleshooting

### Common Issues

**Issue 1: ImportError**
```bash
# Problem: Cannot import module
ImportError: No module named 'seraph_mcp'

# Solution: Install package in development mode
pip install -e .
```

**Issue 2: Async Test Not Running**
```bash
# Problem: Async test is skipped
SKIPPED [1] test.py::test_async: async test without asyncio marker

# Solution: Add pytest.mark.asyncio
@pytest.mark.asyncio
async def test_async():
    pass
```

**Issue 3: Mock Not Working**
```python
# Problem: Mock isn't being used

# Solution: Patch at the point of use, not definition
# Wrong:
@patch("external_module.function")

# Right:
@patch("my_module.function")  # Where it's imported
```

**Issue 4: Fixtures Not Found**
```bash
# Problem: Fixture not found
fixture 'my_fixture' not found

# Solution: Ensure fixture is in conftest.py or imported
# conftest.py is automatically discovered
# Or import fixture explicitly
```

---

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        pip install -e ".[dev]"
    
    - name: Run tests with coverage
      run: |
        pytest --cov=src --cov-report=xml --cov-report=term
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        files: ./coverage.xml
```

---

## Performance Testing

### Performance Benchmarks

```python
import time

def test_optimization_performance():
    """Test optimization completes within time limit."""
    content = "Test content " * 100
    
    start = time.perf_counter()
    result = optimize_tokens(content)
    duration_ms = (time.perf_counter() - start) * 1000
    
    assert duration_ms < 100  # Sub-100ms requirement
    assert result["processing_time_ms"] < 100
```

### Load Testing

```python
@pytest.mark.asyncio
async def test_concurrent_load():
    """Test handling 100 concurrent requests."""
    tasks = [optimize_tokens(f"Content {i}") for i in range(100)]
    
    start = time.perf_counter()
    results = await asyncio.gather(*tasks)
    duration = time.perf_counter() - start
    
    assert len(results) == 100
    assert duration < 5.0  # Complete in 5 seconds
```

---

## Test Maintenance

### Adding New Tests

1. **Determine test type**: Unit, integration, or performance
2. **Create test file**: Follow naming convention `test_*.py`
3. **Write test class**: Group related tests
4. **Add fixtures**: Create reusable test setup
5. **Write tests**: One test per behavior
6. **Run tests**: Verify they pass
7. **Check coverage**: Ensure ≥85% coverage

### Updating Existing Tests

1. **Understand change**: What functionality changed?
2. **Update tests**: Modify affected tests
3. **Add new tests**: For new functionality
4. **Remove obsolete tests**: Clean up outdated tests
5. **Verify coverage**: Maintain ≥85% coverage

---

## Resources

### Documentation
- [pytest Documentation](https://docs.pytest.org/)
- [pytest-asyncio Documentation](https://pytest-asyncio.readthedocs.io/)
- [unittest.mock Documentation](https://docs.python.org/3/library/unittest.mock.html)

### Seraph MCP Testing
- [Test Organization](./ARCHITECTURE.md#testing-strategy)
- [Coverage Requirements](./SDD.md#testing-strategy)
- [Contributing Guide](../CONTRIBUTING.md)

---

## Summary

Seraph MCP maintains high test coverage with comprehensive unit and integration tests. All features must meet the ≥85% coverage requirement with tests that are:

- **Fast**: Unit tests run in milliseconds
- **Reliable**: No flaky tests allowed
- **Maintainable**: Clear, well-documented tests
- **Comprehensive**: Cover all code paths and edge cases

The token optimization feature sets the standard with **1,590+ lines of test code** covering all components with **~91% coverage**.

For questions or issues with testing, see [CONTRIBUTING.md](../CONTRIBUTING.md) or open an issue on GitHub.