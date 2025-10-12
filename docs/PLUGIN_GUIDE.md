# Seraph MCP — Plugin Developer Guide

Version: 1.0
Last Updated: 2025-01-12
Status: Official plugin development documentation

---

## Table of Contents

1. [Introduction](#introduction)
2. [Plugin Architecture Overview](#plugin-architecture-overview)
3. [Plugin Contract & Requirements](#plugin-contract--requirements)
4. [Getting Started](#getting-started)
5. [Plugin Structure](#plugin-structure)
6. [Developing Your First Plugin](#developing-your-first-plugin)
7. [Integration with Core](#integration-with-core)
8. [Testing Plugins](#testing-plugins)
9. [Best Practices](#best-practices)
10. [Deployment & Distribution](#deployment--distribution)
11. [Examples](#examples)
12. [Troubleshooting](#troubleshooting)

---

## Introduction

Seraph MCP follows a **minimal core + plugins** architecture. The core runtime provides essential caching, observability, and configuration capabilities, while plugins extend the platform with specialized features like semantic search, intelligent routing, quality validation, and advanced analytics.

**Key Principles:**
- **Core is minimal**: Only essential functionality belongs in core
- **Plugins are explicit**: All plugins must be explicitly installed and loaded
- **Fail-safe loading**: Plugin failures never crash the core runtime
- **MCP tools as interface**: Plugins expose functionality via MCP tools
- **Typed contracts**: Clear version compatibility and dependency declarations

**Important Note:**
Redis is **NOT a plugin** — it is a core optional backend selected via `CACHE_BACKEND=redis`. This guide covers application-level plugins that add new capabilities beyond the core runtime.

---

## Plugin Architecture Overview

### Core vs. Plugin Boundary

**Core Runtime (`src/`):**
- FastMCP stdio server
- Typed Pydantic configuration
- Cache factory (memory and Redis backends)
- Observability adapter
- Error types and base MCP tools

**Plugins (`plugins/` or separate packages):**
- Semantic search and vector integrations
- Intelligent model routing
- Context optimization strategies
- Quality validation tools
- Advanced analytics and reporting
- External observability backends (beyond simple)
- Domain-specific functionality

### Communication Pattern

```
┌─────────────────────────────────────────┐
│         MCP Client (Claude)             │
└──────────────┬──────────────────────────┘
               │ stdio (JSON-RPC)
┌──────────────▼──────────────────────────┐
│         Core Server (src/server.py)     │
│  ┌────────────────────────────────────┐ │
│  │  Core Tools (cache, status, etc.)  │ │
│  └────────────────────────────────────┘ │
│  ┌────────────────────────────────────┐ │
│  │  Plugin Registry & Loader          │ │
│  └────────────────────────────────────┘ │
└──────────────┬──────────────────────────┘
               │ Direct import/calls
┌──────────────▼──────────────────────────┐
│            Plugins                       │
│  ┌──────────┐  ┌──────────┐  ┌────────┐│
│  │ Plugin A │  │ Plugin B │  │ ...    ││
│  │ @mcp.tool│  │ @mcp.tool│  │        ││
│  └──────────┘  └──────────┘  └────────┘│
└─────────────────────────────────────────┘
```

---

## Plugin Contract & Requirements

Every plugin MUST satisfy the following contract:

### 1. Expose MCP Tools

Plugins provide functionality via MCP tools decorated with `@mcp.tool()`:

```python
from fastmcp import FastMCP

mcp = FastMCP("Plugin Name")

@mcp.tool()
async def my_plugin_tool(param: str) -> dict:
    """Tool description."""
    return {"result": "value"}
```

### 2. Declare Metadata

Plugins must provide a metadata dictionary with:

```python
PLUGIN_METADATA = {
    "name": "my-plugin",
    "version": "1.0.0",
    "description": "Plugin description",
    "author": "Your Name",
    "core_version_min": "1.0.0",  # Minimum compatible core version
    "core_version_max": "2.0.0",  # Maximum compatible core version
    "dependencies": [
        "package>=1.0.0",
        "another-package>=2.0.0",
    ],
}
```

### 3. Provide Setup & Teardown

Plugins should implement setup and teardown functions:

```python
async def setup(config: dict) -> None:
    """Initialize plugin resources."""
    pass

async def teardown() -> None:
    """Clean up plugin resources."""
    pass
```

### 4. Handle Errors Gracefully

Plugin failures must not crash the core:

```python
from src.errors import SeraphError, PluginError

@mcp.tool()
async def my_tool() -> dict:
    try:
        result = await do_work()
        return {"success": True, "data": result}
    except Exception as e:
        # Log error but return structured response
        logger.error(f"Plugin error: {e}")
        raise PluginError(
            f"Tool failed: {e}",
            details={"plugin": "my-plugin", "tool": "my_tool"}
        )
```

### 5. Use Typed Configuration

Plugin configuration must use Pydantic models:

```python
from pydantic import BaseModel, Field

class MyPluginConfig(BaseModel):
    """Plugin configuration."""
    api_key: str = Field(..., description="API key for service")
    timeout: int = Field(30, description="Timeout in seconds")
    enabled: bool = Field(True, description="Enable plugin")
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- Seraph MCP core installed
- Understanding of async Python and MCP protocol

### Quick Start

1. **Create plugin directory:**

```bash
mkdir -p plugins/my-plugin/src/my_plugin
cd plugins/my-plugin
```

2. **Create plugin structure:**

```bash
# Create necessary files
touch src/my_plugin/__init__.py
touch src/my_plugin/plugin.py
touch src/my_plugin/config.py
touch pyproject.toml
touch README.md
```

3. **Implement plugin:**

See [Plugin Structure](#plugin-structure) and [Developing Your First Plugin](#developing-your-first-plugin).

4. **Install plugin:**

```bash
# Development installation
uv pip install -e .

# Or production installation
uv pip install .
```

5. **Register with core:**

Add plugin to your FastMCP configuration or dynamically load it.

---

## Plugin Structure

### Recommended Directory Layout

```
plugins/my-plugin/
├── src/
│   └── my_plugin/
│       ├── __init__.py          # Package exports
│       ├── plugin.py            # Main plugin implementation
│       ├── config.py            # Plugin configuration models
│       ├── tools.py             # MCP tool implementations
│       └── utils.py             # Helper functions
├── tests/
│   ├── unit/
│   │   └── test_plugin.py
│   └── integration/
│       └── test_integration.py
├── docs/
│   └── README.md
├── examples/
│   └── example_usage.py
├── pyproject.toml               # Package metadata & dependencies
├── README.md
└── LICENSE
```

### File Responsibilities

**`__init__.py`** — Package exports and metadata:
```python
from .plugin import PLUGIN_METADATA, setup, teardown
from .tools import mcp

__all__ = ["PLUGIN_METADATA", "setup", "teardown", "mcp"]
```

**`plugin.py`** — Core plugin logic:
```python
PLUGIN_METADATA = {...}

async def setup(config: dict) -> None:
    """Initialize plugin."""
    pass

async def teardown() -> None:
    """Cleanup plugin."""
    pass
```

**`config.py`** — Typed configuration:
```python
from pydantic import BaseModel

class MyPluginConfig(BaseModel):
    """Plugin configuration."""
    pass
```

**`tools.py`** — MCP tool definitions:
```python
from fastmcp import FastMCP

mcp = FastMCP("My Plugin")

@mcp.tool()
async def my_tool():
    """Tool implementation."""
    pass
```

---

## Developing Your First Plugin

Let's create a simple **"Hello World"** plugin that demonstrates the essential patterns.

### Step 1: Create Plugin Package

```python
# plugins/hello-world/src/hello_world/__init__.py
"""Hello World Plugin for Seraph MCP."""

from .plugin import PLUGIN_METADATA, setup, teardown
from .tools import mcp

__all__ = ["PLUGIN_METADATA", "setup", "teardown", "mcp"]

__version__ = "1.0.0"
```

### Step 2: Define Metadata

```python
# plugins/hello-world/src/hello_world/plugin.py
"""Hello World Plugin implementation."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

PLUGIN_METADATA = {
    "name": "hello-world",
    "version": "1.0.0",
    "description": "A simple hello world plugin demonstrating plugin development",
    "author": "Seraph Team",
    "core_version_min": "1.0.0",
    "core_version_max": "2.0.0",
    "dependencies": [],
}

# Plugin state
_initialized = False
_config: Optional[dict] = None


async def setup(config: dict) -> None:
    """
    Initialize the Hello World plugin.
    
    Args:
        config: Plugin configuration dictionary
    """
    global _initialized, _config
    
    if _initialized:
        logger.warning("Hello World plugin already initialized")
        return
    
    _config = config
    logger.info("Hello World plugin initialized", extra={"config": config})
    _initialized = True


async def teardown() -> None:
    """Clean up Hello World plugin resources."""
    global _initialized, _config
    
    if not _initialized:
        return
    
    logger.info("Hello World plugin shutting down")
    _config = None
    _initialized = False


def is_initialized() -> bool:
    """Check if plugin is initialized."""
    return _initialized


def get_config() -> Optional[dict]:
    """Get plugin configuration."""
    return _config
```

### Step 3: Implement MCP Tools

```python
# plugins/hello-world/src/hello_world/tools.py
"""Hello World Plugin MCP Tools."""

import logging
from typing import Optional

from fastmcp import FastMCP

from .plugin import get_config, is_initialized

logger = logging.getLogger(__name__)
mcp = FastMCP("Hello World Plugin")


@mcp.tool()
async def say_hello(name: Optional[str] = None) -> dict:
    """
    Say hello to someone or everyone.
    
    Args:
        name: Optional name to greet. If not provided, greets the world.
        
    Returns:
        Greeting message and plugin status
    """
    if not is_initialized():
        logger.warning("say_hello called but plugin not initialized")
        return {
            "error": "Plugin not initialized",
            "success": False,
        }
    
    greeting = f"Hello, {name}!" if name else "Hello, World!"
    
    config = get_config()
    
    return {
        "success": True,
        "greeting": greeting,
        "plugin": "hello-world",
        "version": "1.0.0",
        "config": config,
    }


@mcp.tool()
async def get_plugin_info() -> dict:
    """
    Get information about the Hello World plugin.
    
    Returns:
        Plugin metadata and status
    """
    from .plugin import PLUGIN_METADATA
    
    return {
        "metadata": PLUGIN_METADATA,
        "initialized": is_initialized(),
        "config": get_config(),
    }
```

### Step 4: Add Configuration

```python
# plugins/hello-world/src/hello_world/config.py
"""Hello World Plugin Configuration."""

from pydantic import BaseModel, Field


class HelloWorldConfig(BaseModel):
    """Configuration for Hello World plugin."""
    
    enabled: bool = Field(
        default=True,
        description="Enable or disable the plugin",
    )
    
    greeting_prefix: str = Field(
        default="Hello",
        description="Prefix for greetings",
    )
    
    max_name_length: int = Field(
        default=100,
        description="Maximum length for names",
        ge=1,
        le=1000,
    )
```

### Step 5: Create pyproject.toml

```toml
# plugins/hello-world/pyproject.toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "seraph-mcp-hello-world"
version = "1.0.0"
description = "Hello World plugin for Seraph MCP"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "fastmcp>=2.0.0",
    "pydantic>=2.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
]

[tool.setuptools.packages.find]
where = ["src"]
```

### Step 6: Write Tests

```python
# plugins/hello-world/tests/unit/test_plugin.py
"""Unit tests for Hello World plugin."""

import pytest

from hello_world.plugin import PLUGIN_METADATA, setup, teardown, is_initialized


@pytest.mark.asyncio
class TestHelloWorldPlugin:
    """Test Hello World plugin lifecycle."""
    
    async def test_metadata(self):
        """Test plugin metadata is defined correctly."""
        assert PLUGIN_METADATA["name"] == "hello-world"
        assert PLUGIN_METADATA["version"] == "1.0.0"
        assert "core_version_min" in PLUGIN_METADATA
    
    async def test_setup_and_teardown(self):
        """Test plugin setup and teardown."""
        # Initially not initialized
        assert not is_initialized()
        
        # Setup
        config = {"test": "value"}
        await setup(config)
        assert is_initialized()
        
        # Teardown
        await teardown()
        assert not is_initialized()
    
    async def test_setup_idempotent(self):
        """Test that setup can be called multiple times safely."""
        await setup({})
        await setup({})  # Should not raise
        
        await teardown()


# plugins/hello-world/tests/unit/test_tools.py
"""Unit tests for Hello World plugin tools."""

import pytest

from hello_world.plugin import setup, teardown
from hello_world.tools import say_hello, get_plugin_info


@pytest.mark.asyncio
class TestHelloWorldTools:
    """Test Hello World plugin MCP tools."""
    
    async def test_say_hello_without_name(self):
        """Test say_hello without a name."""
        await setup({})
        
        result = await say_hello()
        
        assert result["success"] is True
        assert result["greeting"] == "Hello, World!"
        assert result["plugin"] == "hello-world"
        
        await teardown()
    
    async def test_say_hello_with_name(self):
        """Test say_hello with a name."""
        await setup({})
        
        result = await say_hello(name="Alice")
        
        assert result["success"] is True
        assert result["greeting"] == "Hello, Alice!"
        
        await teardown()
    
    async def test_say_hello_not_initialized(self):
        """Test say_hello when plugin not initialized."""
        result = await say_hello()
        
        assert result["success"] is False
        assert "error" in result
    
    async def test_get_plugin_info(self):
        """Test get_plugin_info tool."""
        await setup({"test": "config"})
        
        result = await get_plugin_info()
        
        assert "metadata" in result
        assert result["metadata"]["name"] == "hello-world"
        assert result["initialized"] is True
        assert result["config"]["test"] == "config"
        
        await teardown()
```

---

## Integration with Core

### Method 1: Direct Import (Development)

For development and testing, directly import the plugin in your server:

```python
# src/server.py (or a plugin loader module)
from plugins.hello_world import mcp as hello_world_mcp, setup as hello_setup

# Register plugin tools
mcp.add_tools_from(hello_world_mcp)

# Initialize in lifespan
@mcp.lifespan()
async def lifespan():
    await hello_setup({"greeting_prefix": "Hi"})
    yield
    await hello_teardown()
```

### Method 2: Dynamic Plugin Loader

Create a plugin loader system:

```python
# src/plugins/loader.py
"""Dynamic plugin loader for Seraph MCP."""

import importlib
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_loaded_plugins: Dict[str, Any] = {}


async def load_plugin(plugin_name: str, config: dict) -> bool:
    """
    Load and initialize a plugin.
    
    Args:
        plugin_name: Python import path (e.g., 'hello_world')
        config: Plugin configuration
        
    Returns:
        True if loaded successfully
    """
    try:
        # Import plugin module
        plugin_module = importlib.import_module(plugin_name)
        
        # Validate plugin contract
        required_attrs = ["PLUGIN_METADATA", "setup", "teardown", "mcp"]
        for attr in required_attrs:
            if not hasattr(plugin_module, attr):
                logger.error(f"Plugin {plugin_name} missing required attribute: {attr}")
                return False
        
        # Initialize plugin
        await plugin_module.setup(config)
        
        # Store plugin reference
        _loaded_plugins[plugin_name] = plugin_module
        
        logger.info(f"Plugin {plugin_name} loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load plugin {plugin_name}: {e}", exc_info=True)
        return False


async def unload_plugin(plugin_name: str) -> bool:
    """Unload a plugin."""
    if plugin_name not in _loaded_plugins:
        return False
    
    try:
        plugin = _loaded_plugins[plugin_name]
        await plugin.teardown()
        del _loaded_plugins[plugin_name]
        logger.info(f"Plugin {plugin_name} unloaded")
        return True
    except Exception as e:
        logger.error(f"Error unloading plugin {plugin_name}: {e}")
        return False


async def load_all_plugins(plugin_configs: Dict[str, dict]) -> List[str]:
    """
    Load multiple plugins.
    
    Args:
        plugin_configs: Dict mapping plugin names to their configs
        
    Returns:
        List of successfully loaded plugin names
    """
    loaded = []
    for plugin_name, config in plugin_configs.items():
        if await load_plugin(plugin_name, config):
            loaded.append(plugin_name)
    return loaded


def get_loaded_plugins() -> List[str]:
    """Get list of loaded plugin names."""
    return list(_loaded_plugins.keys())
```

### Method 3: Configuration-Based Loading

Load plugins from configuration file:

```json
// fastmcp.json
{
  "name": "Seraph MCP with Plugins",
  "version": "1.0.0",
  "plugins": {
    "hello_world": {
      "enabled": true,
      "config": {
        "greeting_prefix": "Hi"
      }
    },
    "semantic_search": {
      "enabled": true,
      "config": {
        "embedding_model": "text-embedding-ada-002"
      }
    }
  }
}
```

---

## Testing Plugins

### Unit Tests

Test plugin logic in isolation:

```python
import pytest
from my_plugin import setup, teardown, my_tool

@pytest.mark.asyncio
async def test_my_tool():
    await setup({"key": "value"})
    result = await my_tool()
    assert result["success"] is True
    await teardown()
```

### Integration Tests

Test plugin with real core components:

```python
from src.cache import create_cache
from my_plugin import setup, my_cache_tool

@pytest.mark.asyncio
async def test_plugin_with_cache(mock_env_memory):
    cache = create_cache()
    await setup({"cache": cache})
    
    result = await my_cache_tool("key", "value")
    assert result["cached"] is True
    
    await teardown()
```

### End-to-End Tests

Test plugin via MCP protocol:

```python
from fastmcp.testing import MCPTestClient

@pytest.mark.asyncio
async def test_plugin_via_mcp():
    async with MCPTestClient() as client:
        # Call plugin tool via MCP
        result = await client.call_tool("my_tool", {"param": "value"})
        assert result["success"] is True
```

---

## Best Practices

### 1. Follow the Single Responsibility Principle

Each plugin should do one thing well:

```python
# ✅ Good: Focused plugin
PLUGIN_METADATA = {
    "name": "semantic-search",
    "description": "Semantic search using vector embeddings",
}

# ❌ Bad: Kitchen sink plugin
PLUGIN_METADATA = {
    "name": "everything",
    "description": "Search, analytics, routing, and more",
}
```

### 2. Use Typed Interfaces

Always use Pydantic for configuration and type hints for functions:

```python
# ✅ Good
from pydantic import BaseModel

class PluginConfig(BaseModel):
    api_key: str
    timeout: int = 30

@mcp.tool()
async def my_tool(param: str, count: int = 10) -> dict:
    ...

# ❌ Bad
async def my_tool(param, count=10):  # No types!
    ...
```

### 3. Handle Errors Gracefully

Never let plugin errors crash the core:

```python
# ✅ Good
@mcp.tool()
async def risky_operation() -> dict:
    try:
        result = await external_api_call()
        return {"success": True, "data": result}
    except TimeoutError:
        return {"success": False, "error": "timeout"}
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return {"success": False, "error": "internal_error"}

# ❌ Bad
@mcp.tool()
async def risky_operation() -> dict:
    result = await external_api_call()  # Can crash!
    return result
```

### 4. Use Observability

Integrate with core observability:

```python
from src.observability import get_observability

@mcp.tool()
async def my_tool() -> dict:
    obs = get_observability()
    obs.increment("plugin.my_plugin.my_tool.calls")
    
    with obs.trace("my_tool.process"):
        result = await process()
    
    obs.histogram("my_tool.result_size", len(result))
    return result
```

### 5. Respect Resource Limits

Clean up resources properly:

```python
# ✅ Good
_client = None

async def setup(config):
    global _client
    _client = HTTPClient(timeout=config["timeout"])

async def teardown():
    global _client
    if _client:
        await _client.close()
        _client = None

# ❌ Bad: Resource leak
async def setup(config):
    _client = HTTPClient()  # Never closed!
```

### 6. Document Everything

Provide clear documentation:

```python
@mcp.tool()
async def semantic_search(query: str, limit: int = 10) -> dict:
    """
    Perform semantic search using vector embeddings.
    
    This tool searches through cached content using semantic similarity,
    allowing you to find relevant results even when exact keyword matches
    don't exist.
    
    Args:
        query: Search query text (max 1000 characters)
        limit: Maximum number of results to return (1-100, default 10)
        
    Returns:
        dict containing:
            - results: List of matching items with similarity scores
            - query: Original query
            - count: Number of results returned
            
    Example:
        >>> await semantic_search("machine learning models", limit=5)
        {
            "results": [
                {"text": "...", "score": 0.95},
                {"text": "...", "score": 0.87},
            ],
            "count": 5
        }
    """
    ...
```

### 7. Version Compatibility

Declare clear version constraints:

```python
PLUGIN_METADATA = {
    "core_version_min": "1.0.0",
    "core_version_max": "2.0.0",  # Exclude breaking changes
    "dependencies": [
        "httpx>=0.25.0,<1.0.0",  # Pin major versions
    ],
}
```

### 8. Configuration Validation

Validate configuration at startup:

```python
from pydantic import BaseModel, validator

class PluginConfig(BaseModel):
    api_key: str
    endpoint_url: str
    
    @validator("api_key")
    def validate_api_key(cls, v):
        if not v or len(v) < 10:
            raise ValueError("Invalid API key")
        return v
    
    @validator("endpoint_url")
    def validate_url(cls, v):
        if not v.startswith(("http://", "https://")):
            raise ValueError("Invalid URL")
        return v
```

---

## Deployment & Distribution

### Local Development

```bash
# Install in editable mode
cd plugins/my-plugin
uv pip install -e .
```

### PyPI Distribution

```bash
# Build package
python -m build

# Upload to PyPI
python -m twine upload dist/*

# Install from PyPI
uv pip install seraph-mcp-my-plugin
```

### Git-based Installation

```bash
# Install from git
uv pip install git+https://github.com/user/seraph-mcp-my-plugin.git

# Install specific version
uv pip install git+https://github.com/user/seraph-mcp-my-plugin.git@v1.0.0
```

### Docker Deployment

```dockerfile
FROM python:3.11-slim

# Install core
COPY . /app
WORKDIR /app
RUN uv pip install .

# Install plugins
RUN uv pip install seraph-mcp-semantic-search
RUN uv pip install seraph-mcp-routing

CMD ["fastmcp", "run", "fastmcp.json"]
```

---

## Examples

### Example 1: Semantic Search Plugin

```python
# plugins/semantic-search/src/semantic_search/tools.py
from fastmcp import FastMCP
from sentence_transformers import SentenceTransformer

mcp = FastMCP("Semantic Search")
model = None

@mcp.tool()
async def semantic_search(query: str, limit: int = 10) -> dict:
    """Search using semantic similarity."""
    global model
    if model is None:
        model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Generate embeddings and search
    query_embedding = model.encode(query)
    results = await search_vector_db(query_embedding, limit)
    
    return {
        "results": results,
        "count": len(results),
        "query": query,
    }
```

### Example 2: Model Routing Plugin

```python
# plugins/routing/src/routing/tools.py
from fastmcp import FastMCP
from .router import ModelRouter

mcp = FastMCP("Model Routing")
router = ModelRouter()

@mcp.tool()
async def find_best_model(
    task_type: str,
    budget: float,
    quality_threshold: float = 0.8
) -> dict:
    """Find optimal model for task."""
    recommendation = await router.recommend(
        task_type=task_type,
        budget=budget,
        quality_threshold=quality_threshold,
    )
    
    return {
        "model": recommendation.model_name,
        "cost_per_token": recommendation.cost,
        "expected_quality": recommendation.quality_score,
        "reasoning": recommendation.reasoning,
    }
```

### Example 3: Analytics Plugin

```python
# plugins/analytics/src/analytics/tools.py
from fastmcp import FastMCP
import pandas as pd

mcp = FastMCP("Analytics")

@mcp.tool()
async def generate_usage_report(days: int = 7) -> dict:
    """Generate usage analytics report."""
    from src.cache import create_cache
    
    cache = create_cache()
    stats = await cache.get_stats()
    
    # Analyze usage patterns
    report = {
        "period_days": days,
        "total_requests": stats["hits"] + stats["misses"],
        "cache_hit_rate": stats["hit_rate"],
        "recommendations": [],
    }
    
    if stats["hit_rate"] < 50:
        report["recommendations"].append(
            "Consider increasing cache TTL to improve hit rate"
        )
    
    return report
```

---

## Troubleshooting

### Plugin Not Loading

**Symptom:** Plugin doesn't appear or tools unavailable.

**Solutions:**
1. Check plugin is installed: `uv pip list | grep my-plugin`
2. Verify import path is correct
3. Check logs for import errors
4. Ensure plugin exports `PLUGIN_METADATA`, `setup`, `teardown`, `mcp`

### Version Incompatibility

**Symptom:** "Core version incompatible" error.

**Solutions:**
1. Check `core_version_min` and `core_version_max` in plugin metadata
2. Update core: `uv pip install --upgrade seraph-mcp`
3. Or update plugin: `uv pip install --upgrade seraph-mcp-my-plugin`

### Configuration Errors

**Symptom:** Plugin fails during setup.

**Solutions:**
1. Validate configuration against plugin's Pydantic model
2. Check environment variables are set correctly
3. Review logs for validation errors
4. Use typed configuration with helpful error messages

### Resource Leaks

**Symptom:** Memory/connections growing over time.

**Solutions:**
1. Ensure `teardown()` properly closes resources
2. Use context managers for temporary resources
3. Monitor resource usage with observability tools
4. Test plugin lifecycle in isolation

### Tool Not Responding

**Symptom:** Tool calls timeout or hang.

**Solutions:**
1. Add timeout to async operations
2. Check for deadlocks in async code
3. Use `asyncio.wait_for()` for bounded operations
4. Add logging to identify where hangs occur

---

## Conclusion

Plugins are the primary extension mechanism for Seraph MCP. By following this guide and adhering to the plugin contract, you can create powerful, production-ready plugins that extend the platform's capabilities while maintaining the core's stability and simplicity.

**Key Takeaways:**
- Follow the plugin contract strictly
- Use typed interfaces everywhere
- Handle errors gracefully
- Test thoroughly
- Document comprehensively
- Respect resource limits
- Integrate with core observability

**Resources:**
- [SDD.md](./SDD.md) — System design document
- [FastMCP Documentation](https://github.com/jlowin/fastmcp) — MCP framework
- [Core Source](../src/) — Core implementation examples

**Need Help?**
- Open an issue on GitHub
- Check existing plugins for examples
- Review core source code for patterns

Happy plugin development! 🚀