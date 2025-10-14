"""
Seraph MCP — Core Runtime

Minimal AI optimization platform following SDD.md architecture.

This package provides the core runtime with:
- Configuration management (typed Pydantic models)
- Cache abstraction with pluggable backends
- Observability adapter (metrics, traces, logs)
- Error handling with typed exceptions
- HTTP server with health/readiness endpoints

Following SDD.md mandatory rules:
- Minimal core: Only essential functionality
- Single adapters: One factory/adapter per capability
- Plugin architecture: Heavy features as separate plugins
- Type safety: All config via Pydantic validation
- Observability: Structured logs with trace IDs

Usage:
    from src.config import load_config
    from src.cache import create_cache
    from src.observability import get_observability

    config = load_config()
    cache = create_cache()
    obs = get_observability()
"""

__version__ = "1.0.0"
__author__ = "Seraph MCP Team"
__license__ = "MIT"


# Core exports

from . import (
    budget_management,
    cache,
    config,
    context_optimization,
    errors,
    observability,
    providers,
    semantic_cache,
)

# Optional submodule: make import lazy/optional for environments without it
# Note: token_optimization module does not exist yet
token_optimization = None


__all__ = [
    "cache",
    "config",
    "errors",
    "observability",
    "providers",
    "semantic_cache",
    "budget_management",
    "token_optimization",
    "context_optimization",
    "__version__",
]
