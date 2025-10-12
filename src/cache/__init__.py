"""
Seraph MCP — Cache Module

Provides caching functionality with pluggable backends.

Canonical exports per SDD.md:
- factory.py: Single source of truth for cache creation
- interface.py: Abstract cache interface all backends must implement
- backends/: Cache backend implementations (memory in core, others as plugins)

Usage:
    from src.cache import create_cache, get_cache

    cache = create_cache()
    await cache.set("key", "value", ttl=3600)
    value = await cache.get("key")
"""

from .factory import (
    close_all_caches,
    create_cache,
    get_cache,
    list_cache_instances,
    reset_cache_factory,
)
from .interface import CacheInterface

__all__ = [
    # Factory functions (canonical per SDD.md)
    "create_cache",
    "get_cache",
    "close_all_caches",
    "list_cache_instances",
    "reset_cache_factory",
    # Interface
    "CacheInterface",
]
