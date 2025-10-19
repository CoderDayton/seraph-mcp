"""
Seraph MCP — Cache Backends

Exports available cache backend implementations.

Redis backend is lazy-loaded via factory.py to avoid import overhead.
"""

from .memory import MemoryCacheBackend

__all__ = [
    "MemoryCacheBackend",
]
