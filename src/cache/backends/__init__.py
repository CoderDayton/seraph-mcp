"""
Seraph MCP — Cache Backends

Exports available cache backend implementations.
"""

from .memory import MemoryCacheBackend

try:
    from .redis import RedisCacheBackend
except Exception:
    RedisCacheBackend = None  # type: ignore


__all__ = [
    "MemoryCacheBackend",
]

if "RedisCacheBackend" in globals() and RedisCacheBackend is not None:
    __all__.append("RedisCacheBackend")
