"""
Seraph MCP — Cache Factory

Canonical factory for creating cache instances based on configuration.
This is the ONLY way to obtain cache backends in the core runtime.

Key points:
- Single adapter rule: This is the only cache factory
- Optional Redis via env toggle: Select backend with CACHE_BACKEND=memory|redis
  - Defaults to memory if not set (CACHE_BACKEND=memory)
  - When redis is selected, Redis must be installed and REDIS_URL must be set
- All configuration is typed and validated via Pydantic models

Examples:
    from src.cache.factory import create_cache, get_cache

    # Uses env-configured backend (memory by default)
    cache = create_cache()

    # Or explicitly supply a CacheConfig (e.g., for tests)
    from src.config import CacheConfig, CacheBackend
    cfg = CacheConfig(backend=CacheBackend.MEMORY, ttl_seconds=600)
    mem_cache = create_cache(cfg, name="test")

    # Redis toggle via env:
    #   CACHE_BACKEND=redis REDIS_URL=redis://localhost:6379 fastmcp dev src/server.py
"""

from __future__ import annotations

import logging

from ..config import CacheBackend, CacheConfig, get_config
from ..errors import ConfigurationError
from .backends.memory import (
    MemoryCacheBackend,  # Import memory eagerly (always available)
)
from .interface import CacheInterface

logger = logging.getLogger(__name__)

# Global cache instances registry
_cache_instances: dict[str, CacheInterface] = {}


def _create_memory_cache(config: CacheConfig) -> CacheInterface:
    """Internal helper to construct a memory cache backend."""
    return MemoryCacheBackend(
        max_size=config.max_size,
        default_ttl=config.ttl_seconds,
        namespace=config.namespace,
    )


def _create_redis_cache(config: CacheConfig) -> CacheInterface:
    """Internal helper to construct a redis cache backend with lazy import."""
    # Validate minimal requirements (Pydantic validator also enforces this)
    if not config.redis_url:
        raise ConfigurationError(
            "REDIS_URL must be set when CACHE_BACKEND=redis",
            details={"env": "REDIS_URL", "backend": "redis"},
        )

    # Lazy import to avoid hard dependency when memory backend is used
    try:
        from .backends.redis import RedisCacheBackend
    except ImportError as e:
        logger.error(
            "Redis backend selected but redis client is not installed",
            extra={"package": "redis>=5.0.0", "error": str(e)},
        )
        raise ConfigurationError(
            "Redis backend selected but redis client is unavailable. "
            "Install with: pip install 'redis>=5.0.0' or add to dependencies.",
            details={"package": "redis>=5.0.0", "error": str(e), "backend": "redis"},
        ) from e
    except Exception as e:
        logger.error(
            "Failed to import Redis backend module",
            extra={"error": str(e)},
            exc_info=True,
        )
        raise ConfigurationError(
            f"Failed to load Redis backend: {e}",
            details={"package": "redis", "error": str(e), "backend": "redis"},
        ) from e

    return RedisCacheBackend(
        redis_url=config.redis_url,
        namespace=config.namespace,
        default_ttl=config.ttl_seconds,
        max_connections=config.redis_max_connections,
        socket_timeout=config.redis_socket_timeout,
    )


def create_cache(
    config: CacheConfig | None = None,
    name: str = "default",
) -> CacheInterface:
    """
    Create a cache backend instance based on configuration.

    This is the canonical way to obtain cache instances per SDD.md.
    All cache creation MUST go through this factory.

    Args:
        config: Cache configuration (uses global config if not provided)
        name: Cache instance name (for multiple cache instances)

    Returns:
        Configured cache backend instance

    Raises:
        ConfigurationError: If cache configuration is invalid or backend unavailable
    """
    # Return existing instance if already created
    if name in _cache_instances:
        logger.debug("Returning existing cache instance: %s", name)
        return _cache_instances[name]

    # Use global config if not provided
    if config is None:
        config = get_config().cache

    logger.info(
        "Creating cache instance '%s' with backend: %s",
        name,
        config.backend,
        extra={"cache_name": name, "backend": str(config.backend)},
    )

    # Create backend based on configuration
    try:
        if config.backend == CacheBackend.MEMORY:
            cache = _create_memory_cache(config)
        elif config.backend == CacheBackend.REDIS:
            cache = _create_redis_cache(config)
        else:
            raise ConfigurationError(
                f"Unknown cache backend: {config.backend}",
                details={
                    "backend": str(config.backend),
                    "supported": ["memory", "redis"],
                },
            )

        # Store instance in registry
        _cache_instances[name] = cache

        logger.info(
            "Cache instance '%s' created successfully",
            name,
            extra={"cache_name": name, "backend": str(config.backend)},
        )

        return cache

    except ConfigurationError:
        # Re-raise configuration errors as-is (already logged)
        raise
    except Exception as e:
        logger.error(
            "Unexpected error creating cache instance '%s': %s",
            name,
            e,
            extra={
                "cache_name": name,
                "backend": str(config.backend) if config else "unknown",
                "error": str(e),
            },
            exc_info=True,
        )
        raise ConfigurationError(
            f"Failed to create cache instance '{name}': {e}",
            details={
                "cache_name": name,
                "backend": str(config.backend) if config else "unknown",
                "error": str(e),
            },
        ) from e


def get_cache(name: str = "default") -> CacheInterface:
    """
    Get an existing cache instance by name.

    If the instance doesn't exist, it will be created automatically
    using the global configuration.

    Args:
        name: Cache instance name

    Returns:
        Cache backend instance
    """
    if name not in _cache_instances:
        logger.debug("Cache instance '%s' not found, creating new instance", name)
        return create_cache(name=name)

    return _cache_instances[name]


async def close_all_caches() -> None:
    """
    Close all cache instances and release resources.

    MUST be called during graceful shutdown per SDD.md.
    This ensures proper cleanup of connections and resources.
    """
    if not _cache_instances:
        logger.debug("No cache instances to close")
        return

    logger.info("Closing %d cache instance(s)...", len(_cache_instances))

    for name, cache in list(_cache_instances.items()):
        try:
            await cache.close()
            logger.info("Closed cache instance: %s", name)
        except Exception as e:
            logger.error(
                "Error closing cache instance '%s': %s",
                name,
                e,
                extra={"cache_name": name, "error": str(e)},
                exc_info=True,
            )

    _cache_instances.clear()
    logger.info("All cache instances closed")


def reset_cache_factory() -> None:
    """
    Reset the cache factory by clearing all instance references.

    Used for testing and hot-reload scenarios.
    Does NOT call close() on instances - use close_all_caches() for proper cleanup.

    Warning: Only use this in testing contexts.
    """
    count = len(_cache_instances)
    _cache_instances.clear()
    logger.debug("Reset cache factory, cleared %d instance reference(s)", count)


def clear_cache_registry() -> None:
    """
    Clear all cache instance references without closing them.

    Alias for reset_cache_factory() for backward compatibility.

    Warning: Only use this in testing contexts.
    """
    reset_cache_factory()


def list_cache_instances() -> list[str]:
    """
    List all registered cache instance names.

    Returns:
        List of cache instance names
    """
    return list(_cache_instances.keys())
