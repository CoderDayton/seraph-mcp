"""
Seraph MCP — Server

FastMCP server using stdio transport (Model Context Protocol).
This is the ONLY server entrypoint per SDD.md.

Following SDD.md mandatory rules:
- Single entrypoint: This is the canonical server
- MCP stdio protocol (not HTTP)
- Graceful shutdown with resource cleanup
- Structured logging with trace IDs
- Configuration via typed Pydantic models only
- Monolithic architecture with feature flags for enabling/disabling capabilities
"""

import logging
from typing import Any

from fastmcp import FastMCP

from .cache import close_all_caches, create_cache
from .config import load_config
from .observability import get_observability, initialize_observability
from .token_optimization.tools import get_tools as get_token_optimization_tools

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastMCP server
mcp = FastMCP("Seraph MCP - AI Optimization Platform")

# Global state
_initialized = False
_token_optimization_tools = None


@mcp.tool()
async def check_status(include_details: bool = False) -> dict[str, Any]:
    """
    Check system health and status.

    Args:
        include_details: Include detailed cache and observability stats

    Returns:
        System status information
    """
    obs = get_observability()
    obs.increment("tools.check_status")

    cache = create_cache()
    cache_stats = await cache.get_stats()

    status = {
        "status": "healthy",
        "service": "seraph-mcp",
        "version": "1.0.0",
        "cache": {
            "backend": cache_stats.get("backend"),
            "size": cache_stats.get("size"),
            "hit_rate": cache_stats.get("hit_rate", 0.0),
        },
    }

    if include_details:
        status["cache_details"] = cache_stats
        status["observability"] = {
            "backend": obs.backend,
            "metrics_enabled": obs.enable_metrics,
            "tracing_enabled": obs.enable_tracing,
        }

    return status


@mcp.tool()
async def get_cache_stats() -> dict[str, Any]:
    """
    Get detailed cache statistics.

    Returns:
        Cache performance metrics and statistics
    """
    obs = get_observability()
    obs.increment("tools.get_cache_stats")

    cache = create_cache()
    stats = await cache.get_stats()

    return stats


@mcp.tool()
async def cache_get(key: str) -> Any | None:
    """
    Retrieve value from cache.

    Args:
        key: Cache key to retrieve

    Returns:
        Cached value or None if not found
    """
    obs = get_observability()
    obs.increment("tools.cache_get")

    cache = create_cache()

    with obs.trace("cache.get", tags={"key": key}):
        value = await cache.get(key)

    if value is not None:
        obs.increment("cache.hits")
    else:
        obs.increment("cache.misses")

    return value


@mcp.tool()
async def cache_set(key: str, value: Any, ttl: int | None = None) -> bool:
    """
    Store value in cache.

    Args:
        key: Cache key
        value: Value to store
        ttl: Time-to-live in seconds (None = use default)

    Returns:
        True if successful
    """
    obs = get_observability()
    obs.increment("tools.cache_set")

    cache = create_cache()

    with obs.trace("cache.set", tags={"key": key}):
        success = await cache.set(key, value, ttl)

    if success:
        obs.increment("cache.sets")

    return success


@mcp.tool()
async def cache_delete(key: str) -> bool:
    """
    Delete key from cache.

    Args:
        key: Cache key to delete

    Returns:
        True if key was deleted, False if not found
    """
    obs = get_observability()
    obs.increment("tools.cache_delete")

    cache = create_cache()

    with obs.trace("cache.delete", tags={"key": key}):
        deleted = await cache.delete(key)

    if deleted:
        obs.increment("cache.deletes")

    return deleted


@mcp.tool()
async def cache_clear() -> bool:
    """
    Clear all entries from cache.

    Returns:
        True if cache was cleared
    """
    obs = get_observability()
    obs.increment("tools.cache_clear")

    cache = create_cache()

    with obs.trace("cache.clear"):
        success = await cache.clear()

    return success


@mcp.tool()
async def get_metrics() -> dict[str, Any]:
    """
    Get observability metrics.

    Returns:
        Current metrics snapshot
    """
    obs = get_observability()
    obs.increment("tools.get_metrics")

    metrics = obs.get_metrics()

    return metrics


# ============================================================================
# Token Optimization Tools (conditionally registered based on feature flags)
# ============================================================================


@mcp.tool()
async def optimize_tokens(
    content: str,
    target_reduction: float | None = None,
    model: str = "gpt-4",
    strategies: list[str] | None = None,
) -> dict[str, Any]:
    """
    Optimize content to reduce token count while preserving quality.

    Applies multiple optimization strategies to reduce token usage
    while maintaining content quality above the configured threshold.

    Args:
        content: Content to optimize
        target_reduction: Target reduction ratio (0.0-0.5, default: 0.20)
        model: Model to optimize for (default: "gpt-4")
        strategies: List of strategies to apply (default: config value)

    Returns:
        Optimization result with metrics and optimized content
    """
    if _token_optimization_tools is None:
        return {"error": "Token optimization is not enabled"}

    return _token_optimization_tools.optimize_tokens(
        content=content,
        target_reduction=target_reduction,
        model=model,
        strategies=strategies,
    )


@mcp.tool()
async def count_tokens(
    content: str,
    model: str = "gpt-4",
    include_breakdown: bool = False,
) -> dict[str, Any]:
    """
    Count tokens in content for specified model.

    Provides accurate token counting using provider-specific libraries
    (tiktoken for OpenAI, anthropic for Claude) with fallback to estimation.

    Args:
        content: Content to count tokens for
        model: Model name (e.g., "gpt-4", "claude-3-opus")
        include_breakdown: Include detailed breakdown with metadata

    Returns:
        Token count and optional metadata
    """
    if _token_optimization_tools is None:
        return {"error": "Token optimization is not enabled"}

    return _token_optimization_tools.count_tokens(
        content=content,
        model=model,
        include_breakdown=include_breakdown,
    )


@mcp.tool()
async def estimate_cost(
    content: str,
    model: str,
    operation: str = "completion",
    output_tokens: int | None = None,
) -> dict[str, Any]:
    """
    Estimate cost for LLM API call.

    Calculates expected costs based on current pricing data for
    major LLM providers (OpenAI, Anthropic, Google, Mistral).

    Args:
        content: Input content
        model: Model name
        operation: Type of operation (completion, embedding, etc.)
        output_tokens: Expected output tokens (None = estimate)

    Returns:
        Cost estimation with detailed breakdown
    """
    if _token_optimization_tools is None:
        return {"error": "Token optimization is not enabled"}

    return _token_optimization_tools.estimate_cost(
        content=content,
        model=model,
        operation=operation,
        output_tokens=output_tokens,
    )


@mcp.tool()
async def analyze_token_efficiency(
    content: str,
    model: str = "gpt-4",
) -> dict[str, Any]:
    """
    Analyze content for optimization opportunities.

    Identifies potential token savings by analyzing content structure,
    redundancy, and optimization opportunities. Includes cost implications.

    Args:
        content: Content to analyze
        model: Model to analyze for

    Returns:
        Analysis with optimization suggestions and potential savings
    """
    if _token_optimization_tools is None:
        return {"error": "Token optimization is not enabled"}

    return _token_optimization_tools.analyze_token_efficiency(
        content=content,
        model=model,
    )


async def initialize_server():
    """Initialize server resources on startup."""
    global _initialized, _token_optimization_tools

    if _initialized:
        return

    logger.info("Initializing Seraph MCP server...")

    try:
        # Load configuration
        config = load_config()
        logger.info(f"Configuration loaded: environment={config.environment}")

        # Initialize observability
        obs = initialize_observability(
            enable_metrics=config.observability.enable_metrics,
            enable_tracing=config.observability.enable_tracing,
            backend=config.observability.backend,
        )
        logger.info(f"Observability initialized: backend={obs.backend}")

        # Initialize cache
        cache = create_cache()
        stats = await cache.get_stats()
        logger.info(f"Cache initialized: backend={stats['backend']}")

        # Initialize token optimization if enabled
        if config.features.token_optimization:
            from .token_optimization.config import TokenOptimizationConfig

            token_config = TokenOptimizationConfig(**config.token_optimization.model_dump())
            _token_optimization_tools = get_token_optimization_tools(config=token_config)
            logger.info("Token optimization tools initialized")
        else:
            logger.info("Token optimization disabled via feature flags")

        # Record startup
        obs.increment("server.startup")
        obs.event(
            "server_started",
            {
                "environment": config.environment,
                "cache_backend": config.cache.backend,
                "features_enabled": {
                    "token_optimization": config.features.token_optimization,
                    "model_routing": config.features.model_routing,
                    "semantic_cache": config.features.semantic_cache,
                    "context_optimization": config.features.context_optimization,
                    "budget_management": config.features.budget_management,
                    "quality_preservation": config.features.quality_preservation,
                },
            },
        )

        _initialized = True
        logger.info("Seraph MCP server initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize server: {e}", exc_info=True)
        raise


async def cleanup_server():
    """Cleanup server resources on shutdown."""
    global _initialized, _token_optimization_tools

    if not _initialized:
        return

    logger.info("Cleaning up Seraph MCP server...")

    try:
        obs = get_observability()

        # Close all caches
        await close_all_caches()
        logger.info("All caches closed")

        # Cleanup tool instances
        _token_optimization_tools = None

        # Record shutdown
        obs.increment("server.shutdown")
        obs.event("server_stopped", {})

        _initialized = False
        logger.info("Seraph MCP server cleanup complete")

    except Exception as e:
        logger.error(f"Error during cleanup: {e}", exc_info=True)


# Register lifecycle hooks
@mcp.lifespan()
async def lifespan():
    """Server lifespan manager (startup/shutdown)."""
    await initialize_server()
    yield
    await cleanup_server()


if __name__ == "__main__":
    # Run the MCP server
    mcp.run()
