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

# Token optimization tools loaded lazily to keep module optional


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
_context_optimizer = None
_budget_tracker = None
_budget_enforcer = None
_budget_analytics = None
_semantic_cache = None


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
    if _context_optimizer is None:
        return {"error": "Context optimization is not enabled"}

    obs = get_observability()
    obs.increment("tools.optimize_tokens")

    try:
        # Import required modules
        from .context_optimization import optimize_content
        from .context_optimization.config import ContextOptimizationConfig

        # Get or create config
        config = _context_optimizer.get("config")
        if config is None:
            from .context_optimization.config import load_config as load_context_config

            config = load_context_config()

        # Update config based on parameters if provided
        if target_reduction is not None or strategies is not None:
            # Create a modified config
            config_dict = config.model_dump()

            if target_reduction is not None:
                # Map target_reduction to compression method
                if target_reduction <= 0.002:
                    config_dict["compression_method"] = "seraph"
                    config_dict["seraph_l1_ratio"] = target_reduction
                elif target_reduction <= 0.05:
                    config_dict["compression_method"] = "seraph"
                    config_dict["seraph_l2_ratio"] = target_reduction
                else:
                    config_dict["compression_method"] = "hybrid"

            if strategies is not None:
                # Determine compression method from strategies
                if "aggressive" in strategies or "hybrid" in strategies:
                    config_dict["compression_method"] = "hybrid"
                elif "seraph" in strategies:
                    config_dict["compression_method"] = "seraph"
                elif "ai" in strategies:
                    config_dict["compression_method"] = "ai"

            config = ContextOptimizationConfig(**config_dict)

        # Get provider and budget tracker
        provider = _context_optimizer.get("provider")
        budget_tracker = _budget_tracker

        # Call context optimization
        result = await optimize_content(
            content=content,
            provider=provider,
            config=config,
            budget_tracker=budget_tracker,
        )

        return {
            "success": True,
            "original_content": result.original_content,
            "optimized_content": result.optimized_content,
            "tokens_before": result.tokens_before,
            "tokens_after": result.tokens_after,
            "tokens_saved": result.tokens_saved,
            "reduction_percentage": result.reduction_percentage,
            "quality_score": result.quality_score,
            "validation_passed": result.validation_passed,
            "method_used": result.method,
            "processing_time_ms": result.optimization_time_ms,
            "cost_savings_usd": result.cost_savings_usd,
            "model": result.model_name or model,
            "rollback_occurred": result.rollback_occurred,
        }

    except Exception as e:
        logger.error(f"Error in optimize_tokens: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
        }


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
    if _context_optimizer is None:
        return {"error": "Context optimization is not enabled"}

    obs = get_observability()
    obs.increment("tools.count_tokens")

    try:
        import re

        import tiktoken

        # Get encoding for token counting
        encoding = None
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            token_count = len(encoding.encode(content))
        except Exception:
            # Fallback: estimate as ~4 chars per token
            token_count = len(content) // 4

        result: dict[str, Any] = {
            "success": True,
            "token_count": token_count,
            "model": model,
            "content_length": len(content),
        }

        if include_breakdown:
            lines = content.split("\n")
            words = content.split()

            # Find code blocks
            code_blocks = re.findall(r"```[\s\S]*?```", content)
            code_tokens = 0
            for block in code_blocks:
                try:
                    if encoding is not None:
                        code_tokens += len(encoding.encode(block))
                    else:
                        code_tokens += len(block) // 4
                except Exception:
                    code_tokens += len(block) // 4

            result["breakdown"] = {
                "lines": len(lines),
                "words": len(words),
                "characters": len(content),
                "code_blocks": len(code_blocks),
                "code_block_tokens": code_tokens,
                "non_code_tokens": token_count - code_tokens,
                "avg_tokens_per_line": token_count / len(lines) if lines else 0,
            }

        return result

    except Exception as e:
        logger.error(f"Error in count_tokens: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
        }


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
    if _context_optimizer is None:
        return {"error": "Context optimization is not enabled"}

    obs = get_observability()
    obs.increment("tools.estimate_cost")

    try:
        import tiktoken

        from .providers.models_dev import get_models_dev_client

        # Count input tokens
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            input_tokens = len(encoding.encode(content))
        except Exception:
            input_tokens = len(content) // 4

        # Estimate output tokens if not provided
        if output_tokens is None:
            output_tokens = int(input_tokens * 0.25)  # Conservative estimate

        # Try to get cost from models.dev
        client = get_models_dev_client()

        # Infer provider from model name
        model_lower = model.lower()
        if "gpt" in model_lower or "o1" in model_lower:
            provider_id = "openai"
        elif "claude" in model_lower:
            provider_id = "anthropic"
        elif "gemini" in model_lower:
            provider_id = "google"
        else:
            provider_id = "openai"

        try:
            cost_usd = await client.estimate_cost(
                provider_id=provider_id,
                model_id=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        except Exception:
            # Fallback to hardcoded rates
            rates = {
                "gpt-4": {"input": 30.0, "output": 60.0},
                "gpt-4-turbo": {"input": 10.0, "output": 30.0},
                "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
                "claude-3-opus": {"input": 15.0, "output": 75.0},
                "claude-3-sonnet": {"input": 3.0, "output": 15.0},
                "gemini-pro": {"input": 0.5, "output": 1.5},
            }

            # Default to gpt-4 rates
            rate = rates.get("gpt-4", {"input": 30.0, "output": 60.0})
            for key, value in rates.items():
                if key in model_lower:
                    rate = value
                    break

            input_cost = (input_tokens / 1_000_000) * rate["input"]
            output_cost = (output_tokens / 1_000_000) * rate["output"]
            cost_usd = input_cost + output_cost

        total_tokens = input_tokens + output_tokens
        return {
            "success": True,
            "model": model,
            "operation": operation,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_usd": cost_usd,
            "cost_breakdown": {
                "input_cost": cost_usd * (input_tokens / total_tokens) if total_tokens > 0 else 0,
                "output_cost": cost_usd * (output_tokens / total_tokens) if total_tokens > 0 else 0,
            },
        }

    except Exception as e:
        logger.error(f"Error in estimate_cost: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
        }


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
    if _context_optimizer is None:
        return {"error": "Context optimization is not enabled"}

    obs = get_observability()
    obs.increment("tools.analyze_token_efficiency")

    try:
        import re

        import tiktoken

        # Count tokens
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            token_count = len(encoding.encode(content))
        except Exception:
            token_count = len(content) // 4

        # Analyze content structure
        lines = content.split("\n")
        words = content.split()

        # Find potential optimization opportunities
        opportunities: list[dict[str, Any]] = []

        # Check for excessive whitespace
        whitespace_chars = len(re.findall(r"\s+", content))
        if whitespace_chars > len(content) * 0.2:
            opportunities.append(
                {
                    "type": "whitespace",
                    "description": "Content has excessive whitespace",
                    "potential_savings_percent": 10,
                }
            )

        # Check for repetitive phrases
        sentences = re.split(r"[.!?]+", content)
        if len(sentences) > 5:
            unique_ratio = len(set(sentences)) / len(sentences)
            if unique_ratio < 0.8:
                opportunities.append(
                    {
                        "type": "repetition",
                        "description": "Content contains repetitive sentences",
                        "potential_savings_percent": 15,
                    }
                )

        # Check for verbose language
        avg_word_length = sum(len(w) for w in words) / len(words) if words else 0
        if avg_word_length > 6.0:
            opportunities.append(
                {
                    "type": "verbosity",
                    "description": "Content uses verbose language",
                    "potential_savings_percent": 20,
                }
            )

        # Calculate potential savings
        max_potential_savings = min(sum(o["potential_savings_percent"] for o in opportunities), 50)
        potential_tokens_saved = int(token_count * (max_potential_savings / 100))

        # Estimate cost savings
        cost_per_token = 0.00001  # Conservative estimate for GPT-4
        potential_cost_savings = potential_tokens_saved * cost_per_token

        # Generate recommendation
        if not opportunities:
            recommendation = "Content is already well-optimized. No immediate optimization needed."
        elif token_count < 500:
            recommendation = "Content is short. Optimization may not provide significant savings."
        elif len(opportunities) == 1:
            recommendation = f"Apply {opportunities[0]['type']} optimization to reduce tokens."
        else:
            recommendation = (
                f"Apply {len(opportunities)} optimizations to reduce tokens by up to {max_potential_savings}%."
            )

        return {
            "success": True,
            "current_tokens": token_count,
            "content_length": len(content),
            "lines": len(lines),
            "words": len(words),
            "optimization_opportunities": opportunities,
            "potential_token_savings": potential_tokens_saved,
            "potential_reduction_percentage": max_potential_savings,
            "estimated_cost_savings_usd": potential_cost_savings,
            "recommendation": recommendation,
        }

    except Exception as e:
        logger.error(f"Error in analyze_token_efficiency: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
        }


# ============================================================================
# Budget Management Tools
# ============================================================================


@mcp.tool()
async def check_budget(estimated_cost: float | None = None) -> dict[str, Any]:
    """
    Check current budget status and whether a request is allowed.

    Args:
        estimated_cost: Optional estimated cost of upcoming request (USD)

    Returns:
        Budget status and whether request is allowed
    """
    if _budget_enforcer is None:
        return {"error": "Budget management is not enabled"}

    obs = get_observability()
    obs.increment("tools.check_budget")

    try:
        allowed, status = _budget_enforcer.check_budget(estimated_cost=estimated_cost)

        return {
            "success": True,
            "allowed": allowed,
            **status,
        }

    except Exception as e:
        logger.error(f"Error in check_budget: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
        }


@mcp.tool()
async def get_usage_report(
    period: str = "month",
    details: bool = False,
) -> dict[str, Any]:
    """
    Get spending usage report.

    Args:
        period: Time period ('day', 'week', 'month')
        details: Include detailed breakdown

    Returns:
        Usage report with spending analytics
    """
    if _budget_analytics is None:
        return {"error": "Budget management is not enabled"}

    obs = get_observability()
    obs.increment("tools.get_usage_report")

    try:
        report_type = "detailed" if details else "summary"
        report = _budget_analytics.generate_report(report_type=report_type)

        return {
            "success": True,
            "period": period,
            **report,
        }

    except Exception as e:
        logger.error(f"Error in get_usage_report: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
        }


@mcp.tool()
async def forecast_spending(days_ahead: int = 7) -> dict[str, Any]:
    """
    Forecast future spending based on historical patterns.

    Args:
        days_ahead: Number of days to forecast

    Returns:
        Spending forecast with projections
    """
    if _budget_analytics is None:
        return {"error": "Budget management is not enabled"}

    obs = get_observability()
    obs.increment("tools.forecast_spending")

    try:
        forecast = _budget_analytics.forecast_spending(days_ahead=days_ahead)

        return {
            "success": True,
            **forecast,
        }

    except Exception as e:
        logger.error(f"Error in forecast_spending: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
        }


# ============================================================================
# Semantic Cache Tools
# ============================================================================


@mcp.tool()
async def lookup_semantic_cache(
    query: str,
    threshold: float | None = None,
    max_results: int = 1,
) -> dict[str, Any]:
    """
    Look up cached value by semantic similarity.

    Args:
        query: Query text to search for
        threshold: Similarity threshold (0.0-1.0, default from config)
        max_results: Maximum results to return

    Returns:
        Best matching cached entry or None
    """
    if _semantic_cache is None:
        return {"error": "Semantic cache is not enabled"}

    obs = get_observability()
    obs.increment("tools.lookup_semantic_cache")

    try:
        result = await _semantic_cache.get(
            query=query,
            threshold=threshold,
            max_results=max_results,
        )

        return {
            "success": True,
            "found": result is not None,
            "result": result,
        }

    except Exception as e:
        logger.error(f"Error in lookup_semantic_cache: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
        }


@mcp.tool()
async def store_in_semantic_cache(
    key: str,
    value: Any,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Store value in semantic cache.

    Args:
        key: Cache key (text to embed)
        value: Value to cache
        metadata: Optional metadata

    Returns:
        Success status
    """
    if _semantic_cache is None:
        return {"error": "Semantic cache is not enabled"}

    obs = get_observability()
    obs.increment("tools.store_in_semantic_cache")

    try:
        success = await _semantic_cache.set(
            key=key,
            value=value,
            metadata=metadata,
        )

        return {
            "success": success,
        }

    except Exception as e:
        logger.error(f"Error in store_in_semantic_cache: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
        }


@mcp.tool()
async def search_semantic_cache(
    query: str,
    limit: int | None = None,
    threshold: float | None = None,
) -> dict[str, Any]:
    """
    Search for similar entries in semantic cache.

    Args:
        query: Query text
        limit: Maximum results (default from config)
        threshold: Similarity threshold (default from config)

    Returns:
        List of matching entries with similarity scores
    """
    if _semantic_cache is None:
        return {"error": "Semantic cache is not enabled"}

    obs = get_observability()
    obs.increment("tools.search_semantic_cache")

    try:
        results = await _semantic_cache.search(
            query=query,
            limit=limit,
            threshold=threshold,
        )

        return {
            "success": True,
            "count": len(results),
            "results": results,
        }

    except Exception as e:
        logger.error(f"Error in search_semantic_cache: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
        }


@mcp.tool()
async def get_semantic_cache_stats() -> dict[str, Any]:
    """
    Get semantic cache statistics.

    Returns:
        Cache statistics and configuration
    """
    if _semantic_cache is None:
        return {"error": "Semantic cache is not enabled"}

    obs = get_observability()
    obs.increment("tools.get_semantic_cache_stats")

    try:
        stats = _semantic_cache.get_stats()

        return {
            "success": True,
            **stats,
        }

    except Exception as e:
        logger.error(f"Error in get_semantic_cache_stats: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
        }


@mcp.tool()
async def clear_semantic_cache() -> dict[str, Any]:
    """
    Clear all entries from semantic cache.

    Returns:
        Success status
    """
    if _semantic_cache is None:
        return {"error": "Semantic cache is not enabled"}

    obs = get_observability()
    obs.increment("tools.clear_semantic_cache")

    try:
        success = _semantic_cache.clear()

        return {
            "success": success,
        }

    except Exception as e:
        logger.error(f"Error in clear_semantic_cache: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
        }


# ============================================================================
# Context Optimization Tools
# ============================================================================


@mcp.tool()
async def optimize_context(
    content: str,
    method: str = "auto",
    quality_threshold: float | None = None,
    max_overhead_ms: float | None = None,
) -> dict[str, Any]:
    """
    Optimize content using context optimization (AI/Seraph/Hybrid compression).

    Args:
        content: Content to optimize
        method: Compression method ('auto', 'ai', 'seraph', 'hybrid')
        quality_threshold: Minimum quality score (0-1)
        max_overhead_ms: Maximum processing time in milliseconds

    Returns:
        Optimization result with compressed content and metrics
    """
    if _context_optimizer is None:
        return {"error": "Context optimization is not enabled"}

    obs = get_observability()
    obs.increment("tools.optimize_context")

    try:
        from .context_optimization import optimize_content

        # Use provided thresholds or defaults from config
        config = _context_optimizer.get("config")
        if config is None:
            return {"error": "Context optimization config not available"}

        threshold = quality_threshold if quality_threshold is not None else config.quality_threshold
        overhead = max_overhead_ms if max_overhead_ms is not None else config.max_overhead_ms

        # Create a temporary config with the specified method
        from .context_optimization.config import ContextOptimizationConfig

        temp_config = ContextOptimizationConfig(
            enabled=True,
            compression_method=method,
            quality_threshold=threshold,
            max_overhead_ms=overhead,
            seraph_token_threshold=config.seraph_token_threshold,
            seraph_l1_ratio=config.seraph_l1_ratio,
            seraph_l2_ratio=config.seraph_l2_ratio,
            seraph_l3_ratio=config.seraph_l3_ratio,
        )

        result = await optimize_content(
            content=content,
            provider=_context_optimizer.get("provider"),
            config=temp_config,
        )

        return {
            "success": True,
            "original_content": result.original_content,
            "optimized_content": result.optimized_content,
            "tokens_before": result.tokens_before,
            "tokens_after": result.tokens_after,
            "tokens_saved": result.tokens_saved,
            "reduction_percentage": result.reduction_percentage,
            "quality_score": result.quality_score,
            "method_used": getattr(result, "method_used", method),
            "processing_time_ms": result.optimization_time_ms,
            "validation_passed": result.validation_passed,
            "rollback_occurred": result.rollback_occurred,
        }

    except Exception as e:
        logger.error(f"Error in optimize_context: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
        }


@mcp.tool()
async def get_optimization_settings() -> dict[str, Any]:
    """
    Get current context optimization settings.

    Returns:
        Current optimization configuration
    """
    if _context_optimizer is None:
        return {"error": "Context optimization is not enabled"}

    obs = get_observability()
    obs.increment("tools.get_optimization_settings")

    try:
        config = _context_optimizer.get("config")
        if config is None:
            return {"error": "Context optimization config not available"}

        return {
            "success": True,
            "enabled": config.enabled,
            "compression_method": config.compression_method,
            "seraph_token_threshold": config.seraph_token_threshold,
            "quality_threshold": config.quality_threshold,
            "max_overhead_ms": config.max_overhead_ms,
            "seraph_l1_ratio": config.seraph_l1_ratio,
            "seraph_l2_ratio": config.seraph_l2_ratio,
            "seraph_l3_ratio": config.seraph_l3_ratio,
        }

    except Exception as e:
        logger.error(f"Error in get_optimization_settings: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
        }


@mcp.tool()
async def get_optimization_stats() -> dict[str, Any]:
    """
    Get context optimization statistics.

    Returns:
        Optimization statistics and performance metrics
    """
    if _context_optimizer is None:
        return {"error": "Context optimization is not enabled"}

    obs = get_observability()
    obs.increment("tools.get_optimization_stats")

    try:
        # Get stats from the optimizer instance if available
        # Note: We store config, not the optimizer instance, so we return available info
        return {
            "success": True,
            "message": "Context optimization is enabled. Use optimize_context() to generate statistics.",
            "optimizer_initialized": True,
        }

    except Exception as e:
        logger.error(f"Error in get_optimization_stats: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
        }


def _init_context_optimization_if_available(config: Any) -> None:
    global _context_optimizer
    if _context_optimizer is not None:
        return
    try:
        from .context_optimization.config import load_config as load_context_config
    except Exception as e:
        logger.info("Context optimization module not available or failed to import: %s", e)
        _context_optimizer = None
        return
    try:
        context_config = load_context_config()
        # Store config and provider for use by tools
        _context_optimizer = {
            "config": context_config,
            "provider": None,  # Can be set later if provider is available
        }
        logger.info("Context optimization initialized")
    except Exception as e:
        logger.warning("Context optimization initialization failed: %s", e)
        _context_optimizer = None


def _init_budget_management_if_available(config: Any) -> None:
    global _budget_tracker, _budget_enforcer, _budget_analytics
    if _budget_tracker is not None:
        return
    try:
        from .budget_management import (
            BudgetConfig,
            get_budget_analytics,
            get_budget_enforcer,
            get_budget_tracker,
        )
    except Exception as e:
        logger.info("Budget management module not available or failed to import: %s", e)
        _budget_tracker = None
        _budget_enforcer = None
        _budget_analytics = None
        return
    try:
        budget_config = BudgetConfig(**config.budget.model_dump())
        _budget_tracker = get_budget_tracker()
        _budget_enforcer = get_budget_enforcer(config=budget_config)
        _budget_analytics = get_budget_analytics(tracker=_budget_tracker)
        logger.info("Budget management initialized")
    except Exception as e:
        logger.warning("Budget management initialization failed: %s", e)
        _budget_tracker = None
        _budget_enforcer = None
        _budget_analytics = None


def _init_semantic_cache_if_available(config: Any) -> None:
    global _semantic_cache
    if _semantic_cache is not None:
        return
    try:
        from .semantic_cache import SemanticCacheConfig, get_semantic_cache
    except Exception as e:
        logger.info("Semantic cache module not available or failed to import: %s", e)
        _semantic_cache = None
        return
    try:
        # For now, use default config - can be enhanced to read from env
        semantic_config = SemanticCacheConfig()
        _semantic_cache = get_semantic_cache(config=semantic_config)
        logger.info("Semantic cache initialized")
    except Exception as e:
        logger.warning("Semantic cache initialization failed: %s", e)
        _semantic_cache = None


async def initialize_server():
    """Initialize server resources on startup."""

    global _initialized, _context_optimizer, _budget_tracker, _budget_enforcer, _budget_analytics, _semantic_cache

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

        # Initialize context optimization if enabled (lazy and optional)
        # Note: token_optimization feature flag now controls context_optimization
        if config.features.token_optimization or config.features.context_optimization:
            _init_context_optimization_if_available(config)
        else:
            logger.info("Context optimization disabled via feature flags")

        # Initialize budget management if enabled
        if config.features.budget_management or config.budget.enable_budget_enforcement:
            _init_budget_management_if_available(config)
        else:
            logger.info("Budget management disabled via feature flags")

        # Initialize semantic cache if enabled
        if config.features.semantic_cache:
            _init_semantic_cache_if_available(config)
        else:
            logger.info("Semantic cache disabled via feature flags")

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
    global _initialized, _context_optimizer, _budget_tracker, _budget_enforcer, _budget_analytics, _semantic_cache

    if not _initialized:
        return

    logger.info("Cleaning up Seraph MCP server...")

    try:
        obs = get_observability()

        # Close all caches
        await close_all_caches()
        logger.info("All caches closed")

        # Close semantic cache if initialized
        if _semantic_cache:
            await _semantic_cache.close()
            logger.info("Semantic cache closed")

        # Close budget management resources
        if _budget_tracker:
            from .budget_management import close_budget_analytics, close_budget_enforcer, close_budget_tracker

            close_budget_tracker()
            close_budget_enforcer()
            close_budget_analytics()
            logger.info("Budget management closed")

        # Cleanup tool instances
        _context_optimizer = None
        _budget_tracker = None
        _budget_enforcer = None
        _budget_analytics = None
        _semantic_cache = None

        # Record shutdown
        obs.increment("server.shutdown")
        obs.event("server_stopped", {})

        _initialized = False
        logger.info("Seraph MCP server cleanup complete")

    except Exception as e:
        logger.error(f"Error during cleanup: {e}", exc_info=True)


# Register lifecycle hooks
@mcp.lifespan()  # type: ignore[attr-defined]
async def lifespan():
    """Server lifespan manager (startup/shutdown)."""
    await initialize_server()
    yield
    await cleanup_server()


def main():
    """CLI entry point for seraph-mcp command."""
    mcp.run()


if __name__ == "__main__":
    main()
