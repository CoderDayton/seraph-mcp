"""
Token Optimization MCP Tools

Exposes token optimization capabilities as MCP tools for the Seraph MCP server.
These tools are automatically registered when the feature is enabled.

Per SDD.md specifications:
- All tools use typed Pydantic models for inputs/outputs
- Integrate with core cache and observability
- Support feature flags for enabling/disabling
- Provide comprehensive error handling
"""

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.cache import create_cache
from src.observability import get_observability
from .config import TokenOptimizationConfig
from .cost_estimator import get_cost_estimator
from .counter import get_token_counter
from .optimizer import get_optimizer

logger = logging.getLogger(__name__)


# ============================================================================
# Tool Input/Output Models
# ============================================================================


class OptimizeTokensInput(BaseModel):
    """Input for optimize_tokens tool."""

    content: str = Field(..., description="Content to optimize")
    target_reduction: Optional[float] = Field(
        None,
        ge=0.0,
        le=0.5,
        description="Target reduction ratio (0.0-0.5, default: config value)",
    )
    model: str = Field("gpt-4", description="Model to optimize for")
    strategies: Optional[List[str]] = Field(
        None, description="Optimization strategies to apply (default: config value)"
    )


class CountTokensInput(BaseModel):
    """Input for count_tokens tool."""

    content: str = Field(..., description="Content to count tokens for")
    model: str = Field("gpt-4", description="Model name")
    include_breakdown: bool = Field(
        False, description="Include detailed token breakdown"
    )


class EstimateCostInput(BaseModel):
    """Input for estimate_cost tool."""

    content: str = Field(..., description="Input content")
    model: str = Field(..., description="Model name")
    operation: str = Field("completion", description="Operation type")
    output_tokens: Optional[int] = Field(
        None, description="Expected output tokens (None = estimate)"
    )


class AnalyzeEfficiencyInput(BaseModel):
    """Input for analyze_token_efficiency tool."""

    content: str = Field(..., description="Content to analyze")
    model: str = Field("gpt-4", description="Model to analyze for")


# ============================================================================
# Token Optimization Tools
# ============================================================================


class TokenOptimizationTools:
    """
    MCP tools for token optimization.

    Provides 4 core tools:
    - optimize_tokens: Reduce token count while preserving quality
    - count_tokens: Accurate token counting for any model
    - estimate_cost: Calculate API costs before making requests
    - analyze_token_efficiency: Identify optimization opportunities
    """

    def __init__(self, config: Optional[TokenOptimizationConfig] = None) -> None:
        """
        Initialize token optimization tools.

        Args:
            config: Configuration (uses defaults if None)
        """
        self.config = config or TokenOptimizationConfig()
        self.config.validate_strategies()

        # Initialize components
        self.counter = get_token_counter()
        self.optimizer = get_optimizer(
            quality_threshold=self.config.quality_threshold,
            preserve_code_blocks=self.config.preserve_code_blocks,
            preserve_formatting=self.config.preserve_formatting,
        )
        self.estimator = get_cost_estimator()

        # Cache and observability
        self.cache = create_cache()
        self.obs = get_observability()

        logger.info(
            f"Token optimization tools initialized: "
            f"enabled={self.config.enabled}, "
            f"target_reduction={self.config.default_reduction_target}"
        )

    def optimize_tokens(
        self,
        content: str,
        target_reduction: Optional[float] = None,
        model: str = "gpt-4",
        strategies: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Optimize content to reduce token count.

        Applies multiple optimization strategies to reduce token usage
        while maintaining content quality above the configured threshold.

        Args:
            content: Content to optimize
            target_reduction: Target reduction ratio (0.0-0.5)
            model: Model to optimize for
            strategies: List of strategies to apply

        Returns:
            Optimization result with metrics and optimized content
        """
        self.obs.increment("tools.optimize_tokens")

        if not self.config.enabled:
            logger.warning("Token optimization is disabled")
            return {
                "error": "Token optimization is disabled",
                "enabled": False,
                "original_content": content,
            }

        # Use defaults from config if not specified
        if target_reduction is None:
            target_reduction = self.config.default_reduction_target

        if strategies is None:
            strategies = self.config.optimization_strategies

        # Validate target reduction
        if not 0.0 <= target_reduction <= 0.5:
            return {
                "error": f"Invalid target_reduction: {target_reduction}. Must be 0.0-0.5",
                "original_content": content,
            }

        # Check cache if enabled
        cache_key = None
        if self.config.cache_optimizations:
            cache_key = f"opt:{model}:{target_reduction}:{hash(content)}"
            cached_result = None
            try:
                # Cache operations are async, but we'll handle sync for now
                # In production, this should be properly awaited
                import asyncio
                try:
                    cached_result = asyncio.get_event_loop().run_until_complete(
                        self.cache.get(cache_key)
                    )
                except RuntimeError:
                    # No event loop running
                    pass
            except Exception as e:
                logger.warning(f"Cache lookup failed: {e}")

            if cached_result:
                self.obs.increment("cache.hits")
                logger.debug(f"Cache hit for optimization: {cache_key}")
                return cached_result

        try:
            # Perform optimization
            with self.obs.trace("token_optimization.optimize", tags={"model": model}):
                result = self.optimizer.optimize(
                    content=content,
                    target_reduction=target_reduction,
                    model=model,
                    strategies=strategies,
                    aggressive=self.config.enable_aggressive_mode,
                )

            # Check quality threshold
            if result.quality_score < self.config.quality_threshold:
                logger.warning(
                    f"Optimization quality below threshold: "
                    f"{result.quality_score:.2f} < {self.config.quality_threshold:.2f}"
                )
                self.obs.increment("optimization.quality_failures")
                return {
                    "error": "Optimization quality below threshold",
                    "quality_score": result.quality_score,
                    "quality_threshold": self.config.quality_threshold,
                    "original_content": content,
                }

            # Check overhead
            if result.processing_time_ms > self.config.max_overhead_ms:
                logger.warning(
                    f"Optimization overhead exceeded: "
                    f"{result.processing_time_ms:.2f}ms > {self.config.max_overhead_ms}ms"
                )
                self.obs.increment("optimization.overhead_exceeded")

            # Build response
            response = result.to_dict()
            response["optimized_content"] = result.optimized_content
            response["success"] = True

            # Cache result if enabled
            if self.config.cache_optimizations and cache_key:
                try:
                    import asyncio
                    try:
                        asyncio.get_event_loop().run_until_complete(
                            self.cache.set(
                                cache_key, response, ttl=self.config.cache_ttl_seconds
                            )
                        )
                    except RuntimeError:
                        pass
                except Exception as e:
                    logger.warning(f"Failed to cache result: {e}")

            # Record metrics
            self.obs.increment("optimization.success")
            self.obs.histogram(
                "optimization.tokens_saved",
                result.tokens_saved
            )
            self.obs.histogram(
                "optimization.processing_time_ms",
                result.processing_time_ms
            )

            return response

        except Exception as e:
            logger.error(f"Error optimizing tokens: {e}", exc_info=True)
            self.obs.increment("optimization.errors")
            return {
                "error": f"Optimization failed: {str(e)}",
                "original_content": content,
            }

    def count_tokens(
        self,
        content: str,
        model: str = "gpt-4",
        include_breakdown: bool = False,
    ) -> Dict[str, Any]:
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
        self.obs.increment("tools.count_tokens")

        try:
            with self.obs.trace("token_optimization.count", tags={"model": model}):
                if include_breakdown:
                    result = self.counter.get_token_breakdown(content, model)
                else:
                    token_count = self.counter.count_tokens(content, model)
                    result = {
                        "token_count": token_count,
                        "model": model,
                        "character_count": len(content),
                    }

            self.obs.increment("token_counting.success")
            return result

        except Exception as e:
            logger.error(f"Error counting tokens: {e}", exc_info=True)
            self.obs.increment("token_counting.errors")
            return {
                "error": f"Token counting failed: {str(e)}",
                "model": model,
            }

    def estimate_cost(
        self,
        content: str,
        model: str,
        operation: str = "completion",
        output_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
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
        self.obs.increment("tools.estimate_cost")

        try:
            with self.obs.trace("token_optimization.estimate_cost", tags={"model": model}):
                result = self.estimator.estimate_cost(
                    content=content,
                    model=model,
                    operation=operation,
                    output_tokens=output_tokens,
                )

            self.obs.increment("cost_estimation.success")
            return result

        except Exception as e:
            logger.error(f"Error estimating cost: {e}", exc_info=True)
            self.obs.increment("cost_estimation.errors")
            return {
                "error": f"Cost estimation failed: {str(e)}",
                "model": model,
            }

    def analyze_token_efficiency(
        self,
        content: str,
        model: str = "gpt-4",
    ) -> Dict[str, Any]:
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
        self.obs.increment("tools.analyze_token_efficiency")

        try:
            with self.obs.trace("token_optimization.analyze", tags={"model": model}):
                # Get efficiency analysis
                analysis = self.optimizer.analyze_efficiency(content, model)

                # Add cost implications
                current_cost = self.estimator.estimate_cost(
                    content, model, operation="completion"
                )

                if "error" not in current_cost:
                    # Estimate cost after optimization
                    potential_tokens = (
                        analysis["current_tokens"] - analysis["total_potential_savings"]
                    )
                    potential_savings_pct = (
                        analysis["potential_reduction_percentage"] / 100
                    )

                    analysis["cost_analysis"] = {
                        "current_cost_usd": current_cost["total_cost_usd"],
                        "potential_cost_usd": round(
                            current_cost["total_cost_usd"] * (1 - potential_savings_pct),
                            6,
                        ),
                        "potential_savings_usd": round(
                            current_cost["total_cost_usd"] * potential_savings_pct,
                            6,
                        ),
                    }

            self.obs.increment("efficiency_analysis.success")
            return analysis

        except Exception as e:
            logger.error(f"Error analyzing token efficiency: {e}", exc_info=True)
            self.obs.increment("efficiency_analysis.errors")
            return {
                "error": f"Analysis failed: {str(e)}",
                "model": model,
            }


# ============================================================================
# Singleton Instance
# ============================================================================

_tools_instance: Optional[TokenOptimizationTools] = None


def get_tools(
    config: Optional[TokenOptimizationConfig] = None
) -> TokenOptimizationTools:
    """
    Get singleton TokenOptimizationTools instance.

    Args:
        config: Configuration (uses defaults if None)

    Returns:
        Shared TokenOptimizationTools instance
    """
    global _tools_instance
    if _tools_instance is None:
        _tools_instance = TokenOptimizationTools(config=config)
    return _tools_instance
