"""
Token Optimization Module

Provides automatic token reduction and cost estimation for LLM requests.
Integrated into the main Seraph MCP package.

Public API:
    - TokenOptimizationConfig: Configuration schema
    - TokenCounter: Multi-provider token counter
    - TokenOptimizer: Token optimization engine
    - CostEstimator: LLM cost estimation
    - count_tokens: Convenience function
    - get_token_counter: Get counter instance
    - get_optimizer: Get optimizer instance
    - get_cost_estimator: Get estimator instance

Example:
    >>> from seraph_mcp.token_optimization import count_tokens, get_optimizer
    >>> tokens = count_tokens("Hello, world!", model="gpt-4")
    >>> optimizer = get_optimizer()
    >>> result = optimizer.optimize("Your content here...", model="gpt-4")
"""

from .config import TokenOptimizationConfig
from .cost_estimator import CostEstimator, ModelPricing, PricingTier, get_cost_estimator
from .counter import ModelProvider, TokenCounter, count_tokens, get_token_counter
from .optimizer import (
    OptimizationResult,
    OptimizationStrategy,
    TokenOptimizer,
    get_optimizer,
)

__all__ = [
    # Configuration
    "TokenOptimizationConfig",
    # Token counter
    "TokenCounter",
    "ModelProvider",
    "get_token_counter",
    "count_tokens",
    # Optimizer
    "TokenOptimizer",
    "OptimizationStrategy",
    "OptimizationResult",
    "get_optimizer",
    # Cost estimator
    "CostEstimator",
    "ModelPricing",
    "PricingTier",
    "get_cost_estimator",
]
