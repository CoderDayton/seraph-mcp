"""
Cost Estimator Module

Provides cost estimation for LLM API calls across multiple providers.
Includes real-time pricing data and cost comparison utilities.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .counter import get_token_counter

logger = logging.getLogger(__name__)


class PricingTier(str, Enum):
    """Pricing tiers for different model capabilities."""

    FREE = "free"
    BUDGET = "budget"
    STANDARD = "standard"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


@dataclass
class ModelPricing:
    """Pricing information for a specific model."""

    model_name: str
    provider: str
    input_price_per_1k: float  # USD per 1K tokens
    output_price_per_1k: float  # USD per 1K tokens
    tier: PricingTier
    context_window: int
    supports_streaming: bool
    last_updated: str

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate total cost for token usage.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Total cost in USD
        """
        input_cost = (input_tokens / 1000) * self.input_price_per_1k
        output_cost = (output_tokens / 1000) * self.output_price_per_1k
        return input_cost + output_cost


class CostEstimator:
    """
    Multi-provider cost estimator for LLM APIs.

    Provides accurate cost estimation based on current pricing.
    Pricing data as of January 2024.
    """

    # Pricing database (USD per 1K tokens)
    PRICING_DATABASE: dict[str, ModelPricing] = {
        # OpenAI GPT-4 models
        "gpt-4": ModelPricing(
            model_name="gpt-4",
            provider="openai",
            input_price_per_1k=0.03,
            output_price_per_1k=0.06,
            tier=PricingTier.PREMIUM,
            context_window=8192,
            supports_streaming=True,
            last_updated="2024-01",
        ),
        "gpt-4-turbo": ModelPricing(
            model_name="gpt-4-turbo",
            provider="openai",
            input_price_per_1k=0.01,
            output_price_per_1k=0.03,
            tier=PricingTier.PREMIUM,
            context_window=128000,
            supports_streaming=True,
            last_updated="2024-01",
        ),
        "gpt-4o": ModelPricing(
            model_name="gpt-4o",
            provider="openai",
            input_price_per_1k=0.005,
            output_price_per_1k=0.015,
            tier=PricingTier.STANDARD,
            context_window=128000,
            supports_streaming=True,
            last_updated="2024-01",
        ),
        "gpt-4o-mini": ModelPricing(
            model_name="gpt-4o-mini",
            provider="openai",
            input_price_per_1k=0.00015,
            output_price_per_1k=0.0006,
            tier=PricingTier.BUDGET,
            context_window=128000,
            supports_streaming=True,
            last_updated="2024-01",
        ),
        # OpenAI GPT-3.5 models
        "gpt-3.5-turbo": ModelPricing(
            model_name="gpt-3.5-turbo",
            provider="openai",
            input_price_per_1k=0.0005,
            output_price_per_1k=0.0015,
            tier=PricingTier.BUDGET,
            context_window=16385,
            supports_streaming=True,
            last_updated="2024-01",
        ),
        # Anthropic Claude models
        "claude-3-opus": ModelPricing(
            model_name="claude-3-opus",
            provider="anthropic",
            input_price_per_1k=0.015,
            output_price_per_1k=0.075,
            tier=PricingTier.PREMIUM,
            context_window=200000,
            supports_streaming=True,
            last_updated="2024-01",
        ),
        "claude-3-sonnet": ModelPricing(
            model_name="claude-3-sonnet",
            provider="anthropic",
            input_price_per_1k=0.003,
            output_price_per_1k=0.015,
            tier=PricingTier.STANDARD,
            context_window=200000,
            supports_streaming=True,
            last_updated="2024-01",
        ),
        "claude-3-5-sonnet": ModelPricing(
            model_name="claude-3-5-sonnet",
            provider="anthropic",
            input_price_per_1k=0.003,
            output_price_per_1k=0.015,
            tier=PricingTier.STANDARD,
            context_window=200000,
            supports_streaming=True,
            last_updated="2024-01",
        ),
        "claude-3-haiku": ModelPricing(
            model_name="claude-3-haiku",
            provider="anthropic",
            input_price_per_1k=0.00025,
            output_price_per_1k=0.00125,
            tier=PricingTier.BUDGET,
            context_window=200000,
            supports_streaming=True,
            last_updated="2024-01",
        ),
        "claude-3-5-haiku": ModelPricing(
            model_name="claude-3-5-haiku",
            provider="anthropic",
            input_price_per_1k=0.0008,
            output_price_per_1k=0.004,
            tier=PricingTier.BUDGET,
            context_window=200000,
            supports_streaming=True,
            last_updated="2024-01",
        ),
        # Google Gemini models
        "gemini-pro": ModelPricing(
            model_name="gemini-pro",
            provider="google",
            input_price_per_1k=0.00025,
            output_price_per_1k=0.0005,
            tier=PricingTier.BUDGET,
            context_window=32000,
            supports_streaming=True,
            last_updated="2024-01",
        ),
        "gemini-1.5-pro": ModelPricing(
            model_name="gemini-1.5-pro",
            provider="google",
            input_price_per_1k=0.00125,
            output_price_per_1k=0.005,
            tier=PricingTier.STANDARD,
            context_window=1000000,
            supports_streaming=True,
            last_updated="2024-01",
        ),
        "gemini-1.5-flash": ModelPricing(
            model_name="gemini-1.5-flash",
            provider="google",
            input_price_per_1k=0.000075,
            output_price_per_1k=0.0003,
            tier=PricingTier.BUDGET,
            context_window=1000000,
            supports_streaming=True,
            last_updated="2024-01",
        ),
        # Mistral models
        "mistral-large": ModelPricing(
            model_name="mistral-large",
            provider="mistral",
            input_price_per_1k=0.004,
            output_price_per_1k=0.012,
            tier=PricingTier.STANDARD,
            context_window=32000,
            supports_streaming=True,
            last_updated="2024-01",
        ),
        "mistral-medium": ModelPricing(
            model_name="mistral-medium",
            provider="mistral",
            input_price_per_1k=0.0027,
            output_price_per_1k=0.0081,
            tier=PricingTier.STANDARD,
            context_window=32000,
            supports_streaming=True,
            last_updated="2024-01",
        ),
        "mistral-small": ModelPricing(
            model_name="mistral-small",
            provider="mistral",
            input_price_per_1k=0.001,
            output_price_per_1k=0.003,
            tier=PricingTier.BUDGET,
            context_window=32000,
            supports_streaming=True,
            last_updated="2024-01",
        ),
    }

    def __init__(self) -> None:
        """Initialize cost estimator."""
        self.counter = get_token_counter()

    def estimate_cost(
        self,
        content: str,
        model: str,
        operation: str = "completion",
        output_tokens: int | None = None,
    ) -> dict[str, Any]:
        """
        Estimate cost for an LLM API call.

        Args:
            content: Input content
            model: Model name
            operation: Type of operation (completion, embedding, etc.)
            output_tokens: Expected output tokens (None = estimate)

        Returns:
            Cost estimation with breakdown
        """
        # Get pricing info
        pricing = self._get_pricing(model)
        if pricing is None:
            logger.warning(f"No pricing data for model: {model}")
            return {
                "model": model,
                "error": "Pricing data not available",
                "estimated_cost": 0.0,
            }

        # Count input tokens
        input_tokens = self.counter.count_tokens(content, model)

        # Estimate output tokens if not provided
        if output_tokens is None:
            if operation == "completion":
                # Estimate: response typically 30-50% of input
                output_tokens = int(input_tokens * 0.4)
            else:
                output_tokens = 0

        # Calculate costs
        input_cost = (input_tokens / 1000) * pricing.input_price_per_1k
        output_cost = (output_tokens / 1000) * pricing.output_price_per_1k
        total_cost = input_cost + output_cost

        return {
            "model": model,
            "provider": pricing.provider,
            "tier": pricing.tier.value,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_cost_usd": round(input_cost, 6),
            "output_cost_usd": round(output_cost, 6),
            "total_cost_usd": round(total_cost, 6),
            "pricing": {
                "input_per_1k": pricing.input_price_per_1k,
                "output_per_1k": pricing.output_price_per_1k,
            },
            "operation": operation,
        }

    def compare_model_costs(
        self,
        content: str,
        models: list[str] | None = None,
        output_tokens: int | None = None,
    ) -> dict[str, Any]:
        """
        Compare costs across multiple models.

        Args:
            content: Input content
            models: Models to compare (None = use defaults)
            output_tokens: Expected output tokens

        Returns:
            Cost comparison with recommendations
        """
        if models is None:
            models = [
                "gpt-4o-mini",
                "gpt-3.5-turbo",
                "claude-3-haiku",
                "claude-3-5-haiku",
                "gemini-1.5-flash",
            ]

        # Estimate costs for each model
        estimates = []
        for model in models:
            estimate = self.estimate_cost(content, model, output_tokens=output_tokens)
            if "error" not in estimate:
                estimates.append(estimate)

        # Sort by cost
        estimates.sort(key=lambda x: x["total_cost_usd"])

        # Find cheapest and most expensive
        cheapest = estimates[0] if estimates else None
        most_expensive = estimates[-1] if estimates else None

        # Calculate savings
        savings = None
        if cheapest and most_expensive:
            savings = {
                "absolute_usd": round(most_expensive["total_cost_usd"] - cheapest["total_cost_usd"], 6),
                "percentage": round(
                    (
                        (most_expensive["total_cost_usd"] - cheapest["total_cost_usd"])
                        / most_expensive["total_cost_usd"]
                        * 100
                    )
                    if most_expensive["total_cost_usd"] > 0
                    else 0,
                    2,
                ),
            }

        return {
            "estimates": estimates,
            "cheapest": cheapest,
            "most_expensive": most_expensive,
            "potential_savings": savings,
            "recommendation": self._generate_recommendation(estimates),
        }

    def calculate_monthly_cost(
        self,
        daily_requests: int,
        avg_input_tokens: int,
        avg_output_tokens: int,
        model: str,
    ) -> dict[str, Any]:
        """
        Calculate projected monthly costs.

        Args:
            daily_requests: Average requests per day
            avg_input_tokens: Average input tokens per request
            avg_output_tokens: Average output tokens per request
            model: Model name

        Returns:
            Monthly cost projection
        """
        pricing = self._get_pricing(model)
        if pricing is None:
            return {"error": "Pricing data not available"}

        # Calculate daily cost
        daily_input_cost = (daily_requests * avg_input_tokens / 1000) * pricing.input_price_per_1k
        daily_output_cost = (daily_requests * avg_output_tokens / 1000) * pricing.output_price_per_1k
        daily_cost = daily_input_cost + daily_output_cost

        # Project to monthly (30 days)
        monthly_cost = daily_cost * 30

        return {
            "model": model,
            "provider": pricing.provider,
            "daily_requests": daily_requests,
            "daily_cost_usd": round(daily_cost, 2),
            "monthly_cost_usd": round(monthly_cost, 2),
            "annual_cost_usd": round(monthly_cost * 12, 2),
            "breakdown": {
                "daily_input_cost": round(daily_input_cost, 4),
                "daily_output_cost": round(daily_output_cost, 4),
                "avg_cost_per_request": round(daily_cost / daily_requests, 6),
            },
        }

    def get_pricing_info(self, model: str) -> dict[str, Any] | None:
        """
        Get pricing information for a model.

        Args:
            model: Model name

        Returns:
            Pricing information or None
        """
        pricing = self._get_pricing(model)
        if pricing is None:
            return None

        return {
            "model": pricing.model_name,
            "provider": pricing.provider,
            "tier": pricing.tier.value,
            "input_price_per_1k": pricing.input_price_per_1k,
            "output_price_per_1k": pricing.output_price_per_1k,
            "context_window": pricing.context_window,
            "supports_streaming": pricing.supports_streaming,
            "last_updated": pricing.last_updated,
        }

    def list_models_by_tier(self, tier: PricingTier | None = None) -> list[str]:
        """
        List models by pricing tier.

        Args:
            tier: Pricing tier (None = all models)

        Returns:
            List of model names
        """
        if tier is None:
            return list(self.PRICING_DATABASE.keys())

        return [model for model, pricing in self.PRICING_DATABASE.items() if pricing.tier == tier]

    def _get_pricing(self, model: str) -> ModelPricing | None:
        """
        Get pricing for a model.

        Args:
            model: Model name

        Returns:
            ModelPricing or None
        """
        # Exact match
        if model in self.PRICING_DATABASE:
            return self.PRICING_DATABASE[model]

        # Partial match (e.g., "gpt-4-0125-preview" -> "gpt-4")
        for model_key in self.PRICING_DATABASE:
            if model.startswith(model_key):
                return self.PRICING_DATABASE[model_key]

        return None

    def _generate_recommendation(self, estimates: list[dict[str, Any]]) -> str:
        """
        Generate cost optimization recommendation.

        Args:
            estimates: List of cost estimates

        Returns:
            Recommendation string
        """
        if not estimates:
            return "No cost data available"

        cheapest = estimates[0]

        if len(estimates) == 1:
            return f"Using {cheapest['model']} (${cheapest['total_cost_usd']:.6f})"

        savings_vs_most_expensive = estimates[-1]["total_cost_usd"] - cheapest["total_cost_usd"]
        savings_pct = (
            savings_vs_most_expensive / estimates[-1]["total_cost_usd"] * 100
            if estimates[-1]["total_cost_usd"] > 0
            else 0
        )

        return (
            f"Use {cheapest['model']} for lowest cost (${cheapest['total_cost_usd']:.6f}). "
            f"Saves ${savings_vs_most_expensive:.6f} ({savings_pct:.1f}%) vs most expensive option."
        )


# Singleton instance
_estimator_instance: CostEstimator | None = None


def get_cost_estimator() -> CostEstimator:
    """
    Get singleton CostEstimator instance.

    Returns:
        Shared CostEstimator instance
    """
    global _estimator_instance
    if _estimator_instance is None:
        _estimator_instance = CostEstimator()
    return _estimator_instance
