"""
Unit Tests for Cost Estimator

Tests the CostEstimator class with pricing database and cost calculations.
Covers all major LLM providers and cost comparison features.
"""

from unittest.mock import Mock, patch

import pytest

from src.token_optimization.cost_estimator import (
    CostEstimator,
    ModelPricing,
    PricingTier,
    get_cost_estimator,
)


class TestPricingTier:
    """Tests for PricingTier enum."""

    def test_tier_values(self):
        """Test that all tier values are strings."""
        assert PricingTier.FREE.value == "free"
        assert PricingTier.BUDGET.value == "budget"
        assert PricingTier.STANDARD.value == "standard"
        assert PricingTier.PREMIUM.value == "premium"
        assert PricingTier.ENTERPRISE.value == "enterprise"


class TestModelPricing:
    """Tests for ModelPricing dataclass."""

    @pytest.fixture
    def sample_pricing(self):
        """Create a sample ModelPricing instance."""
        return ModelPricing(
            model_name="gpt-4",
            provider="openai",
            input_price_per_1k=0.03,
            output_price_per_1k=0.06,
            tier=PricingTier.PREMIUM,
            context_window=8192,
            supports_streaming=True,
            last_updated="2024-01",
        )

    def test_initialization(self, sample_pricing):
        """Test ModelPricing initializes correctly."""
        assert sample_pricing.model_name == "gpt-4"
        assert sample_pricing.provider == "openai"
        assert sample_pricing.input_price_per_1k == 0.03
        assert sample_pricing.output_price_per_1k == 0.06
        assert sample_pricing.tier == PricingTier.PREMIUM
        assert sample_pricing.context_window == 8192
        assert sample_pricing.supports_streaming is True

    def test_calculate_cost_basic(self, sample_pricing):
        """Test basic cost calculation."""
        # 1000 input tokens, 500 output tokens
        cost = sample_pricing.calculate_cost(1000, 500)

        expected = (1000 / 1000) * 0.03 + (500 / 1000) * 0.06
        assert cost == expected
        assert cost == 0.03 + 0.03  # 0.06 total

    def test_calculate_cost_zero_tokens(self, sample_pricing):
        """Test cost calculation with zero tokens."""
        cost = sample_pricing.calculate_cost(0, 0)
        assert cost == 0.0

    def test_calculate_cost_only_input(self, sample_pricing):
        """Test cost calculation with only input tokens."""
        cost = sample_pricing.calculate_cost(2000, 0)

        expected = (2000 / 1000) * 0.03
        assert cost == expected
        assert cost == 0.06

    def test_calculate_cost_only_output(self, sample_pricing):
        """Test cost calculation with only output tokens."""
        cost = sample_pricing.calculate_cost(0, 1000)

        expected = (1000 / 1000) * 0.06
        assert cost == expected
        assert cost == 0.06

    def test_calculate_cost_large_numbers(self, sample_pricing):
        """Test cost calculation with large token counts."""
        cost = sample_pricing.calculate_cost(100000, 50000)

        expected = (100000 / 1000) * 0.03 + (50000 / 1000) * 0.06
        assert cost == expected
        assert cost == 3.0 + 3.0  # 6.0 total


class TestCostEstimator:
    """Tests for CostEstimator class."""

    @pytest.fixture
    def estimator(self):
        """Create a CostEstimator instance."""
        with patch("src.token_optimization.cost_estimator.get_token_counter") as mock:
            mock_counter = Mock()
            mock_counter.count_tokens.return_value = 100
            mock.return_value = mock_counter
            return CostEstimator()

    def test_initialization(self, estimator):
        """Test CostEstimator initializes correctly."""
        assert estimator is not None
        assert estimator.counter is not None

    def test_pricing_database_complete(self, estimator):
        """Test that pricing database contains expected models."""
        expected_models = [
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-3.5-turbo",
            "claude-3-opus",
            "claude-3-sonnet",
            "claude-3-5-sonnet",
            "claude-3-haiku",
            "claude-3-5-haiku",
            "gemini-pro",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "mistral-large",
            "mistral-medium",
            "mistral-small",
        ]

        for model in expected_models:
            assert model in estimator.PRICING_DATABASE

    def test_pricing_database_structure(self, estimator):
        """Test that all pricing entries have correct structure."""
        for model_name, pricing in estimator.PRICING_DATABASE.items():
            assert isinstance(pricing, ModelPricing)
            assert pricing.model_name == model_name
            assert pricing.input_price_per_1k >= 0
            assert pricing.output_price_per_1k >= 0
            assert isinstance(pricing.tier, PricingTier)
            assert pricing.context_window > 0
            assert isinstance(pricing.supports_streaming, bool)

    def test_get_pricing_exact_match(self, estimator):
        """Test _get_pricing with exact model match."""
        pricing = estimator._get_pricing("gpt-4")

        assert pricing is not None
        assert pricing.model_name == "gpt-4"
        assert pricing.provider == "openai"

    def test_get_pricing_partial_match(self, estimator):
        """Test _get_pricing with partial model match."""
        # Should match "gpt-4" from "gpt-4-0125-preview"
        pricing = estimator._get_pricing("gpt-4-0125-preview")

        assert pricing is not None
        assert pricing.model_name == "gpt-4"

    def test_get_pricing_unknown_model(self, estimator):
        """Test _get_pricing with unknown model."""
        pricing = estimator._get_pricing("unknown-model-xyz")

        assert pricing is None

    @patch("src.token_optimization.cost_estimator.get_token_counter")
    def test_estimate_cost_basic(self, mock_get_counter):
        """Test basic cost estimation."""
        mock_counter = Mock()
        mock_counter.count_tokens.return_value = 1000
        mock_get_counter.return_value = mock_counter

        estimator = CostEstimator()
        result = estimator.estimate_cost(
            content="Test content",
            model="gpt-4",
            operation="completion",
            output_tokens=500,
        )

        assert "model" in result
        assert result["model"] == "gpt-4"
        assert "provider" in result
        assert result["provider"] == "openai"
        assert "input_tokens" in result
        assert result["input_tokens"] == 1000
        assert "output_tokens" in result
        assert result["output_tokens"] == 500
        assert "total_cost_usd" in result
        assert result["total_cost_usd"] > 0

    @patch("src.token_optimization.cost_estimator.get_token_counter")
    def test_estimate_cost_estimates_output_tokens(self, mock_get_counter):
        """Test that output tokens are estimated if not provided."""
        mock_counter = Mock()
        mock_counter.count_tokens.return_value = 1000
        mock_get_counter.return_value = mock_counter

        estimator = CostEstimator()
        result = estimator.estimate_cost(
            content="Test content",
            model="gpt-4",
            operation="completion",
            output_tokens=None,  # Should be estimated
        )

        # Should estimate output tokens as ~40% of input
        assert result["output_tokens"] > 0
        assert result["output_tokens"] == int(1000 * 0.4)

    @patch("src.token_optimization.cost_estimator.get_token_counter")
    def test_estimate_cost_unknown_model(self, mock_get_counter):
        """Test cost estimation with unknown model."""
        mock_counter = Mock()
        mock_counter.count_tokens.return_value = 100
        mock_get_counter.return_value = mock_counter

        estimator = CostEstimator()
        result = estimator.estimate_cost(
            content="Test",
            model="unknown-model",
            operation="completion",
        )

        assert "error" in result
        assert "Pricing data not available" in result["error"]

    @patch("src.token_optimization.cost_estimator.get_token_counter")
    def test_estimate_cost_cost_breakdown(self, mock_get_counter):
        """Test that cost breakdown is included."""
        mock_counter = Mock()
        mock_counter.count_tokens.return_value = 1000
        mock_get_counter.return_value = mock_counter

        estimator = CostEstimator()
        result = estimator.estimate_cost(
            content="Test",
            model="gpt-4",
            output_tokens=500,
        )

        assert "input_cost_usd" in result
        assert "output_cost_usd" in result
        assert "total_cost_usd" in result

        # Total should equal sum of input and output
        total = result["input_cost_usd"] + result["output_cost_usd"]
        assert abs(total - result["total_cost_usd"]) < 0.000001  # Float precision

    @patch("src.token_optimization.cost_estimator.get_token_counter")
    def test_estimate_cost_pricing_details(self, mock_get_counter):
        """Test that pricing details are included."""
        mock_counter = Mock()
        mock_counter.count_tokens.return_value = 1000
        mock_get_counter.return_value = mock_counter

        estimator = CostEstimator()
        result = estimator.estimate_cost(
            content="Test",
            model="gpt-4",
            output_tokens=500,
        )

        assert "pricing" in result
        assert "input_per_1k" in result["pricing"]
        assert "output_per_1k" in result["pricing"]

    @patch("src.token_optimization.cost_estimator.get_token_counter")
    def test_compare_model_costs_basic(self, mock_get_counter):
        """Test basic model cost comparison."""
        mock_counter = Mock()
        mock_counter.count_tokens.return_value = 1000
        mock_get_counter.return_value = mock_counter

        estimator = CostEstimator()
        result = estimator.compare_model_costs(
            content="Test content",
            models=["gpt-4", "gpt-3.5-turbo", "claude-3-haiku"],
        )

        assert "estimates" in result
        assert len(result["estimates"]) == 3
        assert "cheapest" in result
        assert "most_expensive" in result
        assert "potential_savings" in result

    @patch("src.token_optimization.cost_estimator.get_token_counter")
    def test_compare_model_costs_sorted_by_cost(self, mock_get_counter):
        """Test that comparison results are sorted by cost."""
        mock_counter = Mock()
        mock_counter.count_tokens.return_value = 1000
        mock_get_counter.return_value = mock_counter

        estimator = CostEstimator()
        result = estimator.compare_model_costs(
            content="Test",
            models=["gpt-4", "gpt-4o-mini", "claude-3-opus"],
        )

        estimates = result["estimates"]

        # Should be sorted from cheapest to most expensive
        for i in range(len(estimates) - 1):
            assert estimates[i]["total_cost_usd"] <= estimates[i + 1]["total_cost_usd"]

    @patch("src.token_optimization.cost_estimator.get_token_counter")
    def test_compare_model_costs_uses_defaults(self, mock_get_counter):
        """Test that default model list is used when not specified."""
        mock_counter = Mock()
        mock_counter.count_tokens.return_value = 1000
        mock_get_counter.return_value = mock_counter

        estimator = CostEstimator()
        result = estimator.compare_model_costs(content="Test")

        # Should use default models
        assert len(result["estimates"]) > 0

    @patch("src.token_optimization.cost_estimator.get_token_counter")
    def test_compare_model_costs_savings_calculation(self, mock_get_counter):
        """Test that savings are calculated correctly."""
        mock_counter = Mock()
        mock_counter.count_tokens.return_value = 1000
        mock_get_counter.return_value = mock_counter

        estimator = CostEstimator()
        result = estimator.compare_model_costs(
            content="Test",
            models=["gpt-4o-mini", "gpt-4"],  # Different price tiers
        )

        savings = result["potential_savings"]

        assert savings is not None
        assert "absolute_usd" in savings
        assert "percentage" in savings
        assert savings["absolute_usd"] >= 0
        assert savings["percentage"] >= 0

    @patch("src.token_optimization.cost_estimator.get_token_counter")
    def test_compare_model_costs_recommendation(self, mock_get_counter):
        """Test that a recommendation is generated."""
        mock_counter = Mock()
        mock_counter.count_tokens.return_value = 1000
        mock_get_counter.return_value = mock_counter

        estimator = CostEstimator()
        result = estimator.compare_model_costs(
            content="Test",
            models=["gpt-4", "gpt-3.5-turbo"],
        )

        assert "recommendation" in result
        assert isinstance(result["recommendation"], str)
        assert len(result["recommendation"]) > 0

    @patch("src.token_optimization.cost_estimator.get_token_counter")
    def test_calculate_monthly_cost(self, mock_get_counter):
        """Test monthly cost calculation."""
        mock_get_counter.return_value = Mock()

        estimator = CostEstimator()
        result = estimator.calculate_monthly_cost(
            daily_requests=100,
            avg_input_tokens=1000,
            avg_output_tokens=500,
            model="gpt-4",
        )

        assert "model" in result
        assert "daily_requests" in result
        assert result["daily_requests"] == 100
        assert "daily_cost_usd" in result
        assert "monthly_cost_usd" in result
        assert "annual_cost_usd" in result

        # Monthly should be daily * 30
        assert abs(result["monthly_cost_usd"] - result["daily_cost_usd"] * 30) < 0.01

    @patch("src.token_optimization.cost_estimator.get_token_counter")
    def test_calculate_monthly_cost_breakdown(self, mock_get_counter):
        """Test that monthly cost includes detailed breakdown."""
        mock_get_counter.return_value = Mock()

        estimator = CostEstimator()
        result = estimator.calculate_monthly_cost(
            daily_requests=50,
            avg_input_tokens=2000,
            avg_output_tokens=1000,
            model="gpt-3.5-turbo",
        )

        assert "breakdown" in result
        breakdown = result["breakdown"]

        assert "daily_input_cost" in breakdown
        assert "daily_output_cost" in breakdown
        assert "avg_cost_per_request" in breakdown

    @patch("src.token_optimization.cost_estimator.get_token_counter")
    def test_calculate_monthly_cost_unknown_model(self, mock_get_counter):
        """Test monthly cost with unknown model."""
        mock_get_counter.return_value = Mock()

        estimator = CostEstimator()
        result = estimator.calculate_monthly_cost(
            daily_requests=10,
            avg_input_tokens=100,
            avg_output_tokens=50,
            model="unknown-model",
        )

        assert "error" in result

    def test_get_pricing_info(self, estimator):
        """Test get_pricing_info returns model pricing details."""
        result = estimator.get_pricing_info("gpt-4")

        assert result is not None
        assert result["model"] == "gpt-4"
        assert result["provider"] == "openai"
        assert result["tier"] == "premium"
        assert "input_price_per_1k" in result
        assert "output_price_per_1k" in result
        assert "context_window" in result
        assert "supports_streaming" in result

    def test_get_pricing_info_unknown_model(self, estimator):
        """Test get_pricing_info with unknown model."""
        result = estimator.get_pricing_info("unknown-model")

        assert result is None

    def test_list_models_by_tier_all(self, estimator):
        """Test listing all models without tier filter."""
        result = estimator.list_models_by_tier(tier=None)

        assert len(result) > 0
        assert "gpt-4" in result
        assert "claude-3-opus" in result

    def test_list_models_by_tier_budget(self, estimator):
        """Test listing budget tier models."""
        result = estimator.list_models_by_tier(tier=PricingTier.BUDGET)

        assert len(result) > 0
        # Budget models should be in the list
        assert any("mini" in model or "3.5" in model or "haiku" in model for model in result)

    def test_list_models_by_tier_premium(self, estimator):
        """Test listing premium tier models."""
        result = estimator.list_models_by_tier(tier=PricingTier.PREMIUM)

        assert len(result) > 0
        # Premium models like gpt-4 and claude-3-opus
        premium_models = ["gpt-4", "claude-3-opus"]
        assert any(model in result for model in premium_models)

    def test_generate_recommendation_single_estimate(self, estimator):
        """Test recommendation generation with single estimate."""
        estimates = [{"model": "gpt-4", "total_cost_usd": 0.03}]

        recommendation = estimator._generate_recommendation(estimates)

        assert isinstance(recommendation, str)
        assert "gpt-4" in recommendation
        assert "$0.03" in recommendation

    def test_generate_recommendation_multiple_estimates(self, estimator):
        """Test recommendation generation with multiple estimates."""
        estimates = [
            {"model": "gpt-4o-mini", "total_cost_usd": 0.0001},
            {"model": "gpt-3.5-turbo", "total_cost_usd": 0.002},
            {"model": "gpt-4", "total_cost_usd": 0.06},
        ]

        recommendation = estimator._generate_recommendation(estimates)

        assert "gpt-4o-mini" in recommendation  # Should recommend cheapest
        assert "Saves" in recommendation or "saves" in recommendation

    def test_generate_recommendation_empty_estimates(self, estimator):
        """Test recommendation with empty estimates."""
        recommendation = estimator._generate_recommendation([])

        assert "No cost data available" in recommendation


class TestSingletonFunction:
    """Tests for singleton helper function."""

    def test_get_cost_estimator_returns_instance(self):
        """Test get_cost_estimator returns CostEstimator instance."""
        with patch("src.token_optimization.cost_estimator.get_token_counter"):
            estimator = get_cost_estimator()

            assert isinstance(estimator, CostEstimator)

    def test_get_cost_estimator_singleton(self):
        """Test get_cost_estimator returns singleton instance."""
        with patch("src.token_optimization.cost_estimator.get_token_counter"):
            estimator1 = get_cost_estimator()
            estimator2 = get_cost_estimator()

            assert estimator1 is estimator2


class TestRealWorldScenarios:
    """Tests for real-world usage scenarios."""

    @pytest.fixture
    def estimator(self):
        """Create a CostEstimator instance."""
        with patch("src.token_optimization.cost_estimator.get_token_counter") as mock:
            mock_counter = Mock()
            mock_counter.count_tokens.return_value = 1000
            mock.return_value = mock_counter
            return CostEstimator()

    @patch("src.token_optimization.cost_estimator.get_token_counter")
    def test_cost_comparison_for_api_migration(self, mock_get_counter, estimator):
        """Test comparing costs when migrating between providers."""
        mock_counter = Mock()
        mock_counter.count_tokens.return_value = 5000
        mock_get_counter.return_value = mock_counter

        # Compare OpenAI vs Anthropic vs Google
        result = estimator.compare_model_costs(
            content="Long document content",
            models=["gpt-4", "claude-3-opus", "gemini-1.5-pro"],
            output_tokens=2000,
        )

        assert len(result["estimates"]) == 3
        assert result["cheapest"] is not None
        assert result["potential_savings"] is not None

    @patch("src.token_optimization.cost_estimator.get_token_counter")
    def test_budget_planning(self, mock_get_counter, estimator):
        """Test budget planning for production deployment."""
        mock_get_counter.return_value = Mock()

        # Plan for 1000 requests per day
        result = estimator.calculate_monthly_cost(
            daily_requests=1000,
            avg_input_tokens=500,
            avg_output_tokens=300,
            model="gpt-4o",
        )

        assert result["monthly_cost_usd"] > 0
        assert result["annual_cost_usd"] == result["monthly_cost_usd"] * 12

    @patch("src.token_optimization.cost_estimator.get_token_counter")
    def test_cost_optimization_opportunity(self, mock_get_counter, estimator):
        """Test identifying cost optimization opportunities."""
        mock_counter = Mock()
        mock_counter.count_tokens.return_value = 10000  # Large content
        mock_get_counter.return_value = mock_counter

        # Compare expensive vs budget models
        result = estimator.compare_model_costs(
            content="Very large document",
            models=["gpt-4", "gpt-4o", "gpt-4o-mini"],
        )

        savings = result["potential_savings"]

        # Should show significant savings potential
        assert savings["percentage"] > 50  # Significant difference


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.fixture
    def estimator(self):
        """Create a CostEstimator instance."""
        with patch("src.token_optimization.cost_estimator.get_token_counter") as mock:
            mock_counter = Mock()
            mock_counter.count_tokens.return_value = 0
            mock.return_value = mock_counter
            return CostEstimator()

    @patch("src.token_optimization.cost_estimator.get_token_counter")
    def test_zero_token_cost(self, mock_get_counter):
        """Test cost estimation with zero tokens."""
        mock_counter = Mock()
        mock_counter.count_tokens.return_value = 0
        mock_get_counter.return_value = mock_counter

        estimator = CostEstimator()
        result = estimator.estimate_cost("", "gpt-4", output_tokens=0)

        assert result["input_tokens"] == 0
        assert result["output_tokens"] == 0
        assert result["total_cost_usd"] == 0.0

    @patch("src.token_optimization.cost_estimator.get_token_counter")
    def test_very_large_token_count(self, mock_get_counter):
        """Test cost estimation with very large token counts."""
        mock_counter = Mock()
        mock_counter.count_tokens.return_value = 1000000  # 1M tokens
        mock_get_counter.return_value = mock_counter

        estimator = CostEstimator()
        result = estimator.estimate_cost(
            "Huge content",
            "gpt-4",
            output_tokens=500000,
        )

        assert result["total_cost_usd"] > 0
        assert isinstance(result["total_cost_usd"], int | float)

    def test_pricing_database_no_negative_prices(self, estimator):
        """Test that pricing database has no negative prices."""
        for _model, pricing in estimator.PRICING_DATABASE.items():
            assert pricing.input_price_per_1k >= 0
            assert pricing.output_price_per_1k >= 0
