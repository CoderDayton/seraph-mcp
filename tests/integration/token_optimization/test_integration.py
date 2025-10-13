"""
Integration Tests for Token Optimization Feature

Tests the complete token optimization feature integration with the MCP server.
Covers end-to-end workflows, feature flags, cache integration, and observability.
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.token_optimization.config import TokenOptimizationConfig
from src.token_optimization.tools import TokenOptimizationTools


class TestTokenOptimizationIntegration:
    """Integration tests for token optimization feature."""

    @pytest.fixture
    async def tools(self):
        """Create TokenOptimizationTools instance with mocked dependencies."""
        config = TokenOptimizationConfig(
            enabled=True,
            default_reduction_target=0.20,
            quality_threshold=0.90,
            cache_optimizations=True,
            optimization_strategies=["whitespace", "redundancy", "compression"],
        )

        with patch("src.token_optimization.tools.create_cache") as mock_cache_factory:
            with patch("src.token_optimization.tools.get_observability") as mock_obs_factory:
                # Mock cache
                mock_cache = AsyncMock()
                mock_cache.get.return_value = None
                mock_cache.set.return_value = True
                mock_cache_factory.return_value = mock_cache

                # Mock observability
                mock_obs = Mock()
                mock_obs.increment = Mock()
                mock_obs.histogram = Mock()
                mock_obs.trace = Mock()
                mock_obs.trace.return_value.__enter__ = Mock()
                mock_obs.trace.return_value.__exit__ = Mock()
                mock_obs_factory.return_value = mock_obs

                tools = TokenOptimizationTools(config=config)
                tools.cache = mock_cache
                tools.obs = mock_obs

                yield tools

    @pytest.mark.asyncio
    async def test_optimize_tokens_end_to_end(self, tools):
        """Test complete optimization workflow."""
        content = "This  is   a  test   with    extra     spaces and redundancy."

        with patch.object(tools.counter, "count_tokens", side_effect=[50, 42]):
            result = tools.optimize_tokens(
                content=content,
                target_reduction=0.20,
                model="gpt-4",
                strategies=["whitespace", "redundancy"],
            )

            assert result["success"] is True
            assert result["original_tokens"] == 50
            assert result["optimized_tokens"] == 42
            assert result["tokens_saved"] == 8
            assert "optimized_content" in result
            assert len(result["strategies_applied"]) > 0

    @pytest.mark.asyncio
    async def test_optimize_tokens_with_caching(self, tools):
        """Test optimization with cache hit."""
        content = "Test content for caching"

        # First call - cache miss
        with patch.object(tools.counter, "count_tokens", side_effect=[20, 18]):
            result1 = tools.optimize_tokens(content=content, model="gpt-4")

            # Verify cache set was called
            assert tools.cache.set.called

        # Second call - should attempt cache lookup
        tools.cache.get.return_value = result1

        result2 = tools.optimize_tokens(content=content, model="gpt-4")

        # Both results should be similar
        assert result2 == result1

    @pytest.mark.asyncio
    async def test_optimize_tokens_quality_threshold_failure(self, tools):
        """Test optimization fails when quality below threshold."""
        content = "Test content"

        with patch.object(tools.optimizer, "optimize") as mock_optimize:
            from src.token_optimization.optimizer import OptimizationResult

            # Simulate low quality result
            mock_result = OptimizationResult(
                original_content=content,
                optimized_content="T",
                original_tokens=20,
                optimized_tokens=1,
                reduction_ratio=0.95,
                strategies_applied=["aggressive"],
                quality_score=0.50,  # Below threshold
                processing_time_ms=50.0,
                metadata={},
            )
            mock_optimize.return_value = mock_result

            result = tools.optimize_tokens(content=content, model="gpt-4")

            assert "error" in result
            assert "quality below threshold" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_optimize_tokens_disabled(self, tools):
        """Test optimization returns error when disabled."""
        tools.config.enabled = False

        result = tools.optimize_tokens(content="Test", model="gpt-4")

        assert "error" in result
        assert result["enabled"] is False

    @pytest.mark.asyncio
    async def test_count_tokens_basic(self, tools):
        """Test token counting integration."""
        with patch.object(tools.counter, "count_tokens", return_value=42):
            result = tools.count_tokens(
                content="Test content here",
                model="gpt-4",
                include_breakdown=False,
            )

            assert result["token_count"] == 42
            assert result["model"] == "gpt-4"
            assert "character_count" in result

    @pytest.mark.asyncio
    async def test_count_tokens_with_breakdown(self, tools):
        """Test token counting with detailed breakdown."""
        breakdown_data = {
            "token_count": 42,
            "model": "gpt-4",
            "provider": "openai",
            "character_count": 100,
            "chars_per_token": 2.38,
            "method": "exact",
        }

        with patch.object(tools.counter, "get_token_breakdown", return_value=breakdown_data):
            result = tools.count_tokens(
                content="Test content",
                model="gpt-4",
                include_breakdown=True,
            )

            assert result["provider"] == "openai"
            assert result["method"] == "exact"
            assert result["chars_per_token"] == 2.38

    @pytest.mark.asyncio
    async def test_estimate_cost_basic(self, tools):
        """Test cost estimation integration."""
        cost_data = {
            "model": "gpt-4",
            "provider": "openai",
            "tier": "premium",
            "input_tokens": 1000,
            "output_tokens": 500,
            "total_tokens": 1500,
            "input_cost_usd": 0.03,
            "output_cost_usd": 0.03,
            "total_cost_usd": 0.06,
            "pricing": {
                "input_per_1k": 0.03,
                "output_per_1k": 0.06,
            },
            "operation": "completion",
        }

        with patch.object(tools.estimator, "estimate_cost", return_value=cost_data):
            result = tools.estimate_cost(
                content="Test content",
                model="gpt-4",
                operation="completion",
                output_tokens=500,
            )

            assert result["total_cost_usd"] == 0.06
            assert result["input_tokens"] == 1000
            assert result["output_tokens"] == 500

    @pytest.mark.asyncio
    async def test_analyze_token_efficiency_basic(self, tools):
        """Test efficiency analysis integration."""
        analysis_data = {
            "current_tokens": 100,
            "potential_savings": {
                "whitespace": 10,
                "redundancy": 15,
                "compression": 5,
            },
            "total_potential_savings": 30,
            "potential_reduction_percentage": 30.0,
            "suggestions": [
                "Remove excessive whitespace (save ~10 tokens)",
                "Remove redundant content (save ~15 tokens)",
            ],
        }

        with patch.object(tools.optimizer, "analyze_efficiency", return_value=analysis_data):
            with patch.object(tools.estimator, "estimate_cost") as mock_cost:
                mock_cost.return_value = {
                    "total_cost_usd": 0.05,
                }

                result = tools.analyze_token_efficiency(
                    content="Test content with analysis",
                    model="gpt-4",
                )

                assert result["current_tokens"] == 100
                assert result["total_potential_savings"] == 30
                assert len(result["suggestions"]) == 2
                assert "cost_analysis" in result

    @pytest.mark.asyncio
    async def test_observability_metrics_recorded(self, tools):
        """Test that observability metrics are recorded."""
        with patch.object(tools.counter, "count_tokens", return_value=20):
            tools.count_tokens(content="Test", model="gpt-4")

            # Verify observability calls
            tools.obs.increment.assert_called()
            calls = [call[0][0] for call in tools.obs.increment.call_args_list]
            assert "tools.count_tokens" in calls
            assert "token_counting.success" in calls

    @pytest.mark.asyncio
    async def test_error_handling_logs_failures(self, tools):
        """Test that errors are logged to observability."""
        with patch.object(tools.counter, "count_tokens", side_effect=Exception("Test error")):
            result = tools.count_tokens(content="Test", model="gpt-4")

            assert "error" in result
            tools.obs.increment.assert_any_call("token_counting.errors")


class TestFeatureFlagIntegration:
    """Tests for feature flag integration."""

    @pytest.mark.asyncio
    async def test_feature_enabled(self):
        """Test tools work when feature is enabled."""
        config = TokenOptimizationConfig(enabled=True)

        with patch("src.token_optimization.tools.create_cache"):
            with patch("src.token_optimization.tools.get_observability"):
                tools = TokenOptimizationTools(config=config)

                with patch.object(tools.counter, "count_tokens", return_value=10):
                    result = tools.count_tokens(content="Test", model="gpt-4")

                    assert "token_count" in result
                    assert "error" not in result

    @pytest.mark.asyncio
    async def test_feature_disabled(self):
        """Test tools return error when feature is disabled."""
        config = TokenOptimizationConfig(enabled=False)

        with patch("src.token_optimization.tools.create_cache"):
            with patch("src.token_optimization.tools.get_observability"):
                tools = TokenOptimizationTools(config=config)

                result = tools.optimize_tokens(content="Test", model="gpt-4")

                assert "error" in result
                assert result["enabled"] is False


class TestConfigurationManagement:
    """Tests for configuration management."""

    @pytest.mark.asyncio
    async def test_default_configuration(self):
        """Test tools work with default configuration."""
        with patch("src.token_optimization.tools.create_cache"):
            with patch("src.token_optimization.tools.get_observability"):
                tools = TokenOptimizationTools()

                assert tools.config.enabled is True
                assert tools.config.default_reduction_target == 0.20
                assert tools.config.quality_threshold == 0.90

    @pytest.mark.asyncio
    async def test_custom_configuration(self):
        """Test tools respect custom configuration."""
        config = TokenOptimizationConfig(
            enabled=True,
            default_reduction_target=0.30,
            quality_threshold=0.85,
            cache_optimizations=False,
        )

        with patch("src.token_optimization.tools.create_cache"):
            with patch("src.token_optimization.tools.get_observability"):
                tools = TokenOptimizationTools(config=config)

                assert tools.config.default_reduction_target == 0.30
                assert tools.config.quality_threshold == 0.85
                assert tools.config.cache_optimizations is False


class TestPerformanceIntegration:
    """Performance-related integration tests."""

    @pytest.mark.asyncio
    async def test_optimization_performance(self, tools):
        """Test optimization completes within time limits."""
        import time

        with patch.object(tools.counter, "count_tokens", side_effect=[100, 90]):
            start = time.perf_counter()

            result = tools.optimize_tokens(
                content="Test content " * 100,
                model="gpt-4",
            )

            duration_ms = (time.perf_counter() - start) * 1000

            # Should complete quickly
            assert duration_ms < 1000  # Less than 1 second
            assert result["processing_time_ms"] < 500

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, tools):
        """Test handling concurrent optimization requests."""

        async def optimize_task(content: str, idx: int) -> dict[str, Any]:
            with patch.object(tools.counter, "count_tokens", side_effect=[50, 45]):
                return tools.optimize_tokens(
                    content=f"{content} {idx}",
                    model="gpt-4",
                )

        # Create multiple concurrent tasks
        tasks = [optimize_task("Test content", i) for i in range(10)]
        results = await asyncio.gather(*tasks)

        # All should succeed
        assert len(results) == 10
        assert all("success" in r for r in results)


class TestErrorRecovery:
    """Tests for error recovery and resilience."""

    @pytest.fixture
    async def tools(self):
        """Create TokenOptimizationTools instance."""
        config = TokenOptimizationConfig(enabled=True)

        with patch("src.token_optimization.tools.create_cache"):
            with patch("src.token_optimization.tools.get_observability"):
                yield TokenOptimizationTools(config=config)

    @pytest.mark.asyncio
    async def test_counter_failure_recovery(self, tools):
        """Test recovery from token counter failures."""
        with patch.object(tools.counter, "count_tokens", side_effect=Exception("Counter error")):
            result = tools.count_tokens(content="Test", model="gpt-4")

            assert "error" in result
            assert "failed" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_optimizer_failure_recovery(self, tools):
        """Test recovery from optimizer failures."""
        with patch.object(tools.optimizer, "optimize", side_effect=Exception("Optimizer error")):
            result = tools.optimize_tokens(content="Test", model="gpt-4")

            assert "error" in result
            assert "failed" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_estimator_failure_recovery(self, tools):
        """Test recovery from cost estimator failures."""
        with patch.object(tools.estimator, "estimate_cost", side_effect=Exception("Estimator error")):
            result = tools.estimate_cost(content="Test", model="gpt-4")

            assert "error" in result
            assert "failed" in result["error"].lower()


class TestCacheIntegration:
    """Tests for cache system integration."""

    @pytest.fixture
    async def tools_with_cache(self):
        """Create TokenOptimizationTools with working cache."""
        config = TokenOptimizationConfig(
            enabled=True,
            cache_optimizations=True,
            cache_ttl_seconds=3600,
        )

        mock_cache = AsyncMock()
        mock_cache.get.return_value = None
        mock_cache.set.return_value = True

        with patch("src.token_optimization.tools.create_cache", return_value=mock_cache):
            with patch("src.token_optimization.tools.get_observability"):
                tools = TokenOptimizationTools(config=config)
                tools.cache = mock_cache
                yield tools

    @pytest.mark.asyncio
    async def test_cache_stores_results(self, tools_with_cache):
        """Test that optimization results are cached."""
        with patch.object(tools_with_cache.counter, "count_tokens", side_effect=[20, 18]):
            tools_with_cache.optimize_tokens(
                content="Test content",
                model="gpt-4",
            )

            # Verify cache set was called
            assert tools_with_cache.cache.set.called

    @pytest.mark.asyncio
    async def test_cache_disabled_skips_caching(self):
        """Test that caching is skipped when disabled."""
        config = TokenOptimizationConfig(
            enabled=True,
            cache_optimizations=False,
        )

        mock_cache = AsyncMock()

        with patch("src.token_optimization.tools.create_cache", return_value=mock_cache):
            with patch("src.token_optimization.tools.get_observability"):
                tools = TokenOptimizationTools(config=config)
                tools.cache = mock_cache

                with patch.object(tools.counter, "count_tokens", side_effect=[20, 18]):
                    tools.optimize_tokens(content="Test", model="gpt-4")

                    # Cache should not be called
                    assert not mock_cache.set.called


class TestRealWorldScenarios:
    """Real-world usage scenario tests."""

    @pytest.fixture
    async def tools(self):
        """Create TokenOptimizationTools instance."""
        config = TokenOptimizationConfig(enabled=True)

        with patch("src.token_optimization.tools.create_cache"):
            with patch("src.token_optimization.tools.get_observability"):
                yield TokenOptimizationTools(config=config)

    @pytest.mark.asyncio
    async def test_api_request_optimization_workflow(self, tools):
        """Test typical API request optimization workflow."""
        # 1. Count tokens in prompt
        with patch.object(tools.counter, "count_tokens", return_value=1500):
            token_result = tools.count_tokens(
                content="Long API prompt...",
                model="gpt-4",
            )

            assert token_result["token_count"] == 1500

        # 2. Estimate cost
        with patch.object(tools.estimator, "estimate_cost") as mock_cost:
            mock_cost.return_value = {"total_cost_usd": 0.15}

            cost_result = tools.estimate_cost(
                content="Long API prompt...",
                model="gpt-4",
                output_tokens=500,
            )

            assert cost_result["total_cost_usd"] == 0.15

        # 3. Optimize to reduce cost
        with patch.object(tools.counter, "count_tokens", side_effect=[1500, 1200]):
            opt_result = tools.optimize_tokens(
                content="Long API prompt...",
                model="gpt-4",
                target_reduction=0.20,
            )

            assert opt_result["tokens_saved"] == 300

    @pytest.mark.asyncio
    async def test_cost_comparison_workflow(self, tools):
        """Test cost comparison across models workflow."""
        # Analyze efficiency
        with patch.object(tools.optimizer, "analyze_efficiency") as mock_analyze:
            mock_analyze.return_value = {
                "current_tokens": 2000,
                "total_potential_savings": 400,
                "potential_reduction_percentage": 20.0,
                "suggestions": ["Optimize whitespace"],
                "potential_savings": {"whitespace": 400},
            }

            with patch.object(tools.estimator, "estimate_cost") as mock_cost:
                mock_cost.return_value = {"total_cost_usd": 0.20}

                analysis = tools.analyze_token_efficiency(
                    content="Content to analyze",
                    model="gpt-4",
                )

                assert analysis["current_tokens"] == 2000
                assert "cost_analysis" in analysis
