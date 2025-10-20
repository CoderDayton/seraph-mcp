"""
Tests for observability metrics integration in context optimization middleware.

Verifies that compression metrics are properly tracked to SQLite.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.context_optimization.middleware import OptimizedProvider
from src.context_optimization.models import OptimizationResult
from src.providers.base import CompletionResponse


def create_mock_response(content: str = "test response") -> CompletionResponse:
    """Helper to create properly structured CompletionResponse objects."""
    return CompletionResponse(
        content=content,
        model="gpt-4",
        usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        finish_reason="stop",
        provider="test-provider",
        latency_ms=100.0,
        cost_usd=0.001,
    )


class TestMetricsIntegration:
    """Test observability metrics tracking in middleware."""

    @pytest.fixture
    def mock_provider(self):
        """Mock AI provider."""
        provider = MagicMock()
        provider.complete = AsyncMock(return_value=create_mock_response())
        return provider

    @pytest.fixture
    def mock_optimizer(self):
        """Mock context optimizer with realistic result."""
        optimizer = MagicMock()
        result = OptimizationResult(
            original_content="Long prompt here" * 100,
            tokens_before=1000,
            tokens_after=600,
            tokens_saved=400,
            reduction_percentage=40.0,
            compression_ratio=1.67,  # tokens_before/tokens_after = 1000/600
            quality_score=0.92,
            processing_time_ms=45.3,
            method="seraph",
            validation_passed=True,
            optimized_content="Optimized content here",
            metadata={"cost_savings_usd": 0.008, "rollback_occurred": False},
        )
        optimizer.optimize = AsyncMock(return_value=result)
        return optimizer

    @pytest.fixture
    def mock_observability(self):
        """Mock observability adapter."""
        obs = MagicMock()
        obs.increment = MagicMock()
        obs.gauge = MagicMock()
        obs.histogram = MagicMock()
        return obs

    @pytest.mark.asyncio
    async def test_successful_optimization_tracks_compression_metrics(
        self, mock_provider, mock_optimizer, mock_observability
    ):
        """Test that successful optimization emits all compression metrics."""
        with patch("src.context_optimization.middleware.get_observability", return_value=mock_observability):
            middleware = OptimizedProvider(
                provider=mock_provider,
                optimizer=mock_optimizer,
            )

            # Generate with optimization (messages only)
            messages = [{"role": "user", "content": "Long prompt here" * 100}]
            await middleware.generate(messages=messages)

            # Verify compression metrics were tracked
            mock_observability.histogram.assert_any_call(
                "optimization.compression_ratio",
                value=1.67,
                tags={"method": "seraph"},
            )
            mock_observability.histogram.assert_any_call(
                "optimization.processing_time_ms",
                value=45.3,
                tags={"method": "seraph"},
            )
            mock_observability.gauge.assert_called_with(
                "optimization.quality_score",
                value=0.92,
                tags={"method": "seraph"},
            )
            mock_observability.increment.assert_any_call(
                "optimization.tokens_saved",
                value=400.0,
                tags={"method": "seraph"},
            )
            mock_observability.increment.assert_any_call(
                "optimization.method_selected",
                value=1.0,
                tags={"method": "seraph"},
            )

    # Note: Injection detection test removed - feature requires SecurityConfig.injection_detection_enabled=True
    # and is tested separately in security module tests (tests/unit/security/)

    @pytest.mark.asyncio
    async def test_validation_failure_tracks_security_metric(self, mock_provider, mock_optimizer, mock_observability):
        """Test that validation failure emits security metrics."""
        with patch("src.context_optimization.middleware.get_observability", return_value=mock_observability):
            # Mock content validator
            with patch("src.context_optimization.middleware.ContentValidator") as mock_validator_class:
                mock_validator = MagicMock()
                validation_result = MagicMock()
                validation_result.passed = False
                validation_result.reasons = ["quality_too_low", "excessive_truncation"]
                mock_validator.validate_compressed_content.return_value = validation_result
                mock_validator_class.return_value = mock_validator

                middleware = OptimizedProvider(
                    provider=mock_provider,
                    optimizer=mock_optimizer,
                )

                # Generate with validation failure (messages only)
                messages = [{"role": "user", "content": "Long prompt here" * 100}]
                await middleware.generate(messages=messages)

                # Verify validation failure metric was tracked
                mock_observability.increment.assert_any_call(
                    "optimization.validation_failed",
                    value=1.0,
                    tags={"reasons": "quality_too_low,excessive_truncation"},
                )

    @pytest.mark.asyncio
    async def test_rollback_tracks_rollback_metric(self, mock_provider, mock_observability):
        """Test that rollback events emit rollback metrics."""
        # Create optimizer with rollback result
        optimizer = MagicMock()
        result = OptimizationResult(
            original_content="Long prompt here" * 100,
            tokens_before=1000,
            tokens_after=600,
            tokens_saved=400,
            reduction_percentage=40.0,
            compression_ratio=1.67,
            quality_score=0.92,
            processing_time_ms=45.3,
            method="ai",
            validation_passed=True,
            optimized_content="Optimized content here",
            metadata={"cost_savings_usd": 0.008, "rollback_occurred": True},  # Rollback occurred
        )
        optimizer.optimize = AsyncMock(return_value=result)

        with patch("src.context_optimization.middleware.get_observability", return_value=mock_observability):
            middleware = OptimizedProvider(
                provider=mock_provider,
                optimizer=optimizer,
            )

            # Generate with rollback (messages only)
            messages = [{"role": "user", "content": "Long prompt here" * 100}]
            await middleware.generate(messages=messages)

            # Verify rollback metric was tracked
            mock_observability.increment.assert_any_call(
                "optimization.rollback_occurred",
                value=1.0,
                tags={"method": "ai"},
            )

    @pytest.mark.asyncio
    async def test_messages_optimization_tracks_same_metrics(self, mock_provider, mock_optimizer, mock_observability):
        """Test that message optimization emits compression metrics."""
        with patch("src.context_optimization.middleware.get_observability", return_value=mock_observability):
            middleware = OptimizedProvider(
                provider=mock_provider,
                optimizer=mock_optimizer,
            )

            # Generate with messages
            messages = [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Long user message here" * 100},
            ]
            await middleware.generate(messages=messages)

            # Verify same compression metrics were tracked
            mock_observability.histogram.assert_any_call(
                "optimization.compression_ratio",
                value=1.67,
                tags={"method": "seraph"},
            )
            mock_observability.gauge.assert_called_with(
                "optimization.quality_score",
                value=0.92,
                tags={"method": "seraph"},
            )

    @pytest.mark.asyncio
    async def test_different_methods_have_distinct_tags(self, mock_provider, mock_observability):
        """Test that different optimization methods are tagged correctly."""
        methods = ["ai", "seraph", "hybrid"]
        for method in methods:
            optimizer = MagicMock()
            result = OptimizationResult(
                original_content="Long prompt here" * 100,
                tokens_before=1000,
                tokens_after=600,
                tokens_saved=400,
                reduction_percentage=40.0,
                compression_ratio=1.67,
                quality_score=0.92,
                processing_time_ms=45.3,
                method=method,
                validation_passed=True,
                optimized_content="Optimized content here",
                metadata={"cost_savings_usd": 0.008, "rollback_occurred": False},
            )
            optimizer.optimize = AsyncMock(return_value=result)

            with patch("src.context_optimization.middleware.get_observability", return_value=mock_observability):
                middleware = OptimizedProvider(
                    provider=mock_provider,
                    optimizer=optimizer,
                )

                # Generate with messages
                messages = [{"role": "user", "content": "Long prompt here" * 100}]
                await middleware.generate(messages=messages)

                # Verify method tag is correct
                mock_observability.increment.assert_any_call(
                    "optimization.method_selected",
                    value=1.0,
                    tags={"method": method},
                )

                # Reset mock for next iteration
                mock_observability.reset_mock()
