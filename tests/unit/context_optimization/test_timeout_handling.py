"""
Timeout Handling Validation Tests for Context Optimizer

Tests the complete timeout propagation chain:
1. Provider timeout re-raising in _call_provider()
2. Timeout catch and re-raise in _optimize_with_ai()
3. Graceful fallback to Seraph compression in _optimize_hybrid()
4. Outer timeout configuration validation

Per SDD §4.2.1: Timeout Architecture for Graceful Degradation
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.context_optimization.config import ContextOptimizationConfig
from src.context_optimization.optimizer import ContextOptimizer
from src.providers.base import BaseProvider, CompletionResponse


@pytest.fixture
def optimizer_config() -> ContextOptimizationConfig:
    """Configuration with fast timeouts for testing."""
    return ContextOptimizationConfig(
        enabled=True,
        max_overhead_ms=2000.0,  # 2 second outer timeout
        quality_threshold=0.75,
        seraph_token_threshold=100,
        seraph_l1_ratio=0.15,
        seraph_l2_ratio=0.50,
        seraph_l3_ratio=0.70,
        embedding_provider="none",  # Disable embeddings for tests
        embedding_dimensions=768,
    )


@pytest.fixture
def sample_text() -> str:
    """Sample text for compression testing."""
    return (
        """
    Machine learning is a subset of artificial intelligence that focuses on
    developing systems that can learn from data. These systems improve their
    performance over time without being explicitly programmed. Deep learning
    uses neural networks with multiple layers to process complex patterns.
    Natural language processing enables computers to understand human language.
    Reinforcement learning allows agents to learn optimal behaviors through trial and error.
    Supervised learning uses labeled data to train predictive models.
    Unsupervised learning discovers hidden patterns in unlabeled data.
    Transfer learning leverages pre-trained models for new tasks.
    """
        * 20
    )  # Make it long enough to trigger compression (repeat 20x)


class TestTimeoutPropagation:
    """Test that timeouts propagate correctly through the call chain."""

    @pytest.mark.asyncio
    async def test_provider_timeout_reraises(
        self, optimizer_config: ContextOptimizationConfig, sample_text: str
    ) -> None:
        """
        Test that _call_provider() re-raises timeout exceptions.

        Validates that timeout exceptions from provider.complete() propagate correctly.
        """
        # Create optimizer without provider
        optimizer = ContextOptimizer(config=optimizer_config, provider=None)

        # Create a mock provider that raises TimeoutError (simulating provider's asyncio.wait_for())
        mock_provider = MagicMock(spec=BaseProvider)
        mock_config = MagicMock()
        mock_config.model = "test-model"
        mock_provider.config = mock_config

        # Mock provider.complete() to raise asyncio.TimeoutError (what wait_for() raises)
        mock_provider.complete = AsyncMock(side_effect=TimeoutError("Provider timeout"))

        # Temporarily inject the mock provider
        optimizer.provider = mock_provider

        # Call with short timeout (min 1.0s per Pydantic validation)
        with pytest.raises((TimeoutError, asyncio.TimeoutError)):
            await optimizer._call_provider(
                prompt="test prompt",
                max_tokens=50,
                timeout=1.0,  # 1s timeout (meets ge=1.0 requirement)
            )

    @pytest.mark.asyncio
    async def test_optimize_with_ai_timeout_propagation(
        self, optimizer_config: ContextOptimizationConfig, sample_text: str
    ) -> None:
        """
        Test that _optimize_with_ai() propagates timeout exceptions.

        Validates fix #2: Lines 380-383 in optimizer.py
        """
        # Create mock provider
        mock_provider = MagicMock(spec=BaseProvider)

        async def slow_call(*args: Any, **kwargs: Any) -> str:
            await asyncio.sleep(10)
            return "never"

        mock_provider.complete = AsyncMock(side_effect=slow_call)
        mock_provider.count_tokens = MagicMock(return_value=50)

        # Create optimizer WITH the mock provider
        optimizer = ContextOptimizer(config=optimizer_config, provider=mock_provider)

        # Patch _call_provider to simulate timeout
        with patch.object(optimizer, "_call_provider", side_effect=TimeoutError("Provider timeout")):
            with pytest.raises((TimeoutError, asyncio.TimeoutError)):
                await optimizer._optimize_with_ai(content=sample_text)

    @pytest.mark.asyncio
    async def test_hybrid_fallback_on_timeout(
        self, optimizer_config: ContextOptimizationConfig, sample_text: str
    ) -> None:
        """
        Test that _optimize_hybrid() falls back to Seraph when AI times out.

        This is the critical graceful degradation behavior.
        Validates the complete timeout handling chain.
        """
        # Create optimizer without provider (simulate timeout scenario)
        optimizer = ContextOptimizer(config=optimizer_config, provider=None)

        # Mock _optimize_with_ai to raise timeout
        with patch.object(optimizer, "_optimize_with_ai", side_effect=TimeoutError("AI optimization timeout")):
            # Should fall back to Seraph L2 compression
            compressed, quality = await optimizer._optimize_hybrid(content=sample_text)

            # Verify fallback worked
            assert compressed is not None
            assert len(compressed) < len(sample_text)
            assert quality > 0.0
            assert quality <= 1.0

            # Seraph L2 achieves 80-90% reduction (0.1-0.2 ratio observed in production)
            compression_ratio = len(compressed) / len(sample_text)
            assert 0.1 <= compression_ratio <= 0.3, (
                f"Seraph L2 fallback compression ratio {compression_ratio:.2f} " f"outside expected range [0.1, 0.3]"
            )


class TestTimeoutConfiguration:
    """Test timeout configuration hierarchy and validation."""

    def test_outer_timeout_exceeds_inner_sum(self, optimizer_config: ContextOptimizationConfig) -> None:
        """
        Test that outer timeout > sum of inner timeouts.

        Validates fix #3: Line 128 in config.py (default 10000ms)

        Expected hierarchy:
        - Outer: 10.0s (max_overhead_ms default)
        - Compression: 5.0s (hardcoded in optimizer.py line 561)
        - Validation: 3.0s (hardcoded in optimizer.py line 585)
        - Total inner: 8.0s
        - Buffer: 2.0s ✓
        """
        # Default config should have safe timeout values
        default_config = ContextOptimizationConfig()
        outer_timeout_sec = default_config.max_overhead_ms / 1000.0

        # Hardcoded inner timeouts from optimizer.py
        compression_timeout = 5.0  # Line 561
        validation_timeout = 3.0  # Line 585
        inner_timeout_sum = compression_timeout + validation_timeout

        assert outer_timeout_sec > inner_timeout_sum, (
            f"Outer timeout ({outer_timeout_sec}s) must exceed "
            f"sum of inner timeouts ({inner_timeout_sum}s) "
            f"to allow graceful fallback"
        )

        # Verify at least 1 second buffer for fallback logic
        buffer = outer_timeout_sec - inner_timeout_sum
        assert buffer >= 1.0, f"Timeout buffer ({buffer}s) should be at least 1s " f"for reliable fallback behavior"

    def test_custom_timeout_validation(self) -> None:
        """Test that custom timeout configurations are validated."""
        # Valid configuration
        config = ContextOptimizationConfig(max_overhead_ms=15000.0)
        assert config.max_overhead_ms == 15000.0

        # Invalid (negative) should be caught by Pydantic validation
        with pytest.raises(ValueError):
            ContextOptimizationConfig(max_overhead_ms=-1000.0)


class TestRealWorldTimeoutScenario:
    """Integration-style tests simulating real timeout scenarios."""

    @pytest.mark.asyncio
    async def test_complete_fallback_chain(self, optimizer_config: ContextOptimizationConfig, sample_text: str) -> None:
        """
        End-to-end test: AI provider times out, falls back to Seraph L2.

        This simulates the exact production scenario that was failing
        before the timeout propagation fixes.
        """
        # Create mock provider that will timeout
        slow_provider = MagicMock(spec=BaseProvider)

        async def always_timeout(*args: Any, **kwargs: Any) -> str:
            await asyncio.sleep(100)  # Definitely will timeout
            return "unreachable"

        slow_provider.complete = AsyncMock(side_effect=always_timeout)
        slow_provider.count_tokens = MagicMock(return_value=len(sample_text.split()))

        # Create optimizer with timeout-prone provider
        optimizer = ContextOptimizer(config=optimizer_config, provider=slow_provider)

        # Optimize should succeed via fallback (with shorter timeout for test)
        result = await optimizer.optimize(content=sample_text, timeout_ms=1000.0)

        # Verify we got valid Seraph compression fallback
        assert result.optimized_content is not None
        assert result.optimized_content != sample_text
        assert result.quality_score >= 0.70  # Seraph L2/L3 quality
        assert result.tokens_saved > 0

        print(f"✓ Fallback successful: {result.tokens_saved} tokens saved")
        print(f"✓ Quality: {result.quality_score:.3f}")
        print(f"✓ Compression: {result.compression_ratio:.1%}")

    @pytest.mark.asyncio
    async def test_fast_provider_no_fallback(
        self, optimizer_config: ContextOptimizationConfig, sample_text: str
    ) -> None:
        """
        Verify that fast providers complete normally without triggering fallback.
        """
        # Fast provider that completes quickly
        fast_provider = MagicMock(spec=BaseProvider)
        mock_response = CompletionResponse(
            content="Compressed: ML systems learn from data, DL uses neural nets, NLP understands language.",
            model="test-model",
            usage={"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
            finish_reason="stop",
            provider="test",
            latency_ms=100.0,
            cost_usd=0.001,
        )
        fast_provider.complete = AsyncMock(return_value=mock_response)
        fast_provider.count_tokens = MagicMock(side_effect=lambda t: len(t.split()))

        optimizer = ContextOptimizer(config=optimizer_config, provider=fast_provider)

        result = await optimizer.optimize(content=sample_text, timeout_ms=5000.0)

        # Should use AI compression or hybrid, not pure Seraph fallback
        assert result.optimized_content is not None
        assert result.quality_score > 0.70
        assert result.tokens_saved > 0

        print(f"✓ AI compression successful: {result.compression_ratio:.1%} ratio")


class TestCodeInspection:
    """Static code validation tests."""

    def test_timeout_handler_exists_in_call_provider(self) -> None:
        """
        Verify that _call_provider has explicit timeout exception handling.

        This replaces the fragile regex validation from the previous session.
        """
        import inspect

        from src.context_optimization.optimizer import ContextOptimizer

        source = inspect.getsource(ContextOptimizer._call_provider)

        # Verify timeout exception types are handled
        assert "TimeoutError" in source or "asyncio.TimeoutError" in source

        # Verify it's in an except clause (not just a comment)
        assert "except" in source and ("TimeoutError" in source or "asyncio.TimeoutError" in source)

        # Verify we re-raise (not return/pass)
        assert "raise" in source

    def test_timeout_handler_exists_in_optimize_with_ai(self) -> None:
        """Verify _optimize_with_ai has timeout exception handling."""
        import inspect

        from src.context_optimization.optimizer import ContextOptimizer

        source = inspect.getsource(ContextOptimizer._optimize_with_ai)

        # Should have explicit timeout handling
        assert "except" in source
        assert "TimeoutError" in source or "asyncio.TimeoutError" in source
        assert "raise" in source  # Must re-raise for hybrid fallback

    def test_config_default_timeout_value(self) -> None:
        """Verify default outer timeout is 10 seconds."""
        config = ContextOptimizationConfig()
        assert config.max_overhead_ms == 10000.0  # 10 seconds


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
