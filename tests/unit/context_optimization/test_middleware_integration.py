"""
Tests for OptimizedProvider middleware integration with server initialization.

Verifies that providers are properly wrapped with optimization middleware
at server startup and that automatic optimization occurs transparently.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.context_optimization.config import ContextOptimizationConfig
from src.context_optimization.middleware import OptimizedProvider, wrap_provider


class TestMiddlewareWrapperFunction:
    """Test the wrap_provider convenience function."""

    def test_wrap_provider_creates_optimized_provider(self):
        """Verify wrap_provider returns OptimizedProvider instance."""
        mock_provider = MagicMock()
        config = ContextOptimizationConfig(enabled=True)

        wrapped = wrap_provider(provider=mock_provider, config=config)

        assert isinstance(wrapped, OptimizedProvider)
        assert wrapped.provider is mock_provider
        assert wrapped.config == config

    def test_wrap_provider_with_budget_tracker(self):
        """Verify budget tracker is passed through correctly."""
        mock_provider = MagicMock()
        mock_tracker = MagicMock()
        config = ContextOptimizationConfig(enabled=True)

        wrapped = wrap_provider(provider=mock_provider, config=config, budget_tracker=mock_tracker)

        assert wrapped.budget_tracker is mock_tracker

    def test_wrap_provider_loads_config_if_none(self):
        """Verify config is loaded from environment if not provided."""
        mock_provider = MagicMock()

        wrapped = wrap_provider(provider=mock_provider)

        assert wrapped.config is not None
        assert isinstance(wrapped.config, ContextOptimizationConfig)


class TestOptimizedProviderInterface:
    """Test that OptimizedProvider preserves base provider interface."""

    def test_proxies_unknown_methods_to_base_provider(self):
        """Verify unknown methods are proxied to base provider via __getattr__."""
        mock_provider = MagicMock()
        mock_provider.custom_method = MagicMock(return_value="custom_result")

        wrapped = wrap_provider(provider=mock_provider)

        result = wrapped.custom_method()
        assert result == "custom_result"
        mock_provider.custom_method.assert_called_once()

    def test_preserves_provider_attributes(self):
        """Verify base provider attributes are accessible."""
        mock_provider = MagicMock()
        mock_provider.api_key = "test-key"
        mock_provider.model = "gpt-4"

        wrapped = wrap_provider(provider=mock_provider)

        assert wrapped.api_key == "test-key"
        assert wrapped.model == "gpt-4"


class TestAutomaticOptimization:
    """Test that middleware automatically optimizes prompts."""

    @pytest.mark.asyncio
    async def test_generate_with_prompt_triggers_optimization(self):
        """Verify generate() with prompt triggers automatic optimization."""
        # Setup mock provider
        mock_provider = MagicMock()
        mock_provider.generate = AsyncMock(return_value={"text": "response"})

        # Setup mock optimizer
        mock_optimizer = MagicMock()
        mock_result = MagicMock()
        mock_result.optimized_content = "optimized prompt"
        mock_result.tokens_saved = 100
        mock_result.reduction_percentage = 30.0
        mock_result.quality_score = 0.95
        mock_result.validation_passed = True
        mock_result.processing_time_ms = 50
        mock_result.metadata = {"cost_savings_usd": 0.002, "rollback_occurred": False}
        mock_optimizer.optimize = AsyncMock(return_value=mock_result)

        config = ContextOptimizationConfig(enabled=True)
        wrapped = OptimizedProvider(provider=mock_provider, optimizer=mock_optimizer, config=config)

        # Call generate
        response = await wrapped.generate(prompt="original long prompt")

        # Verify optimization was called
        mock_optimizer.optimize.assert_called_once_with("original long prompt")

        # Verify provider received optimized prompt
        mock_provider.generate.assert_called_once()
        call_kwargs = mock_provider.generate.call_args[1]
        assert call_kwargs["prompt"] == "optimized prompt"

        # Verify response includes optimization metadata
        assert "optimization" in response
        assert response["optimization"]["tokens_saved"] == 100
        assert response["optimization"]["reduction_percentage"] == 30.0

    @pytest.mark.asyncio
    async def test_generate_with_messages_triggers_optimization(self):
        """Verify generate() with messages optimizes last user message."""
        mock_provider = MagicMock()
        mock_provider.generate = AsyncMock(return_value={"text": "response"})

        mock_optimizer = MagicMock()
        mock_result = MagicMock()
        mock_result.optimized_content = "optimized message"
        mock_result.tokens_saved = 50
        mock_result.reduction_percentage = 20.0
        mock_result.quality_score = 0.92
        mock_result.validation_passed = True
        mock_result.processing_time_ms = 40
        mock_result.metadata = {"cost_savings_usd": 0.001, "rollback_occurred": False}
        mock_optimizer.optimize = AsyncMock(return_value=mock_result)

        config = ContextOptimizationConfig(enabled=True)
        wrapped = OptimizedProvider(provider=mock_provider, optimizer=mock_optimizer, config=config)

        # NOTE: Message must be >100 chars to trigger optimization (per middleware logic)
        long_message = "This is a very long user message that must exceed one hundred characters to trigger the optimization logic in the middleware implementation code."
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": long_message},
        ]

        await wrapped.generate(messages=messages)

        # Verify optimization was called on user message
        mock_optimizer.optimize.assert_called_once()

        # Verify provider received optimized messages
        mock_provider.generate.assert_called_once()
        call_kwargs = mock_provider.generate.call_args[1]
        assert call_kwargs["messages"][1]["content"] == "optimized message"

    @pytest.mark.asyncio
    async def test_skip_optimization_flag_bypasses_middleware(self):
        """Verify skip_optimization=True bypasses optimization."""
        mock_provider = MagicMock()
        mock_provider.generate = AsyncMock(return_value={"text": "response"})

        mock_optimizer = MagicMock()
        mock_optimizer.optimize = AsyncMock()

        config = ContextOptimizationConfig(enabled=True)
        wrapped = OptimizedProvider(provider=mock_provider, optimizer=mock_optimizer, config=config)

        await wrapped.generate(prompt="test prompt", skip_optimization=True)

        # Verify optimizer was NOT called
        mock_optimizer.optimize.assert_not_called()

        # Verify provider received original prompt
        mock_provider.generate.assert_called_once()
        call_kwargs = mock_provider.generate.call_args[1]
        assert call_kwargs["prompt"] == "test prompt"

    @pytest.mark.asyncio
    async def test_disabled_config_skips_optimization(self):
        """Verify optimization is skipped when config.enabled=False."""
        mock_provider = MagicMock()
        mock_provider.generate = AsyncMock(return_value={"text": "response"})

        mock_optimizer = MagicMock()
        mock_optimizer.optimize = AsyncMock()

        config = ContextOptimizationConfig(enabled=False)
        wrapped = OptimizedProvider(provider=mock_provider, optimizer=mock_optimizer, config=config)

        await wrapped.generate(prompt="test prompt")

        # Verify optimizer was NOT called
        mock_optimizer.optimize.assert_not_called()


class TestMiddlewareStats:
    """Test middleware statistics tracking."""

    @pytest.mark.asyncio
    async def test_tracks_total_calls(self):
        """Verify middleware tracks total call count."""
        mock_provider = MagicMock()
        mock_provider.generate = AsyncMock(return_value={"text": "response"})

        config = ContextOptimizationConfig(enabled=False)  # Disable to simplify
        wrapped = OptimizedProvider(provider=mock_provider, config=config)

        await wrapped.generate(prompt="test1")
        await wrapped.generate(prompt="test2")
        await wrapped.generate(prompt="test3")

        stats = wrapped.get_stats()
        assert stats["middleware"]["total_calls"] == 3

    @pytest.mark.asyncio
    async def test_tracks_tokens_saved(self):
        """Verify middleware tracks cumulative tokens saved."""
        mock_provider = MagicMock()
        mock_provider.generate = AsyncMock(return_value={"text": "response"})

        mock_optimizer = MagicMock()
        mock_result = MagicMock()
        mock_result.optimized_content = "tst"
        mock_result.tokens_saved = 100
        mock_result.quality_score = 0.85
        mock_result.validation_passed = True
        mock_result.metadata = {"rollback_occurred": False, "cost_savings_usd": 0.002}
        mock_optimizer.optimize = AsyncMock(return_value=mock_result)

        config = ContextOptimizationConfig(enabled=True)
        wrapped = OptimizedProvider(provider=mock_provider, optimizer=mock_optimizer, config=config)

        await wrapped.generate(prompt="test1")
        await wrapped.generate(prompt="test2")

        stats = wrapped.get_stats()
        assert stats["middleware"]["total_tokens_saved"] == 200  # 100 * 2

    @pytest.mark.asyncio
    async def test_tracks_cost_savings(self):
        """Verify middleware tracks cumulative cost savings."""
        mock_provider = MagicMock()
        mock_provider.generate = AsyncMock(return_value={"text": "response"})

        mock_optimizer = MagicMock()
        mock_result = MagicMock()
        mock_result.optimized_content = "tst"
        mock_result.tokens_saved = 100
        mock_result.quality_score = 0.85
        mock_result.validation_passed = True
        mock_result.metadata = {"rollback_occurred": False, "cost_savings_usd": 0.005}
        mock_optimizer.optimize = AsyncMock(return_value=mock_result)

        config = ContextOptimizationConfig(enabled=True)
        wrapped = OptimizedProvider(provider=mock_provider, optimizer=mock_optimizer, config=config)

        await wrapped.generate(prompt="test1")
        await wrapped.generate(prompt="test2")

        stats = wrapped.get_stats()
        assert stats["middleware"]["total_cost_saved"] == 0.010  # 0.005 * 2


class TestErrorHandling:
    """Test middleware error handling and fallback behavior."""

    @pytest.mark.asyncio
    async def test_optimization_failure_uses_original_prompt(self):
        """Verify optimization errors fall back to original prompt."""
        mock_provider = MagicMock()
        mock_provider.generate = AsyncMock(return_value={"text": "response"})

        mock_optimizer = MagicMock()
        mock_optimizer.optimize = AsyncMock(side_effect=Exception("Optimization failed"))

        config = ContextOptimizationConfig(enabled=True)
        wrapped = OptimizedProvider(provider=mock_provider, optimizer=mock_optimizer, config=config)

        response = await wrapped.generate(prompt="test prompt")

        # Verify provider received ORIGINAL prompt (fallback)
        mock_provider.generate.assert_called_once()
        call_kwargs = mock_provider.generate.call_args[1]
        assert call_kwargs["prompt"] == "test prompt"

        # Verify response has no optimization metadata
        assert "optimization" not in response

    @pytest.mark.asyncio
    async def test_provider_error_propagates(self):
        """Verify provider errors propagate correctly."""
        mock_provider = MagicMock()
        mock_provider.generate = AsyncMock(side_effect=ValueError("Provider error"))

        config = ContextOptimizationConfig(enabled=False)
        wrapped = OptimizedProvider(provider=mock_provider, config=config)

        with pytest.raises(ValueError, match="Provider error"):
            await wrapped.generate(prompt="test prompt")


class TestServerIntegration:
    """Test server initialization properly wires middleware."""

    @patch("src.context_optimization.config.load_config")
    @patch("src.providers.factory.create_provider")
    def test_server_initialization_wraps_provider(self, mock_create_provider, mock_load_config):
        """Verify server startup wraps provider with middleware.

        This is an integration test that verifies the critical fix:
        - Provider is created via factory
        - Provider is wrapped with OptimizedProvider middleware
        - Wrapped provider is stored in _context_optimizer dict
        """
        from src.server import _init_context_optimization_if_available

        # Setup mocks
        mock_base_provider = MagicMock()
        mock_create_provider.return_value = mock_base_provider

        mock_context_config = ContextOptimizationConfig(enabled=True)
        mock_load_config.return_value = mock_context_config

        # Create minimal config with mock providers
        mock_config = MagicMock()
        mock_providers_cfg = MagicMock()
        mock_openai_cfg = MagicMock()
        mock_openai_cfg.enabled = True
        mock_openai_cfg.api_key = "test-key"
        mock_openai_cfg.model = "gpt-4"
        mock_openai_cfg.base_url = None
        mock_openai_cfg.timeout = 30
        mock_openai_cfg.max_retries = 3

        mock_providers_cfg.openai = mock_openai_cfg
        mock_providers_cfg.anthropic = MagicMock(enabled=False)
        mock_providers_cfg.gemini = MagicMock(enabled=False)
        mock_providers_cfg.openai_compatible = MagicMock(enabled=False)

        mock_config.providers = mock_providers_cfg
        mock_config.context_optimization = MagicMock(enabled=True)

        # Initialize (this should wrap the provider)
        with patch("src.server._context_optimizer", None):
            with patch("src.server._budget_tracker", None):
                _init_context_optimization_if_available(mock_config)

                # Verify provider factory was called
                mock_create_provider.assert_called_once()

                # Import and check global state
                from src.server import _context_optimizer

                assert _context_optimizer is not None
                assert "provider" in _context_optimizer

                # Critical assertion: Provider is wrapped with OptimizedProvider
                wrapped_provider = _context_optimizer["provider"]
                assert isinstance(wrapped_provider, OptimizedProvider)
                assert wrapped_provider.provider is mock_base_provider
