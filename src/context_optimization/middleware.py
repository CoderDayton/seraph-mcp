"""
Context Optimization Middleware

Automatic middleware that wraps AI providers to transparently optimize
prompts before sending to LLMs.

Usage:
    from src.context_optimization.middleware import OptimizedProvider

    base_provider = OpenAIProvider(...)
    optimizer = ContextOptimizer(...)

    # Wrap provider with optimization
    optimized_provider = OptimizedProvider(
        provider=base_provider,
        optimizer=optimizer
    )

    # Use as normal - optimization happens automatically
    response = await optimized_provider.generate(prompt="Long prompt...")
"""

import logging
from typing import Any

from .config import ContextOptimizationConfig, load_config
from .optimizer import ContextOptimizer

logger = logging.getLogger(__name__)


class OptimizedProvider:
    """
    Middleware that wraps an AI provider to automatically optimize prompts.

    This class intercepts all LLM calls and applies context optimization
    transparently, tracking cost savings and metrics.
    """

    def __init__(
        self,
        provider: Any,
        optimizer: ContextOptimizer | None = None,
        config: ContextOptimizationConfig | None = None,
        budget_tracker: Any | None = None,
    ):
        """
        Initialize optimized provider wrapper.

        Args:
            provider: Base AI provider to wrap
            optimizer: Context optimizer instance (creates default if None)
            config: Optimization config (loads from env if None)
            budget_tracker: Optional budget tracker for cost savings
        """
        self.provider = provider
        self.budget_tracker = budget_tracker

        # Load config if not provided
        if config is None:
            config = load_config()
        self.config = config

        # Create optimizer if not provided
        if optimizer is None:
            optimizer = ContextOptimizer(config=config, provider=provider, budget_tracker=budget_tracker)
        self.optimizer = optimizer

        # Track middleware stats
        self.middleware_stats = {
            "total_calls": 0,
            "optimized_calls": 0,
            "total_tokens_saved": 0,
            "total_cost_saved": 0.0,
        }

    async def generate(
        self,
        prompt: str | None = None,
        messages: list[dict[str, str]] | None = None,
        skip_optimization: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Generate completion with automatic prompt optimization.

        Args:
            prompt: Single prompt string (for completion-style APIs)
            messages: Chat messages (for chat-style APIs)
            skip_optimization: Set to True to bypass optimization
            **kwargs: Additional arguments passed to base provider

        Returns:
            Provider response with optimization metadata added
        """
        self.middleware_stats["total_calls"] += 1

        # Skip optimization if disabled or explicitly requested
        if not self.config.enabled or skip_optimization:
            return await self._call_provider(prompt=prompt, messages=messages, **kwargs)

        # Optimize prompt or messages
        if prompt:
            optimized_prompt, opt_result = await self._optimize_prompt(prompt)
            return await self._call_provider(
                prompt=optimized_prompt,
                messages=None,
                optimization_result=opt_result,
                **kwargs,
            )
        elif messages:
            optimized_messages, opt_result = await self._optimize_messages(messages)
            return await self._call_provider(
                prompt=None,
                messages=optimized_messages,
                optimization_result=opt_result,
                **kwargs,
            )
        else:
            # No prompt or messages - pass through
            return await self._call_provider(prompt=prompt, messages=messages, **kwargs)

    async def chat(
        self, messages: list[dict[str, str]], skip_optimization: bool = False, **kwargs: Any
    ) -> dict[str, Any]:
        """
        Chat completion with automatic message optimization.

        Args:
            messages: Chat messages
            skip_optimization: Set to True to bypass optimization
            **kwargs: Additional arguments passed to base provider

        Returns:
            Provider response with optimization metadata
        """
        return await self.generate(messages=messages, skip_optimization=skip_optimization, **kwargs)

    async def _optimize_prompt(self, prompt: str) -> tuple[str, Any | None]:
        """
        Optimize a single prompt string.

        Returns:
            Tuple of (optimized_prompt, optimization_result)
        """
        try:
            result = await self.optimizer.optimize(prompt)

            # Update middleware stats
            rollback_occurred = result.metadata.get("rollback_occurred", False)
            if result.validation_passed and not rollback_occurred:
                self.middleware_stats["optimized_calls"] += 1
                self.middleware_stats["total_tokens_saved"] += result.tokens_saved
                cost_savings = result.metadata.get("cost_savings_usd", 0.0)
                self.middleware_stats["total_cost_saved"] += cost_savings

                logger.info(
                    f"Optimized prompt: {result.tokens_saved} tokens saved "
                    f"({result.reduction_percentage:.1f}%), "
                    f"quality: {result.quality_score:.2f}"
                )

            return result.optimized_content, result

        except Exception as e:
            logger.error(f"Optimization error, using original prompt: {e}")
            return prompt, None

    async def _optimize_messages(self, messages: list[dict[str, str]]) -> tuple[list[dict[str, str]], Any | None]:
        """
        Optimize chat messages (typically the last user message).

        Returns:
            Tuple of (optimized_messages, optimization_result)
        """
        if not messages:
            return messages, None

        try:
            # Find last user message to optimize
            optimized_messages = messages.copy()
            last_user_idx = None

            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get("role") == "user":
                    last_user_idx = i
                    break

            if last_user_idx is None:
                return messages, None

            # Optimize the last user message
            user_content = messages[last_user_idx].get("content", "")
            if not user_content or len(user_content) < 100:
                # Too short to optimize
                return messages, None

            result = await self.optimizer.optimize(user_content)

            # Update message with optimized content
            optimized_messages[last_user_idx] = {
                **messages[last_user_idx],
                "content": result.optimized_content,
            }

            # Update middleware stats
            rollback_occurred = result.metadata.get("rollback_occurred", False)
            if result.validation_passed and not rollback_occurred:
                self.middleware_stats["optimized_calls"] += 1
                self.middleware_stats["total_tokens_saved"] += result.tokens_saved
                cost_savings = result.metadata.get("cost_savings_usd", 0.0)
                self.middleware_stats["total_cost_saved"] += cost_savings

                logger.info(
                    f"Optimized message: {result.tokens_saved} tokens saved "
                    f"({result.reduction_percentage:.1f}%), "
                    f"quality: {result.quality_score:.2f}"
                )

            return optimized_messages, result

        except Exception as e:
            logger.error(f"Message optimization error, using original: {e}")
            return messages, None

    async def _call_provider(
        self,
        prompt: str | None = None,
        messages: list[dict[str, str]] | None = None,
        optimization_result: Any | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Call the underlying provider with optimized content.

        Returns:
            Provider response with optimization metadata added
        """
        try:
            # Call provider based on its interface
            if hasattr(self.provider, "generate"):
                response = await self.provider.generate(prompt=prompt, messages=messages, **kwargs)
            elif hasattr(self.provider, "chat") and messages:
                response = await self.provider.chat(messages=messages, **kwargs)
            elif hasattr(self.provider, "complete") and prompt:
                response = await self.provider.complete(prompt=prompt, **kwargs)
            else:
                raise ValueError("Provider does not have a supported interface")

            # Add optimization metadata to response
            if optimization_result:
                if isinstance(response, dict):
                    response["optimization"] = {
                        "tokens_saved": optimization_result.tokens_saved,
                        "reduction_percentage": optimization_result.reduction_percentage,
                        "quality_score": optimization_result.quality_score,
                        "cost_savings_usd": optimization_result.metadata.get("cost_savings_usd", 0.0),
                        "processing_time_ms": optimization_result.processing_time_ms,
                    }

            result: dict[str, Any] = response
            return result

        except Exception as e:
            logger.error(f"Provider call error: {e}")
            raise

    def get_stats(self) -> dict[str, Any]:
        """
        Get middleware statistics including base optimizer stats.

        Returns:
            Combined statistics dictionary
        """
        optimizer_stats = self.optimizer.get_stats()

        result: dict[str, Any] = {
            "middleware": self.middleware_stats,
            "optimizer": optimizer_stats,
            "optimization_rate": (
                self.middleware_stats["optimized_calls"] / self.middleware_stats["total_calls"]
                if self.middleware_stats["total_calls"] > 0
                else 0.0
            ),
        }
        return result

    def reset_stats(self) -> None:
        """Reset all statistics"""
        self.middleware_stats = {
            "total_calls": 0,
            "optimized_calls": 0,
            "total_tokens_saved": 0,
            "total_cost_saved": 0.0,
        }

    # Proxy other provider methods
    def __getattr__(self, name: str) -> Any:
        """Proxy unknown methods to base provider"""
        return getattr(self.provider, name)


def wrap_provider(
    provider: Any,
    config: ContextOptimizationConfig | None = None,
    budget_tracker: Any | None = None,
) -> OptimizedProvider:
    """
    Convenience function to wrap a provider with optimization middleware.

    Args:
        provider: Base AI provider to wrap
        config: Optional optimization config (loads from env if None)
        budget_tracker: Optional budget tracker

    Returns:
        OptimizedProvider wrapping the base provider

    Example:
        >>> from src.providers import OpenAIProvider
        >>> from src.context_optimization import wrap_provider
        >>>
        >>> base = OpenAIProvider(api_key="...")
        >>> optimized = wrap_provider(base)
        >>>
        >>> # Use as normal - optimization happens automatically
        >>> response = await optimized.generate(prompt="Long prompt...")
    """
    return OptimizedProvider(
        provider=provider,
        config=config,
        budget_tracker=budget_tracker,
    )
