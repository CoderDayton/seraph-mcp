"""
Resilient Provider Wrapper

Wraps provider calls with retry logic and circuit breaker protection.
Implements P0 Phase 1c: Provider Integration with Resilience Patterns.

Architecture:
    Client → ResilientProvider → Circuit Breaker → Retry Logic → BaseProvider

Features:
- Automatic retry with exponential backoff for transient failures
- Circuit breaker per provider to prevent cascading failures
- Graceful degradation when providers fail
- Comprehensive error handling and observability
- Zero-config defaults with full configurability

Per SDD.md P0 Phase 1c:
- Integrate retry logic into provider calls
- Integrate circuit breaker into provider calls
- Graceful degradation when optional features fail
"""

import logging
from typing import Any

from ..errors import CircuitBreakerError
from ..observability.monitoring import get_observability
from ..resilience.circuit_breaker import get_circuit_breaker
from ..resilience.retry import RetryConfig, with_retry
from .base import BaseProvider, CompletionRequest, CompletionResponse, ModelInfo

logger = logging.getLogger(__name__)


class ResilientProvider:
    """
    Wrapper that adds retry and circuit breaker resilience to any provider.

    Automatically handles:
    - Transient failures with exponential backoff retry
    - Circuit breaking to fail fast when provider is unhealthy
    - Graceful degradation with structured error responses
    - Observability metrics for all resilience events

    Usage:
        >>> base_provider = OpenAIProvider(config)
        >>> resilient = ResilientProvider(base_provider)
        >>> response = await resilient.complete(request)
    """

    def __init__(
        self,
        provider: BaseProvider,
        retry_config: RetryConfig | None = None,
        circuit_breaker_enabled: bool = True,
        fail_max: int = 7,
        reset_timeout: int = 60,
    ):
        """
        Initialize resilient provider wrapper.

        Args:
            provider: Base provider instance to wrap
            retry_config: Retry configuration (uses defaults if None)
            circuit_breaker_enabled: Enable circuit breaker protection
            fail_max: Circuit breaker failure threshold (default: 7)
            reset_timeout: Circuit breaker reset timeout in seconds (default: 60)
        """
        self._provider = provider
        self._retry_config = retry_config or RetryConfig(
            max_retries=3,
            base_delay=1.0,
            max_delay=60.0,
            exponential_base=2.0,
            jitter=True,
            jitter_factor=0.1,
        )
        self._circuit_breaker_enabled = circuit_breaker_enabled
        self._circuit_breaker_manager = (
            get_circuit_breaker(fail_max=fail_max, reset_timeout=reset_timeout) if circuit_breaker_enabled else None
        )
        self._obs = get_observability()

        logger.info(
            f"ResilientProvider initialized for {provider.name}",
            extra={
                "provider": provider.name,
                "retry_enabled": True,
                "max_retries": self._retry_config.max_retries,
                "circuit_breaker_enabled": circuit_breaker_enabled,
                "fail_max": fail_max,
                "reset_timeout": reset_timeout,
            },
        )

    @property
    def name(self) -> str:
        """Get provider name."""
        return self._provider.name

    @property
    def provider(self) -> BaseProvider:
        """Get underlying provider instance."""
        return self._provider

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """
        Generate completion with retry and circuit breaker protection.

        Args:
            request: Completion request

        Returns:
            Completion response

        Raises:
            CircuitBreakerError: If circuit is open
            ProviderError: If all retries exhausted
        """
        model = request.model

        # Emit metric for attempt
        self._obs.increment(
            "provider.complete.attempt",
            tags={
                "provider": self.name,
                "model": model,
            },
        )

        try:
            # Wrap provider call with circuit breaker and retry
            result: CompletionResponse
            if self._circuit_breaker_enabled and self._circuit_breaker_manager:
                # Circuit breaker → retry → provider
                result = await self._circuit_breaker_manager.call_async(
                    provider_name=self.name,
                    func=self._complete_with_retry,
                    model=model,
                    request=request,
                )
            else:
                # Just retry → provider
                result = await self._complete_with_retry(request)

            # Success metric
            self._obs.increment(
                "provider.complete.success",
                tags={
                    "provider": self.name,
                    "model": model,
                },
            )

            return result

        except CircuitBreakerError as e:
            # Circuit is open - fail fast
            logger.warning(
                f"Circuit breaker {e.state} for {self.name}",
                extra={
                    "provider": self.name,
                    "model": model,
                    "circuit_state": e.state,
                    "fail_count": e.fail_count,
                },
            )

            self._obs.increment(
                "provider.complete.circuit_open",
                tags={
                    "provider": self.name,
                    "model": model,
                    "circuit_state": e.state,
                },
            )

            # Re-raise with structured error
            raise

        except Exception as e:
            # All retries exhausted or non-retryable error
            logger.error(
                f"Provider call failed after retries: {e}",
                extra={
                    "provider": self.name,
                    "model": model,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )

            self._obs.increment(
                "provider.complete.failed",
                tags={
                    "provider": self.name,
                    "model": model,
                    "error_type": type(e).__name__,
                },
            )

            # Re-raise
            raise

    async def _complete_with_retry(self, request: CompletionRequest) -> CompletionResponse:
        """
        Execute completion with retry logic.

        Args:
            request: Completion request

        Returns:
            Completion response
        """

        def on_retry_callback(attempt: int, error: Exception) -> None:
            """Callback for retry events."""
            logger.info(
                f"Retrying provider call (attempt {attempt})",
                extra={
                    "provider": self.name,
                    "model": request.model,
                    "attempt": attempt,
                    "error": str(error),
                    "error_type": type(error).__name__,
                },
            )

            self._obs.increment(
                "provider.retry.attempt",
                tags={
                    "provider": self.name,
                    "model": request.model,
                    "attempt": str(attempt),
                    "error_type": type(error).__name__,
                },
            )

        # Execute with retry
        result: CompletionResponse = await with_retry(
            self._provider.complete,
            request,
            config=self._retry_config,
            on_retry=on_retry_callback,
        )
        return result

    async def list_models(self) -> list[ModelInfo]:
        """
        List models with graceful degradation.

        Returns:
            List of models or empty list on failure
        """
        try:
            return await self._provider.list_models()
        except Exception as e:
            logger.warning(
                f"Failed to list models for {self.name}: {e}",
                extra={
                    "provider": self.name,
                    "error": str(e),
                },
            )
            # Graceful degradation - return empty list
            return []

    async def get_model_info(self, model_id: str) -> ModelInfo | None:
        """
        Get model info with graceful degradation.

        Args:
            model_id: Model identifier

        Returns:
            Model info or None on failure
        """
        try:
            return await self._provider.get_model_info(model_id)
        except Exception as e:
            logger.warning(
                f"Failed to get model info for {model_id}: {e}",
                extra={
                    "provider": self.name,
                    "model": model_id,
                    "error": str(e),
                },
            )
            # Graceful degradation - return None
            return None

    async def estimate_cost(
        self,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """
        Estimate cost with graceful degradation.

        Args:
            model_id: Model identifier
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Estimated cost or 0.0 on failure
        """
        try:
            return await self._provider.estimate_cost(model_id, input_tokens, output_tokens)
        except Exception as e:
            logger.warning(
                f"Failed to estimate cost: {e}",
                extra={
                    "provider": self.name,
                    "model": model_id,
                    "error": str(e),
                },
            )
            # Graceful degradation - return 0.0
            return 0.0

    async def health_check(self) -> bool:
        """
        Check provider health with graceful degradation.

        Returns:
            True if healthy, False otherwise
        """
        try:
            return await self._provider.health_check()
        except Exception as e:
            logger.warning(
                f"Health check failed for {self.name}: {e}",
                extra={
                    "provider": self.name,
                    "error": str(e),
                },
            )
            # Graceful degradation - return False
            return False

    async def close(self) -> None:
        """Close provider resources with graceful error handling."""
        try:
            await self._provider.close()
        except Exception as e:
            logger.warning(
                f"Error closing provider {self.name}: {e}",
                extra={
                    "provider": self.name,
                    "error": str(e),
                },
            )

    def get_circuit_breaker_state(self) -> str:
        """
        Get current circuit breaker state.

        Returns:
            State string: "closed", "open", "half_open", or "disabled"
        """
        if not self._circuit_breaker_enabled or not self._circuit_breaker_manager:
            return "disabled"

        return self._circuit_breaker_manager.get_state(self.name)

    def reset_circuit_breaker(self) -> None:
        """Manually reset circuit breaker to closed state."""
        if self._circuit_breaker_enabled and self._circuit_breaker_manager:
            self._circuit_breaker_manager.reset(self.name)
            logger.info(f"Circuit breaker reset for {self.name}")

    def get_stats(self) -> dict[str, Any]:
        """
        Get resilience statistics.

        Returns:
            Statistics dictionary with retry and circuit breaker info
        """
        stats: dict[str, Any] = {
            "provider": self.name,
            "retry": {
                "enabled": True,
                "max_retries": self._retry_config.max_retries,
                "base_delay": self._retry_config.base_delay,
                "max_delay": self._retry_config.max_delay,
            },
            "circuit_breaker": {
                "enabled": self._circuit_breaker_enabled,
            },
        }

        if self._circuit_breaker_enabled and self._circuit_breaker_manager:
            cb_stats = self._circuit_breaker_manager.get_stats(self.name)
            stats["circuit_breaker"].update(cb_stats)

        return stats


def wrap_provider_with_resilience(
    provider: BaseProvider,
    retry_config: RetryConfig | None = None,
    circuit_breaker_enabled: bool = True,
    fail_max: int = 7,
    reset_timeout: int = 60,
) -> ResilientProvider:
    """
    Convenience function to wrap provider with resilience patterns.

    Args:
        provider: Base provider to wrap
        retry_config: Optional retry configuration
        circuit_breaker_enabled: Enable circuit breaker
        fail_max: Circuit breaker failure threshold
        reset_timeout: Circuit breaker reset timeout

    Returns:
        Resilient provider wrapper

    Example:
        >>> from src.providers import create_provider, wrap_provider_with_resilience
        >>> from src.providers.base import ProviderConfig
        >>>
        >>> config = ProviderConfig(api_key="sk-...")
        >>> base = create_provider("openai", config)
        >>> resilient = wrap_provider_with_resilience(base)
        >>>
        >>> # Now use resilient provider with automatic retry and circuit breaking
        >>> response = await resilient.complete(request)
    """
    return ResilientProvider(
        provider=provider,
        retry_config=retry_config,
        circuit_breaker_enabled=circuit_breaker_enabled,
        fail_max=fail_max,
        reset_timeout=reset_timeout,
    )
