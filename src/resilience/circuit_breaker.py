"""
Seraph MCP - Circuit Breaker Pattern

Provides circuit breaker implementation using pybreaker library.
Prevents cascading failures by failing fast when a provider is unhealthy.

P0 Implementation:
- Circuit breaker per provider (provider_name + model key)
- Configurable thresholds (fail_max=7, reset_timeout=60s)
- Integration with error detection
- Metrics tracking
"""

import logging
from typing import Any

try:
    import pybreaker

    PYBREAKER_AVAILABLE = True
except ImportError:
    pybreaker = None  # type: ignore
    PYBREAKER_AVAILABLE = False

from ..errors import CircuitBreakerError
from ..observability.monitoring import get_observability

logger = logging.getLogger(__name__)


class CircuitBreakerManager:
    """
    Manages circuit breakers for all providers.

    Each provider gets its own circuit breaker based on (provider_name, model) key.
    Circuit breakers prevent cascading failures by failing fast when providers are unhealthy.
    """

    def __init__(
        self,
        fail_max: int = 7,
        reset_timeout: int = 60,
        exclude_exceptions: tuple[type[Exception], ...] | None = None,
    ):
        """
        Initialize circuit breaker manager.

        Args:
            fail_max: Number of consecutive failures before opening circuit (default: 7)
            reset_timeout: Seconds to wait before attempting reset (default: 60)
            exclude_exceptions: Exception types to exclude from failure count
        """
        if not PYBREAKER_AVAILABLE or pybreaker is None:
            raise RuntimeError(
                "pybreaker is required for circuit breaker functionality. Install with: uv add pybreaker"
            )

        self.fail_max = fail_max
        self.reset_timeout = reset_timeout
        self.exclude_exceptions = exclude_exceptions or ()

        # Store circuit breakers by key (provider_name, model)
        self._breakers: dict[str, Any] = {}

        # Observability
        self._obs = get_observability()

    def _get_breaker_key(self, provider_name: str, model: str | None = None) -> str:
        """Generate unique key for circuit breaker."""
        if model:
            return f"{provider_name}:{model}"
        return provider_name

    def get_breaker(self, provider_name: str, model: str | None = None) -> Any:
        """
        Get or create circuit breaker for provider.

        Args:
            provider_name: Provider name (e.g., "openai", "anthropic")
            model: Optional model name for finer-grained breakers

        Returns:
            pybreaker.CircuitBreaker instance
        """
        if not PYBREAKER_AVAILABLE or pybreaker is None:
            raise RuntimeError("pybreaker not available")

        key = self._get_breaker_key(provider_name, model)

        if key not in self._breakers:
            # Create new circuit breaker
            breaker = pybreaker.CircuitBreaker(
                fail_max=self.fail_max,
                reset_timeout=self.reset_timeout,
                exclude=[*self.exclude_exceptions],
                name=key,
                listeners=[CircuitBreakerListener(self._obs, provider_name, model)],
            )
            self._breakers[key] = breaker
            logger.info(
                f"Created circuit breaker for {key}",
                extra={
                    "provider": provider_name,
                    "model": model,
                    "fail_max": self.fail_max,
                    "reset_timeout": self.reset_timeout,
                },
            )

        return self._breakers[key]

    def call(
        self,
        provider_name: str,
        func: Any,
        *args: Any,
        model: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Execute function with circuit breaker protection.

        Args:
            provider_name: Provider name
            func: Function to execute
            *args: Positional arguments
            model: Optional model name
            **kwargs: Keyword arguments

        Returns:
            Result of function execution

        Raises:
            CircuitBreakerError: If circuit is open
            Original exception: If function fails
        """
        breaker = self.get_breaker(provider_name, model)

        try:
            return breaker.call(func, *args, **kwargs)
        except pybreaker.CircuitBreakerError as e:
            # Circuit is open - fail fast
            state = "open" if "open" in str(e).lower() else "half-open"
            raise CircuitBreakerError(
                provider=provider_name,
                state=state,
                fail_count=breaker.fail_counter,
                reset_timeout=self.reset_timeout,
            ) from e

    async def call_async(
        self,
        provider_name: str,
        func: Any,
        *args: Any,
        model: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Execute async function with circuit breaker protection.

        Args:
            provider_name: Provider name
            func: Async function to execute
            *args: Positional arguments
            model: Optional model name
            **kwargs: Keyword arguments

        Returns:
            Result of function execution

        Raises:
            CircuitBreakerError: If circuit is open
            Original exception: If function fails
        """
        breaker = self.get_breaker(provider_name, model)

        try:
            return await breaker.call_async(func, *args, **kwargs)
        except pybreaker.CircuitBreakerError as e:
            # Circuit is open - fail fast
            state = "open" if "open" in str(e).lower() else "half-open"
            raise CircuitBreakerError(
                provider=provider_name,
                state=state,
                fail_count=breaker.fail_counter,
                reset_timeout=self.reset_timeout,
            ) from e

    def get_state(self, provider_name: str, model: str | None = None) -> str:
        """
        Get current circuit breaker state.

        Args:
            provider_name: Provider name
            model: Optional model name

        Returns:
            State string: "closed", "open", or "half_open"
        """
        key = self._get_breaker_key(provider_name, model)
        if key not in self._breakers:
            return "closed"  # No breaker means healthy

        breaker = self._breakers[key]
        state: str = str(breaker.current_state)
        return state

    def get_stats(self, provider_name: str | None = None, model: str | None = None) -> dict[str, Any]:
        """
        Get circuit breaker statistics.

        Args:
            provider_name: Optional provider name filter
            model: Optional model name filter

        Returns:
            Statistics dictionary
        """
        if provider_name:
            key = self._get_breaker_key(provider_name, model)
            if key not in self._breakers:
                return {"provider": provider_name, "model": model, "state": "closed", "fail_count": 0}

            breaker = self._breakers[key]
            return {
                "provider": provider_name,
                "model": model,
                "state": breaker.current_state,
                "fail_count": breaker.fail_counter,
                "fail_max": self.fail_max,
                "reset_timeout": self.reset_timeout,
            }

        # Return stats for all breakers
        return {
            key: {
                "state": breaker.current_state,
                "fail_count": breaker.fail_counter,
                "fail_max": self.fail_max,
                "reset_timeout": self.reset_timeout,
            }
            for key, breaker in self._breakers.items()
        }

    def reset(self, provider_name: str, model: str | None = None) -> None:
        """
        Manually reset circuit breaker to closed state.

        Args:
            provider_name: Provider name
            model: Optional model name
        """
        key = self._get_breaker_key(provider_name, model)
        if key in self._breakers:
            breaker = self._breakers[key]
            breaker.close()
            logger.info(f"Circuit breaker reset for {key}")

    def reset_all(self) -> None:
        """Reset all circuit breakers to closed state."""
        for key, breaker in self._breakers.items():
            breaker.close()
            logger.info(f"Circuit breaker reset for {key}")


class CircuitBreakerListener(pybreaker.CircuitBreakerListener if PYBREAKER_AVAILABLE else object):  # type: ignore
    """
    Listener for circuit breaker state transitions.

    Logs state changes and emits observability metrics.
    """

    def __init__(self, observability: Any, provider_name: str, model: str | None = None):
        """Initialize listener."""
        self.obs = observability
        self.provider = provider_name
        self.model = model

    def state_change(self, cb: Any, old_state: Any, new_state: Any) -> None:
        """Called when circuit breaker state changes."""
        logger.warning(
            f"Circuit breaker state change: {old_state.name} -> {new_state.name}",
            extra={
                "provider": self.provider,
                "model": self.model,
                "old_state": old_state.name,
                "new_state": new_state.name,
                "fail_count": cb.fail_counter,
            },
        )

        # Emit metric
        self.obs.increment(
            "circuit_breaker.state_change",
            tags={
                "provider": self.provider,
                "model": self.model or "default",
                "old_state": old_state.name,
                "new_state": new_state.name,
            },
        )

    def failure(self, cb: Any, exc: Exception) -> None:
        """Called when a call fails."""
        logger.debug(
            "Circuit breaker failure recorded",
            extra={
                "provider": self.provider,
                "model": self.model,
                "error": str(exc),
                "error_type": type(exc).__name__,
                "fail_count": cb.fail_counter,
            },
        )

        # Emit metric
        self.obs.increment(
            "circuit_breaker.failure",
            tags={
                "provider": self.provider,
                "model": self.model or "default",
                "error_type": type(exc).__name__,
            },
        )

    def success(self, cb: Any) -> None:
        """Called when a call succeeds."""
        # Only log if we were recovering from failures
        if cb.fail_counter > 0:
            logger.info(
                "Circuit breaker success after failures",
                extra={
                    "provider": self.provider,
                    "model": self.model,
                    "previous_fail_count": cb.fail_counter,
                },
            )

        # Emit metric
        self.obs.increment(
            "circuit_breaker.success",
            tags={
                "provider": self.provider,
                "model": self.model or "default",
            },
        )


# Global circuit breaker manager singleton
_circuit_breaker_manager: CircuitBreakerManager | None = None


def get_circuit_breaker(
    fail_max: int = 7,
    reset_timeout: int = 60,
) -> CircuitBreakerManager:
    """
    Get or create global circuit breaker manager.

    Args:
        fail_max: Number of consecutive failures before opening circuit
        reset_timeout: Seconds to wait before attempting reset

    Returns:
        Global CircuitBreakerManager instance
    """
    global _circuit_breaker_manager

    if _circuit_breaker_manager is None:
        _circuit_breaker_manager = CircuitBreakerManager(
            fail_max=fail_max,
            reset_timeout=reset_timeout,
        )

    return _circuit_breaker_manager
