"""
Seraph MCP - Retry Logic with Exponential Backoff

Provides retry utilities with exponential backoff and jitter for transient failures.

P0 Implementation:
- Exponential backoff with jitter to prevent thundering herd
- Configurable max retries and timeouts
- Integration with error detection
- Async/await support
"""

import asyncio
import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..errors import is_retryable_error

logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    """
    Configuration for retry behavior.

    Attributes:
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Initial delay in seconds (default: 1.0)
        max_delay: Maximum delay in seconds (default: 60.0)
        exponential_base: Exponential backoff base (default: 2.0)
        jitter: Add random jitter to prevent thundering herd (default: True)
        jitter_factor: Jitter randomization factor 0-1 (default: 0.1)
    """

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    jitter_factor: float = 0.1

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.base_delay <= 0:
            raise ValueError("base_delay must be positive")
        if self.max_delay < self.base_delay:
            raise ValueError("max_delay must be >= base_delay")
        if self.exponential_base < 1.0:
            raise ValueError("exponential_base must be >= 1.0")
        if not 0 <= self.jitter_factor <= 1:
            raise ValueError("jitter_factor must be between 0 and 1")


def exponential_backoff(
    attempt: int,
    base_delay: float = 1.0,
    exponential_base: float = 2.0,
    max_delay: float = 60.0,
    jitter: bool = True,
    jitter_factor: float = 0.1,
) -> float:
    """
    Calculate exponential backoff delay with optional jitter.

    Args:
        attempt: Current attempt number (0-indexed)
        base_delay: Initial delay in seconds
        exponential_base: Base for exponential calculation
        max_delay: Maximum delay cap
        jitter: Whether to add random jitter
        jitter_factor: Jitter randomization factor (0-1)

    Returns:
        Delay in seconds

    Example:
        >>> exponential_backoff(0)  # ~1.0s
        >>> exponential_backoff(1)  # ~2.0s
        >>> exponential_backoff(2)  # ~4.0s
        >>> exponential_backoff(3)  # ~8.0s
    """
    # Calculate exponential delay
    delay = min(base_delay * (exponential_base**attempt), max_delay)

    # Add jitter to prevent thundering herd
    if jitter and jitter_factor > 0:
        jitter_amount = delay * jitter_factor
        delay = delay + random.uniform(-jitter_amount, jitter_amount)
        # Ensure delay is positive
        delay = max(0.1, delay)

    return delay


async def with_retry(
    func: Callable[..., Any],
    *args: Any,
    config: RetryConfig | None = None,
    on_retry: Callable[[int, Exception], None] | None = None,
    **kwargs: Any,
) -> Any:
    """
    Execute async function with retry logic.

    Args:
        func: Async function to execute
        *args: Positional arguments for func
        config: Retry configuration (uses defaults if None)
        on_retry: Optional callback called on each retry (attempt, error)
        **kwargs: Keyword arguments for func

    Returns:
        Result of successful function execution

    Raises:
        Last exception if all retries exhausted

    Example:
        >>> async def flaky_api_call():
        ...     # May fail transiently
        ...     return await provider.generate(...)
        >>>
        >>> result = await with_retry(
        ...     flaky_api_call,
        ...     config=RetryConfig(max_retries=3)
        ... )
    """
    if config is None:
        config = RetryConfig()

    last_exception: Exception | None = None

    for attempt in range(config.max_retries + 1):
        try:
            # Execute function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            # Success - return result
            if attempt > 0:
                logger.info(
                    f"Retry succeeded after {attempt} attempts",
                    extra={"attempt": attempt, "function": func.__name__},
                )
            return result

        except Exception as e:
            last_exception = e

            # Check if error is retryable
            if not is_retryable_error(e):
                logger.debug(
                    f"Non-retryable error, not retrying: {e}",
                    extra={"error_type": type(e).__name__, "function": func.__name__},
                )
                raise

            # Check if we have retries left
            if attempt >= config.max_retries:
                logger.error(
                    f"All {config.max_retries} retries exhausted",
                    extra={
                        "function": func.__name__,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                )
                raise

            # Calculate backoff delay
            delay = exponential_backoff(
                attempt=attempt,
                base_delay=config.base_delay,
                exponential_base=config.exponential_base,
                max_delay=config.max_delay,
                jitter=config.jitter,
                jitter_factor=config.jitter_factor,
            )

            logger.warning(
                f"Retry attempt {attempt + 1}/{config.max_retries} after {delay:.2f}s",
                extra={
                    "attempt": attempt + 1,
                    "max_retries": config.max_retries,
                    "delay_seconds": delay,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "function": func.__name__,
                },
            )

            # Call retry callback if provided
            if on_retry:
                try:
                    on_retry(attempt + 1, e)
                except Exception as callback_error:
                    logger.error(f"Retry callback failed: {callback_error}")

            # Wait before retry
            await asyncio.sleep(delay)

    # Should never reach here, but just in case
    if last_exception:
        raise last_exception
    raise RuntimeError("Retry logic error: no exception and no result")


def with_retry_sync[T](
    func: Callable[..., T],
    *args: Any,
    config: RetryConfig | None = None,
    on_retry: Callable[[int, Exception], None] | None = None,
    **kwargs: Any,
) -> T:
    """
    Execute synchronous function with retry logic.

    Args:
        func: Synchronous function to execute
        *args: Positional arguments for func
        config: Retry configuration (uses defaults if None)
        on_retry: Optional callback called on each retry (attempt, error)
        **kwargs: Keyword arguments for func

    Returns:
        Result of successful function execution

    Raises:
        Last exception if all retries exhausted
    """
    if config is None:
        config = RetryConfig()

    last_exception: Exception | None = None

    for attempt in range(config.max_retries + 1):
        try:
            # Execute function
            result = func(*args, **kwargs)

            # Success - return result
            if attempt > 0:
                logger.info(
                    f"Retry succeeded after {attempt} attempts",
                    extra={"attempt": attempt, "function": func.__name__},
                )
            return result

        except Exception as e:
            last_exception = e

            # Check if error is retryable
            if not is_retryable_error(e):
                logger.debug(
                    f"Non-retryable error, not retrying: {e}",
                    extra={"error_type": type(e).__name__, "function": func.__name__},
                )
                raise

            # Check if we have retries left
            if attempt >= config.max_retries:
                logger.error(
                    f"All {config.max_retries} retries exhausted",
                    extra={
                        "function": func.__name__,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                )
                raise

            # Calculate backoff delay
            delay = exponential_backoff(
                attempt=attempt,
                base_delay=config.base_delay,
                exponential_base=config.exponential_base,
                max_delay=config.max_delay,
                jitter=config.jitter,
                jitter_factor=config.jitter_factor,
            )

            logger.warning(
                f"Retry attempt {attempt + 1}/{config.max_retries} after {delay:.2f}s",
                extra={
                    "attempt": attempt + 1,
                    "max_retries": config.max_retries,
                    "delay_seconds": delay,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "function": func.__name__,
                },
            )

            # Call retry callback if provided
            if on_retry:
                try:
                    on_retry(attempt + 1, e)
                except Exception as callback_error:
                    logger.error(f"Retry callback failed: {callback_error}")

            # Wait before retry
            time.sleep(delay)

    # Should never reach here, but just in case
    if last_exception:
        raise last_exception
    raise RuntimeError("Retry logic error: no exception and no result")
