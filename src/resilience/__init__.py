"""
Seraph MCP - Resilience Module

Provides resilience patterns for production reliability:
- Exponential backoff retry logic
- Circuit breaker pattern integration
- Error detection and classification

P0 Implementation for robust error handling.
"""

from .circuit_breaker import CircuitBreakerManager, get_circuit_breaker
from .retry import RetryConfig, exponential_backoff, with_retry

__all__ = [
    # Circuit breaker
    "CircuitBreakerManager",
    "get_circuit_breaker",
    # Retry logic
    "RetryConfig",
    "exponential_backoff",
    "with_retry",
]
