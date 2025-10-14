"""
Seraph MCP - Core Error Types

Defines standard exception hierarchy for the Seraph MCP core runtime.
All exceptions should inherit from SeraphError for consistent error handling.

P0 Enhancements:
- ErrorCode enum for structured error responses
- Circuit breaker error types
- Enhanced provider error handling
- Validation error utilities
"""

from enum import Enum
from typing import Any


class ErrorCode(str, Enum):
    """
    Standard error codes for MCP tool responses.

    Used for structured error handling and client-side error recovery.
    """

    # Input validation errors
    INVALID_INPUT = "INVALID_INPUT"
    MISSING_PARAMETER = "MISSING_PARAMETER"
    INVALID_PARAMETER_TYPE = "INVALID_PARAMETER_TYPE"
    INVALID_PARAMETER_VALUE = "INVALID_PARAMETER_VALUE"

    # Provider errors
    PROVIDER_ERROR = "PROVIDER_ERROR"
    PROVIDER_TIMEOUT = "PROVIDER_TIMEOUT"
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
    RATE_LIMITED = "RATE_LIMITED"

    # Circuit breaker errors
    CIRCUIT_OPEN = "CIRCUIT_OPEN"
    CIRCUIT_HALF_OPEN = "CIRCUIT_HALF_OPEN"

    # Feature availability errors
    FEATURE_DISABLED = "FEATURE_DISABLED"
    OPTIMIZATION_DISABLED = "OPTIMIZATION_DISABLED"
    CACHE_DISABLED = "CACHE_DISABLED"
    BUDGET_DISABLED = "BUDGET_DISABLED"

    # Cache errors
    CACHE_FAILURE = "CACHE_FAILURE"
    CACHE_MISS = "CACHE_MISS"

    # Budget errors
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"

    # Internal errors
    INTERNAL_ERROR = "INTERNAL_ERROR"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"


class SeraphError(Exception):
    """Base exception for all Seraph MCP errors."""

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        status_code: int = 500,
    ):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.status_code = status_code

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
        }


class ConfigurationError(SeraphError):
    """Raised when configuration is invalid or missing."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, details, status_code=500)


class CacheError(SeraphError):
    """Base exception for cache-related errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, details, status_code=500)


class CacheConnectionError(CacheError):
    """Raised when cache backend connection fails."""

    def __init__(self, backend: str, details: dict[str, Any] | None = None):
        message = f"Failed to connect to cache backend: {backend}"
        super().__init__(message, details)


class CacheOperationError(CacheError):
    """Raised when cache operation fails."""

    pass


class RoutingError(SeraphError):
    """Base exception for routing-related errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, details, status_code=500)


class ModelNotFoundError(RoutingError):
    """Raised when requested model is not available."""

    def __init__(self, model_id: str, details: dict[str, Any] | None = None):
        message = f"Model not found: {model_id}"
        super().__init__(message, details)


class RoutingPolicyError(RoutingError):
    """Raised when routing policy fails to select a model."""

    pass


class OptimizationError(SeraphError):
    """Base exception for optimization-related errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, details, status_code=500)


class QualityValidationError(SeraphError):
    """Raised when quality validation fails."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, details, status_code=422)


class BudgetExceededError(SeraphError):
    """Raised when budget limits are exceeded."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, details, status_code=429)


class ProviderError(SeraphError):
    """Base exception for provider-related errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, details, status_code=502)


class ProviderTimeoutError(ProviderError):
    """Raised when provider call times out."""

    def __init__(self, provider: str, timeout: float):
        message = f"Provider {provider} timed out after {timeout}s"
        super().__init__(message, {"provider": provider, "timeout": timeout})


class ProviderRateLimitError(ProviderError):
    """Raised when provider rate limit is hit."""

    def __init__(self, provider: str, retry_after: int | None = None):
        message = f"Provider {provider} rate limit exceeded"
        details: dict[str, Any] = {"provider": provider, "error_code": ErrorCode.RATE_LIMITED}
        if retry_after:
            details["retry_after"] = retry_after
        super().__init__(message, details)
        self.status_code = 429


class CircuitBreakerError(SeraphError):
    """Raised when circuit breaker is open or half-open."""

    def __init__(
        self,
        provider: str,
        state: str,
        fail_count: int | None = None,
        reset_timeout: int | None = None,
    ):
        message = f"Circuit breaker {state} for provider {provider}"
        details: dict[str, Any] = {
            "provider": provider,
            "circuit_state": state,
            "error_code": ErrorCode.CIRCUIT_OPEN if state == "open" else ErrorCode.CIRCUIT_HALF_OPEN,
        }
        if fail_count is not None:
            details["fail_count"] = fail_count
        if reset_timeout is not None:
            details["reset_timeout_seconds"] = reset_timeout
        super().__init__(message, details, status_code=503)

        # Store as instance attributes for access in exception handlers
        self.provider = provider
        self.state = state
        self.fail_count = fail_count
        self.reset_timeout = reset_timeout


class ValidationError(SeraphError):
    """Raised when input validation fails."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, details, status_code=400)


class NotFoundError(SeraphError):
    """Raised when a requested resource is not found."""

    def __init__(self, resource: str, identifier: str):
        message = f"{resource} not found: {identifier}"
        super().__init__(message, {"resource": resource, "id": identifier}, status_code=404)


class UnauthorizedError(SeraphError):
    """Raised when authentication fails."""

    def __init__(self, message: str = "Unauthorized"):
        super().__init__(message, status_code=401)


class ForbiddenError(SeraphError):
    """Raised when access is forbidden."""

    def __init__(self, message: str = "Forbidden"):
        super().__init__(message, status_code=403)


class DependencyError(SeraphError):
    """Raised when a required dependency is missing or fails to load."""

    def __init__(
        self,
        package: str,
        feature: str | None = None,
        install_hint: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        if feature:
            message = f"Required dependency '{package}' is missing for {feature}"
        else:
            message = f"Required dependency '{package}' is missing"

        if install_hint:
            message += f". Install with: {install_hint}"

        error_details = details or {}
        error_details.update(
            {
                "package": package,
                "feature": feature,
                "install_hint": install_hint,
            }
        )

        super().__init__(message, error_details, status_code=500)


class CompressionError(SeraphError):
    """Raised when compression operations fail."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, details, status_code=500)


def make_error_response(
    error_code: ErrorCode,
    message: str,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Create a standardized error response for MCP tools.

    Args:
        error_code: Standard error code
        message: Human-readable error message
        context: Additional context/details

    Returns:
        Standardized error response dictionary

    Example:
        >>> make_error_response(
        ...     ErrorCode.INVALID_INPUT,
        ...     "Query parameter is required",
        ...     {"parameter": "query", "provided_value": None}
        ... )
        {
            "success": False,
            "error_code": "INVALID_INPUT",
            "message": "Query parameter is required",
            "details": {"parameter": "query", "provided_value": None}
        }
    """
    return {
        "success": False,
        "error_code": error_code.value,
        "message": message,
        "details": context or {},
    }


def is_retryable_error(error: Exception) -> bool:
    """
    Check if an error is transient and should be retried.

    Args:
        error: Exception to check

    Returns:
        True if error is retryable (transient)
    """
    # Network/timeout errors are retryable
    if isinstance(error, ProviderTimeoutError):
        return True

    # Rate limits are retryable after backoff
    if isinstance(error, ProviderRateLimitError):
        return True

    # Generic provider errors might be transient
    if isinstance(error, ProviderError):
        # Check for common transient error indicators
        error_msg = str(error).lower()
        transient_indicators = [
            "timeout",
            "connection",
            "network",
            "503",
            "502",
            "504",
            "unavailable",
            "temporary",
        ]
        return any(indicator in error_msg for indicator in transient_indicators)

    return False


def extract_error_code(error: Exception) -> ErrorCode:
    """
    Extract appropriate ErrorCode from an exception.

    Args:
        error: Exception to categorize

    Returns:
        Appropriate ErrorCode for the exception
    """
    if isinstance(error, CircuitBreakerError):
        return ErrorCode.CIRCUIT_OPEN

    if isinstance(error, ProviderRateLimitError):
        return ErrorCode.RATE_LIMITED

    if isinstance(error, ProviderTimeoutError):
        return ErrorCode.PROVIDER_TIMEOUT

    if isinstance(error, ProviderError):
        return ErrorCode.PROVIDER_ERROR

    if isinstance(error, ValidationError):
        return ErrorCode.INVALID_INPUT

    if isinstance(error, CacheError):
        return ErrorCode.CACHE_FAILURE

    if isinstance(error, BudgetExceededError):
        return ErrorCode.BUDGET_EXCEEDED

    if isinstance(error, ConfigurationError):
        return ErrorCode.FEATURE_DISABLED

    return ErrorCode.INTERNAL_ERROR
