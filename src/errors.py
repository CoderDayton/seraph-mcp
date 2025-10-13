"""
Seraph MCP - Core Error Types

Defines standard exception hierarchy for the Seraph MCP core runtime.
All exceptions should inherit from SeraphError for consistent error handling.
"""

from typing import Any


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
        details = {"provider": provider}
        if retry_after:
            details["retry_after"] = retry_after
        super().__init__(message, details, status_code=429)


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
