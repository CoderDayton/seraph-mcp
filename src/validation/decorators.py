"""
Seraph MCP - Validation Decorators

Provides decorators for applying Pydantic validation to MCP tools.

P0 Implementation:
- validate_input decorator for automatic input validation
- Structured error responses with ErrorCode
- Integration with observability for validation failures
"""

import functools
import logging
from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from ..errors import ErrorCode, make_error_response
from ..observability.monitoring import get_observability

logger = logging.getLogger(__name__)

T = TypeVar("T")


def validate_input(schema: type[BaseModel]) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator to validate tool inputs using Pydantic schema.

    Args:
        schema: Pydantic model class for input validation

    Returns:
        Decorated function with automatic validation

    Example:
        >>> @validate_input(CountTokensInput)
        ... async def count_tokens(content: str, model: str = "gpt-4", **kwargs):
        ...     # Function receives validated inputs
        ...     pass

    Error Response:
        {
            "success": False,
            "error_code": "INVALID_INPUT",
            "message": "Input validation failed",
            "details": {
                "validation_errors": [
                    {
                        "field": "content",
                        "message": "String should have at least 1 character",
                        "type": "string_too_short"
                    }
                ]
            }
        }
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            obs = get_observability()

            try:
                # Validate input parameters
                validated = schema(**kwargs)

                # Convert validated model back to dict for function call
                validated_kwargs = validated.model_dump(exclude_unset=False)

                # Call original function with validated inputs
                return await func(*args, **validated_kwargs)

            except ValidationError as e:
                # Extract validation errors
                validation_errors = []
                for error in e.errors():
                    field_path = " -> ".join(str(loc) for loc in error["loc"])
                    validation_errors.append(
                        {
                            "field": field_path,
                            "message": error["msg"],
                            "type": error["type"],
                        }
                    )

                # Log validation failure
                logger.warning(
                    f"Input validation failed for {func.__name__}",
                    extra={
                        "function": func.__name__,
                        "validation_errors": validation_errors,
                        "input_kwargs": kwargs,
                    },
                )

                # Emit metric
                obs.increment(
                    "validation.failed",
                    tags={
                        "function": func.__name__,
                        "error_count": str(len(validation_errors)),
                    },
                )

                # Return structured error response
                return make_error_response(
                    error_code=ErrorCode.INVALID_INPUT,
                    message="Input validation failed",
                    context={
                        "validation_errors": validation_errors,
                        "function": func.__name__,
                    },
                )

            except Exception as e:
                # Unexpected error during validation
                logger.error(
                    f"Unexpected error during validation for {func.__name__}: {e}",
                    extra={
                        "function": func.__name__,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                    exc_info=True,
                )

                obs.increment(
                    "validation.error",
                    tags={
                        "function": func.__name__,
                        "error_type": type(e).__name__,
                    },
                )

                # Return generic error
                return make_error_response(
                    error_code=ErrorCode.INTERNAL_ERROR,
                    message=f"Validation error: {str(e)}",
                    context={"function": func.__name__},
                )

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            obs = get_observability()

            try:
                # Validate input parameters
                validated = schema(**kwargs)

                # Convert validated model back to dict for function call
                validated_kwargs = validated.model_dump(exclude_unset=False)

                # Call original function with validated inputs
                return func(*args, **validated_kwargs)

            except ValidationError as e:
                # Extract validation errors
                validation_errors = []
                for error in e.errors():
                    field_path = " -> ".join(str(loc) for loc in error["loc"])
                    validation_errors.append(
                        {
                            "field": field_path,
                            "message": error["msg"],
                            "type": error["type"],
                        }
                    )

                # Log validation failure
                logger.warning(
                    f"Input validation failed for {func.__name__}",
                    extra={
                        "function": func.__name__,
                        "validation_errors": validation_errors,
                        "input_kwargs": kwargs,
                    },
                )

                # Emit metric
                obs.increment(
                    "validation.failed",
                    tags={
                        "function": func.__name__,
                        "error_count": str(len(validation_errors)),
                    },
                )

                # Return structured error response
                return make_error_response(
                    error_code=ErrorCode.INVALID_INPUT,
                    message="Input validation failed",
                    context={
                        "validation_errors": validation_errors,
                        "function": func.__name__,
                    },
                )

            except Exception as e:
                # Unexpected error during validation
                logger.error(
                    f"Unexpected error during validation for {func.__name__}: {e}",
                    extra={
                        "function": func.__name__,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                    exc_info=True,
                )

                obs.increment(
                    "validation.error",
                    tags={
                        "function": func.__name__,
                        "error_type": type(e).__name__,
                    },
                )

                # Return generic error
                return make_error_response(
                    error_code=ErrorCode.INTERNAL_ERROR,
                    message=f"Validation error: {str(e)}",
                    context={"function": func.__name__},
                )

        # Return appropriate wrapper based on function type
        import inspect

        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator
