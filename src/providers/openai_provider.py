"""
OpenAI Provider Implementation

Provides integration with OpenAI's API for GPT models.
Uses Models.dev API for dynamic model information and pricing.

Per SDD.md:
- Minimal, clean implementation
- Typed with Pydantic
- Async-first design
- Comprehensive error handling
- Dynamic model loading from Models.dev
"""

import asyncio
import logging
import time
from typing import Any

try:
    import openai
    from openai import AsyncOpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    openai = None  # type: ignore[assignment]
    AsyncOpenAI = None  # type: ignore[assignment, misc]
    OPENAI_AVAILABLE = False

from ..errors import DependencyError, ProviderError, ProviderRateLimitError, ProviderTimeoutError
from .base import (
    BaseProvider,
    CompletionRequest,
    CompletionResponse,
    ProviderConfig,
)

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseProvider):
    """OpenAI API provider implementation with dynamic model loading."""

    def __init__(self, config: ProviderConfig) -> None:
        """
        Initialize OpenAI provider.

        Args:
            config: Provider configuration

        Raises:
            DependencyError: If OpenAI SDK is not installed
            ValueError: If configuration is invalid
        """
        super().__init__(config)

        if not OPENAI_AVAILABLE or AsyncOpenAI is None:
            logger.error("OpenAI SDK is not installed", extra={"package": "openai", "provider": "openai"})
            raise DependencyError(
                package="openai",
                feature="OpenAI provider",
                install_hint="pip install 'openai>=1.0.0'",
                details={"provider": "openai"},
            )

        try:
            # Initialize async client
            client_kwargs: dict[str, Any] = {
                "api_key": config.api_key,
                "timeout": config.timeout,
                "max_retries": config.max_retries,
            }

            if config.base_url:
                client_kwargs["base_url"] = config.base_url

            self.client = AsyncOpenAI(**client_kwargs)

            logger.info(
                "OpenAI provider initialized",
                extra={
                    "provider": "openai",
                    "base_url": config.base_url or "default",
                    "timeout": config.timeout,
                    "max_retries": config.max_retries,
                },
            )
        except Exception as e:
            logger.error(
                f"Failed to initialize OpenAI client: {e}", extra={"provider": "openai", "error": str(e)}, exc_info=True
            )
            raise RuntimeError(f"Failed to initialize OpenAI provider: {e}") from e

    def _validate_config(self) -> None:
        """
        Validate OpenAI-specific configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        if not self.config.api_key:
            logger.error("OpenAI API key is missing", extra={"provider": "openai"})
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")

        if len(self.config.api_key) < 20:
            logger.error(
                "OpenAI API key has invalid format",
                extra={"provider": "openai", "key_length": len(self.config.api_key)},
            )
            raise ValueError(
                "Invalid OpenAI API key format. OpenAI keys should start with 'sk-' and be at least 20 characters."
            )

    @property
    def name(self) -> str:
        """Return provider name."""
        return "openai"

    @property
    def models_dev_provider_id(self) -> str | None:
        """Return Models.dev provider ID."""
        return "openai"

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """
        Generate completion using OpenAI API.

        Args:
            request: Completion request parameters

        Returns:
            CompletionResponse with generated content and metadata

        Raises:
            RuntimeError: If provider is disabled
            ProviderError: If API call fails
            ProviderRateLimitError: If rate limit is exceeded
            ProviderTimeoutError: If request times out
        """
        if not self.config.enabled:
            logger.warning(
                f"Attempted to use disabled provider: {self.name}",
                extra={"provider": self.name, "model": request.model},
            )
            raise RuntimeError(f"Provider {self.name} is disabled")

        start_time = time.time()

        try:
            # Prepare API request
            api_params: dict[str, Any] = {
                "model": request.model,
                "messages": request.messages,
                "temperature": request.temperature,
                "stream": request.stream,
            }

            if request.max_tokens:
                api_params["max_tokens"] = request.max_tokens

            # Add any extra provider-specific parameters
            extra = getattr(request, "extra", None)
            if extra and isinstance(extra, dict):
                api_params.update(extra)

            logger.debug(
                f"Calling OpenAI API with model {request.model}",
                extra={
                    "provider": self.name,
                    "model": request.model,
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens,
                    "message_count": len(request.messages),
                },
            )

            # Make API call
            response = await self.client.chat.completions.create(**api_params)

            # Extract response data
            content = response.choices[0].message.content or ""
            finish_reason = response.choices[0].finish_reason or "unknown"

            # Calculate usage and cost
            usage = {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            }

            cost = await self.estimate_cost(
                request.model,
                usage["prompt_tokens"],
                usage["completion_tokens"],
            )

            latency_ms = (time.time() - start_time) * 1000

            logger.info(
                "OpenAI completion successful",
                extra={
                    "provider": self.name,
                    "model": request.model,
                    "latency_ms": latency_ms,
                    "total_tokens": usage["total_tokens"],
                    "finish_reason": finish_reason,
                    "cost_usd": cost,
                },
            )

            return CompletionResponse(
                content=content,
                model=request.model,
                usage=usage,
                finish_reason=finish_reason,
                provider=self.name,
                latency_ms=latency_ms,
                cost_usd=cost,
            )

        except TimeoutError as e:
            logger.error(
                f"OpenAI API request timed out after {self.config.timeout}s",
                extra={
                    "provider": self.name,
                    "model": request.model,
                    "timeout": self.config.timeout,
                },
                exc_info=True,
            )
            raise ProviderTimeoutError(self.name, self.config.timeout) from e

        except Exception as e:
            if openai is not None:
                if isinstance(e, openai.AuthenticationError):
                    logger.error(
                        f"OpenAI authentication failed: {e}",
                        extra={"provider": self.name, "error": str(e)},
                        exc_info=True,
                    )
                    raise ProviderError(
                        "OpenAI authentication failed. Check your API key.",
                        details={"provider": self.name, "error": str(e)},
                    ) from e

                elif isinstance(e, openai.RateLimitError):
                    retry_after = getattr(e, "retry_after", None)
                    logger.warning(
                        "OpenAI rate limit exceeded",
                        extra={
                            "provider": self.name,
                            "model": request.model,
                            "retry_after": retry_after,
                        },
                    )
                    raise ProviderRateLimitError(self.name, retry_after) from e

                elif isinstance(e, openai.APIError):
                    logger.error(
                        f"OpenAI API error: {e}",
                        extra={
                            "provider": self.name,
                            "model": request.model,
                            "error": str(e),
                            "status_code": getattr(e, "status_code", None),
                        },
                        exc_info=True,
                    )
                    raise ProviderError(
                        f"OpenAI API error: {e}",
                        details={
                            "provider": self.name,
                            "model": request.model,
                            "error": str(e),
                            "status_code": getattr(e, "status_code", None),
                        },
                    ) from e

            logger.error(
                f"Unexpected error calling OpenAI: {e}",
                extra={
                    "provider": self.name,
                    "model": request.model,
                    "error": str(e),
                },
                exc_info=True,
            )
            raise ProviderError(
                f"Unexpected error calling OpenAI: {e}",
                details={"provider": self.name, "model": request.model, "error": str(e)},
            ) from e

    async def health_check(self) -> bool:
        """
        Check if OpenAI API is accessible.

        Returns:
            True if API is accessible, False otherwise
        """
        try:
            # Try to list models as a simple health check
            await asyncio.wait_for(self.client.models.list(), timeout=5.0)
            logger.debug("OpenAI health check passed", extra={"provider": self.name})
            return True
        except TimeoutError:
            logger.warning("OpenAI health check timed out", extra={"provider": self.name, "timeout": 5.0})
            return False
        except Exception as e:
            logger.warning(f"OpenAI health check failed: {e}", extra={"provider": self.name, "error": str(e)})
            return False

    async def close(self) -> None:
        """Clean up OpenAI client resources."""
        try:
            if hasattr(self.client, "close"):
                await self.client.close()
                logger.debug("Closed OpenAI client", extra={"provider": self.name})
        except Exception as e:
            logger.warning(f"Error closing OpenAI client: {e}", extra={"provider": self.name, "error": str(e)})
