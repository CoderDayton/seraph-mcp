"""
Anthropic Provider Implementation

Provides integration with Anthropic's API for Claude models.
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
    import anthropic
    from anthropic import AsyncAnthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    anthropic = None  # type: ignore[assignment]
    AsyncAnthropic = None  # type: ignore[assignment, misc]
    ANTHROPIC_AVAILABLE = False

from ..errors import DependencyError, ProviderError, ProviderRateLimitError, ProviderTimeoutError
from .base import (
    BaseProvider,
    CompletionRequest,
    CompletionResponse,
    ProviderConfig,
)

logger = logging.getLogger(__name__)


class AnthropicProvider(BaseProvider):
    """Anthropic (Claude) API provider implementation with dynamic model loading."""

    def __init__(self, config: ProviderConfig) -> None:
        """
        Initialize Anthropic provider.

        Args:
            config: Provider configuration

        Raises:
            DependencyError: If Anthropic SDK is not installed
            ValueError: If configuration is invalid
        """
        super().__init__(config)

        if not ANTHROPIC_AVAILABLE or AsyncAnthropic is None:
            logger.error("Anthropic SDK is not installed", extra={"package": "anthropic", "provider": "anthropic"})
            raise DependencyError(
                package="anthropic",
                feature="Anthropic provider",
                install_hint="pip install 'anthropic>=0.25.0'",
                details={"provider": "anthropic"},
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

            self.client = AsyncAnthropic(**client_kwargs)

            logger.info(
                "Anthropic provider initialized",
                extra={
                    "provider": "anthropic",
                    "base_url": config.base_url or "default",
                    "timeout": config.timeout,
                    "max_retries": config.max_retries,
                },
            )
        except Exception as e:
            logger.error(
                f"Failed to initialize Anthropic client: {e}",
                extra={"provider": "anthropic", "error": str(e)},
                exc_info=True,
            )
            raise RuntimeError(f"Failed to initialize Anthropic provider: {e}") from e

    def _validate_config(self) -> None:
        """
        Validate Anthropic-specific configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        if not self.config.api_key:
            logger.error("Anthropic API key is missing", extra={"provider": "anthropic"})
            raise ValueError("Anthropic API key is required. Set ANTHROPIC_API_KEY environment variable.")

        if len(self.config.api_key) < 20:
            logger.error(
                "Anthropic API key has invalid format",
                extra={"provider": "anthropic", "key_length": len(self.config.api_key)},
            )
            raise ValueError("Invalid Anthropic API key format. Anthropic keys should be at least 20 characters.")

    @property
    def name(self) -> str:
        """Return provider name."""
        return "anthropic"

    @property
    def models_dev_provider_id(self) -> str | None:
        """Return Models.dev provider ID."""
        return "anthropic"

    def _convert_messages(self, messages: list[dict[str, str]]) -> list[dict[str, Any]]:
        """Convert standard message format to Anthropic format."""
        # Anthropic expects messages in a specific format
        converted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Anthropic uses "user" and "assistant" roles
            if role == "system":
                # System messages are handled separately in Anthropic API
                continue

            converted.append(
                {
                    "role": role,
                    "content": content,
                }
            )

        return converted

    def _extract_system_message(self, messages: list[dict[str, str]]) -> str | None:
        """Extract system message from messages list."""
        for msg in messages:
            if msg.get("role") == "system":
                return msg.get("content", "")
        return None

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """
        Generate completion using Anthropic API.

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
            # Convert messages to Anthropic format
            converted_messages = self._convert_messages(request.messages)
            system_message = self._extract_system_message(request.messages)

            # Prepare API request
            api_params: dict[str, Any] = {
                "model": request.model,
                "messages": converted_messages,
                "temperature": request.temperature,
                "stream": request.stream,
                "max_tokens": request.max_tokens or 4096,  # Required by Anthropic
            }

            if system_message:
                api_params["system"] = system_message

            # Add any extra provider-specific parameters
            extra = getattr(request, "extra", None)
            if extra and isinstance(extra, dict):
                api_params.update(extra)

            logger.debug(
                f"Calling Anthropic API with model {request.model}",
                extra={
                    "provider": self.name,
                    "model": request.model,
                    "temperature": request.temperature,
                    "max_tokens": api_params["max_tokens"],
                    "message_count": len(converted_messages),
                    "has_system": bool(system_message),
                },
            )

            # Make API call
            response = await self.client.messages.create(**api_params)

            # Extract response data
            content = ""
            if response.content:
                # Anthropic returns content as a list of blocks
                for block in response.content:
                    if hasattr(block, "text"):
                        content += block.text

            finish_reason = response.stop_reason or "unknown"

            # Calculate usage and cost
            usage = {
                "prompt_tokens": response.usage.input_tokens if response.usage else 0,
                "completion_tokens": response.usage.output_tokens if response.usage else 0,
                "total_tokens": ((response.usage.input_tokens + response.usage.output_tokens) if response.usage else 0),
            }

            cost = await self.estimate_cost(
                request.model,
                usage["prompt_tokens"],
                usage["completion_tokens"],
            )

            latency_ms = (time.time() - start_time) * 1000

            logger.info(
                "Anthropic completion successful",
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
                f"Anthropic API request timed out after {self.config.timeout}s",
                extra={
                    "provider": self.name,
                    "model": request.model,
                    "timeout": self.config.timeout,
                },
                exc_info=True,
            )
            raise ProviderTimeoutError(self.name, self.config.timeout) from e

        except Exception as e:
            if anthropic is not None:
                if isinstance(e, anthropic.AuthenticationError):
                    logger.error(
                        f"Anthropic authentication failed: {e}",
                        extra={"provider": self.name, "error": str(e)},
                        exc_info=True,
                    )
                    raise ProviderError(
                        "Anthropic authentication failed. Check your API key.",
                        details={"provider": self.name, "error": str(e)},
                    ) from e

                elif isinstance(e, anthropic.RateLimitError):
                    retry_after = getattr(e, "retry_after", None)
                    logger.warning(
                        "Anthropic rate limit exceeded",
                        extra={
                            "provider": self.name,
                            "model": request.model,
                            "retry_after": retry_after,
                        },
                    )
                    raise ProviderRateLimitError(self.name, retry_after) from e

                elif isinstance(e, anthropic.APIError):
                    logger.error(
                        f"Anthropic API error: {e}",
                        extra={
                            "provider": self.name,
                            "model": request.model,
                            "error": str(e),
                            "status_code": getattr(e, "status_code", None),
                        },
                        exc_info=True,
                    )
                    raise ProviderError(
                        f"Anthropic API error: {e}",
                        details={
                            "provider": self.name,
                            "model": request.model,
                            "error": str(e),
                            "status_code": getattr(e, "status_code", None),
                        },
                    ) from e

            logger.error(
                f"Unexpected error calling Anthropic: {e}",
                extra={
                    "provider": self.name,
                    "model": request.model,
                    "error": str(e),
                },
                exc_info=True,
            )
            raise ProviderError(
                f"Unexpected error calling Anthropic: {e}",
                details={"provider": self.name, "model": request.model, "error": str(e)},
            ) from e

    async def health_check(self) -> bool:
        """
        Check if Anthropic API is accessible.

        Returns:
            True if API is accessible, False otherwise
        """
        try:
            # Use a very short max_tokens to minimize cost
            await asyncio.wait_for(
                self.client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=10,
                    messages=[{"role": "user", "content": "ping"}],
                ),
                timeout=5.0,
            )
            logger.debug("Anthropic health check passed", extra={"provider": self.name})
            return True
        except TimeoutError:
            logger.warning("Anthropic health check timed out", extra={"provider": self.name, "timeout": 5.0})
            return False
        except Exception as e:
            logger.warning(f"Anthropic health check failed: {e}", extra={"provider": self.name, "error": str(e)})
            return False

    async def close(self) -> None:
        """Clean up Anthropic client resources."""
        try:
            if hasattr(self.client, "close"):
                await self.client.close()
                logger.debug("Closed Anthropic client", extra={"provider": self.name})
        except Exception as e:
            logger.warning(f"Error closing Anthropic client: {e}", extra={"provider": self.name, "error": str(e)})
