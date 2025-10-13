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

from .base import (
    BaseProvider,
    CompletionRequest,
    CompletionResponse,
    ProviderConfig,
)


class AnthropicProvider(BaseProvider):
    """Anthropic (Claude) API provider implementation with dynamic model loading."""

    def __init__(self, config: ProviderConfig) -> None:
        """Initialize Anthropic provider."""
        super().__init__(config)

        if not ANTHROPIC_AVAILABLE or AsyncAnthropic is None:
            raise RuntimeError("Anthropic SDK not available. Install with: pip install anthropic>=0.25.0")

        # Initialize async client
        client_kwargs: dict[str, Any] = {
            "api_key": config.api_key,
            "timeout": config.timeout,
            "max_retries": config.max_retries,
        }

        if config.base_url:
            client_kwargs["base_url"] = config.base_url

        self.client = AsyncAnthropic(**client_kwargs)

    def _validate_config(self) -> None:
        """Validate Anthropic-specific configuration."""
        if not self.config.api_key:
            raise ValueError("Anthropic API key is required")

        if len(self.config.api_key) < 20:
            raise ValueError("Invalid Anthropic API key format")

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
        """Generate completion using Anthropic API."""
        if not self.config.enabled:
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

            return CompletionResponse(
                content=content,
                model=request.model,
                usage=usage,
                finish_reason=finish_reason,
                provider=self.name,
                latency_ms=latency_ms,
                cost_usd=cost,
            )

        except Exception as e:
            if anthropic is not None:
                if isinstance(e, anthropic.AuthenticationError):
                    raise RuntimeError(f"Anthropic authentication failed: {e}") from e
                elif isinstance(e, anthropic.RateLimitError):
                    raise RuntimeError(f"Anthropic rate limit exceeded: {e}") from e
                elif isinstance(e, anthropic.APIError):
                    raise RuntimeError(f"Anthropic API error: {e}") from e
            raise RuntimeError(f"Unexpected error calling Anthropic: {e}") from e

    async def health_check(self) -> bool:
        """Check if Anthropic API is accessible."""
        try:
            # Use a very short max_tokens to minimize cost
            await self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=10,
                messages=[{"role": "user", "content": "ping"}],
            )
            return True
        except Exception:
            return False

    async def close(self) -> None:
        """Clean up Anthropic client resources."""
        if hasattr(self.client, "close"):
            await self.client.close()
