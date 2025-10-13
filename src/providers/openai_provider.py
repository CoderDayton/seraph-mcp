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

from .base import (
    BaseProvider,
    CompletionRequest,
    CompletionResponse,
    ProviderConfig,
)


class OpenAIProvider(BaseProvider):
    """OpenAI API provider implementation with dynamic model loading."""

    def __init__(self, config: ProviderConfig) -> None:
        """Initialize OpenAI provider."""
        super().__init__(config)

        if not OPENAI_AVAILABLE or AsyncOpenAI is None:
            raise RuntimeError("OpenAI SDK not available. Install with: pip install openai>=1.0.0")

        # Initialize async client
        client_kwargs: dict[str, Any] = {
            "api_key": config.api_key,
            "timeout": config.timeout,
            "max_retries": config.max_retries,
        }

        if config.base_url:
            client_kwargs["base_url"] = config.base_url

        self.client = AsyncOpenAI(**client_kwargs)

    def _validate_config(self) -> None:
        """Validate OpenAI-specific configuration."""
        if not self.config.api_key:
            raise ValueError("OpenAI API key is required")

        if len(self.config.api_key) < 20:
            raise ValueError("Invalid OpenAI API key format")

    @property
    def name(self) -> str:
        """Return provider name."""
        return "openai"

    @property
    def models_dev_provider_id(self) -> str | None:
        """Return Models.dev provider ID."""
        return "openai"

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Generate completion using OpenAI API."""
        if not self.config.enabled:
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
            if openai is not None:
                if isinstance(e, openai.AuthenticationError):
                    raise RuntimeError(f"OpenAI authentication failed: {e}") from e
                elif isinstance(e, openai.RateLimitError):
                    raise RuntimeError(f"OpenAI rate limit exceeded: {e}") from e
                elif isinstance(e, openai.APIError):
                    raise RuntimeError(f"OpenAI API error: {e}") from e
            raise RuntimeError(f"Unexpected error calling OpenAI: {e}") from e

    async def health_check(self) -> bool:
        """Check if OpenAI API is accessible."""
        try:
            # Try to list models as a simple health check
            await asyncio.wait_for(self.client.models.list(), timeout=5.0)
            return True
        except Exception:
            return False

    async def close(self) -> None:
        """Clean up OpenAI client resources."""
        if hasattr(self.client, "close"):
            await self.client.close()
