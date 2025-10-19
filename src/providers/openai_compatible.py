"""
OpenAI-Compatible Provider Implementation

Provides integration with OpenAI-compatible API endpoints (e.g., LM Studio, Ollama, vLLM, etc.).
This provider uses the OpenAI SDK but allows custom base URLs.
Uses Models.dev API to discover all OpenAI-compatible providers and their models.

Per SDD.md:
- Minimal, clean implementation
- Typed with Pydantic
- Async-first design
- Comprehensive error handling
"""

import logging
import time
from typing import Any

from .base import (
    BaseProvider,
    CompletionRequest,
    CompletionResponse,
    ModelInfo,
    ProviderConfig,
)

try:
    import openai
    from openai import AsyncOpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    openai = None  # type: ignore[assignment]
    AsyncOpenAI = None  # type: ignore[assignment, misc]
    OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)


class OpenAICompatibleProvider(BaseProvider):
    """
    OpenAI-compatible API provider implementation.

    Works with any service that implements the OpenAI API specification,
    such as LM Studio, Ollama, vLLM, and others.
    """

    def __init__(self, config: ProviderConfig) -> None:
        """Initialize OpenAI-compatible provider."""
        super().__init__(config)

        if not OPENAI_AVAILABLE or AsyncOpenAI is None:
            raise RuntimeError("OpenAI SDK not available. Install with: pip install openai>=1.0.0")

        if not config.base_url:
            raise ValueError("base_url is required for OpenAI-compatible provider. Example: http://localhost:1234/v1")

        # Initialize async client with custom base URL
        client_kwargs: dict[str, Any] = {
            "api_key": config.api_key or "not-needed",  # Some services don't require API keys
            "base_url": config.base_url,
            "timeout": config.timeout,
            "max_retries": config.max_retries,
        }

        self.client = AsyncOpenAI(**client_kwargs)
        self._models_cache: list[ModelInfo] | None = None

    def _validate_config(self) -> None:
        """Validate OpenAI-compatible provider configuration."""
        if not self.config.base_url:
            raise ValueError("base_url is required for OpenAI-compatible provider")

        if not self.config.base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")

    @property
    def name(self) -> str:
        """Return provider name."""
        return "openai-compatible"

    @property
    def models_dev_provider_id(self) -> str | None:
        """
        Return Models.dev provider ID based on base_url.

        Attempts to match the base_url to a known provider in Models.dev.
        """
        if not self.config.base_url:
            return None

        # Store provider ID if we've already determined it
        if hasattr(self, "_cached_provider_id"):
            return self._cached_provider_id

        return None

    async def _discover_provider_id(self) -> str | None:
        """
        Discover the Models.dev provider ID by matching base_url.

        Returns:
            Provider ID or None if not found
        """
        if not self.config.base_url:
            return None

        try:
            # Load all providers from Models.dev
            providers = await self._models_dev_client.load_providers()

            # Try to match base URL to a provider
            base_url_lower = self.config.base_url.lower().rstrip("/")

            for provider_id, provider in providers.items():
                provider_api = provider.api.lower().rstrip("/")

                # Check if URLs match or are similar
                if base_url_lower == provider_api or base_url_lower in provider_api or provider_api in base_url_lower:
                    self._cached_provider_id = provider_id
                    return provider_id

            return None
        except Exception:
            return None

    async def _get_token_costs(self, model_id: str) -> tuple[float, float]:
        """
        Get input and output token costs per 1k tokens for a model.

        Returns:
            Tuple of (input_cost_per_1k, output_cost_per_1k)
        """
        # Try to get model info which contains pricing
        model_info = await self.get_model_info(model_id)
        if model_info:
            return (model_info.input_cost_per_1k, model_info.output_cost_per_1k)

        # Fallback to zero costs if model info unavailable
        return (0.0, 0.0)

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Generate completion using OpenAI-compatible API."""
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

            # Determine timeout: use request timeout if provided, otherwise provider default
            timeout = request.timeout if request.timeout is not None else self.config.timeout

            # Make API call with timeout
            import asyncio

            response = await asyncio.wait_for(self.client.chat.completions.create(**api_params), timeout=timeout)

            # Extract response data
            content = response.choices[0].message.content or ""
            finish_reason = response.choices[0].finish_reason or "unknown"

            # Calculate usage (some services may not provide this)
            usage = {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            }

            # Cost estimation for custom endpoints
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
                    raise RuntimeError(f"Authentication failed: {e}") from e
                elif isinstance(e, openai.APIConnectionError):
                    raise RuntimeError(f"Connection failed to {self.config.base_url}: {e}") from e
                elif isinstance(e, openai.APIError):
                    raise RuntimeError(f"API error: {e}") from e
            raise RuntimeError(f"Unexpected error calling OpenAI-compatible API: {e}") from e

    async def list_models(self) -> list[ModelInfo]:
        """
        List all available models from the OpenAI-compatible endpoint.

        First tries to get models from Models.dev API based on provider detection.
        Falls back to querying the endpoint's /v1/models if Models.dev doesn't have the provider.
        """
        if self._models_cache:
            return self._models_cache

        # Try to get models from Models.dev first
        provider_id = await self._discover_provider_id()
        if provider_id:
            try:
                provider = await self._models_dev_client.get_provider(provider_id)
                if provider and provider.models:
                    models = [ModelInfo.from_models_dev(model_info) for model_info in provider.models.values()]
                    self._models_cache = models
                    return models
            except Exception as e:
                logger.debug(f"Failed to fetch models from Models.dev for provider {provider_id}: {e}")

        # Fallback: try to fetch models from the endpoint directly
        try:
            models_response = await self.client.models.list()

            models = []
            for model in models_response.data:
                # Create ModelInfo with default/estimated values
                # since OpenAI-compatible APIs may not provide full details
                model_info = ModelInfo(
                    model_id=model.id,
                    display_name=model.id,
                    context_window=4096,  # Default assumption
                    max_output_tokens=2048,  # Default assumption
                    supports_streaming=True,
                    input_cost_per_1k=0.0,
                    output_cost_per_1k=0.0,
                    capabilities=["chat"],
                )
                models.append(model_info)

            self._models_cache = models
            return models

        except Exception:
            # If we can't fetch models, return empty list
            return []

    async def get_model_info(self, model_id: str) -> ModelInfo | None:
        """
        Get information about a specific model.

        First tries Models.dev, then checks endpoint's model list,
        then returns conservative defaults.
        """
        # Try Models.dev first
        provider_id = await self._discover_provider_id()
        if provider_id:
            try:
                model_info = await self._models_dev_client.get_model(provider_id, model_id)
                if model_info:
                    return ModelInfo.from_models_dev(model_info)
            except Exception as e:
                logger.debug(f"Failed to get model info from Models.dev for {model_id}: {e}")

        # Try from cached/fetched models
        models = await self.list_models()
        for model in models:
            if model.model_id == model_id:
                return model

        # Return a default ModelInfo for unknown models
        return ModelInfo(
            model_id=model_id,
            display_name=model_id,
            context_window=4096,
            max_output_tokens=2048,
            supports_streaming=True,
            input_cost_per_1k=0.0,
            output_cost_per_1k=0.0,
            capabilities=["chat"],
        )

    async def estimate_cost(self, model_id: str, input_tokens: int, output_tokens: int) -> float:
        """
        Estimate cost for API call.

        Uses Models.dev pricing if available, otherwise assumes free.
        """
        # Try Models.dev pricing first
        provider_id = await self._discover_provider_id()
        if provider_id:
            try:
                cost = await self._models_dev_client.estimate_cost(provider_id, model_id, input_tokens, output_tokens)
                if cost > 0:
                    return cost
            except Exception as e:
                logger.debug(f"Failed to estimate cost from Models.dev: {e}")

        # Fallback: assumes free for undetected providers
        return 0.0

    async def health_check(self) -> bool:
        """Check if the OpenAI-compatible API is accessible."""
        try:
            # Try to list models as a simple health check
            await self.client.models.list()
            return True
        except Exception:
            return False

    async def close(self) -> None:
        """Clean up client resources."""
        if hasattr(self.client, "close"):
            await self.client.close()
