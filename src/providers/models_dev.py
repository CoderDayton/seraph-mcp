"""
Models.dev API Client

Provides dynamic model and provider information from models.dev API.
This eliminates hardcoded model data and keeps pricing/capabilities up-to-date.

Per SDD.md:
- Minimal, clean implementation
- Cached responses to avoid excessive API calls
- Typed with Pydantic
- Comprehensive error handling
"""

import logging
import time
from typing import Any

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ModelCost(BaseModel):
    """Cost information for a model."""

    input: float = Field(0.0, description="Input cost per million tokens (USD)")
    output: float = Field(0.0, description="Output cost per million tokens (USD)")
    cache_read: float | None = Field(None, description="Cache read cost per million tokens (USD)")
    cache_write: float | None = Field(None, description="Cache write cost per million tokens (USD)")


class ModelLimit(BaseModel):
    """Context and output limits for a model."""

    context: int = Field(..., description="Maximum context window in tokens")
    output: int = Field(..., description="Maximum output tokens")


class ModelModalities(BaseModel):
    """Input/output modalities supported by model."""

    input: list[str] = Field(default_factory=list, description="Input modalities (text, image, audio, video)")
    output: list[str] = Field(default_factory=list, description="Output modalities (text, image, audio)")


class ModelInfo(BaseModel):
    """Information about a specific model from models.dev."""

    id: str = Field(..., description="Model identifier")
    name: str = Field(..., description="Human-readable model name")
    attachment: bool = Field(False, description="Supports file attachments")
    reasoning: bool = Field(False, description="Has reasoning capabilities")
    temperature: bool = Field(True, description="Supports temperature parameter")
    tool_call: bool = Field(False, description="Supports function/tool calling")
    knowledge: str | None = Field(None, description="Knowledge cutoff date")
    release_date: str | None = Field(None, description="Release date")
    last_updated: str | None = Field(None, description="Last updated date")
    modalities: ModelModalities = Field(default_factory=ModelModalities, description="Supported modalities")
    open_weights: bool = Field(False, description="Model has open weights")
    cost: ModelCost | None = Field(None, description="Pricing information (optional)")
    limit: ModelLimit = Field(..., description="Context and output limits")


class ProviderInfo(BaseModel):
    """Information about a provider from models.dev."""

    id: str = Field(..., description="Provider identifier")
    name: str = Field(..., description="Provider display name")
    api: str = Field(..., description="API base URL")
    env: list[str] = Field(default_factory=list, description="Required environment variables")
    npm: str = Field(..., description="NPM package name")
    doc: str | None = Field(None, description="Documentation URL")
    models: dict[str, ModelInfo] = Field(default_factory=dict, description="Available models")


class ModelsDevClient:
    """
    Client for models.dev API.

    Fetches and caches provider and model information dynamically.
    """

    API_URL = "https://models.dev/api.json"
    CACHE_TTL_SECONDS = 3600  # Cache for 1 hour

    def __init__(self) -> None:
        """Initialize Models.dev client."""
        self._cache: dict[str, ProviderInfo] | None = None
        self._cache_time: float = 0.0
        self._client = httpx.AsyncClient(timeout=30.0)

    async def _fetch_data(self) -> dict[str, Any]:
        """
        Fetch raw data from models.dev API.

        Returns:
            Raw API response data

        Raises:
            RuntimeError: If API call fails
        """
        try:
            response = await self._client.get(self.API_URL)
            response.raise_for_status()
            return response.json()  # type: ignore[no-any-return]
        except httpx.HTTPError as e:
            raise RuntimeError(f"Failed to fetch models.dev API: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error fetching models.dev API: {e}") from e

    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid."""
        if self._cache is None:
            return False
        elapsed = time.time() - self._cache_time
        return elapsed < self.CACHE_TTL_SECONDS

    async def load_providers(self, force_refresh: bool = False) -> dict[str, ProviderInfo]:
        """
        Load all providers and their models.

        Args:
            force_refresh: Force refresh even if cache is valid

        Returns:
            Dictionary of provider_id -> ProviderInfo

        Raises:
            RuntimeError: If loading fails
        """
        if not force_refresh and self._is_cache_valid():
            if self._cache is not None:
                return self._cache

        logger.info("Fetching provider data from models.dev API...")

        try:
            raw_data = await self._fetch_data()

            providers = {}
            for provider_id, provider_data in raw_data.items():
                # Parse models
                models = {}
                for model_id, model_data in provider_data.get("models", {}).items():
                    try:
                        models[model_id] = ModelInfo(**model_data)
                    except Exception as e:
                        logger.warning(f"Failed to parse model {model_id}: {e}")
                        continue

                # Parse provider
                try:
                    provider_info = ProviderInfo(
                        id=provider_data.get("id", provider_id),
                        name=provider_data.get("name", provider_id),
                        api=provider_data.get("api", ""),
                        env=provider_data.get("env", []),
                        npm=provider_data.get("npm", ""),
                        doc=provider_data.get("doc"),
                        models=models,
                    )
                    providers[provider_id] = provider_info
                except Exception as e:
                    logger.warning(f"Failed to parse provider {provider_id}: {e}")
                    continue

            self._cache = providers
            self._cache_time = time.time()

            logger.info(
                f"Loaded {len(providers)} providers with {sum(len(p.models) for p in providers.values())} models"
            )

            return providers

        except Exception as e:
            logger.error(f"Failed to load providers: {e}")
            # Return cached data if available, even if expired
            if self._cache:
                logger.warning("Returning stale cache due to fetch failure")
                return self._cache
            raise

    async def get_provider(self, provider_id: str) -> ProviderInfo | None:
        """
        Get information about a specific provider.

        Args:
            provider_id: Provider identifier

        Returns:
            ProviderInfo or None if not found
        """
        providers = await self.load_providers()
        return providers.get(provider_id)

    async def get_model(self, provider_id: str, model_id: str) -> ModelInfo | None:
        """
        Get information about a specific model.

        Args:
            provider_id: Provider identifier
            model_id: Model identifier

        Returns:
            ModelInfo or None if not found
        """
        provider = await self.get_provider(provider_id)
        if not provider:
            return None
        return provider.models.get(model_id)

    async def search_model(self, model_id: str) -> tuple[str, ModelInfo] | None:
        """
        Search for a model across all providers.

        Args:
            model_id: Model identifier to search for

        Returns:
            Tuple of (provider_id, ModelInfo) or None if not found
        """
        providers = await self.load_providers()

        for provider_id, provider in providers.items():
            if model_id in provider.models:
                return (provider_id, provider.models[model_id])

        return None

    async def list_all_models(self) -> list[tuple[str, str, ModelInfo]]:
        """
        List all available models across all providers.

        Returns:
            List of (provider_id, model_id, ModelInfo) tuples
        """
        providers = await self.load_providers()

        models = []
        for provider_id, provider in providers.items():
            for model_id, model_info in provider.models.items():
                models.append((provider_id, model_id, model_info))

        return models

    async def get_models_by_provider_type(self, provider_type: str) -> dict[str, ModelInfo]:
        """
        Get all models for providers matching a type (e.g., 'openai', 'anthropic').

        Args:
            provider_type: Provider type to match (case-insensitive)

        Returns:
            Dictionary of model_id -> ModelInfo
        """
        providers = await self.load_providers()
        provider_type_lower = provider_type.lower()

        models = {}
        for provider_id, provider in providers.items():
            # Match provider type (e.g., 'openai' matches 'openai', 'openai-compatible', etc.)
            if provider_type_lower in provider_id.lower() or provider_type_lower in provider.name.lower():
                models.update(provider.models)

        return models

    async def estimate_cost(
        self,
        provider_id: str,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """
        Estimate cost for a model request.

        Args:
            provider_id: Provider identifier
            model_id: Model identifier
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Estimated cost in USD
        """
        model = await self.get_model(provider_id, model_id)
        if not model or not model.cost:
            return 0.0

        # Models.dev costs are per million tokens
        input_cost = (input_tokens / 1_000_000) * model.cost.input
        output_cost = (output_tokens / 1_000_000) * model.cost.output

        return input_cost + output_cost

    async def close(self) -> None:
        """Clean up client resources."""
        await self._client.aclose()


# Global singleton instance
_client: ModelsDevClient | None = None


def get_models_dev_client() -> ModelsDevClient:
    """
    Get the global Models.dev client instance.

    Returns:
        ModelsDevClient singleton
    """
    global _client

    if _client is None:
        _client = ModelsDevClient()

    return _client


async def close_models_dev_client() -> None:
    """Close the global Models.dev client."""
    global _client

    if _client:
        await _client.close()
        _client = None
