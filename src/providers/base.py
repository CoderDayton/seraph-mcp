"""
Base Provider Interface

Defines the abstract interface for AI model providers.
All concrete providers (OpenAI, Anthropic, Gemini, etc.) must implement this interface.

Per SDD.md:
- Minimal, clean implementation
- Typed with Pydantic
- Async-first design
- Comprehensive error handling
- Dynamic model loading from Models.dev API
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .models_dev import get_models_dev_client, ModelInfo as ModelsDevModelInfo


class ProviderConfig(BaseModel):
    """Base configuration for all providers."""

    api_key: str = Field(..., description="API key for the provider")
    base_url: Optional[str] = Field(None, description="Custom base URL (optional)")
    timeout: float = Field(30.0, ge=1.0, description="Request timeout in seconds")
    max_retries: int = Field(3, ge=0, description="Maximum retry attempts")
    enabled: bool = Field(True, description="Whether this provider is enabled")


class ModelInfo(BaseModel):
    """Information about a specific model (converted from Models.dev format)."""

    model_id: str = Field(..., description="Unique model identifier")
    display_name: str = Field(..., description="Human-readable model name")
    context_window: int = Field(..., description="Maximum context window in tokens")
    max_output_tokens: int = Field(..., description="Maximum output tokens")
    supports_streaming: bool = Field(True, description="Whether model supports streaming")
    input_cost_per_1k: float = Field(..., description="Cost per 1K input tokens (USD)")
    output_cost_per_1k: float = Field(..., description="Cost per 1K output tokens (USD)")
    capabilities: List[str] = Field(
        default_factory=list, description="Model capabilities (e.g., 'chat', 'completion', 'vision')"
    )

    @classmethod
    def from_models_dev(cls, model_info: ModelsDevModelInfo) -> "ModelInfo":
        """
        Convert Models.dev ModelInfo to provider ModelInfo.

        Args:
            model_info: Models.dev model information

        Returns:
            Provider ModelInfo instance
        """
        # Determine capabilities from Models.dev flags
        capabilities = []
        if "text" in model_info.modalities.input:
            capabilities.append("chat")
        if "image" in model_info.modalities.input:
            capabilities.append("vision")
        if model_info.tool_call:
            capabilities.append("function_calling")
        if model_info.reasoning:
            capabilities.append("reasoning")

        # Convert cost from per-million to per-1K (handle optional cost)
        if model_info.cost:
            input_cost_per_1k = model_info.cost.input / 1000
            output_cost_per_1k = model_info.cost.output / 1000
        else:
            # Default to $0 if no cost information available
            input_cost_per_1k = 0.0
            output_cost_per_1k = 0.0

        return cls(
            model_id=model_info.id,
            display_name=model_info.name,
            context_window=model_info.limit.context,
            max_output_tokens=model_info.limit.output,
            supports_streaming=True,  # Assume all models support streaming
            input_cost_per_1k=input_cost_per_1k,
            output_cost_per_1k=output_cost_per_1k,
            capabilities=capabilities,
        )


class CompletionRequest(BaseModel):
    """Standardized completion request."""

    model: str = Field(..., description="Model identifier")
    messages: List[Dict[str, str]] = Field(..., description="Chat messages")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")
    stream: bool = Field(False, description="Whether to stream the response")

    class Config:
        extra = "allow"  # Allow provider-specific parameters


class CompletionResponse(BaseModel):
    """Standardized completion response."""

    content: str = Field(..., description="Generated content")
    model: str = Field(..., description="Model used")
    usage: Dict[str, int] = Field(..., description="Token usage statistics")
    finish_reason: str = Field(..., description="Why generation stopped")
    provider: str = Field(..., description="Provider name")
    latency_ms: float = Field(..., description="Request latency in milliseconds")
    cost_usd: float = Field(..., description="Estimated cost in USD")


class BaseProvider(ABC):
    """
    Abstract base class for AI model providers.

    All providers must implement this interface to ensure consistency
    and enable intelligent routing across multiple providers.
    """

    def __init__(self, config: ProviderConfig) -> None:
        """
        Initialize the provider.

        Args:
            config: Provider configuration
        """
        self.config = config
        self._models_dev_client = get_models_dev_client()
        self._validate_config()

    @abstractmethod
    def _validate_config(self) -> None:
        """
        Validate provider-specific configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider name (e.g., 'openai', 'anthropic')."""
        pass

    @property
    def models_dev_provider_id(self) -> Optional[str]:
        """
        Return the Models.dev provider ID for this provider.

        Override this to map provider names to Models.dev provider IDs.
        Returns None if provider doesn't map to Models.dev.
        """
        return None

    @abstractmethod
    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """
        Generate a completion for the given request.

        Args:
            request: Standardized completion request

        Returns:
            Standardized completion response

        Raises:
            ValueError: If request is invalid
            RuntimeError: If API call fails
        """
        pass

    async def list_models(self) -> List[ModelInfo]:
        """
        List all available models for this provider.

        Default implementation fetches from Models.dev API.
        Override if provider has custom model listing logic.

        Returns:
            List of model information
        """
        if not self.models_dev_provider_id:
            return []

        try:
            provider = await self._models_dev_client.get_provider(self.models_dev_provider_id)
            if not provider:
                return []

            return [
                ModelInfo.from_models_dev(model_info)
                for model_info in provider.models.values()
            ]
        except Exception as e:
            # Fallback to empty list on error
            return []

    async def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """
        Get information about a specific model.

        Default implementation fetches from Models.dev API.
        Override if provider has custom model info logic.

        Args:
            model_id: Model identifier

        Returns:
            Model information or None if not found
        """
        if not self.models_dev_provider_id:
            return None

        try:
            model_info = await self._models_dev_client.get_model(
                self.models_dev_provider_id, model_id
            )
            if not model_info:
                return None

            return ModelInfo.from_models_dev(model_info)
        except Exception:
            return None

    async def estimate_cost(
        self, model_id: str, input_tokens: int, output_tokens: int
    ) -> float:
        """
        Estimate the cost for a completion request.

        Default implementation uses Models.dev pricing.
        Override if provider has custom pricing logic.

        Args:
            model_id: Model identifier
            input_tokens: Number of input tokens
            output_tokens: Estimated output tokens

        Returns:
            Estimated cost in USD
        """
        if not self.models_dev_provider_id:
            return 0.0

        try:
            return await self._models_dev_client.estimate_cost(
                self.models_dev_provider_id,
                model_id,
                input_tokens,
                output_tokens,
            )
        except Exception:
            return 0.0

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the provider is healthy and accessible.

        Returns:
            True if healthy, False otherwise
        """
        pass

    async def close(self) -> None:
        """Clean up resources. Override if needed."""
        pass

    def __repr__(self) -> str:
        """Return string representation."""
        return f"{self.__class__.__name__}(name={self.name}, enabled={self.config.enabled})"
