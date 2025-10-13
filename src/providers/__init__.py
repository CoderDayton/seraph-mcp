"""
Providers Module

Provides unified interface for AI model providers (OpenAI, Anthropic, Gemini, etc.).

This module implements a clean provider abstraction that allows Seraph MCP to
work with multiple AI model providers through a consistent interface.

Per SDD.md:
- Minimal, clean implementation
- Type-safe provider creation
- Singleton factory pattern
- Comprehensive error handling

Public API:
    - BaseProvider: Abstract base class for all providers
    - ProviderConfig: Base configuration for providers
    - ModelInfo: Model metadata
    - CompletionRequest: Standardized request format
    - CompletionResponse: Standardized response format

    Provider implementations:
    - OpenAIProvider: OpenAI GPT models
    - AnthropicProvider: Anthropic Claude models
    - GeminiProvider: Google Gemini models
    - OpenAICompatibleProvider: Custom OpenAI-compatible endpoints

    Factory functions:
    - create_provider(): Create a provider instance
    - get_provider(): Get existing provider instance
    - list_providers(): List all active providers
    - get_factory(): Get the global factory instance
    - close_all_providers(): Clean up all providers
    - reset_factory(): Reset the factory state

Usage:
    >>> from src.providers import create_provider, ProviderConfig
    >>>
    >>> # Create OpenAI provider
    >>> config = ProviderConfig(api_key="sk-...", timeout=30.0)
    >>> provider = create_provider("openai", config)
    >>>
    >>> # Make a completion request
    >>> from src.providers import CompletionRequest
    >>> request = CompletionRequest(
    ...     model="gpt-4",
    ...     messages=[{"role": "user", "content": "Hello!"}],
    ...     temperature=0.7,
    ... )
    >>> response = await provider.complete(request)
    >>> print(response.content)
"""

# Provider implementations
from .anthropic_provider import AnthropicProvider
from .base import (
    BaseProvider,
    CompletionRequest,
    CompletionResponse,
    ModelInfo,
    ProviderConfig,
)
from .factory import (
    ProviderFactory,
    close_all_providers,
    create_provider,
    get_factory,
    get_provider,
    list_providers,
    reset_factory,
)
from .gemini_provider import GeminiProvider
from .models_dev import (
    ModelsDevClient,
    close_models_dev_client,
    get_models_dev_client,
)
from .openai_compatible import OpenAICompatibleProvider
from .openai_provider import OpenAIProvider

__all__ = [
    # Base classes and models
    "BaseProvider",
    "ProviderConfig",
    "ModelInfo",
    "CompletionRequest",
    "CompletionResponse",
    # Provider implementations
    "OpenAIProvider",
    "AnthropicProvider",
    "GeminiProvider",
    "OpenAICompatibleProvider",
    # Factory and utilities
    "ProviderFactory",
    "create_provider",
    "get_provider",
    "list_providers",
    "get_factory",
    "close_all_providers",
    "reset_factory",
    # Models.dev client
    "ModelsDevClient",
    "get_models_dev_client",
    "close_models_dev_client",
]
