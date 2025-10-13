"""
Provider Factory

Manages provider instances and provides a unified interface for creating
and accessing AI model providers.

Per SDD.md:
- Minimal, clean implementation
- Singleton pattern for provider management
- Type-safe provider creation
- Comprehensive error handling
"""

import logging

from .anthropic_provider import AnthropicProvider
from .base import BaseProvider, ProviderConfig
from .gemini_provider import GeminiProvider
from .openai_compatible import OpenAICompatibleProvider
from .openai_provider import OpenAIProvider

logger = logging.getLogger(__name__)


class ProviderFactory:
    """
    Factory for creating and managing AI model providers.

    Implements singleton pattern to ensure providers are reused
    and properly managed throughout the application lifecycle.
    """

    # Registry of available providers
    _PROVIDERS: dict[str, type[BaseProvider]] = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "gemini": GeminiProvider,
        "openai-compatible": OpenAICompatibleProvider,
    }

    def __init__(self) -> None:
        """Initialize provider factory."""
        self._instances: dict[str, BaseProvider] = {}

    def create_provider(
        self,
        provider_name: str,
        config: ProviderConfig,
    ) -> BaseProvider:
        """
        Create or retrieve a provider instance.

        Args:
            provider_name: Name of the provider (e.g., "openai", "anthropic")
            config: Provider configuration

        Returns:
            Provider instance

        Raises:
            ValueError: If provider name is not supported
            RuntimeError: If provider initialization fails
        """
        # Normalize provider name
        provider_name = provider_name.lower().strip()

        # Check if provider is supported
        if provider_name not in self._PROVIDERS:
            supported = ", ".join(self._PROVIDERS.keys())
            raise ValueError(f"Unsupported provider: {provider_name}. Supported providers: {supported}")

        # Return cached instance if exists and is enabled
        cache_key = f"{provider_name}:{config.api_key[:8] if config.api_key else 'none'}"
        if cache_key in self._instances:
            instance = self._instances[cache_key]
            # Update enabled status
            instance.config.enabled = config.enabled
            return instance

        # Create new instance
        try:
            provider_class = self._PROVIDERS[provider_name]
            instance = provider_class(config)
            self._instances[cache_key] = instance

            logger.info(f"Created {provider_name} provider instance (enabled={config.enabled})")

            return instance

        except Exception as e:
            logger.error(f"Failed to create {provider_name} provider: {e}")
            raise RuntimeError(f"Failed to create {provider_name} provider: {e}") from e

    def get_provider(self, provider_name: str) -> BaseProvider | None:
        """
        Get an existing provider instance by name.

        Args:
            provider_name: Name of the provider

        Returns:
            Provider instance or None if not found
        """
        provider_name = provider_name.lower().strip()

        for cache_key, instance in self._instances.items():
            if cache_key.startswith(f"{provider_name}:"):
                return instance

        return None

    def list_providers(self) -> dict[str, BaseProvider]:
        """
        List all active provider instances.

        Returns:
            Dictionary of provider name -> instance
        """
        return {instance.name: instance for instance in self._instances.values() if instance.config.enabled}

    async def close_all(self) -> None:
        """Close all provider instances and clean up resources."""
        for provider in self._instances.values():
            try:
                await provider.close()
            except Exception as e:
                logger.warning(f"Error closing provider {provider.name}: {e}")

        self._instances.clear()
        logger.info("All providers closed")

    def reset(self) -> None:
        """Reset the factory (clear all instances without closing them)."""
        self._instances.clear()
        logger.info("Provider factory reset")

    @classmethod
    def get_supported_providers(cls) -> list[str]:
        """
        Get list of supported provider names.

        Returns:
            List of provider names
        """
        return list(cls._PROVIDERS.keys())


# Global factory instance
_factory: ProviderFactory | None = None


def get_factory() -> ProviderFactory:
    """
    Get the global provider factory instance.

    Returns:
        ProviderFactory singleton instance
    """
    global _factory

    if _factory is None:
        _factory = ProviderFactory()

    return _factory


def create_provider(provider_name: str, config: ProviderConfig) -> BaseProvider:
    """
    Convenience function to create a provider using the global factory.

    Args:
        provider_name: Name of the provider
        config: Provider configuration

    Returns:
        Provider instance
    """
    factory = get_factory()
    return factory.create_provider(provider_name, config)


def get_provider(provider_name: str) -> BaseProvider | None:
    """
    Convenience function to get a provider using the global factory.

    Args:
        provider_name: Name of the provider

    Returns:
        Provider instance or None
    """
    factory = get_factory()
    return factory.get_provider(provider_name)


def list_providers() -> dict[str, BaseProvider]:
    """
    Convenience function to list all providers.

    Returns:
        Dictionary of provider name -> instance
    """
    factory = get_factory()
    return factory.list_providers()


async def close_all_providers() -> None:
    """Convenience function to close all providers."""
    factory = get_factory()
    await factory.close_all()


def reset_factory() -> None:
    """Convenience function to reset the factory."""
    global _factory
    if _factory:
        _factory.reset()
    _factory = None
