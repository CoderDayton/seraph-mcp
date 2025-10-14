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

            logger.info(
                f"Created {provider_name} provider instance (enabled={config.enabled})",
                extra={"provider": provider_name, "enabled": config.enabled, "has_api_key": bool(config.api_key)},
            )

            return instance

        except KeyError as e:
            # This should never happen due to earlier check, but handle defensively
            logger.error(
                f"Provider class not found for {provider_name}",
                extra={"provider": provider_name, "error": str(e)},
                exc_info=True,
            )
            raise RuntimeError(f"Provider implementation not found: {provider_name}") from e
        except TypeError as e:
            logger.error(
                f"Invalid configuration for {provider_name} provider: {e}",
                extra={"provider": provider_name, "config_keys": list(config.model_dump().keys()), "error": str(e)},
                exc_info=True,
            )
            raise RuntimeError(f"Invalid configuration for {provider_name}: {e}") from e
        except Exception as e:
            logger.error(
                f"Failed to create {provider_name} provider: {e}",
                extra={"provider": provider_name, "error": str(e)},
                exc_info=True,
            )
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
        if not self._instances:
            logger.debug("No providers to close")
            return

        errors = []
        for provider in self._instances.values():
            try:
                await provider.close()
                logger.debug(f"Closed provider: {provider.name}")
            except Exception as e:
                error_msg = f"Error closing provider {provider.name}: {e}"
                logger.error(error_msg, extra={"provider": provider.name, "error": str(e)}, exc_info=True)
                errors.append(error_msg)

        self._instances.clear()

        if errors:
            logger.warning(f"Closed all providers with {len(errors)} error(s)")
        else:
            logger.info("All providers closed successfully")

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
