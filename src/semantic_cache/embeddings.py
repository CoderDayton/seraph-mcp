"""
Semantic Cache Embedding Generator (Unified)

This module provides embeddings for semantic cache using the unified
provider-backed embedding service from context_optimization.

Initial Release (v1.0.0):
- Removed local (sentence-transformers) support to reduce dependencies
- All embeddings now go through provider-backed services
- Supported providers: openai, openai-compatible, gemini

For backward compatibility, this module wraps the unified ProviderEmbeddingService
and provides the same API that semantic_cache expects.

Per SDD.md:
- Unified embedding pathway across all modules
- Provider-backed architecture (no direct SDK coupling)
- Consistent auth/config/error handling
"""

import logging

from ..context_optimization.embeddings import (
    create_embedding_service,
    is_provider_available,
)
from ..providers import ProviderConfig

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Embedding generator for semantic cache.

    This is a compatibility wrapper around the unified ProviderEmbeddingService
    from context_optimization. It provides the API that semantic_cache expects
    while delegating to the unified service.

    Supported providers:
    - openai: OpenAI embeddings API
    - openai-compatible: OpenAI-compatible endpoints (Ollama, LM Studio, etc.)
    - gemini: Google Gemini embeddings

    Note: Local (sentence-transformers) support has been removed to reduce
    dependencies. Use openai-compatible with a local endpoint (Ollama, etc.)
    as an alternative.
    """

    def __init__(
        self,
        provider_name: str = "openai",
        model_name: str = "text-embedding-3-small",
        provider_config: ProviderConfig | None = None,
        cache_embeddings: bool = True,
    ):
        """
        Initialize embedding generator.

        Args:
            provider_name: Provider to use ('openai', 'openai-compatible', 'gemini')
            model_name: Model/endpoint to use
            provider_config: Provider configuration (API key, base URL, etc.)
            cache_embeddings: Cache embeddings in memory

        Raises:
            ValueError: If provider_name is 'local' (no longer supported)
            RuntimeError: If provider initialization fails
        """
        if provider_name.lower() == "local":
            raise ValueError(
                "Local (sentence-transformers) embeddings are no longer supported. "
                "Use 'openai-compatible' with a local endpoint (e.g., Ollama) instead. "
                "Example: provider_name='openai-compatible', "
                "provider_config=ProviderConfig(base_url='http://localhost:11434/v1', ...)"
            )

        if not provider_config:
            raise ValueError(f"Provider config required for {provider_name}")

        self.provider_name = provider_name.lower()
        self.model_name = model_name
        self.provider_config = provider_config
        self.cache_embeddings = cache_embeddings

        # Initialize the unified embedding service
        try:
            self._service = create_embedding_service(
                provider=self.provider_name,
                provider_config=self.provider_config,
                model=self.model_name,
                cache_embeddings=self.cache_embeddings,
            )
            logger.info(f"Semantic cache embedding generator initialized: {self.provider_name}/{self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding generator: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize embedding generator: {e}") from e

    async def generate(self, text: str) -> list[float]:
        """
        Generate embedding for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        return await self._service.embed_text(text)

    async def generate_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts (batch processing).

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        return await self._service.embed_texts(texts)

    def get_dimension(self) -> int:
        """
        Get embedding dimension for this provider.

        Returns:
            Embedding dimension
        """
        # Common embedding dimensions by provider/model
        if self.provider_name == "openai":
            dims = {
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072,
                "text-embedding-ada-002": 1536,
            }
            return dims.get(self.model_name, 1536)
        elif self.provider_name == "gemini":
            # Gemini text-embedding-004 defaults to 768
            return 768
        else:
            # For openai-compatible, dimensions vary by endpoint
            # Return a safe default
            return 384

    def clear_cache(self) -> None:
        """Clear embedding cache."""
        if self.cache_embeddings and hasattr(self._service, "_cache"):
            self._service._cache.clear()
            logger.info("Embedding cache cleared")


async def get_embedding_generator(
    provider_name: str = "openai",
    model_name: str = "text-embedding-3-small",
    provider_config: ProviderConfig | None = None,
    cache_embeddings: bool = True,
) -> EmbeddingGenerator:
    """
    Factory function to create embedding generator.

    Args:
        provider_name: Provider to use ('openai', 'openai-compatible', 'gemini')
        model_name: Model/endpoint name
        provider_config: Provider configuration
        cache_embeddings: Cache embeddings in memory

    Returns:
        EmbeddingGenerator instance
    """
    return EmbeddingGenerator(
        provider_name=provider_name,
        model_name=model_name,
        provider_config=provider_config,
        cache_embeddings=cache_embeddings,
    )


# Re-export for convenience
__all__ = [
    "EmbeddingGenerator",
    "get_embedding_generator",
    "is_provider_available",
]
