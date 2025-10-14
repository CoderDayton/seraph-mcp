"""
Provider-backed Embedding Service for Context Optimization

This module refactors context optimization embeddings to use the existing
providers factory, removing any direct SDK coupling from this layer.

Key points:
- No direct imports of vendor SDKs (OpenAI, Google GenAI, etc.).
- Uses the providers factory to create providers and leverage their underlying clients.
- Supports providers that expose an embeddings-compatible client surface.
  - OpenAI and OpenAI-compatible: client.embeddings.create(...)
  - Gemini: best-effort dynamic call via provider client if available
- Batch and single embedding helpers with optional in-memory caching.
- Cosine similarity helper for downstream quality/validation logic.

Expected provider behaviors (based on the existing providers module):
- OpenAIProvider/OpenAICompatibleProvider:
  - provider.client is an AsyncOpenAI client that exposes embeddings.create(...)
- GeminiProvider:
  - provider.client is a genai.Client; embeddings are accessible via embed_content on models.
  - This module avoids importing google-genai; instead, it accesses attributes dynamically.

If a provider does not support embeddings, this module will raise a clear error.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np

try:
    # Prefer importing via providers package root for stability (mirrors semantic_cache usage)
    from ..providers import ProviderConfig, create_provider
    from ..providers.base import BaseProvider
except Exception:
    # Fallback to direct module paths if the package root doesn't export these symbols
    from ..providers.base import BaseProvider, ProviderConfig
    from ..providers.factory import create_provider

logger = logging.getLogger(__name__)


# ---------- Utilities ------------------------------------------------------------


def cosine_similarity(vec1: list[float] | np.ndarray, vec2: list[float] | np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity score (0-1, higher = more similar)
    """
    v1 = np.array(vec1, dtype=float)
    v2 = np.array(vec2, dtype=float)

    # Normalize
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)

    if v1_norm == 0 or v2_norm == 0:
        return 0.0

    # Compute cosine similarity
    similarity = float(np.dot(v1, v2) / (v1_norm * v2_norm))
    # Clamp to [0, 1] for safety against tiny numeric drift
    return max(0.0, min(1.0, similarity))


# ---------- Provider-driven Embeddings -------------------------------------------


class ProviderEmbeddingService:
    """
    Provider-backed embedding generator.

    This adapter uses the providers factory to create a provider instance and
    then routes embedding requests through the provider's underlying client.

    Supported providers and methods (indirectly via provider.client):
    - openai, openai-compatible:
        await provider.client.embeddings.create(model=..., input=..., dimensions=optional)
    - gemini:
        response = provider.client.models.embed_content(model=..., content=..., config={...})
        (invoked asynchronously via asyncio.to_thread if sync)

    If a provider does not expose a compatible embeddings surface, a RuntimeError is raised.
    """

    def __init__(
        self,
        provider_name: str,
        model_name: str,
        provider_config: ProviderConfig,
        *,
        dimensions: int | None = None,
        task_type: str | None = None,
        cache_embeddings: bool = True,
    ) -> None:
        """
        Initialize a provider-backed embedding service.

        Args:
            provider_name: Provider identifier (e.g., 'openai', 'openai-compatible', 'gemini')
            model_name: Embedding model identifier/name for the provider
            provider_config: Provider configuration (API key, base URL, etc.)
            dimensions: Optional dimensionality reduction (provider-specific)
            task_type: Optional task type for providers that support it (e.g., Gemini)
            cache_embeddings: Enable simple in-memory caching
        """
        if not provider_name:
            raise ValueError("provider_name is required")
        if not model_name:
            raise ValueError("model_name is required")
        if provider_config is None or not isinstance(provider_config, ProviderConfig):
            raise ValueError("provider_config (ProviderConfig) is required")

        self.provider_name = provider_name.lower().strip()
        self.model_name = model_name
        self.dimensions = dimensions
        self.task_type = task_type
        self.cache_embeddings = cache_embeddings

        self._provider: BaseProvider = create_provider(self.provider_name, provider_config)
        self._cache: dict[str, list[float]] = {} if cache_embeddings else {}

        logger.info(
            "Embedding service initialized via providers factory",
            extra={
                "provider": self.provider_name,
                "model": self.model_name,
                "dimensions": self.dimensions or "default",
                "cache_enabled": self.cache_embeddings,
            },
        )

    # ----- Public API ------------------------------------------------------------

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors (floats)
        """
        if not texts:
            return []

        # Resolve cached first for reuse
        to_compute: list[tuple[int, str]] = []
        result: list[list[float] | None] = [None] * len(texts)

        for idx, text in enumerate(texts):
            cached = self._get_cache(text)
            if cached is not None:
                result[idx] = cached
            else:
                # Normalize empty/whitespace-only
                normalized = text if (text and text.strip()) else " "
                to_compute.append((idx, normalized))

        if not to_compute:
            # All texts were cached
            return [vec for vec in result if vec is not None]

        # Dispatch to provider-specific embedding pathway
        provider = self._provider.name
        if provider in ("openai", "openai-compatible"):
            computed = await self._embed_openai_batch([t for _, t in to_compute])
        elif provider == "gemini":
            computed = await self._embed_gemini_batch([t for _, t in to_compute])
        else:
            raise RuntimeError(
                f"Embeddings not supported for provider '{provider}'. "
                "Supported: 'openai', 'openai-compatible', 'gemini'."
            )

        # Merge computed results back and cache
        for (idx, original_text), embedding in zip(to_compute, computed, strict=False):
            result[idx] = embedding
            self._put_cache(original_text, embedding)

        # All positions should now be filled
        return [vec if vec is not None else [] for vec in result]

    async def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        cached = self._get_cache(text)
        if cached is not None:
            return cached

        vectors = await self.embed_texts([text])
        vector = vectors[0] if vectors else []
        self._put_cache(text, vector)
        return vector

    def similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Cosine similarity helper (0..1)."""
        return cosine_similarity(vec1, vec2)

    # ----- Internal helpers ------------------------------------------------------

    def _get_cache(self, text: str) -> list[float] | None:
        """Get embedding from cache if enabled."""
        if not self.cache_embeddings:
            return None
        return self._cache.get(text)

    def _put_cache(self, text: str, embedding: list[float]) -> None:
        """Insert embedding into cache if enabled."""
        if self.cache_embeddings:
            self._cache[text] = embedding

    # ----- Provider-specific embed paths ----------------------------------------

    async def _embed_openai_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed using OpenAI-style client (OpenAI or OpenAI-compatible).

        Relies on provider.client.embeddings.create(...) being available.
        """
        client = getattr(self._provider, "client", None)
        if client is None:
            raise RuntimeError("OpenAI-compatible provider client is not available")

        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "input": texts,
        }
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions

        # Call embeddings API; expected to be async on AsyncOpenAI
        try:
            response = await client.embeddings.create(**kwargs)
            # Extract embeddings in input order
            data = getattr(response, "data", None)
            if not data:
                raise RuntimeError("Embeddings response missing 'data'")
            embeddings = [item.embedding for item in data]
            if not embeddings or not isinstance(embeddings[0], list):
                raise RuntimeError("Embeddings response format unexpected")
            return embeddings
        except Exception as e:
            logger.error(
                "OpenAI-compatible embeddings call failed",
                extra={"provider": self._provider.name, "model": self.model_name, "error": str(e)},
                exc_info=True,
            )
            raise RuntimeError(f"OpenAI-compatible embeddings call failed: {e}") from e

    async def _embed_gemini_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed using a Gemini-style client via provider.client.

        Accessed dynamically to avoid direct SDK imports. Falls back to per-text calls.
        """
        client = getattr(self._provider, "client", None)
        if client is None:
            raise RuntimeError("Gemini provider client is not available")

        # Prefer client.models.embed_content(model=..., content=..., config=...) if available
        models_obj = getattr(client, "models", None)
        if models_obj is None or not hasattr(models_obj, "embed_content"):
            raise RuntimeError(
                "Gemini embeddings API is not available on provider client (missing models.embed_content)"
            )

        results: list[list[float]] = []
        for text in texts:
            try:
                # Build config dynamically without importing SDK types
                config: dict[str, Any] = {}
                if self.task_type:
                    config["task_type"] = self.task_type
                if self.dimensions:
                    # Name chosen to match google-genai's embed_content API parameter
                    config["output_dimensionality"] = self.dimensions

                # models.embed_content is synchronous in google-genai; run in thread to avoid blocking
                response = await asyncio.to_thread(
                    models_obj.embed_content,
                    model=self.model_name,
                    content=text,
                    config=config if config else None,
                )

                # Extract the first embedding vector
                embeddings_attr = getattr(response, "embeddings", None)
                if not embeddings_attr or not isinstance(embeddings_attr, list):
                    raise RuntimeError("Gemini embeddings response missing 'embeddings'")

                first = embeddings_attr[0]
                values = getattr(first, "values", None)
                if not values or not isinstance(values, list):
                    raise RuntimeError("Gemini embeddings response missing 'values' in first embedding")

                # Ensure list[float]
                vec: list[float] = [float(x) for x in values]
                results.append(vec)
            except Exception as e:
                logger.error(
                    "Gemini embedding generation failed",
                    extra={"provider": self._provider.name, "model": self.model_name, "error": str(e)},
                    exc_info=True,
                )
                raise RuntimeError(f"Gemini embedding generation failed: {e}") from e

        return results


# ---------- Factory and Availability --------------------------------------------


def create_embedding_service(
    provider: str,
    provider_config: ProviderConfig,
    *,
    model: str,
    dimensions: int | None = None,
    task_type: str | None = None,
    cache_embeddings: bool = True,
) -> ProviderEmbeddingService:
    """
    Factory function to create provider-backed embedding service.

    Args:
        provider: Provider name ('openai', 'openai-compatible', 'gemini')
        provider_config: ProviderConfig with authentication and endpoints
        model: Embedding model identifier/name
        dimensions: Optional dimensionality reduction parameter
        task_type: Optional task type (e.g., for Gemini)
        cache_embeddings: Cache embeddings in memory

    Returns:
        ProviderEmbeddingService instance
    """
    return ProviderEmbeddingService(
        provider_name=provider,
        model_name=model,
        provider_config=provider_config,
        dimensions=dimensions,
        task_type=task_type,
        cache_embeddings=cache_embeddings,
    )


def is_provider_available(provider: str) -> bool:
    """
    Lightweight availability check.

    This function intentionally avoids importing vendor SDKs. It simply
    acknowledges supported provider names for embedding via provider clients.

    Args:
        provider: Provider name

    Returns:
        True if provider is recognized for embeddings
    """
    return provider.lower() in {"openai", "openai-compatible", "gemini"}


# Backward-compatible export aliases (to avoid breaking older imports)
# Both alias to ProviderEmbeddingService; direct SDK classes are intentionally removed.
OpenAIEmbeddingService = ProviderEmbeddingService
GeminiEmbeddingService = ProviderEmbeddingService


__all__ = [
    "ProviderEmbeddingService",
    "OpenAIEmbeddingService",
    "GeminiEmbeddingService",
    "create_embedding_service",
    "is_provider_available",
    "cosine_similarity",
]
