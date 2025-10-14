"""
Semantic Cache Embedding Generator

Generates embeddings for semantic similarity search using the provider system.
Supports both API providers (OpenAI, Cohere) and local providers (Ollama, LM Studio).

Per SDD.md:
- Minimal, functional implementation
- Uses existing provider infrastructure
- Local and remote embedding support
- Caching for performance
"""

import logging
from typing import TYPE_CHECKING

from ..providers import ProviderConfig, create_provider

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

    from ..providers.base import BaseProvider

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generates embeddings using the provider system.

    Supports:
    - API providers: OpenAI, Cohere (with embeddings API)
    - Local providers: Ollama, LM Studio (via openai-compatible)
    - Sentence-transformers fallback
    """

    def __init__(
        self,
        provider_name: str = "local",
        model_name: str = "all-MiniLM-L6-v2",
        provider_config: ProviderConfig | None = None,
        cache_embeddings: bool = True,
    ):
        """
        Initialize embedding generator.

        Args:
            provider_name: Provider to use ('openai', 'openai-compatible', 'local')
            model_name: Model/endpoint to use
            provider_config: Provider configuration (API key, base URL, etc.)
            cache_embeddings: Cache embeddings in memory
        """
        self.provider_name = provider_name.lower()
        self.model_name = model_name
        self.provider_config = provider_config
        self.cache_embeddings = cache_embeddings
        self._cache: dict[str, list[float]] = {} if cache_embeddings else {}
        self._provider: BaseProvider | None = None
        self._local_model: SentenceTransformer | None = None

        self._initialize()

    def _initialize(self) -> None:
        """Initialize the embedding provider."""
        if self.provider_name == "local":
            self._initialize_local()
        else:
            self._initialize_provider()

    def _initialize_local(self) -> None:
        """Initialize local embedding model (sentence-transformers)."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers not installed. Install with: pip install sentence-transformers"
            ) from e

        logger.info(f"Loading local embedding model: {self.model_name}")
        try:
            self._local_model = SentenceTransformer(self.model_name)
            logger.info(f"Local embedding model loaded: {self.model_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to load local model {self.model_name}: {e}") from e

    def _initialize_provider(self) -> None:
        """Initialize provider for embeddings."""
        if not self.provider_config:
            raise ValueError(f"Provider config required for {self.provider_name}")

        try:
            self._provider = create_provider(self.provider_name, self.provider_config)
            logger.info(f"Embedding provider initialized: {self.provider_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize provider {self.provider_name}: {e}") from e

    def _get_from_cache(self, text: str) -> list[float] | None:
        """Get embedding from cache if available."""
        if not self.cache_embeddings:
            return None
        return self._cache.get(text)

    def _add_to_cache(self, text: str, embedding: list[float]) -> None:
        """Add embedding to cache."""
        if self.cache_embeddings:
            self._cache[text] = embedding

    async def generate(self, text: str) -> list[float]:
        """
        Generate embedding for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        # Check cache
        cached = self._get_from_cache(text)
        if cached is not None:
            return cached

        # Generate embedding
        if self.provider_name == "local":
            embedding = self._generate_local(text)
        else:
            embedding = await self._generate_via_provider(text)

        # Cache result
        self._add_to_cache(text, embedding)

        return embedding

    def _generate_local(self, text: str) -> list[float]:
        """Generate embedding using local sentence-transformers model."""
        if self._local_model is None:
            raise RuntimeError("Local model not initialized")
        try:
            embedding = self._local_model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            raise RuntimeError(f"Failed to generate local embedding: {e}") from e

    async def _generate_via_provider(self, text: str) -> list[float]:
        """Generate embedding using provider system."""
        if self.provider_name == "openai":
            return await self._generate_openai_embedding(text)
        elif self.provider_name == "openai-compatible":
            return await self._generate_compatible_embedding(text)
        else:
            raise ValueError(f"Unsupported provider for embeddings: {self.provider_name}")

    async def _generate_openai_embedding(self, text: str) -> list[float]:
        """Generate embedding using OpenAI embeddings API."""
        if self._provider is None or not hasattr(self._provider, "client"):
            raise RuntimeError("OpenAI provider not initialized")
        try:
            # Use OpenAI's embeddings endpoint directly
            response = await self._provider.client.embeddings.create(  # type: ignore[attr-defined]
                model=self.model_name,
                input=text,
            )
            return response.data[0].embedding
        except Exception as e:
            raise RuntimeError(f"Failed to generate OpenAI embedding: {e}") from e

    async def _generate_compatible_embedding(self, text: str) -> list[float]:
        """Generate embedding using OpenAI-compatible endpoint."""
        if self._provider is None or not hasattr(self._provider, "client"):
            raise RuntimeError("OpenAI-compatible provider not initialized")
        try:
            # OpenAI-compatible endpoints support embeddings too
            response = await self._provider.client.embeddings.create(  # type: ignore[attr-defined]
                model=self.model_name,
                input=text,
            )
            return response.data[0].embedding
        except Exception as e:
            raise RuntimeError(f"Failed to generate embedding via compatible endpoint: {e}") from e

    async def generate_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts (batch processing).

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if self.provider_name == "local":
            return self._generate_local_batch(texts)
        else:
            # For remote providers, could optimize with batch API calls
            embeddings = []
            for text in texts:
                embedding = await self.generate(text)
                embeddings.append(embedding)
            return embeddings

    def _generate_local_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings in batch using local model."""
        if self._local_model is None:
            raise RuntimeError("Local model not initialized")
        try:
            embeddings = self._local_model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()
        except Exception as e:
            raise RuntimeError(f"Failed to generate batch embeddings: {e}") from e

    def get_dimension(self) -> int:
        """
        Get embedding dimension for this provider.

        Returns:
            Embedding dimension
        """
        if self.provider_name == "local":
            if self._local_model is None:
                raise RuntimeError("Local model not initialized")
            dim = self._local_model.get_sentence_embedding_dimension()
            return dim if dim is not None else 384  # Default dimension
        elif self.provider_name == "openai":
            # OpenAI embedding dimensions
            dims = {
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072,
                "text-embedding-ada-002": 1536,
            }
            return dims.get(self.model_name, 1536)
        else:
            # For custom endpoints, we might need to query or assume
            return 384  # Safe default

    def clear_cache(self) -> None:
        """Clear embedding cache."""
        if self.cache_embeddings:
            self._cache.clear()
            logger.info("Embedding cache cleared")


async def get_embedding_generator(
    provider_name: str = "local",
    model_name: str = "all-MiniLM-L6-v2",
    provider_config: ProviderConfig | None = None,
    cache_embeddings: bool = True,
) -> EmbeddingGenerator:
    """
    Factory function to create embedding generator.

    Args:
        provider_name: Provider to use ('local', 'openai', 'openai-compatible')
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
