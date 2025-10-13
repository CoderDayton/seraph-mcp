"""
Semantic Cache Implementation

Minimal ChromaDB-backed semantic cache with vector similarity search.
Uses the provider system for embeddings.

Per SDD.md:
- Minimal, functional implementation
- ChromaDB for vector storage
- Provider system for embeddings
- Similarity-based cache hits
"""

import hashlib
import logging
import time
from typing import Any

try:
    import chromadb
    from chromadb.config import Settings

    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

from ..providers import ProviderConfig
from .config import SemanticCacheConfig
from .embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)


class SemanticCache:
    """
    Semantic cache with vector similarity search.

    Uses ChromaDB for vector storage and provider system for embeddings.
    """

    def __init__(self, config: SemanticCacheConfig):
        """
        Initialize semantic cache.

        Args:
            config: Semantic cache configuration
        """
        if not CHROMADB_AVAILABLE:
            raise RuntimeError("chromadb not installed. Install with: pip install chromadb>=0.4.0")

        self.config = config
        self._embedding_generator: EmbeddingGenerator | None = None
        self._client = None
        self._collection = None

        self._initialize()

    def _initialize(self) -> None:
        """Initialize ChromaDB and embedding generator."""
        # Initialize ChromaDB
        logger.info(f"Initializing ChromaDB at {self.config.persist_directory}")
        self._client = chromadb.PersistentClient(
            path=self.config.persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )

        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=self.config.collection_name,
            metadata={"description": "Seraph MCP semantic cache"},
        )

        logger.info(f"ChromaDB collection '{self.config.collection_name}' ready")

    def _get_embedding_generator(self) -> EmbeddingGenerator:
        """Get or create embedding generator."""
        if self._embedding_generator is None:
            # Create provider config if needed
            provider_config = None
            if self.config.embedding_provider != "local":
                provider_config = ProviderConfig(
                    api_key=self.config.embedding_api_key or "",
                    base_url=self.config.embedding_base_url,
                    timeout=30.0,
                    enabled=True,
                )

            self._embedding_generator = EmbeddingGenerator(
                provider_name=self.config.embedding_provider,
                model_name=self.config.embedding_model,
                provider_config=provider_config,
                cache_embeddings=self.config.cache_embeddings,
            )

        return self._embedding_generator

    def _generate_id(self, text: str) -> str:
        """Generate stable ID for cache entry."""
        return hashlib.sha256(text.encode()).hexdigest()

    async def get(
        self,
        query: str,
        threshold: float | None = None,
        max_results: int = 1,
    ) -> dict[str, Any] | None:
        """
        Get cached value by semantic similarity.

        Args:
            query: Query text
            threshold: Similarity threshold (uses config default if None)
            max_results: Maximum results to return

        Returns:
            Best matching cache entry or None if no match above threshold
        """
        if not self.config.enabled:
            return None

        threshold = threshold or self.config.similarity_threshold

        try:
            # Generate embedding for query
            generator = self._get_embedding_generator()
            embedding = await generator.generate(query)

            # Search for similar entries
            results = self._collection.query(
                query_embeddings=[embedding],
                n_results=max_results,
            )

            # Check if we have results
            if not results["ids"] or not results["ids"][0]:
                return None

            # Get best match
            best_distance = results["distances"][0][0]
            best_similarity = 1 - best_distance  # Convert distance to similarity

            if best_similarity < threshold:
                return None

            # Return cached value
            return {
                "value": results["documents"][0][0],
                "metadata": results["metadatas"][0][0],
                "similarity": best_similarity,
                "id": results["ids"][0][0],
            }

        except Exception as e:
            logger.error(f"Error getting from semantic cache: {e}")
            return None

    async def set(
        self,
        key: str,
        value: Any,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        Store value in semantic cache.

        Args:
            key: Cache key (text to embed)
            value: Value to cache (stored as string)
            metadata: Optional metadata

        Returns:
            True if successful
        """
        if not self.config.enabled:
            return False

        try:
            # Generate embedding
            generator = self._get_embedding_generator()
            embedding = await generator.generate(key)

            # Generate stable ID
            entry_id = self._generate_id(key)

            # Prepare metadata
            meta = metadata or {}
            meta.update(
                {
                    "timestamp": time.time(),
                    "key": key,
                }
            )

            # Convert value to string if needed
            value_str = str(value)

            # Store in ChromaDB
            self._collection.add(
                ids=[entry_id],
                embeddings=[embedding],
                documents=[value_str],
                metadatas=[meta],
            )

            return True

        except Exception as e:
            logger.error(f"Error setting semantic cache: {e}")
            return False

    async def search(
        self,
        query: str,
        limit: int | None = None,
        threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for similar entries.

        Args:
            query: Query text
            limit: Maximum results (uses config default if None)
            threshold: Similarity threshold (uses config default if None)

        Returns:
            List of matching entries with similarity scores
        """
        if not self.config.enabled:
            return []

        limit = limit or self.config.max_results
        threshold = threshold or self.config.similarity_threshold

        try:
            # Generate embedding
            generator = self._get_embedding_generator()
            embedding = await generator.generate(query)

            # Search
            results = self._collection.query(
                query_embeddings=[embedding],
                n_results=limit,
            )

            # Format results
            entries = []
            if results["ids"] and results["ids"][0]:
                for i, entry_id in enumerate(results["ids"][0]):
                    distance = results["distances"][0][i]
                    similarity = 1 - distance

                    if similarity >= threshold:
                        entries.append(
                            {
                                "id": entry_id,
                                "value": results["documents"][0][i],
                                "metadata": results["metadatas"][0][i],
                                "similarity": similarity,
                            }
                        )

            return entries

        except Exception as e:
            logger.error(f"Error searching semantic cache: {e}")
            return []

    def clear(self, namespace: str | None = None) -> bool:
        """
        Clear cache entries.

        Args:
            namespace: Optional namespace filter (not implemented yet)

        Returns:
            True if successful
        """
        try:
            # Delete and recreate collection
            self._client.delete_collection(name=self.config.collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self.config.collection_name,
                metadata={"description": "Seraph MCP semantic cache"},
            )
            logger.info("Semantic cache cleared")
            return True
        except Exception as e:
            logger.error(f"Error clearing semantic cache: {e}")
            return False

    def get_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        try:
            count = self._collection.count()

            return {
                "enabled": self.config.enabled,
                "total_entries": count,
                "collection_name": self.config.collection_name,
                "embedding_provider": self.config.embedding_provider,
                "embedding_model": self.config.embedding_model,
                "similarity_threshold": self.config.similarity_threshold,
                "max_cache_entries": self.config.max_cache_entries,
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"error": str(e)}

    async def close(self) -> None:
        """Clean up resources."""
        # ChromaDB client cleanup
        if self._embedding_generator:
            self._embedding_generator.clear_cache()

        logger.info("Semantic cache closed")


# Global singleton
_cache: SemanticCache | None = None


def get_semantic_cache(config: SemanticCacheConfig | None = None) -> SemanticCache:
    """
    Get global semantic cache instance.

    Args:
        config: Cache configuration (creates new instance if provided)

    Returns:
        SemanticCache instance
    """
    global _cache

    if config is not None or _cache is None:
        if config is None:
            config = SemanticCacheConfig()
        _cache = SemanticCache(config)

    return _cache


async def close_semantic_cache() -> None:
    """Close global semantic cache."""
    global _cache

    if _cache:
        await _cache.close()
        _cache = None
