"""
Semantic Cache Implementation

ChromaDB-backed semantic cache with vector similarity search and multi-layer eviction.
Uses the provider system for embeddings.

Per SDD.md:
- ChromaDB for vector storage (lazy imported)
- Provider system for embeddings
- Similarity-based cache hits
- Multi-layer LRU+FIFO eviction (P0 Phase 3)
- Optional TTL support (disabled by default)
"""

import hashlib
import logging
import time
from typing import Any

from ..providers import ProviderConfig
from .config import SemanticCacheConfig
from .embeddings import EmbeddingGenerator
from .eviction import MultiLayerCache

logger = logging.getLogger(__name__)

# Lazy import flag - ChromaDB only loaded when SemanticCache is instantiated
_CHROMADB_CHECKED = False
_CHROMADB_AVAILABLE = False


def _check_chromadb_available() -> bool:
    """Check if ChromaDB is available (lazy check on first use)."""
    global _CHROMADB_CHECKED, _CHROMADB_AVAILABLE

    if not _CHROMADB_CHECKED:
        try:
            import chromadb  # noqa: F401

            _CHROMADB_AVAILABLE = True
        except ImportError:
            _CHROMADB_AVAILABLE = False
        _CHROMADB_CHECKED = True

    return _CHROMADB_AVAILABLE


class SemanticCache:
    """
    Semantic cache with vector similarity search and multi-layer eviction.

    Uses ChromaDB for vector storage, provider system for embeddings,
    and multi-layer LRU+FIFO cache for efficient access patterns.

    Architecture:
        - Hot tier (LRU): Recently accessed items
        - Cold tier (FIFO): Less frequent items
        - ChromaDB: Persistent vector storage
        - Automatic promotion on re-access
        - Optional TTL (disabled by default)
    """

    def __init__(self, config: SemanticCacheConfig):
        """
        Initialize semantic cache.

        Args:
            config: Semantic cache configuration
        """
        if not _check_chromadb_available():
            raise RuntimeError("chromadb not installed. Install with: pip install chromadb>=0.4.0")

        self.config = config
        self._embedding_generator: EmbeddingGenerator | None = None
        self._client: Any = None
        self._collection: Any = None

        # Multi-layer cache for efficient access (P0 Phase 3)
        self._multi_cache: MultiLayerCache | None = None

        self._initialize()

    def _initialize(self) -> None:
        """Initialize ChromaDB and embedding generator (lazy imports ChromaDB)."""
        # Lazy import ChromaDB - only loaded when SemanticCache is instantiated
        import chromadb
        from chromadb.config import Settings

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

        # Initialize multi-layer cache (P0 Phase 3)
        self._multi_cache = MultiLayerCache(
            lru_size=self.config.lru_cache_size,
            fifo_size=self.config.fifo_cache_size,
            ttl_seconds=self.config.entry_ttl_seconds,
            high_watermark_pct=self.config.high_watermark_pct,
            cleanup_batch_size=self.config.cleanup_batch_size,
        )
        logger.info("Multi-layer cache initialized")

    def _get_embedding_generator(self) -> EmbeddingGenerator:
        """Get or create embedding generator."""
        if self._embedding_generator is None:
            # Provider config is now always required (no more local mode)
            if not self.config.embedding_api_key:
                raise RuntimeError(
                    f"embedding_api_key is required for provider '{self.config.embedding_provider}'. "
                    "Local (sentence-transformers) embeddings are no longer supported. "
                    "Configure an API provider (openai, openai-compatible, gemini) or use "
                    "openai-compatible with a local endpoint like Ollama."
                )

            provider_config = ProviderConfig(
                api_key=self.config.embedding_api_key,
                model=self.config.embedding_model,
                base_url=self.config.embedding_base_url,
                timeout=30.0,
                max_retries=3,
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
            # Try multi-layer cache first (P0 Phase 3)
            if self._multi_cache:
                cached_result = self._multi_cache.get(query)
                if cached_result is not None:
                    logger.debug(f"Multi-layer cache hit for query: {query[:50]}...")
                    return cached_result

            # Generate embedding for query
            generator = self._get_embedding_generator()
            embedding = await generator.generate(query)

            # Search for similar entries in ChromaDB
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

            # Build result
            result = {
                "value": results["documents"][0][0],
                "metadata": results["metadatas"][0][0],
                "similarity": best_similarity,
                "id": results["ids"][0][0],
            }

            # Store in multi-layer cache (P0 Phase 3)
            if self._multi_cache:
                self._multi_cache.set(query, result, metadata=result["metadata"])

            # Process eviction queue for batch ChromaDB deletes
            if self._multi_cache:
                evicted_keys = self._multi_cache.get_eviction_queue()
                if evicted_keys:
                    await self._batch_delete_chromadb(evicted_keys)

            return result

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

            # Store in multi-layer cache (P0 Phase 3)
            if self._multi_cache:
                cache_entry = {
                    "value": value_str,
                    "metadata": meta,
                    "similarity": 1.0,  # Exact match
                    "id": entry_id,
                }
                self._multi_cache.set(key, cache_entry, metadata=meta)

            # Process eviction queue for batch ChromaDB deletes
            if self._multi_cache:
                evicted_keys = self._multi_cache.get_eviction_queue()
                if evicted_keys:
                    await self._batch_delete_chromadb(evicted_keys)

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

            # Query ChromaDB
            if self._collection is None:
                raise RuntimeError("Collection not initialized")
            results = self._collection.query(
                query_embeddings=[embedding],
                n_results=limit,
            )

            # Format results
            entries = []
            if results["ids"] and results["ids"][0]:
                for i, cache_id in enumerate(results.get("ids", [[]])[0]):
                    distance = results.get("distances", [[]])[0][i]
                    similarity = 1 - distance

                    if similarity >= threshold:
                        entries.append(
                            {
                                "id": cache_id,
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
            # Clear ChromaDB collection
            if self._client is None:
                raise RuntimeError("Client not initialized")
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

    async def _batch_delete_chromadb(self, keys: list[str]) -> None:
        """
        Batch delete entries from ChromaDB.

        Args:
            keys: List of entry IDs to delete
        """
        if not keys:
            return

        try:
            # Generate IDs from keys
            ids = [self._generate_id(key) for key in keys]

            # Batch delete from ChromaDB
            if self._collection is not None:
                self._collection.delete(ids=ids)
                logger.info(f"Batch deleted {len(ids)} entries from ChromaDB")

        except Exception as e:
            logger.error(f"Error batch deleting from ChromaDB: {e}")

    def get_stats(self) -> dict[str, Any]:
        """
        Get cache statistics including multi-layer cache metrics.

        Returns:
            Dictionary with cache stats including eviction metrics
        """
        if self._collection is None:
            raise RuntimeError("Collection not initialized")
        count = self._collection.count()

        base_stats = {
            "enabled": self.config.enabled,
            "total_entries": count,
            "collection_name": self.config.collection_name,
            "embedding_provider": self.config.embedding_provider,
            "embedding_model": self.config.embedding_model,
            "similarity_threshold": self.config.similarity_threshold,
            "max_cache_entries": self.config.max_cache_entries,
        }

        # Add multi-layer cache stats (P0 Phase 3)
        if self._multi_cache:
            multi_cache_stats = self._multi_cache.get_stats()
            base_stats["multi_layer_cache"] = multi_cache_stats

        return base_stats

    async def close(self) -> None:
        """Clean up resources."""
        # ChromaDB client cleanup
        if self._embedding_generator:
            self._embedding_generator.clear_cache()

        # Clear multi-layer cache
        if self._multi_cache:
            self._multi_cache.clear()

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
