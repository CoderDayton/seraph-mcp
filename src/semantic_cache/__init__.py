"""
Semantic Cache Module

Provides semantic similarity-based caching using ChromaDB and the provider system.

Per SDD.md:
- Minimal, functional implementation
- ChromaDB for vector storage
- Provider system for embeddings (OpenAI, Ollama, LM Studio, local)
- Similarity-based cache hits

Public API:
    - SemanticCache: Main cache interface
    - SemanticCacheConfig: Configuration schema
    - EmbeddingGenerator: Embedding generation
    - get_semantic_cache(): Get global cache instance
    - close_semantic_cache(): Cleanup

Usage:
    >>> from src.semantic_cache import get_semantic_cache, SemanticCacheConfig
    >>>
    >>> # Create cache with local embeddings
    >>> config = SemanticCacheConfig(
    ...     embedding_provider="local",
    ...     embedding_model="all-MiniLM-L6-v2",
    ... )
    >>> cache = get_semantic_cache(config)
    >>>
    >>> # Store value
    >>> await cache.set("What is Python?", "Python is a programming language")
    >>>
    >>> # Search with semantic similarity
    >>> result = await cache.get("Tell me about Python")
    >>> if result:
    ...     print(f"Found: {result['value']} (similarity: {result['similarity']:.2f})")
    >>>
    >>> # Use with API provider (OpenAI)
    >>> config = SemanticCacheConfig(
    ...     embedding_provider="openai",
    ...     embedding_model="text-embedding-3-small",
    ...     embedding_api_key="sk-...",
    ... )
    >>> cache = get_semantic_cache(config)
    >>>
    >>> # Use with local Ollama
    >>> config = SemanticCacheConfig(
    ...     embedding_provider="openai-compatible",
    ...     embedding_model="nomic-embed-text",
    ...     embedding_base_url="http://localhost:11434/v1",
    ... )
    >>> cache = get_semantic_cache(config)
"""

from .cache import SemanticCache, close_semantic_cache, get_semantic_cache
from .config import SemanticCacheConfig
from .embeddings import EmbeddingGenerator, get_embedding_generator
from .eviction import EvictionStats, MultiLayerCache

__all__ = [
    # Main cache interface
    "SemanticCache",
    "get_semantic_cache",
    "close_semantic_cache",
    # Configuration
    "SemanticCacheConfig",
    # Embeddings
    "EmbeddingGenerator",
    "get_embedding_generator",
    # Eviction (P0 Phase 3)
    "MultiLayerCache",
    "EvictionStats",
]
