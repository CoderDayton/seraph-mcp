"""
Semantic Cache Configuration

Defines typed configuration for semantic caching with vector similarity search.
Uses the unified provider system for embeddings (OpenAI, OpenAI-compatible, Gemini).

Per SDD.md (v2.0.0):
- Minimal, functional implementation
- Uses unified provider-backed embedding infrastructure
- No local (sentence-transformers) support (removed to reduce dependencies)
- ChromaDB for vector storage
"""

from pydantic import BaseModel, Field, field_validator


class SemanticCacheConfig(BaseModel):
    """Configuration for semantic cache system."""

    enabled: bool = Field(default=True, description="Enable semantic caching")

    # Embedding provider configuration
    embedding_provider: str = Field(
        default="openai",
        description="Provider for embeddings: 'openai', 'openai-compatible', 'gemini'",
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Embedding model name (e.g., text-embedding-3-small for OpenAI)",
    )

    # For API providers (openai, openai-compatible)
    embedding_api_key: str | None = Field(default=None, description="API key for embedding provider (if using API)")
    embedding_base_url: str | None = Field(
        default=None, description="Base URL for openai-compatible embedding endpoints"
    )

    # Similarity search configuration
    similarity_threshold: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for cache hits (0.0-1.0)",
    )
    max_results: int = Field(default=10, ge=1, description="Maximum results to return from similarity search")

    # ChromaDB configuration
    collection_name: str = Field(default="seraph_semantic_cache", description="ChromaDB collection name")
    persist_directory: str = Field(default="./data/chromadb", description="Directory for ChromaDB persistence")
    max_cache_entries: int = Field(default=10000, ge=1, description="Maximum entries before cleanup")

    # Multi-layer cache eviction configuration (P0 Phase 3)
    # Based on S3-FIFO research: 10% hot (LRU) + 90% cold (FIFO) optimal for 95% workloads
    lru_cache_size: int = Field(
        default=1000,
        ge=1,
        description="Size of hot LRU tier for frequently accessed items (10% of total capacity)",
    )
    fifo_cache_size: int = Field(
        default=9000,
        ge=1,
        description="Size of cold FIFO tier for less frequent items (90% of total capacity)",
    )
    entry_ttl_seconds: int = Field(
        default=0,
        ge=0,
        description="Time-to-live for cache entries in seconds (0 = disabled, recommended default)",
    )
    high_watermark_pct: int = Field(
        default=90,
        ge=50,
        le=100,
        description="Percentage of capacity to trigger cleanup (50-100)",
    )
    cleanup_batch_size: int = Field(
        default=100,
        ge=1,
        description="Number of entries to evict per cleanup operation",
    )

    # Performance configuration
    batch_size: int = Field(default=32, ge=1, description="Batch size for embedding generation")
    cache_embeddings: bool = Field(default=True, description="Cache generated embeddings in memory")

    @field_validator("similarity_threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        """Ensure threshold is reasonable for semantic search."""
        if v < 0.5:
            raise ValueError("Similarity threshold should be >= 0.5 for meaningful results")
        return v

    @field_validator("embedding_provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """Ensure provider is supported."""
        allowed = ["openai", "openai-compatible", "gemini"]
        if v.lower() not in allowed:
            raise ValueError(
                f"embedding_provider must be one of: {', '.join(allowed)}. "
                "Note: 'local' (sentence-transformers) is no longer supported. "
                "Use 'openai-compatible' with a local endpoint (e.g., Ollama) instead."
            )
        return v.lower()
