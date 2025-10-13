"""
Semantic Cache Configuration

Defines typed configuration for semantic caching with vector similarity search.
Uses the provider system for embeddings (OpenAI, local Ollama/LM Studio, etc.).

Per SDD.md:
- Minimal, functional implementation
- Uses existing provider infrastructure
- Local and API embedding support
- ChromaDB for vector storage
"""

from pydantic import BaseModel, Field, field_validator


class SemanticCacheConfig(BaseModel):
    """Configuration for semantic cache system."""

    enabled: bool = Field(default=True, description="Enable semantic caching")

    # Embedding provider configuration
    embedding_provider: str = Field(
        default="local",
        description="Provider for embeddings: 'local', 'openai', 'openai-compatible'",
    )
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Model name (local: sentence-transformers, API: model ID)",
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
        allowed = ["local", "openai", "openai-compatible"]
        if v.lower() not in allowed:
            raise ValueError(f"embedding_provider must be one of: {', '.join(allowed)}")
        return v.lower()
