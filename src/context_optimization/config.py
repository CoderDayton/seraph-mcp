"""
Context Optimization Configuration

Minimal configuration - only essential settings.
AI-powered optimization with automatic learning.
"""

import os

from pydantic import BaseModel, ConfigDict, Field


class ContextOptimizationConfig(BaseModel):
    """Minimal context optimization configuration"""

    # Core toggle
    enabled: bool = Field(default=True, description="Enable automatic context optimization")

    # Compression method selection
    compression_method: str = Field(
        default="auto",
        description="Compression method: 'ai' (AI-powered), 'seraph' (deterministic multi-layer), 'hybrid' (seraph + AI), 'auto' (size-based selection)",
    )

    # Token threshold for method selection
    seraph_token_threshold: int = Field(
        default=3000,
        ge=100,
        description="Token count threshold for auto mode: <=threshold uses AI, >threshold uses seraph",
    )

    # Quality target
    quality_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Minimum quality score (0-1) to accept optimization",
    )

    # Performance limit
    # NOTE: This is the OUTER timeout for the entire optimization process.
    # Inner operations (AI provider calls) have their own timeouts (5s for compression, 3s for validation).
    # This should be longer than the sum of inner timeouts to allow graceful fallback.
    max_overhead_ms: float = Field(
        default=10000.0, ge=0.0, description="Maximum processing time in milliseconds (default: 10s)"
    )

    # Seraph compression ratios (research-backed: arxiv papers on context compression)
    # Target: 40-60% retention for optimal quality-compression balance
    seraph_l1_ratio: float = Field(
        default=0.15,
        ge=0.10,
        le=0.25,
        description="L1 layer ratio (15% retention - ultra-compressed summary)",
    )

    seraph_l2_ratio: float = Field(
        default=0.50,
        ge=0.40,
        le=0.60,
        description="L2 layer ratio (50% retention - balanced compression, default output)",
    )

    seraph_l3_ratio: float = Field(
        default=0.70,
        ge=0.60,
        le=0.85,
        description="L3 layer ratio (70% retention - light compression with high fidelity)",
    )

    # Embedding configuration for semantic similarity
    embedding_provider: str = Field(
        default="gemini",
        description="Embedding provider: 'openai', 'gemini', or 'none' (disables embeddings)",
    )

    embedding_model: str | None = Field(
        default=None,
        description="Embedding model name (provider-specific default if None)",
    )

    embedding_api_key: str | None = Field(
        default=None,
        description="API key for embedding provider (uses provider's key if None)",
    )

    embedding_dimensions: int | None = Field(
        default=None,
        ge=256,
        le=3072,
        description="Optional dimension reduction for embeddings",
    )

    model_config = ConfigDict(frozen=False)


def load_config() -> ContextOptimizationConfig:
    """
    Load configuration from environment variables.

    Auto-detection:
    - Compression method: If not specified, uses 'auto' which selects based on token count
    - Works without any provider (uses Seraph-only deterministic compression)
    - Enables hybrid mode automatically if provider is available

    Environment Variables:
        CONTEXT_OPTIMIZATION_ENABLED: Enable/disable (default: true)
        CONTEXT_OPTIMIZATION_COMPRESSION_METHOD: Method selection (default: auto)
        CONTEXT_OPTIMIZATION_SERAPH_TOKEN_THRESHOLD: Token threshold for auto mode (default: 3000)
        CONTEXT_OPTIMIZATION_QUALITY_THRESHOLD: Min quality 0-1 (default: 0.90)
        CONTEXT_OPTIMIZATION_MAX_OVERHEAD_MS: Max time ms (default: 100.0)
        CONTEXT_OPTIMIZATION_SERAPH_L1_RATIO: L1 ratio (default: 0.15)
        CONTEXT_OPTIMIZATION_SERAPH_L2_RATIO: L2 ratio (default: 0.50)
        CONTEXT_OPTIMIZATION_SERAPH_L3_RATIO: L3 ratio (default: 0.70)
        CONTEXT_OPTIMIZATION_EMBEDDING_PROVIDER: Provider (default: gemini)
        CONTEXT_OPTIMIZATION_EMBEDDING_MODEL: Model name (optional)
        CONTEXT_OPTIMIZATION_EMBEDDING_API_KEY: API key (optional)
        CONTEXT_OPTIMIZATION_EMBEDDING_DIMENSIONS: Dimensions (optional)
    """
    # Auto-detect compression method based on provider availability
    # If no provider configured, default to 'seraph' (works without AI)
    # If provider exists, default to 'auto' (smart selection)
    default_method = "auto"  # 'auto' falls back to seraph if no provider

    return ContextOptimizationConfig(
        enabled=os.getenv("CONTEXT_OPTIMIZATION_ENABLED", "true").lower() == "true",
        compression_method=os.getenv("CONTEXT_OPTIMIZATION_COMPRESSION_METHOD", default_method).lower(),
        seraph_token_threshold=int(os.getenv("CONTEXT_OPTIMIZATION_SERAPH_TOKEN_THRESHOLD", "3000")),
        quality_threshold=float(os.getenv("CONTEXT_OPTIMIZATION_QUALITY_THRESHOLD", "0.85")),
        max_overhead_ms=float(os.getenv("CONTEXT_OPTIMIZATION_MAX_OVERHEAD_MS", "10000.0")),
        seraph_l1_ratio=float(os.getenv("CONTEXT_OPTIMIZATION_SERAPH_L1_RATIO", "0.15")),
        seraph_l2_ratio=float(os.getenv("CONTEXT_OPTIMIZATION_SERAPH_L2_RATIO", "0.50")),
        seraph_l3_ratio=float(os.getenv("CONTEXT_OPTIMIZATION_SERAPH_L3_RATIO", "0.70")),
        embedding_provider=os.getenv("CONTEXT_OPTIMIZATION_EMBEDDING_PROVIDER", "gemini"),
        embedding_model=os.getenv("CONTEXT_OPTIMIZATION_EMBEDDING_MODEL"),
        embedding_api_key=os.getenv("CONTEXT_OPTIMIZATION_EMBEDDING_API_KEY"),
        embedding_dimensions=int(os.getenv("CONTEXT_OPTIMIZATION_EMBEDDING_DIMENSIONS", 768)),
    )
