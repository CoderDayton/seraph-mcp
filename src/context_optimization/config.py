"""
Context Optimization Configuration

Minimal configuration - only essential settings.
AI-powered optimization with automatic learning.
"""

import os
from pydantic import BaseModel, Field


class ContextOptimizationConfig(BaseModel):
    """Minimal context optimization configuration"""

    # Core toggle
    enabled: bool = Field(
        default=True,
        description="Enable automatic context optimization"
    )

    # Compression method selection
    compression_method: str = Field(
        default="auto",
        description="Compression method: 'ai' (AI-powered), 'seraph' (deterministic multi-layer), 'hybrid' (seraph + AI), 'auto' (size-based selection)"
    )

    # Token threshold for method selection
    seraph_token_threshold: int = Field(
        default=3000,
        ge=100,
        description="Token count threshold for auto mode: <=threshold uses AI, >threshold uses seraph"
    )

    # Quality target
    quality_threshold: float = Field(
        default=0.90,
        ge=0.0,
        le=1.0,
        description="Minimum quality score (0-1) to accept optimization"
    )

    # Performance limit
    max_overhead_ms: float = Field(
        default=100.0,
        ge=0.0,
        description="Maximum processing time in milliseconds"
    )

    # Seraph compression ratios
    seraph_l1_ratio: float = Field(
        default=0.002,
        ge=0.001,
        le=0.01,
        description="L1 layer ratio (ultra-small skeleton)"
    )

    seraph_l2_ratio: float = Field(
        default=0.01,
        ge=0.005,
        le=0.05,
        description="L2 layer ratio (compact abstracts)"
    )

    seraph_l3_ratio: float = Field(
        default=0.05,
        ge=0.02,
        le=0.15,
        description="L3 layer ratio (larger factual extracts)"
    )

    class Config:
        frozen = False


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
        CONTEXT_OPTIMIZATION_SERAPH_L1_RATIO: L1 ratio (default: 0.002)
        CONTEXT_OPTIMIZATION_SERAPH_L2_RATIO: L2 ratio (default: 0.01)
        CONTEXT_OPTIMIZATION_SERAPH_L3_RATIO: L3 ratio (default: 0.05)
    """
    # Auto-detect compression method based on provider availability
    # If no provider configured, default to 'seraph' (works without AI)
    # If provider exists, default to 'auto' (smart selection)
    default_method = "auto"  # 'auto' falls back to seraph if no provider

    return ContextOptimizationConfig(
        enabled=os.getenv("CONTEXT_OPTIMIZATION_ENABLED", "true").lower() == "true",
        compression_method=os.getenv("CONTEXT_OPTIMIZATION_COMPRESSION_METHOD", default_method).lower(),
        seraph_token_threshold=int(os.getenv("CONTEXT_OPTIMIZATION_SERAPH_TOKEN_THRESHOLD", "3000")),
        quality_threshold=float(os.getenv("CONTEXT_OPTIMIZATION_QUALITY_THRESHOLD", "0.90")),
        max_overhead_ms=float(os.getenv("CONTEXT_OPTIMIZATION_MAX_OVERHEAD_MS", "100.0")),
        seraph_l1_ratio=float(os.getenv("CONTEXT_OPTIMIZATION_SERAPH_L1_RATIO", "0.002")),
        seraph_l2_ratio=float(os.getenv("CONTEXT_OPTIMIZATION_SERAPH_L2_RATIO", "0.01")),
        seraph_l3_ratio=float(os.getenv("CONTEXT_OPTIMIZATION_SERAPH_L3_RATIO", "0.05")),
    )
