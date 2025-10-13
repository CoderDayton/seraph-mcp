"""
Context Optimization Module

Hybrid compression system combining AI-powered and deterministic multi-layer compression.

Two approaches:
1. AI Compression: Fast, nuanced, best for short prompts (≤3k tokens)
2. Seraph Compression: Deterministic, cacheable, multi-layer (L1/L2/L3), best for long/recurring contexts (>3k tokens)
3. Hybrid Mode: Seraph pre-compress + AI polish for optimal results

Target: <100ms, >=90% quality, 20-40% token reduction
"""

from .config import ContextOptimizationConfig, load_config
from .middleware import OptimizedProvider, wrap_provider
from .models import FeedbackRecord, OptimizationResult
from .optimizer import ContextOptimizer, optimize_content
from .seraph_compression import (
    CompressionResult,
    SeraphCompressor,
    Tier2DCP,
    Tier3Hierarchical,
    Tier1500x,
)

__all__ = [
    "ContextOptimizationConfig",
    "load_config",
    "OptimizationResult",
    "FeedbackRecord",
    "ContextOptimizer",
    "optimize_content",
    "OptimizedProvider",
    "wrap_provider",
    "SeraphCompressor",
    "CompressionResult",
    "Tier1500x",
    "Tier2DCP",
    "Tier3Hierarchical",
]
