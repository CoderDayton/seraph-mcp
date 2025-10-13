"""
Hybrid Compression Demo
========================

Demonstrates the three compression methods in Seraph MCP:
1. AI Compression: Fast, nuanced, best for short prompts (≤3k tokens)
2. Seraph Compression: Deterministic, cacheable, multi-layer (L1/L2/L3)
3. Hybrid Mode: Seraph pre-compress + AI polish

Shows when to use each method and their performance characteristics.
"""

import asyncio
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.context_optimization import (
    ContextOptimizationConfig,
    ContextOptimizer,
    SeraphCompressor,
)
from src.providers import OpenAIProvider, ProviderConfig

# Sample content of different sizes
SHORT_CONTENT = """
AI agents are transforming software development. They can write code,
debug issues, and even optimize their own performance. The future of
programming involves more collaboration between humans and AI systems.
"""

MEDIUM_CONTENT = (
    """
# AI Optimization Best Practices

## Token Management
Effective token management is crucial for cost control in AI applications.
By compressing prompts, removing redundancy, and caching results, you can
reduce API costs by 40-60% while maintaining quality.

## Quality Preservation
Always validate that optimizations don't degrade output quality. Use
multi-dimensional scoring: semantic similarity, structure preservation,
and information completeness. Aim for ≥90% quality scores.

## Performance Optimization
Sub-100ms processing is achievable with proper caching and parallel
processing. Use deterministic methods for repeated queries and AI
compression for one-shot requests requiring nuance.

## Cost-Performance Tradeoffs
Balance compression ratio against quality. Aggressive compression saves
money but may lose nuance. Conservative compression preserves quality
but saves less. Use hybrid approaches for best results.
"""
    * 10
)  # ~3k tokens

LONG_CONTENT = (
    """
# Comprehensive AI System Architecture Guide

## Introduction
Building production AI systems requires careful consideration of multiple
factors: performance, cost, reliability, and quality. This guide provides
best practices for each area.

## System Components

### 1. Model Selection
Choose models based on task requirements. GPT-4 for complex reasoning,
GPT-3.5 for simple tasks, Claude for long context, Gemini for multimodal.

### 2. Prompt Engineering
Effective prompts are clear, specific, and include examples. Use system
messages to set behavior. Include constraints and output format requirements.

### 3. Token Optimization
Reduce tokens through compression, caching, and smart routing. Track
usage metrics to identify optimization opportunities.

### 4. Quality Assurance
Implement validation layers: syntax checking, semantic verification,
and output scoring. Use automatic rollback for low-quality results.

### 5. Cost Management
Set budgets, track spending, forecast future costs. Use cheaper models
where appropriate. Implement rate limiting and circuit breakers.

## Implementation Details

### Caching Strategy
Multi-level caching: memory for hot data, Redis for persistence,
semantic similarity for fuzzy matching. TTL-based expiration.

### Monitoring & Observability
Track latency, error rates, quality scores, and cost per request.
Alert on anomalies. Log all decisions for debugging.

### Error Handling
Graceful degradation, automatic retries with exponential backoff,
fallback to simpler models on failure.

### Security
API key rotation, rate limiting, input sanitization, output filtering.
Never log sensitive data. Use environment variables for secrets.

## Performance Benchmarks

### Latency Targets
- Optimization: <100ms
- Cache lookup: <10ms
- API calls: <2s
- Total request: <5s

### Quality Targets
- Semantic similarity: ≥0.90
- Structure preservation: ≥0.95
- Information completeness: ≥0.92

### Cost Targets
- 40-60% reduction via optimization
- 60-80% cache hit rate
- <$0.01 per request average

## Conclusion
Building robust AI systems requires attention to detail across multiple
dimensions. Use the patterns and practices in this guide to create
production-ready applications.
"""
    * 20
)  # ~10k tokens


async def demo_ai_compression():
    """Demonstrate AI compression on short content"""
    print("\n" + "=" * 70)
    print("DEMO 1: AI Compression (Fast & Nuanced)")
    print("=" * 70)
    print("\nBest for: Short prompts (≤3k tokens), one-shot use, nuance preservation")

    # Create provider
    config = ProviderConfig(api_key="sk-test-key")  # Use real key in production
    provider = OpenAIProvider(config)

    # Create optimizer with AI mode
    opt_config = ContextOptimizationConfig(
        enabled=True,
        compression_method="ai",
        quality_threshold=0.90,
        max_overhead_ms=100.0,
    )
    optimizer = ContextOptimizer(opt_config, provider=provider)

    print(f"\nOriginal content: {len(SHORT_CONTENT)} chars")
    print(f"Content preview: {SHORT_CONTENT[:100]}...")

    start = time.perf_counter()
    result = await optimizer.optimize(SHORT_CONTENT)
    elapsed_ms = (time.perf_counter() - start) * 1000

    print("\nResults:")
    print(f"  Method: {result.method}")
    print(f"  Tokens saved: {result.tokens_saved}")
    print(f"  Reduction: {result.reduction_percentage:.1f}%")
    print(f"  Quality score: {result.quality_score:.2f}")
    print(f"  Time: {elapsed_ms:.1f}ms")
    print(f"  Validation passed: {result.validation_passed}")

    if result.validation_passed:
        print(f"\nCompressed: {result.optimized_content[:150]}...")


async def demo_seraph_compression():
    """Demonstrate Seraph compression on long content"""
    print("\n" + "=" * 70)
    print("DEMO 2: Seraph Compression (Deterministic & Cacheable)")
    print("=" * 70)
    print("\nBest for: Long content (>3k tokens), repeated queries, multi-session")

    # Create optimizer with Seraph mode
    opt_config = ContextOptimizationConfig(
        enabled=True,
        compression_method="seraph",
        seraph_token_threshold=3000,
        seraph_l1_ratio=0.002,  # Ultra-small skeleton
        seraph_l2_ratio=0.01,  # Compact abstracts
        seraph_l3_ratio=0.05,  # Factual extracts
        quality_threshold=0.90,
        max_overhead_ms=100.0,
    )
    optimizer = ContextOptimizer(opt_config, provider=None)  # No provider needed

    print(f"\nOriginal content: {len(LONG_CONTENT)} chars")

    # First compression (cold start)
    start = time.perf_counter()
    result1 = await optimizer.optimize(LONG_CONTENT)
    elapsed_ms1 = (time.perf_counter() - start) * 1000

    print("\nFirst compression (cold start):")
    print(f"  Method: {result1.method}")
    print(f"  Tokens saved: {result1.tokens_saved}")
    print(f"  Reduction: {result1.reduction_percentage:.1f}%")
    print(f"  Quality score: {result1.quality_score:.2f}")
    print(f"  Time: {elapsed_ms1:.1f}ms")

    # Second compression (cached)
    start = time.perf_counter()
    result2 = await optimizer.optimize(LONG_CONTENT)
    elapsed_ms2 = (time.perf_counter() - start) * 1000

    print("\nSecond compression (cached):")
    print(f"  Time: {elapsed_ms2:.1f}ms")
    print(f"  Speedup: {elapsed_ms1 / elapsed_ms2:.1f}x faster")
    print(f"  Compression ratio: {result2.compression_ratio:.1%}")

    # Show multi-layer structure
    print("\nMulti-layer structure:")
    if hasattr(optimizer, "seraph_cache") and optimizer.seraph_cache:
        cached_data = list(optimizer.seraph_cache.values())[0]
        print(f"  L1 (skeleton): {len(cached_data['l1'])} chars")
        print(f"  L2 (abstracts): {len(cached_data['l2'])} chars")
        print(f"  L3 (extracts): {len(cached_data['l3'])} chars")
        print(f"\n  L1 preview: {cached_data['l1'][:100]}...")


async def demo_hybrid_compression():
    """Demonstrate hybrid compression"""
    print("\n" + "=" * 70)
    print("DEMO 3: Hybrid Compression (Best of Both)")
    print("=" * 70)
    print("\nBest for: Tight budgets + quality requirements, iterative refinement")

    # Create provider
    config = ProviderConfig(api_key="sk-test-key")  # Use real key in production
    provider = OpenAIProvider(config)

    # Create optimizer with hybrid mode
    opt_config = ContextOptimizationConfig(
        enabled=True,
        compression_method="hybrid",
        quality_threshold=0.90,
        max_overhead_ms=150.0,  # Allow more time for hybrid
    )
    optimizer = ContextOptimizer(opt_config, provider=provider)

    print(f"\nOriginal content: {len(MEDIUM_CONTENT)} chars")

    start = time.perf_counter()
    result = await optimizer.optimize(MEDIUM_CONTENT)
    elapsed_ms = (time.perf_counter() - start) * 1000

    print("\nResults:")
    print(f"  Method: {result.method}")
    print(f"  Tokens saved: {result.tokens_saved}")
    print(f"  Reduction: {result.reduction_percentage:.1f}%")
    print(f"  Quality score: {result.quality_score:.2f}")
    print(f"  Time: {elapsed_ms:.1f}ms")
    print(f"  Validation passed: {result.validation_passed}")

    print("\nHow it works:")
    print("  1. Seraph pre-compresses to L2 (deterministic structure)")
    print("  2. AI polishes the compressed content (semantic enhancement)")
    print("  3. Quality validation ensures improvement")


async def demo_auto_selection():
    """Demonstrate automatic method selection"""
    print("\n" + "=" * 70)
    print("DEMO 4: Auto Method Selection (Smart Routing)")
    print("=" * 70)
    print("\nAuto mode: ≤3k tokens → AI, >3k tokens → Seraph")

    # Create provider
    config = ProviderConfig(api_key="sk-test-key")
    provider = OpenAIProvider(config)

    # Create optimizer with auto mode
    opt_config = ContextOptimizationConfig(
        enabled=True,
        compression_method="auto",
        seraph_token_threshold=3000,
        quality_threshold=0.90,
    )
    optimizer = ContextOptimizer(opt_config, provider=provider)

    # Test short content (should use AI)
    result_short = await optimizer.optimize(SHORT_CONTENT)
    print(f"\nShort content ({result_short.tokens_before} tokens):")
    print(f"  Selected method: {result_short.method}")
    print(f"  Reduction: {result_short.reduction_percentage:.1f}%")

    # Test long content (should use Seraph)
    result_long = await optimizer.optimize(LONG_CONTENT)
    print(f"\nLong content ({result_long.tokens_before} tokens):")
    print(f"  Selected method: {result_long.method}")
    print(f"  Reduction: {result_long.reduction_percentage:.1f}%")

    # Show statistics
    stats = optimizer.get_stats()
    print("\nMethod usage statistics:")
    print(f"  AI compressions: {stats['method_usage']['ai']}")
    print(f"  Seraph compressions: {stats['method_usage']['seraph']}")
    print(f"  Hybrid compressions: {stats['method_usage']['hybrid']}")


async def demo_seraph_standalone():
    """Demonstrate standalone Seraph compressor"""
    print("\n" + "=" * 70)
    print("DEMO 5: Standalone Seraph Compressor (Advanced)")
    print("=" * 70)
    print("\nDirect access to multi-layer compression and querying")

    # Create standalone compressor
    compressor = SeraphCompressor(seed=7)

    print("\nCompressing long document...")
    start = time.perf_counter()
    result = compressor.build(LONG_CONTENT)
    elapsed_ms = (time.perf_counter() - start) * 1000

    print(f"\nCompression complete in {elapsed_ms:.1f}ms")
    print("\nLayer sizes:")
    print(f"  L1: {len(result.l1)} chars ({result.manifest['budgets']['L1']} tokens)")
    print(f"  L2: {len(result.l2)} chars ({result.manifest['budgets']['L2']} tokens)")
    print(f"  L3: {len(result.l3)} chars ({result.manifest['budgets']['L3']} tokens)")

    print("\nManifest:")
    print(f"  Total original tokens: {result.manifest['total_tokens']}")
    print(f"  Chunks created: {result.manifest['chunks']}")
    print(f"  Anchors extracted: {result.manifest['anchors']}")

    # Demonstrate querying
    print("\nQuerying L3 layer:")
    question = "What are the performance targets?"
    top_results = compressor.query(result, question, k=3)

    print(f"  Query: '{question}'")
    print("  Top results:")
    for score, text in top_results[:3]:
        print(f"    Score {score:.3f}: {text[:80]}...")


async def main():
    """Run all demos"""
    print("\n" + "=" * 70)
    print("Seraph MCP: Hybrid Compression System Demo")
    print("=" * 70)

    try:
        # Note: These demos will fail without real API keys
        # They demonstrate the API and decision logic

        # await demo_ai_compression()
        # await demo_seraph_compression()
        # await demo_hybrid_compression()
        # await demo_auto_selection()
        await demo_seraph_standalone()

        print("\n" + "=" * 70)
        print("Decision Matrix Summary")
        print("=" * 70)
        print("""
        | Scenario           | Tokens | Reuse    | Best Method | Why                     |
        |--------------------|--------|----------|-------------|-------------------------|
        | Chat prompt        | 500    | One-shot | AI          | Fast, preserves nuance  |
        | Document summary   | 10k    | One-shot | AI          | Better semantic grasp   |
        | Multi-doc context  | 50k    | Repeated | Seraph      | Build once, query many  |
        | Tool logs          | 100k   | Persist  | Seraph      | Deterministic, cacheable|
        | Code context       | 5k     | Evolving | Hybrid      | Structure + semantic    |
        | Tight budget       | Any    | Any      | Hybrid      | Max reduction + quality |
        """)

        print("\nConfiguration:")
        print("  CONTEXT_OPTIMIZATION_COMPRESSION_METHOD=auto  # Recommended")
        print("  CONTEXT_OPTIMIZATION_SERAPH_TOKEN_THRESHOLD=3000")
        print("  CONTEXT_OPTIMIZATION_QUALITY_THRESHOLD=0.90")
        print("  CONTEXT_OPTIMIZATION_MAX_OVERHEAD_MS=100.0")

    except Exception as e:
        print(f"\nDemo error (expected without API keys): {e}")
        print("\nTo run with real compression:")
        print("1. Set OPENAI_API_KEY environment variable")
        print("2. Uncomment the demo function calls in main()")
        print("3. Run: python hybrid_compression_demo.py")


if __name__ == "__main__":
    asyncio.run(main())
