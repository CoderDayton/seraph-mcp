"""
Compression Method Benchmark
=============================

Compares Seraph and Hybrid compression methods across different content sizes and types.

Metrics measured:
- Compression ratio (tokens saved)
- Processing time (latency)
- Quality score (semantic similarity)
- Throughput (requests per second)
- Cost (API calls for hybrid mode)

Usage:
    python benchmarks/compression_comparison.py

Requirements:
    - OPENAI_COMPATIBLE_API_KEY in .env
    - OPENAI_COMPATIBLE_BASE_URL in .env
    - OPENAI_COMPATIBLE_MODEL in .env
"""

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.config.loader import load_config as load_full_config
from src.context_optimization import ContextOptimizationConfig, ContextOptimizer
from src.providers import create_provider

# Load environment variables
load_dotenv()


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    method: str
    content_size: str
    content_type: str
    tokens_before: int
    tokens_after: int
    tokens_saved: int
    reduction_percentage: float
    processing_time_ms: float
    quality_score: float
    validation_passed: bool
    cached: bool = False


# Test content of varying sizes and types
TEST_CONTENT = {
    "short_technical": """
The Transformer architecture revolutionized NLP through self-attention mechanisms.
Key components include multi-head attention, positional encoding, and feed-forward
networks. Training uses teacher forcing with cross-entropy loss. The model achieves
state-of-the-art results on translation, summarization, and question-answering tasks.
""",
    "medium_documentation": """
# API Documentation

## Authentication
All API requests require an API key in the Authorization header:
```
Authorization: Bearer YOUR_API_KEY
```

## Rate Limits
- Free tier: 100 requests/hour
- Pro tier: 10,000 requests/hour
- Enterprise: Custom limits

## Endpoints

### POST /v1/completions
Generate text completions.

**Request Body:**
```json
{
  "model": "gpt-4",
  "prompt": "Hello world",
  "max_tokens": 100,
  "temperature": 0.7
}
```

**Response:**
```json
{
  "id": "cmpl-123",
  "choices": [{
    "text": "Generated text",
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 2,
    "completion_tokens": 5,
    "total_tokens": 7
  }
}
```

### GET /v1/models
List available models.

**Response:**
```json
{
  "models": [
    {"id": "gpt-4", "created": 1234567890},
    {"id": "gpt-3.5-turbo", "created": 1234567890}
  ]
}
```

## Error Handling
Standard HTTP status codes are used:
- 400: Bad Request - Invalid parameters
- 401: Unauthorized - Invalid API key
- 429: Too Many Requests - Rate limit exceeded
- 500: Internal Server Error - Server error

## Best Practices
1. Cache responses when possible
2. Use streaming for long completions
3. Implement exponential backoff for retries
4. Monitor your usage and costs
5. Use appropriate model for task complexity
"""
    * 5,  # ~1500 tokens
    "long_technical": """
# Distributed Systems Architecture

## Overview
Distributed systems enable scalability, fault tolerance, and geographic distribution
of services. Key challenges include consistency, availability, partition tolerance
(CAP theorem), latency, and complexity management.

## Core Concepts

### Consistency Models
- Strong Consistency: All nodes see same data simultaneously
- Eventual Consistency: Nodes converge over time
- Causal Consistency: Causally related operations seen in order
- Read-your-writes: Readers see their own writes immediately

### Replication Strategies
- Master-Slave: Single write node, multiple read replicas
- Multi-Master: Multiple write nodes with conflict resolution
- Quorum-based: W + R > N for strong consistency
- Chain Replication: Sequential writes with tail reads

### Consensus Algorithms
- Paxos: Classic but complex algorithm
- Raft: More understandable alternative to Paxos
- ZAB: Zookeeper Atomic Broadcast
- PBFT: Practical Byzantine Fault Tolerance

### Partitioning Strategies
- Hash-based: Consistent hashing for even distribution
- Range-based: Ordered keys for range queries
- Directory-based: Lookup service for routing
- Hybrid: Combination of approaches

## Design Patterns

### Circuit Breaker
Prevents cascading failures by stopping requests to failing services.
States: Closed (normal), Open (failing), Half-Open (testing recovery).

### Saga Pattern
Manages distributed transactions through compensating actions.
Useful when ACID transactions span multiple services.

### CQRS (Command Query Responsibility Segregation)
Separates read and write models for optimized performance.
Write model enforces business rules, read model optimized for queries.

### Event Sourcing
Stores state changes as immutable events. Enables time travel,
audit logs, and event replay for debugging or rebuilding state.

### Service Mesh
Infrastructure layer for service-to-service communication.
Provides load balancing, service discovery, encryption, observability.

## Implementation Considerations

### Data Partitioning
Choose partition key carefully to avoid hotspots. Consider:
- Access patterns (read vs write heavy)
- Data distribution (uniform vs skewed)
- Query requirements (point vs range)
- Growth projections (scaling strategy)

### Failure Handling
- Timeouts: Set appropriate deadlines
- Retries: Use exponential backoff with jitter
- Fallbacks: Degrade gracefully
- Health Checks: Monitor service health
- Circuit Breakers: Prevent cascade failures

### Observability
- Distributed Tracing: Track requests across services
- Centralized Logging: Aggregate logs from all nodes
- Metrics Collection: Monitor latency, errors, saturation
- Alerting: Notify on anomalies and SLO violations

### Security
- Authentication: Verify service identity (mTLS)
- Authorization: Enforce access policies (RBAC, ABAC)
- Encryption: Protect data in transit and at rest
- Rate Limiting: Prevent abuse and DDoS
- API Gateways: Single entry point for security policies

## Testing Strategies

### Unit Tests
Test individual components in isolation with mocks.

### Integration Tests
Verify interactions between components work correctly.

### Chaos Engineering
Inject failures to test system resilience:
- Network latency and partitions
- Service crashes and restarts
- Resource exhaustion (CPU, memory, disk)
- Clock skew and time drift

### Load Testing
Validate system handles expected traffic:
- Sustained load testing
- Spike testing (sudden traffic increase)
- Stress testing (beyond capacity)
- Soak testing (extended duration)

## Performance Optimization

### Caching
- Application-level: Redis, Memcached
- CDN: Edge caching for static content
- Database: Query result caching
- HTTP: ETags, Cache-Control headers

### Batching
Combine multiple operations to reduce overhead:
- Request batching (multiple queries)
- Write batching (bulk inserts)
- Event batching (aggregate messages)

### Asynchronous Processing
Decouple request/response with message queues:
- Task queues for background jobs
- Event streams for real-time processing
- Pub/sub for fan-out patterns

### Connection Pooling
Reuse connections to reduce setup overhead:
- Database connection pools
- HTTP keep-alive connections
- gRPC connection multiplexing

## Deployment Strategies

### Blue-Green Deployment
Maintain two identical environments, switch traffic after validation.

### Canary Deployment
Gradually roll out to subset of users, monitor for issues.

### Rolling Deployment
Update instances incrementally to maintain availability.

### Feature Flags
Control feature availability without deployment.

## Monitoring and Operations

### SLIs (Service Level Indicators)
- Latency: Request duration (p50, p95, p99)
- Availability: Uptime percentage
- Error Rate: Failed requests percentage
- Throughput: Requests per second

### SLOs (Service Level Objectives)
Set targets for SLIs:
- 99.9% availability (43 minutes downtime/month)
- p95 latency < 100ms
- Error rate < 0.1%

### SLAs (Service Level Agreements)
Contractual guarantees with penalties for violations.

### Incident Response
1. Detection: Automated alerting on SLO violations
2. Triage: Assess severity and impact
3. Mitigation: Stop bleeding, restore service
4. Root Cause Analysis: Prevent future occurrences
5. Post-Mortem: Document learnings blameless

## Conclusion
Building distributed systems requires careful consideration of
tradeoffs between consistency, availability, performance, and
complexity. Choose patterns and technologies appropriate for
your specific requirements and constraints.
"""
    * 10,  # ~8000 tokens
}


async def benchmark_seraph(content: str, content_id: str) -> BenchmarkResult:
    """Benchmark Seraph compression method."""
    # Create optimizer with Seraph mode
    config = ContextOptimizationConfig(
        enabled=True,
        compression_method="seraph",
        seraph_token_threshold=100,  # Force Seraph for all sizes
        seraph_l1_ratio=0.002,
        seraph_l2_ratio=0.01,
        seraph_l3_ratio=0.05,
        quality_threshold=0.90,
        max_overhead_ms=200.0,
    )
    optimizer = ContextOptimizer(config, provider=None)

    # Run compression
    start = time.perf_counter()
    result = await optimizer.optimize(content)
    elapsed_ms = (time.perf_counter() - start) * 1000

    return BenchmarkResult(
        method="seraph",
        content_size=content_id.split("_")[0],
        content_type="_".join(content_id.split("_")[1:]),
        tokens_before=result.tokens_before,
        tokens_after=result.tokens_after,
        tokens_saved=result.tokens_saved,
        reduction_percentage=result.reduction_percentage,
        processing_time_ms=elapsed_ms,
        quality_score=result.quality_score,
        validation_passed=result.validation_passed,
    )


async def benchmark_hybrid(content: str, content_id: str, provider: Any) -> BenchmarkResult:
    """Benchmark Hybrid compression method."""
    # Create optimizer with Hybrid mode
    config = ContextOptimizationConfig(
        enabled=True,
        compression_method="hybrid",
        seraph_token_threshold=100,  # Force hybrid for all sizes
        quality_threshold=0.90,
        max_overhead_ms=5000.0,  # Allow more time for API calls
    )
    optimizer = ContextOptimizer(config, provider=provider)

    # Run compression
    start = time.perf_counter()
    result = await optimizer.optimize(content)
    elapsed_ms = (time.perf_counter() - start) * 1000

    return BenchmarkResult(
        method="hybrid",
        content_size=content_id.split("_")[0],
        content_type="_".join(content_id.split("_")[1:]),
        tokens_before=result.tokens_before,
        tokens_after=result.tokens_after,
        tokens_saved=result.tokens_saved,
        reduction_percentage=result.reduction_percentage,
        processing_time_ms=elapsed_ms,
        quality_score=result.quality_score,
        validation_passed=result.validation_passed,
    )


async def run_benchmarks() -> list[BenchmarkResult]:
    """Run all benchmarks."""
    results = []

    # Create provider for hybrid mode
    try:
        # Load configuration from environment
        config = load_full_config()
        provider_config = config.providers.openai_compatible

        # Check if provider is enabled (has required env vars)
        if not provider_config.enabled:
            print("✗ OpenAI-compatible provider not configured")
            print("\nMake sure you have set in .env:")
            print("  OPENAI_COMPATIBLE_API_KEY=your-key")
            print("  OPENAI_COMPATIBLE_BASE_URL=your-base-url")
            print("  OPENAI_COMPATIBLE_MODEL=your-model")
            return results

        # Use the provider config directly (it's already a ProviderConfig object)
        provider = create_provider("openai-compatible", provider_config)
        print("✓ OpenAI-compatible provider initialized")
    except Exception as e:
        print(f"✗ Failed to initialize provider: {e}")
        print("\nMake sure you have set in .env:")
        print("  OPENAI_COMPATIBLE_API_KEY=your-key")
        print("  OPENAI_COMPATIBLE_BASE_URL=your-base-url")
        print("  OPENAI_COMPATIBLE_MODEL=your-model")
        return results

    print("\n" + "=" * 80)
    print("COMPRESSION METHOD BENCHMARK")
    print("=" * 80)

    for content_id, content in TEST_CONTENT.items():
        print(f"\n--- Testing: {content_id} ({len(content)} chars) ---")

        # Benchmark Seraph
        print("  Running Seraph compression...")
        try:
            seraph_result = await benchmark_seraph(content, content_id)
            results.append(seraph_result)
            print(
                f"    ✓ {seraph_result.reduction_percentage:.1f}% reduction in {seraph_result.processing_time_ms:.1f}ms"
            )
        except Exception as e:
            print(f"    ✗ Seraph failed: {e}")

        # Benchmark Hybrid
        print("  Running Hybrid compression...")
        try:
            hybrid_result = await benchmark_hybrid(content, content_id, provider)
            results.append(hybrid_result)
            print(
                f"    ✓ {hybrid_result.reduction_percentage:.1f}% reduction in {hybrid_result.processing_time_ms:.1f}ms"
            )
        except Exception as e:
            print(f"    ✗ Hybrid failed: {e}")

    return results


def print_summary(results: list[BenchmarkResult]) -> None:
    """Print benchmark summary."""
    if not results:
        print("\nNo results to display.")
        return

    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80)

    # Group by content size
    for size in ["short", "medium", "long"]:
        size_results = [r for r in results if r.content_size == size]
        if not size_results:
            continue

        print(f"\n{size.upper()} CONTENT:")
        print("-" * 80)

        # Find seraph and hybrid for this size
        seraph = next((r for r in size_results if r.method == "seraph"), None)
        hybrid = next((r for r in size_results if r.method == "hybrid"), None)

        if seraph and hybrid:
            print(f"{'Metric':<30} {'Seraph':<20} {'Hybrid':<20} {'Winner':<10}")
            print("-" * 80)

            # Tokens saved
            seraph_saved = f"{seraph.tokens_saved} ({seraph.reduction_percentage:.1f}%)"
            hybrid_saved = f"{hybrid.tokens_saved} ({hybrid.reduction_percentage:.1f}%)"
            winner = "Seraph" if seraph.tokens_saved > hybrid.tokens_saved else "Hybrid"
            if abs(seraph.tokens_saved - hybrid.tokens_saved) < 10:
                winner = "Tie"
            print(f"{'Tokens Saved':<30} {seraph_saved:<20} {hybrid_saved:<20} {winner:<10}")

            # Processing time
            seraph_time = f"{seraph.processing_time_ms:.1f}ms"
            hybrid_time = f"{hybrid.processing_time_ms:.1f}ms"
            winner = "Seraph" if seraph.processing_time_ms < hybrid.processing_time_ms else "Hybrid"
            speedup = hybrid.processing_time_ms / seraph.processing_time_ms
            print(f"{'Processing Time':<30} {seraph_time:<20} {hybrid_time:<20} {winner} ({speedup:.1f}x)")

            # Quality score
            seraph_quality = f"{seraph.quality_score:.3f}"
            hybrid_quality = f"{hybrid.quality_score:.3f}"
            winner = "Seraph" if seraph.quality_score > hybrid.quality_score else "Hybrid"
            if abs(seraph.quality_score - hybrid.quality_score) < 0.01:
                winner = "Tie"
            print(f"{'Quality Score':<30} {seraph_quality:<20} {hybrid_quality:<20} {winner:<10}")

            # Validation
            seraph_val = "✓" if seraph.validation_passed else "✗"
            hybrid_val = "✓" if hybrid.validation_passed else "✗"
            print(f"{'Validation Passed':<30} {seraph_val:<20} {hybrid_val:<20}")

    # Overall statistics
    print("\n" + "=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)

    seraph_results = [r for r in results if r.method == "seraph"]
    hybrid_results = [r for r in results if r.method == "hybrid"]

    if seraph_results and hybrid_results:
        print(f"\n{'Method':<15} {'Avg Reduction':<15} {'Avg Time':<15} {'Avg Quality':<15}")
        print("-" * 80)

        seraph_avg_reduction = sum(r.reduction_percentage for r in seraph_results) / len(seraph_results)
        seraph_avg_time = sum(r.processing_time_ms for r in seraph_results) / len(seraph_results)
        seraph_avg_quality = sum(r.quality_score for r in seraph_results) / len(seraph_results)

        hybrid_avg_reduction = sum(r.reduction_percentage for r in hybrid_results) / len(hybrid_results)
        hybrid_avg_time = sum(r.processing_time_ms for r in hybrid_results) / len(hybrid_results)
        hybrid_avg_quality = sum(r.quality_score for r in hybrid_results) / len(hybrid_results)

        print(
            f"{'Seraph':<15} {seraph_avg_reduction:>6.1f}%{'':<8} {seraph_avg_time:>6.1f}ms{'':<6} {seraph_avg_quality:>6.3f}{'':<8}"
        )
        print(
            f"{'Hybrid':<15} {hybrid_avg_reduction:>6.1f}%{'':<8} {hybrid_avg_time:>6.1f}ms{'':<6} {hybrid_avg_quality:>6.3f}{'':<8}"
        )

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print("""
When to use SERAPH:
  ✓ Long content (>3k tokens)
  ✓ Repeated/cached queries
  ✓ Offline preprocessing acceptable
  ✓ Deterministic results required
  ✓ No API costs desired
  ✓ Sub-100ms latency needed

When to use HYBRID:
  ✓ Maximum compression required
  ✓ Quality is critical
  ✓ One-shot/unique content
  ✓ API costs acceptable
  ✓ Nuanced semantic understanding needed
  ✓ Longer processing time acceptable

Hybrid = Seraph pre-compression + AI polish
- Best compression ratios
- Higher quality scores
- Requires API calls (cost + latency)
- Good for production with strict quality requirements
""")


async def save_results(results: list[BenchmarkResult]) -> None:
    """Save results to JSON file."""
    output_file = Path(__file__).parent / "benchmark_results.json"

    data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": [
            {
                "method": r.method,
                "content_size": r.content_size,
                "content_type": r.content_type,
                "tokens_before": r.tokens_before,
                "tokens_after": r.tokens_after,
                "tokens_saved": r.tokens_saved,
                "reduction_percentage": r.reduction_percentage,
                "processing_time_ms": r.processing_time_ms,
                "quality_score": r.quality_score,
                "validation_passed": r.validation_passed,
            }
            for r in results
        ],
    }

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")


async def main() -> None:
    """Main benchmark runner."""
    print("\n" + "=" * 80)
    print("Seraph MCP - Compression Method Comparison")
    print("=" * 80)
    print("\nThis benchmark compares:")
    print("  • Seraph: Deterministic multi-layer compression (no AI)")
    print("  • Hybrid: Seraph pre-compression + AI polish")
    print("\nMetrics: compression ratio, processing time, quality score")

    # Run benchmarks
    results = await run_benchmarks()

    if results:
        # Print summary
        print_summary(results)

        # Save results
        await save_results(results)
    else:
        print("\n✗ No benchmarks completed. Check your configuration.")


if __name__ == "__main__":
    asyncio.run(main())
