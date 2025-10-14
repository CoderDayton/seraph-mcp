# Seraph MCP Benchmarks

This directory contains performance benchmarks for the Seraph MCP compression system.

## Available Benchmarks

### 1. Compression Comparison (`compression_comparison.py`)

Compares Seraph and Hybrid compression methods across different content sizes and types.

**What it measures:**
- Compression ratio (tokens saved)
- Processing time (latency)
- Quality score (semantic similarity)
- Validation success rate

**Content types tested:**
- Short technical content (~200 tokens)
- Medium documentation (~1,500 tokens)
- Long technical content (~8,000 tokens)

**Usage:**
```bash
# Set up environment variables in .env:
OPENAI_COMPATIBLE_API_KEY=your-api-key
OPENAI_COMPATIBLE_BASE_URL=your-base-url
OPENAI_COMPATIBLE_MODEL=your-model

# Run benchmark as a module
python -m benchmarks.compression_comparison
```

**Output:**
- Console summary with detailed metrics
- JSON results file: `benchmarks/benchmark_results.json`

## Understanding the Results

### Compression Methods

**Seraph Compression:**
- Deterministic multi-layer compression (L1/L2/L3)
- No AI/LLM required
- Fast (<100ms typical)
- Cacheable results
- Best for: Long content, repeated queries, offline preprocessing

**Hybrid Compression:**
- Seraph pre-compression + AI polish
- Requires LLM API access
- Higher quality scores
- Better semantic preservation
- Best for: Maximum compression, quality-critical applications

### Key Metrics

**Tokens Saved / Reduction Percentage:**
- Higher is better
- Indicates how much content was compressed
- Typical range: 20-60% reduction

**Processing Time:**
- Lower is better
- Seraph is typically 5-50x faster than Hybrid
- Seraph: <100ms, Hybrid: varies with API latency

**Quality Score:**
- Higher is better (0.0 to 1.0)
- Measures semantic similarity to original
- Target: ≥0.90 for production use

**Validation Passed:**
- Boolean indicating if compression meets quality threshold
- Based on multiple validation checks

## Interpreting Results

### When Seraph Wins
- Faster processing time (always)
- Good enough quality for most use cases
- Zero API costs
- Deterministic results

### When Hybrid Wins
- Higher compression ratios
- Better quality scores
- More nuanced semantic preservation
- Better for complex/technical content

### Decision Matrix

| Use Case | Content Size | Reuse | Method | Why |
|----------|-------------|-------|--------|-----|
| API docs | 10k tokens | High | Seraph | Fast, cacheable |
| Chat context | 2k tokens | Low | Hybrid | Quality critical |
| Code context | 5k tokens | Medium | Hybrid | Semantic nuance |
| Log analysis | 100k tokens | High | Seraph | Speed + no cost |
| Multi-session | Any | High | Seraph | Deterministic cache |

## Adding New Benchmarks

To add a new benchmark:

1. Create a new Python file in this directory
2. Follow the naming convention: `{feature}_benchmark.py`
3. Include docstring with:
   - Purpose and metrics measured
   - Usage instructions
   - Configuration requirements
4. Output results to JSON for tracking over time
5. Update this README with benchmark description

### Benchmark Template

```python
"""
{Feature} Benchmark
==================

Brief description of what this benchmark measures.

Usage:
    python -m benchmarks.{feature}_benchmark

Requirements:
    List any environment variables or setup needed
"""

import asyncio
import json
import time
from pathlib import Path

# Your benchmark implementation here

async def main():
    """Main benchmark runner."""
    # Run tests
    results = await run_benchmark()

    # Save results
    output_file = Path(__file__).parent / f"{feature}_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    asyncio.run(main())
```

## CI Integration

Benchmarks can be run in CI to track performance over time:

```yaml
- name: Run benchmarks
  run: |
    uv run python -m benchmarks.compression_comparison
  env:
    OPENAI_COMPATIBLE_API_KEY: ${{ secrets.OPENAI_COMPATIBLE_API_KEY }}
    OPENAI_COMPATIBLE_BASE_URL: ${{ secrets.OPENAI_COMPATIBLE_BASE_URL }}
    OPENAI_COMPATIBLE_MODEL: ${{ secrets.OPENAI_COMPATIBLE_MODEL }}
```

## Best Practices

1. **Consistent Test Data**: Use the same test content across runs for comparability
2. **Multiple Runs**: Run each test multiple times and report averages
3. **Warmup**: Run a warmup iteration to avoid cold-start effects
4. **Environment**: Document hardware, Python version, and dependencies
5. **Version Results**: Save results with timestamps for historical tracking
6. **Statistical Significance**: Use appropriate statistical tests for comparisons

## Contributing

When adding benchmarks:
- Keep test data representative of real-world usage
- Document all assumptions and limitations
- Include error handling and validation
- Make output human-readable and machine-parseable
- Add visualization if helpful (charts, graphs)

## License

MIT - See LICENSE file in project root
