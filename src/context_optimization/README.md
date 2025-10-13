# Context Optimization Module

Hybrid compression system combining AI-powered and deterministic multi-layer compression for optimal token reduction.

## Overview

The context optimization module provides **three compression approaches** that automatically select the best method based on content characteristics:

1. **AI Compression**: Fast, nuanced, best for short prompts (≤3k tokens)
2. **Seraph Compression**: Deterministic, cacheable, multi-layer (L1/L2/L3), best for long/recurring contexts (>3k tokens)
3. **Hybrid Mode**: Seraph pre-compress + AI polish for optimal results

## Quick Start

### Automatic Operation (Recommended)

```python
from src.context_optimization import wrap_provider
from src.providers import OpenAIProvider, ProviderConfig

# Create base provider
provider = OpenAIProvider(ProviderConfig(api_key="sk-..."))

# Wrap with automatic optimization
optimized_provider = wrap_provider(provider)

# Use normally - optimization happens automatically!
response = await optimized_provider.generate(
    prompt="Your long prompt here..."
)

# Check optimization results
print(f"Method: {response['optimization']['method']}")
print(f"Tokens saved: {response['optimization']['tokens_saved']}")
print(f"Quality: {response['optimization']['quality_score']}")
```

### Manual Control

```python
from src.context_optimization import optimize_content, ContextOptimizationConfig

# Configure optimization
config = ContextOptimizationConfig(
    enabled=True,
    compression_method="auto",  # auto, ai, seraph, hybrid
    seraph_token_threshold=3000,
    quality_threshold=0.90,
    max_overhead_ms=100.0,
)

# Optimize content
result = await optimize_content(content, provider, config)

print(f"Tokens before: {result.tokens_before}")
print(f"Tokens after: {result.tokens_after}")
print(f"Reduction: {result.reduction_percentage:.1f}%")
print(f"Quality: {result.quality_score:.2f}")
print(f"Method used: {result.method}")
```

## Compression Methods

### 1. AI Compression (Fast & Nuanced)

**Best for:**
- Short prompts (≤3k tokens)
- One-shot requests
- Content requiring nuance preservation
- Subtle constraints and cross-sentence references

**How it works:**
1. LLM compresses text using intelligent prompt (LLMLingua approach)
2. Second LLM validates quality (0-1 score)
3. Automatic rollback if quality < threshold

**Performance:**
- **Speed**: Sub-100ms
- **Reduction**: 20-40%
- **Quality**: ≥90%

**Example:**
```python
config = ContextOptimizationConfig(compression_method="ai")
optimizer = ContextOptimizer(config, provider=provider)
result = await optimizer.optimize(short_content)
```

### 2. Seraph Compression (Deterministic & Cacheable)

**Best for:**
- Long prompts (>3k tokens)
- Repeated queries on same content
- Multi-session memory
- Deterministic, reproducible results

**How it works - Three-Tier Pipeline:**

#### Tier-1: Structural Compression (500x-style)
Builds three layers using deterministic algorithms:

- **L1 (0.2% ratio)**: Ultra-small skeleton
  - Bullets from anchor extraction
  - Entities, quantities, dates, URLs
  - SimHash deduplication
  
- **L2 (1% ratio)**: Compact abstracts
  - Section summaries
  - BM25 salience scoring
  - Top-ranked chunks
  
- **L3 (5% ratio)**: Factual extracts
  - Top salient content
  - Structure-preserving
  - Extractive (no generation)

#### Tier-2: Dynamic Context Pruning (DCP)
- Importance + novelty + locality scoring
- Greedy selection under token budget
- Further compresses L3 to ~8% of original

#### Tier-3: Hierarchical Query-Time
- Optional LLMLingua-2 for runtime polish
- Query-specific layer selection
- Falls back to internal rules

**Performance:**
- **Speed**: Sub-100ms for queries after build
- **Reduction**: 20-50% (configurable via ratios)
- **Quality**: 88-95% (structure-preserving)
- **Caching**: Same input → same output, integrity-hashed

**Example:**
```python
config = ContextOptimizationConfig(
    compression_method="seraph",
    seraph_l1_ratio=0.002,  # Ultra-small
    seraph_l2_ratio=0.01,   # Compact
    seraph_l3_ratio=0.05,   # Factual
)
optimizer = ContextOptimizer(config, provider=None)  # No provider needed
result = await optimizer.optimize(long_content)
```

### 3. Hybrid Mode (Best of Both)

**Best for:**
- Tight budgets + quality requirements
- Iterative refinement
- Maximum reduction with quality preservation

**How it works:**
1. Seraph pre-compresses to L2 layer (deterministic structure)
2. AI polishes the compressed content (semantic enhancement)
3. Quality validation ensures improvement

**Example:**
```python
config = ContextOptimizationConfig(
    compression_method="hybrid",
    max_overhead_ms=150.0,  # Allow more time
)
optimizer = ContextOptimizer(config, provider=provider)
result = await optimizer.optimize(medium_content)
```

## Configuration

### Environment Variables

**Core Settings:**
```bash
CONTEXT_OPTIMIZATION_ENABLED=true
CONTEXT_OPTIMIZATION_COMPRESSION_METHOD=auto  # auto, ai, seraph, hybrid
CONTEXT_OPTIMIZATION_SERAPH_TOKEN_THRESHOLD=3000  # AI for ≤3k, Seraph for >3k
```

**Quality & Performance:**
```bash
CONTEXT_OPTIMIZATION_QUALITY_THRESHOLD=0.90  # Min quality score (0-1)
CONTEXT_OPTIMIZATION_MAX_OVERHEAD_MS=100.0   # Max processing time
```

**Seraph Layer Ratios (Advanced):**
```bash
CONTEXT_OPTIMIZATION_SERAPH_L1_RATIO=0.002  # L1: 0.2% (skeleton)
CONTEXT_OPTIMIZATION_SERAPH_L2_RATIO=0.01   # L2: 1% (abstracts)
CONTEXT_OPTIMIZATION_SERAPH_L3_RATIO=0.05   # L3: 5% (extracts)
```

### Programmatic Configuration

```python
from src.context_optimization import ContextOptimizationConfig

config = ContextOptimizationConfig(
    enabled=True,
    compression_method="auto",
    seraph_token_threshold=3000,
    quality_threshold=0.90,
    max_overhead_ms=100.0,
    seraph_l1_ratio=0.002,
    seraph_l2_ratio=0.01,
    seraph_l3_ratio=0.05,
)
```

## Decision Matrix

| Scenario | Tokens | Reuse | Best Method | Why |
|----------|--------|-------|-------------|-----|
| Chat prompt | 500 | One-shot | **AI** | Fast, preserves nuance |
| Document summary | 10k | One-shot | **AI** | Better semantic understanding |
| Multi-doc context | 50k | Repeated | **Seraph** | Build once, query many times |
| Tool logs | 100k | Persistent | **Seraph** | Deterministic, cacheable |
| Transcript compression | 20k | Multiple queries | **Seraph** | BM25 retrieval works well |
| Code context | 5k | Evolving | **Hybrid** | Structure + semantic polish |
| Tight budget | Any | Any | **Hybrid** | Maximize reduction, maintain quality |

## Auto Mode (Recommended)

When `compression_method="auto"` (default), the system automatically selects:

```python
if token_count <= seraph_token_threshold:
    use AI compression      # Fast, nuanced for short content
else:
    use Seraph compression  # Efficient, cacheable for long content
```

## Advanced Usage

### Standalone Seraph Compressor

For advanced use cases requiring direct multi-layer access:

```python
from src.context_optimization import SeraphCompressor

# Create compressor
compressor = SeraphCompressor(seed=7)

# Build all layers
result = compressor.build(corpus_text)

# Access layers
print(f"L1 (skeleton): {result.l1}")
print(f"L2 (abstracts): {result.l2}")
print(f"L3 (extracts): {result.l3}")

# Query L3 layer
top_results = compressor.query(result, "What are the key points?", k=5)
for score, text in top_results:
    print(f"{score:.3f}: {text}")

# Pack for storage
compressor.pack(result, "compressed.json.gz")
```

### Statistics & Monitoring

```python
# Get optimization statistics
stats = optimizer.get_stats()

print(f"Total optimizations: {stats['total_optimizations']}")
print(f"Successful: {stats['successful_optimizations']}")
print(f"Total tokens saved: {stats['total_tokens_saved']}")
print(f"Avg quality: {stats['avg_quality_score']:.2f}")
print(f"Avg reduction: {stats['avg_reduction_percentage']:.1f}%")

# Method usage breakdown
print(f"AI: {stats['method_usage']['ai']}")
print(f"Seraph: {stats['method_usage']['seraph']}")
print(f"Hybrid: {stats['method_usage']['hybrid']}")

# Clear caches
optimizer.clear_cache()
```

## Performance Characteristics

### AI Compression
- **Latency**: 50-100ms per request
- **API Calls**: 2 per optimization (compress + validate)
- **Caching**: Results cached by content hash
- **Deterministic**: No (generative)

### Seraph Compression
- **Build Time**: 100ms - 2s (depending on size)
- **Query Time**: <10ms (BM25 on CPU)
- **API Calls**: 0 (pure algorithmic)
- **Caching**: Same input → same output
- **Deterministic**: Yes (seeded)

### Hybrid Compression
- **Latency**: 100-200ms (Seraph + AI)
- **API Calls**: 2 (for AI polish)
- **Best Of**: Determinism + semantic enhancement

## Quality Validation

All methods include automatic quality validation:

1. **Semantic Similarity**: Compare original vs. compressed
2. **Threshold Check**: Must meet configured quality_threshold
3. **Automatic Rollback**: Revert to original if quality too low
4. **Quality Scoring**: 0-1 scale where:
   - 1.0 = Perfect preservation
   - 0.9-0.95 = Excellent (Seraph typical)
   - 0.8-0.9 = Good
   - <0.8 = Triggers rollback

## Files

- `config.py` - Configuration models and environment loading
- `models.py` - Pydantic data models (OptimizationResult, FeedbackRecord)
- `optimizer.py` - Main ContextOptimizer class with all three methods
- `middleware.py` - Automatic middleware for provider wrapping
- `seraph_compression.py` - Standalone three-tier compression system
- `strategies.py` - Optimization strategies (if applicable)
- `validator.py` - Quality validation logic (if applicable)

## Examples

See `examples/context_optimization/hybrid_compression_demo.py` for comprehensive demonstrations of all three methods.

## Integration with Budget Management

Optimization automatically integrates with budget tracking:

```python
from src.budget_management import BudgetTracker

tracker = BudgetTracker()
optimizer = ContextOptimizer(config, provider=provider, budget_tracker=tracker)

# Optimization savings are automatically recorded!
result = await optimizer.optimize(content)

# Check savings
print(f"Cost saved: ${result.cost_savings_usd:.4f}")
```

## Troubleshooting

**"Optimization not happening"**
- Check `CONTEXT_OPTIMIZATION_ENABLED=true`
- Verify content has >100 tokens (too short skips optimization)
- Check provider is configured (needed for AI/hybrid modes)

**"Quality too low / frequent rollbacks"**
- Lower `quality_threshold` (e.g., 0.85 instead of 0.90)
- Use Seraph method for structure-preserving compression
- Increase Seraph layer ratios for less aggressive compression

**"Too slow"**
- Reduce `max_overhead_ms` to fail faster
- Use Seraph for long content (builds once, queries fast)
- Enable caching (automatic)

**"Seraph compression errors"**
- Optional dependencies may be missing: `pip install tiktoken blake3`
- For full features: `pip install sentence-transformers llmlingua`
- Basic functionality works without optional deps

## License

MIT License - See LICENSE file for details.