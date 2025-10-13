<div align="center">

<img src="https://i.imgur.com/GCSZhST.png" alt="Seraph MCP" width="180"/>

# Seraph MCP

**Intelligent token optimization for AI agents**

[![Version](https://img.shields.io/badge/version-1.0.0-blue?style=flat-square)](https://github.com/yourusername/seraph-mcp)
[![MCP](https://img.shields.io/badge/MCP-Server-green?style=flat-square)](https://modelcontextprotocol.io)
[![License](https://img.shields.io/badge/license-MIT-purple?style=flat-square)](LICENSE)

Hybrid compression system delivering 40-60% cost reduction with >90% quality preservation.
Zero configuration required. Features auto-enable based on what you configure.

[Quick Start](#quick-start) • [Configuration](#configuration) • [Tools](#tools) • [Docs](docs/)

</div>

---

## Features

- **Zero-Config Operation** — Works immediately with memory cache + deterministic compression
- **Hybrid Compression** — AI-powered (fast, nuanced) + Seraph multi-layer (deterministic, cacheable)
- **Auto-Enabling** — Providers, cache, and features activate automatically based on configuration
- **40-60% Cost Reduction** — Through intelligent optimization and compression
- **>90% Quality Preservation** — Multi-dimensional validation with automatic rollback
- **Sub-100ms Processing** — Minimal latency impact on requests
- **18+ Tools** — Comprehensive optimization, budgeting, caching, and analytics
- **Production Ready** — Full monitoring, budget enforcement, and error handling

---

## Install

### Three Usage Modes

#### Zero Config (Works Immediately)
```json
{
  "mcpServers": {
    "seraph": {
      "command": "npx",
      "args": ["-y", "seraph-mcp"],
      "env": {}
    }
  }
}
```
**Auto-enables:** Memory cache + Seraph compression (no AI needed)

#### Basic (Most Common)
```json
{
  "mcpServers": {
    "seraph": {
      "command": "npx",
      "args": ["-y", "seraph-mcp"],
      "env": {
        "OPENAI_API_KEY": "sk-...",
        "OPENAI_MODEL": "gpt-4",
        "MONTHLY_BUDGET_LIMIT": "200.0"
      }
    }
  }
}
```
**Auto-enables:** OpenAI provider + hybrid compression + budget tracking

#### Production
```json
{
  "mcpServers": {
    "seraph": {
      "command": "npx",
      "args": ["-y", "seraph-mcp"],
      "env": {
        "OPENAI_API_KEY": "sk-...",
        "OPENAI_MODEL": "gpt-4",
        "REDIS_URL": "redis://localhost:6379",
        "DAILY_BUDGET_LIMIT": "10.0",
        "MONTHLY_BUDGET_LIMIT": "200.0"
      }
    }
  }
}
```
**Auto-enables:** OpenAI + Redis cache + budget enforcement + hybrid compression

---

## Configuration

### Auto-Enabling System

Features automatically enable when you provide configuration:

| You Configure | Auto-Enables |
|--------------|--------------|
| API Key + Model | That provider + hybrid compression + budget tracking |
| + `REDIS_URL` | Redis persistent cache |
| + Budget Limits | Budget enforcement with alerts |
| Nothing | Memory cache + Seraph-only compression |

### Providers

Configure **one or more** providers. Each requires API key + model:

| Provider | API Key | Model | Base URL |
|----------|---------|-------|----------|
| **OpenAI** | `OPENAI_API_KEY` | `OPENAI_MODEL`<br/>`gpt-4`, `gpt-3.5-turbo` | Optional |
| **Anthropic** | `ANTHROPIC_API_KEY` | `ANTHROPIC_MODEL`<br/>`claude-3-5-sonnet-20241022` | Optional |
| **Google AI** | `GEMINI_API_KEY` | `GEMINI_MODEL`<br/>`gemini-1.5-pro` | Optional |
| **OpenAI-Compatible** | `OPENAI_COMPATIBLE_API_KEY` | `OPENAI_COMPATIBLE_MODEL`<br/>`meta-llama/Llama-3-8b-chat-hf` | **Required**<br/>`OPENAI_COMPATIBLE_BASE_URL` |

**Example (Together AI):**
```json
"OPENAI_COMPATIBLE_API_KEY": "your-key",
"OPENAI_COMPATIBLE_MODEL": "meta-llama/Llama-3-70b-chat-hf",
"OPENAI_COMPATIBLE_BASE_URL": "https://api.together.xyz/v1"
```

### Context Optimization

Hybrid compression with automatic method selection:

| Variable | Default | Description |
|----------|---------|-------------|
| `CONTEXT_OPTIMIZATION_ENABLED` | `true` | Enable/disable optimization |
| `CONTEXT_OPTIMIZATION_COMPRESSION_METHOD` | `auto` | `auto`, `ai`, `seraph`, `hybrid` |
| `CONTEXT_OPTIMIZATION_QUALITY_THRESHOLD` | `0.90` | Min quality (0-1) |
| `CONTEXT_OPTIMIZATION_MAX_OVERHEAD_MS` | `100.0` | Max processing time |

**Compression Methods:**
- **`auto`** — Seraph if no provider; else AI for ≤3k tokens, Seraph for >3k
- **`ai`** — AI-powered (requires provider, fast, nuanced)
- **`seraph`** — Deterministic multi-layer (works without provider, cacheable)
- **`hybrid`** — Seraph pre-compress + AI polish (requires provider)

### Budget Management

| Variable | Description |
|----------|-------------|
| `DAILY_BUDGET_LIMIT` | Daily spending cap (USD) |
| `MONTHLY_BUDGET_LIMIT` | Monthly spending cap (USD) |
| `BUDGET_ALERT_THRESHOLDS` | Alert percentages (default: `50,75,90`) |

### Cache

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | None | Auto-enables Redis when set |
| `CACHE_TTL_SECONDS` | `3600` | Cache expiration (seconds) |
| `CACHE_MAX_SIZE` | `1000` | Max entries (memory only) |

---

## Tools

### Context Optimization
- `optimize_context()` — Compress content with quality preservation
- `get_optimization_settings()` — View current configuration
- `get_optimization_stats()` — Performance metrics

### Budget Management
- `check_budget()` — Current spending and limits
- `set_budget()` — Configure spending limits
- `get_usage_report()` — Detailed cost analytics
- `forecast_spending()` — Predict future costs

### Semantic Cache
- `lookup_cache()` — Find similar cached content
- `store_in_cache()` — Cache with metadata
- `analyze_cache_performance()` — Hit rates and metrics
- `clear_cache()` — Maintenance operations

### Model Intelligence
- `find_best_model()` — Optimal model recommendations
- `compare_model_costs()` — Cross-provider pricing
- `estimate_request_cost()` — Cost prediction
- `get_model_recommendations()` — AI-powered selection

### System
- `check_status()` — System health overview
- `get_performance_metrics()` — Real-time metrics
- `run_health_check()` — Comprehensive diagnostics

---

## Performance

| Metric | Target | Typical |
|--------|--------|---------|
| Cost Reduction | 40-60% | 45% |
| Quality Preservation | >90% | 92-95% |
| Optimization Overhead | <100ms | 60-80ms |
| Cache Hit Rate | 60-80% | 70% |
| Cache Lookup | <10ms | 5ms |

---

## Architecture

### Compression System

**Three-Tier Seraph Compression:**
- **L1 (0.2%)** — Skeleton bullets from anchor extraction
- **L2 (1%)** — Compact section summaries via BM25 salience
- **L3 (5%)** — Top factual chunks with structure preservation

**AI Compression:**
- LLM-powered prompt compression (LLMLingua approach)
- Quality validation with automatic rollback
- 20-40% reduction with semantic preservation

**Hybrid Mode:**
- Seraph pre-compression to L2
- AI polish for readability
- Combined benefits: determinism + semantic enhancement

### Auto-Detection

```
Provider configured? → Enable hybrid compression + budget tracking
REDIS_URL set? → Switch to Redis cache
No provider? → Use Seraph-only (works without AI)
```

---

## Development

```bash
# Clone
git clone https://github.com/yourusername/seraph-mcp.git
cd seraph-mcp

# Install
pip install uv
uv sync

# Test
uv run pytest

# Run
fastmcp dev src/server.py
```

---

## Documentation

- **[System Design](docs/SDD.md)** — Architecture and design decisions
- **[Context Optimization](src/context_optimization/README.md)** — Compression system details
- **[Provider Guide](docs/PROVIDERS.md)** — Provider integration reference

---

## License

MIT — See [LICENSE](LICENSE) file.

---

<div align="center">

**[⬆ Back to top](#seraph-mcp)**

</div>
