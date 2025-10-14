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
- **22+ Tools** — Comprehensive optimization, budgeting, caching, and analytics
- **Production Ready** — Full monitoring, budget enforcement, and error handling

---

## Install

### Quick Start

Install from PyPI:
```bash
pip install seraph-mcp
```

Or run directly without installing:
```bash
uvx seraph-mcp
```

### Claude Desktop Integration

#### Zero Config (Works Immediately)
```json
{
  "mcpServers": {
    "seraph": {
      "command": "uvx",
      "args": ["seraph-mcp"],
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
      "command": "uvx",
      "args": ["seraph-mcp"],
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
      "command": "uvx",
      "args": ["seraph-mcp"],
      "env": {
        "OPENAI_API_KEY": "sk-...",
        "OPENAI_MODEL": "gpt-4",
        "REDIS_URL": "redis://localhost:6379/0",
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

### Core System (7 tools)
- `check_status()` — System health and status overview
- `get_cache_stats()` — Detailed cache performance metrics
- `cache_get()` — Retrieve values from cache
- `cache_set()` — Store values in cache with TTL
- `cache_delete()` — Delete specific cache keys
- `cache_clear()` — Clear entire cache
- `get_metrics()` — Observability metrics snapshot

### Context Optimization (4 tools)
- `count_tokens()` — Accurate token counting per model
- `estimate_cost()` — LLM API cost prediction
- `analyze_token_efficiency()` — Optimization opportunity analysis
- `optimize_context()` — AI/Seraph hybrid compression

### Budget Management (3 tools)
- `check_budget()` — Current spending status and limits
- `get_usage_report()` — Detailed cost analytics by period
- `forecast_spending()` — Predictive spending analysis

### Semantic Cache (5 tools)
- `lookup_semantic_cache()` — Find similar cached content
- `store_in_semantic_cache()` — Cache with semantic indexing
- `search_semantic_cache()` — Multi-result similarity search
- `get_semantic_cache_stats()` — Cache performance metrics
- `clear_semantic_cache()` — Maintenance operations

### Context Optimization Settings (2 tools)
- `get_optimization_settings()` — View current configuration
- `get_optimization_stats()` — Performance and savings metrics

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
git clone https://github.com/coderdayton/seraph-mcp.git
cd seraph-mcp

# Install dependencies
pip install uv
uv sync

# Run tests
uv run pytest

# Run in development mode
fastmcp dev src/server.py

# Or use the CLI entry point
uv run seraph-mcp
```

### Optional: Local Redis Setup

By default, Seraph MCP uses an in-memory cache. For persistent caching, you can optionally run a local Redis server:

```bash
# Start Redis (from project root or docker/ directory)
docker-compose -f docker/docker-compose.yml up -d

# Configure Seraph MCP to use Redis
export REDIS_URL=redis://localhost:6379/0

# Redis is now available at localhost:6379
# RedisInsight web UI at http://localhost:8001

# Stop Redis when done
docker-compose -f docker/docker-compose.yml down
```

**See [`docker/README.md`](docker/README.md) for:**
- Complete Docker setup instructions
- Development vs production configurations
- Troubleshooting and advanced options

---

## Documentation

- **[System Design](docs/SDD.md)** — Architecture and design decisions
- **[Context Optimization](src/context_optimization/README.md)** — Compression system details

---

## License

MIT — See [LICENSE](LICENSE) file.

---

<div align="center">

**[⬆ Back to top](#seraph-mcp)**

</div>
