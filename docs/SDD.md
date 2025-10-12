# Seraph MCP — System Design Document (SDD)

Version: 3.0
Author: Senior Software Engineer / Systems Architect
Status: Canonical system design for Seraph MCP with monolithic architecture
Date: 2025-10-12
Last Updated: 2025-01-13 (Refactored to monolithic architecture with feature flags)

---

## Purpose
This document defines the architecture for Seraph MCP, a comprehensive AI optimization platform delivered as a single, integrated package. The platform provides token optimization, model routing, semantic caching, context optimization, budget management, and quality preservation capabilities. All features are included in the main package but can be selectively enabled/disabled via feature flags.

### Architectural Decision: Monolithic with Feature Flags

**Why Monolithic Over Plugins:**

After extensive analysis from user and maintainer perspectives, we chose a monolithic architecture with internal modularity over a plugin system for the following reasons:

**User Benefits:**
- ✅ Single installation: `npx -y seraph-mcp` (or `pip install seraph-mcp`)
- ✅ All features work out-of-the-box without additional plugin management
- ✅ One version number to track (no plugin compatibility matrix)
- ✅ Unified documentation in one place
- ✅ Simpler troubleshooting (no "which plugin?" questions)
- ✅ Matches marketing promise: "Comprehensive AI Optimization Platform"

**Maintainer Benefits:**
- ✅ Single release cycle (one version, one changelog)
- ✅ Simpler CI/CD (one build, one test suite, one deployment)
- ✅ Integrated testing (test features together as users will use them)
- ✅ No cross-package compatibility testing
- ✅ Unified issue tracking
- ✅ 90% less maintenance overhead

**When Plugins Would Make Sense:**
- If third-party developers need to extend the platform
- If there are 50+ features and users need à la carte selection
- If features have conflicting dependencies
- If the team grows to 10+ people working on separate codebases

**Current Reality:** Single team, 6 core features (all part of value proposition), harmonious dependencies, users want the complete platform.

Principles:
- Comprehensive: include all AI optimization features users need
- Modular Internally: organized codebase with clear separation of concerns
- Feature Flags: enable/disable capabilities via configuration
- Single Source of Truth: one factory/adapter per capability (cache, observability)
- Typed & Traceable: typed configuration and comprehensive observability
- Safe-by-default: conservative defaults (timeouts, budgets, circuit breakers)
- Stdio MCP only: uses Model Context Protocol (MCP) over stdio, not HTTP

---

## High-level Architecture

Components (All Integrated):
- **Core Runtime**
  - `src/server.py` — FastMCP stdio server with all MCP tools and lifecycle
  - `src/config/` — typed configuration models (Pydantic) with feature flags
  - `src/cache/` — canonical cache factory and backends (memory, Redis)
  - `src/observability/` — observability adapter (metrics, traces, logs)
  - `src/errors.py` — standardized error types

- **AI Optimization Features** (All included, feature-flagged)
  - `src/token_optimization/` — Token reduction and cost estimation
    - Token counting for 15+ models (OpenAI, Anthropic, Google, Mistral)
    - 5 optimization strategies (whitespace, redundancy, compression, etc.)
    - Cost estimation with real-time pricing data
    - Quality preservation with configurable thresholds
  - `src/model_routing/` — Intelligent model selection (future)
  - `src/semantic_cache/` — Vector-based similarity caching (future)
  - `src/context_optimization/` — AI-powered content reduction (future)
  - `src/budget_management/` — Cost tracking and enforcement (future)
  - `src/quality_preservation/` — Multi-dimensional validation (future)

- **Tooling & Docs**
  - `examples/` — usage samples
  - `tests/` — comprehensive unit/integration tests
  - `docs/` — design docs & SDD (this file)

Data flow:
1. MCP client connects via stdio to `src/server.py`
2. Server validates request and loads config via `src/config`
3. Feature flags determine which tools are active
4. MCP tools route to feature-specific handlers
5. Cache access uses `src/cache/factory.get_cache()` (memory or Redis)
6. All operations emit metrics/traces/logs via `src/observability`
7. Features integrate seamlessly with core cache and observability

Design principles:
- All features included in one package for simplicity
- Feature flags enable/disable capabilities without uninstalling
- Internal modularity maintains code organization
- Shared dependencies (Redis, tiktoken, anthropic) installed once
- Transport is MCP stdio only (no HTTP)

Core Interfaces (stable):
- Cache interface (async):
  - `async def get(key: str) -> Optional[Any]`
  - `async def set(key: str, value: Any, ttl: int | None) -> bool`
  - `async def delete(key: str) -> bool`
  - `async def exists(key: str) -> bool`
  - `async def clear() -> bool`
  - `async def get_stats() -> dict`
  - Optional bulk helpers: `get_many`, `set_many`, `delete_many`
- Observability adapter:
  - `increment(metric, value=1.0, tags=None)`
  - `gauge(metric, value, tags=None)`
  - `histogram(metric, value, tags=None)`
  - `event(name, payload)`
  - `trace(span_name, tags=None)` context manager

---

## Protocol & Transport

- MCP stdio Protocol (mandatory):
  - JSON-RPC 2.0 over stdin/stdout.
  - Compatible with Claude Desktop and other MCP clients.
  - No HTTP endpoints or REST APIs in core.

- FastMCP Framework:
  - Tools via `@mcp.tool()`
  - Lifecycle via `@mcp.lifespan()`
  - Config via `fastmcp.json`

---

## MCP Tools

### Core Tools (Always Available)
- `check_status(include_details: bool)` — System health and status
- `get_cache_stats()` — Cache metrics and stats (backend, hit-rate, etc.)
- `cache_get(key: str)` — Retrieve value from cache
- `cache_set(key: str, value: Any, ttl: Optional[int])` — Store value in cache
- `cache_delete(key: str)` — Delete key from cache
- `cache_clear()` — Clear all cache entries
- `get_metrics()` — Observability metrics snapshot

### Token Optimization Tools (Feature Flag: `features.token_optimization`)
- `optimize_tokens(content: str, target_reduction: Optional[float], model: str, strategies: Optional[List[str]])` — Reduce token count while preserving quality
- `count_tokens(content: str, model: str, include_breakdown: bool)` — Accurate token counting for any model
- `estimate_cost(content: str, model: str, operation: str, output_tokens: Optional[int])` — Calculate API costs before requests
- `analyze_token_efficiency(content: str, model: str)` — Identify optimization opportunities

### Future Tools (Feature Flags: `features.*`)
- Model routing, semantic cache, context optimization, budget management, and quality preservation tools will be added as features are implemented
- `cache_get(key: str)` — Retrieve from cache.
- `cache_set(key, value, ttl)` — Store in cache (optional TTL).
- `cache_delete(key)` — Delete a cache key.
- `cache_clear()` — Clear all cache entries (namespace-scoped).
- `get_metrics()` — Observability metrics snapshot.

---

## Configuration & Environment

### Feature Flags

All features can be enabled/disabled via configuration:

```python
class FeatureFlags(BaseModel):
    token_optimization: bool = True   # Enable token optimization
    model_routing: bool = False       # Enable model routing (future)
    semantic_cache: bool = False      # Enable semantic caching (future)
    context_optimization: bool = False  # Enable context optimization (future)
    budget_management: bool = False   # Enable budget management (future)
    quality_preservation: bool = False  # Enable quality preservation (future)
```

### Configuration Schema

Configuration is typed using Pydantic and loaded in order:
1. Defaults in code
2. `.env` (if present)
3. Environment variables

Mandatory environment variables:
- `ENVIRONMENT` = production | staging | development
- `CACHE_BACKEND` = memory | redis
  - Defaults to `memory` if unset (backward-compatible).
- `CACHE_TTL_SECONDS` = default TTL for cache entries (0 = no expiry)
- `CACHE_MAX_SIZE` = max entries (memory backend)
- `CACHE_NAMESPACE` = key namespace/prefix (applies to all backends)
- `OBSERVABILITY_BACKEND` = simple | prometheus | datadog
- `ENABLE_METRICS` = true | false
- `ENABLE_TRACING` = true | false
- `LOG_LEVEL` = DEBUG | INFO | WARNING | ERROR | CRITICAL

Conditional (required when enabled by the toggle):
- When `CACHE_BACKEND=redis`:
  - `REDIS_URL` (required) e.g., `redis://localhost:6379/0` or `rediss://...`
  - `REDIS_MAX_CONNECTIONS` (default 10)
  - `REDIS_SOCKET_TIMEOUT` seconds (default 5)

- When `OBSERVABILITY_BACKEND=datadog`:
  - `DATADOG_API_KEY` (required)
  - `DATADOG_SITE` (default `datadoghq.com`)

Security and secrets:
- Never commit secrets.
- Use environment-bound secret stores in production.
- Always use TLS for external calls (e.g., `rediss://` for Redis over TLS).

Example `.env` snippets:
- Memory (default):
  - `CACHE_BACKEND=memory`
  - `CACHE_TTL_SECONDS=3600`
  - `CACHE_MAX_SIZE=1000`
  - `CACHE_NAMESPACE=seraph`
- Redis (toggle on):
  - `CACHE_BACKEND=redis`
  - `REDIS_URL=redis://localhost:6379/0`
  - `REDIS_MAX_CONNECTIONS=20`
  - `REDIS_SOCKET_TIMEOUT=5`
  - `CACHE_TTL_SECONDS=3600`
  - `CACHE_NAMESPACE=seraph`

---

## Cache System

Single factory (canonical):
- `src/cache/factory.py` creates and returns cache instances.
- Selection is based on `CACHE_BACKEND`:
  - `memory` → In-process LRU cache with TTL and namespace support.
  - `redis` → Redis-backed cache with JSON serialization, TTL, namespace, and batch ops.

Memory backend:
- LRU eviction at capacity (`CACHE_MAX_SIZE`).
- Per-key TTL with optional default TTL.
- Namespace-prefixed keys.
- Lock-protected, async-safe operations.

Redis backend (core optional):
- Async Redis client (redis-py 4+) with JSON serialization.
- Namespaced keys (`<namespace>:<key>`).
- Per-key TTL (`EX` seconds); `ttl=None` → use default TTL; `ttl=0` → no expiry.
- Batch ops using MGET / pipelined SET/DEL for efficiency.
- `get_stats()` includes hit/miss counters and light Redis INFO when available.
- Toggle behavior:
  - Memory remains default for simplicity and zero external dependencies.
  - Switch to Redis by setting `CACHE_BACKEND=redis` and valid `REDIS_URL`.

Resource lifecycle:
- All cache instances must be closed on shutdown (`close_all_caches()`).
- The server lifecycle hook (`@mcp.lifespan`) initializes cache and closes it gracefully.

---

## Observability & Monitoring

- Centralized in `src/observability`.
- Emit:
  - Tool invocation counts
  - Cache hits/misses
  - Latency histograms
  - Error counts per type
- Structured logs (JSON), include trace IDs.
- `get_metrics()` tool surfaces current metrics snapshot.

---

## Error Handling & Resiliency

- Standardized exceptions in `src/errors.py`.
- MCP tools catch and return structured error responses.
- Async operations have strict timeouts (default 30s; configurable).
- Graceful shutdown flushes observability buffers and closes caches.

---

## Packaging, Deployment & Runtime

- Core package contains only `src/` and minimal runtime dependencies.
- Redis client dependency is allowed in core (as an optional runtime path) to support the Redis backend toggle.
- Deployment via FastMCP:
  - Dev: `fastmcp dev src/server.py`
  - Production: Configure MCP client to run the stdio server.
- Containerization:
  - Use slim Python base image
  - Entrypoint: `fastmcp run fastmcp.json`

---

## CI / CD & Quality Gates

Mandatory on every PR:
- `ruff` linting and formatting
- `mypy` type checking
- Unit test coverage threshold (core default ≥ 85%)
- Integration smoke tests (server starts, tools callable)
- Dependency vulnerability scan
- Secrets scan
- Tests must accompany changes in `src/`

Code review checklist:
- Adherence to SDD (single factory/adapter per capability).
- No HTTP in core (stdio MCP only).
- Config typed and validated.
- Metrics/traces/logging for new code.
- Redis usage guarded behind CACHE_BACKEND toggle.

---

## Testing Strategy

### Test Organization
- `tests/unit/` — Unit tests for individual modules
- `tests/integration/` — Integration tests across features
- `tests/performance/` — Performance benchmarks

### Coverage Requirements
- Overall: ≥85% code coverage
- Core modules: ≥90% coverage
- Feature modules: ≥85% coverage
- Critical paths: 100% coverage

### Test Scope
- Core cache and observability
- Token optimization (counter, optimizer, cost estimator)
- Feature flag behavior
- Configuration validation
- Error handling and edge cases

- Unit tests: `tests/unit/` (cache, config, observability, errors).
- Integration tests: `tests/integration/` (MCP tools end-to-end).
- Smoke tests: cache operations and lifecycle hooks.
- Performance: optional suite before major releases (cache hit rates, latencies).

---

## Feature Module Architecture (Internal Organization)

### Module Structure

Each feature is organized as a self-contained module within `src/`:

```
src/seraph_mcp/
├── <feature_name>/
│   ├── __init__.py       # Public API exports
│   ├── config.py         # Feature-specific configuration
│   ├── tools.py          # MCP tools for the feature
│   └── <implementation>  # Feature logic
```

### Integration Pattern

Features integrate with core systems:

1. **Configuration**: Feature config classes in `src/config/schemas.py`
2. **Tools Registration**: Tools registered in `src/server.py` based on feature flags
3. **Cache Integration**: Use `create_cache()` from core
4. **Observability**: Use `get_observability()` from core
5. **Error Handling**: Use standard error types from `src/errors.py`

### Example: Token Optimization Module

```
src/token_optimization/
├── __init__.py           # Exports: TokenCounter, TokenOptimizer, etc.
├── config.py             # TokenOptimizationConfig
├── tools.py              # MCP tools: optimize_tokens, count_tokens, etc.
├── counter.py            # Multi-provider token counting
├── optimizer.py          # Optimization strategies
└── cost_estimator.py     # LLM pricing and cost calculation
```

- Plugins live under `plugins/<plugin-name>/` or separate packages.
- Contract:
  - Expose MCP tools with `@mcp.tool()`.
  - Declare dependencies and minimum supported core version.
  - Provide setup/teardown lifecycle hooks.
  - Use typed Pydantic configuration.
  - Handle errors gracefully (never crash core).
  - Explicit, fail-safe loading.

Standard Plugin Structure:
```
plugins/my-plugin/
├── src/my_plugin/
│   ├── __init__.py          # Exports and metadata
│   ├── plugin.py            # Setup, teardown, metadata
│   ├── config.py            # Pydantic configuration
│   ├── tools.py             # MCP tool implementations
│   └── utils.py             # Helper functions
├── tests/                   # Plugin-specific tests
├── pyproject.toml           # Dependencies and metadata
└── README.md                # Plugin documentation
```

Note: Redis is not a plugin; it is a core optional backend selected via `CACHE_BACKEND`.

---

## Feature Specifications

### Feature 1: Token Optimization (Implemented)

**Purpose:** Automatic token reduction and cost estimation for LLM requests

**Location:** `src/token_optimization/`

**Dependencies:**
```toml
dependencies = [
    "tiktoken>=0.5.0",         # OpenAI token counting
    "anthropic>=0.25.0",       # Anthropic/Claude token counting
]
```

**MCP Tools:**
- `optimize_tokens(content, target_reduction, model, strategies)` → optimization result
- `count_tokens(content, model, include_breakdown)` → token count
- `estimate_cost(content, model, operation, output_tokens)` → cost estimate
- `analyze_token_efficiency(content, model)` → efficiency analysis

**Configuration:**
```python
class TokenOptimizationConfig(BaseModel):
    enabled: bool = True
    default_reduction_target: float = 0.20  # 20% reduction
    quality_threshold: float = 0.90
    cache_optimizations: bool = True
    optimization_strategies: List[str] = ["whitespace", "redundancy", "compression"]
    max_overhead_ms: float = 100.0
    enable_aggressive_mode: bool = False
    preserve_code_blocks: bool = True
    preserve_formatting: bool = True
    cache_ttl_seconds: int = 3600
```

**Capabilities:**
- Token counting for 15+ models across OpenAI, Anthropic, Google, Mistral
- 5 optimization strategies with quality preservation
- Real-time cost estimation with pricing database
- Sub-100ms processing overhead
- 20-50% token reduction in typical use cases
- >90% quality preservation by default

**Integration:**
- Uses core cache for optimization pattern storage
- Emits metrics via core observability
- Respects feature flag: `features.token_optimization`

This section defines the six major plugins that implement the AI optimization platform capabilities described in the project vision. Each plugin specification includes purpose, dependencies, MCP tools, configuration, and integration points.



### Feature 2: Model Routing (Future Implementation)

**Purpose:** Intelligent model selection across 15+ providers

**Location:** `src/model_routing/`

**Status:** Planned for future release

**Planned Capabilities:**
- Real-time cost-performance optimization
- Support for 15+ AI models
- Sub-25ms routing decisions
- 40-60% cost reduction potential

**Purpose:** Intelligent model selection and routing across 15+ AI providers based on cost, quality, and latency requirements.

**Package Name:** `seraph-mcp-model-routing`

**Location:** `plugins/model-routing/`

**Core Dependencies:**
```toml
[project]
dependencies = [
    "seraph-mcp>=1.0.0",       # Core platform
    "openai>=1.0.0",           # OpenAI models
    "anthropic>=0.25.0",       # Anthropic/Claude models
    "google-generativeai>=0.3.0",  # Google Gemini
    "cohere>=4.0.0",           # Cohere models
    "httpx>=0.25.0",           # HTTP client for API calls
]
```

**MCP Tools Exposed:**
- `find_best_model(task_type: str, requirements: dict) -> dict`
  - Find optimal model for task and requirements
  - Returns model recommendation with reasoning
- `route_request(prompt: str, requirements: dict) -> dict`
  - Automatically route to best model and execute
  - Returns response with model used and cost
- `compare_models(task_description: str, candidate_models: List[str]) -> dict`
  - Compare multiple models for specific task
  - Returns detailed comparison matrix
- `get_model_pricing() -> dict`
  - Current pricing for all supported models
- `get_model_capabilities(model: str) -> dict`
  - Capabilities, context window, features

**Configuration Schema:**
```python
class ModelRoutingConfig(BaseModel):
    enabled: bool = True
    default_providers: List[str] = ["openai", "anthropic", "google"]
    cost_weight: float = Field(0.4, ge=0.0, le=1.0)
    quality_weight: float = Field(0.4, ge=0.0, le=1.0)
    latency_weight: float = Field(0.2, ge=0.0, le=1.0)
    max_cost_per_request: Optional[float] = None
    cache_routing_decisions: bool = True
    fallback_models: List[str] = ["gpt-3.5-turbo"]
```

**Supported Models (15+):**
- OpenAI: GPT-4, GPT-4-Turbo, GPT-3.5-Turbo, GPT-3.5-Turbo-16k
- Anthropic: Claude-3-Opus, Claude-3-Sonnet, Claude-3-Haiku, Claude-2.1
- Google: Gemini-Pro, Gemini-Pro-Vision, PaLM-2
- Cohere: Command, Command-Light
- Mistral: Mistral-Large, Mistral-Medium, Mistral-Small

**Integration Points:**
- Uses core cache for pricing and routing decisions
- Emits detailed cost/performance metrics
- Integrates with budget plugin for cost enforcement

---

### Feature 3: Semantic Cache (Future Implementation)

**Purpose:** Vector-based semantic similarity caching

**Location:** `src/semantic_cache/`

**Status:** Planned for future release

**Planned Capabilities:**
- Redis Stack or ChromaDB integration
- Semantic similarity matching
- Multi-level caching strategy

**Purpose:** Advanced similarity-based caching using vector embeddings for semantic matching beyond exact key matches.

**Package Name:** `seraph-mcp-semantic-cache`

**Location:** `plugins/semantic-cache/`

**Core Dependencies:**
```toml
[project]
dependencies = [
    "seraph-mcp>=1.0.0",           # Core platform
    "chromadb>=0.4.0",             # Vector database
    "sentence-transformers>=2.2.0", # Local embeddings
    "numpy>=1.24.0",               # Vector operations
    "scipy>=1.10.0",               # Similarity calculations
]

[project.optional-dependencies]
remote-embeddings = [
    "openai>=1.0.0",               # OpenAI embeddings API
    "cohere>=4.0.0",               # Cohere embeddings API
]
```

**MCP Tools Exposed:**
- `semantic_search(query: str, limit: int = 10, threshold: float = 0.8) -> dict`
  - Find semantically similar cached content
  - Returns matches with similarity scores
- `semantic_cache_set(key: str, value: Any, embedding: Optional[List[float]] = None) -> bool`
  - Store with automatic or provided embedding
- `analyze_cache_similarity(content: str) -> dict`
  - Find similar cache entries before storing
- `get_embedding(text: str) -> List[float]`
  - Generate embedding for text
- `clear_semantic_cache(namespace: Optional[str] = None) -> bool`
  - Clear vector cache entries

**Configuration Schema:**
```python
class SemanticCacheConfig(BaseModel):
    enabled: bool = True
    embedding_model: str = "all-MiniLM-L6-v2"  # Local by default
    similarity_threshold: float = Field(0.80, ge=0.0, le=1.0)
    use_remote_embeddings: bool = False
    remote_embedding_provider: Optional[str] = None  # "openai" or "cohere"
    max_cache_entries: int = 10000
    collection_name: str = "seraph_semantic_cache"
    persist_directory: str = "./data/chromadb"
```

**Integration Points:**
- Works alongside core cache (complementary, not replacement)
- Uses core observability for hit/miss tracking
- Can be queried before expensive LLM calls

---

### Feature 4: Context Optimization (Future Implementation)

**Purpose:** AI-powered content reduction with quality preservation

**Location:** `src/context_optimization/`

**Status:** Planned for future release

**Purpose:** AI-powered content reduction that preserves meaning and quality while reducing token usage.

**Package Name:** `seraph-mcp-context-optimization`

**Location:** `plugins/context-optimization/`

**Core Dependencies:**
```toml
[project]
dependencies = [
    "seraph-mcp>=1.0.0",       # Core platform
    "anthropic>=0.25.0",       # Claude for intelligent summarization
    "tiktoken>=0.5.0",         # Token counting
    "scikit-learn>=1.3.0",     # Text analysis
]
```

**MCP Tools Exposed:**
- `optimize_context(content: str, optimization_goal: str = "balanced") -> dict`
  - Reduce context size while preserving meaning
  - Goals: "aggressive", "balanced", "conservative"
  - Returns optimized content with quality metrics
- `analyze_content_structure(content: str) -> dict`
  - Analyze content for optimization opportunities
  - Returns structure analysis and recommendations
- `preserve_key_elements(content: str, elements: List[str]) -> dict`
  - Optimize while ensuring specific elements preserved
- `compare_optimization_strategies(content: str) -> dict`
  - Compare different optimization approaches
  - Returns side-by-side comparison

**Configuration Schema:**
```python
class ContextOptimizationConfig(BaseModel):
    enabled: bool = True
    default_optimization_goal: str = "balanced"
    quality_threshold: float = Field(0.90, ge=0.0, le=1.0)
    max_reduction_percentage: float = Field(0.40, ge=0.0, le=0.7)
    preserve_code_blocks: bool = True
    preserve_structured_data: bool = True
    use_ai_summarization: bool = True
    summarization_model: str = "claude-3-haiku-20240307"
```

**Integration Points:**
- Uses model routing plugin for AI summarization
- Caches optimization patterns in core cache
- Emits quality metrics via observability

---

### Feature 5: Budget Management (Future Implementation)

**Purpose:** Cost tracking and enforcement with free tier detection

**Location:** `src/budget_management/`

**Status:** Planned for future release

**Purpose:** Comprehensive cost tracking, forecasting, and enforcement with free tier detection and intelligent alerts.

**Package Name:** `seraph-mcp-budget-management`

**Location:** `plugins/budget-management/`

**Core Dependencies:**
```toml
[project]
dependencies = [
    "seraph-mcp>=1.0.0",       # Core platform
    "sqlalchemy>=2.0.0",       # Database for cost tracking
    "alembic>=1.12.0",         # Database migrations
    "pandas>=2.1.0",           # Data analysis
    "matplotlib>=3.8.0",       # Cost visualization
]
```

**MCP Tools Exposed:**
- `check_budget(detailed: bool = False) -> dict`
  - Current budget status and spending
  - Returns usage, limits, alerts, forecasts
- `set_budget(daily_limit: float, monthly_limit: float, alert_thresholds: List[float]) -> bool`
  - Configure budget limits and alerts
- `get_usage_report(time_period_hours: int = 24, breakdown_by: str = "model") -> dict`
  - Detailed usage analytics and trends
  - Breakdowns: "model", "plugin", "user", "time"
- `forecast_spending(days_ahead: int = 7) -> dict`
  - Predict future spending with confidence intervals
  - Returns forecast with recommendations
- `detect_free_tier() -> dict`
  - Detect and respect API free tier limits
  - Returns free tier status per provider
- `track_cost(operation: str, model: str, tokens: int, cost: float) -> bool`
  - Record cost for operation (auto-called by routing plugin)
- `get_cost_optimization_suggestions() -> dict`
  - AI-generated cost saving recommendations

**Configuration Schema:**
```python
class BudgetManagementConfig(BaseModel):
    enabled: bool = True
    daily_budget_limit: Optional[float] = None
    monthly_budget_limit: Optional[float] = None
    alert_thresholds: List[float] = [50.0, 75.0, 90.0]  # Percentage
    enforce_hard_limits: bool = False  # Block requests when limit hit
    free_tier_detection: bool = True
    cost_tracking_database: str = "sqlite:///data/costs.db"
    forecast_confidence_level: float = 0.95
    alert_email: Optional[str] = None
```

**Integration Points:**
- Receives cost data from model routing plugin
- Uses core cache for pricing lookup
- Emits budget alerts via observability
- Can block requests when limits exceeded

---

### Feature 6: Quality Preservation (Future Implementation)

**Purpose:** Multi-dimensional validation with automatic rollback

**Location:** `src/quality_preservation/`

**Status:** Planned for future release

**Purpose:** Multi-dimensional validation of optimized content with automatic rollback when quality degrades below thresholds.

**Package Name:** `seraph-mcp-quality-preservation`

**Location:** `plugins/quality-preservation/`

**Core Dependencies:**
```toml
[project]
dependencies = [
    "seraph-mcp>=1.0.0",       # Core platform
    "sentence-transformers>=2.2.0",  # Semantic similarity
    "difflib",                 # Built-in text comparison
    "scikit-learn>=1.3.0",     # Text metrics
    "numpy>=1.24.0",           # Numerical operations
]
```

**MCP Tools Exposed:**
- `validate_quality(original: str, optimized: str) -> dict`
  - Multi-dimensional quality assessment
  - Returns quality score and detailed metrics
- `compare_content(original: str, modified: str, dimensions: List[str]) -> dict`
  - Compare content across multiple dimensions
  - Dimensions: "semantic", "structure", "completeness", "accuracy"
- `suggest_improvements(content: str, quality_issues: List[str]) -> dict`
  - Get actionable improvement recommendations
- `analyze_content_quality(content: str) -> dict`
  - Comprehensive quality analysis
  - Returns scores across all dimensions
- `set_quality_thresholds(thresholds: dict) -> bool`
  - Configure quality requirements per dimension

**Configuration Schema:**
```python
class QualityPreservationConfig(BaseModel):
    enabled: bool = True
    semantic_similarity_threshold: float = Field(0.90, ge=0.0, le=1.0)
    structural_similarity_threshold: float = Field(0.85, ge=0.0, le=1.0)
    completeness_threshold: float = Field(0.90, ge=0.0, le=1.0)
    overall_quality_threshold: float = Field(0.90, ge=0.0, le=1.0)
    auto_rollback: bool = True  # Rollback if quality too low
    preserve_code_blocks: bool = True
    preserve_urls: bool = True
    preserve_structured_data: bool = True
    validation_timeout_ms: int = 100
```

**Validation Dimensions:**
- **Semantic Similarity:** Meaning preservation (using embeddings)
- **Structural Similarity:** Format and organization preservation
- **Completeness:** All key information retained
- **Accuracy:** Factual correctness maintained
- **Overall Quality:** Weighted combination of all dimensions

**Integration Points:**
- Called by context optimization plugin after optimization
- Can trigger automatic rollback to original content
- Emits quality metrics via observability
- Works with semantic cache for similarity calculations

---

## Packaging & Deployment

```
┌─────────────────────────────────────────────────────┐
│                  MCP Client                         │
└──────────────────┬──────────────────────────────────┘
                   │ stdio (JSON-RPC)
┌──────────────────▼──────────────────────────────────┐
│            Core Server (src/server.py)              │
│  ┌────────────────────────────────────────────────┐ │
│  │  Core Tools: cache, status, metrics            │ │
│  └────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────┐ │
│  │  Cache Factory (Memory/Redis)                  │ │
│  └────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────┐ │
│  │  Observability Adapter                         │ │
│  └────────────────────────────────────────────────┘ │
└──────────────────┬──────────────────────────────────┘
                   │ Plugin Integration
┌──────────────────▼──────────────────────────────────┐
│                 AI Optimization Plugins              │
│  ┌──────────────┐  ┌─────────────┐  ┌─────────────┐│
│  │ Token Opt.   │  │ Model Route │  │ Semantic    ││
│  │              │◄─┤             │─►│ Cache       ││
│  └──────┬───────┘  └──────┬──────┘  └──────┬──────┘│
│         │                 │                 │       │
│  ┌──────▼───────┐  ┌─────▼──────┐  ┌──────▼──────┐│
│  │ Context Opt. │  │ Budget     │  │ Quality     ││
│  │              │◄─┤ Management │─►│ Preserv.    ││
│  └──────────────┘  └────────────┘  └─────────────┘│
└─────────────────────────────────────────────────────┘

Plugin Communication:
- All plugins can use core cache
- All plugins emit to core observability
- Plugins can call other plugins' tools via MCP
- Budget plugin receives cost data from routing plugin
- Quality plugin validates context optimization output
- Semantic cache works alongside core cache
```

---

## Installation & Configuration

### Installation

**Method 1: pip install**
```bash
# Install individual plugins
pip install seraph-mcp-model-routing
pip install seraph-mcp-semantic-cache

# Install all plugins
pip install seraph-mcp[all-plugins]
```

**Method 2: uv sync**
```toml
# pyproject.toml
```bash
# Standard installation (includes all features)
pip install seraph-mcp

# Or via npm for MCP usage
npx -y seraph-mcp
```

**Optional Dependencies:**
```toml
[project.optional-dependencies]
# Heavy dependencies for advanced features
embeddings = [
    "sentence-transformers>=2.0.0",  # Local embeddings
    "chromadb>=0.4.0",               # Vector database
]

# All optional features
all = [
    "sentence-transformers>=2.0.0",
    "chromadb>=0.4.0",
]
```

### Configuration File

```json
// fastmcp.json
{
  "name": "Seraph MCP with AI Optimization",
  "version": "1.0.0",
  "core": {
    "cache_backend": "redis",
    "redis_url": "redis://localhost:6379/0"
  },
  "plugins": {
    "token_optimization": {
      "enabled": true,
      "default_reduction_target": 0.20
    },
    "model_routing": {
      "enabled": true,
      "default_providers": ["openai", "anthropic", "google"],
      "max_cost_per_request": 0.10
    },
    "semantic_cache": {
      "enabled": true,
      "embedding_model": "all-MiniLM-L6-v2",
      "similarity_threshold": 0.85
    },
    "context_optimization": {
      "enabled": true,
      "default_optimization_goal": "balanced",
      "quality_threshold": 0.90
    },
    "budget_management": {
      "enabled": true,
      "daily_budget_limit": 10.0,
      "monthly_budget_limit": 200.0,
      "alert_thresholds": [50, 75, 90]
    },
    "quality_preservation": {
      "enabled": true,
      "overall_quality_threshold": 0.90,
      "auto_rollback": true
    }
  }
}
```

---

## Governance & Mandatory Rules

### Code Organization
1. All features in `src/<feature_name>/` (not separate packages)
2. Feature flags in `src/config/schemas.py`
3. Tools registered in `src/server.py`
4. Shared dependencies in main `pyproject.toml`

### Development Rules

1. Minimal core: Only features required by the canonical runtime belong in core.
2. Single adapter rule:
   - Cache factory: `src/cache/factory.py`
   - Observability adapter: `src/observability/*`
3. Dependencies:
   - Keep core lean; Redis is allowed as a core optional backend (toggle).
   - Heavy or niche dependencies should be delivered via plugins.
4. Tests required for all core changes.
5. No examples/dev-only files in `src/`.
6. Configuration must be typed and validated at startup.
7. Long-running tasks must be cancellable and time-bounded.
8. Secrets never in source; CI-enforced.
9. Observability required for every MCP tool (invocation, latency, errors).
10. Deprecations must be behind explicit feature flags for one release cycle maximum.
11. MCP stdio only in core; any HTTP servers must live in plugins.

Enforcement:
- CI gates described above.
- Branch protections and CODEOWNERS for `src/`.

---

## Migration & Recovery Plan

- Maintain a `recovery/` branch for archived or plugin-candidate code for one release cycle.
- Reintroduce features as plugins with integration tests.
- Rollback via redeploying previous tags.

---

## Release & Versioning

- Semantic versioning: `MAJOR.MINOR.PATCH` for core.
- Plugins declare `core_version_range` for compatibility.
- Release checklist:
  - CI green
  - Updated changelog
  - Release notes (including deprecations)

---

## File Layout (Monolithic Architecture)

- `src/`
  - `src/server.py` — FastMCP stdio server (ONLY entrypoint)
  - `src/config/`
    - `src/config/schemas.py` — Pydantic models
    - `src/config/loader.py` — Config loading
    - `src/config/__init__.py` — Exports
  - `src/cache/`
    - `src/cache/factory.py` — ONLY cache factory
    - `src/cache/interface.py` — Cache interface
    - `src/cache/backends/memory.py` — Memory backend
    - `src/cache/backends/redis.py` — Redis backend (core optional)
    - `src/cache/__init__.py` — Exports
  - `src/observability/`
    - `src/observability/monitoring.py` — Observability adapter
    - `src/observability/__init__.py` — Exports
  - `src/errors.py` — Error types
  - `src/__init__.py` — Core exports
- `plugins/` — Optional features as separate packages
- `tests/`
  - `tests/unit/`
  - `tests/integration/`
- `examples/`
- `docs/`
  - `docs/SDD.md` (this file)
  - `docs/PLUGIN_GUIDE.md` (plugin development guide)
- `fastmcp.json` — FastMCP configuration
- `pyproject.toml` — Python package configuration
- `.env.example` — Example environment configuration

---

## Operational Playbooks

- Incident response:
  1. Increase log verbosity (if safe).
  2. Inspect metrics via `get_metrics()` and `get_cache_stats()`.
  3. Validate cache connectivity (Redis ping in stats).
  4. Roll back to last green tag if persistent.
- Deploy:
  - Run CI → tag release → test via MCP client → rollout.
- Emergency rollback:
  - Re-deploy previous green tag.

---

## Appendix — Implementation Checklist

### Core Implementation (Complete ✅)
1. ✅ MCP stdio server and tools
2. ✅ Typed Pydantic config and loader
3. ✅ Cache factory + memory backend
4. ✅ Observability adapter with structured logs
5. ✅ Standardized error types
6. ✅ Redis backend implemented as core optional; toggle via `CACHE_BACKEND`
7. ✅ Minimal tests for Redis backend (unit + integration)
8. ✅ `.env.example` updated with Redis toggle and variables
9. ✅ Plugin developer guide in `docs/PLUGIN_GUIDE.md`
10. ✅ CI enhancements: coverage gates, secret scanning, dependency checks

### Feature Implementation

**Token Optimization (✅ Complete):**
- [x] Token counter with multi-provider support
- [x] Token optimizer with 5 strategies
- [x] Cost estimator with pricing database
- [x] MCP tools integration
- [x] Feature flag support
- [x] Configuration schema
- [x] Documentation

**Future Features (📋 Planned):**
1. ⬜ Token Optimization Plugin
   - Token counting and reduction
   - Cost estimation
   - Optimization pattern caching
2. ⬜ Model Routing Plugin
   - 15+ model integrations (OpenAI, Anthropic, Google, etc.)
   - Intelligent routing algorithm
   - Real-time pricing integration
   - Fallback and retry logic
3. ⬜ Semantic Cache Plugin
   - ChromaDB integration
   - Embedding generation (local + remote)
   - Similarity search with configurable thresholds
   - Performance monitoring
4. ⬜ Context Optimization Plugin
   - AI-powered summarization
   - Content structure analysis
   - Key element preservation
   - Multiple optimization strategies
5. ⬜ Budget Management Plugin
   - Cost tracking database
   - Usage analytics and reporting
   - Spending forecasts with confidence intervals
   - Free tier detection
   - Alert system
6. ⬜ Quality Preservation Plugin
   - Multi-dimensional validation
   - Semantic similarity calculation
   - Automatic rollback mechanism
   - Quality metrics dashboard

### Documentation Updates (Current)
1. ✅ SDD.md updated with plugin specifications
2. ⬜ Individual plugin README.md files
3. ⬜ Plugin integration examples
4. ⬜ End-to-end workflow documentation
5. ⬜ Performance benchmarks and optimization guides

---