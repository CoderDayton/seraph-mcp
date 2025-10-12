# Seraph MCP Architecture

**Version:** 3.0  
**Date:** 2025-01-13  
**Status:** Active

---

## Overview

Seraph MCP is a **comprehensive AI optimization platform** delivered as a single, integrated package. The platform provides token optimization, model routing, semantic caching, context optimization, budget management, and quality preservation capabilities. All features are included in the main package but can be selectively enabled/disabled via feature flags.

This document explains the architectural decisions, component organization, and design principles.

---

## Architectural Decision: Monolithic with Internal Modularity

### Why Monolithic Over Plugins

After extensive analysis from user and maintainer perspectives, we chose a **monolithic architecture with internal modularity** over a plugin system.

#### User Benefits

✅ **Single Installation**
- One command: `npx -y seraph-mcp` or `pip install seraph-mcp`
- No plugin management, dependency resolution, or version compatibility matrix
- Everything works out-of-the-box

✅ **Simplified Experience**
- One version number to track
- Unified documentation in one place
- No "which plugin do I need?" confusion
- Simpler troubleshooting (no cross-package issues)

✅ **Matches Marketing Promise**
- Platform is marketed as "Comprehensive AI Optimization Platform"
- Users expect all features when they install "Seraph MCP"
- No surprises or hidden costs

#### Maintainer Benefits

✅ **Single Release Cycle**
- One version number for everything
- One changelog
- No cross-package compatibility testing
- Release everything in sync

✅ **Simpler CI/CD**
- One build pipeline
- One test suite
- One deployment process
- No plugin version matrix

✅ **Integrated Testing**
- Test features together as users will use them
- No cross-package integration nightmares
- Clear test coverage metrics

✅ **Unified Development**
- One codebase, one issue tracker
- Clear ownership and accountability
- Simpler onboarding for contributors
- 90% less maintenance overhead

#### When Plugins Would Make Sense

Plugins are appropriate when:
1. Third-party developers need to extend the platform
2. There are 50+ features and users want à la carte selection
3. Features have conflicting dependencies
4. Team grows to 10+ people on separate codebases
5. Core is minimal and extensions add specialized domains

**Current Reality:**
- Single team maintaining all features
- 6 core features (all part of value proposition)
- Harmonious dependencies
- Features work better together than separately
- Users want the complete platform

#### Comparison with Industry

**Successful Monolithic Projects:**
- **FastAPI** - Includes validation, docs, testing, security
- **Django** - "Batteries included" is the selling point
- **Next.js** - Routing, rendering, optimization all built-in
- **Pandas** - Massive library, no plugins

**Plugin-Based Projects (Different Use Cases):**
- **Babel** - Compiler needing language-specific transforms
- **ESLint** - Linter for different languages/frameworks
- **VS Code** - IDE with thousands of third-party extensions

Seraph MCP is more like FastAPI/Django (comprehensive platform) than Babel/ESLint (extensible tool).

---

## Architecture Principles

1. **Comprehensive by Default**
   - All AI optimization features included
   - Works out-of-the-box without configuration
   - Sensible defaults for all settings

2. **Internal Modularity**
   - Clear separation of concerns
   - Self-contained feature modules
   - Easy to work on individual features

3. **Feature Flags**
   - Enable/disable capabilities via configuration
   - No need to uninstall features
   - Fine-grained control over platform behavior

4. **Typed and Validated**
   - Pydantic models for all configuration
   - Runtime validation of inputs/outputs
   - Type safety throughout

5. **Observable and Traceable**
   - Comprehensive metrics and logging
   - Distributed tracing support
   - Performance monitoring built-in

6. **Safe by Default**
   - Conservative defaults
   - Automatic rollback on quality failures
   - Budget enforcement and alerts

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Seraph MCP Platform                   │
│                  (Single Package v1.0.0)                 │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌─────────────────────────────────────────────────┐   │
│  │              Core Runtime                        │   │
│  │  - FastMCP Server (MCP stdio protocol)          │   │
│  │  - Configuration Management (Pydantic)          │   │
│  │  - Cache System (Memory + Redis)                │   │
│  │  - Observability (Metrics, Traces, Logs)        │   │
│  └─────────────────────────────────────────────────┘   │
│                          │                               │
│  ┌───────────────────────┴──────────────────────────┐  │
│  │                                                    │  │
│  ▼                                                    ▼  │
│  ┌──────────────────────┐    ┌──────────────────────┐  │
│  │ Token Optimization   │    │  Model Routing       │  │
│  │ (Implemented)        │    │  (Future)            │  │
│  │                      │    │                      │  │
│  │ - Token Counter      │    │ - Cost Analysis      │  │
│  │ - Optimizer          │    │ - Provider Selection │  │
│  │ - Cost Estimator     │    │ - Load Balancing     │  │
│  │                      │    │                      │  │
│  │ Tools:               │    │ Tools:               │  │
│  │ • optimize_tokens    │    │ • find_best_model    │  │
│  │ • count_tokens       │    │ • route_request      │  │
│  │ • estimate_cost      │    │ • compare_models     │  │
│  │ • analyze_efficiency │    │                      │  │
│  └──────────────────────┘    └──────────────────────┘  │
│                                                           │
│  ┌──────────────────────┐    ┌──────────────────────┐  │
│  │ Semantic Cache       │    │ Context Optimization │  │
│  │ (Future)             │    │ (Future)             │  │
│  │                      │    │                      │  │
│  │ - Vector Store       │    │ - Content Reducer    │  │
│  │ - Embeddings         │    │ - Quality Validator  │  │
│  │ - Similarity Search  │    │ - Strategy Selector  │  │
│  └──────────────────────┘    └──────────────────────┘  │
│                                                           │
│  ┌──────────────────────┐    ┌──────────────────────┐  │
│  │ Budget Management    │    │ Quality Preservation │  │
│  │ (Future)             │    │ (Future)             │  │
│  │                      │    │                      │  │
│  │ - Cost Tracking      │    │ - Validation Engine  │  │
│  │ - Forecasting        │    │ - Rollback Manager   │  │
│  │ - Enforcement        │    │ - Metrics Analyzer   │  │
│  └──────────────────────┘    └──────────────────────┘  │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

---

## Component Organization

### Directory Structure

```
seraph-mcp/
├── src/
│   ├── server.py              # Main MCP server with all tools
│   ├── errors.py              # Standardized error types
│   │
│   ├── config/                # Configuration management
│   │   ├── __init__.py
│   │   ├── schemas.py         # Pydantic models
│   │   └── loader.py          # Config loading logic
│   │
│   ├── cache/                 # Cache system
│   │   ├── __init__.py
│   │   ├── interface.py       # Cache interface
│   │   ├── factory.py         # Cache factory
│   │   ├── memory.py          # Memory backend
│   │   └── redis_backend.py   # Redis backend
│   │
│   ├── observability/         # Observability system
│   │   ├── __init__.py
│   │   ├── adapter.py         # Observability adapter
│   │   └── backends/          # Different backends
│   │
│   ├── token_optimization/    # Token optimization feature
│   │   ├── __init__.py        # Public API
│   │   ├── config.py          # Feature config
│   │   ├── tools.py           # MCP tools
│   │   ├── counter.py         # Token counter
│   │   ├── optimizer.py       # Optimizer engine
│   │   └── cost_estimator.py  # Cost calculator
│   │
│   ├── model_routing/         # Future: Model routing
│   ├── semantic_cache/        # Future: Semantic cache
│   ├── context_optimization/  # Future: Context optimization
│   ├── budget_management/     # Future: Budget management
│   └── quality_preservation/  # Future: Quality preservation
│
├── tests/
│   ├── unit/                  # Unit tests per module
│   ├── integration/           # Integration tests
│   └── performance/           # Performance benchmarks
│
├── docs/
│   ├── ARCHITECTURE.md        # This file
│   ├── SDD.md                 # System design document
│   └── PLUGIN_GUIDE.md        # Feature development guide
│
├── examples/                  # Usage examples
├── pyproject.toml            # Package definition
└── README.md                 # User documentation
```

### Feature Module Structure

Each feature follows a consistent internal structure:

```
src/<feature_name>/
├── __init__.py       # Public API exports
├── config.py         # Feature-specific configuration (Pydantic)
├── tools.py          # MCP tools for the feature
└── <implementation>  # Feature-specific logic
```

**Example: Token Optimization**

```
src/token_optimization/
├── __init__.py           # Exports: TokenCounter, TokenOptimizer, etc.
├── config.py             # TokenOptimizationConfig
├── tools.py              # TokenOptimizationTools class
├── counter.py            # Multi-provider token counting
├── optimizer.py          # 5 optimization strategies
└── cost_estimator.py     # LLM pricing and cost calculation
```

---

## Feature Integration Pattern

### 1. Configuration

Feature configurations are defined in `src/config/schemas.py`:

```python
class FeatureFlags(BaseModel):
    """Feature flags for enabling/disabling capabilities."""
    token_optimization: bool = True
    model_routing: bool = False
    semantic_cache: bool = False
    # ...

class TokenOptimizationConfig(BaseModel):
    """Token optimization feature configuration."""
    enabled: bool = True
    default_reduction_target: float = 0.20
    quality_threshold: float = 0.90
    # ...
```

### 2. Tools Registration

Tools are registered in `src/server.py` based on feature flags:

```python
async def initialize_server():
    config = load_config()
    
    # Initialize token optimization if enabled
    if config.features.token_optimization:
        token_config = TokenOptimizationConfig(**config.token_optimization.model_dump())
        _token_optimization_tools = get_token_optimization_tools(config=token_config)
        logger.info("Token optimization tools initialized")
```

### 3. MCP Tool Wrappers

```python
@mcp.tool()
async def optimize_tokens(content: str, ...) -> dict[str, Any]:
    """Optimize content to reduce token count."""
    if _token_optimization_tools is None:
        return {"error": "Token optimization is not enabled"}
    
    return _token_optimization_tools.optimize_tokens(...)
```

### 4. Core System Integration

Features integrate with core systems:

```python
class TokenOptimizationTools:
    def __init__(self, config):
        # Use core cache
        self.cache = create_cache()
        
        # Use core observability
        self.obs = get_observability()
    
    def optimize_tokens(self, content, ...):
        # Emit metrics
        self.obs.increment("tools.optimize_tokens")
        
        # Cache results
        if self.config.cache_optimizations:
            await self.cache.set(cache_key, result)
```

---

## Configuration System

### Hierarchical Configuration

Configuration is loaded in order of precedence:

1. **Code Defaults** (in Pydantic models)
2. **Environment Variables** (via `pydantic-settings`)
3. **Configuration File** (`seraph-mcp-config.json`)
4. **Runtime Updates** (via MCP tools)

### Feature Flags

```json
{
  "features": {
    "token_optimization": true,
    "model_routing": false,
    "semantic_cache": false,
    "context_optimization": false,
    "budget_management": false,
    "quality_preservation": false
  }
}
```

### Feature-Specific Configuration

```json
{
  "token_optimization": {
    "enabled": true,
    "default_reduction_target": 0.20,
    "quality_threshold": 0.90,
    "optimization_strategies": ["whitespace", "redundancy", "compression"],
    "max_overhead_ms": 100.0,
    "cache_optimizations": true
  }
}
```

---

## Data Flow

### Request Processing Flow

```
1. MCP Client (Claude Desktop, etc.)
   │
   ├─── stdin/stdout ───┐
   │                     ▼
2. FastMCP Server (src/server.py)
   │
   ├─── Load Config ───┐
   │                   ▼
3. Configuration System
   │  - Feature flags
   │  - Feature configs
   │
   ├─── Check Feature Flag ───┐
   │                           ▼
4. Feature Tools (if enabled)
   │  - Token Optimization
   │  - Model Routing (future)
   │  - etc.
   │
   ├─── Cache Check ───┐
   │                   ▼
5. Cache System
   │  - Memory or Redis
   │  - Hit/Miss
   │
   ├─── Process Request ───┐
   │                        ▼
6. Feature Logic
   │  - Token counting
   │  - Optimization
   │  - Cost estimation
   │
   ├─── Emit Metrics ───┐
   │                    ▼
7. Observability System
   │  - Metrics
   │  - Traces
   │  - Logs
   │
   └─── Return Response ───┐
                            ▼
8. MCP Client (receives result)
```

---

## Dependency Management

### Core Dependencies

All features share a single set of dependencies:

```toml
[project]
dependencies = [
    # Core MCP server
    "fastmcp>=2.0.0",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    "python-dotenv>=1.0.0",
    
    # Cache
    "redis>=5.0.0",
    
    # Token optimization
    "tiktoken>=0.5.0",
    "anthropic>=0.25.0",
]
```

### Optional Dependencies

Heavy dependencies for advanced features:

```toml
[project.optional-dependencies]
embeddings = [
    "sentence-transformers>=2.0.0",  # For semantic cache
    "chromadb>=0.4.0",               # Vector database
]

all = [
    "sentence-transformers>=2.0.0",
    "chromadb>=0.4.0",
]
```

### Installation

```bash
# Standard installation (all core features)
pip install seraph-mcp

# With optional heavy dependencies
pip install seraph-mcp[all]

# Development installation
pip install seraph-mcp[dev]
```

---

## Testing Strategy

### Test Organization

```
tests/
├── unit/                   # Unit tests for individual modules
│   ├── test_cache.py
│   ├── test_config.py
│   ├── test_token_counter.py
│   ├── test_optimizer.py
│   └── test_cost_estimator.py
│
├── integration/            # Integration tests across features
│   ├── test_server.py
│   ├── test_token_optimization_integration.py
│   └── test_feature_flags.py
│
└── performance/            # Performance benchmarks
    ├── test_optimization_performance.py
    └── test_cache_performance.py
```

### Coverage Requirements

- **Overall:** ≥85% code coverage
- **Core modules:** ≥90% coverage
- **Feature modules:** ≥85% coverage
- **Critical paths:** 100% coverage

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=src/seraph_mcp --cov-report=html

# Specific test file
pytest tests/unit/test_counter.py -v

# Performance tests
pytest tests/performance/ -v
```

---

## Development Workflow

### Adding a New Feature

1. **Create Feature Module**
   ```bash
   mkdir -p src/seraph_mcp/<feature_name>
   touch src/seraph_mcp/<feature_name>/{__init__.py,config.py,tools.py}
   ```

2. **Define Configuration**
   - Add config class to `src/config/schemas.py`
   - Add feature flag to `FeatureFlags`

3. **Implement Feature Logic**
   - Create implementation files in feature directory
   - Follow existing patterns (counter.py, optimizer.py, etc.)

4. **Create MCP Tools**
   - Define tools in `tools.py`
   - Integrate with core cache and observability

5. **Register in Server**
   - Update `src/server.py` to initialize feature
   - Register MCP tool wrappers

6. **Write Tests**
   - Unit tests for each component
   - Integration tests for MCP tools
   - Performance benchmarks

7. **Document**
   - Update ARCHITECTURE.md
   - Update README.md
   - Add examples

### Code Quality

```bash
# Format code
ruff format .

# Lint code
ruff check --fix .

# Type check
mypy src/

# Security check
bandit -r src/
```

---

## Performance Considerations

### Design Goals

- **Sub-100ms Processing:** Most operations complete in <100ms
- **Minimal Overhead:** Core runtime overhead <10ms
- **Efficient Caching:** Cache access <5ms (memory), <10ms (Redis)
- **Scalable:** Handle 1000+ requests/second

### Optimization Strategies

1. **Lazy Loading**
   - Features initialized only when enabled
   - Heavy dependencies loaded on-demand

2. **Caching**
   - Optimization patterns cached for reuse
   - Token counts cached by content hash
   - Cost estimates cached by model

3. **Async Operations**
   - All I/O operations are async
   - Concurrent processing where possible

4. **Memory Management**
   - LRU cache for memory backend
   - Configurable cache sizes
   - Automatic cleanup

---

## Security Considerations

### Input Validation

- All inputs validated with Pydantic
- Type safety enforced at runtime
- Bounds checking on numeric values

### Dependency Management

- Regular security audits with `safety`
- Pin dependencies for reproducibility
- Monitor for CVEs

### Rate Limiting (Future)

- Per-client rate limits
- Budget enforcement
- Automatic throttling

---

## Deployment

### Environment Setup

```bash
# Install dependencies
pip install seraph-mcp

# Set environment variables
export REDIS_URL=redis://localhost:6379
export LOG_LEVEL=INFO
export ENVIRONMENT=production
```

### Running the Server

```bash
# Via FastMCP CLI
fastmcp run

# Via Python
python -m seraph_mcp.server

# With custom config
fastmcp run --config prod.fastmcp.json
```

### Docker Deployment

```bash
# Build image
docker build -t seraph-mcp .

# Run container
docker run -d \
  -e REDIS_URL=redis://redis:6379 \
  -e LOG_LEVEL=INFO \
  seraph-mcp
```

---

## Monitoring and Observability

### Metrics

The platform emits metrics for:
- Tool invocations
- Cache hit/miss rates
- Optimization performance
- Token counts and savings
- Cost estimates
- Processing times

### Logging

Structured logging includes:
- Request traces
- Error details
- Performance data
- Configuration changes

### Health Checks

```bash
# Via MCP tool
mcp call check_status '{"include_details": true}'

# Response includes:
# - System health
# - Cache status
# - Feature availability
# - Performance metrics
```

---

## Future Roadmap

### Q1 2025
- ✅ Token Optimization (Complete)
- ⬜ Model Routing
- ⬜ Basic Budget Management

### Q2 2025
- ⬜ Semantic Cache
- ⬜ Context Optimization
- ⬜ Advanced Budget Features

### Q3 2025
- ⬜ Quality Preservation
- ⬜ Performance Optimizations
- ⬜ Enhanced Monitoring

### Q4 2025
- ⬜ Advanced Analytics
- ⬜ Multi-tenancy
- ⬜ Enterprise Features

---

## Contributing

### Guidelines

1. Follow the existing code structure
2. Add comprehensive tests (≥85% coverage)
3. Document public APIs
4. Update ARCHITECTURE.md for significant changes
5. Submit PRs to the main repository

### Code Standards

- Use `ruff` for formatting and linting
- Type all functions with Python type hints
- Write docstrings for public APIs
- Follow existing naming conventions

---

## Conclusion

Seraph MCP's monolithic architecture with internal modularity provides the best balance of:
- **User Experience:** Simple installation and usage
- **Developer Experience:** Clear structure and tooling
- **Maintainability:** Single codebase and release cycle
- **Extensibility:** Easy to add new features
- **Performance:** Optimized integration between components

This architecture supports the project's goal of being a comprehensive AI optimization platform while remaining simple to install, use, and maintain.

---

**For detailed API documentation, see [SDD.md](./SDD.md)**  
**For feature development, see [PLUGIN_GUIDE.md](./PLUGIN_GUIDE.md)**