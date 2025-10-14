# Seraph MCP — AI Model Providers Guide

Version: 1.0
Last Updated: 2025-01-13

---

## Overview

Seraph MCP includes a sophisticated AI model provider system that enables seamless integration with multiple AI model providers. The system dynamically loads model information, pricing, and capabilities from the **Models.dev API**, ensuring always up-to-date information without hardcoded model data.

### Key Features

- **Dynamic Model Discovery**: Automatic loading of 750+ models from 50+ providers via Models.dev API
- **Real-time Pricing**: Always current pricing information (per-million token costs)
- **Unified Interface**: Consistent API across all providers (OpenAI, Anthropic, Google, etc.)
- **Automatic Fallback**: Cached data ensures reliability if Models.dev API is unavailable
- **Zero Configuration**: Works out-of-the-box with sensible defaults
- **Custom Endpoints**: Support for self-hosted models (LM Studio, Ollama, vLLM, etc.)

---

## Supported Providers

### Official Providers

1. **OpenAI** (`openai`)
   - GPT-4, GPT-4 Turbo, GPT-3.5-Turbo, O1, and more
   - Function calling, vision capabilities
   - Streaming support

2. **Anthropic** (`anthropic`)
   - Claude 3 (Opus, Sonnet, Haiku)
   - Claude 4 models
   - Extended context windows (200K+ tokens)

3. **Google Gemini** (`gemini`)
   - Gemini Pro, Gemini 1.5 Pro/Flash
   - Vision and multimodal capabilities
   - Ultra-long context (1M+ tokens)

4. **OpenAI-Compatible** (`openai-compatible`)
   - Any OpenAI-compatible endpoint
   - Local models (LM Studio, Ollama, vLLM)
   - Custom deployments
   - Automatic Models.dev discovery

### Models.dev Integration

All providers use the **Models.dev API** (https://models.dev/api.json) for:

- Model metadata (context windows, capabilities)
- Real-time pricing (per-million token costs)
- Model availability and status
- Feature support (function calling, vision, reasoning)

The API response is cached for **1 hour** to minimize network requests while keeping data fresh.

---

## Configuration

### Environment Variables

Configure providers via `.env` file:

```bash
# OpenAI
OPENAI_ENABLED=true
OPENAI_API_KEY=sk-...
OPENAI_TIMEOUT=30.0
OPENAI_MAX_RETRIES=3

# Anthropic
ANTHROPIC_ENABLED=true
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_TIMEOUT=30.0
ANTHROPIC_MAX_RETRIES=3

# Google Gemini
GEMINI_ENABLED=true
GEMINI_API_KEY=AIza...
GEMINI_TIMEOUT=30.0
GEMINI_MAX_RETRIES=3

# OpenAI-Compatible Providers (Together AI, Fireworks, Anyscale, DeepInfra, local LLMs, etc)
OPENAI_COMPATIBLE_ENABLED=true
OPENAI_COMPATIBLE_BASE_URL=https://api.together.xyz/v1  # Or any OpenAI-compatible endpoint
OPENAI_COMPATIBLE_API_KEY=your-api-key-here  # Required for most services, optional for local
OPENAI_COMPATIBLE_TIMEOUT=60.0
OPENAI_COMPATIBLE_MAX_RETRIES=3
```

### Configuration Schema

All providers use the `ProviderConfig` schema:

```python
class ProviderConfig(BaseModel):
    enabled: bool = True          # Enable/disable provider
    api_key: str                   # API key (required)
    base_url: Optional[str] = None # Custom base URL (optional)
    timeout: float = 30.0          # Request timeout (seconds)
    max_retries: int = 3           # Maximum retry attempts
```

---

## Usage Examples

### Basic Usage

```python
from src.providers import create_provider, ProviderConfig, CompletionRequest

# Create provider
config = ProviderConfig(
    api_key="sk-your-api-key-here",
    timeout=30.0,
    max_retries=3,
)
provider = create_provider("openai", config)

# Make completion request
request = CompletionRequest(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "Explain quantum computing in simple terms."}
    ],
    temperature=0.7,
    max_tokens=500,
)

response = await provider.complete(request)
print(f"Response: {response.content}")
print(f"Cost: ${response.cost_usd:.6f}")
print(f"Latency: {response.latency_ms:.2f}ms")
```

### Model Discovery

```python
from src.providers import create_provider, ProviderConfig

config = ProviderConfig(api_key="sk-...")
provider = create_provider("openai", config)

# List all available models (from Models.dev)
models = await provider.list_models()

print(f"Available models: {len(models)}")
for model in models:
    print(f"- {model.display_name} ({model.model_id})")
    print(f"  Context: {model.context_window:,} tokens")
    print(f"  Cost: ${model.input_cost_per_1k:.4f} in / ${model.output_cost_per_1k:.4f} out per 1K")
    print(f"  Capabilities: {', '.join(model.capabilities)}")
```

### Cost Estimation

```python
from src.providers import create_provider, ProviderConfig

config = ProviderConfig(api_key="sk-...")
provider = create_provider("openai", config)

# Estimate cost before making request
input_tokens = 1000
output_tokens = 500

cost = await provider.estimate_cost("gpt-4", input_tokens, output_tokens)
print(f"Estimated cost: ${cost:.6f}")

# Compare costs across models
models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
for model in models:
    cost = await provider.estimate_cost(model, input_tokens, output_tokens)
    print(f"{model:20s} → ${cost:.6f}")
```

### Multiple Providers

```python
from src.providers import create_provider, ProviderConfig, list_providers

# Create multiple providers
providers = {}

providers["openai"] = create_provider(
    "openai",
    ProviderConfig(api_key="sk-...")
)

providers["anthropic"] = create_provider(
    "anthropic",
    ProviderConfig(api_key="sk-ant-...")
)

providers["gemini"] = create_provider(
    "gemini",
    ProviderConfig(api_key="AIza...")
)

# List all active providers
active = list_providers()
print(f"Active providers: {list(active.keys())}")

# Use different providers
request = CompletionRequest(
    model="gpt-4",  # or "claude-3-opus-20240229", "gemini-pro"
    messages=[{"role": "user", "content": "Hello!"}],
)

# Route to different providers
for name, provider in providers.items():
    response = await provider.complete(request)
    print(f"{name}: {response.content} (${response.cost_usd:.6f})")
```

### Custom Endpoints (LM Studio, Ollama, etc.)

```python
from src.providers import create_provider, ProviderConfig

# LM Studio
config = ProviderConfig(
    api_key="not-needed",  # Local services often don't need keys
    base_url="http://localhost:1234/v1",
    timeout=60.0,  # Local inference may take longer
)
provider = create_provider("openai-compatible", config)

# List local models
models = await provider.list_models()
print(f"Local models: {[m.model_id for m in models]}")

# Use local model
request = CompletionRequest(
    model=models[0].model_id,
    messages=[{"role": "user", "content": "Hello!"}],
)
response = await provider.complete(request)
print(f"Response: {response.content}")
print(f"Cost: ${response.cost_usd:.6f}")  # Usually $0.00 for local
```

---

## Models.dev Client

### Direct Access

```python
from src.providers import get_models_dev_client

client = get_models_dev_client()

# Load all providers and models
providers = await client.load_providers()
print(f"Total providers: {len(providers)}")
print(f"Total models: {sum(len(p.models) for p in providers.values())}")

# Get specific provider
provider = await client.get_provider("openai")
if provider:
    print(f"Provider: {provider.name}")
    print(f"API: {provider.api}")
    print(f"Models: {len(provider.models)}")

# Search for a model
result = await client.search_model("gpt-4")
if result:
    provider_id, model_info = result
    print(f"Found {model_info.name} from {provider_id}")
    print(f"Cost: ${model_info.cost.input}/M input, ${model_info.cost.output}/M output")

# List all models
all_models = await client.list_all_models()
print(f"Total models across all providers: {len(all_models)}")

# Get models by provider type
openai_models = await client.get_models_by_provider_type("openai")
print(f"OpenAI-compatible models: {len(openai_models)}")
```

### Model Information Schema

```python
class ModelInfo(BaseModel):
    id: str                          # Model identifier
    name: str                        # Human-readable name
    attachment: bool                 # Supports file attachments
    reasoning: bool                  # Has reasoning capabilities
    temperature: bool                # Supports temperature parameter
    tool_call: bool                  # Supports function/tool calling
    knowledge: Optional[str]         # Knowledge cutoff date
    release_date: Optional[str]      # Release date
    modalities: ModelModalities      # Supported input/output types
    open_weights: bool               # Open source weights
    cost: Optional[ModelCost]        # Pricing (per million tokens)
    limit: ModelLimit                # Context and output limits

class ModelCost(BaseModel):
    input: float                     # Input cost per million tokens (USD)
    output: float                    # Output cost per million tokens (USD)
    cache_read: Optional[float]      # Cache read cost (if supported)
    cache_write: Optional[float]     # Cache write cost (if supported)

class ModelLimit(BaseModel):
    context: int                     # Maximum context window (tokens)
    output: int                      # Maximum output tokens
```

---

## Architecture

### Provider Hierarchy

```
BaseProvider (abstract)
├── Dynamic model loading from Models.dev
├── Cost estimation
├── Health checks
└── Standardized interface

├── OpenAIProvider
│   └── Models.dev provider_id: "openai"
├── AnthropicProvider
│   └── Models.dev provider_id: "anthropic"
├── GeminiProvider
│   └── Models.dev provider_id: "google"
└── OpenAICompatibleProvider
    └── Automatic Models.dev discovery by base_url
```

### Request/Response Flow

```
1. User creates CompletionRequest
   ↓
2. Provider validates model (checks Models.dev)
   ↓
3. Provider makes API call to model endpoint
   ↓
4. Response converted to CompletionResponse
   ↓
5. Cost calculated using Models.dev pricing
   ↓
6. Return standardized response to user
```

### Caching Strategy

- **Models.dev API**: Cached for 1 hour
- **Model Lists**: Cached per provider instance
- **Model Info**: Fetched on-demand, cached for session
- **Pricing**: Calculated using cached Models.dev data

---

## Advanced Topics

### Health Checks

```python
from src.providers import create_provider, ProviderConfig

providers_to_check = [
    ("openai", ProviderConfig(api_key="sk-...")),
    ("anthropic", ProviderConfig(api_key="sk-ant-...")),
    ("gemini", ProviderConfig(api_key="AIza...")),
]

for name, config in providers_to_check:
    provider = create_provider(name, config)
    is_healthy = await provider.health_check()
    print(f"{name:15s} → {'✓ Healthy' if is_healthy else '✗ Unhealthy'}")
```

### Custom Retry Logic

```python
from src.providers import create_provider, ProviderConfig

config = ProviderConfig(
    api_key="sk-...",
    max_retries=5,  # Increase retries
    timeout=60.0,   # Longer timeout
)
provider = create_provider("openai", config)
```

### Error Handling

```python
from src.providers import create_provider, CompletionRequest

try:
    response = await provider.complete(request)
except RuntimeError as e:
    if "authentication failed" in str(e).lower():
        print("Invalid API key")
    elif "rate limit" in str(e).lower():
        print("Rate limit exceeded - wait before retrying")
    elif "api error" in str(e).lower():
        print("Provider API error")
    else:
        print(f"Unexpected error: {e}")
```

### Resource Cleanup

```python
from src.providers import close_all_providers

# At application shutdown
await close_all_providers()
```

---

## Best Practices

### 1. API Key Security

```bash
# Never hardcode API keys
# ❌ BAD
config = ProviderConfig(api_key="sk-1234567890abcdef")

# ✅ GOOD - Use environment variables
import os
config = ProviderConfig(api_key=os.getenv("OPENAI_API_KEY"))
```

### 2. Cost Management

```python
# Always estimate costs first
cost = await provider.estimate_cost(model_id, input_tokens, output_tokens)
if cost > threshold:
    print(f"Warning: Estimated cost ${cost:.4f} exceeds threshold")

# Monitor actual costs
response = await provider.complete(request)
print(f"Actual cost: ${response.cost_usd:.6f}")
```

### 3. Model Selection

```python
# Choose models based on requirements
models = await provider.list_models()

# Filter by context window
long_context = [m for m in models if m.context_window >= 100000]

# Filter by capabilities
vision_models = [m for m in models if "vision" in m.capabilities]

# Sort by cost
by_cost = sorted(models, key=lambda m: m.input_cost_per_1k)
cheapest = by_cost[0]
```

### 4. Error Resilience

```python
# Implement exponential backoff for retries
import asyncio

async def complete_with_backoff(provider, request, max_attempts=3):
    for attempt in range(max_attempts):
        try:
            return await provider.complete(request)
        except RuntimeError as e:
            if attempt < max_attempts - 1:
                wait_time = 2 ** attempt
                print(f"Attempt {attempt+1} failed, retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                raise
```

### 5. Provider Factory Pattern

```python
from src.providers import get_factory

# Use factory for provider management
factory = get_factory()

# Create providers on-demand
provider = factory.create_provider("openai", config)

# List all active providers
active = factory.list_providers()

# Clean up when done
await factory.close_all()
```

---

## Troubleshooting

### Models.dev API Issues

```python
from src.providers import get_models_dev_client

client = get_models_dev_client()

# Force refresh cache
providers = await client.load_providers(force_refresh=True)

# Check if specific provider is available
provider = await client.get_provider("openai")
if not provider:
    print("Provider not found in Models.dev")
```

### Provider Connection Issues

```bash
# Check if provider is reachable
curl https://api.openai.com/v1/models

# Check custom endpoint
curl http://localhost:1234/v1/models
```

### Model Not Found

```python
# List all available models
models = await provider.list_models()
available = [m.model_id for m in models]
print(f"Available: {available}")

# Search across all providers
client = get_models_dev_client()
result = await client.search_model("gpt-4")
if result:
    provider_id, model_info = result
    print(f"Found in provider: {provider_id}")
```

---

## Performance Optimization

### Caching

- Models.dev API responses are cached for 1 hour
- Provider instances are reused via factory pattern
- Model lists are cached per provider instance

### Async Operations

All provider operations are async for optimal performance:

```python
import asyncio

# Parallel requests to multiple providers
tasks = [
    provider1.complete(request),
    provider2.complete(request),
    provider3.complete(request),
]
responses = await asyncio.gather(*tasks)
```

### Connection Pooling

Providers use connection pooling for HTTP requests:

```python
config = ProviderConfig(
    api_key="sk-...",
    timeout=30.0,        # Per-request timeout
    max_retries=3,       # Automatic retries
)
```

---

## File Reference

### Core Files

- `src/providers/base.py` - BaseProvider interface and Models.dev integration
- `src/providers/models_dev.py` - Models.dev API client
- `src/providers/factory.py` - Provider factory and management
- `src/providers/__init__.py` - Public API exports

### Provider Implementations

- `src/providers/openai_provider.py` - OpenAI GPT models
- `src/providers/anthropic_provider.py` - Anthropic Claude models
- `src/providers/gemini_provider.py` - Google Gemini models
- `src/providers/openai_compatible.py` - Any OpenAI-compatible provider (Together AI, Fireworks, Anyscale, DeepInfra, local LLMs, etc)

### Examples

- `examples/provider_example.py` - Comprehensive usage examples

---

## Additional Resources

- **Models.dev API**: https://models.dev/api.json
- **OpenAI API**: https://platform.openai.com/docs
- **Anthropic API**: https://docs.anthropic.com
- **Google Gemini**: https://ai.google.dev/docs
- **OpenAI Compatibility**: https://github.com/openai/openai-openapi

---

**Version**: 1.0
**Last Updated**: 2025-01-13
**Maintained by**: Seraph MCP Team
