"""
Seraph MCP — Configuration Schemas

Defines typed configuration models using Pydantic for validation and type safety.
All configuration must be defined here and validated at startup.

Following SDD.md:
- MCP stdio protocol (no HTTP server config needed)
- Only minimal core configuration
- All config via environment variables
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Unified runtime budget configuration:
# We import the canonical BudgetConfig from budget_management to avoid drift
# and maintain a single source of truth.
from ..budget_management.config import BudgetConfig as BudgetConfig


class Environment(str, Enum):
    """Runtime environment."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


class CacheBackend(str, Enum):
    """Supported cache backends."""

    MEMORY = "memory"
    REDIS = "redis"  # Requires plugin


class ObservabilityBackend(str, Enum):
    """Supported observability backends."""

    SIMPLE = "simple"
    PROMETHEUS = "prometheus"  # Requires plugin
    DATADOG = "datadog"  # Requires plugin


class LogLevel(str, Enum):
    """Log levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class CacheConfig(BaseModel):
    """Cache configuration."""

    backend: CacheBackend = Field(default=CacheBackend.MEMORY, description="Cache backend to use")
    ttl_seconds: int = Field(default=3600, ge=0, description="Default TTL in seconds (0 = no expiry)")
    max_size: int = Field(default=1000, ge=1, description="Max cache entries (memory backend)")
    namespace: str = Field(default="seraph", description="Cache key namespace/prefix")

    # Redis-specific settings (only used when backend=redis)
    redis_url: str | None = Field(default=None, description="Redis connection URL")
    redis_max_connections: int = Field(default=10, ge=1, description="Redis connection pool size")
    redis_socket_timeout: int = Field(default=5, ge=1, description="Redis socket timeout in seconds")

    @field_validator("redis_url")
    @classmethod
    def validate_redis_url(cls, v: str | None, info: Any) -> str | None:
        """Ensure redis_url is provided when backend is redis."""
        backend = info.data.get("backend")
        if backend == CacheBackend.REDIS and not v:
            raise ValueError("redis_url is required when cache backend is 'redis'")
        return v


class ObservabilityConfig(BaseModel):
    """Observability and monitoring configuration."""

    backend: ObservabilityBackend = Field(default=ObservabilityBackend.SIMPLE, description="Observability backend")
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    enable_tracing: bool = Field(default=False, description="Enable distributed tracing")
    metrics_port: int = Field(
        default=9090,
        ge=1,
        le=65535,
        description="Metrics endpoint port (for prometheus exporter)",
    )

    # Prometheus-specific
    prometheus_path: str = Field(default="/metrics", description="Prometheus metrics path")

    # Datadog-specific
    datadog_api_key: str | None = Field(default=None, description="Datadog API key")
    datadog_site: str = Field(default="datadoghq.com", description="Datadog site")


class FeatureFlags(BaseModel):
    """Feature flags for enabling/disabling platform capabilities."""

    semantic_cache: bool = Field(default=False, description="Enable semantic caching")
    context_optimization: bool = Field(default=False, description="Enable context optimization")
    budget_management: bool = Field(default=False, description="Enable budget management")
    quality_preservation: bool = Field(default=False, description="Enable quality preservation (future)")


class SecurityConfig(BaseModel):
    """Security configuration."""

    enable_auth: bool = Field(default=False, description="Enable authentication (MCP client-side)")
    api_keys: list[str] = Field(default_factory=list, description="Valid API keys (for plugin HTTP adapters)")
    allowed_hosts: list[str] = Field(
        default_factory=lambda: ["*"],
        description="Allowed host headers (for plugin HTTP adapters)",
    )


class ProviderConfig(BaseModel):
    """Configuration for a single AI model provider."""

    enabled: bool = Field(default=True, description="Whether this provider is enabled")
    api_key: str | None = Field(default=None, description="API key for the provider")
    model: str | None = Field(
        default=None,
        description="Model name to use (e.g., 'gpt-4', 'claude-3-opus', 'llama-3-8b'). Required for operation.",
    )
    base_url: str | None = Field(default=None, description="Custom base URL (optional, for openai-compatible)")
    timeout: float = Field(default=30.0, ge=1.0, description="Request timeout in seconds")
    max_retries: int = Field(default=3, ge=0, description="Maximum retry attempts")


class ProvidersConfig(BaseModel):
    """Configuration for AI model providers."""

    openai: ProviderConfig = Field(default_factory=ProviderConfig, description="OpenAI provider configuration")
    anthropic: ProviderConfig = Field(default_factory=ProviderConfig, description="Anthropic provider configuration")
    gemini: ProviderConfig = Field(
        default_factory=ProviderConfig,
        description="Google Gemini provider configuration",
    )
    openai_compatible: ProviderConfig = Field(
        default_factory=ProviderConfig,
        description="OpenAI-compatible provider configuration (for custom endpoints)",
    )


class SeraphConfig(BaseModel):
    """Root configuration for Seraph MCP."""

    environment: Environment = Field(default=Environment.DEVELOPMENT, description="Runtime environment")
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Logging level")

    cache: CacheConfig = Field(default_factory=CacheConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    features: FeatureFlags = Field(default_factory=FeatureFlags)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)

    # Context Optimization (lazy-loaded to avoid import issues)
    @property
    def context_optimization(self) -> Any:
        """Load context optimization config on demand."""
        from ..context_optimization.config import load_config as load_context_config

        return load_context_config()

    @field_validator("security")
    @classmethod
    def validate_security(cls, v: SecurityConfig, info: Any) -> SecurityConfig:
        """Enforce security requirements in production."""
        environment = info.data.get("environment")
        if environment == Environment.PRODUCTION:
            if v.enable_auth and not v.api_keys:
                raise ValueError("At least one API key must be configured when auth is enabled in production")
        return v

    model_config = ConfigDict(use_enum_values=True, validate_assignment=True)
