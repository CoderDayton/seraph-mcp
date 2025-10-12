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
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class Environment(str, Enum):
    """Runtime environment."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


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

    backend: CacheBackend = Field(
        default=CacheBackend.MEMORY,
        description="Cache backend to use"
    )
    ttl_seconds: int = Field(
        default=3600,
        ge=0,
        description="Default TTL in seconds (0 = no expiry)"
    )
    max_size: int = Field(
        default=1000,
        ge=1,
        description="Max cache entries (memory backend)"
    )
    namespace: str = Field(
        default="seraph",
        description="Cache key namespace/prefix"
    )

    # Redis-specific settings (only used when backend=redis)
    redis_url: Optional[str] = Field(
        default=None,
        description="Redis connection URL"
    )
    redis_max_connections: int = Field(
        default=10,
        ge=1,
        description="Redis connection pool size"
    )
    redis_socket_timeout: int = Field(
        default=5,
        ge=1,
        description="Redis socket timeout in seconds"
    )

    @field_validator("redis_url")
    @classmethod
    def validate_redis_url(cls, v: Optional[str], info) -> Optional[str]:
        """Ensure redis_url is provided when backend is redis."""
        backend = info.data.get("backend")
        if backend == CacheBackend.REDIS and not v:
            raise ValueError("redis_url is required when cache backend is 'redis'")
        return v


class ObservabilityConfig(BaseModel):
    """Observability and monitoring configuration."""

    backend: ObservabilityBackend = Field(
        default=ObservabilityBackend.SIMPLE,
        description="Observability backend"
    )
    enable_metrics: bool = Field(
        default=True,
        description="Enable metrics collection"
    )
    enable_tracing: bool = Field(
        default=False,
        description="Enable distributed tracing"
    )
    metrics_port: int = Field(
        default=9090,
        ge=1,
        le=65535,
        description="Metrics endpoint port (for prometheus exporter)"
    )

    # Prometheus-specific
    prometheus_path: str = Field(
        default="/metrics",
        description="Prometheus metrics path"
    )

    # Datadog-specific
    datadog_api_key: Optional[str] = Field(
        default=None,
        description="Datadog API key"
    )
    datadog_site: str = Field(
        default="datadoghq.com",
        description="Datadog site"
    )


class FeatureFlags(BaseModel):
    """Feature flags for enabling/disabling platform capabilities."""

    token_optimization: bool = Field(
        default=True,
        description="Enable token optimization features"
    )
    model_routing: bool = Field(
        default=False,
        description="Enable intelligent model routing (future)"
    )
    semantic_cache: bool = Field(
        default=False,
        description="Enable semantic caching (future)"
    )
    context_optimization: bool = Field(
        default=False,
        description="Enable context optimization (future)"
    )
    budget_management: bool = Field(
        default=False,
        description="Enable budget management (future)"
    )
    quality_preservation: bool = Field(
        default=False,
        description="Enable quality preservation (future)"
    )


class TokenOptimizationConfig(BaseModel):
    """Token optimization feature configuration."""

    enabled: bool = Field(
        default=True,
        description="Enable token optimization features"
    )
    default_reduction_target: float = Field(
        default=0.20,
        ge=0.0,
        le=0.5,
        description="Default token reduction target (0.0-0.5 = 0-50%)"
    )
    quality_threshold: float = Field(
        default=0.90,
        ge=0.0,
        le=1.0,
        description="Minimum quality threshold for optimizations (0.0-1.0)"
    )
    cache_optimizations: bool = Field(
        default=True,
        description="Cache optimization patterns for reuse"
    )
    optimization_strategies: list[str] = Field(
        default_factory=lambda: ["whitespace", "redundancy", "compression"],
        description="Active optimization strategies"
    )
    max_overhead_ms: float = Field(
        default=100.0,
        ge=0.0,
        description="Maximum acceptable processing overhead in milliseconds"
    )
    enable_aggressive_mode: bool = Field(
        default=False,
        description="Enable aggressive optimization (may reduce quality)"
    )
    preserve_code_blocks: bool = Field(
        default=True,
        description="Preserve code blocks and structured content"
    )
    preserve_formatting: bool = Field(
        default=True,
        description="Preserve important formatting (lists, tables, etc.)"
    )
    cache_ttl_seconds: int = Field(
        default=3600,
        ge=0,
        description="TTL for cached optimizations in seconds"
    )


class BudgetConfig(BaseModel):
    """Budget enforcement configuration (plugin-provided features)."""

    enable_budget_enforcement: bool = Field(
        default=False,
        description="Enable budget limits (requires plugin)"
    )
    daily_budget_limit: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Daily budget in USD"
    )
    monthly_budget_limit: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Monthly budget in USD"
    )
    alert_thresholds: list[float] = Field(
        default_factory=lambda: [0.5, 0.75, 0.9],
        description="Budget alert thresholds (0.0-1.0)"
    )

    @field_validator("alert_thresholds")
    @classmethod
    def validate_thresholds(cls, v: list[float]) -> list[float]:
        """Ensure all thresholds are between 0 and 1."""
        for threshold in v:
            if not 0.0 <= threshold <= 1.0:
                raise ValueError(f"Alert threshold must be between 0.0 and 1.0, got {threshold}")
        return sorted(v)


class SecurityConfig(BaseModel):
    """Security configuration."""

    enable_auth: bool = Field(
        default=False,
        description="Enable authentication (MCP client-side)"
    )
    api_keys: list[str] = Field(
        default_factory=list,
        description="Valid API keys (for plugin HTTP adapters)"
    )
    allowed_hosts: list[str] = Field(
        default_factory=lambda: ["*"],
        description="Allowed host headers (for plugin HTTP adapters)"
    )


class SeraphConfig(BaseModel):
    """Root configuration for Seraph MCP."""

    environment: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Runtime environment"
    )
    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Logging level"
    )

    cache: CacheConfig = Field(default_factory=CacheConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    features: FeatureFlags = Field(default_factory=FeatureFlags)
    token_optimization: TokenOptimizationConfig = Field(default_factory=TokenOptimizationConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)

    @field_validator("security")
    @classmethod
    def validate_security(cls, v: SecurityConfig, info) -> SecurityConfig:
        """Enforce security requirements in production."""
        environment = info.data.get("environment")
        if environment == Environment.PRODUCTION:
            if v.enable_auth and not v.api_keys:
                raise ValueError("At least one API key must be configured when auth is enabled in production")
        return v

    class Config:
        use_enum_values = True
        validate_assignment = True
