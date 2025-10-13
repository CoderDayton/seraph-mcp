"""
Seraph MCP — Configuration Loader

Loads and validates configuration from environment variables and .env files.
Provides a singleton configuration instance for the runtime.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from .schemas import SeraphConfig

logger = logging.getLogger(__name__)

_config_instance: SeraphConfig | None = None


def load_config(
    env_file: str | None = None,
    reload: bool = False,
) -> SeraphConfig:
    """
    Load configuration from environment variables and .env file.

    Args:
        env_file: Path to .env file (default: .env in project root)
        reload: Force reload even if config already loaded

    Returns:
        Validated SeraphConfig instance

    Raises:
        ConfigurationError: If configuration is invalid
    """
    global _config_instance

    if _config_instance is not None and not reload:
        return _config_instance

    # Load .env file if exists
    if env_file:
        env_path = Path(env_file)
    else:
        env_path = Path.cwd() / ".env"

    if env_path.exists():
        logger.info(f"Loading environment from {env_path}")
        load_dotenv(env_path, override=True)
    else:
        logger.info("No .env file found, using environment variables only")

    # Build configuration from environment variables
    # Auto-detect cache backend: Redis if REDIS_URL is set, else memory
    redis_url = os.getenv("REDIS_URL")
    cache_backend = "redis" if redis_url else "memory"

    config_dict = {
        "environment": os.getenv("ENVIRONMENT", "development"),
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
        "cache": {
            "backend": os.getenv("CACHE_BACKEND", cache_backend),  # Auto-detected
            "ttl_seconds": int(os.getenv("CACHE_TTL_SECONDS", "3600")),
            "max_size": int(os.getenv("CACHE_MAX_SIZE", "1000")),
            "namespace": os.getenv("CACHE_NAMESPACE", "seraph"),
            "redis_url": redis_url,
            "redis_max_connections": int(os.getenv("REDIS_MAX_CONNECTIONS", "10")),
            "redis_socket_timeout": int(os.getenv("REDIS_SOCKET_TIMEOUT", "5")),
        },
        "observability": {
            "backend": os.getenv("OBSERVABILITY_BACKEND", "simple"),
            "enable_metrics": os.getenv("ENABLE_METRICS", "true").lower() == "true",
            "enable_tracing": os.getenv("ENABLE_TRACING", "false").lower() == "true",
            "metrics_port": int(os.getenv("METRICS_PORT", "9090")),
            "prometheus_path": os.getenv("PROMETHEUS_PATH", "/metrics"),
            "datadog_api_key": os.getenv("DATADOG_API_KEY"),
            "datadog_site": os.getenv("DATADOG_SITE", "datadoghq.com"),
        },
        "optimization": {
            "enable_optimization": os.getenv("ENABLE_OPTIMIZATION", "true").lower() == "true",
            "optimization_mode": os.getenv("OPTIMIZATION_MODE", "balanced"),
            "quality_threshold": float(os.getenv("QUALITY_THRESHOLD", "0.90")),
            "max_overhead_ms": float(os.getenv("MAX_OVERHEAD_MS", "100.0")),
        },
        "budget": {
            "enable_budget_enforcement": os.getenv("ENABLE_BUDGET_ENFORCEMENT", "false").lower() == "true",
            "daily_budget_limit": float(os.getenv("DAILY_BUDGET_LIMIT")) if os.getenv("DAILY_BUDGET_LIMIT") else None,
            "monthly_budget_limit": float(os.getenv("MONTHLY_BUDGET_LIMIT"))
            if os.getenv("MONTHLY_BUDGET_LIMIT")
            else None,
            "alert_thresholds": [
                float(x.strip()) for x in os.getenv("BUDGET_ALERT_THRESHOLDS", "0.5,0.75,0.9").split(",")
            ],
        },
        "security": {
            "enable_auth": os.getenv("ENABLE_AUTH", "false").lower() == "true",
            "api_keys": [k.strip() for k in os.getenv("API_KEYS", "").split(",") if k.strip()],
            "require_https": os.getenv("REQUIRE_HTTPS", "false").lower() == "true",
            "allowed_hosts": [h.strip() for h in os.getenv("ALLOWED_HOSTS", "*").split(",")],
        },
        "providers": {
            # Auto-enable providers if api_key AND model are both set
            "openai": {
                "enabled": bool(os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_MODEL")),
                "api_key": os.getenv("OPENAI_API_KEY"),
                "model": os.getenv("OPENAI_MODEL"),
                "base_url": os.getenv("OPENAI_BASE_URL"),
                "timeout": float(os.getenv("OPENAI_TIMEOUT", "30.0")),
                "max_retries": int(os.getenv("OPENAI_MAX_RETRIES", "3")),
            },
            "anthropic": {
                "enabled": bool(os.getenv("ANTHROPIC_API_KEY") and os.getenv("ANTHROPIC_MODEL")),
                "api_key": os.getenv("ANTHROPIC_API_KEY"),
                "model": os.getenv("ANTHROPIC_MODEL"),
                "base_url": os.getenv("ANTHROPIC_BASE_URL"),
                "timeout": float(os.getenv("ANTHROPIC_TIMEOUT", "30.0")),
                "max_retries": int(os.getenv("ANTHROPIC_MAX_RETRIES", "3")),
            },
            "gemini": {
                "enabled": bool(os.getenv("GEMINI_API_KEY") and os.getenv("GEMINI_MODEL")),
                "api_key": os.getenv("GEMINI_API_KEY"),
                "model": os.getenv("GEMINI_MODEL"),
                "base_url": os.getenv("GEMINI_BASE_URL"),
                "timeout": float(os.getenv("GEMINI_TIMEOUT", "30.0")),
                "max_retries": int(os.getenv("GEMINI_MAX_RETRIES", "3")),
            },
            "openai_compatible": {
                # For openai_compatible, also require base_url
                "enabled": bool(
                    os.getenv("OPENAI_COMPATIBLE_API_KEY")
                    and os.getenv("OPENAI_COMPATIBLE_MODEL")
                    and os.getenv("OPENAI_COMPATIBLE_BASE_URL")
                ),
                "api_key": os.getenv("OPENAI_COMPATIBLE_API_KEY"),
                "model": os.getenv("OPENAI_COMPATIBLE_MODEL"),
                "base_url": os.getenv("OPENAI_COMPATIBLE_BASE_URL"),
                "timeout": float(os.getenv("OPENAI_COMPATIBLE_TIMEOUT", "30.0")),
                "max_retries": int(os.getenv("OPENAI_COMPATIBLE_MAX_RETRIES", "3")),
            },
        },
    }

    try:
        _config_instance = SeraphConfig(**config_dict)
        logger.info(f"Configuration loaded successfully (environment: {_config_instance.environment})")
        return _config_instance
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        raise


def get_config() -> SeraphConfig:
    """
    Get the current configuration instance.

    Returns:
        Current SeraphConfig instance

    Raises:
        RuntimeError: If config not loaded yet
    """
    global _config_instance

    if _config_instance is None:
        # Auto-load on first access
        return load_config()

    return _config_instance


def reload_config(env_file: str | None = None) -> SeraphConfig:
    """
    Force reload configuration.

    Args:
        env_file: Optional path to .env file

    Returns:
        Reloaded SeraphConfig instance
    """
    return load_config(env_file=env_file, reload=True)
