"""
Seraph MCP — Observability Module

Single observability adapter for the entire runtime per SDD.md.
All metrics, traces, and logs MUST go through this module.

Following SDD.md mandatory rules:
- Single adapter rule: This is the ONLY observability module
- No other module may create metrics/tracing independently
- All instrumentation goes through get_observability()

Usage:
    from src.observability import get_observability

    obs = get_observability()
    obs.increment("cache.hits")
    obs.gauge("cache.size", 100)

    with obs.trace("operation"):
        # traced code here
        pass
"""

from .monitoring import (
    ObservabilityAdapter,
    get_observability,
    initialize_observability,
)

__all__ = [
    # Canonical observability adapter
    "ObservabilityAdapter",
    "get_observability",
    "initialize_observability",
]
