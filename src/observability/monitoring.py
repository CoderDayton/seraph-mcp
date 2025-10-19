"""
Seraph MCP — Observability Monitoring

Single observability adapter for metrics, traces, and logs per SDD.md.
This is the ONLY place for instrumentation in the core runtime.

Following SDD.md mandatory rules:
- Single adapter rule: This is the only observability module
- All modules must use this adapter, not create their own
- Structured JSON logging with trace IDs
- Prometheus-compatible metrics
"""

import contextvars
import json
import logging
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

# Trace ID context variable for distributed tracing
_trace_id_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar("trace_id", default=None)

# Request ID context variable
_request_id_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)


@dataclass
class MetricValue:
    """Container for metric data."""

    name: str
    value: float
    tags: dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class ObservabilityAdapter:
    """
    Single observability adapter for the entire runtime.

    Provides:
    - Metrics (counters, gauges, histograms)
    - Distributed tracing
    - Structured logging with trace context
    """

    def __init__(
        self,
        enable_metrics: bool = True,
        enable_tracing: bool = True,
        backend: str = "simple",
    ):
        """
        Initialize observability adapter.

        Args:
            enable_metrics: Enable metrics collection
            enable_tracing: Enable distributed tracing
            backend: Backend type (simple, prometheus, datadog)
        """
        self.enable_metrics = enable_metrics
        self.enable_tracing = enable_tracing
        self.backend = backend

        # In-memory metrics store (for simple backend)
        self._metrics: dict[str, list[MetricValue]] = {}

        # Logger with JSON formatting
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Setup structured JSON logger."""
        logger = logging.getLogger("src")

        # Remove existing handlers
        logger.handlers.clear()

        # Create handler with JSON formatter
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        return logger

    def increment(
        self,
        metric: str,
        value: float = 1.0,
        tags: dict[str, str] | None = None,
    ) -> None:
        """
        Increment a counter metric.

        Args:
            metric: Metric name (e.g., "src.cache.hits")
            value: Value to increment by
            tags: Optional metric tags/labels
        """
        if not self.enable_metrics:
            return

        tags = tags or {}
        metric_name = f"src.{metric}"

        if self.backend == "simple":
            self._store_metric(metric_name, value, tags)
        elif self.backend == "prometheus":
            # Prometheus integration would go here (plugin)
            pass
        elif self.backend == "datadog":
            # Datadog integration would go here (plugin)
            pass

    def gauge(
        self,
        metric: str,
        value: float,
        tags: dict[str, str] | None = None,
    ) -> None:
        """
        Set a gauge metric.

        Args:
            metric: Metric name
            value: Current value
            tags: Optional metric tags
        """
        if not self.enable_metrics:
            return

        tags = tags or {}
        metric_name = f"src.{metric}"

        if self.backend == "simple":
            self._store_metric(metric_name, value, tags)

    def histogram(
        self,
        metric: str,
        value: float,
        tags: dict[str, str] | None = None,
    ) -> None:
        """
        Record a histogram metric (for latencies, sizes, etc.).

        Args:
            metric: Metric name
            value: Value to record
            tags: Optional metric tags
        """
        if not self.enable_metrics:
            return

        tags = tags or {}
        metric_name = f"src.{metric}"

        if self.backend == "simple":
            self._store_metric(metric_name, value, tags)

    def event(self, name: str, payload: dict[str, Any]) -> None:
        """
        Record an event.

        Args:
            name: Event name
            payload: Event data
        """
        self.logger.info(
            f"Event: {name}",
            extra={
                "event_name": name,
                "event_payload": payload,
                "trace_id": self.get_trace_id(),
            },
        )

    @contextmanager
    def trace(self, span_name: str, tags: dict[str, str] | None = None) -> Generator[None, None, None]:
        """
        Context manager for tracing a span.

        Args:
            span_name: Name of the span
            tags: Optional span tags

        Example:
            with observability.trace("cache.get"):
                value = await cache.get(key)
        """
        if not self.enable_tracing:
            yield
            return

        start_time = time.perf_counter()
        trace_id = self.get_trace_id()
        tags = tags or {}

        self.logger.debug(
            f"Span started: {span_name}",
            extra={
                "span_name": span_name,
                "trace_id": trace_id,
                "tags": tags,
            },
        )

        try:
            yield
        except Exception as e:
            # Record span error
            self.logger.error(
                f"Span error: {span_name}",
                extra={
                    "span_name": span_name,
                    "trace_id": trace_id,
                    "error": str(e),
                    "tags": tags,
                },
                exc_info=True,
            )
            raise
        finally:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.histogram(
                "span.duration",
                duration_ms,
                tags={"span_name": span_name, **tags},
            )

            self.logger.debug(
                f"Span completed: {span_name}",
                extra={
                    "span_name": span_name,
                    "trace_id": trace_id,
                    "duration_ms": round(duration_ms, 2),
                    "tags": tags,
                },
            )

    def get_trace_id(self) -> str | None:
        """Get current trace ID from context."""
        return _trace_id_ctx.get()

    def set_trace_id(self, trace_id: str) -> None:
        """Set trace ID in context."""
        _trace_id_ctx.set(trace_id)

    def generate_trace_id(self) -> str:
        """Generate a new trace ID and set it in context."""
        trace_id = str(uuid4())
        self.set_trace_id(trace_id)
        return trace_id

    def get_request_id(self) -> str | None:
        """Get current request ID from context."""
        return _request_id_ctx.get()

    def set_request_id(self, request_id: str) -> None:
        """Set request ID in context."""
        _request_id_ctx.set(request_id)

    def get_metrics(self) -> dict[str, list[MetricValue]]:
        """
        Get all collected metrics (simple backend only).

        Returns:
            Dictionary of metric name to list of values
        """
        return self._metrics.copy()

    def clear_metrics(self) -> None:
        """Clear all collected metrics (testing/reset)."""
        self._metrics.clear()

    def _store_metric(
        self,
        name: str,
        value: float,
        tags: dict[str, str],
    ) -> None:
        """Store metric in memory (simple backend)."""
        if name not in self._metrics:
            self._metrics[name] = []

        self._metrics[name].append(MetricValue(name=name, value=value, tags=tags))

        # Keep only last 1000 values per metric to prevent memory bloat
        if len(self._metrics[name]) > 1000:
            self._metrics[name] = self._metrics[name][-1000:]


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add trace ID if available
        trace_id = _trace_id_ctx.get()
        if trace_id:
            log_data["trace_id"] = trace_id

        # Add request ID if available
        request_id = _request_id_ctx.get()
        if request_id:
            log_data["request_id"] = request_id

        # Add any extra fields
        if hasattr(record, "extra") and isinstance(getattr(record, "extra", None), dict):
            log_data.update(record.extra)

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


# Global observability adapter instance (singleton per SDD.md)
_observability_adapter: ObservabilityAdapter | None = None


def get_observability() -> ObservabilityAdapter:
    """
    Get the global observability adapter instance.

    This is the canonical way to access observability per SDD.md.
    All modules MUST use this function, not create their own adapters.

    Returns:
        Global ObservabilityAdapter instance
    """
    global _observability_adapter

    if _observability_adapter is None:
        # Auto-initialize with defaults
        from ..config import get_config

        config = get_config().observability
        _observability_adapter = ObservabilityAdapter(
            enable_metrics=config.enable_metrics,
            enable_tracing=config.enable_tracing,
            backend=config.backend,
        )

    return _observability_adapter


def initialize_observability(
    enable_metrics: bool = True,
    enable_tracing: bool = True,
    backend: str = "simple",
) -> ObservabilityAdapter:
    """
    Initialize the global observability adapter.

    Args:
        enable_metrics: Enable metrics collection
        enable_tracing: Enable distributed tracing
        backend: Backend type

    Returns:
        Initialized ObservabilityAdapter instance
    """
    global _observability_adapter

    _observability_adapter = ObservabilityAdapter(
        enable_metrics=enable_metrics,
        enable_tracing=enable_tracing,
        backend=backend,
    )

    return _observability_adapter
