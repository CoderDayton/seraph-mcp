"""
Seraph MCP — Observability Monitoring

Simple metrics collection with SQLite persistence.
All metrics stored in local database for easy querying and maintenance.
"""

import asyncio
import contextvars
import json
import logging
import time
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import delete, select

from .database import MetricsDatabase
from .db_models import MetricRecord

# Trace ID context variable for distributed tracing
_trace_id_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar("trace_id", default=None)

# Request ID context variable
_request_id_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)


class ObservabilityAdapter:
    """
    Simple observability adapter with SQLite persistence.

    Provides:
    - Metrics (counters, gauges, histograms) → SQLite
    - Distributed tracing (trace IDs)
    - Structured JSON logging
    """

    def __init__(
        self,
        enable_metrics: bool = True,
        enable_tracing: bool = True,
        metrics_db_path: str = "./data/metrics.db",
    ):
        """
        Initialize observability adapter.

        Args:
            enable_metrics: Enable metrics collection
            enable_tracing: Enable distributed tracing
            metrics_db_path: Path to SQLite database
        """
        self.enable_metrics = enable_metrics
        self.enable_tracing = enable_tracing

        # SQLite database for metrics
        self._db = MetricsDatabase(db_path=metrics_db_path)
        self._db_initialized = False

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

    async def _ensure_db_initialized(self) -> None:
        """Ensure database is initialized (idempotent)."""
        if not self._db_initialized:
            await self._db.initialize()
            self._db_initialized = True

    def increment(
        self,
        metric: str,
        value: float = 1.0,
        tags: dict[str, str] | None = None,
    ) -> None:
        """
        Increment a counter metric.

        Args:
            metric: Metric name (e.g., "cache.hits")
            value: Value to increment by
            tags: Optional metric tags/labels
        """
        if not self.enable_metrics:
            return

        tags = tags or {}
        metric_name = f"src.{metric}"

        # Store to SQLite (fire and forget)
        asyncio.create_task(self._store_metric(metric_name, value, tags))

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

        # Store to SQLite (fire and forget)
        asyncio.create_task(self._store_metric(metric_name, value, tags))

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

        # Store to SQLite (fire and forget)
        asyncio.create_task(self._store_metric(metric_name, value, tags))

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
                value = cache.get(key)
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

    async def get_metrics(
        self,
        metric_name: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """
        Get metrics from database.

        Args:
            metric_name: Optional filter by metric name
            limit: Maximum number of records to return

        Returns:
            List of metric records as dictionaries
        """
        await self._ensure_db_initialized()

        async with self._db.get_session() as session:
            query = select(MetricRecord).order_by(MetricRecord.timestamp.desc()).limit(limit)

            if metric_name:
                query = query.where(MetricRecord.name == metric_name)

            result = await session.execute(query)
            records = result.scalars().all()

            return [record.to_dict() for record in records]

    async def clear_metrics(self) -> None:
        """Clear all metrics from database (testing/reset)."""
        await self._ensure_db_initialized()

        async with self._db.get_session() as session:
            await session.execute(delete(MetricRecord))
            await session.commit()

    async def close(self) -> None:
        """Close database connections gracefully."""
        await self._db.close()

    async def _store_metric(
        self,
        name: str,
        value: float,
        tags: dict[str, str],
    ) -> None:
        """Store metric to SQLite database."""
        try:
            await self._ensure_db_initialized()

            record = MetricRecord.from_metric_value(
                name=name,
                value=value,
                tags=tags,
                timestamp=time.time(),
            )

            async with self._db.get_session() as session:
                session.add(record)
                await session.commit()

        except Exception as e:
            # Log error but don't fail the main flow
            self.logger.error(f"Failed to store metric {name}: {e}")


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

        # Add any extra fields from record.__dict__
        for key, value in record.__dict__.items():
            if (
                key not in log_data
                and not key.startswith("_")
                and key
                not in (
                    "args",
                    "msg",
                    "levelname",
                    "levelno",
                    "pathname",
                    "filename",
                    "module",
                    "exc_info",
                    "exc_text",
                    "stack_info",
                    "lineno",
                    "funcName",
                    "created",
                    "msecs",
                    "relativeCreated",
                    "thread",
                    "threadName",
                    "processName",
                    "process",
                    "name",
                    "message",
                )
            ):
                log_data[key] = value

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


# Global observability adapter instance (singleton)
_observability_adapter: ObservabilityAdapter | None = None


def get_observability() -> ObservabilityAdapter:
    """
    Get the global observability adapter instance.

    This is the canonical way to access observability.
    All modules MUST use this function, not create their own adapters.

    Returns:
        Global ObservabilityAdapter instance
    """
    global _observability_adapter

    if _observability_adapter is None:
        # Auto-initialize with defaults from config
        from ..config import get_config

        config = get_config().observability
        _observability_adapter = ObservabilityAdapter(
            enable_metrics=config.enable_metrics,
            enable_tracing=config.enable_tracing,
            metrics_db_path=config.metrics_db_path,
        )

    return _observability_adapter


def initialize_observability(
    enable_metrics: bool = True,
    enable_tracing: bool = True,
    metrics_db_path: str = "./data/metrics.db",
) -> ObservabilityAdapter:
    """
    Initialize the global observability adapter.

    Args:
        enable_metrics: Enable metrics collection
        enable_tracing: Enable distributed tracing
        metrics_db_path: Path to SQLite database

    Returns:
        Initialized ObservabilityAdapter instance
    """
    global _observability_adapter

    _observability_adapter = ObservabilityAdapter(
        enable_metrics=enable_metrics,
        enable_tracing=enable_tracing,
        metrics_db_path=metrics_db_path,
    )

    return _observability_adapter
