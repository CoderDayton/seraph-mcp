"""
Seraph MCP — Metrics Database Models

SQLAlchemy models for persistent metrics storage.
Simple SQLite backend for maintainability and zero-ops deployment.
"""

import json
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Float, Index, Integer, String, Text
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(AsyncAttrs, DeclarativeBase):
    """Base class for all database models."""

    pass


class MetricRecord(Base):
    """
    Persistent metric storage.

    Schema optimized for:
    - Fast writes (single INSERT per metric)
    - Fast queries (indexed by name + timestamp)
    - JSON tags for flexibility
    - Aggregations (AVG, PERCENTILE on value column)
    """

    __tablename__ = "metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    tags: Mapped[str] = mapped_column(Text, nullable=False, default="{}")  # JSON string
    timestamp: Mapped[datetime] = mapped_column(nullable=False, index=True, default=lambda: datetime.now(UTC))

    # Composite index for time-series queries
    __table_args__ = (Index("idx_metric_name_timestamp", "name", "timestamp"),)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "value": self.value,
            "tags": json.loads(self.tags) if self.tags else {},
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_metric_value(cls, name: str, value: float, tags: dict[str, str], timestamp: float) -> "MetricRecord":
        """Create MetricRecord from MetricValue data."""
        return cls(
            name=name,
            value=value,
            tags=json.dumps(tags),
            timestamp=datetime.fromtimestamp(timestamp, UTC),
        )
