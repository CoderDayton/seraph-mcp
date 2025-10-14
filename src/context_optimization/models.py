"""
Context Optimization Models

Data models for context optimization with LLMLingua-2 and Seraph compression.
"""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class OptimizationResult(BaseModel):
    """Result of context optimization."""

    # Content
    original_content: str = Field(..., description="Original content")
    optimized_content: str = Field(..., description="Optimized content")

    # Token metrics
    tokens_before: int = Field(..., ge=0, description="Original token count")
    tokens_after: int = Field(..., ge=0, description="Optimized token count")
    tokens_saved: int = Field(..., ge=0, description="Tokens saved")
    reduction_percentage: float = Field(..., ge=0, le=100, description="Percentage reduction")
    compression_ratio: float = Field(..., ge=1.0, description="Compression ratio (original/compressed)")

    # Quality
    quality_score: float = Field(..., ge=0, le=1, description="Quality score (0-1)")
    validation_passed: bool = Field(..., description="Whether quality threshold met")

    # Performance
    processing_time_ms: float = Field(..., ge=0, description="Processing time in milliseconds")

    # Method
    method: str = Field(..., description="Compression method: 'ai', 'seraph', 'hybrid', or 'none'")

    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC), description="When optimized")

    model_config = ConfigDict(frozen=False)


class FeedbackRecord(BaseModel):
    """User feedback for optimization quality assessment."""

    # Identification
    content_hash: str = Field(..., description="Hash of original content")
    method: str = Field(..., description="Compression method used")

    # Performance metrics
    tokens_saved: int = Field(..., ge=0, description="Tokens saved")
    quality_score: float = Field(..., ge=0, le=1, description="Automated quality score")

    # User feedback
    user_rating: float = Field(..., ge=0, le=1, description="User quality rating (0-1)")

    # Metadata
    timestamp: float = Field(..., description="Unix timestamp when recorded")

    model_config = ConfigDict(frozen=False)
