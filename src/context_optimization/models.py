"""
Context Optimization Models

Minimal data models for AI-powered context optimization.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class OptimizationResult(BaseModel):
    """Result of AI-powered context optimization"""

    # Content
    original_content: str = Field(..., description="Original content")
    optimized_content: str = Field(..., description="Optimized content")

    # Metrics
    tokens_before: int = Field(..., ge=0, description="Original token count")
    tokens_after: int = Field(..., ge=0, description="Optimized token count")
    tokens_saved: int = Field(..., ge=0, description="Tokens saved")
    reduction_percentage: float = Field(..., ge=0, le=100, description="Percentage reduction")

    # Quality
    quality_score: float = Field(..., ge=0, le=1, description="Quality score (0-1)")
    validation_passed: bool = Field(..., description="Whether quality threshold met")

    # Performance
    optimization_time_ms: float = Field(..., ge=0, description="Time taken in milliseconds")

    # Cost tracking (for budget integration)
    cost_savings_usd: float = Field(default=0.0, ge=0, description="Cost savings in USD")
    model_name: Optional[str] = Field(default=None, description="Model used for optimization")

    # Method used
    method: str = Field(default="ai", description="Compression method used: 'ai', 'seraph', or 'hybrid'")

    # Metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When optimized")
    rollback_occurred: bool = Field(default=False, description="Whether rollback happened")

    class Config:
        frozen = False


class FeedbackRecord(BaseModel):
    """Learning feedback for adaptive optimization"""

    # Identification
    record_id: str = Field(..., description="Unique record ID")
    content_hash: str = Field(..., description="Hash of content")

    # Performance
    tokens_saved: int = Field(..., ge=0, description="Tokens saved")
    reduction_percentage: float = Field(..., ge=0, le=100, description="Reduction percentage")
    quality_score: float = Field(..., ge=0, le=1, description="Quality score")

    # Outcome
    validation_passed: bool = Field(..., description="Validation passed")
    optimization_time_ms: float = Field(..., ge=0, description="Time taken")

    # Learning signal
    success_score: float = Field(..., ge=0, le=1, description="Overall success (0-1)")

    # Metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When recorded")

    class Config:
        frozen = False
