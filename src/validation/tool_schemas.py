"""
Seraph MCP - Tool Input Validation Schemas

Pydantic models for validating all MCP tool inputs.

P0 Implementation:
- Strict type validation
- Field constraints (min/max lengths, value ranges)
- Default values where appropriate
- Comprehensive field descriptions
"""

from typing import Any

from pydantic import BaseModel, Field, field_validator


class GetOptimizationStatsInput(BaseModel):
    """Input validation for get_optimization_stats tool (no parameters)."""

    pass


class GetOptimizationSettingsInput(BaseModel):
    """Input validation for get_optimization_settings tool (no parameters)."""

    pass


class LookupSemanticCacheInput(BaseModel):
    """Input validation for lookup_semantic_cache tool."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=10_000,
        description="Query text to search for (1-10K characters)",
    )
    threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Similarity threshold (0-1), uses config default if None",
    )
    max_results: int = Field(
        default=1,
        ge=1,
        le=100,
        description="Maximum number of results to return (1-100)",
    )

    @field_validator("query")
    @classmethod
    def validate_query_not_empty(cls, v: str) -> str:
        """Ensure query is not just whitespace."""
        if not v.strip():
            raise ValueError("Query cannot be empty or only whitespace")
        return v


class StoreInSemanticCacheInput(BaseModel):
    """Input validation for store_in_semantic_cache tool."""

    key: str = Field(
        ...,
        min_length=1,
        max_length=10_000,
        description="Cache key (text to embed) (1-10K characters)",
    )
    value: Any = Field(
        ...,
        description="Value to cache (will be converted to string)",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Optional metadata dictionary",
    )

    @field_validator("key")
    @classmethod
    def validate_key_not_empty(cls, v: str) -> str:
        """Ensure key is not just whitespace."""
        if not v.strip():
            raise ValueError("Key cannot be empty or only whitespace")
        return v


class SearchSemanticCacheInput(BaseModel):
    """Input validation for search_semantic_cache tool."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=10_000,
        description="Query text (1-10K characters)",
    )
    limit: int | None = Field(
        default=None,
        ge=1,
        le=100,
        description="Maximum results (1-100), uses config default if None",
    )
    threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Similarity threshold (0-1), uses config default if None",
    )

    @field_validator("query")
    @classmethod
    def validate_query_not_empty(cls, v: str) -> str:
        """Ensure query is not just whitespace."""
        if not v.strip():
            raise ValueError("Query cannot be empty or only whitespace")
        return v


class GetSemanticCacheStatsInput(BaseModel):
    """Input validation for get_semantic_cache_stats tool (no parameters)."""

    pass


class ClearSemanticCacheInput(BaseModel):
    """Input validation for clear_semantic_cache tool (no parameters)."""

    pass


class CountTokensInput(BaseModel):
    """Input validation for count_tokens tool."""

    content: str = Field(
        ...,
        min_length=1,
        max_length=1_000_000,
        description="Content to count tokens for (1-1M characters)",
    )
    model: str = Field(
        default="gpt-4",
        min_length=1,
        max_length=100,
        description="Model name for tokenization (e.g., 'gpt-4', 'claude-3-opus')",
    )
    include_breakdown: bool = Field(
        default=False,
        description="Include character/word breakdown",
    )

    @field_validator("content")
    @classmethod
    def validate_content_not_empty(cls, v: str) -> str:
        """Ensure content is not just whitespace."""
        if not v.strip():
            raise ValueError("Content cannot be empty or only whitespace")
        return v


class EstimateCostInput(BaseModel):
    """Input validation for estimate_cost tool."""

    content: str = Field(
        ...,
        min_length=1,
        max_length=1_000_000,
        description="Content to estimate cost for (1-1M characters)",
    )
    model: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Model name (e.g., 'gpt-4', 'claude-3-opus')",
    )
    operation: str = Field(
        default="completion",
        pattern="^(completion|embedding)$",
        description="Operation type: completion or embedding",
    )
    output_tokens: int | None = Field(
        default=None,
        ge=1,
        le=100_000,
        description="Expected output tokens for cost calculation (1-100K)",
    )

    @field_validator("content")
    @classmethod
    def validate_content_not_empty(cls, v: str) -> str:
        """Ensure content is not just whitespace."""
        if not v.strip():
            raise ValueError("Content cannot be empty or only whitespace")
        return v


class AnalyzeTokenEfficiencyInput(BaseModel):
    """Input validation for analyze_token_efficiency tool."""

    content: str = Field(
        ...,
        min_length=1,
        max_length=1_000_000,
        description="Content to analyze (1-1M characters)",
    )
    model: str = Field(
        default="gpt-4",
        min_length=1,
        max_length=100,
        description="Model name for analysis",
    )

    @field_validator("content")
    @classmethod
    def validate_content_not_empty(cls, v: str) -> str:
        """Ensure content is not just whitespace."""
        if not v.strip():
            raise ValueError("Content cannot be empty or only whitespace")
        return v


class CheckBudgetInput(BaseModel):
    """Input validation for check_budget tool."""

    estimated_cost: float | None = Field(
        default=None,
        ge=0.0,
        description="Estimated cost of upcoming request in USD (non-negative)",
    )


class GetBudgetStatsInput(BaseModel):
    """Input validation for get_budget_stats tool (no parameters)."""

    pass


class GetUsageReportInput(BaseModel):
    """Input validation for get_usage_report tool."""

    period: str = Field(
        default="month",
        pattern="^(day|week|month)$",
        description="Time period: day, week, or month",
    )
    details: bool = Field(
        default=False,
        description="Include detailed breakdown",
    )


class ForecastSpendingInput(BaseModel):
    """Input validation for forecast_spending tool."""

    days_ahead: int = Field(
        default=7,
        ge=1,
        le=90,
        description="Number of days to forecast (1-90)",
    )


class CacheGetInput(BaseModel):
    """Input validation for cache_get tool."""

    key: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Cache key to retrieve (1-1000 characters)",
    )


class CacheSetInput(BaseModel):
    """Input validation for cache_set tool."""

    key: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Cache key (1-1000 characters)",
    )
    value: Any = Field(
        ...,
        description="Value to store",
    )
    ttl: int | None = Field(
        default=None,
        ge=0,
        le=86400 * 30,  # 30 days max
        description="Time-to-live in seconds (0-2592000, None=use default)",
    )


class CacheDeleteInput(BaseModel):
    """Input validation for cache_delete tool."""

    key: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Cache key to delete (1-1000 characters)",
    )


class CacheClearInput(BaseModel):
    """Input validation for cache_clear tool (no parameters)."""

    pass


class GetCacheStatsInput(BaseModel):
    """Input validation for get_cache_stats tool (no parameters)."""

    pass


class CheckStatusInput(BaseModel):
    """Input validation for check_status tool."""

    include_details: bool = Field(
        default=False,
        description="Include detailed cache and observability stats",
    )


class GetMetricsInput(BaseModel):
    """Input validation for get_metrics tool (no parameters)."""

    pass
