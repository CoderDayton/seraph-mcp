"""
Token Optimization Plugin Configuration

Configuration schema and validation for the Token Optimization Plugin.
Follows SDD.md specifications for typed configuration with Pydantic.
"""

from typing import List

from pydantic import BaseModel, Field


class TokenOptimizationConfig(BaseModel):
    """
    Configuration for Token Optimization Plugin.

    Per SDD.md specifications:
    - All fields typed and validated
    - Sensible defaults provided
    - Integration with core cache system
    """

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

    optimization_strategies: List[str] = Field(
        default=["whitespace", "redundancy", "compression"],
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

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "enabled": True,
                    "default_reduction_target": 0.20,
                    "quality_threshold": 0.90,
                    "cache_optimizations": True,
                    "optimization_strategies": ["whitespace", "redundancy", "compression"],
                    "max_overhead_ms": 100.0,
                }
            ]
        }
    }

    def validate_strategies(self) -> None:
        """Validate optimization strategies are supported."""
        valid_strategies = {
            "whitespace",
            "redundancy",
            "compression",
            "summarization",
            "deduplication"
        }

        for strategy in self.optimization_strategies:
            if strategy not in valid_strategies:
                raise ValueError(
                    f"Invalid optimization strategy: {strategy}. "
                    f"Valid strategies: {valid_strategies}"
                )
