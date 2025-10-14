"""
Seraph MCP - Input Validation Module

Provides Pydantic-based validation for all MCP tool inputs.

P0 Implementation:
- Strict type validation for all tool parameters
- Field constraints (lengths, ranges, types)
- Standardized error responses with ErrorCode
- Decorator for applying validation to tools
"""

from .decorators import validate_input
from .tool_schemas import (
    GetBudgetStatsInput,
    GetOptimizationSettingsInput,
    GetOptimizationStatsInput,
    LookupSemanticCacheInput,
    OptimizeContextInput,
    StoreInSemanticCacheInput,
)

__all__ = [
    # Decorator
    "validate_input",
    # Tool input schemas
    "OptimizeContextInput",
    "GetOptimizationStatsInput",
    "GetOptimizationSettingsInput",
    "LookupSemanticCacheInput",
    "StoreInSemanticCacheInput",
    "GetBudgetStatsInput",
]
