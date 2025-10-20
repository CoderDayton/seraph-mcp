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
    StoreInSemanticCacheInput,
)

__all__ = [
    # Decorator
    "validate_input",
    # Tool input schemas
    "GetOptimizationStatsInput",
    "GetOptimizationSettingsInput",
    "LookupSemanticCacheInput",
    "StoreInSemanticCacheInput",
    "GetBudgetStatsInput",
]
