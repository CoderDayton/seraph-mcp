"""
Seraph MCP — AI Optimization Platform

Comprehensive token optimization, model routing, semantic caching,
and cost management for LLM APIs.
"""

__version__ = "1.0.0"
__author__ = "Seraph Team"
__email__ = "team@seraph-mcp.dev"

# Export main components for external use
from .server import mcp

__all__ = ["mcp"]
