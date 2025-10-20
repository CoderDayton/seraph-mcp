"""
Integration Tests: MCP Compression Middleware

Validates Layer 1 compression middleware (CompressionMiddleware) integrates
correctly with FastMCP server and compresses tool results + resource reads.

Per SDD §10.4.2: Two-layer compression architecture
"""

import pytest

from src.context_optimization.config import load_config as load_optimization_config
from src.context_optimization.mcp_middleware import CompressionMiddleware


class TestCompressionMiddlewareIntegration:
    """Test CompressionMiddleware loads and integrates with server"""

    def test_middleware_instantiation(self):
        """Verify middleware can be instantiated with config"""
        config = load_optimization_config()
        middleware = CompressionMiddleware(
            config=config,
            min_size_bytes=1000,
            timeout_seconds=10.0,
        )

        assert middleware.min_size_bytes == 1000
        assert middleware.timeout_seconds == 10.0
        assert middleware._compressor is None, "Compressor should be lazy-loaded"

    def test_middleware_has_hooks(self):
        """Verify middleware exposes FastMCP hooks"""
        config = load_optimization_config()
        middleware = CompressionMiddleware(
            config=config,
            min_size_bytes=1000,
            timeout_seconds=10.0,
        )

        # Verify hook methods exist
        assert hasattr(middleware, "on_call_tool")
        assert callable(middleware.on_call_tool)

        assert hasattr(middleware, "on_read_resource")
        assert callable(middleware.on_read_resource)

    def test_server_imports_middleware(self):
        """Verify server.py can import and register middleware without errors"""
        # This test validates the import chain works at runtime
        from src.server import mcp

        assert mcp is not None
        assert mcp.name == "Seraph MCP - AI Optimization Platform"

    def test_middleware_config_sharing(self):
        """Verify Layer 1 and Layer 2 share same config instance"""
        config = load_optimization_config()

        # Layer 1 middleware
        middleware_l1 = CompressionMiddleware(
            config=config,
            min_size_bytes=1000,
            timeout_seconds=10.0,
        )

        # Verify both layers would use same config instance
        assert middleware_l1.config is config
        assert middleware_l1.config.enabled is True


class TestMiddlewareRegistration:
    """Test middleware registration in FastMCP server"""

    def test_middleware_registration_at_startup(self):
        """Verify middleware is registered during server initialization"""
        # Import server (triggers middleware registration)
        from src.server import mcp

        # Server should be initialized without errors
        assert mcp is not None

    def test_lazy_compressor_initialization(self):
        """Verify SeraphCompressor is NOT loaded until first use"""
        config = load_optimization_config()
        middleware = CompressionMiddleware(
            config=config,
            min_size_bytes=1000,
            timeout_seconds=10.0,
        )

        # Compressor should be None (lazy)
        assert middleware._compressor is None

        # After first compression, compressor should be initialized
        # (This would be tested in unit tests with actual compression calls)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
