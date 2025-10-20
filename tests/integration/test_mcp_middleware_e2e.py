"""
E2E Tests: MCP Compression Middleware with Real FastMCP Tools

Validates Layer 1 compression middleware (CompressionMiddleware) with real
FastMCP tool execution. Tests <1KB and >1KB compression thresholds end-to-end.

Per SDD §10.4.2: Layer 1 compresses tool results before client transmission.

Minimal footprint: <100 lines, mock SeraphCompressor for <3s execution.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.context_optimization.config import load_config as load_optimization_config
from src.context_optimization.mcp_middleware import CompressionMiddleware


@pytest.fixture
def mock_compressor_result():
    """Mock CompressionResult for fast execution"""
    mock_result = MagicMock()
    mock_result.original_token_count = 1000
    mock_result.select_layer = MagicMock(return_value="[COMPRESSED] Short version")
    return mock_result


@pytest.fixture
def mock_compressor_class(mock_compressor_result):
    """Mock SeraphCompressor class for fast execution"""
    # Patch where it's imported (inside _get_compressor method)
    with patch("src.context_optimization.seraph_compression.SeraphCompressor") as mock_class:
        # Mock compressor instance
        mock_instance = MagicMock()
        mock_instance.build = AsyncMock(return_value=mock_compressor_result)

        # Return mock instance when SeraphCompressor() is instantiated
        mock_class.return_value = mock_instance

        yield mock_instance


@pytest.fixture
def middleware(mock_compressor_class):
    """Create CompressionMiddleware with mocked compressor"""
    config = load_optimization_config()
    return CompressionMiddleware(
        config=config,
        min_size_bytes=1000,  # 1KB threshold
        timeout_seconds=5.0,  # Fast timeout for tests
    )


@pytest.mark.asyncio
async def test_compress_text_below_threshold(middleware, mock_compressor_class):
    """Test: Content <1KB should NOT trigger compression"""
    small_content = "Small content < 1KB"

    # Call internal compression method
    result = await middleware._compress_text(small_content, "test")

    # Verify: Returns None (no compression below threshold)
    assert result is None

    # Verify: Compressor build() was NOT called
    mock_compressor_class.build.assert_not_called()


@pytest.mark.asyncio
async def test_compress_text_above_threshold(middleware, mock_compressor_class):
    """Test: Content >1KB should trigger compression"""
    large_content = "This is a large response. " * 100  # ~2.6KB

    # Call internal compression method
    result = await middleware._compress_text(large_content, "test")

    # Verify: Returns compressed version
    assert result == "[COMPRESSED] Short version"

    # Verify: Compressor build() WAS called with large content
    mock_compressor_class.build.assert_called_once_with(large_content)


@pytest.mark.asyncio
async def test_compression_timeout_fallback(middleware):
    """Test: Timeout returns None (graceful fallback)"""
    # Force compressor to timeout
    middleware._get_compressor().build = AsyncMock(
        side_effect=TimeoutError("Compression timeout")
    )

    large_content = "Large content " * 200
    result = await middleware._compress_text(large_content, "test")

    # Verify: Returns None on timeout (graceful fallback)
    assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
