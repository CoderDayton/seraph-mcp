"""
Tests for Context Optimization module
"""

import pytest
import asyncio
from src.context_optimization import (
    ContextOptimizationConfig,
    load_config,
    ContextOptimizer,
    optimize_content,
    OptimizationResult,
)


def test_config_loading():
    """Test configuration loading with defaults"""
    config = load_config()

    assert config.enabled is True
    assert config.quality_threshold == 0.90
    assert config.max_overhead_ms == 100.0


def test_config_creation():
    """Test manual config creation"""
    config = ContextOptimizationConfig(
        enabled=False,
        quality_threshold=0.95,
        max_overhead_ms=50.0,
    )

    assert config.enabled is False
    assert config.quality_threshold == 0.95
    assert config.max_overhead_ms == 50.0


@pytest.mark.asyncio
async def test_optimizer_passthrough_when_disabled():
    """Test that optimizer passes through content when disabled"""
    config = ContextOptimizationConfig(enabled=False)
    optimizer = ContextOptimizer(config=config, provider=None)

    content = "This is test content that should not be modified."
    result = await optimizer.optimize(content)

    assert result.original_content == content
    assert result.optimized_content == content
    assert result.tokens_saved == 0
    assert result.validation_passed is True
    assert result.rollback_occurred is False


@pytest.mark.asyncio
async def test_optimizer_skips_small_content():
    """Test that optimizer skips very small content"""
    config = ContextOptimizationConfig(enabled=True)
    optimizer = ContextOptimizer(config=config, provider=None)

    content = "Small"
    result = await optimizer.optimize(content)

    assert result.optimized_content == content
    assert result.tokens_saved == 0


@pytest.mark.asyncio
async def test_optimizer_without_provider():
    """Test optimizer behavior when no provider is available"""
    config = ContextOptimizationConfig(enabled=True)
    optimizer = ContextOptimizer(config=config, provider=None)

    content = "This is a longer piece of content that would normally be optimized but we have no provider so it should pass through cleanly without errors."
    result = await optimizer.optimize(content)

    # Without provider, should return original content safely
    assert result.optimized_content == content
    assert result.validation_passed is True


def test_optimizer_stats():
    """Test statistics tracking"""
    config = ContextOptimizationConfig(enabled=True)
    optimizer = ContextOptimizer(config=config, provider=None)

    stats = optimizer.get_stats()

    assert stats["total_optimizations"] == 0
    assert stats["successful_optimizations"] == 0
    assert stats["total_tokens_saved"] == 0
    assert stats["success_rate"] == 0.0
    assert stats["cache_size"] == 0


def test_optimizer_cache_operations():
    """Test cache clear functionality"""
    config = ContextOptimizationConfig(enabled=True)
    optimizer = ContextOptimizer(config=config, provider=None)

    # Add something to cache manually
    optimizer.cache["test_hash"] = "test_value"
    assert len(optimizer.cache) == 1

    # Clear cache
    optimizer.clear_cache()
    assert len(optimizer.cache) == 0


@pytest.mark.asyncio
async def test_optimizer_timeout_handling():
    """Test that optimizer respects timeout"""
    config = ContextOptimizationConfig(enabled=True, max_overhead_ms=1.0)  # Very short timeout
    
    # Mock provider that takes too long
    class SlowProvider:
        async def generate(self, **kwargs):
            await asyncio.sleep(1.0)  # Sleep longer than timeout
            return {"content": "compressed"}
    
    optimizer = ContextOptimizer(config=config, provider=SlowProvider())
    
    content = "This content would take too long to optimize."
    result = await optimizer.optimize(content, timeout_ms=1.0)
    
    # Should rollback due to timeout
    assert result.optimized_content == content
    assert result.rollback_occurred is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
