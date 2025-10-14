"""
Test OptimizationResult Field Population

Verifies that OptimizationResult instances include all required fields,
especially compression_ratio and processing_time_ms.
"""

import pytest

from src.context_optimization.config import ContextOptimizationConfig
from src.context_optimization.models import OptimizationResult
from src.context_optimization.optimizer import ContextOptimizer


@pytest.fixture
def config() -> ContextOptimizationConfig:
    """Create test configuration"""
    return ContextOptimizationConfig(
        enabled=True,
        compression_method="seraph",  # Use deterministic method for testing
        quality_threshold=0.85,
        max_overhead_ms=200.0,
    )


@pytest.fixture
def optimizer(config: ContextOptimizationConfig) -> ContextOptimizer:
    """Create optimizer instance"""
    return ContextOptimizer(config=config, provider=None)


@pytest.mark.asyncio
async def test_optimization_result_has_all_required_fields(optimizer: ContextOptimizer) -> None:
    """Test that OptimizationResult includes all required fields"""
    content = "This is a test content that should be optimized. " * 50

    result = await optimizer.optimize(content)

    # Verify all required fields are present
    assert hasattr(result, "original_content")
    assert hasattr(result, "optimized_content")
    assert hasattr(result, "tokens_before")
    assert hasattr(result, "tokens_after")
    assert hasattr(result, "tokens_saved")
    assert hasattr(result, "reduction_percentage")
    assert hasattr(result, "compression_ratio")
    assert hasattr(result, "quality_score")
    assert hasattr(result, "validation_passed")
    assert hasattr(result, "processing_time_ms")
    assert hasattr(result, "method")
    assert hasattr(result, "metadata")
    assert hasattr(result, "timestamp")


@pytest.mark.asyncio
async def test_compression_ratio_calculation(optimizer: ContextOptimizer) -> None:
    """Test that compression_ratio is calculated correctly"""
    content = "Test content for compression ratio. " * 100

    result = await optimizer.optimize(content)

    # Compression ratio should be >= 1.0
    assert result.compression_ratio >= 1.0

    # Should equal tokens_before / tokens_after
    if result.tokens_after > 0:
        expected_ratio = result.tokens_before / result.tokens_after
        assert abs(result.compression_ratio - expected_ratio) < 0.01
    else:
        # If no tokens after, ratio should be 1.0
        assert result.compression_ratio == 1.0


@pytest.mark.asyncio
async def test_processing_time_ms_is_populated(optimizer: ContextOptimizer) -> None:
    """Test that processing_time_ms is populated and valid"""
    content = "Test content for timing. " * 50

    result = await optimizer.optimize(content)

    # Processing time should be non-negative
    assert result.processing_time_ms >= 0

    # Should be reasonable (not zero unless instant)
    assert result.processing_time_ms >= 0.0


@pytest.mark.asyncio
async def test_metadata_contains_extra_fields(optimizer: ContextOptimizer) -> None:
    """Test that metadata dict contains moved fields"""
    content = "Test content for metadata. " * 50

    result = await optimizer.optimize(content)

    # Metadata should be a dict
    assert isinstance(result.metadata, dict)

    # Should contain expected fields
    assert "cost_savings_usd" in result.metadata
    assert "rollback_occurred" in result.metadata

    # Cost savings should be a number
    assert isinstance(result.metadata["cost_savings_usd"], int | float)

    # Rollback should be a boolean
    assert isinstance(result.metadata["rollback_occurred"], bool)


@pytest.mark.asyncio
async def test_passthrough_result_fields(optimizer: ContextOptimizer) -> None:
    """Test passthrough results when optimization is skipped"""
    # Very short content that might be skipped
    content = "Short"

    result = await optimizer.optimize(content)

    # All fields should still be present
    assert result.compression_ratio == 1.0  # No compression
    assert result.processing_time_ms >= 0
    assert result.method in ["ai", "seraph", "hybrid", "none"]
    assert isinstance(result.metadata, dict)


@pytest.mark.asyncio
async def test_no_compression_gives_ratio_1(optimizer: ContextOptimizer) -> None:
    """Test that no compression yields ratio of 1.0"""
    content = "Test"

    result = await optimizer.optimize(content)

    if result.tokens_before == result.tokens_after:
        # No compression occurred
        assert result.compression_ratio == 1.0
        assert result.tokens_saved == 0
        assert result.reduction_percentage == 0.0


@pytest.mark.asyncio
async def test_compression_gives_ratio_greater_than_1(optimizer: ContextOptimizer) -> None:
    """Test that actual compression yields ratio > 1.0"""
    # Long content that should compress
    content = (
        "This is a long piece of content that should definitely be compressed "
        "by the optimizer. It contains multiple sentences and repeated patterns "
        "that make it a good candidate for token reduction. "
    ) * 100

    result = await optimizer.optimize(content)

    if result.tokens_saved > 0:
        # Compression occurred
        assert result.compression_ratio > 1.0
        assert result.tokens_before > result.tokens_after

        # Verify the math
        expected_ratio = result.tokens_before / result.tokens_after
        assert abs(result.compression_ratio - expected_ratio) < 0.01


@pytest.mark.asyncio
async def test_method_field_is_valid(optimizer: ContextOptimizer) -> None:
    """Test that method field contains valid value"""
    content = "Test content. " * 50

    result = await optimizer.optimize(content)

    # Method should be one of the valid options
    assert result.method in ["ai", "seraph", "hybrid", "none"]


def test_optimization_result_model_validation() -> None:
    """Test OptimizationResult Pydantic validation"""
    # Valid result should pass
    result = OptimizationResult(
        original_content="test",
        optimized_content="test",
        tokens_before=100,
        tokens_after=60,
        tokens_saved=40,
        reduction_percentage=40.0,
        compression_ratio=100 / 60,
        quality_score=0.95,
        validation_passed=True,
        processing_time_ms=50.0,
        method="seraph",
    )

    assert result.compression_ratio > 1.0
    assert result.processing_time_ms >= 0


def test_compression_ratio_must_be_at_least_1() -> None:
    """Test that compression_ratio validates >= 1.0"""
    with pytest.raises(ValueError):
        OptimizationResult(
            original_content="test",
            optimized_content="test",
            tokens_before=100,
            tokens_after=60,
            tokens_saved=40,
            reduction_percentage=40.0,
            compression_ratio=0.5,  # Invalid: less than 1.0
            quality_score=0.95,
            validation_passed=True,
            processing_time_ms=50.0,
            method="seraph",
        )


def test_processing_time_must_be_non_negative() -> None:
    """Test that processing_time_ms validates >= 0"""
    with pytest.raises(ValueError):
        OptimizationResult(
            original_content="test",
            optimized_content="test",
            tokens_before=100,
            tokens_after=60,
            tokens_saved=40,
            reduction_percentage=40.0,
            compression_ratio=100 / 60,
            quality_score=0.95,
            validation_passed=True,
            processing_time_ms=-10.0,  # Invalid: negative
            method="seraph",
        )
