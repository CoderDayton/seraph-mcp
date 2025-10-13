"""
Unit Tests for Token Optimizer

Tests the TokenOptimizer class with all optimization strategies.
Covers whitespace, redundancy, compression, deduplication, and quality metrics.
"""

import time
from unittest.mock import Mock, patch

import pytest

from src.token_optimization.optimizer import (
    OptimizationResult,
    OptimizationStrategy,
    TokenOptimizer,
    get_optimizer,
)


class TestOptimizationStrategy:
    """Tests for OptimizationStrategy enum."""

    def test_strategy_values(self):
        """Test that all strategy values are strings."""
        assert OptimizationStrategy.WHITESPACE.value == "whitespace"
        assert OptimizationStrategy.REDUNDANCY.value == "redundancy"
        assert OptimizationStrategy.COMPRESSION.value == "compression"
        assert OptimizationStrategy.SUMMARIZATION.value == "summarization"
        assert OptimizationStrategy.DEDUPLICATION.value == "deduplication"


class TestOptimizationResult:
    """Tests for OptimizationResult dataclass."""

    @pytest.fixture
    def sample_result(self):
        """Create a sample OptimizationResult."""
        return OptimizationResult(
            original_content="This is a test content.",
            optimized_content="This is test content.",
            original_tokens=10,
            optimized_tokens=8,
            reduction_ratio=0.20,
            strategies_applied=["whitespace"],
            quality_score=0.95,
            processing_time_ms=25.5,
            metadata={"model": "gpt-4"},
        )

    def test_tokens_saved_property(self, sample_result):
        """Test tokens_saved calculated property."""
        assert sample_result.tokens_saved == 2

    def test_reduction_percentage_property(self, sample_result):
        """Test reduction_percentage calculated property."""
        assert sample_result.reduction_percentage == 20.0

    def test_reduction_percentage_with_zero_tokens(self):
        """Test reduction_percentage with zero original tokens."""
        result = OptimizationResult(
            original_content="",
            optimized_content="",
            original_tokens=0,
            optimized_tokens=0,
            reduction_ratio=0.0,
            strategies_applied=[],
            quality_score=1.0,
            processing_time_ms=0.0,
            metadata={},
        )
        assert result.reduction_percentage == 0.0

    def test_to_dict(self, sample_result):
        """Test to_dict method."""
        result_dict = sample_result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["original_tokens"] == 10
        assert result_dict["optimized_tokens"] == 8
        assert result_dict["tokens_saved"] == 2
        assert result_dict["reduction_ratio"] == 0.20
        assert result_dict["reduction_percentage"] == 20.0
        assert result_dict["strategies_applied"] == ["whitespace"]
        assert result_dict["quality_score"] == 0.95
        assert result_dict["processing_time_ms"] == 25.5
        assert result_dict["metadata"] == {"model": "gpt-4"}


class TestTokenOptimizer:
    """Tests for TokenOptimizer class."""

    @pytest.fixture
    def optimizer(self):
        """Create a TokenOptimizer instance with default settings."""
        with patch("src.token_optimization.optimizer.get_token_counter") as mock_counter:
            mock_counter_instance = Mock()
            mock_counter_instance.count_tokens.return_value = 10
            mock_counter.return_value = mock_counter_instance
            return TokenOptimizer()

    @pytest.fixture
    def mock_counter(self):
        """Create a mock token counter."""
        counter = Mock()
        counter.count_tokens.side_effect = lambda content, model: len(content.split())
        return counter

    def test_initialization_defaults(self):
        """Test TokenOptimizer initializes with default settings."""
        with patch("src.token_optimization.optimizer.get_token_counter"):
            optimizer = TokenOptimizer()

            assert optimizer.quality_threshold == 0.90
            assert optimizer.preserve_code_blocks is True
            assert optimizer.preserve_formatting is True

    def test_initialization_custom_settings(self):
        """Test TokenOptimizer initializes with custom settings."""
        with patch("src.token_optimization.optimizer.get_token_counter"):
            optimizer = TokenOptimizer(
                quality_threshold=0.85,
                preserve_code_blocks=False,
                preserve_formatting=False,
            )

            assert optimizer.quality_threshold == 0.85
            assert optimizer.preserve_code_blocks is False
            assert optimizer.preserve_formatting is False

    def test_optimize_whitespace_basic(self, optimizer):
        """Test basic whitespace optimization."""
        content = "This  is    a   test\n\n\n\nwith    extra     spaces."
        result = optimizer._optimize_whitespace(content)

        # Should reduce multiple spaces to single spaces
        assert "  " not in result
        # Should reduce multiple newlines to max 2
        assert "\n\n\n" not in result

    def test_optimize_whitespace_preserves_single_spaces(self, optimizer):
        """Test that single spaces are preserved."""
        content = "This is normal text."
        result = optimizer._optimize_whitespace(content)

        assert result == content.strip()

    def test_optimize_whitespace_removes_trailing_spaces(self, optimizer):
        """Test that trailing spaces are removed."""
        content = "Line one   \nLine two   \nLine three   "
        result = optimizer._optimize_whitespace(content)

        for line in result.split("\n"):
            assert not line.endswith(" ")

    def test_optimize_whitespace_converts_tabs(self, optimizer):
        """Test that tabs are converted to single spaces."""
        content = "Tab\t\tseparated\t\t\ttext"
        result = optimizer._optimize_whitespace(content)

        assert "\t" not in result
        assert "  " not in result  # No double spaces

    def test_remove_redundancy_common_phrases(self, optimizer):
        """Test removal of common redundant phrases."""
        test_cases = [
            ("Please note that this is important.", "this is important."),
            ("It is important to note that we must act.", "we must act."),
            ("This is basically correct.", "This is correct."),
            ("In order to proceed, click here.", "to proceed, click here."),
            ("Due to the fact that it rained.", "because it rained."),
        ]

        for input_text, expected_substring in test_cases:
            result = optimizer._remove_redundancy(input_text)
            # Check that redundant phrase was removed or replaced
            assert len(result) < len(input_text) or expected_substring in result.lower()

    def test_remove_redundancy_repeated_words(self, optimizer):
        """Test removal of repeated words."""
        content = "The the quick brown fox fox jumps."
        result = optimizer._remove_redundancy(content)

        # Should remove duplicate consecutive words
        assert "the the" not in result.lower()
        assert "fox fox" not in result.lower()

    def test_compress_content_verbose_phrases(self, optimizer):
        """Test compression of verbose phrases."""
        test_cases = [
            "will be able to",
            "is going to",
            "has the ability to",
            "in spite of the fact that",
            "with regard to",
        ]

        for phrase in test_cases:
            content = f"We {phrase} complete this task."
            result = optimizer._compress_content(content)

            # Should be shorter or have replaced phrase
            assert len(result) <= len(content)

    def test_deduplicate_content_removes_duplicates(self, optimizer):
        """Test deduplication removes duplicate paragraphs."""
        content = """First paragraph.

Second paragraph.

First paragraph.

Third paragraph.

Second paragraph."""

        result = optimizer._deduplicate_content(content)

        # Should have only unique paragraphs
        paragraphs = result.split("\n\n")
        unique_paragraphs = [p.strip() for p in paragraphs if p.strip()]

        assert len(unique_paragraphs) == 3
        assert unique_paragraphs.count("First paragraph.") == 1
        assert unique_paragraphs.count("Second paragraph.") == 1
        assert unique_paragraphs.count("Third paragraph.") == 1

    def test_deduplicate_content_preserves_order(self, optimizer):
        """Test deduplication preserves first occurrence order."""
        content = "A\n\nB\n\nC\n\nB\n\nA"
        result = optimizer._deduplicate_content(content)

        paragraphs = [p.strip() for p in result.split("\n\n") if p.strip()]

        assert paragraphs == ["A", "B", "C"]

    def test_extract_code_blocks_markdown(self, optimizer):
        """Test extraction of markdown code blocks."""
        content = """Some text before.

```python
def hello():
    print("world")
```

Some text after.

```javascript
console.log("test");
```

End."""

        code_blocks = optimizer._extract_code_blocks(content)

        assert len(code_blocks) == 2
        assert "python" in code_blocks[0]
        assert "javascript" in code_blocks[1]

    def test_extract_code_blocks_empty_content(self, optimizer):
        """Test code block extraction with no code blocks."""
        content = "Just regular text without any code blocks."
        code_blocks = optimizer._extract_code_blocks(content)

        assert len(code_blocks) == 0

    def test_estimate_quality_identical_content(self, optimizer):
        """Test quality estimation with identical content."""
        content = "This is test content."
        result = optimizer._estimate_quality(content, content)

        assert result == 1.0

    def test_estimate_quality_empty_original(self, optimizer):
        """Test quality estimation with empty original."""
        result = optimizer._estimate_quality("", "optimized")

        assert result == 1.0

    def test_estimate_quality_significant_reduction(self, optimizer):
        """Test quality estimation with significant reduction."""
        original = "This is a very long piece of content with many words."
        optimized = "Short."

        result = optimizer._estimate_quality(original, optimized)

        # Should have lower quality due to significant reduction
        assert result < 1.0

    @patch("src.token_optimization.optimizer.get_token_counter")
    def test_optimize_basic(self, mock_get_counter):
        """Test basic optimization with default strategies."""
        mock_counter = Mock()
        mock_counter.count_tokens.side_effect = [20, 16]  # Original, then optimized
        mock_get_counter.return_value = mock_counter

        optimizer = TokenOptimizer()
        content = "This  is   a  test   with    extra     spaces."

        result = optimizer.optimize(content, model="gpt-4")

        assert isinstance(result, OptimizationResult)
        assert result.original_tokens == 20
        assert result.optimized_tokens == 16
        assert result.tokens_saved == 4
        assert result.reduction_ratio > 0
        assert len(result.strategies_applied) > 0

    @patch("src.token_optimization.optimizer.get_token_counter")
    def test_optimize_with_specific_strategies(self, mock_get_counter):
        """Test optimization with specific strategies."""
        mock_counter = Mock()
        mock_counter.count_tokens.side_effect = [10, 9]
        mock_get_counter.return_value = mock_counter

        optimizer = TokenOptimizer()
        strategies = [OptimizationStrategy.WHITESPACE, OptimizationStrategy.REDUNDANCY]

        result = optimizer.optimize(
            "Test content",
            strategies=strategies,
            model="gpt-4",
        )

        assert result.strategies_applied == strategies

    @patch("src.token_optimization.optimizer.get_token_counter")
    def test_optimize_aggressive_mode(self, mock_get_counter):
        """Test optimization in aggressive mode applies more strategies."""
        mock_counter = Mock()
        mock_counter.count_tokens.side_effect = [20, 15]
        mock_get_counter.return_value = mock_counter

        optimizer = TokenOptimizer()

        result = optimizer.optimize(
            "Test content with redundancy",
            aggressive=True,
            model="gpt-4",
        )

        # Aggressive mode should apply more strategies
        assert len(result.strategies_applied) >= 2

    @patch("src.token_optimization.optimizer.get_token_counter")
    def test_optimize_preserves_code_blocks(self, mock_get_counter):
        """Test that code blocks are preserved during optimization."""
        mock_counter = Mock()
        mock_counter.count_tokens.side_effect = [30, 28]
        mock_get_counter.return_value = mock_counter

        optimizer = TokenOptimizer(preserve_code_blocks=True)
        content = """Text with  extra   spaces.

```python
def test():
    pass
```

More text."""

        result = optimizer.optimize(content, model="gpt-4")

        # Code block should be preserved
        assert "```python" in result.optimized_content
        assert "def test():" in result.optimized_content

    @patch("src.token_optimization.optimizer.get_token_counter")
    def test_optimize_processing_time_recorded(self, mock_get_counter):
        """Test that processing time is recorded."""
        mock_counter = Mock()
        mock_counter.count_tokens.side_effect = [10, 9]
        mock_get_counter.return_value = mock_counter

        optimizer = TokenOptimizer()
        result = optimizer.optimize("Test content", model="gpt-4")

        assert result.processing_time_ms > 0
        assert isinstance(result.processing_time_ms, float)

    @patch("src.token_optimization.optimizer.get_token_counter")
    def test_optimize_empty_content(self, mock_get_counter):
        """Test optimization of empty content."""
        mock_counter = Mock()
        mock_counter.count_tokens.return_value = 0
        mock_get_counter.return_value = mock_counter

        optimizer = TokenOptimizer()
        result = optimizer.optimize("", model="gpt-4")

        assert result.original_tokens == 0
        assert result.optimized_tokens == 0
        assert result.reduction_ratio == 0.0

    @patch("src.token_optimization.optimizer.get_token_counter")
    def test_optimize_handles_strategy_errors(self, mock_get_counter, caplog):
        """Test that strategy errors are handled gracefully."""
        mock_counter = Mock()
        mock_counter.count_tokens.side_effect = [10, 10]
        mock_get_counter.return_value = mock_counter

        optimizer = TokenOptimizer()

        # Mock a strategy to raise an error
        with patch.object(optimizer, "_optimize_whitespace", side_effect=Exception("Strategy error")):
            result = optimizer.optimize(
                "Test content",
                strategies=[OptimizationStrategy.WHITESPACE],
                model="gpt-4",
            )

            # Should complete without raising
            assert isinstance(result, OptimizationResult)
            # Strategy should not be in applied list
            assert OptimizationStrategy.WHITESPACE not in result.strategies_applied

    def test_analyze_efficiency_basic(self, optimizer):
        """Test basic efficiency analysis."""
        with patch.object(optimizer.counter, "count_tokens", return_value=50):
            content = "Test content with   extra spaces and redundancy."
            result = optimizer.analyze_efficiency(content, "gpt-4")

            assert "current_tokens" in result
            assert "potential_savings" in result
            assert "total_potential_savings" in result
            assert "potential_reduction_percentage" in result
            assert "suggestions" in result

            assert result["current_tokens"] == 50
            assert isinstance(result["suggestions"], list)

    def test_analyze_efficiency_identifies_whitespace(self, optimizer):
        """Test efficiency analysis identifies whitespace opportunities."""
        with patch.object(optimizer.counter, "count_tokens") as mock_count:
            mock_count.side_effect = [50, 5]  # Total tokens, whitespace tokens

            content = "Content     with     lots     of     spaces."
            result = optimizer.analyze_efficiency(content, "gpt-4")

            # Should identify whitespace savings
            assert result["potential_savings"]["whitespace"] > 0

    def test_analyze_efficiency_suggestions_generated(self, optimizer):
        """Test that suggestions are generated based on analysis."""
        with patch.object(optimizer.counter, "count_tokens", return_value=100):
            content = "A" * 1000
            result = optimizer.analyze_efficiency(content, "gpt-4")

            suggestions = result["suggestions"]
            assert len(suggestions) > 0
            assert all(isinstance(s, str) for s in suggestions)

    def test_analyze_efficiency_well_optimized_content(self, optimizer):
        """Test analysis of already well-optimized content."""
        with patch.object(optimizer.counter, "count_tokens") as mock_count:
            mock_count.side_effect = [50, 1, 1, 1]  # Low potential savings

            content = "Well optimized content."
            result = optimizer.analyze_efficiency(content, "gpt-4")

            # Should recognize content is well-optimized
            assert "already well-optimized" in " ".join(result["suggestions"]).lower()


class TestSingletonFunction:
    """Tests for singleton helper function."""

    @pytest.fixture(autouse=True)
    def reset_optimizer_singleton(self):
        """Reset optimizer singleton before each test."""
        import src.token_optimization.optimizer as opt_module
        opt_module._optimizer_instance = None
        yield
        opt_module._optimizer_instance = None

    def test_get_optimizer_returns_instance(self):
        """Test get_optimizer returns TokenOptimizer instance."""
        with patch("src.token_optimization.optimizer.get_token_counter"):
            optimizer = get_optimizer()

            assert isinstance(optimizer, TokenOptimizer)

    def test_get_optimizer_with_custom_settings(self):
        """Test get_optimizer accepts custom settings."""
        with patch("src.token_optimization.optimizer.get_token_counter"):
            optimizer = get_optimizer(
                quality_threshold=0.85,
                preserve_code_blocks=False,
            )

            assert optimizer.quality_threshold == 0.85
            assert optimizer.preserve_code_blocks is False


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    @pytest.fixture
    def optimizer(self):
        """Create a TokenOptimizer instance."""
        with patch("src.token_optimization.optimizer.get_token_counter"):
            return TokenOptimizer()

    def test_optimize_unicode_content(self, optimizer):
        """Test optimization handles Unicode correctly."""
        with patch.object(optimizer.counter, "count_tokens", side_effect=[20, 18]):
            content = "Hello 世界! 🌍 Тест тэкст"
            result = optimizer.optimize(content, model="gpt-4")

            assert isinstance(result, OptimizationResult)
            assert result.optimized_content  # Should have content

    def test_optimize_very_long_content(self, optimizer):
        """Test optimization of very long content."""
        with patch.object(optimizer.counter, "count_tokens", side_effect=[10000, 9000]):
            content = "A" * 100000  # 100k characters
            result = optimizer.optimize(content, model="gpt-4")

            assert isinstance(result, OptimizationResult)
            assert result.processing_time_ms > 0

    def test_optimize_content_with_newlines_only(self, optimizer):
        """Test optimization of content with only newlines."""
        with patch.object(optimizer.counter, "count_tokens", side_effect=[5, 1]):
            content = "\n\n\n\n\n"
            result = optimizer.optimize(content, model="gpt-4")

            assert len(result.optimized_content) < len(content)

    def test_quality_estimation_edge_cases(self, optimizer):
        """Test quality estimation with edge cases."""
        # Both empty
        assert optimizer._estimate_quality("", "") == 1.0

        # Original empty, optimized has content
        assert optimizer._estimate_quality("", "content") == 1.0

        # Very different lengths
        original = "A" * 1000
        optimized = "B"
        quality = optimizer._estimate_quality(original, optimized)
        assert 0.0 <= quality <= 1.0


class TestPerformance:
    """Performance-related tests."""

    @pytest.fixture
    def optimizer(self):
        """Create a TokenOptimizer instance."""
        with patch("src.token_optimization.optimizer.get_token_counter"):
            return TokenOptimizer()

    def test_optimization_completes_quickly(self, optimizer):
        """Test that optimization completes in reasonable time."""
        with patch.object(optimizer.counter, "count_tokens", side_effect=[100, 90]):
            content = "Test content " * 100
            start = time.perf_counter()

            result = optimizer.optimize(content, model="gpt-4")

            duration_ms = (time.perf_counter() - start) * 1000

            # Should complete quickly (allowing some overhead for mocking)
            assert duration_ms < 1000  # Less than 1 second
            assert result.processing_time_ms < 1000

    def test_multiple_optimizations_independent(self, optimizer):
        """Test that multiple optimizations don't interfere."""
        with patch.object(optimizer.counter, "count_tokens", side_effect=[10, 9, 20, 18]):
            result1 = optimizer.optimize("Content 1", model="gpt-4")
            result2 = optimizer.optimize("Content 2", model="gpt-4")

            # Results should be independent
            assert result1.original_tokens != result2.original_tokens
            assert result1.optimized_content != result2.optimized_content
