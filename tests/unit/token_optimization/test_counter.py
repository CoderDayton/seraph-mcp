"""
Unit Tests for Token Counter

Tests the TokenCounter class with multi-provider token counting.
Covers OpenAI, Anthropic, Google, and Mistral models with mocking.
"""

from unittest.mock import Mock, patch

import pytest

from src.token_optimization.counter import (
    ModelProvider,
    TokenCounter,
    count_tokens,
    get_token_counter,
)


class TestModelProvider:
    """Tests for ModelProvider enum."""

    def test_provider_values(self):
        """Test that all provider values are strings."""
        assert ModelProvider.OPENAI.value == "openai"
        assert ModelProvider.ANTHROPIC.value == "anthropic"
        assert ModelProvider.GOOGLE.value == "google"
        assert ModelProvider.MISTRAL.value == "mistral"
        assert ModelProvider.UNKNOWN.value == "unknown"


class TestTokenCounter:
    """Tests for TokenCounter class."""

    @pytest.fixture
    def counter(self):
        """Create a TokenCounter instance."""
        return TokenCounter()

    def test_initialization(self, counter):
        """Test TokenCounter initializes correctly."""
        assert counter is not None
        assert counter._encoding_cache == {}

    def test_count_tokens_empty_string(self, counter):
        """Test counting tokens in empty string."""
        result = counter.count_tokens("", "gpt-4")
        assert result == 0

    def test_get_provider_openai(self, counter):
        """Test provider detection for OpenAI models."""
        assert counter._get_provider("gpt-4") == ModelProvider.OPENAI
        assert counter._get_provider("gpt-4-turbo") == ModelProvider.OPENAI
        assert counter._get_provider("gpt-4o") == ModelProvider.OPENAI
        assert counter._get_provider("gpt-4o-mini") == ModelProvider.OPENAI
        assert counter._get_provider("gpt-3.5-turbo") == ModelProvider.OPENAI
        assert counter._get_provider("text-embedding-ada-002") == ModelProvider.OPENAI

    def test_get_provider_anthropic(self, counter):
        """Test provider detection for Anthropic models."""
        assert counter._get_provider("claude-3-opus") == ModelProvider.ANTHROPIC
        assert counter._get_provider("claude-3-sonnet") == ModelProvider.ANTHROPIC
        assert counter._get_provider("claude-3-haiku") == ModelProvider.ANTHROPIC
        assert counter._get_provider("claude-3-5-sonnet") == ModelProvider.ANTHROPIC
        assert counter._get_provider("claude-2.1") == ModelProvider.ANTHROPIC

    def test_get_provider_google(self, counter):
        """Test provider detection for Google models."""
        assert counter._get_provider("gemini-pro") == ModelProvider.GOOGLE
        assert counter._get_provider("gemini-1.5-pro") == ModelProvider.GOOGLE
        assert counter._get_provider("gemini-1.5-flash") == ModelProvider.GOOGLE

    def test_get_provider_mistral(self, counter):
        """Test provider detection for Mistral models."""
        assert counter._get_provider("mistral-large") == ModelProvider.MISTRAL
        assert counter._get_provider("mistral-medium") == ModelProvider.MISTRAL
        assert counter._get_provider("mistral-small") == ModelProvider.MISTRAL

    def test_get_provider_unknown(self, counter):
        """Test provider detection for unknown models."""
        assert counter._get_provider("unknown-model") == ModelProvider.UNKNOWN

    def test_get_provider_partial_match(self, counter):
        """Test provider detection with partial model name match."""
        # Should match "gpt-4" prefix
        assert counter._get_provider("gpt-4-0125-preview") == ModelProvider.OPENAI
        # Should match "claude-3-opus" prefix
        assert counter._get_provider("claude-3-opus-20240229") == ModelProvider.ANTHROPIC

    @patch("src.token_optimization.counter.tiktoken")
    def test_count_openai_tokens_with_tiktoken(self, mock_tiktoken, counter):
        """Test OpenAI token counting with tiktoken available."""
        # Setup mock encoding
        mock_encoding = Mock()
        mock_encoding.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens
        mock_tiktoken.encoding_for_model.return_value = mock_encoding

        result = counter._count_openai_tokens("Hello, world!", "gpt-4")

        assert result == 5
        mock_tiktoken.encoding_for_model.assert_called_once_with("gpt-4")
        mock_encoding.encode.assert_called_once_with("Hello, world!")

    @patch("src.token_optimization.counter.tiktoken")
    def test_count_openai_tokens_caching(self, mock_tiktoken, counter):
        """Test that encodings are cached for reuse."""
        mock_encoding = Mock()
        mock_encoding.encode.return_value = [1, 2, 3]
        mock_tiktoken.encoding_for_model.return_value = mock_encoding

        # First call
        counter._count_openai_tokens("First", "gpt-4")
        # Second call
        counter._count_openai_tokens("Second", "gpt-4")

        # Should only create encoding once
        assert mock_tiktoken.encoding_for_model.call_count == 1
        # Should encode twice
        assert mock_encoding.encode.call_count == 2

    @patch("src.token_optimization.counter.tiktoken")
    def test_count_openai_tokens_fallback_to_cl100k(self, mock_tiktoken, counter):
        """Test fallback to cl100k_base encoding for unknown models."""
        mock_encoding = Mock()
        mock_encoding.encode.return_value = [1, 2, 3, 4]

        # Raise KeyError for unknown model
        mock_tiktoken.encoding_for_model.side_effect = KeyError("Unknown model")
        mock_tiktoken.get_encoding.return_value = mock_encoding

        result = counter._count_openai_tokens("Test", "gpt-5-new-model")

        assert result == 4
        mock_tiktoken.get_encoding.assert_called_once_with("cl100k_base")

    def test_count_openai_tokens_without_tiktoken(self, counter):
        """Test OpenAI token counting falls back to estimation without tiktoken."""
        with patch("src.token_optimization.counter.tiktoken", None):
            counter_no_tiktoken = TokenCounter()
            result = counter_no_tiktoken._count_openai_tokens("Hello world!", "gpt-4")

            # Should fall back to estimation (12 chars / 4 = 3 tokens)
            assert result == 3

    @patch("src.token_optimization.counter.tiktoken")
    def test_count_anthropic_tokens(self, mock_tiktoken, counter):
        """Test Anthropic token counting uses cl100k_base approximation."""
        mock_encoding = Mock()
        mock_encoding.encode.return_value = [1, 2, 3, 4, 5, 6]
        mock_tiktoken.get_encoding.return_value = mock_encoding

        result = counter._count_anthropic_tokens("Test content", "claude-3-opus")

        assert result == 6
        mock_tiktoken.get_encoding.assert_called_once_with("cl100k_base")

    def test_count_anthropic_tokens_without_libraries(self, counter):
        """Test Anthropic counting falls back to estimation without libraries."""
        with patch("src.token_optimization.counter.tiktoken", None):
            with patch("src.token_optimization.counter.Anthropic", None):
                counter_no_libs = TokenCounter()
                result = counter_no_libs._count_anthropic_tokens("Hello!", "claude-3-opus")

                # Should estimate (6 chars / 4 = 1.5, rounded to 1)
                assert result >= 1

    def test_estimate_tokens(self, counter):
        """Test token estimation heuristic."""
        # Empty string
        assert counter._estimate_tokens("") == 1  # Minimum 1 token

        # Short string (4 chars)
        assert counter._estimate_tokens("Test") == 1

        # Medium string (20 chars)
        assert counter._estimate_tokens("This is a test text.") == 5

        # Long string (100 chars)
        assert counter._estimate_tokens("a" * 100) == 25

    @patch("src.token_optimization.counter.tiktoken")
    def test_count_tokens_openai_model(self, mock_tiktoken, counter):
        """Test count_tokens with OpenAI model."""
        mock_encoding = Mock()
        mock_encoding.encode.return_value = [1, 2, 3, 4]
        mock_tiktoken.encoding_for_model.return_value = mock_encoding

        result = counter.count_tokens("Hello OpenAI", "gpt-4")

        assert result == 4

    @patch("src.token_optimization.counter.tiktoken")
    def test_count_tokens_anthropic_model(self, mock_tiktoken, counter):
        """Test count_tokens with Anthropic model."""
        mock_encoding = Mock()
        mock_encoding.encode.return_value = [1, 2, 3]
        mock_tiktoken.get_encoding.return_value = mock_encoding

        result = counter.count_tokens("Hello Claude", "claude-3-opus")

        assert result == 3

    def test_count_tokens_unknown_model(self, counter):
        """Test count_tokens with unknown model falls back to estimation."""
        result = counter.count_tokens("Hello world!", "unknown-model")

        # Should estimate (12 chars / 4 = 3 tokens)
        assert result == 3

    @patch("src.token_optimization.counter.tiktoken")
    def test_get_token_breakdown(self, mock_tiktoken, counter):
        """Test get_token_breakdown returns detailed metadata."""
        mock_encoding = Mock()
        mock_encoding.encode.return_value = [1, 2, 3, 4, 5]
        mock_tiktoken.encoding_for_model.return_value = mock_encoding

        content = "Hello, world!"
        result = counter.get_token_breakdown(content, "gpt-4")

        assert result["token_count"] == 5
        assert result["model"] == "gpt-4"
        assert result["provider"] == "openai"
        assert result["character_count"] == len(content)
        assert result["chars_per_token"] == len(content) / 5
        assert result["method"] == "exact"

    def test_get_token_breakdown_estimated(self, counter):
        """Test get_token_breakdown with estimation."""
        with patch("src.token_optimization.counter.tiktoken", None):
            counter_no_tiktoken = TokenCounter()
            result = counter_no_tiktoken.get_token_breakdown("Test", "unknown-model")

            assert "token_count" in result
            assert result["method"] == "estimated"

    @patch("src.token_optimization.counter.tiktoken")
    def test_compare_models(self, mock_tiktoken, counter):
        """Test compare_models returns token counts for multiple models."""
        mock_encoding = Mock()
        mock_encoding.encode.return_value = [1, 2, 3, 4]
        mock_tiktoken.encoding_for_model.return_value = mock_encoding
        mock_tiktoken.get_encoding.return_value = mock_encoding

        content = "Test content"
        models = ["gpt-4", "gpt-3.5-turbo", "claude-3-opus"]

        result = counter.compare_models(content, models)

        assert len(result) == 3
        assert "gpt-4" in result
        assert "gpt-3.5-turbo" in result
        assert "claude-3-opus" in result
        assert all(isinstance(count, int) for count in result.values())

    @patch("src.token_optimization.counter.tiktoken")
    def test_compare_models_default_models(self, mock_tiktoken, counter):
        """Test compare_models uses default model list."""
        mock_encoding = Mock()
        mock_encoding.encode.return_value = [1, 2, 3]
        mock_tiktoken.encoding_for_model.return_value = mock_encoding
        mock_tiktoken.get_encoding.return_value = mock_encoding

        result = counter.compare_models("Test")

        # Should use default models
        assert len(result) >= 4  # At least the default models

    def test_encoding_cache_isolation(self):
        """Test that different counter instances have separate caches."""
        counter1 = TokenCounter()
        counter2 = TokenCounter()

        with patch("src.token_optimization.counter.tiktoken") as mock_tiktoken:
            mock_encoding = Mock()
            mock_encoding.encode.return_value = [1, 2, 3]
            mock_tiktoken.encoding_for_model.return_value = mock_encoding

            counter1._count_openai_tokens("Test", "gpt-4")

            # counter2 should have empty cache
            assert len(counter2._encoding_cache) == 0
            assert len(counter1._encoding_cache) > 0


class TestSingletonFunctions:
    """Tests for singleton helper functions."""

    def test_get_token_counter_singleton(self):
        """Test get_token_counter returns singleton instance."""
        counter1 = get_token_counter()
        counter2 = get_token_counter()

        assert counter1 is counter2

    @patch("src.token_optimization.counter.tiktoken")
    def test_count_tokens_convenience_function(self, mock_tiktoken):
        """Test count_tokens convenience function."""
        mock_encoding = Mock()
        mock_encoding.encode.return_value = [1, 2, 3, 4, 5]
        mock_tiktoken.encoding_for_model.return_value = mock_encoding

        result = count_tokens("Hello!", "gpt-4")

        assert result == 5


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    @pytest.fixture
    def counter(self):
        """Create a TokenCounter instance."""
        return TokenCounter()

    @patch("src.token_optimization.counter.tiktoken")
    def test_encoding_error_falls_back_to_estimation(self, mock_tiktoken, counter):
        """Test that encoding errors fall back to estimation."""
        mock_tiktoken.encoding_for_model.side_effect = Exception("Encoding error")

        result = counter._count_openai_tokens("Test content", "gpt-4")

        # Should fall back to estimation
        assert result > 0
        assert isinstance(result, int)

    def test_count_tokens_with_unicode(self, counter):
        """Test token counting with Unicode characters."""
        content = "Hello 世界! 🌍"
        result = counter.count_tokens(content, "gpt-4")

        assert result > 0
        assert isinstance(result, int)

    def test_count_tokens_with_special_characters(self, counter):
        """Test token counting with special characters."""
        content = "Special: @#$%^&*(){}[]|\\:;\"'<>,.?/~`"
        result = counter.count_tokens(content, "gpt-4")

        assert result > 0
        assert isinstance(result, int)

    def test_count_tokens_very_long_text(self, counter):
        """Test token counting with very long text."""
        content = "A" * 100000  # 100k characters
        result = counter.count_tokens(content, "gpt-4")

        assert result > 0
        assert isinstance(result, int)

    def test_get_token_breakdown_division_by_zero(self, counter):
        """Test get_token_breakdown handles zero tokens edge case."""
        with patch.object(counter, "count_tokens", return_value=0):
            result = counter.get_token_breakdown("", "gpt-4")

            assert result["token_count"] == 0
            assert result["chars_per_token"] == 0  # Should handle division by zero


class TestPerformance:
    """Performance-related tests."""

    @pytest.fixture
    def counter(self):
        """Create a TokenCounter instance."""
        return TokenCounter()

    @patch("src.token_optimization.counter.tiktoken")
    def test_caching_improves_performance(self, mock_tiktoken, counter):
        """Test that encoding caching reduces function calls."""
        mock_encoding = Mock()
        mock_encoding.encode.return_value = [1, 2, 3]
        mock_tiktoken.encoding_for_model.return_value = mock_encoding

        # Count tokens multiple times for same model
        for _ in range(10):
            counter._count_openai_tokens("Different content each time", "gpt-4")

        # Should only create encoding once due to caching
        assert mock_tiktoken.encoding_for_model.call_count == 1
        # But should encode 10 times
        assert mock_encoding.encode.call_count == 10

    @patch("src.token_optimization.counter.tiktoken")
    def test_different_models_create_separate_cache_entries(self, mock_tiktoken, counter):
        """Test that different models get separate cache entries."""
        mock_encoding = Mock()
        mock_encoding.encode.return_value = [1, 2, 3]
        mock_tiktoken.encoding_for_model.return_value = mock_encoding

        counter._count_openai_tokens("Test", "gpt-4")
        counter._count_openai_tokens("Test", "gpt-3.5-turbo")
        counter._count_openai_tokens("Test", "gpt-4o")

        # Should create 3 different encodings
        assert mock_tiktoken.encoding_for_model.call_count == 3
        assert len(counter._encoding_cache) == 3
