"""
Token Counter Module

Provides accurate token counting for multiple LLM providers.
Supports OpenAI (via tiktoken) and Anthropic (via anthropic library).
"""

import logging
from enum import Enum

try:
    import tiktoken
except ImportError:
    tiktoken = None

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

logger = logging.getLogger(__name__)


class ModelProvider(str, Enum):
    """Supported model providers for token counting."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    MISTRAL = "mistral"
    UNKNOWN = "unknown"


class TokenCounter:
    """
    Multi-provider token counter with caching.

    Provides accurate token counting for various LLM models.
    Uses provider-specific libraries when available.
    """

    # Model to provider mapping
    MODEL_PROVIDERS: dict[str, ModelProvider] = {
        # OpenAI models
        "gpt-4": ModelProvider.OPENAI,
        "gpt-4-turbo": ModelProvider.OPENAI,
        "gpt-4-turbo-preview": ModelProvider.OPENAI,
        "gpt-4o": ModelProvider.OPENAI,
        "gpt-4o-mini": ModelProvider.OPENAI,
        "gpt-3.5-turbo": ModelProvider.OPENAI,
        "gpt-3.5-turbo-16k": ModelProvider.OPENAI,
        "text-embedding-3-small": ModelProvider.OPENAI,
        "text-embedding-3-large": ModelProvider.OPENAI,
        "text-embedding-ada-002": ModelProvider.OPENAI,
        # Anthropic models
        "claude-3-opus": ModelProvider.ANTHROPIC,
        "claude-3-sonnet": ModelProvider.ANTHROPIC,
        "claude-3-haiku": ModelProvider.ANTHROPIC,
        "claude-3-5-sonnet": ModelProvider.ANTHROPIC,
        "claude-3-5-haiku": ModelProvider.ANTHROPIC,
        "claude-2.1": ModelProvider.ANTHROPIC,
        "claude-2.0": ModelProvider.ANTHROPIC,
        "claude-instant-1.2": ModelProvider.ANTHROPIC,
        # Google models
        "gemini-pro": ModelProvider.GOOGLE,
        "gemini-pro-vision": ModelProvider.GOOGLE,
        "gemini-1.5-pro": ModelProvider.GOOGLE,
        "gemini-1.5-flash": ModelProvider.GOOGLE,
        # Mistral models
        "mistral-large": ModelProvider.MISTRAL,
        "mistral-medium": ModelProvider.MISTRAL,
        "mistral-small": ModelProvider.MISTRAL,
        "mistral-tiny": ModelProvider.MISTRAL,
    }

    def __init__(self) -> None:
        """Initialize token counter."""
        # Instance-level encoding cache
        self._encoding_cache: dict[str, any] = {}
        self._validate_dependencies()

    def _validate_dependencies(self) -> None:
        """Log warnings if optional dependencies are missing."""
        if tiktoken is None:
            logger.warning("tiktoken not installed. OpenAI token counting will use estimation.")
        if Anthropic is None:
            logger.warning("anthropic not installed. Anthropic token counting will use estimation.")

    def count_tokens(self, content: str, model: str = "gpt-4") -> int:
        """
        Count tokens in content for specified model.

        Args:
            content: Text content to count tokens for
            model: Model name (e.g., "gpt-4", "claude-3-opus")

        Returns:
            Token count for the specified model

        Raises:
            ValueError: If model is not supported
        """
        if not content:
            return 0

        provider = self._get_provider(model)

        if provider == ModelProvider.OPENAI:
            return self._count_openai_tokens(content, model)
        elif provider == ModelProvider.ANTHROPIC:
            return self._count_anthropic_tokens(content, model)
        else:
            # Fallback to estimation for unsupported providers
            return self._estimate_tokens(content)

    def _get_provider(self, model: str) -> ModelProvider:
        """
        Get provider for model name.

        Args:
            model: Model name

        Returns:
            Provider enum value
        """
        # Exact match
        if model in self.MODEL_PROVIDERS:
            return self.MODEL_PROVIDERS[model]

        # Partial match (e.g., "gpt-4-0125-preview" -> "gpt-4")
        for model_prefix, provider in self.MODEL_PROVIDERS.items():
            if model.startswith(model_prefix):
                return provider

        logger.warning(f"Unknown model: {model}, using estimation")
        return ModelProvider.UNKNOWN

    def _count_openai_tokens(self, content: str, model: str) -> int:
        """
        Count tokens using tiktoken for OpenAI models.

        Args:
            content: Text content
            model: OpenAI model name

        Returns:
            Exact token count
        """
        if tiktoken is None:
            logger.warning("tiktoken not available, using estimation")
            return self._estimate_tokens(content)

        try:
            # Get or create encoding
            if model not in self._encoding_cache:
                try:
                    encoding = tiktoken.encoding_for_model(model)
                except KeyError:
                    # Fall back to cl100k_base for newer models
                    encoding = tiktoken.get_encoding("cl100k_base")
                self._encoding_cache[model] = encoding
            else:
                encoding = self._encoding_cache[model]

            # Count tokens
            tokens = encoding.encode(content)
            return len(tokens)

        except Exception as e:
            logger.error(f"Error counting OpenAI tokens: {e}")
            return self._estimate_tokens(content)

    def _count_anthropic_tokens(self, content: str, model: str) -> int:
        """
        Count tokens for Anthropic models.

        Args:
            content: Text content
            model: Anthropic model name

        Returns:
            Token count (estimated or exact)
        """
        try:
            # Anthropic uses similar tokenization to GPT-4
            # Use tiktoken with cl100k_base as approximation
            if tiktoken is not None:
                if "cl100k_base" not in self._encoding_cache:
                    self._encoding_cache["cl100k_base"] = tiktoken.get_encoding("cl100k_base")

                encoding = self._encoding_cache["cl100k_base"]
                tokens = encoding.encode(content)
                return len(tokens)
            else:
                logger.warning("tiktoken not available, using estimation for Anthropic")
                return self._estimate_tokens(content)

        except Exception as e:
            logger.error(f"Error counting Anthropic tokens: {e}")
            return self._estimate_tokens(content)

    def _estimate_tokens(self, content: str) -> int:
        """
        Estimate token count using simple heuristic.

        Rule of thumb: ~4 characters per token for English text.

        Args:
            content: Text content

        Returns:
            Estimated token count
        """
        # Simple estimation: 4 chars ≈ 1 token
        return max(1, len(content) // 4)

    def get_token_breakdown(self, content: str, model: str = "gpt-4") -> dict[str, any]:
        """
        Get detailed token breakdown with metadata.

        Args:
            content: Text content
            model: Model name

        Returns:
            Dictionary with token count and metadata
        """
        token_count = self.count_tokens(content, model)
        provider = self._get_provider(model)

        return {
            "token_count": token_count,
            "model": model,
            "provider": provider.value,
            "character_count": len(content),
            "chars_per_token": len(content) / token_count if token_count > 0 else 0,
            "method": "exact" if provider in [ModelProvider.OPENAI, ModelProvider.ANTHROPIC] else "estimated",
        }

    def compare_models(self, content: str, models: list[str] | None = None) -> dict[str, int]:
        """
        Compare token counts across multiple models.

        Args:
            content: Text content
            models: List of model names (default: common models)

        Returns:
            Dictionary mapping model names to token counts
        """
        if models is None:
            models = ["gpt-4", "gpt-3.5-turbo", "claude-3-opus", "claude-3-haiku"]

        return {model: self.count_tokens(content, model) for model in models}


# Singleton instance
_counter_instance: TokenCounter | None = None


def get_token_counter() -> TokenCounter:
    """
    Get singleton TokenCounter instance.

    Returns:
        Shared TokenCounter instance
    """
    global _counter_instance
    if _counter_instance is None:
        _counter_instance = TokenCounter()
    return _counter_instance


def count_tokens(content: str, model: str = "gpt-4") -> int:
    """
    Convenience function to count tokens.

    Args:
        content: Text content
        model: Model name

    Returns:
        Token count
    """
    counter = get_token_counter()
    return counter.count_tokens(content, model)
