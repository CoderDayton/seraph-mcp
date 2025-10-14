"""
Gemini Provider Implementation

Provides integration with Google's Gemini API.
Uses Models.dev API for dynamic model information and pricing.

Per SDD.md:
- Minimal, clean implementation
- Typed with Pydantic
- Async-first design
- Comprehensive error handling
- Dynamic model loading from Models.dev
"""

import asyncio
import logging
import time

try:
    from google import genai
    from google.genai.types import GenerateContentConfig

    GEMINI_AVAILABLE = True
except ImportError:
    genai = None  # type: ignore[assignment]
    GenerateContentConfig = None  # type: ignore[assignment, misc]
    GEMINI_AVAILABLE = False

from ..errors import DependencyError, ProviderError, ProviderRateLimitError, ProviderTimeoutError
from .base import (
    BaseProvider,
    CompletionRequest,
    CompletionResponse,
    ProviderConfig,
)

logger = logging.getLogger(__name__)


class GeminiProvider(BaseProvider):
    """Google Gemini API provider implementation with dynamic model loading."""

    def __init__(self, config: ProviderConfig) -> None:
        """
        Initialize Gemini provider.

        Args:
            config: Provider configuration

        Raises:
            DependencyError: If Gemini SDK is not installed
            ValueError: If configuration is invalid
        """
        super().__init__(config)

        if not GEMINI_AVAILABLE or genai is None:
            logger.error("Google Gemini SDK is not installed", extra={"package": "google-genai", "provider": "gemini"})
            raise DependencyError(
                package="google-genai",
                feature="Gemini provider",
                install_hint="pip install 'google-genai>=0.2.0'",
                details={"provider": "gemini"},
            )

        try:
            # Create async client
            self.client = genai.Client(api_key=config.api_key)

            logger.info(
                "Gemini provider initialized",
                extra={
                    "provider": "gemini",
                    "timeout": config.timeout,
                    "max_retries": config.max_retries,
                },
            )
        except Exception as e:
            logger.error(
                f"Failed to initialize Gemini client: {e}",
                extra={"provider": "gemini", "error": str(e)},
                exc_info=True,
            )
            raise RuntimeError(f"Failed to initialize Gemini provider: {e}") from e

    def _validate_config(self) -> None:
        """
        Validate Gemini-specific configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        if not self.config.api_key:
            logger.error("Gemini API key is missing", extra={"provider": "gemini"})
            raise ValueError("Gemini API key is required. Set GEMINI_API_KEY environment variable.")

        if len(self.config.api_key) < 20:
            logger.error(
                "Gemini API key has invalid format",
                extra={"provider": "gemini", "key_length": len(self.config.api_key)},
            )
            raise ValueError("Invalid Gemini API key format. Gemini keys should be at least 20 characters.")

    @property
    def name(self) -> str:
        """Return provider name."""
        return "gemini"

    @property
    def models_dev_provider_id(self) -> str | None:
        """Return Models.dev provider ID."""
        return "google"

    def _convert_messages_to_gemini_format(
        self, messages: list[dict[str, str]]
    ) -> tuple[str | None, list[dict[str, str | list[str]]]]:
        """
        Convert standard message format to Gemini format.

        Returns:
            Tuple of (system_instruction, converted_messages)
        """
        system_instruction: str | None = None
        converted: list[dict[str, str | list[str]]] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                # Gemini handles system messages separately
                system_instruction = content
            elif role == "assistant":
                # Gemini uses "model" instead of "assistant"
                converted.append({"role": "model", "parts": [content]})
            else:
                # "user" role
                converted.append({"role": "user", "parts": [content]})

        return system_instruction, converted

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation for Gemini (1 token ≈ 4 chars)."""
        return len(text) // 4

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """
        Generate completion using Gemini API.

        Args:
            request: Completion request parameters

        Returns:
            CompletionResponse with generated content and metadata

        Raises:
            RuntimeError: If provider is disabled or GenerateContentConfig unavailable
            ProviderError: If API call fails
            ProviderTimeoutError: If request times out
        """
        if not self.config.enabled:
            logger.warning(
                f"Attempted to use disabled provider: {self.name}",
                extra={"provider": self.name, "model": request.model},
            )
            raise RuntimeError(f"Provider {self.name} is disabled")

        start_time = time.time()

        try:
            # Convert messages
            system_instruction, converted_messages = self._convert_messages_to_gemini_format(request.messages)

            # Prepare generation config
            if GenerateContentConfig is None:
                logger.error("GenerateContentConfig not available", extra={"provider": self.name})
                raise RuntimeError("GenerateContentConfig not available - check google-genai installation")
            config = GenerateContentConfig(
                temperature=request.temperature,
                max_output_tokens=request.max_tokens or 8192,
                system_instruction=system_instruction,
            )

            # Build contents from messages
            contents: list[dict[str, str | list[str]]] = []
            for msg in converted_messages:
                contents.append(
                    {
                        "role": msg["role"],
                        "parts": msg["parts"],
                    }
                )

            logger.debug(
                f"Calling Gemini API with model {request.model}",
                extra={
                    "provider": self.name,
                    "model": request.model,
                    "temperature": request.temperature,
                    "max_tokens": config.max_output_tokens,
                    "message_count": len(converted_messages),
                    "has_system": bool(system_instruction),
                },
            )

            # Generate content
            response = await self.client.aio.models.generate_content(
                model=request.model,
                contents=contents,  # type: ignore[arg-type]
                config=config,
            )

            # Extract response data
            content = response.text if hasattr(response, "text") else ""

            # Estimate tokens (Gemini doesn't always provide exact counts)
            prompt_text = " ".join([msg.get("content", "") for msg in request.messages])
            prompt_tokens = self._estimate_tokens(prompt_text) if prompt_text else 0
            completion_tokens = self._estimate_tokens(content) if content else 0

            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }

            # Determine finish reason
            finish_reason = "stop"
            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if (
                    hasattr(candidate, "finish_reason")
                    and candidate.finish_reason is not None
                    and hasattr(candidate.finish_reason, "name")
                ):
                    finish_reason = str(candidate.finish_reason.name).lower()

            cost = await self.estimate_cost(
                request.model,
                usage["prompt_tokens"],
                usage["completion_tokens"],
            )

            latency_ms = (time.time() - start_time) * 1000

            logger.info(
                "Gemini completion successful",
                extra={
                    "provider": self.name,
                    "model": request.model,
                    "latency_ms": latency_ms,
                    "total_tokens": usage["total_tokens"],
                    "finish_reason": finish_reason,
                    "cost_usd": cost,
                },
            )

            return CompletionResponse(
                content=content or "",
                model=request.model,
                usage=usage,
                finish_reason=finish_reason,
                provider=self.name,
                latency_ms=latency_ms,
                cost_usd=cost,
            )

        except TimeoutError as e:
            logger.error(
                f"Gemini API request timed out after {self.config.timeout}s",
                extra={
                    "provider": self.name,
                    "model": request.model,
                    "timeout": self.config.timeout,
                },
                exc_info=True,
            )
            raise ProviderTimeoutError(self.name, self.config.timeout) from e

        except Exception as e:
            # Check for common error types
            error_str = str(e).lower()

            if "authentication" in error_str or "api key" in error_str or "unauthorized" in error_str:
                logger.error(
                    f"Gemini authentication failed: {e}",
                    extra={"provider": self.name, "error": str(e)},
                    exc_info=True,
                )
                raise ProviderError(
                    "Gemini authentication failed. Check your API key.",
                    details={"provider": self.name, "error": str(e)},
                ) from e

            elif "rate limit" in error_str or "quota" in error_str:
                logger.warning(
                    "Gemini rate limit exceeded",
                    extra={"provider": self.name, "model": request.model},
                )
                raise ProviderRateLimitError(self.name, None) from e

            else:
                logger.error(
                    f"Error calling Gemini API: {e}",
                    extra={
                        "provider": self.name,
                        "model": request.model,
                        "error": str(e),
                    },
                    exc_info=True,
                )
                raise ProviderError(
                    f"Error calling Gemini API: {e}",
                    details={"provider": self.name, "model": request.model, "error": str(e)},
                ) from e

    async def health_check(self) -> bool:
        """
        Check if Gemini API is accessible.

        Returns:
            True if API is accessible, False otherwise
        """
        if not GEMINI_AVAILABLE or GenerateContentConfig is None:
            logger.warning("Gemini SDK not available for health check", extra={"provider": self.name})
            return False

        try:
            # Get available models from models.dev
            from .models_dev import get_models_dev_client

            models_api = get_models_dev_client()
            available_models = await models_api.get_models_by_provider_type(provider_type="google")
            fallback_model = "gemini-flash-latest"

            # Use first available model or fallback to gemini-pro
            if available_models:
                models_list = list(available_models.values())
                if models_list:
                    test_model = models_list[0].name
                else:
                    test_model = fallback_model
            else:
                test_model = fallback_model
            # Try a minimal generation
            config = GenerateContentConfig(max_output_tokens=1)
            response = await asyncio.wait_for(
                self.client.aio.models.generate_content(
                    model=test_model,
                    contents="Hi",
                    config=config,
                ),
                timeout=5.0,
            )
            # Check if response is valid
            result = bool(response and hasattr(response, "text"))
            logger.debug(
                f"Gemini health check {'passed' if result else 'failed'}",
                extra={"provider": self.name, "test_model": test_model},
            )
            return result
        except TimeoutError:
            logger.warning("Gemini health check timed out", extra={"provider": self.name, "timeout": 5.0})
            return False
        except Exception as e:
            logger.warning(f"Gemini health check failed: {e}", extra={"provider": self.name, "error": str(e)})
            return False

    async def close(self) -> None:
        """Clean up Gemini client resources."""
        try:
            # Gemini SDK doesn't require explicit cleanup
            logger.debug("Closed Gemini client", extra={"provider": self.name})
        except Exception as e:
            logger.warning(f"Error closing Gemini client: {e}", extra={"provider": self.name, "error": str(e)})
