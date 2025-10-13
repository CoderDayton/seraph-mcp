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

import time

try:
    from google import genai
    from google.genai.types import GenerateContentConfig

    GEMINI_AVAILABLE = True
except ImportError:
    genai = None  # type: ignore[assignment]
    GenerateContentConfig = None  # type: ignore[assignment, misc]
    GEMINI_AVAILABLE = False

from .base import (
    BaseProvider,
    CompletionRequest,
    CompletionResponse,
    ProviderConfig,
)


class GeminiProvider(BaseProvider):
    """Google Gemini API provider implementation with dynamic model loading."""

    def __init__(self, config: ProviderConfig) -> None:
        """Initialize Gemini provider."""
        super().__init__(config)

        if not GEMINI_AVAILABLE or genai is None:
            raise RuntimeError("Google Gemini SDK not available. Install with: pip install google-genai>=0.2.0")

        # Create async client
        self.client = genai.Client(api_key=config.api_key)

    def _validate_config(self) -> None:
        """Validate Gemini-specific configuration."""
        if not self.config.api_key:
            raise ValueError("Gemini API key is required")

        if len(self.config.api_key) < 20:
            raise ValueError("Invalid Gemini API key format")

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
    ) -> tuple[str | None, list[dict[str, str]]]:
        """
        Convert standard message format to Gemini format.

        Returns:
            Tuple of (system_instruction, converted_messages)
        """
        system_instruction = None
        converted = []

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
        """Generate completion using Gemini API."""
        if not self.config.enabled:
            raise RuntimeError(f"Provider {self.name} is disabled")

        start_time = time.time()

        try:
            # Convert messages
            system_instruction, converted_messages = self._convert_messages_to_gemini_format(request.messages)

            # Prepare generation config
            if GenerateContentConfig is None:
                raise RuntimeError("GenerateContentConfig not available")
            config = GenerateContentConfig(
                temperature=request.temperature,
                max_output_tokens=request.max_tokens or 8192,
                system_instruction=system_instruction,
            )

            # Build contents from messages
            contents = []
            for msg in converted_messages:
                contents.append(
                    {
                        "role": msg["role"],
                        "parts": msg["parts"],
                    }
                )

            # Generate content
            response = await self.client.aio.models.generate_content(
                model=request.model,
                contents=contents,
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

            return CompletionResponse(
                content=content or "",
                model=request.model,
                usage=usage,
                finish_reason=finish_reason,
                provider=self.name,
                latency_ms=latency_ms,
                cost_usd=cost,
            )

        except Exception as e:
            raise RuntimeError(f"Error calling Gemini API: {e}") from e

    async def health_check(self) -> bool:
        """Check if Gemini API is accessible."""
        try:
            if GenerateContentConfig is None:
                return False
            # Try a minimal generation
            config = GenerateContentConfig(max_output_tokens=1)
            await self.client.aio.models.generate_content(
                model="gemini-pro",
                contents="Hi",
                config=config,
            )
            return True
        except Exception:
            return False

    async def close(self) -> None:
        """Clean up Gemini client resources."""
        # Gemini SDK doesn't require explicit cleanup
        pass
