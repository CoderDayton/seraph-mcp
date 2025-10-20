"""
MCP Protocol Middleware for Context Optimization
-------------------------------------------------

Compresses MCP protocol payloads (tool results, resource reads) at the protocol
layer before they're transmitted to clients. This is SEPARATE from the provider
wrapper (middleware.py) which compresses LLM prompts.

Architecture (Two-Layer Compression):
  Layer 1 (THIS FILE): MCP protocol message compression
    - Compresses: Tool call results, resource read content
    - When: Response payloads >1KB
    - Goal: Reduce client token consumption (40-50% compression)

  Layer 2 (middleware.py): LLM provider prompt compression
    - Compresses: User prompts before sending to LLM
    - When: Prompts >100 chars
    - Goal: Reduce LLM context window usage (20% compression)

Per SDD §10.4.2: Automatic optimization with separate metrics namespaces.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
import time
from collections import Counter
from typing import TYPE_CHECKING, Any

from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from fastmcp.tools.tool import ToolResult
from mcp.server.lowlevel.helper_types import ReadResourceContents

from ..observability import get_observability

if TYPE_CHECKING:
    from .config import ContextOptimizationConfig
    from .seraph_compression import CompressionResult, SeraphCompressor

logger = logging.getLogger(__name__)


class CompressionMiddleware(Middleware):
    """
    MCP protocol middleware that compresses large text responses.

    Hooks into FastMCP message processing to compress:
    - Tool call results (tools/call)
    - Resource read content (resources/read)

    Configuration:
        min_size_bytes: Only compress responses larger than this (default: 1000)
        compression_ratio: Target compression ratio 0-1 (default: 0.5 = 50% retention)
        timeout_seconds: Max time for compression (default: 10s)

    Metrics: Tracks compression stats in 'mcp.middleware.*' namespace.
    """

    def __init__(
        self,
        config: ContextOptimizationConfig,
        min_size_bytes: int = 1000,
        timeout_seconds: float = 10.0,
    ):
        """
        Initialize compression middleware with automatic quality scoring.

        Args:
            config: Context optimization config (for compression ratios)
            min_size_bytes: Minimum response size to compress (bytes)
            timeout_seconds: Max compression time before fallback

        Note:
            Compression ratio is dynamically computed per-request based on
            content characteristics (structure, entropy, code density, etc.)
        """
        super().__init__()
        self.config = config
        self.min_size_bytes = min_size_bytes
        self.timeout_seconds = timeout_seconds

        # Lazy-load compressor on first use (avoid startup overhead)
        self._compressor: SeraphCompressor | None = None
        self._compression_available = True

        # Observability
        self._obs = get_observability()

        logger.info(
            f"CompressionMiddleware initialized (min_size={min_size_bytes}B, "
            f"auto_quality=enabled, timeout={timeout_seconds}s)"
        )

    def _get_compressor(self) -> SeraphCompressor:
        """Lazy-initialize compressor on first compression attempt."""
        if self._compressor is None:
            try:
                from .seraph_compression import SeraphCompressor

                self._compressor = SeraphCompressor(
                    l1_ratio=self.config.seraph_l1_ratio,
                    l2_ratio=self.config.seraph_l2_ratio,
                    l3_ratio=self.config.seraph_l3_ratio,
                )
                logger.debug("SeraphCompressor initialized for MCP middleware")
            except Exception as e:
                logger.error(f"Failed to initialize compressor: {e}", exc_info=True)
                self._compression_available = False
                raise RuntimeError(f"Cannot initialize SeraphCompressor: {e}") from e

        return self._compressor

    async def _compress_text(self, text: str, context_label: str) -> str | None:
        """
        Compress text content using SeraphCompressor.

        Args:
            text: Content to compress
            context_label: Label for logging/metrics (e.g., "tool_result", "resource")

        Returns:
            Compressed text or None if compression failed/skipped
        """
        if not self._compression_available:
            return None

        # Check size threshold
        text_bytes = len(text.encode("utf-8"))
        if text_bytes < self.min_size_bytes:
            logger.debug(f"Skipping compression for {context_label}: {text_bytes}B < {self.min_size_bytes}B")
            return None

        start_time = time.perf_counter()

        try:
            # Get compressor (lazy-init)
            compressor = self._get_compressor()

            # Compress with timeout
            result: CompressionResult = await asyncio.wait_for(
                compressor.build(text),
                timeout=self.timeout_seconds,
            )

            # Automatically determine optimal compression ratio from content
            target_ratio = self._analyze_content_quality(text)

            # Select layer based on computed ratio
            compressed: str = result.select_layer(target_ratio=target_ratio)

            # Calculate metrics
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            original_tokens = result.original_token_count
            compressed_tokens = self._count_tokens(compressed)
            compression_pct = (
                ((original_tokens - compressed_tokens) / original_tokens * 100) if original_tokens > 0 else 0
            )

            # Track metrics
            self._obs.histogram(
                f"mcp.middleware.{context_label}.compression_time_ms",
                elapsed_ms,
            )
            self._obs.gauge(
                f"mcp.middleware.{context_label}.tokens_saved",
                original_tokens - compressed_tokens,
            )
            self._obs.gauge(
                f"mcp.middleware.{context_label}.compression_ratio",
                compression_pct / 100,
            )

            logger.info(
                f"Compressed {context_label}: {original_tokens}→{compressed_tokens} tokens "
                f"({compression_pct:.1f}% reduction, ratio={target_ratio:.2f}, {elapsed_ms:.0f}ms)"
            )

            return compressed

        except TimeoutError:
            logger.warning(f"Compression timeout for {context_label} ({self.timeout_seconds}s), using original content")
            self._obs.increment(f"mcp.middleware.{context_label}.timeout")
            return None

        except Exception as e:
            logger.error(
                f"Compression failed for {context_label}: {e}",
                exc_info=True,
            )
            self._obs.increment(f"mcp.middleware.{context_label}.error")
            return None

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text (reuses SeraphCompressor's tokenizer)."""
        try:
            from .seraph_compression import count_tokens

            token_count: int = count_tokens(text)
            return token_count
        except Exception:
            # Fallback: rough word-based estimate (1.3 tokens per word)
            return int(len(text.split()) * 1.3)

    def _analyze_content_quality(self, text: str) -> float:
        """
        Automatically score content to determine optimal compression ratio.

        Scoring dimensions (0.0-1.0 scale, higher = preserve more):
            1. Structure density: Code/JSON/tables → high preservation
            2. Information entropy: Random/compressed → high preservation
            3. Redundancy: Repeated patterns → aggressive compression
            4. Semantic density: Technical terms → moderate preservation

        Returns:
            Target compression ratio (0.3-0.85):
                0.30-0.45: Aggressive (prose, logs, verbose output)
                0.50-0.60: Balanced (mixed content)
                0.65-0.85: Conservative (code, structured data, dense info)
        """
        if not text or len(text) < 100:
            return 0.70  # Default conservative for small content

        # === 1. Structure Density Score (0-1) ===
        # Detect code, JSON, YAML, tables, formatted output
        structure_markers = sum(
            [
                text.count("{") + text.count("}"),  # JSON/objects
                text.count("[") + text.count("]"),  # Arrays/indexing
                text.count("(") + text.count(")"),  # Function calls
                text.count("\n|"),  # Markdown tables
                text.count("```"),  # Code blocks
                text.count("::"),  # Type annotations
                text.count("->"),  # Arrows (types/logic)
                text.count("=>"),  # Fat arrows
            ]
        )
        structure_score = min(1.0, structure_markers / (len(text) / 50))

        # === 2. Information Entropy (0-1) ===
        # High entropy = already compressed/random → preserve more
        char_freq = Counter(text.lower())
        total_chars = len(text)
        entropy = 0.0
        for count in char_freq.values():
            prob = count / total_chars
            entropy -= prob * math.log2(prob) if prob > 0 else 0
        # Normalize: English prose ~4.1 bits, random ~4.7 bits
        entropy_score = min(1.0, entropy / 4.7)

        # === 3. Redundancy Score (0-1, inverted) ===
        # Detect repeated lines/patterns → compress aggressively
        lines = text.split("\n")
        unique_lines = len(set(lines))
        redundancy = 1.0 - (unique_lines / max(len(lines), 1))
        redundancy_score = 1.0 - redundancy  # Invert: high uniqueness = preserve

        # === 4. Semantic Density (0-1) ===
        # Technical terms, camelCase, specific vocabulary → preserve
        words = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", text)
        if not words:
            semantic_score = 0.5
        else:
            # Count technical indicators
            camel_case = sum(1 for w in words if any(c.isupper() for c in w[1:]))
            snake_case = sum(1 for w in words if "_" in w)
            long_words = sum(1 for w in words if len(w) > 8)

            semantic_score = min(1.0, (camel_case + snake_case + long_words) / (len(words) * 0.3))

        # === Weighted Combination ===
        # Structure and entropy are most important for preservation
        weights = {
            "structure": 0.35,
            "entropy": 0.25,
            "redundancy": 0.20,
            "semantic": 0.20,
        }

        quality_score = (
            structure_score * weights["structure"]
            + entropy_score * weights["entropy"]
            + redundancy_score * weights["redundancy"]
            + semantic_score * weights["semantic"]
        )

        # === Map to Compression Ratio ===
        # quality_score ∈ [0, 1] → compression_ratio ∈ [0.30, 0.85]
        # Low quality (verbose prose) → 0.30-0.45 (aggressive)
        # High quality (structured data) → 0.65-0.85 (conservative)
        min_ratio = 0.30
        max_ratio = 0.85
        compression_ratio = min_ratio + (quality_score * (max_ratio - min_ratio))

        # Track quality breakdown for observability
        self._obs.gauge("mcp.middleware.quality.structure", structure_score)
        self._obs.gauge("mcp.middleware.quality.entropy", entropy_score)
        self._obs.gauge("mcp.middleware.quality.redundancy", redundancy_score)
        self._obs.gauge("mcp.middleware.quality.semantic", semantic_score)
        self._obs.gauge("mcp.middleware.quality.final_score", quality_score)
        self._obs.gauge("mcp.middleware.quality.compression_ratio", compression_ratio)

        logger.debug(
            f"Content quality analysis: structure={structure_score:.2f}, "
            f"entropy={entropy_score:.2f}, redundancy={redundancy_score:.2f}, "
            f"semantic={semantic_score:.2f} → ratio={compression_ratio:.2f}"
        )

        return compression_ratio

    async def on_call_tool(
        self,
        context: MiddlewareContext,
        call_next: CallNext,
    ) -> ToolResult:
        """
        Compress tool call results (tools/call).

        Intercepts tool execution results and compresses large text content
        before returning to the client.
        """
        # Execute tool normally
        result: ToolResult = await call_next(context)

        # Compress text content if present
        if hasattr(result, "content") and result.content:
            for content_item in result.content:
                if hasattr(content_item, "type") and content_item.type == "text":
                    if hasattr(content_item, "text") and content_item.text:
                        compressed = await self._compress_text(
                            content_item.text,
                            context_label="tool_result",
                        )
                        if compressed:
                            # Replace with compressed version
                            content_item.text = compressed

        return result

    async def on_read_resource(
        self,
        context: MiddlewareContext,
        call_next: CallNext,
    ) -> list[ReadResourceContents]:
        """
        Compress resource read content (resources/read).

        Intercepts resource reads and compresses large text content before
        returning to the client.
        """
        # Execute resource read normally
        result: list[ReadResourceContents] = await call_next(context)

        # Compress text content in each resource
        for resource_content in result:
            # ReadResourceContents has 'content' field (str | bytes)
            if isinstance(resource_content.content, str):
                compressed = await self._compress_text(
                    resource_content.content,
                    context_label="resource",
                )
                if compressed:
                    # Replace with compressed version
                    resource_content.content = compressed

        return result

    async def on_message(
        self,
        context: MiddlewareContext,
        call_next: CallNext,
    ) -> Any:
        """
        Fallback hook for unhandled message types.

        This catches any MCP messages not handled by specific hooks
        (on_call_tool, on_read_resource). Currently just passes through.
        """
        # Log for debugging (optional)
        if context.method not in {"tools/call", "resources/read"}:
            logger.debug(f"MCP middleware: pass-through for {context.method}")

        return await call_next(context)
