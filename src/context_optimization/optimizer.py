"""
AI-Powered Context Optimizer with Seraph Multi-Layer Compression

Hybrid compression system:
- AI compression: Fast, nuanced, best for short prompts (≤3k tokens)
- Seraph compression: Deterministic, cacheable, multi-layer (L1/L2/L3), best for long/recurring contexts (>3k tokens)
- Hybrid mode: Seraph pre-compress + AI polish for optimal results

Target: <100ms processing, >=90% quality, 20-40% token reduction
"""

import asyncio
import hashlib
import time

import tiktoken

from .config import ContextOptimizationConfig
from .models import FeedbackRecord, OptimizationResult
from .seraph_compression import CompressionResult, SeraphCompressor

# Optimization prompt based on LLMLingua and 2025 research
OPTIMIZATION_PROMPT = """You are an expert at compressing text while preserving all important information and meaning.

CRITICAL RULES:
1. NEVER remove or alter code blocks (```...```)
2. NEVER remove structured data (JSON, YAML, XML)
3. NEVER remove technical terms or key concepts
4. Preserve all numbers, metrics, and specific details
5. Keep the same language and tone

YOUR TASK: Compress the following text by 20-40% while maintaining 100% of the meaning.

Compression techniques to use:
- Remove filler words and redundant phrases
- Use more concise phrasing
- Merge similar sentences
- Remove unnecessary explanations while keeping core information
- Compress verbose language to terse but complete statements

Original text:
{content}

Provide ONLY the compressed version. No explanations, no preamble, just the compressed text."""

# Quality validation prompt
VALIDATION_PROMPT = """Compare these two texts and rate how well the compressed version preserves the original meaning.

Original:
{original}

Compressed:
{compressed}

Rate the compression quality from 0.0 to 1.0 where:
- 1.0 = Perfect preservation (no information lost)
- 0.9 = Excellent (tiny details lost but all key info intact)
- 0.8 = Good (minor info lost but meaning preserved)
- 0.7 = Acceptable (some info lost but main ideas clear)
- <0.7 = Poor (significant information or meaning lost)

Respond with ONLY a single number between 0.0 and 1.0. Nothing else."""


class ContextOptimizer:
    """AI-powered context optimizer with automatic learning"""

    def __init__(self, config: ContextOptimizationConfig, provider=None, budget_tracker=None):
        """
        Initialize optimizer.

        Args:
            config: Optimization configuration
            provider: AI provider for optimization (uses existing provider system)
            budget_tracker: Optional BudgetTracker for cost savings tracking
        """
        self.config = config
        self.provider = provider
        self.budget_tracker = budget_tracker

        # Token counter
        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.encoding = None

        # Simple in-memory cache for optimization results
        self.cache = {}

        # Seraph compressor for deterministic multi-layer compression
        self.seraph_compressor = SeraphCompressor(seed=7)
        self.seraph_cache = {}  # Cache for seraph compression results

        # Learning statistics
        self.stats = {
            "total_optimizations": 0,
            "successful_optimizations": 0,
            "total_tokens_saved": 0,
            "avg_quality_score": 0.0,
            "avg_reduction_percentage": 0.0,
            "method_usage": {
                "ai": 0,
                "seraph": 0,
                "hybrid": 0,
            },
        }

    async def optimize(self, content: str, timeout_ms: float | None = None) -> OptimizationResult:
        """
        Optimize content using hybrid compression system.

        Automatically selects best method based on:
        - Content size (token count)
        - Configured compression_method
        - Performance constraints

        Args:
            content: Text content to optimize
            timeout_ms: Override default timeout

        Returns:
            OptimizationResult with optimized content and metrics
        """
        start_time = time.perf_counter()
        timeout = timeout_ms or self.config.max_overhead_ms

        # Check if optimization is enabled
        if not self.config.enabled:
            return self._create_passthrough_result(content, start_time)

        # Check cache
        content_hash = self._hash_content(content)
        if content_hash in self.cache:
            cached = self.cache[content_hash]
            # Update cache hit stats
            return cached

        # Count original tokens
        tokens_before = self._count_tokens(content)

        # Skip very small content (not worth optimizing)
        if tokens_before < 100:
            return self._create_passthrough_result(content, start_time)

        # Determine compression method
        method = self._select_compression_method(tokens_before)

        try:
            # Route to appropriate compression method
            if method == "ai":
                optimized_content, quality_score = await asyncio.wait_for(
                    self._optimize_with_ai(content), timeout=timeout / 1000.0
                )
            elif method == "seraph":
                optimized_content, quality_score = await asyncio.wait_for(
                    self._optimize_with_seraph(content), timeout=timeout / 1000.0
                )
            elif method == "hybrid":
                optimized_content, quality_score = await asyncio.wait_for(
                    self._optimize_hybrid(content), timeout=timeout / 1000.0
                )
            else:
                # Fallback to AI
                optimized_content, quality_score = await asyncio.wait_for(
                    self._optimize_with_ai(content), timeout=timeout / 1000.0
                )

            # Count optimized tokens
            tokens_after = self._count_tokens(optimized_content)
            tokens_saved = max(0, tokens_before - tokens_after)
            reduction_percentage = (tokens_saved / tokens_before * 100) if tokens_before > 0 else 0

            # Validate quality
            validation_passed = quality_score >= self.config.quality_threshold

            # Rollback if quality too low
            rollback_occurred = False
            if not validation_passed:
                optimized_content = content
                tokens_after = tokens_before
                tokens_saved = 0
                reduction_percentage = 0
                rollback_occurred = True

            # Calculate time
            optimization_time_ms = (time.perf_counter() - start_time) * 1000

            # Calculate cost savings
            cost_savings_usd = await self._calculate_cost_savings(
                tokens_saved, getattr(self.provider, "model_name", None)
            )

            # Create result
            result = OptimizationResult(
                original_content=content,
                optimized_content=optimized_content,
                tokens_before=tokens_before,
                tokens_after=tokens_after,
                tokens_saved=tokens_saved,
                reduction_percentage=reduction_percentage,
                quality_score=quality_score,
                validation_passed=validation_passed,
                method=method,
                optimization_time_ms=optimization_time_ms,
                cost_savings_usd=cost_savings_usd,
                model_name=getattr(self.provider, "model_name", None),
                rollback_occurred=rollback_occurred,
            )

            # Track cost savings in budget if available
            if self.budget_tracker and cost_savings_usd > 0 and validation_passed:
                try:
                    await self._record_budget_savings(result)
                except Exception as e:
                    print(f"Budget tracking error: {e}")

            # Cache successful optimizations
            if validation_passed and not rollback_occurred:
                self.cache[content_hash] = result

            # Update stats
            self._update_stats(result)

            # Store feedback for learning (async, don't wait)
            asyncio.create_task(self._store_feedback(result))

            return result

        except asyncio.TimeoutError:
            # Timeout - return original content
            optimization_time_ms = (time.perf_counter() - start_time) * 1000
            return OptimizationResult(
                original_content=content,
                optimized_content=content,
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                tokens_saved=0,
                reduction_percentage=0.0,
                quality_score=1.0,  # No change = perfect preservation
                validation_passed=True,
                optimization_time_ms=optimization_time_ms,
                rollback_occurred=True,
            )

        except Exception as e:
            # Error - return original content
            optimization_time_ms = (time.perf_counter() - start_time) * 1000
            print(f"Optimization error: {e}")
            return self._create_passthrough_result(content, start_time)

    async def _optimize_with_ai(self, content: str) -> tuple[str, float]:
        """
        Perform AI-powered optimization using provider.

        Returns:
            Tuple of (optimized_content, quality_score)
        """
        if not self.provider:
            # No provider available - return original
            return content, 1.0

        # Step 1: Compress using AI
        optimization_prompt = OPTIMIZATION_PROMPT.format(content=content)

        try:
            # Use provider to generate compressed version
            compressed = await self._call_provider(optimization_prompt, max_tokens=len(content))

            # Step 2: Validate quality using AI
            validation_prompt = VALIDATION_PROMPT.format(original=content, compressed=compressed)

            quality_response = await self._call_provider(validation_prompt, max_tokens=10)

            # Parse quality score
            try:
                quality_score = float(quality_response.strip())
                quality_score = max(0.0, min(1.0, quality_score))  # Clamp to [0, 1]
            except ValueError:
                # If parsing fails, assume moderate quality
                quality_score = 0.85

            return compressed, quality_score

        except Exception as e:
            print(f"AI optimization error: {e}")
            return content, 1.0

    def _select_compression_method(self, token_count: int) -> str:
        """
        Select compression method based on content size and configuration.

        Decision matrix:
        - "ai": AI-powered compression (fast, nuanced) - requires provider
        - "seraph": Deterministic multi-layer compression (cacheable, scalable) - no provider needed
        - "hybrid": Seraph pre-compress + AI polish (best of both) - requires provider
        - "auto": Automatic selection based on token_count and provider availability

        Args:
            token_count: Number of tokens in content

        Returns:
            Method name: "ai", "seraph", or "hybrid"
        """
        method = self.config.compression_method.lower()

        # Check if provider is available
        has_provider = self.provider is not None

        if method == "auto":
            # Auto-select based on token threshold AND provider availability
            if not has_provider:
                return "seraph"  # No provider: always use Seraph (works without AI)
            elif token_count <= self.config.seraph_token_threshold:
                return "ai"  # Short content + provider: AI is faster and more nuanced
            else:
                return "seraph"  # Long content: Seraph is more efficient and cacheable
        elif method == "ai":
            # AI requested but no provider available - fallback to seraph
            if not has_provider:
                print("Warning: AI compression requested but no provider available, using Seraph")
                return "seraph"
            return "ai"
        elif method == "hybrid":
            # Hybrid requested but no provider available - fallback to seraph
            if not has_provider:
                print("Warning: Hybrid compression requested but no provider available, using Seraph")
                return "seraph"
            return "hybrid"
        elif method == "seraph":
            return "seraph"  # Seraph always works
        else:
            # Unknown method - fallback based on provider availability
            return "seraph" if not has_provider else "ai"

    async def _optimize_with_seraph(self, content: str) -> tuple[str, float]:
        """
        Perform deterministic multi-layer compression using Seraph.

        Returns L3 layer (largest compressed layer) which typically achieves
        20-40% reduction while preserving factual content.

        Returns:
            Tuple of (optimized_content, quality_score)
        """
        # Check seraph cache
        content_hash = self._hash_content(content)
        if content_hash in self.seraph_cache:
            cached = self.seraph_cache[content_hash]
            return cached["l3"], cached["quality"]

        try:
            # Build compression layers
            result: CompressionResult = self.seraph_compressor.build(content)

            # Use L3 (largest layer) as the optimized output
            # L3 typically preserves ~60-80% of tokens with high quality
            optimized_content = result.l3

            # Seraph quality estimation based on compression ratio
            # Seraph is deterministic and structure-preserving, so quality is consistently high
            tokens_original = self._count_tokens(content)
            tokens_compressed = self._count_tokens(optimized_content)
            compression_ratio = tokens_compressed / tokens_original if tokens_original > 0 else 1.0

            # Quality score: Seraph preserves structure well, estimate based on ratio
            # Higher compression = slightly lower quality (but still good)
            if compression_ratio >= 0.7:
                quality_score = 0.95  # Excellent preservation
            elif compression_ratio >= 0.5:
                quality_score = 0.92  # Very good preservation
            else:
                quality_score = 0.88  # Good preservation

            # Cache result
            self.seraph_cache[content_hash] = {
                "l1": result.l1,
                "l2": result.l2,
                "l3": result.l3,
                "quality": quality_score,
                "manifest": result.manifest,
            }

            return optimized_content, quality_score

        except Exception as e:
            print(f"Seraph compression error: {e}")
            # Fallback: return original with perfect quality
            return content, 1.0

    async def _optimize_hybrid(self, content: str) -> tuple[str, float]:
        """
        Hybrid compression: Seraph pre-compress + AI polish.

        Best of both worlds:
        1. Seraph deterministically compresses to L2 or L3
        2. AI polishes the compressed content for better readability

        Returns:
            Tuple of (optimized_content, quality_score)
        """
        try:
            # Step 1: Seraph pre-compression to L2 (more aggressive than L3)
            result: CompressionResult = self.seraph_compressor.build(content)

            # Use L2 for hybrid (more compressed, leaves room for AI polish)
            seraph_compressed = result.l2

            # If L2 is too short or empty, use L3
            if not seraph_compressed or len(seraph_compressed.strip()) < 50:
                seraph_compressed = result.l3

            # Step 2: AI polish the seraph-compressed content
            if self.provider and seraph_compressed:
                polished, quality = await self._optimize_with_ai(seraph_compressed)

                # Verify hybrid result is actually better than seraph alone
                tokens_hybrid = self._count_tokens(polished)
                tokens_seraph = self._count_tokens(seraph_compressed)

                # Only use hybrid if it provides additional compression with good quality
                if tokens_hybrid < tokens_seraph and quality >= self.config.quality_threshold:
                    return polished, quality
                else:
                    # AI didn't improve - use seraph result
                    return seraph_compressed, 0.92
            else:
                # No AI provider - just return seraph result
                return seraph_compressed, 0.92

        except Exception as e:
            print(f"Hybrid compression error: {e}")
            # Fallback to seraph only
            return await self._optimize_with_seraph(content)

    async def _call_provider(self, prompt: str, max_tokens: int = 500) -> str:
        """Call AI provider with prompt"""
        if not self.provider:
            return ""

        try:
            # Use provider's generate method (adjust based on actual provider interface)
            if hasattr(self.provider, "generate"):
                response = await self.provider.generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=0.3,  # Low temperature for consistent compression
                )
                return response.get("content", "") or response.get("text", "")
            else:
                # Fallback to chat-style interface
                messages = [{"role": "user", "content": prompt}]
                response = await self.provider.chat(messages=messages, max_tokens=max_tokens)
                return response.get("content", "") or response.get("message", {}).get("content", "")

        except Exception as e:
            print(f"Provider call error: {e}")
            return ""

    def _count_tokens(self, content: str) -> int:
        """Count tokens in content"""
        if self.encoding:
            try:
                return len(self.encoding.encode(content))
            except Exception:
                pass

        # Approximate: ~4 chars per token
        return len(content) // 4

    def _hash_content(self, content: str) -> str:
        """Generate hash for content"""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _create_passthrough_result(self, content: str, start_time: float) -> OptimizationResult:
        """Create result for content that wasn't optimized"""
        tokens = self._count_tokens(content)
        optimization_time_ms = (time.perf_counter() - start_time) * 1000

        return OptimizationResult(
            original_content=content,
            optimized_content=content,
            tokens_before=tokens,
            tokens_after=tokens,
            tokens_saved=0,
            reduction_percentage=0.0,
            quality_score=1.0,
            validation_passed=True,
            optimization_time_ms=optimization_time_ms,
            rollback_occurred=False,
        )

    def _update_stats(self, result: OptimizationResult):
        """Update running statistics"""
        self.stats["total_optimizations"] += 1

        # Track method usage
        method = getattr(result, "method", "ai")
        if method in self.stats["method_usage"]:
            self.stats["method_usage"][method] += 1

        if result.validation_passed and not result.rollback_occurred:
            self.stats["successful_optimizations"] += 1
            self.stats["total_tokens_saved"] += result.tokens_saved

            # Running average for quality
            n = self.stats["successful_optimizations"]
            current_avg_quality = self.stats["avg_quality_score"]
            self.stats["avg_quality_score"] = (current_avg_quality * (n - 1) + result.quality_score) / n

            # Running average for reduction
            current_avg_reduction = self.stats["avg_reduction_percentage"]
            self.stats["avg_reduction_percentage"] = (current_avg_reduction * (n - 1) + result.reduction_percentage) / n

    async def _store_feedback(self, result: OptimizationResult):
        """Store feedback for adaptive learning (async)"""
        try:
            # Calculate success score
            success_score = result.quality_score if result.validation_passed else 0.0

            # Record feedback for future improvements
            _feedback = FeedbackRecord(
                record_id=f"opt_{int(time.time() * 1000)}",
                content_hash=self._hash_content(result.original_content),
                tokens_saved=result.tokens_saved,
                reduction_percentage=result.reduction_percentage,
                quality_score=result.quality_score,
                validation_passed=result.validation_passed,
                optimization_time_ms=result.optimization_time_ms,
                success_score=success_score,
            )

            # TODO: Store in database for long-term learning
            # For now, just keep in memory

        except Exception as e:
            print(f"Feedback storage error: {e}")

    def get_stats(self) -> dict:
        """Get current optimization statistics"""
        success_rate = (
            self.stats["successful_optimizations"] / self.stats["total_optimizations"]
            if self.stats["total_optimizations"] > 0
            else 0.0
        )

        return {
            **self.stats,
            "success_rate": success_rate,
            "cache_size": len(self.cache),
        }

    def clear_cache(self):
        """Clear optimization caches (both AI and Seraph)"""
        self.cache.clear()
        self.seraph_cache.clear()

    async def _calculate_cost_savings(self, tokens_saved: int, model_name: str | None) -> float:
        """
        Calculate cost savings from token reduction.

        Args:
            tokens_saved: Number of tokens saved
            model_name: Model name for pricing lookup

        Returns:
            Cost savings in USD
        """
        if tokens_saved <= 0 or not model_name:
            return 0.0

        # Simple pricing lookup (can be enhanced with models.dev integration)
        # Average input token prices for common models (per 1M tokens)
        pricing = {
            "gpt-4": 30.0,
            "gpt-4-turbo": 10.0,
            "gpt-3.5-turbo": 0.50,
            "claude-3-opus": 15.0,
            "claude-3-sonnet": 3.0,
            "claude-3-haiku": 0.25,
            "gemini-pro": 0.50,
        }

        # Find best matching price
        price_per_million = 1.0  # Default fallback
        for model_key, price in pricing.items():
            if model_key in model_name.lower():
                price_per_million = price
                break

        # Calculate savings: (tokens_saved / 1_000_000) * price_per_million
        cost_savings = (tokens_saved / 1_000_000) * price_per_million
        return round(cost_savings, 6)

    async def _record_budget_savings(self, result: OptimizationResult):
        """Record cost savings in budget tracker"""
        if not self.budget_tracker:
            return

        try:
            # Import here to avoid circular dependency
            from ..budget_management import record_transaction

            # Record as a negative cost (savings)
            await record_transaction(
                tracker=self.budget_tracker,
                amount=-result.cost_savings_usd,  # Negative = savings
                transaction_type="optimization_savings",
                model=result.model_name or "unknown",
                tokens=result.tokens_saved,
                metadata={
                    "reduction_percentage": result.reduction_percentage,
                    "quality_score": result.quality_score,
                },
            )
        except ImportError:
            # Budget management not available
            pass
        except Exception as e:
            print(f"Failed to record budget savings: {e}")


# Convenience function
async def optimize_content(
    content: str,
    provider=None,
    config: ContextOptimizationConfig | None = None,
    budget_tracker=None,
) -> OptimizationResult:
    """
    Optimize content with AI-powered compression.

    Args:
        content: Content to optimize
        provider: AI provider instance
        config: Optional configuration (uses defaults if not provided)
        budget_tracker: Optional budget tracker for cost savings

    Returns:
        OptimizationResult
    """
    from .config import load_config

    if config is None:
        config = load_config()

    optimizer = ContextOptimizer(config=config, provider=provider, budget_tracker=budget_tracker)
    return await optimizer.optimize(content)
