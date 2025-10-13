"""
Token Optimizer Module

Implements various token optimization strategies to reduce token count
while preserving content quality and meaning.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from .counter import get_token_counter

logger = logging.getLogger(__name__)


class OptimizationStrategy(str, Enum):
    """Available optimization strategies."""
    WHITESPACE = "whitespace"
    REDUNDANCY = "redundancy"
    COMPRESSION = "compression"
    SUMMARIZATION = "summarization"
    DEDUPLICATION = "deduplication"


@dataclass
class OptimizationResult:
    """Result from token optimization."""

    original_content: str
    optimized_content: str
    original_tokens: int
    optimized_tokens: int
    reduction_ratio: float
    strategies_applied: List[str]
    quality_score: float
    processing_time_ms: float
    metadata: Dict[str, Any]

    @property
    def tokens_saved(self) -> int:
        """Calculate tokens saved."""
        return self.original_tokens - self.optimized_tokens

    @property
    def reduction_percentage(self) -> float:
        """Calculate reduction percentage."""
        if self.original_tokens == 0:
            return 0.0
        return (self.tokens_saved / self.original_tokens) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_tokens": self.original_tokens,
            "optimized_tokens": self.optimized_tokens,
            "tokens_saved": self.tokens_saved,
            "reduction_ratio": self.reduction_ratio,
            "reduction_percentage": self.reduction_percentage,
            "strategies_applied": self.strategies_applied,
            "quality_score": self.quality_score,
            "processing_time_ms": self.processing_time_ms,
            "metadata": self.metadata,
        }


class TokenOptimizer:
    """
    Token optimizer with multiple strategies.

    Provides various optimization techniques to reduce token count
    while maintaining content quality above threshold.
    """

    def __init__(
        self,
        quality_threshold: float = 0.90,
        preserve_code_blocks: bool = True,
        preserve_formatting: bool = True,
    ) -> None:
        """
        Initialize token optimizer.

        Args:
            quality_threshold: Minimum quality threshold (0.0-1.0)
            preserve_code_blocks: Preserve code blocks during optimization
            preserve_formatting: Preserve important formatting
        """
        self.quality_threshold = quality_threshold
        self.preserve_code_blocks = preserve_code_blocks
        self.preserve_formatting = preserve_formatting
        self.counter = get_token_counter()

    def optimize(
        self,
        content: str,
        target_reduction: float = 0.20,
        model: str = "gpt-4",
        strategies: Optional[List[str]] = None,
        aggressive: bool = False,
    ) -> OptimizationResult:
        """
        Optimize content to reduce token count.

        Args:
            content: Original content to optimize
            target_reduction: Target reduction ratio (0.0-0.5)
            model: Model to optimize for
            strategies: List of strategies to apply (None = use defaults)
            aggressive: Enable aggressive optimization

        Returns:
            OptimizationResult with optimized content and metrics
        """
        import time
        start_time = time.perf_counter()

        # Count original tokens
        original_tokens = self.counter.count_tokens(content, model)

        # Default strategies
        if strategies is None:
            strategies = [
                OptimizationStrategy.WHITESPACE,
                OptimizationStrategy.REDUNDANCY,
            ]
            if aggressive:
                strategies.extend([
                    OptimizationStrategy.COMPRESSION,
                    OptimizationStrategy.DEDUPLICATION,
                ])

        # Apply optimization strategies
        optimized = content
        applied_strategies: List[str] = []

        for strategy in strategies:
            try:
                if strategy == OptimizationStrategy.WHITESPACE:
                    optimized = self._optimize_whitespace(optimized)
                    applied_strategies.append(strategy)
                elif strategy == OptimizationStrategy.REDUNDANCY:
                    optimized = self._remove_redundancy(optimized)
                    applied_strategies.append(strategy)
                elif strategy == OptimizationStrategy.COMPRESSION:
                    optimized = self._compress_content(optimized)
                    applied_strategies.append(strategy)
                elif strategy == OptimizationStrategy.DEDUPLICATION:
                    optimized = self._deduplicate_content(optimized)
                    applied_strategies.append(strategy)
                else:
                    logger.warning(f"Unknown strategy: {strategy}")
            except Exception as e:
                logger.error(f"Error applying strategy {strategy}: {e}")

        # Count optimized tokens
        optimized_tokens = self.counter.count_tokens(optimized, model)

        # Calculate metrics
        reduction_ratio = (
            (original_tokens - optimized_tokens) / original_tokens
            if original_tokens > 0
            else 0.0
        )

        # Estimate quality score
        quality_score = self._estimate_quality(content, optimized)

        # Processing time
        processing_time = (time.perf_counter() - start_time) * 1000

        return OptimizationResult(
            original_content=content,
            optimized_content=optimized,
            original_tokens=original_tokens,
            optimized_tokens=optimized_tokens,
            reduction_ratio=reduction_ratio,
            strategies_applied=applied_strategies,
            quality_score=quality_score,
            processing_time_ms=processing_time,
            metadata={
                "model": model,
                "target_reduction": target_reduction,
                "aggressive": aggressive,
            },
        )

    def _optimize_whitespace(self, content: str) -> str:
        """
        Optimize whitespace without changing meaning.

        Args:
            content: Content to optimize

        Returns:
            Content with optimized whitespace
        """
        if self.preserve_code_blocks:
            # Extract code blocks
            code_blocks = self._extract_code_blocks(content)

            # Replace with placeholders
            for i, _ in enumerate(code_blocks):
                content = content.replace(
                    code_blocks[i],
                    f"__CODE_BLOCK_{i}__",
                    1
                )

        # Remove excessive whitespace
        content = re.sub(r'\n{3,}', '\n\n', content)  # Max 2 newlines
        content = re.sub(r' {2,}', ' ', content)       # Max 1 space
        content = re.sub(r'\t+', ' ', content)         # Tabs to single space

        # Remove trailing whitespace
        lines = content.split('\n')
        lines = [line.rstrip() for line in lines]
        content = '\n'.join(lines)

        if self.preserve_code_blocks:
            # Restore code blocks
            for i, block in enumerate(code_blocks):
                content = content.replace(f"__CODE_BLOCK_{i}__", block, 1)

        return content.strip()

    def _remove_redundancy(self, content: str) -> str:
        """
        Remove redundant phrases and repetition.

        Args:
            content: Content to optimize

        Returns:
            Content with reduced redundancy
        """
        # Common redundant phrases
        redundant_phrases = [
            (r'\bplease note that\b', ''),
            (r'\bit is important to note that\b', ''),
            (r'\bbasically\b', ''),
            (r'\bactually\b', ''),
            (r'\bin order to\b', 'to'),
            (r'\bdue to the fact that\b', 'because'),
            (r'\bat this point in time\b', 'now'),
            (r'\bfor the purpose of\b', 'for'),
            (r'\bin the event that\b', 'if'),
        ]

        for pattern, replacement in redundant_phrases:
            content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)

        # Remove repeated words
        content = re.sub(r'\b(\w+)\s+\1\b', r'\1', content)

        return content

    def _compress_content(self, content: str) -> str:
        """
        Apply compression techniques.

        Args:
            content: Content to compress

        Returns:
            Compressed content
        """
        # Replace common verbose constructions
        replacements = [
            (r'\bwill be able to\b', 'can'),
            (r'\bis going to\b', 'will'),
            (r'\bhas the ability to\b', 'can'),
            (r'\bin spite of the fact that\b', 'although'),
            (r'\buntil such time as\b', 'until'),
            (r'\bwith regard to\b', 'regarding'),
            (r'\bwith reference to\b', 'regarding'),
        ]

        for pattern, replacement in replacements:
            content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)

        return content

    def _deduplicate_content(self, content: str) -> str:
        """
        Remove duplicate lines and paragraphs.

        Args:
            content: Content to deduplicate

        Returns:
            Deduplicated content
        """
        # Split into paragraphs
        paragraphs = content.split('\n\n')

        # Track seen paragraphs
        seen = set()
        unique_paragraphs = []

        for para in paragraphs:
            # Normalize for comparison
            normalized = para.strip().lower()

            if normalized and normalized not in seen:
                seen.add(normalized)
                unique_paragraphs.append(para)

        return '\n\n'.join(unique_paragraphs)

    def _extract_code_blocks(self, content: str) -> List[str]:
        """
        Extract code blocks from content.

        Args:
            content: Content containing code blocks

        Returns:
            List of code blocks
        """
        # Match markdown code blocks
        pattern = r'```[\s\S]*?```'
        matches = re.findall(pattern, content)
        return matches

    def _estimate_quality(self, original: str, optimized: str) -> float:
        """
        Estimate quality of optimization.

        Simple heuristic based on:
        - Length ratio
        - Preserved structure
        - Content similarity

        Args:
            original: Original content
            optimized: Optimized content

        Returns:
            Quality score (0.0-1.0)
        """
        if not original:
            return 1.0

        # Length ratio (avoid over-compression)
        length_ratio = len(optimized) / len(original)
        length_score = min(1.0, length_ratio * 1.2)  # Penalize >17% reduction

        # Structure preservation
        original_lines = len(original.split('\n'))
        optimized_lines = len(optimized.split('\n'))
        structure_score = min(1.0, optimized_lines / original_lines) if original_lines > 0 else 1.0

        # Simple similarity (character overlap)
        common_chars = sum(1 for a, b in zip(original, optimized) if a == b)
        similarity_score = common_chars / len(original) if len(original) > 0 else 1.0

        # Weighted average
        quality = (
            length_score * 0.3 +
            structure_score * 0.3 +
            similarity_score * 0.4
        )

        return min(1.0, max(0.0, quality))

    def analyze_efficiency(
        self,
        content: str,
        model: str = "gpt-4"
    ) -> Dict[str, Any]:
        """
        Analyze content for optimization opportunities.

        Args:
            content: Content to analyze
            model: Model to analyze for

        Returns:
            Analysis with suggestions
        """
        token_count = self.counter.count_tokens(content, model)

        # Count various elements
        whitespace_tokens = self.counter.count_tokens(
            re.sub(r'\S', '', content), model
        )

        # Estimate redundancy
        lines = content.split('\n')
        unique_lines = len(set(line.strip() for line in lines if line.strip()))
        redundancy_ratio = 1.0 - (unique_lines / len(lines)) if lines else 0.0

        # Estimate potential savings
        potential_savings = {
            "whitespace": whitespace_tokens,
            "redundancy": int(token_count * redundancy_ratio),
            "compression": int(token_count * 0.1),  # Conservative estimate
        }

        total_potential = sum(potential_savings.values())

        return {
            "current_tokens": token_count,
            "potential_savings": potential_savings,
            "total_potential_savings": total_potential,
            "potential_reduction_percentage": (
                (total_potential / token_count * 100) if token_count > 0 else 0.0
            ),
            "suggestions": self._generate_suggestions(content, potential_savings),
        }

    def _generate_suggestions(
        self,
        content: str,
        potential_savings: Dict[str, int]
    ) -> List[str]:
        """
        Generate optimization suggestions.

        Args:
            content: Content to analyze
            potential_savings: Estimated savings per strategy

        Returns:
            List of suggestions
        """
        suggestions = []

        if potential_savings["whitespace"] > 10:
            suggestions.append(
                f"Remove excessive whitespace (save ~{potential_savings['whitespace']} tokens)"
            )

        if potential_savings["redundancy"] > 20:
            suggestions.append(
                f"Remove redundant content (save ~{potential_savings['redundancy']} tokens)"
            )

        if potential_savings["compression"] > 15:
            suggestions.append(
                f"Apply compression techniques (save ~{potential_savings['compression']} tokens)"
            )

        if not suggestions:
            suggestions.append("Content is already well-optimized")

        return suggestions


# Singleton instance
_optimizer_instance: Optional[TokenOptimizer] = None


def get_optimizer(
    quality_threshold: float = 0.90,
    preserve_code_blocks: bool = True,
    preserve_formatting: bool = True,
) -> TokenOptimizer:
    """
    Get TokenOptimizer instance.

    Args:
        quality_threshold: Minimum quality threshold
        preserve_code_blocks: Preserve code blocks
        preserve_formatting: Preserve formatting

    Returns:
        TokenOptimizer instance
    """
    global _optimizer_instance
    if _optimizer_instance is None:
        _optimizer_instance = TokenOptimizer(
            quality_threshold=quality_threshold,
            preserve_code_blocks=preserve_code_blocks,
            preserve_formatting=preserve_formatting,
        )
    return _optimizer_instance
