"""
Content Validators

Post-compression validation to ensure output quality and safety.
"""

import logging
from dataclasses import dataclass

from .config import SecurityConfig

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result from content validation"""

    passed: bool
    reasons: list[str]  # Failure reasons if any
    metrics: dict[str, float]  # Validation metrics


class ContentValidator:
    """
    Post-compression content validator.

    Ensures compressed content meets quality and safety thresholds.
    """

    def __init__(self, config: SecurityConfig | None = None):
        """
        Initialize content validator.

        Args:
            config: Security configuration (creates default if None)
        """
        if config is None:
            from .config import load_security_config

            config = load_security_config()

        self.config = config

    def validate_compressed_content(
        self,
        original: str,
        compressed: str,
        quality_score: float,
    ) -> ValidationResult:
        """
        Validate compressed content against original.

        Checks:
        - Length ratio (compressed shouldn't be longer than original * max_ratio)
        - Quality score meets minimum threshold
        - Content similarity (basic heuristic)

        Args:
            original: Original content
            compressed: Compressed content
            quality_score: Quality score from compression (0-1)

        Returns:
            Validation result
        """
        reasons = []
        metrics = {}

        # Check 1: Length ratio
        orig_len = len(original)
        comp_len = len(compressed)
        length_ratio = comp_len / orig_len if orig_len > 0 else 0.0

        metrics["length_ratio"] = length_ratio
        metrics["original_length"] = orig_len
        metrics["compressed_length"] = comp_len

        if length_ratio > self.config.max_length_ratio:
            reasons.append(
                f"Compressed content too long (ratio: {length_ratio:.2f}, max: {self.config.max_length_ratio})"
            )

        # Check 2: Quality score
        metrics["quality_score"] = quality_score

        if quality_score < self.config.min_quality_score:
            reasons.append(f"Quality score below threshold ({quality_score:.2f} < {self.config.min_quality_score})")

        # Check 3: Empty content protection
        if not compressed.strip():
            reasons.append("Compressed content is empty")
            metrics["is_empty"] = 1.0
        else:
            metrics["is_empty"] = 0.0

        # Check 4: Structural similarity (basic heuristic)
        structure_score = self._check_structural_similarity(original, compressed)
        metrics["structure_similarity"] = structure_score

        if structure_score < 0.3:  # Very low structural similarity
            reasons.append(f"Structural similarity too low ({structure_score:.2f})")

        # Determine pass/fail
        passed = len(reasons) == 0

        return ValidationResult(
            passed=passed,
            reasons=reasons,
            metrics=metrics,
        )

    def _check_structural_similarity(self, original: str, compressed: str) -> float:
        """
        Basic structural similarity check.

        Compares:
        - Line count ratio
        - Word count ratio
        - Common word preservation

        Returns:
            Similarity score (0.0 - 1.0)
        """
        # Count lines
        orig_lines = len(original.split("\n"))
        comp_lines = len(compressed.split("\n"))
        line_ratio = min(comp_lines / orig_lines, 1.0) if orig_lines > 0 else 0.0

        # Count words
        orig_words = set(original.lower().split())
        comp_words = set(compressed.lower().split())

        # Jaccard similarity (word overlap)
        if not orig_words:
            word_similarity = 0.0
        else:
            intersection = len(orig_words & comp_words)
            union = len(orig_words | comp_words)
            word_similarity = intersection / union if union > 0 else 0.0

        # Weighted average (favor word similarity)
        return 0.3 * line_ratio + 0.7 * word_similarity
