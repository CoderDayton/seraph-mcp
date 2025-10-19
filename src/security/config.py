"""
Security Configuration

Heuristic-based injection detection with configurable thresholds.
Zero ML dependencies - pure pattern matching.
"""

import os

from pydantic import BaseModel, Field


class SecurityConfig(BaseModel):
    """Security and injection detection configuration"""

    # Core toggle
    enabled: bool = Field(default=True, description="Enable security validation for compression inputs")

    # Injection detection thresholds
    injection_detection_enabled: bool = Field(default=True, description="Enable prompt injection pattern detection")

    injection_risk_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Risk score threshold (0-1) to reject inputs"
    )

    # Content validation
    max_length_ratio: float = Field(
        default=1.5, ge=1.0, le=3.0, description="Max allowed ratio: compressed_length / original_length"
    )

    min_quality_score: float = Field(
        default=0.75, ge=0.0, le=1.0, description="Minimum quality score for compressed content"
    )

    # Behavioral flags
    log_security_events: bool = Field(
        default=True, description="Log all security detections (for security team analysis)"
    )

    fail_safe_on_detection: bool = Field(
        default=True, description="Use original content when injection detected (rollback)"
    )

    # Pattern matching intensity
    strict_mode: bool = Field(default=False, description="Enable strict pattern matching (higher false positive rate)")


def load_security_config() -> SecurityConfig:
    """
    Load security configuration from environment variables.

    Environment Variables:
        SECURITY_ENABLED: Enable/disable security module (default: true)
        SECURITY_INJECTION_DETECTION_ENABLED: Enable injection detection (default: true)
        SECURITY_INJECTION_RISK_THRESHOLD: Risk threshold 0-1 (default: 0.7)
        SECURITY_MAX_LENGTH_RATIO: Max compressed/original ratio (default: 1.5)
        SECURITY_MIN_QUALITY_SCORE: Min quality score (default: 0.75)
        SECURITY_LOG_EVENTS: Log security events (default: true)
        SECURITY_FAIL_SAFE_ON_DETECTION: Rollback on detection (default: true)
        SECURITY_STRICT_MODE: Enable strict patterns (default: false)
    """
    return SecurityConfig(
        enabled=os.getenv("SECURITY_ENABLED", "true").lower() == "true",
        injection_detection_enabled=os.getenv("SECURITY_INJECTION_DETECTION_ENABLED", "true").lower() == "true",
        injection_risk_threshold=float(os.getenv("SECURITY_INJECTION_RISK_THRESHOLD", "0.7")),
        max_length_ratio=float(os.getenv("SECURITY_MAX_LENGTH_RATIO", "1.5")),
        min_quality_score=float(os.getenv("SECURITY_MIN_QUALITY_SCORE", "0.75")),
        log_security_events=os.getenv("SECURITY_LOG_EVENTS", "true").lower() == "true",
        fail_safe_on_detection=os.getenv("SECURITY_FAIL_SAFE_ON_DETECTION", "true").lower() == "true",
        strict_mode=os.getenv("SECURITY_STRICT_MODE", "false").lower() == "true",
    )
