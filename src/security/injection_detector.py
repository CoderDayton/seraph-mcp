"""
Prompt Injection Detector

Heuristic pattern-based detection for prompt injection attacks.
Based on OWASP LLM Top 10 and research from Rebuff, Lakera, HiddenLayer.

Detection Categories:
- Instruction Override: Attempts to ignore system prompts
- Role Confusion: Mimicking system/assistant roles
- Data Exfiltration: Attempts to leak sensitive data
- Code Injection: Shell commands, SQL, XSS patterns
- Encoding Attacks: Base64, unicode escapes, obfuscation
"""

import base64
import logging
import re
from dataclasses import dataclass

from .config import SecurityConfig

logger = logging.getLogger(__name__)


@dataclass
class InjectionDetectionResult:
    """Result from injection detection scan"""

    detected: bool
    risk_score: float  # 0.0 (safe) to 1.0 (high risk)
    matched_patterns: list[dict[str, str]]  # [{"category": "...", "pattern": "...", "match": "..."}]
    explanation: str


class InjectionDetector:
    """
    Heuristic-based prompt injection detector.

    Uses pattern matching against known attack signatures.
    Zero ML dependencies - fast and deterministic.
    """

    # Pattern database: (category, pattern, weight, description)
    PATTERNS = [
        # Instruction Override Attacks
        (
            "instruction_override",
            r"ignore\s+(all\s+)?(previous|above|your)\s+(instruction|prompt|rule)s?",
            0.8,
            "Ignore instructions directive",
        ),
        ("instruction_override", r"forget\s+(everything|all|previous)", 0.8, "Forget directive"),
        (
            "instruction_override",
            r"disregard\s+(your|the)\s+(previous\s+)?(instruction|rule)s?",
            0.8,
            "Disregard directive",
        ),
        ("instruction_override", r"instead,?\s+(do|tell|say|write)", 0.6, "Instruction replacement"),
        ("instruction_override", r"new\s+(instruction|task|prompt|rule)", 0.7, "New instruction injection"),
        # Role Confusion
        ("role_confusion", r"(you\s+are\s+now|act\s+as|simulate|pretend\s+to\s+be)\s+(a|an)", 0.7, "Role redefinition"),
        ("role_confusion", r"system\s*:\s*", 0.9, "System role mimicry"),
        ("role_confusion", r"assistant\s*:\s*", 0.6, "Assistant role mimicry"),
        ("role_confusion", r"<\s*system\s*>", 0.9, "System tag injection"),
        # Data Exfiltration (Higher priority than role_confusion for "system prompt" matches)
        (
            "data_exfiltration",
            r"(show|reveal|display)\s+(me\s+)?(your|the)\s+(original\s+)?(system\s+)?prompt",
            0.9,
            "Prompt leakage attempt",
        ),
        (
            "data_exfiltration",
            r"what\s+(were|are)\s+your\s+(original|initial)\s+instruction",
            0.8,
            "Instruction disclosure",
        ),
        (
            "data_exfiltration",
            r"(output|print|dump)\s+(all\s+)?(your\s+)?(memory|context|data)",
            0.9,
            "Data dump attempt",
        ),
        # Code Injection
        ("code_injection", r"<\s*script[^>]*>", 0.9, "XSS script tag"),
        ("code_injection", r"(exec|eval|system|shell_exec)\s*\(", 0.9, "Code execution function"),
        ("code_injection", r"(;|\||&&)\s*(rm|del|format|shutdown)", 0.9, "Destructive shell command"),
        ("code_injection", r"(SELECT|INSERT|UPDATE|DELETE|DROP)\s+(FROM|INTO|TABLE)", 0.8, "SQL injection"),
        ("code_injection", r"__import__\s*\(", 0.8, "Python import injection"),
        # Encoding/Obfuscation
        ("encoding_attack", r"\\u[0-9a-fA-F]{4}", 0.5, "Unicode escape sequences"),
        ("encoding_attack", r"\\x[0-9a-fA-F]{2}", 0.5, "Hex escape sequences"),
        ("encoding_attack", r"&#x?[0-9a-fA-F]+;", 0.5, "HTML entity encoding"),
        ("encoding_attack", r"%[0-9a-fA-F]{2}", 0.4, "URL encoding"),
        # Delimiter Confusion
        ("delimiter_confusion", r"```[^`]{0,20}(ignore|system|execute)", 0.6, "Code block with suspicious content"),
        ("delimiter_confusion", r"---\s*\n\s*(system|instruction|rule)", 0.7, "Markdown separator injection"),
        ("delimiter_confusion", r"={3,}\s*\n", 0.3, "Delimiter flood"),
        # Prompt Leakage
        ("prompt_leakage", r"repeat\s+(everything|all|the\s+above)", 0.7, "Repeat instruction"),
        (
            "prompt_leakage",
            r"(tell|show)\s+me\s+the\s+(first|exact)\s+(word|instruction)",
            0.8,
            "Exact content request",
        ),
    ]

    # Strict mode patterns (higher false positive rate, but catches advanced attacks)
    STRICT_PATTERNS = [
        ("strict_override", r"(from\s+now\s+on|starting\s+now)", 0.5, "Temporal instruction change"),
        ("strict_override", r"(must|should|will)\s+(always|never|only)", 0.4, "Absolute directive"),
        ("strict_structure", r"<[^>]+>.*</[^>]+>", 0.3, "Nested HTML/XML tags"),
        ("strict_structure", r"\{\{[^}]+\}\}", 0.3, "Template injection pattern"),
    ]

    def __init__(self, config: SecurityConfig | None = None):
        """
        Initialize injection detector.

        Args:
            config: Security configuration (creates default if None)
        """
        if config is None:
            from .config import load_security_config

            config = load_security_config()

        self.config = config
        self.compiled_patterns = self._compile_patterns()

    def _compile_patterns(self) -> list[tuple[str, re.Pattern[str], float, str]]:
        """Compile regex patterns for performance"""
        patterns = self.PATTERNS.copy()

        # Add strict patterns if enabled
        if self.config.strict_mode:
            patterns.extend(self.STRICT_PATTERNS)

        return [
            (category, re.compile(pattern, re.IGNORECASE | re.MULTILINE), weight, desc)
            for category, pattern, weight, desc in patterns
        ]

    def detect(self, content: str) -> InjectionDetectionResult:
        """
        Scan content for injection patterns.

        Args:
            content: Input content to scan

        Returns:
            Detection result with risk score and matched patterns
        """
        if not self.config.injection_detection_enabled:
            return InjectionDetectionResult(
                detected=False, risk_score=0.0, matched_patterns=[], explanation="Injection detection disabled"
            )

        # Detect patterns
        matched_patterns = []
        total_weight = 0.0

        for category, pattern, weight, description in self.compiled_patterns:
            matches = pattern.finditer(content)
            for match in matches:
                matched_patterns.append(
                    {
                        "category": category,
                        "pattern": description,
                        "match": match.group(0)[:100],  # Truncate long matches
                    }
                )
                total_weight += weight

        # Normalize risk score (cap at 1.0)
        # Use cumulative weighting: each pattern contributes fully up to threshold
        risk_score = min(total_weight, 1.0)

        # Check for base64 encoded payloads (potential obfuscation)
        base64_score = self._check_base64_encoding(content)
        risk_score = min(risk_score + base64_score, 1.0)

        # Determine if injection detected
        detected = risk_score >= self.config.injection_risk_threshold

        # Generate explanation
        if not detected:
            explanation = f"No injection detected (risk: {risk_score:.2f})"
        else:
            categories: set[str] = {m["category"] for m in matched_patterns}
            explanation = f"Injection detected (risk: {risk_score:.2f}, categories: {', '.join(categories)})"

        return InjectionDetectionResult(
            detected=detected, risk_score=risk_score, matched_patterns=matched_patterns, explanation=explanation
        )

    def _check_base64_encoding(self, content: str) -> float:
        """
        Detect potential base64-encoded payloads.

        Returns:
            Risk score contribution (0.0 - 0.3)
        """
        # Look for base64-like strings (20+ chars, alphanumeric + / + =)
        base64_pattern = re.compile(r"[A-Za-z0-9+/]{20,}={0,2}")
        matches = base64_pattern.findall(content)

        if not matches:
            return 0.0

        # Try decoding suspicious strings
        decoded_suspicious = 0
        for match in matches[:5]:  # Check first 5 matches only (performance)
            try:
                decoded = base64.b64decode(match).decode("utf-8", errors="ignore")
                # Check if decoded content contains suspicious keywords
                if any(keyword in decoded.lower() for keyword in ["exec", "system", "import", "ignore", "disregard"]):
                    decoded_suspicious += 1
            except Exception:
                continue

        # Return risk contribution (0.1 per suspicious decode, max 0.3)
        return min(decoded_suspicious * 0.1, 0.3)
