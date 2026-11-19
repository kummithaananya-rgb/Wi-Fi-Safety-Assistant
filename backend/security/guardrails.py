"""
Security & Guardrails Module
Implements: input validation, PII protection, prompt injection detection,
rate limiting, output validation, and safe handling of cybersecurity requests.
"""
import re
import time
import html
import logging
import threading
from typing import Dict, Any, List, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PII patterns to detect and block
# ---------------------------------------------------------------------------
PII_PATTERNS = {
    "phone_number": re.compile(r"\b[6-9]\d{9}\b"),
    "aadhaar": re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"),
    "pan_card": re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"),
    "credit_card": re.compile(r"\b(?:\d{4}[\s-]?){3}\d{4}\b"),
    "password_hint": re.compile(
        r"\b(password|passwd|pwd|secret|credential|pin)\s*[:=]\s*\S+",
        re.IGNORECASE,
    ),
    "bank_account": re.compile(r"\b\d{9,18}\b"),  # generic bank account
    "email_in_context": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "ifsc_code": re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b"),
}

# ---------------------------------------------------------------------------
# Prompt injection patterns
# ---------------------------------------------------------------------------
INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"ignore\s+previous\s+instructions",
        r"ignore\s+all\s+instructions",
        r"disregard\s+(your|the)\s+(previous|above|prior)",
        r"you\s+are\s+now\s+a",
        r"act\s+as\s+(a\s+)?(different|new|evil|hacker|malicious)",
        r"jailbreak",
        r"dan\s+mode",
        r"developer\s+mode",
        r"system\s+prompt",
        r"reveal\s+your\s+instructions",
        r"print\s+your\s+system\s+prompt",
        r"\\n\\n(human|assistant|user):",
        r"</?system>",
        r"</?prompt>",
    ]
]

# ---------------------------------------------------------------------------
# Dangerous request patterns (things we must never do)
# ---------------------------------------------------------------------------
DANGEROUS_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"scan\s+(the\s+)?network",
        r"nmap",
        r"ping\s+sweep",
        r"port\s+scan",
        r"vulnerability\s+scan",
        r"exploit",
        r"hack\s+(the\s+)?network",
        r"attack\s+(the\s+)?router",
        r"brute\s+force",
        r"crack\s+(the\s+)?(password|wifi)",
        r"intercept\s+(traffic|packets)",
        r"sniff\s+(traffic|packets)",
    ]
]

# ---------------------------------------------------------------------------
# Input length and content limits
# ---------------------------------------------------------------------------
MAX_FIELD_LENGTHS = {
    "wifi_name": 100,
    "connection_type": 50,
    "location": 200,
    "user_context": 1000,
}

ALLOWED_CONNECTION_TYPES = {
    "open", "wep", "wpa", "wpa2", "wpa2-personal", "wpa2-enterprise",
    "wpa3", "wpa3-personal", "wpa3-enterprise", "802.1x", "unknown", "other",
    "unsecured", "none", "secured",
}


class RateLimiter:
    """Simple in-memory token bucket rate limiter."""

    def __init__(self, max_requests_per_minute: int = 30):
        self.max_rpm = max_requests_per_minute
        self._requests: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def is_allowed(self, client_id: str) -> Tuple[bool, int]:
        """
        Returns (allowed, requests_remaining).
        """
        with self._lock:
            now = time.time()
            window = 60.0  # 1 minute
            self._requests[client_id] = [
                t for t in self._requests[client_id] if now - t < window
            ]
            count = len(self._requests[client_id])
            if count >= self.max_rpm:
                return False, 0
            self._requests[client_id].append(now)
            return True, self.max_rpm - count - 1


class OutputValidator:
    """Validates AI model output before returning to users."""

    HALLUCINATION_TRIGGERS = [
        re.compile(p, re.IGNORECASE)
        for p in [
            r"as\s+an\s+ai\s+language\s+model\s+i\s+cannot",
            r"i\s+(will|can)\s+hack",
            r"i\s+have\s+scanned\s+the\s+network",
            r"the\s+network\s+is\s+100%\s+safe",
            r"guaranteed\s+safe",
            r"no\s+risk\s+whatsoever",
        ]
    ]

    REQUIRED_DISCLAIMER = (
        "Note: This assessment is based on user-provided information only. "
        "No network scanning or testing was performed."
    )

    def validate(self, text: str, risk_level: str) -> Dict[str, Any]:
        """
        Returns {valid, issues, sanitized_text}
        """
        issues = []
        sanitized = text if isinstance(text, str) else ""
        if not isinstance(text, str):
            issues.append("Model output is not text")

        # Check for hallucination triggers
        for pat in self.HALLUCINATION_TRIGGERS:
            if pat.search(sanitized):
                issues.append(f"Hallucination trigger detected: {pat.pattern[:40]}")

        # Strip any leaked PII from outputs
        for name, pat in PII_PATTERNS.items():
            if name in (
                "phone_number", "aadhaar", "pan_card", "credit_card",
                "password_hint", "email_in_context", "ifsc_code",
            ):
                sanitized = pat.sub(f"[{name.upper()} REDACTED]", sanitized)

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "sanitized_text": sanitized,
        }


class Guardrails:
    """
    Main guardrails class used by the Supervisor.
    Validates inputs before passing to agents.
    """

    def __init__(
        self,
        max_requests_per_minute: int = 30,
        enable: bool = True,
        max_input_length: int = 2000,
    ):
        if max_input_length < 1:
            raise ValueError("max_input_length must be greater than zero")
        self.enabled = enable
        self.max_input_length = max_input_length
        self.rate_limiter = RateLimiter(max_requests_per_minute)
        self.output_validator = OutputValidator()
        logger.info(f"Guardrails initialized (enabled={enable})")

    def validate_input(
        self,
        wifi_name: str,
        connection_type: str,
        location: str,
        user_context: str,
        client_ip: str = "unknown",
    ) -> Dict[str, Any]:
        """
        Full input validation pipeline.
        Returns {allowed, reason, flags, sanitized_inputs}
        """
        inputs = {
            "wifi_name": str(wifi_name or ""),
            "connection_type": str(connection_type or ""),
            "location": str(location or ""),
            "user_context": str(user_context or ""),
        }
        return self._validate_inputs(
            inputs=inputs,
            client_id=client_ip,
            required_field="wifi_name",
            field_limits=MAX_FIELD_LENGTHS,
        )

    def validate_chat_input(
        self,
        message: str,
        client_id: str = "unknown",
    ) -> Dict[str, Any]:
        """Validate a chat message with the same limits and protections as assessments."""
        inputs = {
            "wifi_name": "",
            "connection_type": "",
            "location": "",
            "user_context": str(message or ""),
        }
        field_limits = dict(MAX_FIELD_LENGTHS)
        field_limits["user_context"] = self.max_input_length
        return self._validate_inputs(
            inputs=inputs,
            client_id=client_id,
            required_field="user_context",
            field_limits=field_limits,
            required_message="A message is required",
        )

    def _validate_inputs(
        self,
        inputs: Dict[str, str],
        client_id: str,
        required_field: str,
        field_limits: Dict[str, int],
        required_message: str = "Wi-Fi network name is required",
    ) -> Dict[str, Any]:
        flags: List[str] = []
        input_length = sum(len(value) for value in inputs.values())
        if input_length > self.max_input_length:
            return {
                "allowed": False,
                "reason": (
                    "Input exceeds the maximum permitted length of "
                    f"{self.max_input_length} characters."
                ),
                "flags": ["input_too_long"],
            }

        if not inputs[required_field].strip():
            return {
                "allowed": False,
                "reason": required_message,
                "flags": ["missing_required_field"],
            }

        remaining = None
        if self.enabled:
            allowed, remaining = self.rate_limiter.is_allowed(client_id)
            if not allowed:
                return {
                    "allowed": False,
                    "reason": "Rate limit exceeded. Please wait before trying again.",
                    "flags": ["rate_limit"],
                }

        # Inspect the full submitted text before field-level truncation or HTML escaping.
        full_text = " ".join(inputs.values())
        if self.enabled:
            for pat in INJECTION_PATTERNS:
                if pat.search(html.unescape(full_text)):
                    logger.warning(
                        "Prompt injection attempt from %s: %s",
                        client_id,
                        pat.pattern[:50],
                    )
                    return {
                        "allowed": False,
                        "reason": (
                            "Invalid input detected. Please describe your Wi-Fi "
                            "connection normally."
                        ),
                        "flags": ["prompt_injection"],
                    }

            for pat in DANGEROUS_PATTERNS:
                if pat.search(full_text):
                    logger.warning(
                        "Dangerous request from %s: %s",
                        client_id,
                        pat.pattern[:50],
                    )
                    return {
                        "allowed": False,
                        "reason": (
                            "This application does not perform network scanning or "
                            "testing. Please describe your Wi-Fi connection details "
                            "for a safety assessment."
                        ),
                        "flags": ["dangerous_request"],
                    }

        for field, max_len in field_limits.items():
            if len(inputs[field]) > max_len:
                inputs[field] = inputs[field][:max_len]
                flags.append(f"field_truncated:{field}")

        sanitized = {key: html.escape(value.strip()) for key, value in inputs.items()}
        pii_found = []
        for name, pat in PII_PATTERNS.items():
            if name == "bank_account":
                continue
            for field, value in sanitized.items():
                if pat.search(value):
                    pii_found.append(name)
                    replacement = (
                        "[PASSWORD REDACTED]"
                        if name == "password_hint"
                        else f"[{name.upper()} REDACTED]"
                    )
                    sanitized[field] = pat.sub(replacement, value)
                    flags.append(f"pii_redacted:{name}")

        if pii_found:
            logger.warning("PII detected from %s: %s", client_id, sorted(set(pii_found)))

        return {
            "allowed": True,
            "reason": "Input validated" if self.enabled else "Guardrails disabled",
            "flags": flags,
            "sanitized": sanitized,
            "rate_limit_remaining": remaining,
        }

    def validate_output(self, text: str, risk_level: str) -> Dict[str, Any]:
        return self.output_validator.validate(text, risk_level)
