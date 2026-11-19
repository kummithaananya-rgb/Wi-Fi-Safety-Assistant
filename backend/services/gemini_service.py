"""
Google Gemini service – secondary AI provider / fallback.
Uses the official google-genai SDK (v3+).
"""
import time
import logging
from typing import Dict, Any

from services.provider_errors import classify_provider_error

logger = logging.getLogger(__name__)

try:
    from google import genai
    from google.genai import types as genai_types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("google-genai not installed; Gemini unavailable")


class GeminiService:
    """Wrapper around Google Gemini API (google-genai SDK)."""

    REQUEST_TIMEOUT_MS = 30_000

    def __init__(self, api_key: str, model_id: str):
        self.api_key = api_key
        self.model_id = model_id
        self._client = None
        self._available = False
        self._init()

    def _init(self):
        if not GEMINI_AVAILABLE:
            logger.error("google-genai package not available")
            return
        if not self.api_key:
            logger.warning("Gemini API key not configured; Gemini unavailable")
            return
        try:
            self._client = genai.Client(
                api_key=self.api_key,
                http_options=genai_types.HttpOptions(
                    timeout=self.REQUEST_TIMEOUT_MS
                ),
            )
            self._available = True
            logger.info(f"Google Gemini initialized: {self.model_id}")
        except Exception as exc:
            error_code = classify_provider_error(exc)
            logger.error("Gemini initialization failed (%s)", error_code)
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> Dict[str, Any]:
        if not self._available:
            return self._failure("not_configured")

        t0 = time.monotonic()
        try:
            gen_config = genai_types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            )
            response = self._client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config=gen_config,
            )
            duration_ms = round((time.monotonic() - t0) * 1000, 1)
            # response.text can be None for some models; fall back to candidates
            text = response.text
            if text is None:
                try:
                    parts = response.candidates[0].content.parts
                    text = "".join(
                        p.text for p in parts if hasattr(p, "text") and p.text
                    )
                except (IndexError, AttributeError, TypeError):
                    text = ""
            if not isinstance(text, str) or not text.strip():
                return self._failure(
                    "invalid_response",
                    round((time.monotonic() - t0) * 1000, 1),
                )
            usage = getattr(response, "usage_metadata", None)
            logger.debug(f"Gemini OK in {duration_ms}ms")
            return {
                "success": True,
                "text": (text or "").strip(),
                "model": self.model_id,
                "provider": "gemini",
                "duration_ms": duration_ms,
                "prompt_tokens": getattr(usage, "prompt_token_count", 0) or 0,
                "completion_tokens": getattr(usage, "candidates_token_count", 0) or 0,
                "error": None,
            }
        except Exception as exc:
            duration_ms = round((time.monotonic() - t0) * 1000, 1)
            error_code = classify_provider_error(exc)
            logger.warning("Gemini request failed (%s)", error_code)
            return self._failure(error_code, duration_ms)

    def _failure(self, error: str, duration_ms: float = 0) -> Dict[str, Any]:
        return {
            "success": False,
            "text": "",
            "model": self.model_id,
            "provider": "gemini",
            "duration_ms": duration_ms,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "error": error,
        }

    def test_connection(self) -> Dict[str, Any]:
        """Check connectivity with a prompt that must produce non-empty text."""
        result = self.generate(
            "Say the words 'Google Gemini operational' and nothing else.",
            max_tokens=20,
            temperature=0.1,
        )
        result["test"] = True
        return result
