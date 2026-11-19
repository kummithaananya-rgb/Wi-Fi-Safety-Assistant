"""
Centralized AI Model Manager.
Active providers: IBM Granite (primary) + Google Gemini (secondary/fallback).
Groq is disabled in this configuration.
"""
import logging
import time
from typing import Dict, Any, Literal

from services.granite_service import GraniteService
from services.gemini_service import GeminiService
from services.provider_errors import classify_provider_error

logger = logging.getLogger(__name__)

Provider = Literal["granite", "gemini", "groq", "auto"]
PRIMARY_PROVIDER = "granite"
FALLBACK_PROVIDER = "gemini"


class AIModelManager:
    """
    Single entry point for all AI model calls.
    IBM Granite is the primary model for all AI tasks.
    Google Gemini is the secondary fallback only.
    'auto' tries granite → gemini.
    Legacy 'groq' requests use the Granite → Gemini route; Groq is disabled.
    """

    def __init__(self, config):
        self.granite = GraniteService(
            api_key=config.IBM_CLOUD_API_KEY,
            project_id=config.IBM_WATSONX_PROJECT_ID,
            url=config.IBM_WATSONX_URL,
            model_id=config.IBM_GRANITE_MODEL_ID,
        )
        self.gemini = GeminiService(
            api_key=config.GOOGLE_GEMINI_API_KEY,
            model_id=config.GEMINI_MODEL_ID,
        )
        # Groq stub – always unavailable; prevents AttributeError in legacy code
        self.groq = _DisabledProvider("Groq (disabled)")

    def status(self) -> Dict[str, Any]:
        return {
            "granite": {
                "available": self.granite.is_available,
                "configured": self.granite.is_available,
                "status": "configured" if self.granite.is_available else "not_configured",
                "model": self.granite.model_id,
                "provider": "IBM watsonx.ai",
                "role": "primary",
            },
            "gemini": {
                "available": self.gemini.is_available,
                "configured": self.gemini.is_available,
                "status": "configured" if self.gemini.is_available else "not_configured",
                "model": self.gemini.model_id,
                "provider": "Google Gemini",
                "role": "secondary/fallback",
            },
            "groq": {
                "available": False,
                "model": "disabled",
                "provider": "Groq (disabled)",
                "role": "disabled",
            },
        }

    def generate(
        self,
        prompt: str,
        provider: Provider = "granite",
        max_tokens: int = 1024,
        temperature: float = 0.3,
        system_prompt: str = None,
    ) -> Dict[str, Any]:
        """
        Call Granite first and retry once with Gemini when requested.
        The same prompt and generation options are sent to both providers.
        """
        started = time.monotonic()
        if provider == "groq":
            provider = "auto"

        order = (
            [(PRIMARY_PROVIDER, self.granite), (FALLBACK_PROVIDER, self.gemini)]
            if provider in ("granite", "auto")
            else [(FALLBACK_PROVIDER, self.gemini)]
            if provider == "gemini"
            else []
        )
        if not order:
            return self._failed_result(
                started, [], "invalid_provider"
            )

        attempts = []
        for provider_name, svc in order:
            if not svc.is_available:
                attempts.append({
                    "provider": provider_name,
                    "status": "unavailable",
                    "error": "not_configured",
                })
                continue

            try:
                result = svc.generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            except Exception as exc:
                error_code = classify_provider_error(exc)
                logger.warning("%s request failed (%s)", provider_name, error_code)
                attempts.append({
                    "provider": provider_name,
                    "status": "failed",
                    "error": error_code,
                })
                continue

            if not isinstance(result, dict):
                error_code = "invalid_response"
            elif result.get("success") and isinstance(result.get("text"), str):
                text = result["text"].strip()
                if text:
                    fallback_reason = next(
                        (
                            attempt["error"] for attempt in attempts
                            if attempt["provider"] == PRIMARY_PROVIDER
                        ),
                        None,
                    )
                    return {
                        **result,
                        "success": True,
                        "text": text,
                        "model": result.get("model") or svc.model_id,
                        "provider": provider_name,
                        "provider_status": (
                            "primary_success"
                            if provider_name == PRIMARY_PROVIDER
                            else "fallback_success"
                        ),
                        "provider_attempts": attempts + [{
                            "provider": provider_name,
                            "status": "success",
                        }],
                        "fallback_reason": fallback_reason,
                        "error": None,
                    }
                error_code = "invalid_response"
            else:
                error_code = classify_provider_error(
                    result.get("error") if isinstance(result, dict) else None
                )

            logger.warning("%s request failed (%s)", provider_name, error_code)
            attempts.append({
                "provider": provider_name,
                "status": "failed",
                "error": error_code,
            })

        return self._failed_result(started, attempts, "all_providers_failed")

    @staticmethod
    def _failed_result(started: float, attempts: list, error: str) -> Dict[str, Any]:
        logger.error("AI provider attempts exhausted (%s)", error)
        return {
            "success": False,
            "text": "",
            "model": "none",
            "provider": "none",
            "provider_status": "failed",
            "provider_attempts": attempts,
            "fallback_reason": (
                attempts[0]["error"] if attempts else error
            ),
            "duration_ms": round((time.monotonic() - started) * 1000, 1),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "error": error,
        }

    def test_all(self) -> Dict[str, Any]:
        """Test both active providers independently."""
        return {
            "granite": self.granite.test_connection(),
            "gemini": self.gemini.test_connection(),
            "groq": {
                "success": False,
                "error": "Groq is disabled in this configuration",
                "test": True,
            },
        }


class _DisabledProvider:
    """Stub that satisfies attribute lookups for the disabled Groq provider."""
    def __init__(self, name: str):
        self.name = name
        self.is_available = False
        self.model_id = "disabled"

    def generate(self, **kwargs) -> Dict[str, Any]:
        return {
            "success": False, "text": "", "model": "disabled",
            "provider": "groq", "duration_ms": 0,
            "prompt_tokens": 0, "completion_tokens": 0,
            "error": f"{self.name} is disabled",
        }

    def test_connection(self) -> Dict[str, Any]:
        return {"success": False, "error": f"{self.name} is disabled", "test": True}


__all__ = ["AIModelManager"]
