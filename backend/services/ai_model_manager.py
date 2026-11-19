"""
Centralized AI Model Manager.
IBM Granite is the only active AI provider.
Groq is disabled in this configuration.
"""
import logging
import time
from typing import Dict, Any, Literal

from services.granite_service import GraniteService
from services.provider_errors import classify_provider_error

logger = logging.getLogger(__name__)

Provider = Literal["granite", "groq", "auto"]
PRIMARY_PROVIDER = "granite"


class AIModelManager:
    """
    Single entry point for all AI model calls.
    IBM Granite handles all AI tasks. 'auto' and legacy 'groq' requests use
    Granite; Groq remains disabled.
    """

    def __init__(self, config):
        self.granite = GraniteService(
            api_key=config.IBM_CLOUD_API_KEY,
            project_id=config.IBM_WATSONX_PROJECT_ID,
            url=config.IBM_WATSONX_URL,
            model_id=config.IBM_GRANITE_MODEL_ID,
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
        Send the request to Granite once; failures remain explicit and safe.
        """
        started = time.monotonic()
        if provider == "groq":
            provider = "auto"

        if provider not in ("granite", "auto"):
            return self._failed_result(
                started, [], "invalid_provider"
            )

        if not self.granite.is_available:
            return self._failed_result(
                started,
                [{
                    "provider": PRIMARY_PROVIDER,
                    "status": "unavailable",
                    "error": "not_configured",
                }],
                "all_providers_failed",
            )

        try:
            result = self.granite.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as exc:
            error_code = classify_provider_error(exc)
            logger.warning("%s request failed (%s)", PRIMARY_PROVIDER, error_code)
            attempts = [{
                "provider": PRIMARY_PROVIDER,
                "status": "failed",
                "error": error_code,
            }]
            return self._failed_result(started, attempts, "all_providers_failed")

        if (
            isinstance(result, dict)
            and result.get("success")
            and isinstance(result.get("text"), str)
            and result["text"].strip()
        ):
            return {
                **result,
                "success": True,
                "text": result["text"].strip(),
                "model": result.get("model") or self.granite.model_id,
                "provider": PRIMARY_PROVIDER,
                "provider_status": "primary_success",
                "provider_attempts": [{
                    "provider": PRIMARY_PROVIDER,
                    "status": "success",
                }],
                "fallback_reason": None,
                "error": None,
            }

        error_code = (
            classify_provider_error(result.get("error"))
            if isinstance(result, dict)
            else "invalid_response"
        )
        if isinstance(result, dict) and result.get("success"):
            error_code = "invalid_response"
        logger.warning("%s request failed (%s)", PRIMARY_PROVIDER, error_code)
        attempts = [{
            "provider": PRIMARY_PROVIDER,
            "status": "failed",
            "error": error_code,
        }]
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
        """Test Granite and report the disabled legacy Groq stub."""
        return {
            "granite": self.granite.test_connection(),
            "groq": self.groq.test_connection(),
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
