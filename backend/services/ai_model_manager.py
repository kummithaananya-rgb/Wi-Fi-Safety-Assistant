"""
Centralized AI Model Manager.
Active providers: IBM Granite (primary) + Google Gemini (secondary/fallback).
Groq is disabled in this configuration.
"""
import logging
from typing import Dict, Any, Literal

from services.granite_service import GraniteService
from services.gemini_service import GeminiService

logger = logging.getLogger(__name__)

Provider = Literal["granite", "gemini", "groq", "auto"]


class AIModelManager:
    """
    Single entry point for all AI model calls.
    IBM Granite is the primary model for all AI tasks.
    Google Gemini is the secondary fallback only.
    'auto' tries granite → gemini.
    'groq' is remapped to gemini (Groq is disabled).
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
                "model": self.granite.model_id,
                "provider": "IBM watsonx.ai",
                "role": "primary",
            },
            "gemini": {
                "available": self.gemini.is_available,
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
        Call the specified provider.
        - 'granite' → IBM Granite (primary)
        - 'gemini'  → Google Gemini (secondary)
        - 'groq'    → remapped to gemini (Groq disabled)
        - 'auto'    → granite → gemini
        """
        # Remap groq to gemini since Groq is disabled
        if provider == "groq":
            provider = "gemini"

        order = {
            "granite": [self.granite],
            "gemini":  [self.gemini],
            "auto":    [self.granite, self.gemini],
        }.get(provider, [self.granite])

        last_error = None
        for svc in order:
            if not svc.is_available:
                continue
            result = svc.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            if result["success"]:
                return result
            last_error = result.get("error")

        logger.error(f"All providers failed for provider={provider}: {last_error}")
        return {
            "success": False,
            "text": (
                "AI services are currently unavailable. "
                "Please verify your IBM and Gemini API credentials in .env and try again."
            ),
            "model": "none",
            "provider": "none",
            "duration_ms": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "error": last_error or "No AI provider available",
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
