"""
Groq service – secondary AI provider.
Used for fast inference and incident response guidance.
"""
import time
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logger.warning("groq package not installed; Groq unavailable")


class GroqService:
    """Wrapper around Groq API."""

    def __init__(self, api_key: str, model_id: str):
        self.api_key = api_key
        self.model_id = model_id
        self._client = None
        self._available = False
        self._init()

    def _init(self):
        if not GROQ_AVAILABLE:
            logger.error("groq package not available")
            return
        if not self.api_key:
            logger.warning("Groq API key not configured; Groq unavailable")
            return
        try:
            self._client = Groq(api_key=self.api_key)
            self._available = True
            logger.info(f"Groq initialized: {self.model_id}")
        except Exception as e:
            logger.error(f"Groq init failed: {e}")
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        system_prompt: str = None,
    ) -> Dict[str, Any]:
        if not self._available:
            return {
                "success": False,
                "text": "",
                "model": self.model_id,
                "provider": "groq",
                "duration_ms": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "error": "Groq not available – check API key",
            }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        t0 = time.time()
        try:
            response = self._client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            duration_ms = round((time.time() - t0) * 1000, 1)
            text = response.choices[0].message.content
            logger.debug(f"Groq OK in {duration_ms}ms")
            return {
                "success": True,
                "text": text.strip(),
                "model": self.model_id,
                "provider": "groq",
                "duration_ms": duration_ms,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "error": None,
            }
        except Exception as e:
            duration_ms = round((time.time() - t0) * 1000, 1)
            logger.error(f"Groq generate error: {e}")
            return {
                "success": False,
                "text": "",
                "model": self.model_id,
                "provider": "groq",
                "duration_ms": duration_ms,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "error": str(e),
            }

    def test_connection(self) -> Dict[str, Any]:
        result = self.generate(
            "Respond with exactly: Groq operational",
            max_tokens=20,
            temperature=0.0,
        )
        result["test"] = True
        return result
