"""
IBM Granite / watsonx.ai service.
Primary AI model for all three agents.
"""
import time
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

try:
    from ibm_watsonx_ai import APIClient, Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference
    from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
    IBM_AVAILABLE = True
except ImportError:
    IBM_AVAILABLE = False
    logger.warning("ibm-watsonx-ai not installed; Granite unavailable")


class GraniteService:
    """Wrapper around IBM watsonx.ai Granite model."""

    def __init__(self, api_key: str, project_id: str, url: str, model_id: str):
        self.api_key = api_key
        self.project_id = project_id
        self.url = url
        self.model_id = model_id
        self._model: Optional[Any] = None
        self._available = False
        self._init()

    def _init(self):
        if not IBM_AVAILABLE:
            logger.error("ibm-watsonx-ai package not available")
            return
        if not self.api_key or not self.project_id:
            logger.warning("IBM credentials not configured; Granite unavailable")
            return
        try:
            credentials = Credentials(url=self.url, api_key=self.api_key)
            self._model = ModelInference(
                model_id=self.model_id,
                credentials=credentials,
                project_id=self.project_id,
            )
            self._available = True
            logger.info(f"IBM Granite initialized: {self.model_id}")
        except Exception as e:
            logger.error(f"IBM Granite init failed: {e}")
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        stop_sequences: list = None,
    ) -> Dict[str, Any]:
        """
        Call Granite and return a structured result dict.
        Returns: {success, text, model, duration_ms, prompt_tokens, completion_tokens, error}
        """
        if not self._available:
            return {
                "success": False,
                "text": "",
                "model": self.model_id,
                "duration_ms": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "error": "Granite not available – check IBM credentials",
            }

        params = {
            GenParams.MAX_NEW_TOKENS: max_tokens,
            GenParams.TEMPERATURE: temperature,
            GenParams.REPETITION_PENALTY: 1.1,
        }
        if stop_sequences:
            params[GenParams.STOP_SEQUENCES] = stop_sequences

        t0 = time.time()
        try:
            # Use chat() to call the current /ml/v1/text/chat endpoint (non-deprecated)
            response = self._model.chat(
                messages=[{"role": "user", "content": prompt}],
                params=params,
            )
            duration_ms = round((time.time() - t0) * 1000, 1)
            # chat() returns a dict; extract the assistant message content
            text = ""
            try:
                text = response["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                text = str(response)
            logger.debug(f"Granite OK in {duration_ms}ms, ~{len(text.split())} words")
            return {
                "success": True,
                "text": text.strip(),
                "model": self.model_id,
                "provider": "granite",
                "duration_ms": duration_ms,
                "prompt_tokens": len(prompt.split()),       # estimate
                "completion_tokens": len(text.split()),     # estimate
                "error": None,
            }
        except Exception as e:
            duration_ms = round((time.time() - t0) * 1000, 1)
            logger.error(f"Granite generate error: {e}")
            return {
                "success": False,
                "text": "",
                "model": self.model_id,
                "provider": "granite",
                "duration_ms": duration_ms,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "error": str(e),
            }

    def test_connection(self) -> Dict[str, Any]:
        """Quick connectivity test."""
        result = self.generate(
            "Respond with exactly: IBM Granite operational",
            max_tokens=20,
            temperature=0.0,
        )
        result["test"] = True
        return result
