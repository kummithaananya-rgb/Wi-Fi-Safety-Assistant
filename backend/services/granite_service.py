"""
IBM Granite / watsonx.ai service.
Primary AI model for all three agents.
"""
import time
import logging
from typing import Optional, Dict, Any

from services.provider_errors import (
    classify_provider_error,
    safe_exception_name,
    safe_http_status,
)

logger = logging.getLogger(__name__)

try:
    import httpx
    from ibm_watsonx_ai import APIClient, Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference
    from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
    IBM_AVAILABLE = True
except ImportError:
    IBM_AVAILABLE = False
    logger.warning("ibm-watsonx-ai not installed; Granite unavailable")


class GraniteService:
    """Wrapper around IBM watsonx.ai Granite model."""

    REQUEST_TIMEOUT_SECONDS = 30

    def __init__(self, api_key: str, project_id: str, url: str, model_id: str):
        self.api_key = api_key
        self.project_id = project_id
        self.url = url
        self.model_id = model_id
        self._model: Optional[Any] = None
        self._api_client: Optional[Any] = None
        self._available = False
        self._initialization_stage = "sdk_check"
        self._initialization_error: Optional[str] = None
        self._initialization_exception_type: Optional[str] = None
        self._initialization_http_status: Optional[int] = None
        self._init()

    def _init(self):
        if not IBM_AVAILABLE:
            self._initialization_error = "not_configured"
            logger.error("ibm-watsonx-ai package not available")
            return
        self._initialization_stage = "credentials"
        if not self.api_key or not self.project_id:
            self._initialization_error = "not_configured"
            logger.warning("IBM credentials not configured; Granite unavailable")
            return
        self._initialization_stage = "api_client"
        if not self.url:
            self._initialization_error = "not_configured"
            logger.warning("IBM watsonx.ai endpoint not configured; Granite unavailable")
            return
        try:
            self._initialization_stage = "credentials"
            credentials = Credentials(url=self.url, api_key=self.api_key)
            self._initialization_stage = "api_client"
            timeout = httpx.Timeout(
                connect=5.0,
                read=self.REQUEST_TIMEOUT_SECONDS,
                write=self.REQUEST_TIMEOUT_SECONDS,
                pool=5.0,
            )
            http_client = httpx.Client(timeout=timeout)
            self._api_client = APIClient(
                credentials=credentials,
                project_id=self.project_id,
                httpx_client=http_client,
            )
            self._initialization_stage = "model_client"
            self._model = ModelInference(
                model_id=self.model_id,
                api_client=self._api_client,
                project_id=self.project_id,
                max_retries=0,
            )
            self._available = True
            self._initialization_stage = "complete"
            logger.info("IBM Granite client initialized: %s", self.model_id)
        except Exception as exc:
            error_code = classify_provider_error(exc)
            self._initialization_error = error_code
            self._initialization_exception_type = safe_exception_name(exc)
            self._initialization_http_status = safe_http_status(exc)
            logger.error("IBM Granite initialization failed (%s)", error_code)
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def initialization_diagnostic(self) -> Dict[str, Any]:
        return {
            "stage": self._initialization_stage,
            "error_category": self._initialization_error,
            "exception_type": self._initialization_exception_type,
            "http_status": self._initialization_http_status,
        }

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
            return self._failure("not_configured")

        params = {
            GenParams.MAX_NEW_TOKENS: max_tokens,
            GenParams.TEMPERATURE: temperature,
            GenParams.REPETITION_PENALTY: 1.1,
        }
        if stop_sequences:
            params[GenParams.STOP_SEQUENCES] = stop_sequences

        t0 = time.monotonic()
        try:
            # Use chat() to call the current /ml/v1/text/chat endpoint (non-deprecated)
            response = self._model.chat(
                messages=[{"role": "user", "content": prompt}],
                params=params,
            )
            duration_ms = round((time.monotonic() - t0) * 1000, 1)
            # chat() returns a dict; extract the assistant message content
            text = ""
            try:
                text = response["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                return self._failure("invalid_response", duration_ms)
            if not isinstance(text, str) or not text.strip():
                return self._failure("invalid_response", duration_ms)
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
        except Exception as exc:
            duration_ms = round((time.monotonic() - t0) * 1000, 1)
            error_code = classify_provider_error(exc)
            logger.warning("Granite request failed (%s)", error_code)
            return self._failure(error_code, duration_ms)

    def _failure(self, error: str, duration_ms: float = 0) -> Dict[str, Any]:
        return {
            "success": False,
            "text": "",
            "model": self.model_id,
            "provider": "granite",
            "duration_ms": duration_ms,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "error": error,
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
