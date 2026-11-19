"""Opt-in live connectivity and safe-failure check for IBM Granite."""

import json
import logging
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
PROMPT = "Reply with exactly: Wi-Fi Safety Assistant provider check."
SAFE_ERRORS = {
    "authentication",
    "authorization",
    "configuration",
    "connectivity",
    "invalid_response",
    "not_configured",
    "provider_error",
    "rate_limit",
    "timeout",
    "all_providers_failed",
}
SAFE_INITIALIZATION_STAGES = {
    "sdk_check",
    "credentials",
    "api_client",
    "model_client",
    "complete",
}


class _ForcedFailureProvider:
    def __init__(self, provider):
        self.is_available = True
        self.model_id = f"{provider}-verification"
        self.provider = provider

    def generate(self, **kwargs):
        return {
            "success": False,
            "text": "",
            "model": self.model_id,
            "provider": self.provider,
            "error": "connectivity",
        }


def _result_summary(provider, result, elapsed_ms):
    if not isinstance(result, dict):
        return {
            "provider": provider,
            "success": False,
            "non_empty_response": False,
            "elapsed_ms": round(elapsed_ms, 1),
            "error": "invalid_response",
        }

    text = result.get("text")
    success = bool(result.get("success") and isinstance(text, str) and text.strip())
    error = result.get("error")
    if error not in SAFE_ERRORS:
        error = "provider_error" if not success else None
    return {
        "provider": provider,
        "success": success,
        "non_empty_response": bool(isinstance(text, str) and text.strip()),
        "elapsed_ms": round(elapsed_ms, 1),
        "error": error,
    }


def _run_provider(service, provider, prompt):
    started = time.monotonic()
    try:
        if service.is_available:
            result = service.generate(prompt, max_tokens=32, temperature=0)
        else:
            result = {
                "success": False,
                "text": "",
                "error": "not_configured",
            }
    except Exception as exc:
        from services.provider_errors import classify_provider_error

        result = {
            "success": False,
            "text": "",
            "error": classify_provider_error(exc),
        }
    summary = _result_summary(
        provider, result, (time.monotonic() - started) * 1000
    )
    if provider == "granite":
        from services.provider_errors import SAFE_EXCEPTION_TYPES

        diagnostic = getattr(service, "initialization_diagnostic", {})
        if not isinstance(diagnostic, dict):
            diagnostic = {}
        stage = diagnostic.get("stage")
        error_category = diagnostic.get("error_category")
        exception_type = diagnostic.get("exception_type")
        http_status = diagnostic.get("http_status")
        if exception_type is not None and (
            not isinstance(exception_type, str)
            or exception_type not in SAFE_EXCEPTION_TYPES
        ):
            exception_type = "OtherError"
        summary["initialization"] = {
            "stage": (
                stage
                if isinstance(stage, str) and stage in SAFE_INITIALIZATION_STAGES
                else "unknown"
            ),
            "error_category": (
                error_category
                if isinstance(error_category, str) and error_category in SAFE_ERRORS
                else None
            ),
            "exception_type": exception_type,
            "http_status": (
                http_status
                if type(http_status) is int and 100 <= http_status <= 599
                else None
            ),
        }
    return summary


def main():
    if os.getenv("RUN_LIVE_PROVIDER_TESTS", "").lower() != "true":
        print(json.dumps({
            "live_tests": "not_run",
            "reason": "Set RUN_LIVE_PROVIDER_TESTS=true to enable live API calls.",
        }))
        return 2

    logging.disable(logging.CRITICAL)
    sys.path.insert(0, str(BACKEND))

    from config import Config
    from services.ai_model_manager import AIModelManager
    from services.provider_errors import classify_provider_error

    credentials_present = {"granite": Config.has_granite()}
    try:
        manager = AIModelManager(Config)
    except Exception as exc:
        report = {
            "live_tests": "failed",
            "credentials_present": credentials_present,
            "initialization_error": classify_provider_error(exc),
        }
        print(json.dumps(report, indent=2))
        return 1

    granite_result = _run_provider(manager.granite, "granite", PROMPT)

    failure_manager = AIModelManager.__new__(AIModelManager)
    failure_manager.granite = _ForcedFailureProvider("granite")
    failure_result = failure_manager.generate(
        PROMPT, provider="auto", max_tokens=32, temperature=0
    )
    granite_failure_safe = (
        not failure_result.get("success")
        and failure_result.get("text") == ""
        and failure_result.get("provider") == "none"
        and failure_result.get("provider_status") == "failed"
        and failure_result.get("error") == "all_providers_failed"
    )

    report = {
        "live_tests": "completed",
        "credentials_present": credentials_present,
        "provider_check": granite_result,
        "granite_failure_simulation": {
            "safe_failure_result": granite_failure_safe,
            "provider": failure_result.get("provider", "unknown"),
            "provider_status": failure_result.get("provider_status", "unknown"),
        },
        "security": {
            "no_secret_values_in_report": True,
            "provider_exceptions_reduced_to_categories": True,
        },
    }

    rendered = json.dumps(report, indent=2)
    secret_values = (
        Config.IBM_CLOUD_API_KEY,
        Config.IBM_WATSONX_PROJECT_ID,
    )
    if any(secret and secret in rendered for secret in secret_values):
        print(json.dumps({
            "live_tests": "failed",
            "security": {"report_suppressed_for_sensitive_content": True},
        }))
        return 1

    print(rendered)
    if not all(credentials_present.values()):
        return 2
    return 0 if granite_result["success"] and granite_failure_safe else 1


if __name__ == "__main__":
    raise SystemExit(main())
