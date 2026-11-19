"""
Agent 1: Network Risk Assessment Agent
Analyzes user-provided Wi-Fi connection details.
Produces a structured risk assessment (LOW / MEDIUM / HIGH).
Uses IBM Granite as primary model.
"""
import re
import json
import logging
import math
from typing import Dict, Any, List

from security.guardrails import OutputValidator

logger = logging.getLogger(__name__)

ASSESSMENT_OUTPUT_FIELDS = frozenset({
    "risk_level",
    "risk_score",
    "summary",
    "warning_signs",
    "risk_factors",
    "positive_indicators",
    "explanation",
    "cert_in_reference",
})

NO_AUTHENTICATION_TYPES = {"open", "unsecured", "none"}
WEAK_ENCRYPTION_TYPES = {"wep"}
UNKNOWN_SECURITY_TYPES = {"", "unknown", "other", "captive_portal"}
MODERATE_SECURITY_TYPES = {"wpa", "wpa2", "wpa2-personal"}
SECURED_TYPES = {
    "wpa2-enterprise", "wpa3", "wpa3-personal", "wpa3-enterprise",
    "802.1x", "secured",
}

BASE_PUBLIC_WIFI_SCORE = 10
UNKNOWN_PROVIDER_SCORE = 10
UNKNOWN_SECURITY_SCORE = 30
NO_AUTHENTICATION_SCORE = 60
WEAK_ENCRYPTION_SCORE = 60
MODERATE_SECURITY_SCORE = 10
REPORTED_WARNING_SCORE = 60

HIGH_RISK_INDICATORS = (
    "certificate warning",
    "ssl error",
    "https warning",
    "redirected to unknown site",
    "redirects to unknown site",
    "unexpected login page",
    "asks for password",
    "asked for password",
    "asked for my password",
    "asks for otp",
    "asks for bank details",
)


class NetworkRiskAgent:
    """
    Agent 1 – Network Risk Assessment.
    
    Responsibilities:
    - Collect non-sensitive Wi-Fi details from user
    - Analyze suspicious Wi-Fi names and connection types
    - Identify security warning signs
    - Generate Low/Medium/High risk assessment with explanation
    - Return structured assessment for Agent 2
    """

    AGENT_NAME = "Agent 1: Network Risk Assessment"
    PRIMARY_PROVIDER = "auto"

    SYSTEM_PROMPT = """You are a cybersecurity expert specializing in public Wi-Fi security in India.
Your role is to analyze Wi-Fi connection details provided by users and assess the risk level.

IMPORTANT RULES:
- Never ask for or use passwords, banking credentials, or sensitive personal data
- Base assessment only on user-provided non-sensitive information
- Always classify risk as exactly: LOW, MEDIUM, or HIGH
- Do not infer maliciousness from a generic SSID, network brand, or location
- Treat user-provided network details and context as unverified reports
- The deterministic rule-based risk level and score are authoritative; do not change them
- Never claim any network is 100% safe
- Reference CERT-In and I4C guidelines where applicable
- Be specific, practical and educational

FORMAT your response as JSON with these exact keys:
{
  "risk_level": "LOW|MEDIUM|HIGH",
  "risk_score": 0-100,
  "summary": "one sentence summary",
  "warning_signs": ["list", "of", "specific", "warning", "signs", "detected"],
  "risk_factors": ["specific", "risk", "factors"],
  "positive_indicators": ["any", "good", "signs"],
  "explanation": "2-3 sentence explanation for a non-technical user",
  "cert_in_reference": "relevant CERT-In advisory or guideline if applicable"
}"""

    def __init__(self, ai_manager):
        self.ai = ai_manager
        self.output_validator = OutputValidator()
        logger.info(f"{self.AGENT_NAME} initialized")

    def _pre_analyze(self, wifi_name: str, connection_type: str,
                     location: str, user_context: str) -> Dict[str, Any]:
        """
        Score only reported security properties and explicit user-reported warnings.
        """
        conn_lower = (connection_type or "").strip().lower()
        ctx_lower = (user_context or "").strip().lower()
        score = BASE_PUBLIC_WIFI_SCORE + UNKNOWN_PROVIDER_SCORE
        score_breakdown = [
            f"Public Wi-Fi baseline: +{BASE_PUBLIC_WIFI_SCORE}",
            f"Provider not supplied/unverified: +{UNKNOWN_PROVIDER_SCORE}",
        ]
        risk_factors = [
            "Public Wi-Fi is a shared medium; use encryption and avoid sensitive activity.",
            "Network operator/provider was not supplied or independently verified.",
        ]
        potential_threats = [
            "The operator's legitimacy is unknown; this alone does not establish malicious activity."
        ]

        if conn_lower in NO_AUTHENTICATION_TYPES:
            score += NO_AUTHENTICATION_SCORE
            score_breakdown.append(
                f"No Wi-Fi password authentication: +{NO_AUTHENTICATION_SCORE}"
            )
            risk_factors.append("The reported connection has no Wi-Fi password authentication.")
            potential_threats.append(
                "On an open network, other users may be able to observe unencrypted traffic."
            )
        elif conn_lower in WEAK_ENCRYPTION_TYPES:
            score += WEAK_ENCRYPTION_SCORE
            score_breakdown.append(f"Weak WEP security: +{WEAK_ENCRYPTION_SCORE}")
            risk_factors.append("WEP is a weak, outdated Wi-Fi security protocol.")
            potential_threats.append(
                "WEP's known weaknesses can make network traffic more exposed."
            )
        elif conn_lower in UNKNOWN_SECURITY_TYPES or conn_lower not in (
            MODERATE_SECURITY_TYPES | SECURED_TYPES
        ):
            score += UNKNOWN_SECURITY_SCORE
            score_breakdown.append(
                f"Unknown Wi-Fi security/authentication: +{UNKNOWN_SECURITY_SCORE}"
            )
            risk_factors.append("Wi-Fi encryption and authentication status are unknown.")
            potential_threats.append(
                "Because Wi-Fi security is unknown, the connection's protections cannot be assessed."
            )
        elif conn_lower in MODERATE_SECURITY_TYPES:
            score += MODERATE_SECURITY_SCORE
            score_breakdown.append(
                f"Shared-credential security mode: +{MODERATE_SECURITY_SCORE}"
            )
            risk_factors.append(
                f"The reported {connection_type} security mode may use credentials shared with other users."
            )

        reported_indicators = [
            indicator for indicator in HIGH_RISK_INDICATORS
            if self._is_affirmative_report(ctx_lower, indicator)
        ]
        warning_signs = [
            f"User-reported concern: {indicator} (not independently verified)."
            for indicator in reported_indicators
        ]
        if reported_indicators:
            score += REPORTED_WARNING_SCORE
            score_breakdown.append(
                f"User-reported security warning: +{REPORTED_WARNING_SCORE}"
            )
            potential_threats.append(
                "A security or unexpected-page warning was reported; it may indicate a risk, "
                "but has not been independently verified."
            )

        score = min(100, score)
        pre_risk = "HIGH" if score >= 80 else ("MEDIUM" if score >= 50 else "LOW")
        risk_factors = self._dedupe_text(risk_factors)
        warning_signs = self._dedupe_text(warning_signs)
        flags = self._dedupe_text(risk_factors + warning_signs)

        submitted = [
            f"SSID entered by the user: {wifi_name or 'not provided'}.",
            f"Security type entered by the user: {connection_type or 'not provided'}.",
            f"Location entered by the user: {location or 'not provided'}.",
            "Network operator/provider was not supplied or independently verified.",
        ]
        user_context_items = (
            [f"User-reported context: {user_context.strip()} (not independently verified)."]
            if user_context and user_context.strip() else []
        )
        return {
            "pre_risk": pre_risk,
            "risk_score": score,
            "flags": flags,
            "risk_factors": risk_factors,
            "warning_signs": warning_signs,
            "confirmed_information": submitted,
            "potential_threats": self._dedupe_text(potential_threats),
            "user_provided_assumptions": user_context_items,
            "score_breakdown": score_breakdown,
        }

    @staticmethod
    def _dedupe_text(values: List[str]) -> List[str]:
        unique = []
        seen = set()
        for value in values:
            normalized = " ".join(value.split()).casefold()
            for prefix in (
                "user-reported concern: ",
                "ai-suggested potential concern (unverified): ",
                "ai-suggested risk factor (unverified): ",
                "ai-suggested positive indicator (unverified): ",
            ):
                if normalized.startswith(prefix):
                    normalized = normalized[len(prefix):]
                    break
            normalized = normalized.replace(" (not independently verified)", "")
            normalized = normalized.strip().rstrip(".")
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique.append(value.strip())
        return unique

    @staticmethod
    def _is_affirmative_report(context: str, indicator: str) -> bool:
        start = context.find(indicator)
        if start < 0:
            return False
        prefix = context[max(0, start - 48):start]
        return not re.search(
            r"\b(?:no|not|never|without|didn't|did not|doesn't|does not|"
            r"haven't|have not|hasn't|has not)\s+(?:\w+\s+){0,3}$",
            prefix,
        )

    def _assessment_basis(self, pre: Dict[str, Any]) -> Dict[str, List[str]]:
        return {
            "confirmed_information": pre["confirmed_information"],
            "potential_threats": pre["potential_threats"],
            "user_provided_assumptions": pre["user_provided_assumptions"],
            "score_breakdown": pre["score_breakdown"],
        }

    def _format_explanation(self, model_explanation: str, pre: Dict[str, Any]) -> str:
        sections = [
            (
                f"Deterministic rule-based assessment: {pre['pre_risk']} risk "
                f"({pre['risk_score']}/100)."
            ),
            (
                "Confirmed input (the application confirms only that these values "
                "were entered; it has not independently verified them):\n- "
                + "\n- ".join(pre["confirmed_information"])
            ),
            "Score components:\n- " + "\n- ".join(pre["score_breakdown"]),
            (
                "Potential threats (possible, not confirmed or tested):\n- "
                + "\n- ".join(pre["potential_threats"])
            ),
            (
                "User-provided context/assumptions (unverified):\n- "
                + (
                    "\n- ".join(pre["user_provided_assumptions"])
                    if pre["user_provided_assumptions"] else "None supplied."
                )
            ),
            (
                "Safety recommendation: Prefer mobile data for banking and other "
                "sensitive activities. A VPN may add a layer of protection but does "
                "not guarantee security."
            ),
        ]
        if model_explanation:
            sections.append(
                f"AI interpretation (unverified): {model_explanation.strip()}"
            )
        return "\n\n".join(sections)

    def assess(
        self,
        wifi_name: str,
        connection_type: str,
        location: str,
        user_context: str,
        task_id: str = None,
    ) -> Dict[str, Any]:
        """
        Main entry point.
        Returns structured assessment dict for Supervisor → Agent 2.
        """
        logger.info(f"[{self.AGENT_NAME}] Assessing user-provided Wi-Fi details")

        pre = self._pre_analyze(wifi_name, connection_type, location, user_context)

        prompt = f"""{self.SYSTEM_PROMPT}

## Wi-Fi Connection Details Provided by User

- **Wi-Fi Network Name (SSID):** {wifi_name or 'Not provided'}
- **Connection Type / Security:** {connection_type or 'Not provided'}
- **Location:** {location or 'Not provided'}
- **Additional Context:** {user_context or 'None'}

## Pre-Analysis Flags
{chr(10).join('- ' + f for f in pre['flags']) if pre['flags'] else '- None detected by rule engine'}

Initial rule-based risk estimate: {pre['pre_risk']}
Initial rule-based risk score: {pre['risk_score']}/100

Analyze ALL provided details and return a comprehensive JSON risk assessment.
Do not treat an SSID, provider identity, or location as proof of malicious activity.
Consider this is in Bengaluru, India. Reference CERT-In advisories where relevant.
Return ONLY valid JSON, no markdown fences."""

        result = self.ai.generate(
            prompt=prompt,
            provider=self.PRIMARY_PROVIDER,
            max_tokens=800,
            temperature=0.2,
        )

        agent_output: Dict[str, Any] = {
            "agent": self.AGENT_NAME,
            "task_id": task_id,
            "status": "complete" if result["success"] else "error",
            "model_used": result.get("model", "unknown"),
            "provider": result.get("provider", "unknown"),
            "provider_status": result.get("provider_status", "unknown"),
            "provider_attempts": result.get("provider_attempts", []),
            "fallback_reason": result.get("fallback_reason"),
            "duration_ms": result.get("duration_ms", 0),
            "pre_analysis": pre,
            "input": {
                "wifi_name": wifi_name,
                "connection_type": connection_type,
                "location": location,
                "user_context": user_context,
            },
        }

        if result["success"]:
            validation = self.output_validator.validate(result.get("text", ""), pre["pre_risk"])
            if validation["valid"]:
                parsed = self._parse_llm_output(
                    validation["sanitized_text"],
                    pre,
                    wifi_name,
                    connection_type,
                )
                agent_output.update(parsed)
                if parsed.get("output_validation_fallback"):
                    agent_output["status"] = "complete_with_fallback"
                else:
                    agent_output["raw_response"] = validation["sanitized_text"]
            else:
                logger.warning(
                    "Agent1: unsafe model output rejected; using rule-based assessment"
                )
                agent_output.update(
                    self._fallback_assessment(pre, wifi_name, connection_type)
                )
                agent_output["status"] = "complete_with_fallback"
                agent_output["output_validation_issues"] = validation["issues"]
        else:
            # Graceful fallback using rule-based result
            agent_output.update(self._fallback_assessment(pre, wifi_name, connection_type))
            agent_output["error"] = result.get("error")
            agent_output["status"] = "complete_with_fallback"

        logger.info(
            f"[{self.AGENT_NAME}] Done: risk={agent_output.get('risk_level','?')} "
            f"in {agent_output.get('duration_ms',0)}ms"
        )
        return agent_output

    def _parse_llm_output(
        self,
        text: str,
        pre: Dict,
        wifi_name: str,
        connection_type: str,
    ) -> Dict[str, Any]:
        """Parse model JSON while keeping the deterministic score authoritative."""
        # Strip markdown code fences if present
        clean = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
        try:
            data = json.loads(clean)
        except (json.JSONDecodeError, ValueError):
            data = None

        list_fields = ("warning_signs", "risk_factors", "positive_indicators")
        if isinstance(data, dict):
            unexpected_fields = data.keys() - ASSESSMENT_OUTPUT_FIELDS
            if unexpected_fields:
                logger.warning("Agent1: unexpected model fields ignored")
            data = {
                field: data[field]
                for field in ASSESSMENT_OUTPUT_FIELDS
                if field in data
            }

        if (
            not isinstance(data, dict)
            or not isinstance(data.get("risk_level"), str)
            or not isinstance(data.get("risk_score"), (int, float))
            or isinstance(data.get("risk_score"), bool)
            or (
                isinstance(data.get("risk_score"), float)
                and not math.isfinite(data["risk_score"])
            )
            or not all(isinstance(data.get(key), str) for key in ("summary", "explanation"))
            or not all(
                isinstance(data.get(key), list)
                and all(isinstance(item, str) for item in data[key])
                for key in list_fields
            )
            or not isinstance(data.get("cert_in_reference"), str)
        ):
            logger.warning("Agent1: invalid model response; using rule-based assessment")
            fallback = self._fallback_assessment(pre, wifi_name, connection_type)
            fallback["output_validation_fallback"] = True
            return fallback

        risk = data["risk_level"].strip().upper()
        if risk not in ("LOW", "MEDIUM", "HIGH"):
            logger.warning("Agent1: invalid model risk level; using rule-based assessment")
            fallback = self._fallback_assessment(pre, wifi_name, connection_type)
            fallback["output_validation_fallback"] = True
            return fallback

        model_warnings = [
            f"AI-suggested potential concern (unverified): {item}"
            for item in data["warning_signs"]
        ]
        model_factors = [
            f"AI-suggested risk factor (unverified): {item}"
            for item in data["risk_factors"]
        ]
        data["warning_signs"] = self._dedupe_text(
            pre["warning_signs"] + model_warnings
        )
        data["risk_factors"] = self._dedupe_text(
            pre["risk_factors"] + model_factors
        )
        data["positive_indicators"] = self._dedupe_text([
            f"AI-suggested positive indicator (unverified): {item}"
            for item in data["positive_indicators"]
        ])
        data["risk_level"] = pre["pre_risk"]
        data["risk_score"] = pre["risk_score"]
        data["summary"] = (
            f"Rule-based assessment: {pre['pre_risk']} risk "
            f"({pre['risk_score']}/100)"
        )
        model_explanation = data["explanation"] if risk == pre["pre_risk"] else ""
        data["explanation"] = self._format_explanation(model_explanation, pre)
        data["assessment_basis"] = self._assessment_basis(pre)
        return data

    def _fallback_assessment(self, pre: Dict, wifi_name: str,
                             connection_type: str) -> Dict[str, Any]:
        """Pure rule-based fallback when AI is unavailable."""
        risk = pre["pre_risk"]
        score = pre["risk_score"]
        explanation = self._format_explanation("", pre)
        return {
            "risk_level": risk,
            "risk_score": score,
            "summary": f"Rule-based assessment: {risk} risk ({score}/100)",
            "warning_signs": pre["warning_signs"],
            "risk_factors": pre["risk_factors"],
            "positive_indicators": (
                [f"Reported security type: {connection_type} (not independently verified)."]
                if (connection_type or "").strip().lower() in
                (MODERATE_SECURITY_TYPES | SECURED_TYPES) else []
            ),
            "explanation": explanation,
            "assessment_basis": self._assessment_basis(pre),
            "cert_in_reference": "https://www.cert-in.org.in/",
        }
