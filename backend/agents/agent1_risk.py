"""
Agent 1: Network Risk Assessment Agent
Analyzes user-provided Wi-Fi connection details.
Produces a structured risk assessment (LOW / MEDIUM / HIGH).
Uses IBM Granite as primary model.
"""
import re
import json
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Suspicious pattern library (synthetic + CERT-In / I4C advisories)
# ---------------------------------------------------------------------------
SUSPICIOUS_NAME_PATTERNS: List[str] = [
    r"free.*wifi", r"public.*wifi", r"open.*wifi", r"guest.*wifi",
    r"airport.*free", r"train.*free", r"metro.*free", r"namma.*metro.*free",
    r"ksrtc.*free", r"bangalore.*free", r"bengaluru.*free",
    r"starbucks.*free", r"mcdonald.*free", r"hotel.*free",
    r"bank.*wifi", r"secure.*bank", r"sbi.*wifi", r"hdfc.*wifi",
    r"atm.*wifi", r"payment.*wifi",
    r"[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}",   # IP-as-SSID
    r"linksys", r"netgear", r"dlink", r"tp-?link",         # default router names
    r"xfinity", r"optimum",                                # US ISP (unusual in BLR)
    r"evil\s*twin", r"honeypot",
]
SUSPICIOUS_PATTERNS_COMPILED = [
    re.compile(p, re.IGNORECASE) for p in SUSPICIOUS_NAME_PATTERNS
]

RISKY_CONNECTION_TYPES = {"open", "unsecured", "none", "wep"}
MEDIUM_RISK_TYPES = {"wpa", "wpa2-personal"}
LOW_RISK_TYPES = {"wpa2-enterprise", "wpa3", "wpa3-enterprise", "802.1x"}

RISKY_LOCATIONS = [
    "railway station", "metro station", "bus stand", "airport",
    "café", "coffee shop", "mall", "market", "public park",
]

HIGH_RISK_INDICATORS = [
    "asks for login", "login portal", "redirects", "captive portal",
    "asks for personal info", "asks for phone number", "asks for email",
    "certificate warning", "ssl error", "https warning",
    "not familiar", "unfamiliar", "strange", "unusual",
    "same name as", "same as", "duplicate",
]


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
    PRIMARY_PROVIDER = "granite"

    SYSTEM_PROMPT = """You are a cybersecurity expert specializing in public Wi-Fi security in India.
Your role is to analyze Wi-Fi connection details provided by users and assess the risk level.

IMPORTANT RULES:
- Never ask for or use passwords, banking credentials, or sensitive personal data
- Base assessment only on user-provided non-sensitive information
- Always classify risk as exactly: LOW, MEDIUM, or HIGH
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
        logger.info(f"{self.AGENT_NAME} initialized")

    def _pre_analyze(self, wifi_name: str, connection_type: str,
                     location: str, user_context: str) -> Dict[str, Any]:
        """
        Rule-based pre-analysis before sending to LLM.
        Detects obvious patterns quickly and cheaply.
        """
        flags: List[str] = []
        pre_risk = "LOW"

        wifi_lower = (wifi_name or "").lower()
        conn_lower = (connection_type or "").lower()
        ctx_lower = (user_context or "").lower()

        # 1. Suspicious SSID patterns
        for pat in SUSPICIOUS_PATTERNS_COMPILED:
            if pat.search(wifi_lower):
                flags.append(f"Suspicious Wi-Fi name pattern: '{wifi_name}'")
                pre_risk = "MEDIUM"
                break

        # 2. Connection security type
        if conn_lower in RISKY_CONNECTION_TYPES:
            flags.append(f"Insecure connection type: {connection_type}")
            pre_risk = "HIGH" if pre_risk != "HIGH" else "HIGH"
        elif conn_lower in MEDIUM_RISK_TYPES:
            flags.append(f"Moderate security: {connection_type}")
            if pre_risk == "LOW":
                pre_risk = "LOW"  # WPA2-personal alone is acceptable

        # 3. Location risk
        for risky_loc in RISKY_LOCATIONS:
            if risky_loc in (location or "").lower():
                flags.append(f"High-risk public location: {location}")
                if pre_risk == "LOW":
                    pre_risk = "MEDIUM"
                break

        # 4. High-risk context indicators
        for indicator in HIGH_RISK_INDICATORS:
            if indicator in ctx_lower:
                flags.append(f"High-risk indicator: {indicator}")
                pre_risk = "HIGH"

        # 5. No encryption = always HIGH
        if conn_lower in {"open", "none", "unsecured", "wep"}:
            pre_risk = "HIGH"

        return {"pre_risk": pre_risk, "flags": flags}

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
        logger.info(
            f"[{self.AGENT_NAME}] Assessing: wifi='{wifi_name}' "
            f"type='{connection_type}' location='{location}'"
        )

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

Analyze ALL provided details and return a comprehensive JSON risk assessment.
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
            parsed = self._parse_llm_output(result["text"], pre)
            agent_output.update(parsed)
            agent_output["raw_response"] = result["text"]
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

    def _parse_llm_output(self, text: str, pre: Dict) -> Dict[str, Any]:
        """Parse JSON from LLM response; fallback to rule-based if malformed."""
        # Strip markdown code fences if present
        clean = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
        try:
            data = json.loads(clean)
            # Normalize risk_level
            risk = str(data.get("risk_level", pre["pre_risk"])).upper()
            if risk not in ("LOW", "MEDIUM", "HIGH"):
                risk = pre["pre_risk"]
            data["risk_level"] = risk
            return data
        except (json.JSONDecodeError, ValueError):
            logger.warning("Agent1: LLM JSON parse failed, extracting partial data")
            return self._extract_partial(text, pre)

    def _extract_partial(self, text: str, pre: Dict) -> Dict[str, Any]:
        """Heuristic extraction if JSON is malformed."""
        risk = pre["pre_risk"]
        for level in ("HIGH", "MEDIUM", "LOW"):
            if level in text.upper():
                risk = level
                break
        return {
            "risk_level": risk,
            "risk_score": {"HIGH": 85, "MEDIUM": 50, "LOW": 20}[risk],
            "summary": f"Risk assessment: {risk}",
            "warning_signs": pre["flags"],
            "risk_factors": pre["flags"],
            "positive_indicators": [],
            "explanation": text[:400] if text else "Assessment completed.",
            "cert_in_reference": "Refer to https://www.cert-in.org.in/ for advisories",
        }

    def _fallback_assessment(self, pre: Dict, wifi_name: str,
                             connection_type: str) -> Dict[str, Any]:
        """Pure rule-based fallback when AI is unavailable."""
        risk = pre["pre_risk"]
        score = {"HIGH": 80, "MEDIUM": 55, "LOW": 20}[risk]
        explanations = {
            "HIGH": (
                f"The Wi-Fi network '{wifi_name}' shows multiple high-risk indicators "
                "including insecure connection type or suspicious patterns. "
                "Avoid sensitive activities on this network."
            ),
            "MEDIUM": (
                f"The Wi-Fi network '{wifi_name}' has some concerning characteristics. "
                "Exercise caution and avoid banking or sensitive transactions."
            ),
            "LOW": (
                f"The Wi-Fi network '{wifi_name}' appears relatively safe based on "
                "provided information, but public Wi-Fi always carries some risk."
            ),
        }
        return {
            "risk_level": risk,
            "risk_score": score,
            "summary": f"Rule-based assessment: {risk} risk",
            "warning_signs": pre["flags"],
            "risk_factors": pre["flags"],
            "positive_indicators": (
                ["Secured connection type"] if connection_type.upper() in
                {t.upper() for t in LOW_RISK_TYPES} else []
            ),
            "explanation": explanations[risk],
            "cert_in_reference": "https://www.cert-in.org.in/",
        }
