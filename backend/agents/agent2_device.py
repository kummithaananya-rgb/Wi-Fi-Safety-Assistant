"""
Agent 2: Device Safety & Prevention Agent
Receives risk assessment from Agent 1.
Provides personalized device security checklists and safety recommendations.
Uses IBM Granite as primary model.
"""
import re
import json
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Static knowledge base – checklist items and best practices
# ---------------------------------------------------------------------------

DEVICE_CHECKLIST_BY_RISK = {
    "HIGH": [
        {"id": "wifi_off", "priority": "CRITICAL", "category": "Network",
         "action": "Disconnect and use mobile data for banking and other sensitive activity",
         "why": "Avoids routing sensitive activity over this shared public Wi-Fi network"},
        {"id": "vpn", "priority": "HIGH", "category": "Network",
         "action": "If you remain on Wi-Fi, consider a trusted VPN as an additional layer",
         "why": "A VPN adds protection for part of the traffic path but does not prevent phishing, unsafe apps, or all monitoring"},
        {"id": "no_banking", "priority": "CRITICAL", "category": "Banking",
         "action": "Do NOT access banking, UPI, or financial apps while connected to this Wi-Fi",
         "why": "Disconnect and switch to mobile data if you need to transact"},
        {"id": "no_otp", "priority": "CRITICAL", "category": "Banking",
         "action": "Do NOT enter OTPs or passwords on this network",
         "why": "Man-in-the-middle attacks can capture authentication data"},
        {"id": "https_check", "priority": "HIGH", "category": "Browsing",
         "action": "Verify HTTPS (padlock icon) before entering any information",
         "why": "HTTP sites transmit data in plain text"},
        {"id": "auto_connect", "priority": "HIGH", "category": "Settings",
         "action": "Disable 'Auto-connect to open networks' in Wi-Fi settings",
         "why": "Prevents automatic connection to malicious hotspots"},
        {"id": "sharing_off", "priority": "HIGH", "category": "Settings",
         "action": "Turn off file sharing, Bluetooth, and AirDrop",
         "why": "Reduces attack surface on public networks"},
        {"id": "firewall", "priority": "MEDIUM", "category": "Security",
         "action": "Enable device firewall if available",
         "why": "Adds another layer of protection against unauthorized connections"},
        {"id": "logout", "priority": "MEDIUM", "category": "Accounts",
         "action": "Log out of all accounts after use",
         "why": "Prevents session hijacking"},
        {"id": "clear_cache", "priority": "LOW", "category": "Privacy",
         "action": "Clear browser cache and history after session",
         "why": "Removes stored credentials and browsing history"},
    ],
    "MEDIUM": [
        {"id": "vpn_recommended", "priority": "HIGH", "category": "Network",
         "action": "If you remain on Wi-Fi, a trusted VPN may add a layer of protection",
         "why": "A VPN adds a layer of protection but does not guarantee security or replace mobile data"},
        {"id": "banking_caution", "priority": "HIGH", "category": "Banking",
         "action": "Avoid banking and UPI transactions; use mobile data instead",
         "why": "WPA2-Personal networks can still be attacked with dictionary attacks"},
        {"id": "https_check", "priority": "HIGH", "category": "Browsing",
         "action": "Ensure websites show HTTPS and valid certificates",
         "why": "Verifies you're connecting to legitimate sites"},
        {"id": "auto_connect", "priority": "MEDIUM", "category": "Settings",
         "action": "Disable auto-connect to unknown networks",
         "why": "Prevents connecting to evil twin attacks"},
        {"id": "app_updates", "priority": "MEDIUM", "category": "Security",
         "action": "Ensure your device OS and apps are up to date",
         "why": "Security patches close known vulnerabilities"},
        {"id": "2fa", "priority": "MEDIUM", "category": "Accounts",
         "action": "Enable two-factor authentication on important accounts",
         "why": "Even if credentials are captured, 2FA provides an extra barrier"},
        {"id": "sharing_off", "priority": "MEDIUM", "category": "Settings",
         "action": "Turn off file sharing and unnecessary Bluetooth",
         "why": "Reduces exposure to network-based attacks"},
    ],
    "LOW": [
        {"id": "general_awareness", "priority": "MEDIUM", "category": "Awareness",
         "action": "Stay aware of your surroundings and what you're accessing",
         "why": "Physical shoulder-surfing is still a risk on public Wi-Fi"},
        {"id": "https_check", "priority": "MEDIUM", "category": "Browsing",
         "action": "Verify HTTPS for any sensitive page",
         "why": "Good security hygiene regardless of network type"},
        {"id": "vpn_good_practice", "priority": "LOW", "category": "Network",
         "action": "Consider using a VPN as a good practice",
         "why": "A VPN can protect some traffic but does not guarantee security"},
        {"id": "app_updates", "priority": "LOW", "category": "Security",
         "action": "Keep your device OS and security apps updated",
         "why": "Maintains protection against emerging threats"},
        {"id": "2fa", "priority": "LOW", "category": "Accounts",
         "action": "Enable two-factor authentication on all important accounts",
         "why": "Best practice for account security regardless of network"},
    ],
}

BANKING_GUIDANCE = {
    "HIGH": {
        "recommendation": "AVOID all banking and financial transactions",
        "details": [
            "Do NOT open SBI, HDFC, ICICI, Paytm, PhonePe, GPay or any banking app",
            "Do NOT enter UPI PIN, net banking password, or card details",
            "If you must transact, switch to your mobile data (4G/5G) first",
            "CERT-In Advisory: Never perform financial transactions on unsecured networks",
            "Report suspicious activity to your bank immediately",
        ],
        "emergency": "If you already transacted: immediately change passwords and contact your bank",
    },
    "MEDIUM": {
        "recommendation": "CAUTION – Use mobile data for banking",
        "details": [
            "Strongly prefer mobile data over this Wi-Fi for financial transactions",
            "A VPN may add a layer of protection, but it does not guarantee security or replace mobile data",
            "Only use official bank apps (not browser) for transactions",
            "Verify certificate on banking websites before proceeding",
        ],
        "emergency": "If you notice unusual activity: change passwords and contact bank",
    },
    "LOW": {
        "recommendation": "Exercise standard caution",
        "details": [
            "Prefer mobile data for banking and other sensitive activity; no Wi-Fi assessment can guarantee safety",
            "Prefer official banking apps over browser",
            "Verify HTTPS and certificate on banking websites",
            "Log out immediately after completing transactions",
            "Enable transaction SMS alerts on your bank account",
        ],
        "emergency": "Report any suspicious SMS or email alerts to your bank",
    },
}


class DeviceSafetyAgent:
    """
    Agent 2 – Device Safety & Prevention.
    
    Responsibilities:
    - Receive Agent 1 risk assessment
    - Explain cybersecurity threats relevant to the risk level
    - Provide personalized device security checklist
    - Recommend safer browsing and connectivity practices
    - Provide banking and sensitive activity safety guidance
    - Pass all findings to Agent 3
    """

    AGENT_NAME = "Agent 2: Device Safety & Prevention"
    PRIMARY_PROVIDER = "auto"

    SYSTEM_PROMPT = """You are a cybersecurity educator helping Indian commuters stay safe on public Wi-Fi.
You have received a risk assessment from the Network Risk Assessment Agent.
Your job is to provide clear, actionable, personalized safety guidance.

RULES:
- Be specific to the risk level and context provided
- Use simple language accessible to non-technical users
- Reference Indian-specific resources (CERT-In, I4C, cybercrime.gov.in)
- Never ask for passwords, credentials, or sensitive data
- Treat user-provided network details and context as unverified reports
- Do not claim a threat occurred without evidence
- Recommend mobile data for banking and other sensitive activity
- Explain that a VPN is an additional layer, not a guarantee of security
- Focus on education and prevention

Provide a JSON response with these exact keys:
{
  "threat_explanation": "plain-language explanation of relevant threats for this risk level",
  "immediate_actions": ["list of immediate steps to take RIGHT NOW"],
  "browsing_tips": ["specific safer browsing recommendations"],
  "app_security": ["app and device security recommendations"],
  "bengaluru_specific": "advice specific to Bengaluru commuters and local transport hubs",
  "recovery_note": "what to do if already connected and worried"
}"""

    def __init__(self, ai_manager):
        self.ai = ai_manager
        logger.info(f"{self.AGENT_NAME} initialized")

    def analyze(
        self,
        agent1_output: Dict[str, Any],
        task_id: str = None,
    ) -> Dict[str, Any]:
        """
        Main entry point.
        Receives Agent 1 output, returns device safety recommendations.
        """
        risk_level = agent1_output.get("risk_level", "MEDIUM")
        wifi_name = agent1_output.get("input", {}).get("wifi_name", "unknown")
        location = agent1_output.get("input", {}).get("location", "unknown")
        warning_signs = agent1_output.get("warning_signs", [])
        explanation = agent1_output.get("explanation", "")

        logger.info(
            f"[{self.AGENT_NAME}] Processing risk_level={risk_level} "
            f"wifi='{wifi_name}'"
        )

        # Build checklist from static knowledge base
        checklist = DEVICE_CHECKLIST_BY_RISK.get(
            risk_level, DEVICE_CHECKLIST_BY_RISK["MEDIUM"]
        )
        banking = BANKING_GUIDANCE.get(risk_level, BANKING_GUIDANCE["MEDIUM"])

        prompt = f"""{self.SYSTEM_PROMPT}

## Risk Assessment from Agent 1

- **Risk Level:** {risk_level}
- **Network:** {wifi_name}
- **Location:** {location}
- **Warning Signs Detected:** {', '.join(warning_signs) if warning_signs else 'None'}
- **Risk Explanation:** {explanation}

Based on this {risk_level} risk assessment for a Bengaluru commuter, provide personalized 
device safety and cybersecurity prevention guidance. 
Return ONLY valid JSON, no markdown fences."""

        result = self.ai.generate(
            prompt=prompt,
            provider=self.PRIMARY_PROVIDER,
            max_tokens=900,
            temperature=0.25,
        )

        agent_output: Dict[str, Any] = {
            "agent": self.AGENT_NAME,
            "task_id": task_id,
            "status": "complete" if result["success"] else "complete_with_fallback",
            "model_used": result.get("model", "unknown"),
            "provider": result.get("provider", "unknown"),
            "provider_status": result.get("provider_status", "unknown"),
            "provider_attempts": result.get("provider_attempts", []),
            "fallback_reason": result.get("fallback_reason"),
            "duration_ms": result.get("duration_ms", 0),
            "risk_level_received": risk_level,
            "device_checklist": checklist,
            "banking_guidance": banking,
        }

        if result["success"]:
            parsed = self._parse_llm_output(result["text"], risk_level, wifi_name, location)
            agent_output.update(parsed)
        else:
            agent_output.update(self._fallback_guidance(risk_level, wifi_name, location))
            agent_output["error"] = result.get("error")

        logger.info(
            f"[{self.AGENT_NAME}] Done: {len(checklist)} checklist items "
            f"in {agent_output.get('duration_ms',0)}ms"
        )
        return agent_output

    def _parse_llm_output(self, text: str, risk_level: str,
                          wifi_name: str, location: str) -> Dict[str, Any]:
        clean = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
        try:
            data = json.loads(clean)
            return data
        except (json.JSONDecodeError, ValueError):
            logger.warning("Agent2: LLM JSON parse failed, using fallback")
            return self._fallback_guidance(risk_level, wifi_name, location)

    def _fallback_guidance(self, risk_level: str, wifi_name: str,
                           location: str) -> Dict[str, Any]:
        threat_map = {
            "HIGH": (
                "The submitted details produce a HIGH risk score, but do not prove an attack. "
                "An open or weak network can expose unencrypted traffic; phishing or "
                "impersonation are possible threats, not confirmed events."
            ),
            "MEDIUM": (
                "The submitted details produce a MEDIUM score because provider or security "
                "information is unknown. Protections cannot be verified, but this is not "
                "evidence that the network is malicious."
            ),
            "LOW": (
                "No strong risk signal was found in the submitted details. This does not "
                "verify that the network is safe; standard public Wi-Fi precautions still apply."
            ),
        }
        return {
            "threat_explanation": threat_map.get(risk_level, threat_map["MEDIUM"]),
            "immediate_actions": [
                "Use mobile data for banking and other sensitive activities",
                "A trusted VPN may add a layer of protection on Wi-Fi, but does not guarantee security",
                "Verify HTTPS on all websites",
                "Disable file sharing and Bluetooth",
            ],
            "browsing_tips": [
                "Only visit HTTPS websites",
                "Do not click on unusual pop-ups or certificate warnings",
                "Avoid entering passwords on sites you're not 100% sure about",
            ],
            "app_security": [
                "Keep your OS and apps updated",
                "Use official apps for banking, not browser",
                "Enable screen lock and device encryption",
            ],
            "bengaluru_specific": (
                f"At {location}, use Namma Metro or KSRTC Wi-Fi only for general browsing. "
                "For sensitive activities, use Jio/Airtel 4G/5G mobile data. "
                "Report suspicious Wi-Fi to CERT-In at cert-in@cert-in.org.in"
            ),
            "recovery_note": (
                "If you're concerned about this connection: "
                "immediately change passwords for accounts accessed on this network, "
                "check bank statements for unauthorized transactions, "
                "and report incidents at cybercrime.gov.in or call 1930."
            ),
        }
