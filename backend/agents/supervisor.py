"""
Central Supervisor Agent / Orchestrator
Coordinates all three specialized agents, transfers information between them,
resolves conflicting recommendations, and returns one unified response.
"""
import time
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone

from agents.agent2_device import BANKING_GUIDANCE, DEVICE_CHECKLIST_BY_RISK

logger = logging.getLogger(__name__)

AGENT2_OUTPUT_SCHEMA = {
    "threat_explanation": str,
    "immediate_actions": list,
    "browsing_tips": list,
    "app_security": list,
    "bengaluru_specific": str,
    "recovery_note": str,
}

AGENT3_OUTPUT_SCHEMA = {
    "incident_scenario": str,
    "severity": str,
    "situation_summary": str,
    "immediate_steps": list,
    "if_credentials_entered": str,
    "reporting_guidance": str,
    "prevention_summary": str,
    "bengaluru_transit_tip": str,
}

AGENT3_ALLOWED_VALUES = {
    "incident_scenario": {
        "phishing", "account_compromise", "fake_portal",
        "suspicious_network", "general",
    },
    "severity": {"CRITICAL", "HIGH", "MEDIUM", "LOW"},
}


class SupervisorOrchestrator:
    """
    Central Supervisor – orchestrates the three-agent pipeline.
    
    Workflow:
    1. Validate and understand the user request
    2. Assign task to Agent 1 (Risk Assessment) → IBM Granite
    3. Pass Agent 1 output to Agent 2 (Device Safety) → IBM Granite
    4. Pass Agents 1+2 output to Agent 3 (Incident Response) → Granite
    5. Consolidate all outputs into one unified response
    6. Log metrics and persist to database
    """

    def __init__(self, ai_manager, db_manager, guardrails, agents: Dict):
        """
        agents: {
            'agent1': NetworkRiskAgent,
            'agent2': DeviceSafetyAgent,
            'agent3': IncidentResponseAgent,
        }
        """
        self.ai = ai_manager
        self.db = db_manager
        self.guardrails = guardrails
        self.agent1 = agents["agent1"]
        self.agent2 = agents["agent2"]
        self.agent3 = agents["agent3"]
        logger.info("Supervisor/Orchestrator initialized with 3 agents")

    def process_request(
        self,
        wifi_name: str,
        connection_type: str,
        location: str,
        user_context: str,
        client_ip: str = "unknown",
    ) -> Dict[str, Any]:
        """
        Full pipeline execution.
        Returns unified response dict for the frontend.
        """
        t_start = time.time()
        task_id = None
        execution_log = []

        def log_step(step: str, status: str, details: str = ""):
            entry = {
                "step": step,
                "status": status,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "details": details,
            }
            execution_log.append(entry)
            logger.info(f"[Supervisor] {step}: {status} {details}")

        try:
            # ── Step 0: Guardrail checks ──────────────────────────────────
            log_step("Guardrail Check", "running")
            guard_result = self.guardrails.validate_input(
                wifi_name=wifi_name,
                connection_type=connection_type,
                location=location,
                user_context=user_context,
                client_ip=client_ip,
            )
            if not guard_result["allowed"]:
                log_step("Guardrail Check", "blocked", guard_result["reason"])
                return self._blocked_response(guard_result, execution_log)
            sanitized = guard_result["sanitized"]
            wifi_name = sanitized["wifi_name"]
            connection_type = sanitized["connection_type"]
            location = sanitized["location"]
            user_context = sanitized["user_context"]
            log_step("Guardrail Check", "passed",
                     f"flags={guard_result.get('flags', [])}")

            # ── Step 1: Create task record ────────────────────────────────
            log_step("Task Creation", "running")
            task_id = self.db.create_task(
                wifi_name=wifi_name,
                connection_type=connection_type,
                location=location,
                user_context=user_context,
            )
            self.db.update_task(task_id, status="running")
            log_step("Task Creation", "complete", f"task_id={task_id}")

            # ── Step 2: Agent 1 – Network Risk Assessment ─────────────────
            log_step("Agent 1: Network Risk Assessment", "running")
            t1 = time.time()
            agent1_output = self.agent1.assess(
                wifi_name=wifi_name,
                connection_type=connection_type,
                location=location,
                user_context=user_context,
                task_id=task_id,
            )
            agent1_ms = round((time.time() - t1) * 1000, 1)
            self.db.update_task(task_id,
                                agent1_output=agent1_output,
                                risk_level=agent1_output.get("risk_level", "UNKNOWN"),
                                agent1_duration_ms=agent1_ms)
            self._log_model_call(task_id, agent1_output, "Agent 1")
            log_step("Agent 1: Network Risk Assessment", "complete",
                     f"risk={agent1_output.get('risk_level')} in {agent1_ms}ms")

            # ── Step 3: Agent 2 – Device Safety ───────────────────────────
            log_step("Agent 2: Device Safety & Prevention", "running")
            t2 = time.time()
            try:
                agent2_output = self.agent2.analyze(
                    agent1_output=agent1_output,
                    task_id=task_id,
                )
            except (TypeError, ValueError):
                logger.warning("Agent 2: malformed output; using static guidance")
                agent2_output = {}
            if not isinstance(agent2_output, dict):
                agent2_output = {}
            if not self._validate_output_fields(
                agent2_output,
                AGENT2_OUTPUT_SCHEMA,
                agent1_output.get("risk_level", "UNKNOWN"),
            ):
                logger.warning("Agent 2: unsafe output rejected; using static guidance")
                agent2_output.update(self.agent2._fallback_guidance(
                    agent1_output.get("risk_level", "MEDIUM"),
                    agent1_output.get("input", {}).get("wifi_name", "unknown"),
                    agent1_output.get("input", {}).get("location", "unknown"),
                ))
                agent2_output["status"] = "complete_with_fallback"
            risk_level = agent1_output.get("risk_level", "MEDIUM")
            agent2_output["device_checklist"] = DEVICE_CHECKLIST_BY_RISK.get(
                risk_level, DEVICE_CHECKLIST_BY_RISK["MEDIUM"]
            )
            agent2_output["banking_guidance"] = BANKING_GUIDANCE.get(
                risk_level, BANKING_GUIDANCE["MEDIUM"]
            )
            agent2_output["agent"] = self.agent2.AGENT_NAME
            agent2_output["task_id"] = task_id
            agent2_ms = round((time.time() - t2) * 1000, 1)
            self.db.update_task(task_id,
                                agent2_output=agent2_output,
                                agent2_duration_ms=agent2_ms)
            self._log_model_call(task_id, agent2_output, "Agent 2")
            log_step("Agent 2: Device Safety & Prevention", "complete",
                     f"checklist_items={len(agent2_output.get('device_checklist',[]))} "
                     f"in {agent2_ms}ms")

            # ── Step 4: Agent 3 – Incident Response ───────────────────────
            log_step("Agent 3: Incident Response & Safety Guidance", "running")
            t3 = time.time()
            try:
                agent3_output = self.agent3.respond(
                    agent1_output=agent1_output,
                    agent2_output=agent2_output,
                    task_id=task_id,
                )
            except (TypeError, ValueError):
                logger.warning("Agent 3: malformed output; using static incident guidance")
                agent3_output = {}
            if not isinstance(agent3_output, dict):
                agent3_output = {}
            if not self._validate_output_fields(
                agent3_output,
                AGENT3_OUTPUT_SCHEMA,
                agent1_output.get("risk_level", "UNKNOWN"),
                allowed_values=AGENT3_ALLOWED_VALUES,
            ):
                logger.warning(
                    "Agent 3: unsafe output rejected; using static incident guidance"
                )
                scenario = agent3_output.get("identified_scenario", "general")
                if (
                    not isinstance(scenario, str)
                    or scenario not in AGENT3_ALLOWED_VALUES["incident_scenario"]
                ):
                    scenario = "general"
                risk_level = agent1_output.get("risk_level", "MEDIUM")
                agent3_output.update(self.agent3._fallback_response(
                    scenario,
                    risk_level,
                    agent1_output.get("input", {}).get("wifi_name", "unknown"),
                    agent1_output.get("input", {}).get("location", "unknown"),
                ))
                agent3_output["status"] = "complete_with_fallback"
                agent3_output["final_report"] = self.agent3._generate_final_report(
                    agent1_output,
                    agent2_output,
                    agent3_output,
                    task_id,
                )
            agent3_output["agent"] = self.agent3.AGENT_NAME
            agent3_output["task_id"] = task_id
            agent3_ms = round((time.time() - t3) * 1000, 1)
            self.db.update_task(task_id,
                                agent3_output=agent3_output,
                                agent3_duration_ms=agent3_ms)
            self._log_model_call(task_id, agent3_output, "Agent 3")
            log_step("Agent 3: Incident Response & Safety Guidance", "complete",
                     f"scenario={agent3_output.get('identified_scenario')} "
                     f"in {agent3_ms}ms")

            # ── Step 5: Consolidate ────────────────────────────────────────
            log_step("Supervisor Consolidation", "running")
            unified = self._consolidate(
                agent1_output, agent2_output, agent3_output,
                task_id, execution_log, t_start
            )
            total_ms = round((time.time() - t_start) * 1000, 1)

            models_used = list({
                agent1_output.get("model_used"),
                agent2_output.get("model_used"),
                agent3_output.get("model_used"),
            } - {None, ""})

            self.db.update_task(
                task_id,
                status="complete",
                supervisor_output=unified,
                total_duration_ms=total_ms,
                models_used=models_used,
                guardrail_flags=guard_result.get("flags", []),
            )
            log_step("Supervisor Consolidation", "complete", f"total={total_ms}ms")

            unified["execution_log"] = execution_log
            unified["total_duration_ms"] = total_ms
            return unified

        except Exception as e:
            logger.exception(f"[Supervisor] Pipeline error: {e}")
            if task_id:
                self.db.update_task(task_id, status="error",
                                    error_message="Pipeline processing failed")
            log_step("Pipeline", "error", "Pipeline processing failed")
            return self._error_response(task_id, execution_log)

    # ── Private helpers ────────────────────────────────────────────────────

    def _validate_output_fields(
        self,
        output: Any,
        schema: Dict[str, type],
        risk_level: str,
        allowed_values: Optional[Dict[str, set]] = None,
    ) -> bool:
        if not isinstance(output, dict):
            return False

        for field, expected_type in schema.items():
            if field not in output:
                return False

            value = output[field]
            if expected_type is str:
                if not isinstance(value, str):
                    return False
                if allowed_values and field in allowed_values:
                    value = value.strip().upper() if field == "severity" else value.strip().lower()
                    if value not in allowed_values[field]:
                        return False
                    output[field] = value
                validation = self.guardrails.validate_output(value, risk_level)
                if not validation["valid"]:
                    return False
                output[field] = validation["sanitized_text"]
            elif expected_type is list:
                if not isinstance(value, list) or not all(
                    isinstance(item, str) for item in value
                ):
                    return False
                sanitized_items = []
                for item in value:
                    validation = self.guardrails.validate_output(item, risk_level)
                    if not validation["valid"]:
                        return False
                    sanitized_items.append(validation["sanitized_text"])
                output[field] = sanitized_items
            elif not isinstance(value, expected_type):
                return False
        return True

    def _consolidate(
        self,
        a1: Dict, a2: Dict, a3: Dict,
        task_id: str, execution_log: list, t_start: float
    ) -> Dict[str, Any]:
        """Merge outputs from all three agents into one clean response."""
        risk = a1.get("risk_level", "UNKNOWN")
        checklist = a2.get("device_checklist", [])
        final_report = dict(a3.get("final_report", {}))
        warning_signs = self._dedupe_strings(a1.get("warning_signs", []))
        risk_factors = self._dedupe_strings(a1.get("risk_factors", []))
        positive_indicators = self._dedupe_strings(a1.get("positive_indicators", []))
        final_report["warning_signs"] = self._dedupe_strings(
            final_report.get("warning_signs", warning_signs)
        )

        severity_rank = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1, "UNKNOWN": 0}
        incident_severity = a3.get("severity", risk)
        if severity_rank.get(incident_severity, 0) < severity_rank.get(risk, 0):
            incident_severity = risk

        disclaimer = self.guardrails.output_validator.REQUIRED_DISCLAIMER
        risk_explanation = a1.get("explanation", "")
        if disclaimer not in risk_explanation:
            risk_explanation = f"{risk_explanation}\n\n{disclaimer}".strip()
        final_report["risk_level"] = risk
        final_report["disclaimer"] = disclaimer

        return {
            "success": True,
            "task_id": task_id,
            "risk_level": risk,
            "risk_score": a1.get("risk_score", 0),
            "risk_summary": a1.get("summary", ""),
            "risk_explanation": risk_explanation,
            "disclaimer": disclaimer,
            "warning_signs": warning_signs,
            "risk_factors": risk_factors,
            "positive_indicators": positive_indicators,
            "assessment_basis": (
                a1.get("assessment_basis", {})
                if isinstance(a1.get("assessment_basis", {}), dict) else {}
            ),
            "cert_in_reference": a1.get("cert_in_reference", ""),

            "device_checklist": checklist,
            "banking_guidance": a2.get("banking_guidance", {}),
            "threat_explanation": a2.get("threat_explanation", ""),
            "immediate_actions": a2.get("immediate_actions", []),
            "browsing_tips": a2.get("browsing_tips", []),
            "app_security": a2.get("app_security", []),
            "bengaluru_specific": a2.get("bengaluru_specific", ""),
            "recovery_note": a2.get("recovery_note", ""),

            "incident_scenario": a3.get("incident_scenario",
                                        a3.get("identified_scenario", "general")),
            "incident_severity": incident_severity,
            "situation_summary": a3.get("situation_summary", ""),
            "incident_immediate_steps": a3.get("immediate_steps",
                                               a3.get("immediate_response_steps", [])),
            "if_credentials_entered": a3.get("if_credentials_entered", ""),
            "reporting_guidance": a3.get("reporting_guidance", ""),
            "prevention_summary": a3.get("prevention_summary", ""),
            "bengaluru_transit_tip": a3.get("bengaluru_transit_tip", ""),
            "reporting_resources": a3.get("reporting_resources", []),

            "final_report": final_report,

            "agent_statuses": {
                "agent1": {
                    "name": a1.get("agent", "Agent 1"),
                    "status": a1.get("status", "complete"),
                    "model": a1.get("model_used", ""),
                    "provider": a1.get("provider", ""),
                    "provider_status": a1.get("provider_status", "unknown"),
                    "duration_ms": a1.get("duration_ms", 0),
                },
                "agent2": {
                    "name": a2.get("agent", "Agent 2"),
                    "status": a2.get("status", "complete"),
                    "model": a2.get("model_used", ""),
                    "provider": a2.get("provider", ""),
                    "provider_status": a2.get("provider_status", "unknown"),
                    "duration_ms": a2.get("duration_ms", 0),
                },
                "agent3": {
                    "name": a3.get("agent", "Agent 3"),
                    "status": a3.get("status", "complete"),
                    "model": a3.get("model_used", ""),
                    "provider": a3.get("provider", ""),
                    "provider_status": a3.get("provider_status", "unknown"),
                    "duration_ms": a3.get("duration_ms", 0),
                },
            },
        }

    @staticmethod
    def _dedupe_strings(values: Any) -> list:
        if not isinstance(values, list):
            return []
        unique = []
        seen = set()
        for value in values:
            if not isinstance(value, str):
                continue
            normalized = " ".join(value.split()).casefold()
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique.append(value.strip())
        return unique

    def _log_model_call(self, task_id: str, agent_output: Dict, agent_name: str):
        """Log AI model call metrics to database."""
        try:
            self.db.log_model_call(
                task_id=task_id,
                provider=agent_output.get("provider", "unknown"),
                model_id=agent_output.get("model_used", "unknown"),
                agent_name=agent_name,
                duration_ms=agent_output.get("duration_ms", 0),
                success=agent_output.get("status") not in ("error",),
                error=agent_output.get("error"),
            )
        except Exception as e:
            logger.warning(f"Failed to log model call: {e}")

    def _blocked_response(self, guard_result: Dict,
                          execution_log: list) -> Dict[str, Any]:
        return {
            "success": False,
            "blocked": True,
            "reason": guard_result.get("reason", "Request blocked by guardrails"),
            "flags": guard_result.get("flags", []),
            "risk_level": "UNKNOWN",
            "execution_log": execution_log,
        }

    def _error_response(self, task_id: Optional[str],
                        execution_log: list) -> Dict[str, Any]:
        return {
            "success": False,
            "error": "Assessment could not be completed. Please try again.",
            "task_id": task_id,
            "risk_level": "UNKNOWN",
            "execution_log": execution_log,
            "reporting_resources": [
                {
                    "name": "Cyber Crime Helpline",
                    "helpline": "1930",
                    "url": "https://cybercrime.gov.in/",
                }
            ],
        }
