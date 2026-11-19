# Architecture & Workflow Documentation
## AI-Powered Public Wi-Fi Safety Assistant

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        FRONTEND (Browser)                            │
│  HTML5 + CSS3 + JavaScript                                           │
│  ┌──────────────┐ ┌───────────────┐ ┌─────────┐ ┌──────────────┐  │
│  │  Dashboard   │ │Risk Assessment│ │AI Chat  │ │  Analytics   │  │
│  └──────────────┘ └───────────────┘ └─────────┘ └──────────────┘  │
└───────────────────────────────┬──────────────────────────────────────┘
                                │ HTTP/JSON REST API
┌───────────────────────────────▼──────────────────────────────────────┐
│                     FLASK BACKEND (Python)                           │
│                                                                      │
│  POST /api/assess    GET /api/health    POST /api/chat               │
│  GET /api/tasks      GET /api/analytics  POST /api/models/test       │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │               SUPERVISOR / ORCHESTRATOR                      │    │
│  │  • Validates inputs via Guardrails                          │    │
│  │  • Creates task record in DB                                │    │
│  │  • Coordinates agent pipeline                               │    │
│  │  • Transfers structured data between agents                │    │
│  │  • Resolves conflicts                                       │    │
│  │  • Consolidates final response                              │    │
│  └───┬────────────────────┬──────────────────────┬────────────┘    │
│      │                    │                      │                   │
│  ┌───▼─────────┐  ┌───────▼──────┐  ┌──────────▼────────┐         │
│  │   AGENT 1   │→ │   AGENT 2    │→ │      AGENT 3       │         │
│  │Network Risk │  │Device Safety │  │Incident Response   │         │
│  │Assessment   │  │& Prevention  │  │& Safety Guidance   │         │
│  │             │  │              │  │                    │         │
│  │IBM Granite  │  │IBM Granite   │  │IBM Granite         │         │
│  │(only model) │  │(only model)  │  │(only model)        │         │
│  └─────────────┘  └──────────────┘  └────────────────────┘         │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │               AI MODEL MANAGER                               │    │
│  │          IBM Granite (watsonx.ai) only                      │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌────────────┐  ┌──────────────────────┐  ┌───────────────────┐   │
│  │  Database  │  │  Guardrails/Security │  │  Config & Logging │   │
│  │SQLite/MySQL│  │  (input validation,  │  │  (.env, structlog) │  │
│  │(SQLAlchemy)│  │   PII, rate limiting)│  │                   │   │
│  └────────────┘  └──────────────────────┘  └───────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Three-Agent Collaboration Workflow

### Step 1: User Submits Assessment Request
User provides:
- Wi-Fi network name (SSID)
- Connection security type
- Location in Bengaluru
- Any additional context (unusual behavior, login portals, etc.)

### Step 2: Supervisor Validates Input
The **Supervisor/Orchestrator** runs input through **Guardrails**:
- Input length and content validation
- Prompt injection detection
- Dangerous request blocking (no network scanning)
- PII detection and redaction
- Rate limiting (30 requests/minute per IP)

### Step 3: Agent 1 – Network Risk Assessment (IBM Granite)
**Input received from Supervisor:**
- Wi-Fi name, connection type, location, user context

**Processing:**
1. Rule-based pre-analysis (suspicious SSID patterns, insecure connection types)
2. IBM Granite LLM call with structured prompt
3. JSON output parsing with fallback

**Output produced (passed to Supervisor → Agent 2):**
```json
{
  "risk_level": "HIGH|MEDIUM|LOW",
  "risk_score": 0-100,
  "warning_signs": [...],
  "risk_factors": [...],
  "explanation": "...",
  "cert_in_reference": "..."
}
```

### Step 4: Agent 2 – Device Safety & Prevention (IBM Granite)
**Input received:** Agent 1's structured risk assessment

**Processing:**
1. Retrieves risk-appropriate device checklist (10 items for HIGH risk)
2. IBM Granite LLM call for personalized device safety guidance
3. Banking guidance specific to risk level

**Output produced (passed to Supervisor → Agent 3):**
```json
{
  "device_checklist": [...],
  "banking_guidance": {...},
  "threat_explanation": "...",
  "immediate_actions": [...],
  "bengaluru_specific": "..."
}
```

### Step 5: Agent 3 – Incident Response (IBM Granite)
**Input received:** Combined Agent 1 + Agent 2 outputs

**Processing:**
1. Identifies incident scenario (phishing, account_compromise, fake_portal, general)
2. IBM Granite API call for incident response guidance
3. If Granite is unavailable, the agent uses its existing safe response path
4. Generates final comprehensive safety report

**Output produced (returned to Supervisor):**
```json
{
  "incident_scenario": "...",
  "immediate_steps": [...],
  "reporting_guidance": "...",
  "final_report": {...}
}
```

### Step 6: Supervisor Consolidates
- Merges all three agent outputs
- Resolves any conflicting risk levels (highest wins)
- Adds execution metrics
- Returns single unified JSON response to frontend

---

## AI Model Assignment Strategy

| Task | Active Model | Failure behavior |
|------|--------------|------------------|
| Risk Analysis (Agent 1) | IBM Granite | Existing deterministic risk assessment |
| Device Safety (Agent 2) | IBM Granite | Existing static checklist |
| Incident Response (Agent 3) | IBM Granite | Existing safe response path |
| Conversational Chat | IBM Granite | Safe unavailable response |
| Model Testing | IBM Granite | Disabled Groq compatibility check |

**Key principle:** IBM Granite through watsonx.ai is the only active AI provider. There is no secondary model or automatic provider fallback. A Granite failure remains a failure and is handled through the application's existing safe responses. A disabled Groq compatibility stub is retained but is never used to generate responses.

---

## Security Architecture

| Layer | Implementation |
|-------|---------------|
| Input Validation | Field length limits, HTML escaping |
| Prompt Injection | 14 regex patterns block instruction overrides |
| PII Protection | Aadhaar, PAN, credit card, password redaction |
| Dangerous Requests | Blocks network scanning, exploits, brute force |
| Rate Limiting | In-memory token bucket, 30 req/min per IP |
| Output Validation | Strips PII from AI outputs |
| API Security | Backend-only credentials, CORS configured |

---

## Database Schema

| Table | Purpose |
|-------|---------|
| `task_records` | Full task details, all agent outputs, risk level |
| `model_call_logs` | Per-call metrics: provider, duration, tokens, success |

Supports MySQL (production) with automatic SQLite fallback for development.

---

## File Structure
```
ibm hackathon/
├── backend/
│   ├── app.py                    # Flask application + all API routes
│   ├── config.py                 # Typed config from .env
│   ├── database.py               # SQLAlchemy models, DB manager
│   ├── agents/
│   │   ├── supervisor.py         # Central Supervisor/Orchestrator
│   │   ├── agent1_risk.py        # Agent 1: Network Risk Assessment
│   │   ├── agent2_device.py      # Agent 2: Device Safety & Prevention
│   │   └── agent3_incident.py    # Agent 3: Incident Response
│   ├── services/
│   │   ├── granite_service.py    # IBM watsonx.ai / Granite
│   │   ├── groq_service.py       # Legacy disabled service; not routed
│   │   └── ai_model_manager.py   # Centralized provider selection
│   └── security/
│       └── guardrails.py         # All security/validation logic
├── frontend/
│   ├── index.html                # Full SPA dashboard
│   └── static/
│       ├── css/style.css         # Cybersecurity dark theme
│       └── js/app.js             # Dashboard interactivity
├── tests/
│   ├── test_agents.py            # 17 agent + guardrail tests
│   └── test_api.py               # 12 API endpoint tests
├── docs/
│   └── architecture.md           # This document
├── requirements.txt
├── env.template                  # Copy to .env, fill in keys
└── README.md
```
