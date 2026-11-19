AI-POWERED PUBLIC WI-FI SAFETY ASSISTANT
IBM Inter-Institute Hackathon – Bengaluru Commuter Cybersecurity
================================================================

## Overview

An Agentic AI application that helps Bengaluru commuters assess public Wi-Fi risks
using IBM Granite through IBM watsonx.ai as its only active AI provider. Three
specialized AI agents collaborate under a central Supervisor/Orchestrator.

## Architecture

```
User → Frontend Dashboard → Flask Backend → Supervisor/Orchestrator
                                                      ↓
                           ┌──────────────────────────────────────────────┐
                           │  Agent 1: Network Risk Assessment             │
                           │  Agent 2: Device Safety & Prevention          │
                           │  Agent 3: Incident Response & Safety Guidance │
                           └──────────────────────────────────────────────┘
                                                      ↓
                           IBM Granite through IBM watsonx.ai
```

## Three AI Agents

| Agent | Role | AI Model |
|-------|------|----------|
| Agent 1: Network Risk Assessment | Analyzes Wi-Fi context, generates Low/Medium/High risk | IBM Granite |
| Agent 2: Device Safety & Prevention | Device checklist, browsing safety, banking guidance | IBM Granite |
| Agent 3: Incident Response & Safety Guidance | Phishing guidance, incident reporting, final report | IBM Granite |

## Quick Start

### Prerequisites
- Python 3.10+
- IBM Cloud account with watsonx.ai access
- watsonx.ai project ID

### Installation

```bash
# 1. Clone / navigate to project
cd "ibm hackathon"

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
copy .env.example .env
# Edit .env with your actual API keys

# 5. Run the application
python backend/app.py
```

### Access
Open `http://localhost:5000` in your browser.

## IBM watsonx.ai Configuration

1. **IBM Cloud API Key** – https://cloud.ibm.com/iam/apikeys
2. **watsonx.ai Project ID** – https://dataplatform.cloud.ibm.com/projects/

## Project Structure

```
ibm hackathon/
├── backend/
│   ├── app.py                    # Flask application entry point
│   ├── config.py                 # Configuration management
│   ├── database.py               # Database models and connection
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── supervisor.py         # Central Supervisor/Orchestrator
│   │   ├── agent1_risk.py        # Network Risk Assessment Agent
│   │   ├── agent2_device.py      # Device Safety & Prevention Agent
│   │   └── agent3_incident.py    # Incident Response Agent
│   ├── services/
│   │   ├── __init__.py
│   │   ├── granite_service.py    # IBM Granite/watsonx.ai
│   │   ├── groq_service.py       # Legacy disabled provider service
│   │   └── ai_model_manager.py   # Centralized AI model selection
│   └── security/
│       ├── __init__.py
│       └── guardrails.py         # Input validation, PII protection, rate limiting
├── frontend/
│   ├── index.html                # Main dashboard
│   └── static/
│       ├── css/style.css         # Cybersecurity-themed styles
│       └── js/app.js             # Dashboard interactivity
├── tests/
│   ├── test_agents.py
│   ├── test_ai_services.py
│   └── test_api.py
├── docs/
│   └── architecture.md
├── requirements.txt
├── .env.example
└── README.md
```

## Demo Workflow

1. User enters Wi-Fi name and connection details on dashboard
2. Supervisor assigns task to Agent 1 (Risk Assessment)
3. Agent 1 analyzes via IBM Granite → produces risk level + findings
4. Supervisor passes findings to Agent 2 (Device Safety)
5. Agent 2 generates device checklist via IBM Granite
6. Supervisor passes all context to Agent 3 (Incident Response)
7. Agent 3 produces the final safety report using IBM Granite
8. Supervisor consolidates → unified response displayed on dashboard

## Safety Restrictions

- Never scans, probes or tests real networks
- Never collects passwords or banking credentials
- Assessment based only on user-provided information
- Always communicates uncertainty
- Guides users to official resources (CERT-In, I4C)

## Official Resources

- CERT-In: https://www.cert-in.org.in/
- I4C: https://i4c.mha.gov.in/
- Cyber Crime Portal: https://cybercrime.gov.in/
