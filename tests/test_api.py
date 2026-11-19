"""
Flask API endpoint tests.
Run: pytest tests/test_api.py -v
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def app_client():
    """Flask test client with mocked dependencies."""
    # Patch all external AI services before importing app
    with patch('services.granite_service.GraniteService._init'), \
         patch('services.gemini_service.GeminiService._init'), \
         patch('services.groq_service.GroqService._init'), \
         patch('database.DatabaseManager._init_engine'):

        import app as flask_app
        flask_app.app.config['TESTING'] = True
        flask_app.app.config['SECRET_KEY'] = 'test-secret'

        # Mock the supervisor
        flask_app.supervisor = MagicMock()
        flask_app.supervisor.process_request.return_value = {
            "success": True,
            "task_id": "test-123",
            "risk_level": "HIGH",
            "risk_score": 85,
            "risk_summary": "High risk network",
            "risk_explanation": "Open network with no encryption",
            "warning_signs": ["Open network"],
            "device_checklist": [],
            "banking_guidance": {"recommendation": "AVOID all banking"},
            "final_report": {"report_id": "test-123", "risk_level": "HIGH"},
            "agent_statuses": {
                "agent1": {"name": "Agent 1", "status": "complete", "model": "granite"},
                "agent2": {"name": "Agent 2", "status": "complete", "model": "granite"},
                "agent3": {"name": "Agent 3", "status": "complete", "model": "groq"},
            },
            "execution_log": [],
            "total_duration_ms": 3200,
        }

        # Mock AI manager
        flask_app.ai_manager = MagicMock()
        flask_app.ai_manager.status.return_value = {
            "granite": {"available": False, "model": "ibm/granite-4-h-small"},
            "gemini": {"available": False, "model": "gemini-1.5-flash"},
            "groq": {"available": False, "model": "llama-3.1-70b-versatile"},
        }
        flask_app.ai_manager.generate.return_value = {
            "success": True, "text": "Test response", "model": "test", "provider": "test",
            "duration_ms": 100, "prompt_tokens": 10, "completion_tokens": 5, "error": None,
        }
        flask_app.ai_manager.test_all.return_value = {
            "granite": {"success": False, "error": "Not configured", "test": True},
            "gemini": {"success": False, "error": "Not configured", "test": True},
            "groq": {"success": False, "error": "Not configured", "test": True},
        }

        # Mock DB
        flask_app.db = MagicMock()
        flask_app.db.get_recent_tasks.return_value = []
        flask_app.db.get_task.return_value = None
        flask_app.db.get_analytics.return_value = {
            "total_tasks": 0, "complete": 0, "errors": 0,
            "risk_distribution": {}, "provider_stats": {},
        }

        with flask_app.app.test_client() as client:
            yield client


class TestHealthEndpoint:
    def test_health_ok(self, app_client):
        r = app_client.get('/api/health')
        assert r.status_code == 200
        data = r.get_json()
        assert data['status'] == 'healthy'
        assert 'models' in data
        assert 'agents' in data

    def test_health_has_three_agents(self, app_client):
        r = app_client.get('/api/health')
        data = r.get_json()
        assert 'agent1' in data['agents']
        assert 'agent2' in data['agents']
        assert 'agent3' in data['agents']
        assert 'supervisor' in data['agents']


class TestAssessEndpoint:
    def test_assess_requires_wifi_name(self, app_client):
        r = app_client.post('/api/assess',
                            json={},
                            content_type='application/json')
        assert r.status_code == 400

    def test_assess_valid_input(self, app_client):
        r = app_client.post('/api/assess',
                            json={
                                "wifi_name": "TestNet",
                                "connection_type": "open",
                                "location": "Metro",
                                "user_context": "Normal use"
                            },
                            content_type='application/json')
        assert r.status_code == 200
        data = r.get_json()
        assert data['success'] is True
        assert 'risk_level' in data

    def test_assess_returns_agent_statuses(self, app_client):
        r = app_client.post('/api/assess',
                            json={"wifi_name": "TestNet"},
                            content_type='application/json')
        data = r.get_json()
        assert 'agent_statuses' in data


class TestModelEndpoints:
    def test_model_status(self, app_client):
        r = app_client.get('/api/models/status')
        assert r.status_code == 200
        data = r.get_json()
        assert 'providers' in data
        assert data['primary'] == 'IBM Granite (watsonx.ai)'

    def test_model_test_all(self, app_client):
        r = app_client.post('/api/models/test',
                            json={"provider": "all"},
                            content_type='application/json')
        assert r.status_code == 200
        data = r.get_json()
        assert 'results' in data
        assert 'granite' in data['results']
        assert 'gemini' in data['results']
        assert 'groq' in data['results']


class TestTaskEndpoints:
    def test_get_tasks_empty(self, app_client):
        r = app_client.get('/api/tasks')
        assert r.status_code == 200
        data = r.get_json()
        assert data['success'] is True
        assert 'tasks' in data

    def test_get_task_not_found(self, app_client):
        r = app_client.get('/api/tasks/nonexistent-id')
        assert r.status_code == 404


class TestChatEndpoint:
    def test_chat_requires_message(self, app_client):
        r = app_client.post('/api/chat',
                            json={},
                            content_type='application/json')
        assert r.status_code == 400

    def test_chat_valid_message(self, app_client):
        r = app_client.post('/api/chat',
                            json={"message": "Is public WiFi safe?"},
                            content_type='application/json')
        assert r.status_code == 200
        data = r.get_json()
        assert 'response' in data


class TestAnalyticsEndpoint:
    def test_analytics_ok(self, app_client):
        r = app_client.get('/api/analytics')
        assert r.status_code == 200
        data = r.get_json()
        assert data['success'] is True
        assert 'analytics' in data
