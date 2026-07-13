from fastapi.testclient import TestClient

from api.inference import llm_engine
from api.main import app

client = TestClient(app)


def test_health_reports_model_configuration():
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert "model" in payload
    assert payload["live_model_enabled"] is False


def test_generate_uses_safe_fallback_without_live_model():
    response = client.post(
        "/generate",
        json={"prompt": "Explain hierarchical reasoning.", "max_tokens": 16},
    )

    assert response.status_code == 200
    assert "Live model loading is disabled" in response.json()["output"]


def test_generate_rejects_empty_prompt():
    response = client.post("/generate", json={"prompt": "", "max_tokens": 16})

    assert response.status_code == 422


def test_llm_engine_empty_prompt_guard():
    assert llm_engine.generate("   ") == "Prompt cannot be empty."
