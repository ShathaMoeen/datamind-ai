from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check() -> None:
    """The health endpoint should report a healthy service."""

    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "service": "DataMind AI",
        "version": "0.1.0",
    }
    