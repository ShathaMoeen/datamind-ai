"""Tests for the lightweight browser frontend."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_frontend_is_served_at_root() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "DataMind AI" in response.text
    assert "Upload dataset" in response.text
    assert "language-toggle" in response.text


def test_frontend_static_assets_are_served() -> None:
    response = client.get("/static/styles.css")

    assert response.status_code == 200
    assert "--accent" in response.text


def test_frontend_javascript_contains_arabic_translation_and_rtl_support() -> None:
    response = client.get("/static/app.js")

    assert response.status_code == 200
    assert "رفع مجموعة بيانات" in response.text
    assert 'document.documentElement.dir = language === "ar" ? "rtl" : "ltr"' in response.text


def test_frontend_connects_analysis_button_to_api() -> None:
    html = client.get("/")
    javascript = client.get("/static/app.js")

    assert 'id="analysis-button"' in html.text
    assert 'id="analysis-button" type="button"' in html.text
    assert "`${api}/analysis/run`" in javascript.text
