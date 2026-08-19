"""Integration tests for PDF upload endpoints."""

from fastapi.testclient import TestClient

from app.api.routes.documents import get_document_service
from app.main import app
from app.services.document_service import DocumentService


def test_pdf_upload_endpoint_returns_safe_metadata(tmp_path) -> None:
    """A valid PDF upload should return HTTP 201 and a generated ID."""

    app.dependency_overrides[get_document_service] = lambda: DocumentService(
        tmp_path,
        max_size_bytes=1_000,
    )
    client = TestClient(app)

    try:
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("report.pdf", b"%PDF-1.4\nmock", "application/pdf")},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 201
    assert response.json()["original_filename"] == "report.pdf"
    assert response.json()["document_id"]
