"""Integration tests for dataset upload endpoints."""

from fastapi.testclient import TestClient

from app.api.routes.datasets import get_dataset_service
from app.main import app
from app.services.dataset_service import DatasetService


def test_upload_csv_endpoint(tmp_path) -> None:
    """A valid CSV should produce safe dataset metadata."""

    app.dependency_overrides[get_dataset_service] = lambda: DatasetService(
        upload_directory=tmp_path,
        max_size_bytes=1_000,
    )
    client = TestClient(app)

    try:
        response = client.post(
            "/api/v1/datasets/upload",
            files={"file": ("sales.csv", b"region,sales\nWest,10\n", "text/csv")},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 201
    assert response.json()["original_filename"] == "sales.csv"
    assert response.json()["file_type"] == "csv"
    assert response.json()["size_bytes"] == 21
