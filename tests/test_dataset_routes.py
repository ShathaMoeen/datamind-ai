"""Integration tests for dataset upload endpoints."""

from uuid import uuid4

import pandas as pd
from fastapi.testclient import TestClient

from app.api.routes.datasets import get_dataset_loader, get_dataset_service
from app.main import app
from app.services.dataset_loader import DatasetLoader
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


def test_profile_endpoint_returns_calculated_metrics(tmp_path) -> None:
    """The API should expose the deterministic profile tool result."""

    dataset_id = str(uuid4())
    dataframe = pd.DataFrame(
        {
            "region": ["West", "East"],
            "sales": [100.0, None],
        }
    )
    dataframe.to_csv(tmp_path / f"{dataset_id}.csv", index=False)
    app.dependency_overrides[get_dataset_loader] = lambda: DatasetLoader(tmp_path)
    client = TestClient(app)

    try:
        response = client.get(f"/api/v1/datasets/{dataset_id}/profile")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["row_count"] == 2
    assert response.json()["column_count"] == 2
    assert response.json()["columns"][1]["missing_count"] == 1
    assert response.json()["columns"][1]["missing_percentage"] == 50.0


def test_profile_endpoint_returns_not_found_for_unknown_dataset(tmp_path) -> None:
    """A valid but unknown UUID should produce HTTP 404."""

    app.dependency_overrides[get_dataset_loader] = lambda: DatasetLoader(tmp_path)
    client = TestClient(app)

    try:
        response = client.get(f"/api/v1/datasets/{uuid4()}/profile")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 404
