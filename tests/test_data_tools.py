"""Tests for deterministic Pandas data tools."""

from uuid import uuid4

import pandas as pd

from app.services.dataset_loader import DatasetLoader
from app.tools.data_tools import detect_outliers_iqr, profile_dataset


def test_profile_dataset_calculates_known_quality_metrics(tmp_path) -> None:
    """Profile values should equal deterministic Pandas calculations."""

    dataset_id = str(uuid4())
    dataframe = pd.DataFrame(
        {
            "region": ["West", "West", "East", "West"],
            "sales": [100.0, 100.0, None, 100.0],
        }
    )
    dataframe.to_csv(tmp_path / f"{dataset_id}.csv", index=False)

    profile = profile_dataset(dataset_id, DatasetLoader(tmp_path))

    assert profile.row_count == 4
    assert profile.column_count == 2
    assert profile.duplicate_rows == 2
    assert profile.columns[0].name == "region"
    assert profile.columns[0].unique_count == 2
    assert profile.columns[1].name == "sales"
    assert profile.columns[1].missing_count == 1
    assert profile.columns[1].missing_percentage == 25.0


def test_iqr_tool_detects_obvious_numeric_outlier() -> None:
    """The deterministic IQR rule should identify an extreme value."""

    dataframe = pd.DataFrame({"value": [10, 11, 12, 13, 100]})

    result = detect_outliers_iqr(dataframe)

    assert result.method == "iqr"
    assert result.columns[0].column == "value"
    assert result.columns[0].outlier_count == 1
