"""Tests for loading stored datasets by safe identifiers."""

from uuid import uuid4

import pandas as pd
import pytest

from app.services.dataset_loader import (
    DatasetLoader,
    InvalidDatasetIdError,
)


def test_loader_reads_csv_by_uuid(tmp_path) -> None:
    """A UUID should resolve to its stored CSV file."""

    dataset_id = str(uuid4())
    expected = pd.DataFrame({"region": ["West"], "sales": [10]})
    expected.to_csv(tmp_path / f"{dataset_id}.csv", index=False)

    result = DatasetLoader(tmp_path).load(dataset_id)

    pd.testing.assert_frame_equal(result, expected)


def test_loader_reads_excel_by_uuid(tmp_path) -> None:
    """A UUID should resolve to its stored XLSX file."""

    dataset_id = str(uuid4())
    expected = pd.DataFrame({"product": ["A"], "units": [4]})
    expected.to_excel(tmp_path / f"{dataset_id}.xlsx", index=False)

    result = DatasetLoader(tmp_path).load(dataset_id)

    pd.testing.assert_frame_equal(result, expected)


def test_loader_rejects_paths_instead_of_ids(tmp_path) -> None:
    """The loader must reject arbitrary paths from users or agents."""

    loader = DatasetLoader(tmp_path)

    with pytest.raises(InvalidDatasetIdError):
        loader.load("../../private.csv")
