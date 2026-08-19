"""Unit tests for safe dataset storage."""

import asyncio
from io import BytesIO

import pytest
from fastapi import UploadFile

from app.services.dataset_service import (
    DatasetService,
    DatasetTooLargeError,
    UnsupportedDatasetTypeError,
)


def test_service_stores_csv_under_generated_name(tmp_path) -> None:
    """The original filename must not become the stored path."""

    service = DatasetService(upload_directory=tmp_path, max_size_bytes=1_000)
    upload = UploadFile(filename="../sales.csv", file=BytesIO(b"region,sales\nWest,10\n"))

    result = asyncio.run(service.save(upload))

    stored_files = list(tmp_path.iterdir())
    assert result.original_filename == "sales.csv"
    assert result.size_bytes == 21
    assert len(stored_files) == 1
    assert stored_files[0].name == f"{result.dataset_id}.csv"


def test_service_rejects_unsupported_extension(tmp_path) -> None:
    """Executable and unrelated formats must be rejected."""

    service = DatasetService(upload_directory=tmp_path, max_size_bytes=1_000)
    upload = UploadFile(filename="payload.exe", file=BytesIO(b"unsafe"))

    with pytest.raises(UnsupportedDatasetTypeError):
        asyncio.run(service.save(upload))


def test_service_deletes_partial_file_when_limit_is_exceeded(tmp_path) -> None:
    """A rejected oversized upload must not remain on disk."""

    service = DatasetService(upload_directory=tmp_path, max_size_bytes=3)
    upload = UploadFile(filename="large.csv", file=BytesIO(b"1234"))

    with pytest.raises(DatasetTooLargeError):
        asyncio.run(service.save(upload))

    assert list(tmp_path.iterdir()) == []
