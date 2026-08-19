"""Tests for secure PDF upload storage."""

import asyncio
from io import BytesIO

import pytest
from fastapi import UploadFile

from app.services.document_service import (
    DocumentService,
    UnsupportedDocumentTypeError,
)


def test_document_service_stores_pdf_and_metadata(tmp_path) -> None:
    """A signed PDF should be stored under a generated identifier."""

    service = DocumentService(tmp_path, max_size_bytes=1_000)
    upload = UploadFile(
        filename="../report.pdf",
        file=BytesIO(b"%PDF-1.4\nmock content"),
    )

    result = asyncio.run(service.save(upload))
    stored = service.get(result.document_id)

    assert result.original_filename == "report.pdf"
    assert stored.path.name == f"{result.document_id}.pdf"
    assert stored.original_filename == "report.pdf"


def test_document_service_rejects_fake_pdf_signature(tmp_path) -> None:
    """A renamed non-PDF must be rejected before it is retained."""

    service = DocumentService(tmp_path, max_size_bytes=1_000)
    upload = UploadFile(filename="fake.pdf", file=BytesIO(b"not a pdf"))

    with pytest.raises(UnsupportedDocumentTypeError):
        asyncio.run(service.save(upload))

    assert list(tmp_path.iterdir()) == []
