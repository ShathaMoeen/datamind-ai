"""Safe storage and metadata operations for uploaded PDF documents."""

import json
from pathlib import Path
from uuid import UUID, uuid4

import anyio
from fastapi import UploadFile

from app.models.rag import DocumentUploadResponse, StoredDocument

CHUNK_SIZE_BYTES = 1024 * 1024


class UnsupportedDocumentTypeError(ValueError):
    """Raised when an upload is not a PDF by extension or signature."""


class DocumentTooLargeError(ValueError):
    """Raised when a PDF exceeds the configured byte limit."""


class EmptyDocumentError(ValueError):
    """Raised when a PDF upload contains no bytes."""


class DocumentNotFoundError(FileNotFoundError):
    """Raised when a valid document identifier has no stored PDF."""


class DocumentService:
    """Store PDFs under UUID names and persist citation-safe metadata."""

    def __init__(self, upload_directory: Path, max_size_bytes: int) -> None:
        self._upload_directory = upload_directory.resolve()
        self._max_size_bytes = max_size_bytes

    async def save(self, upload: UploadFile) -> DocumentUploadResponse:
        """Validate PDF extension, signature, size, and generated storage path."""

        original_filename = Path(upload.filename or "").name
        if Path(original_filename).suffix.lower() != ".pdf":
            await upload.close()
            raise UnsupportedDocumentTypeError("Only .pdf documents are allowed.")

        document_id = str(uuid4())
        destination = self._upload_directory / f"{document_id}.pdf"
        metadata_path = self._upload_directory / f"{document_id}.json"
        size_bytes = 0
        first_chunk = True
        self._upload_directory.mkdir(parents=True, exist_ok=True)

        try:
            async with await anyio.open_file(destination, "wb") as stored_file:
                while chunk := await upload.read(CHUNK_SIZE_BYTES):
                    if first_chunk:
                        if b"%PDF-" not in chunk[:1024]:
                            raise UnsupportedDocumentTypeError(
                                "The uploaded file does not have a valid PDF signature."
                            )
                        first_chunk = False
                    size_bytes += len(chunk)
                    if size_bytes > self._max_size_bytes:
                        raise DocumentTooLargeError(
                            "The PDF exceeds the configured upload limit."
                        )
                    await stored_file.write(chunk)

            if size_bytes == 0:
                raise EmptyDocumentError("The uploaded PDF is empty.")

            metadata_path.write_text(
                json.dumps(
                    {
                        "document_id": document_id,
                        "original_filename": original_filename,
                        "size_bytes": size_bytes,
                    }
                ),
                encoding="utf-8",
            )
        except Exception:
            destination.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
            raise
        finally:
            await upload.close()

        return DocumentUploadResponse(
            document_id=document_id,
            original_filename=original_filename,
            size_bytes=size_bytes,
        )

    def get(self, document_id: str) -> StoredDocument:
        """Resolve one UUID to its PDF and stored citation metadata."""

        try:
            normalized_id = str(UUID(document_id))
        except ValueError as error:
            raise DocumentNotFoundError("document_id must be a valid UUID.") from error

        pdf_path = (self._upload_directory / f"{normalized_id}.pdf").resolve()
        metadata_path = (self._upload_directory / f"{normalized_id}.json").resolve()
        if (
            pdf_path.parent != self._upload_directory
            or metadata_path.parent != self._upload_directory
            or not pdf_path.is_file()
            or not metadata_path.is_file()
        ):
            raise DocumentNotFoundError(f"Document '{normalized_id}' was not found.")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return StoredDocument(path=pdf_path, **metadata)
