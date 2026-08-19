"""HTTP endpoints for safe PDF uploads."""

from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.core.config import Settings, get_settings
from app.models.rag import DocumentUploadResponse
from app.services.document_service import (
    DocumentService,
    DocumentTooLargeError,
    EmptyDocumentError,
    UnsupportedDocumentTypeError,
)

router = APIRouter(prefix="/documents", tags=["Documents"])


def get_document_service(
    settings: Annotated[Settings, Depends(get_settings)],
) -> DocumentService:
    """Build safe PDF storage from application settings."""

    return DocumentService(
        upload_directory=settings.document_upload_directory,
        max_size_bytes=settings.max_document_upload_size_mb * 1024 * 1024,
    )


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    file: Annotated[UploadFile, File(description="A text-based PDF document")],
    service: Annotated[DocumentService, Depends(get_document_service)],
) -> DocumentUploadResponse:
    """Validate and persist one PDF for later RAG ingestion."""

    try:
        return await service.save(file)
    except UnsupportedDocumentTypeError as error:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=str(error),
        ) from error
    except DocumentTooLargeError as error:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=str(error),
        ) from error
    except EmptyDocumentError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error
