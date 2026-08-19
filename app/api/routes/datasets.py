"""HTTP endpoints for dataset uploads."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.core.config import Settings, get_settings
from app.models.data_profile import DatasetProfile
from app.models.dataset import DatasetUploadResponse
from app.services.dataset_loader import (
    DatasetLoader,
    DatasetNotFoundError,
    DatasetReadError,
)
from app.services.dataset_service import (
    DatasetService,
    DatasetTooLargeError,
    EmptyDatasetError,
    UnsupportedDatasetTypeError,
)
from app.tools.data_tools import profile_dataset

router = APIRouter(prefix="/datasets", tags=["Datasets"])


def get_dataset_service(
    settings: Annotated[Settings, Depends(get_settings)],
) -> DatasetService:
    """Build the upload service from validated application settings."""

    return DatasetService(
        upload_directory=settings.dataset_upload_directory,
        max_size_bytes=settings.max_upload_size_mb * 1024 * 1024,
    )


def get_dataset_loader(
    settings: Annotated[Settings, Depends(get_settings)],
) -> DatasetLoader:
    """Build a loader restricted to the configured upload directory."""

    return DatasetLoader(upload_directory=settings.dataset_upload_directory)


@router.post(
    "/upload",
    response_model=DatasetUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_dataset(
    file: Annotated[UploadFile, File(description="A CSV or XLSX dataset")],
    service: Annotated[DatasetService, Depends(get_dataset_service)],
) -> DatasetUploadResponse:
    """Validate and store one tabular dataset."""

    try:
        return await service.save(file)
    except UnsupportedDatasetTypeError as error:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(error)) from error
    except DatasetTooLargeError as error:
        raise HTTPException(status_code=status.HTTP_413_CONTENT_TOO_LARGE, detail=str(error)) from error
    except EmptyDatasetError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error


@router.get("/{dataset_id}/profile", response_model=DatasetProfile)
def get_dataset_profile(
    dataset_id: UUID,
    loader: Annotated[DatasetLoader, Depends(get_dataset_loader)],
) -> DatasetProfile:
    """Return deterministic structure and quality metrics for one dataset."""

    try:
        return profile_dataset(str(dataset_id), loader)
    except DatasetNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error
    except DatasetReadError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error
