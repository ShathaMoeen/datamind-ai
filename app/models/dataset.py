"""Models describing uploaded datasets."""

from enum import StrEnum

from pydantic import BaseModel, Field


class DatasetFileType(StrEnum):
    """Dataset formats accepted by the first upload endpoint."""

    CSV = "csv"
    EXCEL = "excel"


class DatasetUploadResponse(BaseModel):
    """Safe metadata returned after a dataset is stored."""

    dataset_id: str
    original_filename: str
    file_type: DatasetFileType
    size_bytes: int = Field(gt=0)
