"""Safe storage operations for uploaded tabular datasets."""

from pathlib import Path
from uuid import uuid4

import anyio
from fastapi import UploadFile

from app.models.dataset import DatasetFileType, DatasetUploadResponse

CHUNK_SIZE_BYTES = 1024 * 1024
SUPPORTED_EXTENSIONS = {
    ".csv": DatasetFileType.CSV,
    ".xlsx": DatasetFileType.EXCEL,
}


class UnsupportedDatasetTypeError(ValueError):
    """Raised when a dataset does not have an allowed extension."""


class DatasetTooLargeError(ValueError):
    """Raised when an upload exceeds the configured byte limit."""


class EmptyDatasetError(ValueError):
    """Raised when an uploaded dataset contains no bytes."""


class DatasetService:
    """Validate and persist uploaded datasets under generated names."""

    def __init__(self, upload_directory: Path, max_size_bytes: int) -> None:
        self._upload_directory = upload_directory.resolve()
        self._max_size_bytes = max_size_bytes

    async def save(self, upload: UploadFile) -> DatasetUploadResponse:
        """Store one valid dataset and return non-sensitive metadata."""

        original_filename = Path(upload.filename or "").name
        extension = Path(original_filename).suffix.lower()
        file_type = SUPPORTED_EXTENSIONS.get(extension)
        if file_type is None:
            await upload.close()
            raise UnsupportedDatasetTypeError("Only .csv and .xlsx files are allowed.")

        dataset_id = str(uuid4())
        destination = self._upload_directory / f"{dataset_id}{extension}"
        size_bytes = 0
        self._upload_directory.mkdir(parents=True, exist_ok=True)

        try:
            async with await anyio.open_file(destination, "wb") as stored_file:
                while chunk := await upload.read(CHUNK_SIZE_BYTES):
                    size_bytes += len(chunk)
                    if size_bytes > self._max_size_bytes:
                        raise DatasetTooLargeError(
                            "The dataset exceeds the configured upload limit."
                        )
                    await stored_file.write(chunk)
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        finally:
            await upload.close()

        if size_bytes == 0:
            destination.unlink(missing_ok=True)
            raise EmptyDatasetError("The uploaded dataset is empty.")

        return DatasetUploadResponse(
            dataset_id=dataset_id,
            original_filename=original_filename,
            file_type=file_type,
            size_bytes=size_bytes,
        )
