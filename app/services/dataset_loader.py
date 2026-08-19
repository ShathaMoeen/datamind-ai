"""Resolve stored dataset identifiers and load their tabular content."""

from pathlib import Path
from uuid import UUID
from zipfile import BadZipFile

import pandas as pd


class InvalidDatasetIdError(ValueError):
    """Raised when a dataset identifier is not a UUID."""


class DatasetNotFoundError(FileNotFoundError):
    """Raised when no stored file matches a valid dataset identifier."""


class DatasetReadError(ValueError):
    """Raised when Pandas cannot parse a stored dataset."""


class DatasetLoader:
    """Load CSV or XLSX files by generated identifier, never by user path."""

    def __init__(self, upload_directory: Path) -> None:
        self._upload_directory = upload_directory.resolve()

    def resolve_path(self, dataset_id: str) -> Path:
        """Resolve a UUID to one known dataset path inside the upload directory."""

        try:
            normalized_id = str(UUID(dataset_id))
        except ValueError as error:
            raise InvalidDatasetIdError("dataset_id must be a valid UUID.") from error

        for extension in (".csv", ".xlsx"):
            candidate = (self._upload_directory / f"{normalized_id}{extension}").resolve()
            if candidate.parent != self._upload_directory:
                raise InvalidDatasetIdError("The resolved dataset path is invalid.")
            if candidate.is_file():
                return candidate

        raise DatasetNotFoundError(f"Dataset '{normalized_id}' was not found.")

    def load(self, dataset_id: str) -> pd.DataFrame:
        """Read one stored CSV or XLSX file into a Pandas DataFrame."""

        path = self.resolve_path(dataset_id)
        try:
            if path.suffix == ".csv":
                return pd.read_csv(path)
            return pd.read_excel(path, engine="openpyxl")
        except (
            BadZipFile,
            OSError,
            UnicodeDecodeError,
            ValueError,
            pd.errors.EmptyDataError,
            pd.errors.ParserError,
        ) as error:
            raise DatasetReadError(
                "The stored dataset could not be parsed as a valid table."
            ) from error
