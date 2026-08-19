"""Structured results returned by deterministic dataset profiling."""

from pydantic import BaseModel, Field


class ColumnProfile(BaseModel):
    """Quality and type information for one dataset column."""

    name: str
    data_type: str
    missing_count: int = Field(ge=0)
    missing_percentage: float = Field(ge=0.0, le=100.0)
    unique_count: int = Field(ge=0)


class DatasetProfile(BaseModel):
    """Deterministic structural summary of a tabular dataset."""

    dataset_id: str
    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    duplicate_rows: int = Field(ge=0)
    columns: list[ColumnProfile]
