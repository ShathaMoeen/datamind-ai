"""Structured plans, tool results, and traces for the Data Agent."""

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class DataToolName(StrEnum):
    """Allowlisted tools that the Data Agent may select."""

    PROFILE_DATASET = "profile_dataset"
    DETECT_MISSING_VALUES = "detect_missing_values"
    COUNT_DUPLICATE_ROWS = "count_duplicate_rows"
    DETECT_OUTLIERS_IQR = "detect_outliers_iqr"


class DataAgentPlan(BaseModel):
    """Validated tool choices produced by the language model."""

    tools: list[DataToolName] = Field(min_length=1, max_length=4)
    reason: str = Field(min_length=1)

    @field_validator("tools")
    @classmethod
    def tools_must_be_unique(cls, tools: list[DataToolName]) -> list[DataToolName]:
        """Reject repeated calls that would waste computation."""

        if len(tools) != len(set(tools)):
            raise ValueError("Data Agent tools must be unique.")
        return tools


class MissingColumnResult(BaseModel):
    """Missing-value metrics for one column."""

    column: str
    missing_count: int = Field(ge=0)
    missing_percentage: float = Field(ge=0.0, le=100.0)


class MissingValuesResult(BaseModel):
    """Missing-value metrics for all columns."""

    row_count: int = Field(ge=0)
    columns: list[MissingColumnResult]


class DuplicateRowsResult(BaseModel):
    """Duplicate-row metrics without exposing row contents."""

    row_count: int = Field(ge=0)
    duplicate_count: int = Field(ge=0)
    duplicate_percentage: float = Field(ge=0.0, le=100.0)


class ColumnOutlierResult(BaseModel):
    """IQR outlier boundaries and count for one numeric column."""

    column: str
    lower_bound: float | None
    upper_bound: float | None
    outlier_count: int = Field(ge=0)


class OutlierDetectionResult(BaseModel):
    """Obvious numeric outlier candidates detected with the IQR rule."""

    method: Literal["iqr"] = "iqr"
    columns: list[ColumnOutlierResult]


class DataToolExecution(BaseModel):
    """Normalized output from one allowlisted deterministic tool."""

    tool: DataToolName
    status: Literal["success"] = "success"
    output: dict[str, Any]


class AgentTraceEvent(BaseModel):
    """User-visible execution metadata without private model reasoning."""

    step: int = Field(ge=1)
    action: str
    selected_tool: DataToolName | None = None
    parameters: dict[str, str] = Field(default_factory=dict)
    status: Literal["success", "failed"]


class DataAgentResult(BaseModel):
    """Complete structured result returned by the Data Agent."""

    selected_agent: Literal["data_agent"] = "data_agent"
    dataset_id: str
    plan: DataAgentPlan
    executions: list[DataToolExecution]
    trace: list[AgentTraceEvent]
