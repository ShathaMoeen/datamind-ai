"""Validated tool plans and results for the Analysis Agent."""

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


class AnalysisToolName(StrEnum):
    """Allowlisted statistical tools available to the Analysis Agent."""

    DESCRIBE_NUMERIC = "describe_numeric"
    GROUP_COMPARISON = "group_comparison"
    CORRELATION = "correlation"
    TREND_ANALYSIS = "trend_analysis"


class DescribeNumericCall(BaseModel):
    """Request descriptive statistics for numeric columns."""

    tool: Literal[AnalysisToolName.DESCRIBE_NUMERIC]
    columns: list[str] | None = None


class GroupComparisonCall(BaseModel):
    """Request an aggregation grouped by one categorical column."""

    tool: Literal[AnalysisToolName.GROUP_COMPARISON]
    group_by: str
    metric: str
    aggregation: Literal["sum", "mean", "median", "min", "max", "count"]


class CorrelationCall(BaseModel):
    """Request pairwise Pearson correlations for numeric columns."""

    tool: Literal[AnalysisToolName.CORRELATION]
    columns: list[str] | None = None


class TrendAnalysisCall(BaseModel):
    """Request a time-based aggregation for one metric."""

    tool: Literal[AnalysisToolName.TREND_ANALYSIS]
    date_column: str
    metric: str
    frequency: Literal["day", "week", "month", "quarter", "year"]
    aggregation: Literal["sum", "mean", "median", "count"]


AnalysisToolCall = Annotated[
    DescribeNumericCall | GroupComparisonCall | CorrelationCall | TrendAnalysisCall,
    Field(discriminator="tool"),
]


class AnalysisAgentPlan(BaseModel):
    """Structured analysis plan produced by the language model."""

    tool_calls: list[AnalysisToolCall] = Field(min_length=1, max_length=4)
    reason: str = Field(min_length=1)


class NumericColumnSummary(BaseModel):
    """Descriptive statistics calculated for one numeric column."""

    column: str
    count: int = Field(ge=0)
    mean: float | None
    median: float | None
    standard_deviation: float | None
    minimum: float | None
    maximum: float | None


class NumericDescriptionResult(BaseModel):
    """Descriptive statistics for selected numeric columns."""

    columns: list[NumericColumnSummary]


class GroupMetricResult(BaseModel):
    """One aggregated value for a category."""

    group: str
    value: float | None


class GroupComparisonResult(BaseModel):
    """Grouped aggregation calculated by Pandas."""

    group_by: str
    metric: str
    aggregation: str
    groups: list[GroupMetricResult]


class CorrelationPair(BaseModel):
    """Pearson correlation for two numeric columns."""

    first_column: str
    second_column: str
    coefficient: float | None = Field(ge=-1.0, le=1.0)


class CorrelationResult(BaseModel):
    """Pairwise correlations; correlation does not establish causation."""

    method: Literal["pearson"] = "pearson"
    pairs: list[CorrelationPair]
    warning: str = "Correlation indicates association, not causation."


class TrendPoint(BaseModel):
    """One time period and its aggregated metric."""

    period: str
    value: float | None


class TrendAnalysisResult(BaseModel):
    """Time-ordered aggregation for a metric."""

    date_column: str
    metric: str
    frequency: str
    aggregation: str
    points: list[TrendPoint]


class AnalysisToolExecution(BaseModel):
    """Normalized output from one validated analysis tool call."""

    tool: AnalysisToolName
    parameters: dict[str, Any]
    status: Literal["success"] = "success"
    output: dict[str, Any]


class AnalysisTraceEvent(BaseModel):
    """Observable workflow event without private chain-of-thought."""

    step: int = Field(ge=1)
    action: str
    selected_tool: AnalysisToolName | None = None
    parameters: dict[str, str] = Field(default_factory=dict)
    status: Literal["success", "failed"]


class AnalysisAgentResult(BaseModel):
    """Complete structured result returned by the Analysis Agent."""

    selected_agent: Literal["analysis_agent"] = "analysis_agent"
    dataset_id: str
    plan: AnalysisAgentPlan
    executions: list[AnalysisToolExecution]
    trace: list[AnalysisTraceEvent]
