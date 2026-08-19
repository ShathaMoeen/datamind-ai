"""Validated chart specifications and Visualization Agent results."""

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


class ChartType(StrEnum):
    """Allowlisted chart types supported by DataMind AI."""

    BAR = "bar"
    LINE = "line"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    BOX = "box"
    CORRELATION_HEATMAP = "correlation_heatmap"


class BarChartSpec(BaseModel):
    """Specification for comparing aggregated categories."""

    chart_type: Literal[ChartType.BAR]
    x: str
    y: str
    aggregation: Literal["sum", "mean", "median", "count"] = "sum"
    color: str | None = None
    title: str = Field(min_length=1)


class LineChartSpec(BaseModel):
    """Specification for showing ordered or time-based change."""

    chart_type: Literal[ChartType.LINE]
    x: str
    y: str
    color: str | None = None
    title: str = Field(min_length=1)


class ScatterChartSpec(BaseModel):
    """Specification for examining two numeric variables."""

    chart_type: Literal[ChartType.SCATTER]
    x: str
    y: str
    color: str | None = None
    title: str = Field(min_length=1)


class HistogramChartSpec(BaseModel):
    """Specification for a numeric distribution."""

    chart_type: Literal[ChartType.HISTOGRAM]
    x: str
    color: str | None = None
    bins: int = Field(default=20, ge=5, le=100)
    title: str = Field(min_length=1)


class BoxChartSpec(BaseModel):
    """Specification for spread and potential outliers."""

    chart_type: Literal[ChartType.BOX]
    y: str
    x: str | None = None
    color: str | None = None
    title: str = Field(min_length=1)


class CorrelationHeatmapSpec(BaseModel):
    """Specification for a numeric correlation matrix."""

    chart_type: Literal[ChartType.CORRELATION_HEATMAP]
    columns: list[str] | None = None
    title: str = Field(min_length=1)


ChartSpec = Annotated[
    BarChartSpec
    | LineChartSpec
    | ScatterChartSpec
    | HistogramChartSpec
    | BoxChartSpec
    | CorrelationHeatmapSpec,
    Field(discriminator="chart_type"),
]


class VisualizationPlan(BaseModel):
    """Structured chart decision produced by the language model."""

    chart: ChartSpec
    reason: str = Field(min_length=1)
    explanation: str = Field(min_length=1)


class VisualizationTraceEvent(BaseModel):
    """Observable chart workflow event without chain-of-thought."""

    step: int = Field(ge=1)
    action: str
    chart_type: ChartType | None = None
    parameters: dict[str, str] = Field(default_factory=dict)
    status: Literal["success", "failed"]


class VisualizationAgentResult(BaseModel):
    """Plotly figure JSON and metadata returned by the agent."""

    selected_agent: Literal["visualization_agent"] = "visualization_agent"
    dataset_id: str
    chart_type: ChartType
    explanation: str
    figure: dict[str, Any]
    trace: list[VisualizationTraceEvent]
