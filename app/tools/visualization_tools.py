"""Controlled Plotly chart generation from validated specifications."""

import json

import pandas as pd
import plotly.express as px

from app.models.visualization_agent import (
    BarChartSpec,
    BoxChartSpec,
    ChartSpec,
    CorrelationHeatmapSpec,
    HistogramChartSpec,
    LineChartSpec,
    ScatterChartSpec,
)


class VisualizationInputError(ValueError):
    """Raised when selected columns cannot support a requested chart."""


def _require_columns(dataframe: pd.DataFrame, columns: list[str | None]) -> None:
    requested = [column for column in columns if column is not None]
    missing = [column for column in requested if column not in dataframe.columns]
    if missing:
        raise VisualizationInputError(f"Unknown chart columns: {', '.join(missing)}")


def _require_numeric(dataframe: pd.DataFrame, columns: list[str]) -> None:
    non_numeric = [
        column
        for column in columns
        if not pd.api.types.is_numeric_dtype(dataframe[column])
    ]
    if non_numeric:
        raise VisualizationInputError(
            f"Numeric chart columns required: {', '.join(non_numeric)}"
        )


def _bar_chart(dataframe: pd.DataFrame, spec: BarChartSpec):
    _require_columns(dataframe, [spec.x, spec.y, spec.color])
    if spec.aggregation != "count":
        _require_numeric(dataframe, [spec.y])

    group_columns = [spec.x] + ([spec.color] if spec.color else [])
    grouped = (
        dataframe.groupby(group_columns, dropna=False)[spec.y]
        .agg(spec.aggregation)
        .reset_index(name="value")
    )
    return px.bar(
        grouped,
        x=spec.x,
        y="value",
        color=spec.color,
        title=spec.title,
        labels={"value": f"{spec.aggregation} of {spec.y}"},
    )


def _line_chart(dataframe: pd.DataFrame, spec: LineChartSpec):
    _require_columns(dataframe, [spec.x, spec.y, spec.color])
    _require_numeric(dataframe, [spec.y])
    ordered = dataframe.sort_values(spec.x)
    return px.line(
        ordered,
        x=spec.x,
        y=spec.y,
        color=spec.color,
        markers=True,
        title=spec.title,
    )


def _scatter_chart(dataframe: pd.DataFrame, spec: ScatterChartSpec):
    _require_columns(dataframe, [spec.x, spec.y, spec.color])
    _require_numeric(dataframe, [spec.x, spec.y])
    return px.scatter(
        dataframe,
        x=spec.x,
        y=spec.y,
        color=spec.color,
        title=spec.title,
    )


def _histogram(dataframe: pd.DataFrame, spec: HistogramChartSpec):
    _require_columns(dataframe, [spec.x, spec.color])
    _require_numeric(dataframe, [spec.x])
    return px.histogram(
        dataframe,
        x=spec.x,
        color=spec.color,
        nbins=spec.bins,
        title=spec.title,
    )


def _box_chart(dataframe: pd.DataFrame, spec: BoxChartSpec):
    _require_columns(dataframe, [spec.x, spec.y, spec.color])
    _require_numeric(dataframe, [spec.y])
    return px.box(
        dataframe,
        x=spec.x,
        y=spec.y,
        color=spec.color,
        points="outliers",
        title=spec.title,
    )


def _correlation_heatmap(dataframe: pd.DataFrame, spec: CorrelationHeatmapSpec):
    if spec.columns is not None:
        _require_columns(dataframe, list(spec.columns))
        selected = dataframe[spec.columns].select_dtypes(include="number")
    else:
        selected = dataframe.select_dtypes(include="number")
    if selected.shape[1] < 2:
        raise VisualizationInputError(
            "At least two numeric columns are required for a heatmap."
        )
    return px.imshow(
        selected.corr(),
        text_auto=".2f",
        aspect="auto",
        zmin=-1,
        zmax=1,
        color_continuous_scale="RdBu_r",
        title=spec.title,
    )


def create_chart(dataframe: pd.DataFrame, spec: ChartSpec) -> dict:
    """Create JSON-safe Plotly output from one allowlisted chart spec."""

    if isinstance(spec, BarChartSpec):
        figure = _bar_chart(dataframe, spec)
    elif isinstance(spec, LineChartSpec):
        figure = _line_chart(dataframe, spec)
    elif isinstance(spec, ScatterChartSpec):
        figure = _scatter_chart(dataframe, spec)
    elif isinstance(spec, HistogramChartSpec):
        figure = _histogram(dataframe, spec)
    elif isinstance(spec, BoxChartSpec):
        figure = _box_chart(dataframe, spec)
    elif isinstance(spec, CorrelationHeatmapSpec):
        figure = _correlation_heatmap(dataframe, spec)
    else:
        raise TypeError("Unsupported validated chart specification.")

    figure.update_layout(template="plotly_white")
    return json.loads(figure.to_json())
