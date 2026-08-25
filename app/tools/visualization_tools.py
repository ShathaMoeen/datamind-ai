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


POSITIVE_COLOR = "#A7FF4F"
NEGATIVE_COLOR = "#FF6B6B"
NEUTRAL_COLOR = "#6574F7"
SERIES_COLORS = ["#A7FF4F", "#6574F7", "#F7C948", "#B76EFF", "#35C6E8"]
LOWER_IS_BETTER_TERMS = (
    "expense",
    "cost",
    "debt",
    "loss",
    "risk",
    "stress",
    "error",
    "defect",
    "churn",
)


def _higher_is_better(metric: str) -> bool:
    """Apply a transparent naming heuristic for semantic result colors."""

    normalized = metric.casefold()
    return not any(term in normalized for term in LOWER_IS_BETTER_TERMS)


def _result_colors(values: list[float], metric: str) -> list[str]:
    """Highlight the preferred and least-preferred values deterministically."""

    if not values or min(values) == max(values):
        return [NEUTRAL_COLOR] * len(values)
    maximum = max(values)
    minimum = min(values)
    preferred = maximum if _higher_is_better(metric) else minimum
    least_preferred = minimum if _higher_is_better(metric) else maximum
    return [
        POSITIVE_COLOR
        if value == preferred
        else NEGATIVE_COLOR
        if value == least_preferred
        else NEUTRAL_COLOR
        for value in values
    ]


def _apply_result_colors(figure, spec: ChartSpec) -> None:
    """Apply accessible, result-aware colors to the generated Plotly figure."""

    if isinstance(spec, BarChartSpec) and spec.color is None:
        for trace in figure.data:
            values = [float(value) for value in trace.y]
            trace.marker.color = _result_colors(values, spec.y)
        preferred_label = "Higher" if _higher_is_better(spec.y) else "Lower"
        figure.add_annotation(
            text=(
                f"{preferred_label} is preferred: green | "
                "الأفضل أخضر، والأقل تفضيلًا أحمر"
            ),
            x=1,
            y=1.12,
            xref="paper",
            yref="paper",
            xanchor="right",
            showarrow=False,
            font={"size": 11},
        )
    elif isinstance(spec, LineChartSpec):
        for index, trace in enumerate(figure.data):
            trace.line.color = SERIES_COLORS[index % len(SERIES_COLORS)]
            trace.marker.color = SERIES_COLORS[index % len(SERIES_COLORS)]
    elif isinstance(spec, ScatterChartSpec) and spec.color is None:
        for trace in figure.data:
            trace.marker.color = trace.y
            trace.marker.colorscale = "RdYlGn"
            trace.marker.showscale = True
    elif not isinstance(spec, CorrelationHeatmapSpec):
        for index, trace in enumerate(figure.data):
            trace.marker.color = SERIES_COLORS[index % len(SERIES_COLORS)]


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

    _apply_result_colors(figure, spec)
    figure.update_layout(template="plotly_white")
    return json.loads(figure.to_json())
