"""Tests for controlled Plotly visualization tools."""

import base64
import json

import numpy as np
import pandas as pd
import plotly.io as pio

from app.models.visualization_agent import (
    BarChartSpec,
    CorrelationHeatmapSpec,
)
from app.tools.visualization_tools import create_chart


def _decode_plotly_array(value: list | dict) -> list:
    """Decode Plotly 6 typed-array JSON into ordinary test values."""

    if isinstance(value, list):
        return value
    binary = base64.b64decode(value["bdata"])
    return np.frombuffer(binary, dtype=np.dtype(value["dtype"])).tolist()


def test_bar_chart_aggregates_categories() -> None:
    """A bar chart should aggregate the selected metric deterministically."""

    dataframe = pd.DataFrame(
        {"region": ["West", "West", "East"], "sales": [10, 15, 7]}
    )
    spec = BarChartSpec(
        chart_type="bar",
        x="region",
        y="sales",
        aggregation="sum",
        title="Sales by region",
    )

    figure = create_chart(dataframe, spec)

    assert figure["data"][0]["type"] == "bar"
    y_values = _decode_plotly_array(figure["data"][0]["y"])
    values = dict(zip(figure["data"][0]["x"], y_values, strict=True))
    assert values == {"East": 7, "West": 25}


def test_heatmap_contains_numeric_correlation_matrix() -> None:
    """A heatmap should include one matrix trace for numeric columns."""

    dataframe = pd.DataFrame({"units": [1, 2, 3], "sales": [10, 20, 30]})
    spec = CorrelationHeatmapSpec(
        chart_type="correlation_heatmap",
        title="Numeric correlations",
    )

    figure = create_chart(dataframe, spec)

    chart = pio.from_json(json.dumps(figure))
    assert chart.data[0].type == "heatmap"
    assert list(chart.data[0].x) == ["units", "sales"]
