"""Tests for deterministic statistical analysis tools."""

import pandas as pd

from app.models.analysis_agent import (
    GroupComparisonCall,
    TrendAnalysisCall,
)
from app.tools.analysis_tools import (
    analyze_trend,
    calculate_correlations,
    group_comparison,
)


def test_group_comparison_calculates_known_sums() -> None:
    """Grouped values should match deterministic Pandas sums."""

    dataframe = pd.DataFrame(
        {"region": ["West", "West", "East"], "sales": [10, 15, 7]}
    )
    call = GroupComparisonCall(
        tool="group_comparison",
        group_by="region",
        metric="sales",
        aggregation="sum",
    )

    result = group_comparison(dataframe, call)

    values = {group.group: group.value for group in result.groups}
    assert values == {"East": 7.0, "West": 25.0}


def test_correlation_reports_perfect_positive_association() -> None:
    """Linearly proportional columns should have correlation 1."""

    dataframe = pd.DataFrame({"units": [1, 2, 3], "sales": [10, 20, 30]})

    result = calculate_correlations(dataframe)

    assert result.pairs[0].coefficient == 1.0
    assert "not causation" in result.warning


def test_trend_analysis_aggregates_by_month() -> None:
    """Date rows should be grouped into ordered monthly totals."""

    dataframe = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-20", "2026-02-01"],
            "sales": [10, 15, 7],
        }
    )
    call = TrendAnalysisCall(
        tool="trend_analysis",
        date_column="date",
        metric="sales",
        frequency="month",
        aggregation="sum",
    )

    result = analyze_trend(dataframe, call)

    assert [(point.period, point.value) for point in result.points] == [
        ("2026-01", 25.0),
        ("2026-02", 7.0),
    ]
