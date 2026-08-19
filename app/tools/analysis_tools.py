"""Deterministic Pandas tools for statistical and trend analysis."""

import math

import pandas as pd

from app.models.analysis_agent import (
    CorrelationPair,
    CorrelationResult,
    GroupComparisonCall,
    GroupComparisonResult,
    GroupMetricResult,
    NumericColumnSummary,
    NumericDescriptionResult,
    TrendAnalysisCall,
    TrendAnalysisResult,
    TrendPoint,
)


class AnalysisInputError(ValueError):
    """Raised when requested columns or types cannot support an analysis."""


def _require_columns(dataframe: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in dataframe.columns]
    if missing:
        raise AnalysisInputError(f"Unknown dataset columns: {', '.join(missing)}")


def _finite_float(value: object) -> float | None:
    """Convert a numeric result to JSON-safe float or None."""

    if pd.isna(value):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def describe_numeric(
    dataframe: pd.DataFrame,
    columns: list[str] | None = None,
) -> NumericDescriptionResult:
    """Calculate descriptive statistics using Pandas, not the LLM."""

    if columns is not None:
        _require_columns(dataframe, columns)
        selected = dataframe[columns].select_dtypes(include="number")
        non_numeric = [column for column in columns if column not in selected.columns]
        if non_numeric:
            raise AnalysisInputError(
                f"Numeric columns required: {', '.join(non_numeric)}"
            )
    else:
        selected = dataframe.select_dtypes(include="number")

    if selected.shape[1] == 0:
        raise AnalysisInputError("No numeric columns are available for analysis.")

    summaries = []
    for column in selected.columns:
        values = selected[column].dropna()
        summaries.append(
            NumericColumnSummary(
                column=str(column),
                count=int(values.count()),
                mean=_finite_float(values.mean()),
                median=_finite_float(values.median()),
                standard_deviation=_finite_float(values.std()),
                minimum=_finite_float(values.min()),
                maximum=_finite_float(values.max()),
            )
        )
    return NumericDescriptionResult(columns=summaries)


def group_comparison(
    dataframe: pd.DataFrame,
    call: GroupComparisonCall,
) -> GroupComparisonResult:
    """Aggregate one metric across categories with validated parameters."""

    _require_columns(dataframe, [call.group_by, call.metric])
    if call.aggregation != "count" and not pd.api.types.is_numeric_dtype(
        dataframe[call.metric]
    ):
        raise AnalysisInputError("The selected metric must be numeric.")

    grouped = dataframe.groupby(call.group_by, dropna=False)[call.metric].agg(
        call.aggregation
    )
    groups = [
        GroupMetricResult(
            group="<missing>" if pd.isna(group) else str(group),
            value=_finite_float(value),
        )
        for group, value in grouped.items()
    ]
    return GroupComparisonResult(
        group_by=call.group_by,
        metric=call.metric,
        aggregation=call.aggregation,
        groups=groups,
    )


def calculate_correlations(
    dataframe: pd.DataFrame,
    columns: list[str] | None = None,
) -> CorrelationResult:
    """Calculate pairwise Pearson correlations for numeric columns."""

    if columns is not None:
        _require_columns(dataframe, columns)
        selected = dataframe[columns].select_dtypes(include="number")
    else:
        selected = dataframe.select_dtypes(include="number")

    if selected.shape[1] < 2:
        raise AnalysisInputError("At least two numeric columns are required.")

    matrix = selected.corr(method="pearson")
    pairs = []
    column_names = list(matrix.columns)
    for first_index, first_column in enumerate(column_names):
        for second_column in column_names[first_index + 1 :]:
            pairs.append(
                CorrelationPair(
                    first_column=str(first_column),
                    second_column=str(second_column),
                    coefficient=_finite_float(matrix.loc[first_column, second_column]),
                )
            )
    return CorrelationResult(pairs=pairs)


def analyze_trend(
    dataframe: pd.DataFrame,
    call: TrendAnalysisCall,
) -> TrendAnalysisResult:
    """Aggregate a metric into ordered time periods."""

    _require_columns(dataframe, [call.date_column, call.metric])
    if call.aggregation != "count" and not pd.api.types.is_numeric_dtype(
        dataframe[call.metric]
    ):
        raise AnalysisInputError("The trend metric must be numeric.")

    working = dataframe[[call.date_column, call.metric]].copy()
    working[call.date_column] = pd.to_datetime(
        working[call.date_column], errors="coerce"
    )
    working = working.dropna(subset=[call.date_column])
    if working.empty:
        raise AnalysisInputError("The date column contains no valid dates.")

    period_codes = {
        "day": "D",
        "week": "W",
        "month": "M",
        "quarter": "Q",
        "year": "Y",
    }
    working["__period"] = (
        working[call.date_column]
        .dt.to_period(period_codes[call.frequency])
        .astype(str)
    )
    grouped = working.groupby("__period", sort=True)[call.metric].agg(
        call.aggregation
    )
    return TrendAnalysisResult(
        date_column=call.date_column,
        metric=call.metric,
        frequency=call.frequency,
        aggregation=call.aggregation,
        points=[
            TrendPoint(period=str(period), value=_finite_float(value))
            for period, value in grouped.items()
        ],
    )
