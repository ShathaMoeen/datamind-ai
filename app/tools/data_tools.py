"""Controlled Pandas tools for inspecting uploaded datasets."""

import pandas as pd

from app.models.data_agent import (
    ColumnOutlierResult,
    DuplicateRowsResult,
    MissingColumnResult,
    MissingValuesResult,
    OutlierDetectionResult,
)
from app.models.data_profile import ColumnProfile, DatasetProfile
from app.services.dataset_loader import DatasetLoader


def profile_dataset(dataset_id: str, loader: DatasetLoader) -> DatasetProfile:
    """Calculate a deterministic structural and data-quality summary."""

    dataframe = loader.load(dataset_id)
    return profile_dataframe(dataset_id, dataframe)


def profile_dataframe(dataset_id: str, dataframe: pd.DataFrame) -> DatasetProfile:
    """Calculate profile metrics for an already-loaded DataFrame."""

    row_count, column_count = dataframe.shape
    columns = [
        ColumnProfile(
            name=str(column_name),
            data_type=str(dataframe[column_name].dtype),
            missing_count=int(dataframe[column_name].isna().sum()),
            missing_percentage=(
                round(float(dataframe[column_name].isna().mean() * 100), 2)
                if row_count
                else 0.0
            ),
            unique_count=int(dataframe[column_name].nunique(dropna=True)),
        )
        for column_name in dataframe.columns
    ]

    return DatasetProfile(
        dataset_id=dataset_id,
        row_count=row_count,
        column_count=column_count,
        duplicate_rows=int(dataframe.duplicated().sum()),
        columns=columns,
    )


def detect_missing_values(dataframe: pd.DataFrame) -> MissingValuesResult:
    """Calculate missing counts and percentages for every column."""

    row_count = len(dataframe)
    columns = [
        MissingColumnResult(
            column=str(column),
            missing_count=int(dataframe[column].isna().sum()),
            missing_percentage=(
                round(float(dataframe[column].isna().mean() * 100), 2)
                if row_count
                else 0.0
            ),
        )
        for column in dataframe.columns
    ]
    return MissingValuesResult(row_count=row_count, columns=columns)


def count_duplicate_rows(dataframe: pd.DataFrame) -> DuplicateRowsResult:
    """Count fully duplicated rows without returning their contents."""

    row_count = len(dataframe)
    duplicate_count = int(dataframe.duplicated().sum())
    duplicate_percentage = (
        round(duplicate_count / row_count * 100, 2) if row_count else 0.0
    )
    return DuplicateRowsResult(
        row_count=row_count,
        duplicate_count=duplicate_count,
        duplicate_percentage=duplicate_percentage,
    )


def detect_outliers_iqr(dataframe: pd.DataFrame) -> OutlierDetectionResult:
    """Detect obvious numeric outlier candidates using the 1.5×IQR rule."""

    results: list[ColumnOutlierResult] = []
    for column in dataframe.select_dtypes(include="number").columns:
        values = dataframe[column].dropna()
        if values.empty:
            results.append(
                ColumnOutlierResult(
                    column=str(column),
                    lower_bound=None,
                    upper_bound=None,
                    outlier_count=0,
                )
            )
            continue

        first_quartile = float(values.quantile(0.25))
        third_quartile = float(values.quantile(0.75))
        iqr = third_quartile - first_quartile
        lower_bound = first_quartile - 1.5 * iqr
        upper_bound = third_quartile + 1.5 * iqr
        outlier_count = int(
            ((values < lower_bound) | (values > upper_bound)).sum()
        )
        results.append(
            ColumnOutlierResult(
                column=str(column),
                lower_bound=round(lower_bound, 4),
                upper_bound=round(upper_bound, 4),
                outlier_count=outlier_count,
            )
        )

    return OutlierDetectionResult(columns=results)
