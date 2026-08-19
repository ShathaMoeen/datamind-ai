"""Controlled Pandas tools for inspecting uploaded datasets."""

from app.models.data_profile import ColumnProfile, DatasetProfile
from app.services.dataset_loader import DatasetLoader


def profile_dataset(dataset_id: str, loader: DatasetLoader) -> DatasetProfile:
    """Calculate a deterministic structural and data-quality summary."""

    dataframe = loader.load(dataset_id)
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
