"""Allowlisted registry connecting Data Agent choices to Python functions."""

from collections.abc import Callable

import pandas as pd
from pydantic import BaseModel

from app.models.data_agent import DataToolExecution, DataToolName
from app.services.dataset_loader import DatasetLoader
from app.tools.data_tools import (
    count_duplicate_rows,
    detect_missing_values,
    detect_outliers_iqr,
    profile_dataframe,
)

DataToolFunction = Callable[[str, pd.DataFrame], BaseModel]


def _profile(dataset_id: str, dataframe: pd.DataFrame) -> BaseModel:
    return profile_dataframe(dataset_id, dataframe)


def _missing_values(_: str, dataframe: pd.DataFrame) -> BaseModel:
    return detect_missing_values(dataframe)


def _duplicates(_: str, dataframe: pd.DataFrame) -> BaseModel:
    return count_duplicate_rows(dataframe)


def _outliers(_: str, dataframe: pd.DataFrame) -> BaseModel:
    return detect_outliers_iqr(dataframe)


class DataToolRegistry:
    """Execute only known tools against one safely resolved dataset."""

    def __init__(self, loader: DatasetLoader) -> None:
        self._loader = loader
        self._tools: dict[DataToolName, DataToolFunction] = {
            DataToolName.PROFILE_DATASET: _profile,
            DataToolName.DETECT_MISSING_VALUES: _missing_values,
            DataToolName.COUNT_DUPLICATE_ROWS: _duplicates,
            DataToolName.DETECT_OUTLIERS_IQR: _outliers,
        }

    @property
    def allowed_tools(self) -> tuple[DataToolName, ...]:
        """Return the immutable list exposed to the planner."""

        return tuple(self._tools)

    def execute_many(
        self,
        dataset_id: str,
        selected_tools: list[DataToolName],
    ) -> list[DataToolExecution]:
        """Load the dataset once and run each validated tool."""

        dataframe = self._loader.load(dataset_id)
        return [
            DataToolExecution(
                tool=tool_name,
                output=self._tools[tool_name](dataset_id, dataframe).model_dump(
                    mode="json"
                ),
            )
            for tool_name in selected_tools
        ]
