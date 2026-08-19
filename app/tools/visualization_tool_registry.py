"""Controlled bridge between Visualization Agent plans and Plotly."""

from app.models.visualization_agent import ChartSpec
from app.services.dataset_loader import DatasetLoader
from app.tools.visualization_tools import create_chart


class VisualizationToolRegistry:
    """Load one safe dataset and generate one validated Plotly figure."""

    def __init__(self, loader: DatasetLoader) -> None:
        self._loader = loader

    def create(self, dataset_id: str, spec: ChartSpec) -> dict:
        """Generate chart JSON without accepting arbitrary HTML or JavaScript."""

        dataframe = self._loader.load(dataset_id)
        return create_chart(dataframe, spec)
