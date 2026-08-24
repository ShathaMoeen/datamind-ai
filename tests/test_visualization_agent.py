"""Tests for Visualization Agent planning and chart generation."""

import asyncio
from uuid import uuid4

import pandas as pd
import pytest

from app.agents.visualization_agent import (
    VisualizationAgent,
    VisualizationPlanningError,
)
from app.models.llm import LLMResponse
from app.models.visualization_agent import ChartType
from app.services.dataset_loader import DatasetLoader
from app.services.fake_llm_client import FakeLLMClient
from app.tools.visualization_tool_registry import VisualizationToolRegistry


def test_visualization_agent_generates_validated_bar_chart(tmp_path) -> None:
    """A valid LLM chart plan should produce Plotly figure JSON."""

    dataset_id = str(uuid4())
    pd.DataFrame(
        {"region": ["West", "East"], "sales": [10, 7]}
    ).to_csv(tmp_path / f"{dataset_id}.csv", index=False)
    llm_client = FakeLLMClient(
        LLMResponse(
            content=(
                '{"chart":{"chart_type":"bar","x":"region",'
                '"y":"sales","aggregation":"sum",'
                '"title":"Sales by region"},"reason":"Compare categories.",'
                '"explanation":"The bar height shows total sales."}'
            ),
            model="fake-model",
        )
    )
    agent = VisualizationAgent(
        llm_client=llm_client,
        registry=VisualizationToolRegistry(DatasetLoader(tmp_path)),
    )

    result = asyncio.run(agent.visualize(dataset_id, "Plot sales by region."))

    assert result.chart_type == ChartType.BAR
    assert result.figure["data"][0]["type"] == "bar"
    assert len(result.trace) == 2
    planner_prompt = llm_client.requests[0].messages[0].content
    assert '"name": "region"' in planner_prompt
    assert '"name": "sales"' in planner_prompt


def test_visualization_agent_rejects_executable_chart_plan(tmp_path) -> None:
    """Executable HTML or JavaScript is not a valid chart type."""

    dataset_id = str(uuid4())
    pd.DataFrame({"sales": [10]}).to_csv(
        tmp_path / f"{dataset_id}.csv", index=False
    )
    llm_client = FakeLLMClient(
        LLMResponse(
            content=(
                '{"chart":{"chart_type":"javascript",'
                '"code":"alert(1)"},"reason":"Unsafe.",'
                '"explanation":"Unsafe."}'
            ),
            model="fake-model",
        )
    )
    agent = VisualizationAgent(
        llm_client=llm_client,
        registry=VisualizationToolRegistry(DatasetLoader(tmp_path)),
    )

    with pytest.raises(VisualizationPlanningError):
        asyncio.run(agent.visualize(dataset_id, "Run JavaScript."))


def test_visualization_agent_replaces_hallucinated_column(tmp_path) -> None:
    """An invented chart column should be replaced by a safe chart."""

    dataset_id = str(uuid4())
    pd.DataFrame({"monthly_income": [7000, 9000]}).to_csv(
        tmp_path / f"{dataset_id}.csv", index=False
    )
    llm_client = FakeLLMClient(
        LLMResponse(
            content=(
                '{"chart":{"chart_type":"bar","x":"mean",'
                '"y":"monthly_income","aggregation":"mean",'
                '"title":"Income"},"reason":"Show income.",'
                '"explanation":"Income mean."}'
            ),
            model="fake-model",
        )
    )
    agent = VisualizationAgent(
        llm_client=llm_client,
        registry=VisualizationToolRegistry(DatasetLoader(tmp_path)),
    )

    result = asyncio.run(agent.visualize(dataset_id, "Show mean income."))

    assert result.chart_type == ChartType.HISTOGRAM
    assert result.figure["data"][0]["type"] == "histogram"
