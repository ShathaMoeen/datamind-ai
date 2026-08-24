"""Tests for Analysis Agent planning and controlled execution."""

import asyncio
from uuid import uuid4

import pandas as pd
import pytest

from app.agents.analysis_agent import AnalysisAgent, AnalysisAgentPlanningError
from app.models.analysis_agent import AnalysisToolName
from app.models.llm import LLMResponse
from app.services.dataset_loader import DatasetLoader
from app.services.fake_llm_client import FakeLLMClient
from app.tools.analysis_tool_registry import AnalysisToolRegistry


def test_analysis_agent_executes_typed_group_comparison(tmp_path) -> None:
    """A valid structured plan should execute deterministic aggregation."""

    dataset_id = str(uuid4())
    pd.DataFrame(
        {"region": ["West", "West", "East"], "sales": [10, 15, 7]}
    ).to_csv(tmp_path / f"{dataset_id}.csv", index=False)
    llm_client = FakeLLMClient(
        LLMResponse(
            content=(
                '{"tool_calls":[{"tool":"group_comparison",'
                '"group_by":"region","metric":"sales",'
                '"aggregation":"sum"}],"reason":"Compare regions."}'
            ),
            model="fake-model",
        )
    )
    agent = AnalysisAgent(
        llm_client=llm_client,
        registry=AnalysisToolRegistry(DatasetLoader(tmp_path)),
    )

    result = asyncio.run(agent.analyze(dataset_id, "Compare regional sales."))

    assert result.executions[0].tool == AnalysisToolName.GROUP_COMPARISON
    assert result.executions[0].output["groups"] == [
        {"group": "East", "value": 7.0},
        {"group": "West", "value": 25.0},
    ]
    assert len(result.trace) == 2
    planner_prompt = llm_client.requests[0].messages[0].content
    assert '"name": "region"' in planner_prompt
    assert '"name": "sales"' in planner_prompt


def test_analysis_agent_rejects_untyped_python_plan(tmp_path) -> None:
    """An arbitrary-code plan must fail schema validation before execution."""

    dataset_id = str(uuid4())
    pd.DataFrame({"sales": [10]}).to_csv(
        tmp_path / f"{dataset_id}.csv", index=False
    )
    llm_client = FakeLLMClient(
        LLMResponse(
            content=(
                '{"tool_calls":[{"tool":"run_python",'
                '"code":"import os"}],"reason":"Unsafe."}'
            ),
            model="fake-model",
        )
    )
    agent = AnalysisAgent(
        llm_client=llm_client,
        registry=AnalysisToolRegistry(DatasetLoader(tmp_path)),
    )

    with pytest.raises(AnalysisAgentPlanningError):
        asyncio.run(agent.analyze(dataset_id, "Run Python."))


def test_analysis_agent_removes_hallucinated_statistic_column(tmp_path) -> None:
    """Statistics such as mean must not be treated as dataset columns."""

    dataset_id = str(uuid4())
    pd.DataFrame({"monthly_income": [7000, 9000]}).to_csv(
        tmp_path / f"{dataset_id}.csv", index=False
    )
    llm_client = FakeLLMClient(
        LLMResponse(
            content=(
                '{"tool_calls":[{"tool":"describe_numeric","columns":'
                '["monthly_income","mean"]}],"reason":"Summarize income."}'
            ),
            model="fake-model",
        )
    )
    agent = AnalysisAgent(
        llm_client=llm_client,
        registry=AnalysisToolRegistry(DatasetLoader(tmp_path)),
    )

    result = asyncio.run(agent.analyze(dataset_id, "Calculate mean income."))

    assert result.plan.tool_calls[0].columns == ["monthly_income"]
    assert result.executions[0].output["columns"][0]["mean"] == 8000.0
