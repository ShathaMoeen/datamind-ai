"""Tests for controlled Data Agent planning and tool execution."""

import asyncio
from uuid import uuid4

import pandas as pd
import pytest

from app.agents.data_agent import DataAgent, DataAgentPlanningError
from app.models.data_agent import DataToolName
from app.models.llm import LLMResponse
from app.services.dataset_loader import DatasetLoader
from app.services.fake_llm_client import FakeLLMClient
from app.tools.data_tool_registry import DataToolRegistry


def test_data_agent_runs_only_validated_tools(tmp_path) -> None:
    """A valid plan should execute allowlisted deterministic tools."""

    dataset_id = str(uuid4())
    dataframe = pd.DataFrame(
        {
            "region": ["West", "West", "East"],
            "sales": [10.0, 10.0, None],
        }
    )
    dataframe.to_csv(tmp_path / f"{dataset_id}.csv", index=False)
    llm_client = FakeLLMClient(
        LLMResponse(
            content=(
                '{"tools":["detect_missing_values","count_duplicate_rows"],'
                '"reason":"Check data quality."}'
            ),
            model="fake-model",
        )
    )
    agent = DataAgent(
        llm_client=llm_client,
        registry=DataToolRegistry(DatasetLoader(tmp_path)),
    )

    result = asyncio.run(agent.analyze(dataset_id, "Check missing and duplicate data."))

    assert [execution.tool for execution in result.executions] == [
        DataToolName.DETECT_MISSING_VALUES,
        DataToolName.COUNT_DUPLICATE_ROWS,
    ]
    assert result.executions[0].output["columns"][1]["missing_count"] == 1
    assert result.executions[1].output["duplicate_count"] == 1
    assert len(result.trace) == 3
    assert len(llm_client.requests) == 1


def test_data_agent_rejects_unknown_tool_plan(tmp_path) -> None:
    """An invented tool name must fail before any tool executes."""

    llm_client = FakeLLMClient(
        LLMResponse(
            content=(
                '{"tools":["execute_arbitrary_python"],'
                '"reason":"Unsafe request."}'
            ),
            model="fake-model",
        )
    )
    agent = DataAgent(
        llm_client=llm_client,
        registry=DataToolRegistry(DatasetLoader(tmp_path)),
    )

    with pytest.raises(DataAgentPlanningError):
        asyncio.run(agent.analyze(str(uuid4()), "Run arbitrary Python."))
