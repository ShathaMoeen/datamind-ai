"""Tests for safe, structured Orchestrator routing."""

import asyncio

import pytest

from app.agents.orchestrator import OrchestrationPlanningError, OrchestratorAgent
from app.models.llm import LLMResponse
from app.models.orchestrator import OrchestrationContext
from app.services.fake_llm_client import FakeLLMClient


def test_orchestrator_routes_cross_source_request_in_order() -> None:
    """A combined request may select multiple specialists in execution order."""

    llm_client = FakeLLMClient(
        LLMResponse(
            content=(
                '{"agents":["analysis_agent","rag_agent"],'
                '"reason":"Calculate the trend, then find document evidence."}'
            ),
            model="fake-model",
        )
    )
    orchestrator = OrchestratorAgent(llm_client)

    decision = asyncio.run(
        orchestrator.route(
            "Did sales decline, and do the reports explain why?",
            OrchestrationContext(
                dataset_id="dataset-123",
                document_ids=["document-456"],
            ),
        )
    )

    assert decision.plan.agents == ["analysis_agent", "rag_agent"]
    assert decision.trace[0].selected_agents == decision.plan.agents
    assert llm_client.requests[0].temperature == 0.0


def test_orchestrator_rejects_unknown_agent() -> None:
    """An LLM cannot expand its own permissions by inventing an agent."""

    llm_client = FakeLLMClient(
        LLMResponse(
            content='{"agents":["shell_agent"],"reason":"Run commands."}',
            model="fake-model",
        )
    )

    with pytest.raises(OrchestrationPlanningError):
        asyncio.run(
            OrchestratorAgent(llm_client).route(
                "Inspect the computer.",
                OrchestrationContext(),
            )
        )


def test_orchestrator_rejects_route_without_required_dataset() -> None:
    """A valid agent name is still rejected if its required input is absent."""

    llm_client = FakeLLMClient(
        LLMResponse(
            content=(
                '{"agents":["visualization_agent"],'
                '"reason":"The user requested a chart."}'
            ),
            model="fake-model",
        )
    )

    with pytest.raises(OrchestrationPlanningError, match="uploaded dataset"):
        asyncio.run(
            OrchestratorAgent(llm_client).route(
                "Create a sales chart.",
                OrchestrationContext(),
            )
        )


def test_orchestrator_rejects_duplicate_agents() -> None:
    """Repeated agents are invalid because they would duplicate work."""

    llm_client = FakeLLMClient(
        LLMResponse(
            content=(
                '{"agents":["data_agent","data_agent"],'
                '"reason":"Inspect twice."}'
            ),
            model="fake-model",
        )
    )

    with pytest.raises(OrchestrationPlanningError):
        asyncio.run(
            OrchestratorAgent(llm_client).route(
                "Inspect data.",
                OrchestrationContext(dataset_id="dataset-123"),
            )
        )
