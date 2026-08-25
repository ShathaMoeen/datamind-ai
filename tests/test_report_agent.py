"""Tests for fact-grounded analytical report generation."""

import asyncio

import pytest

from app.agents.report_agent import ReportAgent, ReportValidationError
from app.models.llm import LLMResponse
from app.models.report_agent import ReportFact
from app.services.fake_llm_client import FakeLLMClient


class SequenceLLMClient:
    def __init__(self, contents: list[str]) -> None:
        self._contents = iter(contents)
        self.requests = []

    async def generate(self, request):
        self.requests.append(request)
        return LLMResponse(content=next(self._contents), model="fake-model")


def _facts() -> list[ReportFact]:
    return [
        ReportFact(
            fact_id="analysis-1",
            source_agent="analysis_agent",
            statement="West sales decreased by 20 percent.",
            value=-20.0,
        ),
        ReportFact(
            fact_id="rag-1",
            source_agent="rag_agent",
            statement="The report records a western supply interruption.",
            value={"source": "operations.pdf", "page": 3},
        ),
    ]


def test_report_agent_separates_facts_interpretations_and_advice() -> None:
    """A valid report preserves provenance for findings and interpretations."""

    llm_client = FakeLLMClient(
        LLMResponse(
            content=(
                '{"executive_summary":"Western sales declined.",'
                '"findings":[{"statement":"Sales fell 20 percent.",'
                '"fact_ids":["analysis-1"]}],'
                '"interpretations":[{"statement":"The interruption may have '
                'contributed to the decline.","supporting_fact_ids":'
                '["analysis-1","rag-1"]}],'
                '"recommendations":[{"action":"Review western suppliers.",'
                '"supporting_fact_ids":["rag-1"]}],'
                '"limitations":["The evidence does not prove causation."]}'
            ),
            model="fake-model",
        )
    )

    result = asyncio.run(ReportAgent(llm_client).generate(_facts()))

    assert result.report.findings[0].fact_ids == ["analysis-1"]
    assert result.report.interpretations[0].supporting_fact_ids == [
        "analysis-1",
        "rag-1",
    ]
    assert result.reference_validation_passed is True


def test_report_agent_rejects_invented_fact_reference() -> None:
    """A plausible sentence is rejected if its supporting fact was not supplied."""

    llm_client = FakeLLMClient(
        LLMResponse(
            content=(
                '{"executive_summary":"Sales declined.",'
                '"findings":[{"statement":"Marketing spend fell.",'
                '"fact_ids":["invented-1"]}],'
                '"interpretations":[],"recommendations":[],"limitations":[]}'
            ),
            model="fake-model",
        )
    )

    with pytest.raises(ReportValidationError, match="invented-1"):
        asyncio.run(ReportAgent(llm_client).generate(_facts()))


def test_report_agent_requires_unique_source_facts() -> None:
    """Duplicate IDs are rejected because provenance would be ambiguous."""

    duplicate_facts = [_facts()[0], _facts()[0]]
    llm_client = FakeLLMClient(LLMResponse(content="{}", model="fake-model"))

    with pytest.raises(ReportValidationError, match="unique"):
        asyncio.run(ReportAgent(llm_client).generate(duplicate_facts))


def test_report_agent_requests_selected_output_language() -> None:
    llm_client = FakeLLMClient(
        LLMResponse(
            content=(
                '{"executive_summary":"ملخص", "findings":[],'
                '"interpretations":[],"recommendations":[],"limitations":[]}'
            ),
            model="fake-model",
        )
    )

    asyncio.run(ReportAgent(llm_client).generate(_facts(), language="ar"))

    prompt = llm_client.requests[0].messages[0].content
    assert "in Arabic" in prompt
    assert "Modern Standard Arabic" in prompt
    assert "proofread every Arabic sentence" in prompt
    assert "column names" in prompt


def test_report_agent_repairs_one_invalid_local_model_response() -> None:
    valid_report = (
        '{"executive_summary":"Summary","findings":[],'
        '"interpretations":[],"recommendations":[],"limitations":[]}'
    )
    llm_client = SequenceLLMClient(["{}", valid_report])

    result = asyncio.run(ReportAgent(llm_client).generate(_facts()))

    assert result.report.executive_summary == "Summary"
    assert len(llm_client.requests) == 2
    assert "Validation errors" in llm_client.requests[1].messages[-1].content


def test_report_agent_repairs_vague_categorical_language() -> None:
    facts = [
        ReportFact(
            fact_id="analysis-region-1",
            source_agent="analysis_agent",
            statement="mean of monthly_savings for region 'Riyadh': 2450.0.",
            value={
                "group_by": "region",
                "group": "Riyadh",
                "metric": "monthly_savings",
                "aggregation": "mean",
                "value": 2450.0,
            },
        )
    ]
    vague = (
        '{"executive_summary":"A certain region performed best.",'
        '"findings":[{"statement":"A certain region averaged 2450.",'
        '"fact_ids":["analysis-region-1"]}],"interpretations":[],'
        '"recommendations":[],"limitations":[]}'
    )
    repaired = (
        '{"executive_summary":"Riyadh performed best.",'
        '"findings":[{"statement":"Riyadh averaged 2450 in monthly savings.",'
        '"fact_ids":["analysis-region-1"]}],"interpretations":[],'
        '"recommendations":[],"limitations":[]}'
    )
    llm_client = SequenceLLMClient([vague, repaired])

    result = asyncio.run(ReportAgent(llm_client).generate(facts))

    assert "Riyadh" in result.report.executive_summary
    assert "Riyadh" in result.report.findings[0].statement
    assert len(llm_client.requests) == 2
    assert "exact group labels" in llm_client.requests[1].messages[-1].content
