"""Tests for deterministic DataMind AI evaluation metrics."""

import pytest

from app.evaluation.metrics import (
    citation_correctness,
    execution_success_rate,
    report_grounding_coverage,
    routing_accuracy,
)
from app.models.evaluation import RoutingEvaluationCase
from app.models.rag import RetrievedChunk
from app.models.rag_agent import RAGCitation
from app.models.report_agent import (
    GeneratedReport,
    ReportFact,
    ReportFinding,
    ReportRecommendation,
)
from app.models.workflow import WorkflowTraceEvent


def test_routing_accuracy_requires_exact_ordered_match() -> None:
    cases = [
        RoutingEvaluationCase(
            expected_agents=["data_agent"], predicted_agents=["data_agent"]
        ),
        RoutingEvaluationCase(
            expected_agents=["analysis_agent", "rag_agent"],
            predicted_agents=["rag_agent", "analysis_agent"],
        ),
    ]

    metric = routing_accuracy(cases)

    assert metric.passed == 1
    assert metric.score == 0.5


def test_citation_correctness_checks_all_metadata() -> None:
    retrieved = [
        RetrievedChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            source="report.pdf",
            page_number=2,
            text="Verified evidence.",
            distance=0.1,
        )
    ]
    citations = [
        RAGCitation(chunk_id="chunk-1", source="report.pdf", page_number=2),
        RAGCitation(chunk_id="chunk-1", source="report.pdf", page_number=3),
    ]

    metric = citation_correctness(citations, retrieved)

    assert metric.passed == 1
    assert metric.score == 0.5


def test_report_grounding_coverage_counts_supported_claims() -> None:
    report = GeneratedReport(
        executive_summary="Summary.",
        findings=[ReportFinding(statement="Verified.", fact_ids=["fact-1"])],
        interpretations=[],
        recommendations=[
            ReportRecommendation(action="Supported action.", supporting_fact_ids=["fact-1"]),
            ReportRecommendation(action="Unsupported action."),
        ],
        limitations=[],
    )
    facts = [
        ReportFact(
            fact_id="fact-1",
            source_agent="analysis_agent",
            statement="Calculated fact.",
        )
    ]

    metric = report_grounding_coverage(report, facts)

    assert metric.passed == 2
    assert metric.total == 3
    assert metric.score == pytest.approx(2 / 3)


def test_execution_success_rate_counts_failed_steps() -> None:
    trace = [
        WorkflowTraceEvent(step=1, action="route", status="success"),
        WorkflowTraceEvent(step=2, action="execute", status="failed"),
        WorkflowTraceEvent(step=3, action="report", status="success"),
    ]

    metric = execution_success_rate(trace)

    assert metric.passed == 2
    assert metric.total == 3
    assert metric.score == pytest.approx(2 / 3)
