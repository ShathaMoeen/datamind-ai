"""Pure, reproducible metrics for routing, grounding, and execution."""

from collections.abc import Iterable

from app.models.evaluation import EvaluationMetric, RoutingEvaluationCase
from app.models.rag import RetrievedChunk
from app.models.rag_agent import RAGCitation
from app.models.report_agent import GeneratedReport, ReportFact
from app.models.workflow import WorkflowTraceEvent


def _metric(name: str, passed: int, total: int) -> EvaluationMetric:
    """Build a normalized metric; an empty benchmark scores zero."""

    score = passed / total if total else 0.0
    return EvaluationMetric(name=name, passed=passed, total=total, score=score)


def routing_accuracy(cases: Iterable[RoutingEvaluationCase]) -> EvaluationMetric:
    """Measure exact ordered-route matches across benchmark requests."""

    case_list = list(cases)
    passed = sum(
        case.expected_agents == case.predicted_agents for case in case_list
    )
    return _metric("routing_accuracy", passed, len(case_list))


def citation_correctness(
    citations: Iterable[RAGCitation],
    retrieved_chunks: Iterable[RetrievedChunk],
) -> EvaluationMetric:
    """Measure citations whose ID, source, and page match retrieved evidence."""

    citation_list = list(citations)
    retrieved_by_id = {chunk.chunk_id: chunk for chunk in retrieved_chunks}
    passed = 0
    for citation in citation_list:
        chunk = retrieved_by_id.get(citation.chunk_id)
        if (
            chunk is not None
            and citation.source == chunk.source
            and citation.page_number == chunk.page_number
        ):
            passed += 1
    return _metric("citation_correctness", passed, len(citation_list))


def report_grounding_coverage(
    report: GeneratedReport,
    source_facts: Iterable[ReportFact],
) -> EvaluationMetric:
    """Measure report claims with at least one valid supporting fact reference."""

    known_ids = {fact.fact_id for fact in source_facts}
    references = [finding.fact_ids for finding in report.findings]
    references.extend(
        interpretation.supporting_fact_ids
        for interpretation in report.interpretations
    )
    references.extend(
        recommendation.supporting_fact_ids
        for recommendation in report.recommendations
    )
    passed = sum(bool(set(item).intersection(known_ids)) for item in references)
    return _metric("report_grounding_coverage", passed, len(references))


def execution_success_rate(
    trace: Iterable[WorkflowTraceEvent],
) -> EvaluationMetric:
    """Measure successful observable workflow steps."""

    events = list(trace)
    passed = sum(event.status == "success" for event in events)
    return _metric("execution_success_rate", passed, len(events))
