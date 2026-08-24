"""Integration tests for routed multi-agent execution."""

import asyncio
from typing import Any

from app.models.analysis_agent import (
    AnalysisAgentPlan,
    AnalysisAgentResult,
    AnalysisToolExecution,
    AnalysisToolName,
    DescribeNumericCall,
)
from app.models.orchestrator import (
    OrchestrationContext,
    OrchestrationDecision,
    OrchestrationPlan,
    OrchestratorTraceEvent,
)
from app.models.rag_agent import RAGAgentResult, RAGGeneratedAnswer
from app.models.report_agent import GeneratedReport, ReportAgentResult
from app.services.agent_workflow import AgentWorkflow


class FakeOrchestrator:
    async def route(
        self, question: str, context: OrchestrationContext
    ) -> OrchestrationDecision:
        return OrchestrationDecision(
            question=question,
            context=context,
            plan=OrchestrationPlan(
                agents=["analysis_agent", "rag_agent"],
                reason="Compare data, then retrieve supporting evidence.",
            ),
            trace=[
                OrchestratorTraceEvent(
                    step=1,
                    action="validated_agent_routing",
                    status="success",
                    selected_agents=["analysis_agent", "rag_agent"],
                )
            ],
        )


class FakeAnalysisAgent:
    async def analyze(self, dataset_id: str, question: str) -> AnalysisAgentResult:
        return AnalysisAgentResult(
            dataset_id=dataset_id,
            plan=AnalysisAgentPlan(
                tool_calls=[DescribeNumericCall(tool="describe_numeric")],
                reason="Calculate a verified metric.",
            ),
            executions=[
                AnalysisToolExecution(
                    tool=AnalysisToolName.DESCRIBE_NUMERIC,
                    parameters={},
                    output={"sales_mean": 42.0},
                )
            ],
            trace=[],
        )


class FakeRAGAgent:
    async def answer(
        self, question: str, document_ids: list[str] | None = None
    ) -> RAGAgentResult:
        return RAGAgentResult(
            question=question,
            generated=RAGGeneratedAnswer(
                status="insufficient_evidence",
                answer="No supporting document evidence was found.",
                citations=[],
                limitations=["No relevant chunks."],
            ),
            retrieved_chunks=[],
            citation_validation_passed=True,
            trace=[],
        )


class FakeReportAgent:
    def __init__(self) -> None:
        self.received_facts: list[Any] = []

    async def generate(
        self, facts: list[Any], language: str = "en"
    ) -> ReportAgentResult:
        self.received_facts = facts
        return ReportAgentResult(
            report=GeneratedReport(
                executive_summary="Analysis completed.",
                findings=[],
                interpretations=[],
                recommendations=[],
                limitations=[],
            ),
            source_facts=facts,
            reference_validation_passed=True,
            trace=[],
        )


def test_workflow_executes_route_and_builds_report_facts() -> None:
    """Specialists run in route order and their outputs become report facts."""

    report_agent = FakeReportAgent()
    workflow = AgentWorkflow(
        orchestrator=FakeOrchestrator(),  # type: ignore[arg-type]
        data_agent=object(),  # type: ignore[arg-type]
        analysis_agent=FakeAnalysisAgent(),  # type: ignore[arg-type]
        visualization_agent=object(),  # type: ignore[arg-type]
        rag_agent=FakeRAGAgent(),  # type: ignore[arg-type]
        report_agent=report_agent,  # type: ignore[arg-type]
    )

    result = asyncio.run(
        workflow.run(
            "Analyze sales and check the report.",
            OrchestrationContext(
                dataset_id="dataset-1",
                document_ids=["document-1"],
            ),
        )
    )

    assert [item.selected_agent for item in result.specialist_results] == [
        "analysis_agent",
        "rag_agent",
    ]
    assert [fact.fact_id for fact in report_agent.received_facts] == [
        "analysis-1",
        "rag-1",
    ]
    assert result.trace[-1].agent == "report_agent"
