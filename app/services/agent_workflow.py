"""Execute an Orchestrator plan and combine specialist results into a report."""

from app.agents.analysis_agent import AnalysisAgent
from app.agents.data_agent import DataAgent
from app.agents.orchestrator import OrchestratorAgent
from app.agents.rag_agent import RAGAgent
from app.agents.report_agent import ReportAgent
from app.agents.visualization_agent import VisualizationAgent
from app.models.analysis_agent import AnalysisAgentResult
from app.models.data_agent import DataAgentResult
from app.models.orchestrator import AgentName, OrchestrationContext
from app.models.report_agent import ReportFact
from app.models.visualization_agent import VisualizationAgentResult
from app.models.workflow import (
    AgentWorkflowResult,
    SpecialistResult,
    WorkflowTraceEvent,
)


class AgentWorkflow:
    """Coordinate specialists in validated order and produce a final report."""

    def __init__(
        self,
        orchestrator: OrchestratorAgent,
        data_agent: DataAgent,
        analysis_agent: AnalysisAgent,
        visualization_agent: VisualizationAgent,
        rag_agent: RAGAgent,
        report_agent: ReportAgent,
    ) -> None:
        self._orchestrator = orchestrator
        self._data_agent = data_agent
        self._analysis_agent = analysis_agent
        self._visualization_agent = visualization_agent
        self._rag_agent = rag_agent
        self._report_agent = report_agent

    async def run(
        self,
        question: str,
        context: OrchestrationContext,
    ) -> AgentWorkflowResult:
        """Route, execute selected agents sequentially, and generate a report."""

        decision = await self._orchestrator.route(question, context)
        trace = [
            WorkflowTraceEvent(
                step=1,
                action="orchestrated_request",
                agent="orchestrator",
                status="success",
            )
        ]
        results: list[SpecialistResult] = []
        facts: list[ReportFact] = []

        for agent_name in decision.plan.agents:
            result = await self._execute_agent(agent_name, question, context)
            results.append(result)
            facts.extend(self._to_facts(result))
            trace.append(
                WorkflowTraceEvent(
                    step=len(trace) + 1,
                    action="executed_specialist",
                    agent=agent_name,
                    status="success",
                )
            )

        final_report = await self._report_agent.generate(facts)
        trace.append(
            WorkflowTraceEvent(
                step=len(trace) + 1,
                action="generated_final_report",
                agent="report_agent",
                status="success",
            )
        )
        return AgentWorkflowResult(
            orchestration=decision,
            specialist_results=results,
            final_report=final_report,
            trace=trace,
        )

    async def _execute_agent(
        self,
        agent_name: AgentName,
        question: str,
        context: OrchestrationContext,
    ) -> SpecialistResult:
        if agent_name == "rag_agent":
            return await self._rag_agent.answer(question, context.document_ids)

        dataset_id = context.dataset_id
        if dataset_id is None:
            raise ValueError(f"{agent_name} requires a dataset ID.")
        if agent_name == "data_agent":
            return await self._data_agent.analyze(dataset_id, question)
        if agent_name == "analysis_agent":
            return await self._analysis_agent.analyze(dataset_id, question)
        return await self._visualization_agent.visualize(dataset_id, question)

    @staticmethod
    def _to_facts(result: SpecialistResult) -> list[ReportFact]:
        if isinstance(result, DataAgentResult):
            return [
                ReportFact(
                    fact_id=f"data-{index}",
                    source_agent="data_agent",
                    statement=f"{execution.tool.value} produced verified results.",
                    value=execution.output,
                )
                for index, execution in enumerate(result.executions, start=1)
            ]
        if isinstance(result, AnalysisAgentResult):
            return [
                ReportFact(
                    fact_id=f"analysis-{index}",
                    source_agent="analysis_agent",
                    statement=f"{execution.tool.value} produced calculated results.",
                    value=execution.output,
                )
                for index, execution in enumerate(result.executions, start=1)
            ]
        if isinstance(result, VisualizationAgentResult):
            return [
                ReportFact(
                    fact_id="visualization-1",
                    source_agent="visualization_agent",
                    statement=result.explanation,
                    value={"chart_type": result.chart_type.value},
                )
            ]
        return [
            ReportFact(
                fact_id="rag-1",
                source_agent="rag_agent",
                statement=result.generated.answer,
                value={
                    "status": result.generated.status,
                    "citations": [
                        citation.model_dump(mode="json")
                        for citation in result.generated.citations
                    ],
                },
            )
        ]
