"""Execute an Orchestrator plan and combine specialist results into a report."""

from typing import Literal

from app.agents.analysis_agent import AnalysisAgent
from app.agents.data_agent import DataAgent
from app.agents.orchestrator import OrchestratorAgent
from app.agents.rag_agent import RAGAgent
from app.agents.report_agent import ReportAgent
from app.agents.visualization_agent import VisualizationAgent
from app.models.analysis_agent import AnalysisAgentResult
from app.models.data_agent import DataAgentResult, DataToolName
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
        language: Literal["ar", "en"] = "en",
        dashboard_mode: bool = False,
    ) -> AgentWorkflowResult:
        """Route, execute selected agents sequentially, and generate a report."""

        decision = await self._orchestrator.route(question, context)
        if dashboard_mode and context.dataset_id is not None:
            for required_agent in ("analysis_agent", "visualization_agent"):
                if required_agent not in decision.plan.agents:
                    decision.plan.agents.append(required_agent)
            decision.trace[0].selected_agents = decision.plan.agents
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

        final_report = await self._report_agent.generate(facts, language=language)
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
            return AgentWorkflow._data_facts(result)
        if isinstance(result, AnalysisAgentResult):
            return AgentWorkflow._analysis_facts(result)
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

    @staticmethod
    def _data_facts(result: DataAgentResult) -> list[ReportFact]:
        """Convert verbose tool payloads into small, unambiguous report facts."""

        facts: list[ReportFact] = []
        for execution in result.executions:
            output = execution.output
            if execution.tool == DataToolName.PROFILE_DATASET:
                columns = output.get("columns", [])
                missing_total = sum(item.get("missing_count", 0) for item in columns)
                values = [
                    ("row_count", "Dataset row count", output.get("row_count", 0)),
                    (
                        "column_count",
                        "Dataset column count",
                        output.get("column_count", 0),
                    ),
                    (
                        "duplicate_rows",
                        "Duplicate row count",
                        output.get("duplicate_rows", 0),
                    ),
                    ("missing_values", "Total missing cell count", missing_total),
                ]
                facts.extend(
                    ReportFact(
                        fact_id=f"data-{name}",
                        source_agent="data_agent",
                        statement=f"{label}: {value}.",
                        value=value,
                    )
                    for name, label, value in values
                )
            elif execution.tool == DataToolName.DETECT_MISSING_VALUES:
                columns = output.get("columns", [])
                total = sum(item.get("missing_count", 0) for item in columns)
                affected = sum(item.get("missing_count", 0) > 0 for item in columns)
                facts.append(
                    ReportFact(
                        fact_id="data-missing-summary",
                        source_agent="data_agent",
                        statement=(
                            f"Missing-value scan found {total} missing cells across "
                            f"{affected} columns."
                        ),
                        value={"missing_cells": total, "affected_columns": affected},
                    )
                )
            elif execution.tool == DataToolName.COUNT_DUPLICATE_ROWS:
                count = output.get("duplicate_count", 0)
                facts.append(
                    ReportFact(
                        fact_id="data-duplicate-summary",
                        source_agent="data_agent",
                        statement=f"Duplicate-row scan found {count} duplicate rows.",
                        value=count,
                    )
                )
            else:
                outliers = sum(
                    item.get("outlier_count", 0)
                    for item in output.get("columns", [])
                )
                facts.append(
                    ReportFact(
                        fact_id="data-outlier-summary",
                        source_agent="data_agent",
                        statement=(
                            f"IQR scan flagged {outliers} potential outlier values."
                        ),
                        value=outliers,
                    )
                )
        return facts

    @staticmethod
    def _analysis_facts(result: AnalysisAgentResult) -> list[ReportFact]:
        """Extract dashboard-ready metrics from deterministic calculations."""

        facts: list[ReportFact] = []
        for execution_index, execution in enumerate(result.executions, start=1):
            fact_count_before = len(facts)
            output = execution.output
            if execution.tool.value == "describe_numeric":
                for column in output.get("columns", [])[:6]:
                    name = str(column.get("column", "metric"))
                    for metric in ("mean", "median", "minimum", "maximum"):
                        value = column.get(metric)
                        if value is not None:
                            rounded = round(float(value), 2)
                            facts.append(
                                ReportFact(
                                    fact_id=(
                                        f"analysis-{execution_index}-{name}-{metric}"
                                    ),
                                    source_agent="analysis_agent",
                                    statement=f"{name} {metric}: {rounded}.",
                                    value=rounded,
                                )
                            )
            elif execution.tool.value == "group_comparison":
                metric = str(output.get("metric", "value"))
                groups = sorted(
                    output.get("groups", []),
                    key=lambda item: item.get("value") or 0,
                    reverse=True,
                )[:5]
                for group_index, group in enumerate(groups, start=1):
                    value = group.get("value")
                    if value is not None:
                        rounded = round(float(value), 2)
                        facts.append(
                            ReportFact(
                                fact_id=(
                                    f"analysis-{execution_index}-group-{group_index}"
                                ),
                                source_agent="analysis_agent",
                                statement=(
                                    f"{group.get('group')} {metric}: {rounded}."
                                ),
                                value=rounded,
                            )
                        )
            else:
                facts.append(
                    ReportFact(
                        fact_id=f"analysis-{execution_index}",
                        source_agent="analysis_agent",
                        statement=(
                            f"{execution.tool.value} produced verified calculations."
                        ),
                        value=output,
                    )
                )
            if len(facts) == fact_count_before:
                facts.append(
                    ReportFact(
                        fact_id=f"analysis-{execution_index}",
                        source_agent="analysis_agent",
                        statement=(
                            f"{execution.tool.value} produced verified calculations."
                        ),
                        value=output,
                    )
                )
        return facts[:16]
