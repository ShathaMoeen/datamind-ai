"""Models returned by the integrated multi-agent workflow."""

from typing import Literal

from pydantic import BaseModel, Field

from app.models.analysis_agent import AnalysisAgentResult
from app.models.data_agent import DataAgentResult
from app.models.orchestrator import OrchestrationDecision
from app.models.rag_agent import RAGAgentResult
from app.models.report_agent import ReportAgentResult
from app.models.visualization_agent import VisualizationAgentResult

SpecialistResult = (
    DataAgentResult
    | AnalysisAgentResult
    | VisualizationAgentResult
    | RAGAgentResult
)


class WorkflowTraceEvent(BaseModel):
    """One observable workflow event without private reasoning."""

    step: int = Field(gt=0)
    action: str
    agent: str | None = None
    status: Literal["success", "failed"]


class AgentWorkflowResult(BaseModel):
    """Complete routed execution and its final grounded report."""

    orchestration: OrchestrationDecision
    specialist_results: list[SpecialistResult]
    final_report: ReportAgentResult
    trace: list[WorkflowTraceEvent]
