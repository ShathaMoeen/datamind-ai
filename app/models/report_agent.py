"""Grounded analytical report models used by the Report Agent."""

from typing import Any, Literal

from pydantic import BaseModel, Field

ReportSource = Literal[
    "data_agent",
    "analysis_agent",
    "visualization_agent",
    "rag_agent",
]


class ReportFact(BaseModel):
    """One trusted fact produced by a specialist agent or deterministic tool."""

    fact_id: str = Field(min_length=1)
    source_agent: ReportSource
    statement: str = Field(min_length=1)
    value: Any | None = None


class ReportFinding(BaseModel):
    """A factual finding that must reference one or more supplied facts."""

    statement: str = Field(min_length=1, max_length=400)
    fact_ids: list[str] = Field(min_length=1)


class ReportInterpretation(BaseModel):
    """An explicitly labelled LLM interpretation supported by known facts."""

    statement: str = Field(min_length=1, max_length=400)
    supporting_fact_ids: list[str] = Field(min_length=1)


class ReportRecommendation(BaseModel):
    """An advisory action, kept separate from calculated facts."""

    action: str = Field(min_length=1, max_length=400)
    supporting_fact_ids: list[str] = Field(default_factory=list)


class GeneratedReport(BaseModel):
    """Structured report content generated from an allowlisted fact set."""

    executive_summary: str = Field(min_length=1, max_length=800)
    findings: list[ReportFinding] = Field(max_length=5)
    interpretations: list[ReportInterpretation] = Field(max_length=3)
    recommendations: list[ReportRecommendation] = Field(max_length=3)
    limitations: list[str] = Field(max_length=3)


class ReportTraceEvent(BaseModel):
    """Observable report workflow event without private chain-of-thought."""

    step: int = Field(gt=0)
    action: str
    status: Literal["success", "failed"]
    parameters: dict[str, str] = Field(default_factory=dict)


class ReportAgentResult(BaseModel):
    """A generated report together with its validated source facts."""

    selected_agent: Literal["report_agent"] = "report_agent"
    report: GeneratedReport
    source_facts: list[ReportFact]
    reference_validation_passed: bool
    trace: list[ReportTraceEvent]
