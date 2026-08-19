"""Validated routing plans and observable Orchestrator traces."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

AgentName = Literal[
    "data_agent",
    "analysis_agent",
    "visualization_agent",
    "rag_agent",
]


class OrchestrationContext(BaseModel):
    """Resources available to agents for the current request."""

    dataset_id: str | None = None
    document_ids: list[str] = Field(default_factory=list)


class OrchestrationPlan(BaseModel):
    """Ordered, allowlisted agents selected for one request."""

    agents: list[AgentName] = Field(min_length=1, max_length=4)
    reason: str = Field(min_length=1, max_length=300)

    @model_validator(mode="after")
    def agents_must_be_unique(self) -> "OrchestrationPlan":
        """Reject repeated work in an LLM-generated plan."""

        if len(self.agents) != len(set(self.agents)):
            raise ValueError("An orchestration plan cannot repeat an agent.")
        return self


class OrchestratorTraceEvent(BaseModel):
    """One safe routing event, excluding private model reasoning."""

    step: int = Field(gt=0)
    action: str
    status: Literal["success", "rejected"]
    selected_agents: list[AgentName] = Field(default_factory=list)


class OrchestrationDecision(BaseModel):
    """Validated routing decision returned before agent execution."""

    question: str
    context: OrchestrationContext
    plan: OrchestrationPlan
    trace: list[OrchestratorTraceEvent]
