"""Normalized evaluation results for agents and grounded outputs."""

from pydantic import BaseModel, Field


class EvaluationMetric(BaseModel):
    """One measurable score with transparent numerator and denominator."""

    name: str
    passed: int = Field(ge=0)
    total: int = Field(ge=0)
    score: float = Field(ge=0.0, le=1.0)


class RoutingEvaluationCase(BaseModel):
    """Expected and predicted ordered routes for one benchmark request."""

    expected_agents: list[str]
    predicted_agents: list[str]
