"""Visualization Agent selecting validated chart specifications."""

from pydantic import ValidationError

from app.models.llm import LLMMessage, LLMRequest
from app.models.visualization_agent import (
    VisualizationAgentResult,
    VisualizationPlan,
    VisualizationTraceEvent,
)
from app.services.llm_client import LLMClient
from app.tools.visualization_tool_registry import VisualizationToolRegistry


class VisualizationPlanningError(ValueError):
    """Raised when an LLM returns an invalid or unsafe chart plan."""


class VisualizationAgent:
    """Select a chart and delegate generation to a controlled Plotly tool."""

    def __init__(
        self,
        llm_client: LLMClient,
        registry: VisualizationToolRegistry,
    ) -> None:
        self._llm_client = llm_client
        self._registry = registry

    async def visualize(
        self,
        dataset_id: str,
        question: str,
    ) -> VisualizationAgentResult:
        """Validate a chart plan, create its figure, and expose a safe trace."""

        request = LLMRequest(
            messages=[
                LLMMessage(
                    role="developer",
                    content=(
                        "Return JSON only with chart, reason, and explanation. "
                        "chart.chart_type must be bar, line, scatter, histogram, "
                        "box, or correlation_heatmap. Use only the relevant "
                        "column names and typed chart parameters. Never return "
                        "HTML, JavaScript, Python, or executable code."
                    ),
                ),
                LLMMessage(role="user", content=question),
            ],
            temperature=0.0,
            max_output_tokens=500,
        )
        response = await self._llm_client.generate(request)
        try:
            plan = VisualizationPlan.model_validate_json(response.content)
        except ValidationError as error:
            raise VisualizationPlanningError(
                "The Visualization Agent returned an invalid chart plan."
            ) from error

        figure = self._registry.create(dataset_id, plan.chart)
        trace = [
            VisualizationTraceEvent(
                step=1,
                action="validated_chart_specification",
                chart_type=plan.chart.chart_type,
                parameters={"dataset_id": dataset_id},
                status="success",
            ),
            VisualizationTraceEvent(
                step=2,
                action="generated_plotly_figure",
                chart_type=plan.chart.chart_type,
                parameters={
                    key: str(value)
                    for key, value in plan.chart.model_dump(
                        mode="json", exclude={"chart_type", "title"}
                    ).items()
                    if value is not None
                },
                status="success",
            ),
        ]
        return VisualizationAgentResult(
            dataset_id=dataset_id,
            chart_type=plan.chart.chart_type,
            explanation=plan.explanation,
            figure=figure,
            trace=trace,
        )
