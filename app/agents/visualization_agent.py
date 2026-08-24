"""Visualization Agent selecting validated chart specifications."""

import json

from pydantic import ValidationError

from app.models.llm import LLMMessage, LLMRequest
from app.models.visualization_agent import (
    BarChartSpec,
    BoxChartSpec,
    CorrelationHeatmapSpec,
    HistogramChartSpec,
    LineChartSpec,
    ScatterChartSpec,
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

    @staticmethod
    def _sanitize_plan(
        plan: VisualizationPlan,
        dataset_schema: list[dict[str, str]],
    ) -> VisualizationPlan:
        """Replace hallucinated chart columns with a safe numeric chart."""

        available = {column["name"] for column in dataset_schema}
        numeric = [
            column["name"]
            for column in dataset_schema
            if any(
                marker in column["data_type"].lower()
                for marker in ("int", "float", "decimal", "number")
            )
        ]
        chart = plan.chart
        color = getattr(chart, "color", None)
        if color not in available:
            color = None

        valid = False
        if isinstance(chart, BarChartSpec):
            valid = chart.x in available and (
                chart.aggregation == "count" or chart.y in numeric
            )
        elif isinstance(chart, LineChartSpec):
            valid = chart.x in available and chart.y in numeric
        elif isinstance(chart, ScatterChartSpec):
            valid = chart.x in numeric and chart.y in numeric
        elif isinstance(chart, HistogramChartSpec):
            valid = chart.x in numeric
        elif isinstance(chart, BoxChartSpec):
            valid = chart.y in numeric and (
                chart.x is None or chart.x in available
            )
        elif isinstance(chart, CorrelationHeatmapSpec):
            columns = (
                [column for column in chart.columns if column in numeric]
                if chart.columns is not None
                else numeric
            )
            if len(columns) >= 2:
                return plan.model_copy(
                    update={"chart": chart.model_copy(update={"columns": columns})}
                )

        if valid:
            if hasattr(chart, "color"):
                chart = chart.model_copy(update={"color": color})
            return plan.model_copy(update={"chart": chart})
        if not numeric:
            raise VisualizationPlanningError(
                "No numeric dataset column is available for a safe chart."
            )
        fallback = HistogramChartSpec(
            chart_type="histogram",
            x=numeric[0],
            title=f"Distribution of {numeric[0]}",
        )
        return plan.model_copy(
            update={
                "chart": fallback,
                "explanation": (
                    "A safe distribution chart was selected using an available "
                    "numeric dataset column."
                ),
            }
        )

    async def visualize(
        self,
        dataset_id: str,
        question: str,
    ) -> VisualizationAgentResult:
        """Validate a chart plan, create its figure, and expose a safe trace."""

        dataset_schema = self._registry.describe_columns(dataset_id)

        request = LLMRequest(
            messages=[
                LLMMessage(
                    role="developer",
                    content=(
                        "Return JSON only with chart, reason, and explanation. "
                        "chart.chart_type must be bar, line, scatter, histogram, "
                        "box, or correlation_heatmap. Use only the relevant "
                        "column names and typed chart parameters. Never return "
                        "HTML, JavaScript, Python, or executable code. Use ONLY "
                        "exact column names from this dataset schema; do not "
                        "translate, rename, infer, or invent columns: "
                        f"{json.dumps(dataset_schema, ensure_ascii=False)} "
                        "If the user's question is Arabic, write the chart title, "
                        "reason, and explanation in clear Modern Standard Arabic; "
                        "proofread spelling, grammar, and punctuation before "
                        "returning JSON. Preserve dataset column names exactly."
                    ),
                ),
                LLMMessage(role="user", content=question),
            ],
            temperature=0.0,
            max_output_tokens=500,
            response_schema=VisualizationPlan.model_json_schema(),
        )
        response = await self._llm_client.generate(request)
        try:
            plan = VisualizationPlan.model_validate_json(response.content)
        except ValidationError as error:
            raise VisualizationPlanningError(
                "The Visualization Agent returned an invalid chart plan."
            ) from error

        plan = self._sanitize_plan(plan, dataset_schema)

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
