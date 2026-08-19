"""Analysis Agent using an LLM planner and deterministic Python tools."""

from pydantic import ValidationError

from app.models.analysis_agent import (
    AnalysisAgentPlan,
    AnalysisAgentResult,
    AnalysisTraceEvent,
)
from app.models.llm import LLMMessage, LLMRequest
from app.services.llm_client import LLMClient
from app.tools.analysis_tool_registry import AnalysisToolRegistry


class AnalysisAgentPlanningError(ValueError):
    """Raised when the LLM does not return a valid analysis plan."""


class AnalysisAgent:
    """Plan statistical analysis and execute only typed Python operations."""

    def __init__(
        self,
        llm_client: LLMClient,
        registry: AnalysisToolRegistry,
    ) -> None:
        self._llm_client = llm_client
        self._registry = registry

    async def analyze(self, dataset_id: str, question: str) -> AnalysisAgentResult:
        """Validate an LLM plan, run calculations, and return a safe trace."""

        request = LLMRequest(
            messages=[
                LLMMessage(
                    role="developer",
                    content=(
                        "Return JSON only with keys tool_calls and reason. Each "
                        "tool call must use one of: describe_numeric with optional "
                        "columns; group_comparison with group_by, metric, aggregation; "
                        "correlation with optional columns; trend_analysis with "
                        "date_column, metric, frequency, aggregation. Never generate "
                        "Python, SQL, shell commands, or unsupported parameters."
                    ),
                ),
                LLMMessage(role="user", content=question),
            ],
            temperature=0.0,
            max_output_tokens=600,
        )
        response = await self._llm_client.generate(request)
        try:
            plan = AnalysisAgentPlan.model_validate_json(response.content)
        except ValidationError as error:
            raise AnalysisAgentPlanningError(
                "The Analysis Agent planner returned an invalid tool plan."
            ) from error

        executions = self._registry.execute_many(dataset_id, plan.tool_calls)
        trace = [
            AnalysisTraceEvent(
                step=1,
                action="validated_analysis_plan",
                parameters={"dataset_id": dataset_id},
                status="success",
            )
        ]
        trace.extend(
            AnalysisTraceEvent(
                step=index,
                action="executed_analysis_tool",
                selected_tool=execution.tool,
                parameters={
                    key: str(value) for key, value in execution.parameters.items()
                },
                status="success",
            )
            for index, execution in enumerate(executions, start=2)
        )
        return AnalysisAgentResult(
            dataset_id=dataset_id,
            plan=plan,
            executions=executions,
            trace=trace,
        )
