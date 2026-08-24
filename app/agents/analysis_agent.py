"""Analysis Agent using an LLM planner and deterministic Python tools."""

import json

from pydantic import ValidationError

from app.models.analysis_agent import (
    AnalysisAgentPlan,
    AnalysisAgentResult,
    AnalysisTraceEvent,
    CorrelationCall,
    DescribeNumericCall,
    GroupComparisonCall,
    TrendAnalysisCall,
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

    @staticmethod
    def _sanitize_plan(
        plan: AnalysisAgentPlan,
        dataset_schema: list[dict[str, str]],
    ) -> AnalysisAgentPlan:
        """Remove hallucinated column names before deterministic execution."""

        available = {column["name"] for column in dataset_schema}
        numeric = {
            column["name"]
            for column in dataset_schema
            if any(
                marker in column["data_type"].lower()
                for marker in ("int", "float", "decimal", "number")
            )
        }
        safe_calls = []
        for call in plan.tool_calls:
            if isinstance(call, DescribeNumericCall):
                columns = (
                    [column for column in call.columns if column in numeric]
                    if call.columns is not None
                    else None
                )
                safe_calls.append(call.model_copy(update={"columns": columns or None}))
            elif isinstance(call, CorrelationCall):
                columns = (
                    [column for column in call.columns if column in numeric]
                    if call.columns is not None
                    else None
                )
                safe_calls.append(
                    call.model_copy(
                        update={"columns": columns if len(columns or []) >= 2 else None}
                    )
                )
            elif isinstance(call, GroupComparisonCall):
                metric_is_safe = call.metric in available and (
                    call.aggregation == "count" or call.metric in numeric
                )
                if call.group_by in available and metric_is_safe:
                    safe_calls.append(call)
            elif isinstance(call, TrendAnalysisCall):
                if call.date_column in available and call.metric in numeric:
                    safe_calls.append(call)

        if not safe_calls:
            safe_calls = [DescribeNumericCall(tool="describe_numeric", columns=None)]
        return plan.model_copy(update={"tool_calls": safe_calls})

    async def analyze(self, dataset_id: str, question: str) -> AnalysisAgentResult:
        """Validate an LLM plan, run calculations, and return a safe trace."""

        dataset_schema = self._registry.describe_columns(dataset_id)

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
                        "Python, SQL, shell commands, or unsupported parameters. "
                        "Use ONLY exact column names from this dataset schema; do "
                        "not translate, rename, infer, or invent columns: "
                        f"{json.dumps(dataset_schema, ensure_ascii=False)}"
                    ),
                ),
                LLMMessage(role="user", content=question),
            ],
            temperature=0.0,
            max_output_tokens=600,
            response_schema=AnalysisAgentPlan.model_json_schema(),
        )
        response = await self._llm_client.generate(request)
        try:
            plan = AnalysisAgentPlan.model_validate_json(response.content)
        except ValidationError as error:
            raise AnalysisAgentPlanningError(
                "The Analysis Agent planner returned an invalid tool plan."
            ) from error

        plan = self._sanitize_plan(plan, dataset_schema)

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
