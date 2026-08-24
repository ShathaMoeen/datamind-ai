"""Data Agent that plans and runs allowlisted dataset inspection tools."""

from pydantic import ValidationError

from app.models.data_agent import (
    AgentTraceEvent,
    DataAgentPlan,
    DataAgentResult,
)
from app.models.llm import LLMMessage, LLMRequest
from app.services.llm_client import LLMClient
from app.tools.data_tool_registry import DataToolRegistry


class DataAgentPlanningError(ValueError):
    """Raised when the LLM returns an invalid or unsafe tool plan."""


class DataAgent:
    """Select and execute controlled Pandas tools for one user question."""

    def __init__(self, llm_client: LLMClient, registry: DataToolRegistry) -> None:
        self._llm_client = llm_client
        self._registry = registry

    async def analyze(self, dataset_id: str, question: str) -> DataAgentResult:
        """Plan with an LLM, validate the plan, and run deterministic tools."""

        allowed_tools = ", ".join(tool.value for tool in self._registry.allowed_tools)
        request = LLMRequest(
            messages=[
                LLMMessage(
                    role="developer",
                    content=(
                        "You plan dataset inspection tasks. Return JSON only with "
                        "this shape: {\"tools\": [\"tool_name\"], "
                        "\"reason\": \"short explanation\"}. "
                        f"Allowed tools: {allowed_tools}. Never invent tool names."
                    ),
                ),
                LLMMessage(role="user", content=question),
            ],
            temperature=0.0,
            max_output_tokens=300,
            response_schema=DataAgentPlan.model_json_schema(),
        )
        response = await self._llm_client.generate(request)

        try:
            plan = DataAgentPlan.model_validate_json(response.content)
        except ValidationError as error:
            raise DataAgentPlanningError(
                "The Data Agent planner returned an invalid tool plan."
            ) from error

        executions = self._registry.execute_many(dataset_id, plan.tools)
        trace = [
            AgentTraceEvent(
                step=1,
                action="validated_tool_plan",
                parameters={"dataset_id": dataset_id},
                status="success",
            )
        ]
        trace.extend(
            AgentTraceEvent(
                step=index,
                action="executed_data_tool",
                selected_tool=execution.tool,
                parameters={"dataset_id": dataset_id},
                status="success",
            )
            for index, execution in enumerate(executions, start=2)
        )

        return DataAgentResult(
            dataset_id=dataset_id,
            plan=plan,
            executions=executions,
            trace=trace,
        )
