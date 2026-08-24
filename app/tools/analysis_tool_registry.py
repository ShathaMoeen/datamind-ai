"""Validated dispatcher for Analysis Agent tool calls."""

from app.models.analysis_agent import (
    AnalysisToolCall,
    AnalysisToolExecution,
    CorrelationCall,
    DescribeNumericCall,
    GroupComparisonCall,
    TrendAnalysisCall,
)
from app.services.dataset_loader import DatasetLoader
from app.tools.analysis_tools import (
    analyze_trend,
    calculate_correlations,
    describe_numeric,
    group_comparison,
)


class AnalysisToolRegistry:
    """Load a dataset once and dispatch only typed analysis calls."""

    def __init__(self, loader: DatasetLoader) -> None:
        self._loader = loader

    def describe_columns(self, dataset_id: str) -> list[dict[str, str]]:
        """Return the real dataset schema for safe LLM planning."""

        dataframe = self._loader.load(dataset_id)
        return [
            {"name": str(column), "data_type": str(dataframe[column].dtype)}
            for column in dataframe.columns
        ]

    def execute_many(
        self,
        dataset_id: str,
        tool_calls: list[AnalysisToolCall],
    ) -> list[AnalysisToolExecution]:
        """Execute validated calls without accepting arbitrary Python."""

        dataframe = self._loader.load(dataset_id)
        executions = []
        for call in tool_calls:
            if isinstance(call, DescribeNumericCall):
                result = describe_numeric(dataframe, call.columns)
            elif isinstance(call, GroupComparisonCall):
                result = group_comparison(dataframe, call)
            elif isinstance(call, CorrelationCall):
                result = calculate_correlations(dataframe, call.columns)
            elif isinstance(call, TrendAnalysisCall):
                result = analyze_trend(dataframe, call)
            else:
                raise TypeError("Unsupported validated analysis tool call.")

            executions.append(
                AnalysisToolExecution(
                    tool=call.tool,
                    parameters=call.model_dump(mode="json", exclude={"tool"}),
                    output=result.model_dump(mode="json"),
                )
            )
        return executions
