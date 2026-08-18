"""OpenAI implementation of the provider-independent LLM client."""

from typing import Any

from openai import AsyncOpenAI

from app.models.llm import LLMRequest, LLMResponse, TokenUsage


class OpenAILLMClient:
    """Generate normalized responses through OpenAI's Responses API."""

    def __init__(self, client: AsyncOpenAI, model: str) -> None:
        self._client = client
        self._model = model

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Convert an internal request to OpenAI format and normalize its result."""

        parameters: dict[str, Any] = {
            "model": self._model,
            "input": [message.model_dump() for message in request.messages],
            "max_output_tokens": request.max_output_tokens,
        }
        if request.temperature is not None:
            parameters["temperature"] = request.temperature

        response = await self._client.responses.create(**parameters)
        usage = response.usage

        return LLMResponse(
            content=response.output_text,
            model=response.model,
            usage=TokenUsage(
                input_tokens=usage.input_tokens if usage else 0,
                output_tokens=usage.output_tokens if usage else 0,
            ),
        )
