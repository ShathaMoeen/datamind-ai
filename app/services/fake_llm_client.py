"""Deterministic LLM client used in tests and local development."""

from app.models.llm import LLMRequest, LLMResponse


class FakeLLMClient:
    """Return a configured response without making a network request."""

    def __init__(self, response: LLMResponse) -> None:
        self._response = response
        self.requests: list[LLMRequest] = []

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Record the request and return a copy of the configured response."""

        self.requests.append(request)
        return self._response.model_copy(deep=True)
