"""Contract implemented by all language-model providers."""

from typing import Protocol

from app.models.llm import LLMRequest, LLMResponse


class LLMClient(Protocol):
    """Provider-independent interface for language-model generation."""

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a normalized response for a validated request."""
        ...
