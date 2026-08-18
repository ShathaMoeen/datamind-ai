"""Provider-independent models for language-model interactions."""

from typing import Literal

from pydantic import BaseModel, Field

MessageRole = Literal["system", "developer", "user", "assistant"]


class LLMMessage(BaseModel):
    """A single message sent to or received from a language model."""

    role: MessageRole
    content: str = Field(min_length=1)


class LLMRequest(BaseModel):
    """A provider-independent request for text generation."""

    messages: list[LLMMessage] = Field(min_length=1)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=1_000, gt=0)


class TokenUsage(BaseModel):
    """Normalized token counts reported by an LLM provider."""

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)


class LLMResponse(BaseModel):
    """A normalized response returned by any LLM provider."""

    content: str
    model: str
    usage: TokenUsage = Field(default_factory=TokenUsage)
