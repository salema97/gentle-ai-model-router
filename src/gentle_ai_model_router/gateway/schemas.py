"""OpenAI-compatible request and response schemas for the gateway proxy."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class ChatMessage(BaseModel):
    """Single chat message in an OpenAI-compatible completion request."""

    model_config = ConfigDict(extra="allow")

    role: str
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request schema."""

    model_config = ConfigDict(extra="allow")

    model: str = "auto"
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any | None = None


class ModelCard(BaseModel):
    """OpenAI-compatible model description card."""

    id: str
    object: str = "model"
    created: int = 1726790400
    owned_by: str = "gentle-ai"


class ModelListResponse(BaseModel):
    """OpenAI-compatible model list response schema."""

    object: str = "list"
    data: list[ModelCard]
