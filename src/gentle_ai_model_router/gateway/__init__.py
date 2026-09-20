"""OpenAI-compatible Gateway Proxy package for gentle-ai-model-router."""

from __future__ import annotations

from gentle_ai_model_router.gateway.proxy import (
    extract_task_and_phase,
    handle_chat_completion,
    list_gateway_models,
    resolve_upstream,
)
from gentle_ai_model_router.gateway.schemas import (
    ChatCompletionRequest,
    ChatMessage,
    ModelCard,
    ModelListResponse,
)

__all__ = [
    "ChatCompletionRequest",
    "ChatMessage",
    "ModelCard",
    "ModelListResponse",
    "extract_task_and_phase",
    "handle_chat_completion",
    "list_gateway_models",
    "resolve_upstream",
]
