"""OpenAI-compatible Gateway Proxy package for gentle-ai-model-router."""

from __future__ import annotations

from gentle_ai_model_router.gateway.proxy import (
    clear_exhausted_targets,
    extract_task_and_phase,
    handle_chat_completion,
    is_quota_or_rate_limit_error,
    is_target_exhausted,
    list_gateway_models,
    mark_target_exhausted,
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
    "clear_exhausted_targets",
    "extract_task_and_phase",
    "handle_chat_completion",
    "is_quota_or_rate_limit_error",
    "is_target_exhausted",
    "list_gateway_models",
    "mark_target_exhausted",
    "resolve_upstream",
]
