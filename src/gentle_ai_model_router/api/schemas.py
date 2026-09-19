"""Pydantic wire schemas for the ``/route`` API.

Deterministic by construction: response models serialize in declared field
order, and the policy underneath has no randomness — identical request body
+ same registry content + same policy config ⇒ byte-identical response.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from gentle_ai_model_router.registry.normalize import Effort

_EFFORT_VALUES = tuple(level.value for level in Effort)


class RouteRequest(BaseModel):
    """One routing request.

    ``tools`` and ``budget_policy`` are accepted and logged for provenance but
    NOT consumed by the deterministic baseline policy yet (documented
    extension point for the learned ranker / budget-constrained pass).
    """

    task: str = Field(min_length=1)
    phase: str
    task_type: str | None = None
    context_tokens: int | None = Field(default=None, ge=1)
    repo_features: dict[str, Any] | None = None
    available_models: list[str] | None = None
    available_efforts: list[str] | None = None
    tools: list[str] | None = None
    budget_policy: dict[str, Any] | None = None

    def validate_efforts(self) -> list[str]:
        """Return the effort levels, raising ValueError for non-taxonomy ids.

        Called by the route handler so FastAPI maps the failure to a 422 with
        a clear message listing the valid Effort vocabulary.
        """
        if self.available_efforts is None:
            return []
        invalid = sorted(set(self.available_efforts) - set(_EFFORT_VALUES))
        if invalid:
            raise ValueError(
                f"invalid effort level(s) {invalid}; expected values from the "
                f"Effort taxonomy: {list(_EFFORT_VALUES)}"
            )
        return self.available_efforts


class AlternativeOut(BaseModel):
    """One runner-up candidate that also met the phase quality threshold."""

    model: str
    deployment: str
    effort: str
    score: float


class RouteResponse(BaseModel):
    """Deterministic routing decision + provenance hashes."""

    model: str
    deployment: str
    effort: str
    score: float
    alternatives: list[AlternativeOut]
    reason_codes: list[str]
    estimated_tokens: float
    estimated_cost: float
    policy_version: str
    registry_hash: str


class ExecutionIn(BaseModel):
    """Execution telemetry record submitted by runtime hooks or batch ingestion."""

    execution_id: str = Field(min_length=1)
    session_id: str | None = None
    project_id: str | None = None
    phase: str
    task_type: str | None = None
    model: str | None = None
    deployment: str | None = None
    effort: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    total_tokens: int = 0
    latency_ms: int | None = None
    tool_calls: int = 0
    tool_errors: int = 0
    tests_passed: int | None = None
    tests_failed: int | None = None
    task_success: int | None = None
    quality_score: float | None = None
    escalation_count: int = 0
    repo_features: dict[str, Any] | None = None
    router_version: str | None = None
    decision_id: str | None = None


class ExecutionIngestResponse(BaseModel):
    """Response returned upon ingesting a single execution record."""

    status: str = "ok"
    execution_id: str
    created: bool
    task_success: int | None = None
    quality_score: float | None = None


class BatchExecutionIngestResponse(BaseModel):
    """Response returned upon ingesting multiple execution records in batch."""

    status: str = "ok"
    total: int
    created: int
    updated: int
    execution_ids: list[str]

