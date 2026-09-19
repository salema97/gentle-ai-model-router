"""Decision objects for the deterministic policy ranker.

The decision is the contract surface between the router and its consumers
(the OpenCode write adapter, the telemetry shim, and later the learned
DeBERTa ranker that must beat this baseline).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# The 11 canonical SDD phases (agent_class vocabulary verified in
# docs/telemetry-probe.md §1.2: ``sdd-init`` … ``sdd-onboard``).
CANONICAL_PHASES: tuple[str, ...] = (
    "init",
    "explore",
    "research",
    "propose",
    "spec",
    "design",
    "tasks",
    "apply",
    "verify",
    "archive",
    "onboard",
)


@dataclass(frozen=True)
class TaskContext:
    """Optional task-level context that constrains candidate generation."""

    task_type: str | None = None
    context_tokens: int | None = None
    repo_features: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Alternative:
    """One runner-up candidate that also met the phase quality threshold."""

    model: str
    provider: str
    deployment: str
    effort: str
    score: float
    quality: float
    estimated_tokens: float


@dataclass(frozen=True)
class Decision:
    """A deterministic routing decision for one phase invocation."""

    phase: str
    model: str  # canonical_id, e.g. "anthropic/claude-sonnet-4"
    provider: str
    deployment: str
    effort: str
    score: float
    quality: float
    alternatives: tuple[Alternative, ...]
    reason_codes: tuple[str, ...]
    estimated_tokens: float
    estimated_cost: float
    policy_version: str  # hash of the effective policy configuration

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
