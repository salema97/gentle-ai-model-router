"""Reward computation: join shim decisions to executions and score arms.

This is the second half of the bandit loop (see ``training/__init__.py``):
the bandit minimizes ``tokens_per_success`` subject to per-phase quality
floors, so the reward per execution must make success dominant and token
cost + latency explicit penalties:

    reward = task_success
             - token_weight  * total_tokens / 1_000_000
             - latency_weight * latency_ms / 60_000

Success contributes a full 1.0; the penalties are small corrections (a 1M
token execution at the default weight costs 1.0 — never more than one
"success unit" for realistic sizes, and always dominated by the success term
for sub-100k executions). Lower reward is better in the aggregate view only
through ``tokens_per_success``; the bandit ranks arms by ``mean_reward``.

Fail-closed rules (typed :class:`RewardError`, never silently degraded):

- An execution carrying a ``decision_id`` that does not exist in ``decisions``
  is a dangling join and raises — rewards must never be attributed to a
  decision we did not record.
- A decision whose ``alternatives`` payload is not a list of objects raises.
- Executions without a ``decision_id`` or with ``task_success IS NULL`` are
  simply not attributable/scorable and are skipped (documented, not an error).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from gentle_ai_model_router.integration.telemetry_shim import DecisionRecord, ExecutionRecord

# Default penalty weights: one full success unit per 1M tokens / per minute
# of latency. Documented priors, tunable at the call site.
DEFAULT_TOKEN_WEIGHT = 1.0
DEFAULT_LATENCY_WEIGHT = 1.0


class RewardError(Exception):
    """Fatal reward failure (dangling join, invalid payload). Fails closed."""


@dataclass(frozen=True)
class ExecutionReward:
    """One execution's reward plus the join provenance needed for win rates."""

    decision_id: str
    execution_id: str
    phase: str
    model: str
    effort: str
    task_success: int  # 0/1
    total_tokens: int
    latency_ms: float | None
    reward: float
    # ((model, effort), ...) runner-ups recorded on the decision; used to
    # compute win_rate_vs_alternatives.
    alternatives: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class RewardAggregate:
    """Per-(phase, model, effort) arm statistics fed to the bandit."""

    phase: str
    model: str
    effort: str
    executions: int
    success_rate: float
    mean_reward: float
    mean_total_tokens: float
    tokens_per_success: float | None  # None when the arm never succeeded
    win_rate: float | None  # None when the arm never faced recorded alternatives


def _parse_alternatives(decision_id: str, payload: Any) -> tuple[tuple[str, str], ...]:
    if not isinstance(payload, list):
        raise RewardError(
            f"decision {decision_id}: alternatives payload must be a list, "
            f"got {type(payload).__name__}"
        )
    pairs: list[tuple[str, str]] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise RewardError(
                f"decision {decision_id}: alternatives[{idx}] must be an object, "
                f"got {type(item).__name__}"
            )
        model = item.get("model")
        effort = item.get("effort")
        if not isinstance(model, str) or not isinstance(effort, str):
            raise RewardError(
                f"decision {decision_id}: alternatives[{idx}] requires string "
                f"'model' and 'effort'"
            )
        pairs.append((model, effort))
    return tuple(pairs)


def compute_rewards(
    session: Session,
    *,
    token_weight: float = DEFAULT_TOKEN_WEIGHT,
    latency_weight: float = DEFAULT_LATENCY_WEIGHT,
) -> list[ExecutionReward]:
    """Join ``executions`` to ``decisions`` on ``decision_id`` and score each.

    Deterministic order (phase, model, effort, execution_id). Raises
    :class:`RewardError` on dangling joins or invalid decision payloads.
    """
    rows = session.execute(
        select(ExecutionRecord, DecisionRecord)
        .join(DecisionRecord, ExecutionRecord.decision_id == DecisionRecord.decision_id)
        .order_by(
            ExecutionRecord.phase,
            ExecutionRecord.model,
            ExecutionRecord.effort,
            ExecutionRecord.execution_id,
        )
    ).all()

    # Dangling joins: executions pointing at decisions we never recorded.
    # These would silently mis-attribute rewards, so they fail closed.
    dangling = session.execute(
        select(ExecutionRecord.execution_id, ExecutionRecord.decision_id)
        .where(ExecutionRecord.decision_id.is_not(None))
        .where(~ExecutionRecord.decision_id.in_(select(DecisionRecord.decision_id)))
    ).all()
    if dangling:
        ids = ", ".join(str(row.execution_id) for row in dangling[:5])
        raise RewardError(
            f"dangling decision join on {len(dangling)} execution(s): {ids} "
            "(decision_id not present in decisions)"
        )

    rewards: list[ExecutionReward] = []
    for execution, decision in rows:
        if execution.task_success is None:
            continue  # not yet scored by the outcome rubric
        success = int(execution.task_success)
        latency_ms = float(execution.latency_ms) if execution.latency_ms is not None else None
        token_penalty = token_weight * execution.total_tokens / 1_000_000
        latency_penalty = latency_weight * (latency_ms or 0.0) / 60_000
        reward = float(success) - token_penalty - latency_penalty
        rewards.append(
            ExecutionReward(
                decision_id=decision.decision_id,
                execution_id=execution.execution_id,
                phase=execution.phase,
                model=execution.model or decision.selected.get("model", ""),
                effort=execution.effort or decision.selected.get("effort", ""),
                task_success=success,
                total_tokens=execution.total_tokens,
                latency_ms=latency_ms,
                reward=reward,
                alternatives=_parse_alternatives(decision.decision_id, decision.alternatives),
            )
        )
    return rewards


def aggregate_rewards(
    rewards: list[ExecutionReward],
    *,
    min_executions: int = 1,
) -> list[RewardAggregate]:
    """Aggregate execution rewards per (phase, model, effort) arm.

    ``win_rate_vs_alternatives``: among the arm's executions whose decision
    recorded alternatives, the fraction where the arm's own mean reward is at
    least every alternative arm's mean reward (same phase; an alternative
    with no observations is not considered better). ``None`` when the arm
    never faced a decision with alternatives.

    Arms with fewer than ``min_executions`` executions are omitted (the
    bandit treats them as unobserved). Output is sorted by arm key for
    determinism.
    """
    groups: dict[tuple[str, str, str], list[ExecutionReward]] = {}
    for reward in rewards:
        key = (reward.phase, reward.model, reward.effort)
        groups.setdefault(key, []).append(reward)

    means: dict[tuple[str, str, str], float] = {}
    stats: dict[tuple[str, str, str], tuple[int, int, float, float]] = {}
    for key in sorted(groups):
        items = groups[key]
        executions = len(items)
        successes = sum(item.task_success for item in items)
        total_tokens = sum(item.total_tokens for item in items)
        mean_reward = sum(item.reward for item in items) / executions
        stats[key] = (executions, successes, mean_reward, total_tokens)
        means[key] = mean_reward

    aggregates: list[RewardAggregate] = []
    for key in sorted(groups):
        executions, successes, mean_reward, total_tokens = stats[key]
        if executions < min_executions:
            continue
        phase, model, effort = key
        win_rate: float | None = None
        faced = [item for item in groups[key] if item.alternatives]
        if faced:
            wins = 0
            for item in faced:
                best_alt = max(
                    (means.get((phase, alt_model, alt_effort), float("-inf"))
                     for alt_model, alt_effort in item.alternatives),
                    default=float("-inf"),
                )
                if mean_reward >= best_alt:
                    wins += 1
            win_rate = wins / len(faced)
        aggregates.append(
            RewardAggregate(
                phase=phase,
                model=model,
                effort=effort,
                executions=executions,
                success_rate=successes / executions,
                mean_reward=mean_reward,
                mean_total_tokens=total_tokens / executions,
                tokens_per_success=(
                    total_tokens / successes if successes > 0 else None
                ),
                win_rate=win_rate,
            )
        )
    return aggregates
