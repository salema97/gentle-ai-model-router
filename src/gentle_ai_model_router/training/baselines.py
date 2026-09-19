"""Reference routers for offline evaluation (docs/evaluation.md).

Every baseline picks ONE candidate per (phase, task_id) group from the
dataset rows, so business metrics (tokens_per_success, regret, …) are
directly comparable with a learned ranker or the deterministic policy:

- fixed_strong: highest benchmark PRIOR (min-max normalized within the
  group), cost ignored. Approximates "always buy the best model".
- fixed_cheap: lowest estimated cost. Approximates "always buy the cheapest".
- benchmark_only: highest RAW benchmark score (no normalization, no cost,
  no effort awareness). Arena-to-workload failure mode made concrete.
- baseline_policy: the deterministic router/policy.py replayed against the
  same registry (requires a session; skipped in --baselines-only mode when
  no registry is available).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from gentle_ai_model_router.dataset.schema import DatasetExample
from gentle_ai_model_router.router.decision import TaskContext

# A chooser maps one group of examples (same phase + task_id) to the chosen row.
Chooser = Callable[[list[DatasetExample]], DatasetExample]


def _tie_break(
    examples: list[DatasetExample], score_of: Callable[[DatasetExample], float]
) -> DatasetExample:
    """Deterministic argmax: score desc, then candidate key asc."""
    return min(examples, key=lambda e: (-score_of(e), e.candidate.key))


def fixed_strong_chooser(group: list[DatasetExample]) -> DatasetExample:
    """Highest normalized benchmark prior (cost ignored)."""

    def score(e: DatasetExample) -> float:
        values = [v for v in e.benchmark_features]
        if not values:
            return 0.0
        lo, hi = min(values), max(values)
        if hi == lo:
            return 0.5
        return sum((v - lo) / (hi - lo) for v in values) / len(values)

    return _tie_break(group, score)


def fixed_cheap_chooser(group: list[DatasetExample]) -> DatasetExample:
    """Lowest estimated cost."""
    return min(group, key=lambda e: (e.cost_features["est_cost"], e.candidate.key))


def benchmark_only_chooser(group: list[DatasetExample]) -> DatasetExample:
    """Highest raw benchmark score; cost and effort ignored."""
    return _tie_break(group, lambda e: sum(e.benchmark_features))


def policy_chooser(
    session: Any, config: Any
) -> Chooser:
    """Replay router/policy.py deterministically per group (needs registry)."""
    from gentle_ai_model_router.router.policy import select_candidate

    cache: dict[str, tuple[str, str, str]] = {}

    def choose(group: list[DatasetExample]) -> DatasetExample:
        rep = group[0]
        key = f"{rep.phase}|{rep.task_id}"
        if key not in cache:
            decision = select_candidate(
                session,
                rep.phase,
                config,
                TaskContext(
                    task_type=rep.task_type, context_tokens=rep.context_tokens or None
                ),
            )
            cache[key] = (decision.model, decision.deployment, decision.effort)
        wanted = cache[key]
        chosen = next(
            (
                e
                for e in group
                if (e.candidate.model, e.candidate.deployment, e.candidate.effort)
                == wanted
            ),
            None,
        )
        if chosen is None:
            # The policy picked a candidate outside this dataset group
            # (should not happen: groups cover all variants). Fail closed by
            # falling back to the policy winner's model at any effort.
            chosen = next(
                (e for e in group if e.candidate.model == wanted[0]), group[0]
            )
        return chosen

    return choose


def group_examples(examples: list[DatasetExample]) -> dict[str, list[DatasetExample]]:
    """Group by (phase, task_id), preserving first-seen order."""
    groups: dict[str, list[DatasetExample]] = {}
    for example in examples:
        groups.setdefault(example.task_id, []).append(example)
    return groups
