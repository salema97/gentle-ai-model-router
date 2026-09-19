"""Bandit tests: cold start, UCB reorder, constraint preservation, server fallback."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from gentle_ai_model_router.api.server import create_app
from gentle_ai_model_router.integration import telemetry_shim
from gentle_ai_model_router.registry import db as registry_db
from gentle_ai_model_router.registry.models import Deployment, Model, ModelVariant, Provider
from gentle_ai_model_router.router.bandit import (
    REASON_COLD_START,
    REASON_UCB,
    BanditConfig,
    apply_bandit,
    bandit_version,
)
from gentle_ai_model_router.router.config import RouterConfig, load_config
from gentle_ai_model_router.router.policy import CandidateRanking, RankedCandidate
from gentle_ai_model_router.router.reward import RewardAggregate

AA = "artificial_analysis_intelligence_index"


# --------------------------------------------------------------------------- #
# Synthetic ranking fixtures (no DB needed for the pure bandit core)
# --------------------------------------------------------------------------- #


def _candidate(canonical: str, effort: str = "low") -> RankedCandidate:
    model = Model(id=1, canonical_id=canonical, name=canonical)
    provider = Provider(id=1, registry_key="test", name="test")
    deployment = Deployment(id=1, model_id=1, provider_id=1, deployment_ref="default")
    variant = ModelVariant(id=1, deployment_id=1, effort=effort, provider_value=effort)
    return RankedCandidate(
        model=model,
        provider=provider,
        deployment=deployment,
        variant=variant,
        prior=0.8,
        prior_missing=False,
        quality=0.9,
        estimated_tokens=14_000.0,
        estimated_cost=0.05,
        score=0.85,
        reason_codes=("meets_threshold:low",),
    )


def _ranking(*canonicals: str) -> CandidateRanking:
    return CandidateRanking(
        phase="apply",
        threshold=0.75,
        candidates=tuple(_candidate(c) for c in canonicals),
        pairs_considered=len(canonicals),
        raw_count=len(canonicals),
        policy_version="pol-test",
    )


def _aggregate(model: str, *, executions: int, successes: int,
               mean_reward: float | None = None) -> RewardAggregate:
    if mean_reward is None:
        mean_reward = successes / executions - 0.000_5  # small token penalty
    total_tokens = executions * 1000
    return RewardAggregate(
        phase="apply",
        model=model,
        effort="low",
        executions=executions,
        success_rate=successes / executions,
        mean_reward=mean_reward,
        mean_total_tokens=1000.0,
        tokens_per_success=(total_tokens / successes if successes > 0 else None),
        win_rate=None,
    )


# --------------------------------------------------------------------------- #
# Cold start
# --------------------------------------------------------------------------- #


def test_cold_start_empty_aggregates_is_byte_identical() -> None:
    ranking = _ranking("test/a", "test/b")
    result = apply_bandit(ranking, [], BanditConfig())
    assert result.ranking is ranking  # unchanged object: byte-identical
    assert result.applied is False
    assert result.reason_code == REASON_COLD_START
    assert result.bandit_version.startswith("ban-")


def test_cold_start_insufficient_executions() -> None:
    ranking = _ranking("test/a", "test/b")
    aggregates = [_aggregate("test/a", executions=2, successes=2)]
    config = BanditConfig(min_executions_before_exploit=3)
    result = apply_bandit(ranking, aggregates, config)
    assert result.ranking is ranking
    assert result.applied is False
    assert result.reason_code == REASON_COLD_START


def test_cold_start_ignores_other_phases() -> None:
    """Aggregates for a different phase must not warm up this phase."""
    ranking = _ranking("test/a")
    other = RewardAggregate(
        phase="verify", model="test/a", effort="low", executions=100,
        success_rate=1.0, mean_reward=0.9, mean_total_tokens=1000.0,
        tokens_per_success=1000.0, win_rate=None,
    )
    result = apply_bandit(ranking, [other], BanditConfig())
    assert result.reason_code == REASON_COLD_START


# --------------------------------------------------------------------------- #
# UCB reordering + constraints
# --------------------------------------------------------------------------- #


def test_ucb_reorders_by_reward() -> None:
    ranking = _ranking("test/a", "test/b")  # policy prefers a
    aggregates = [
        _aggregate("test/a", executions=10, successes=8),  # mean ~0.8
        _aggregate("test/b", executions=10, successes=10),  # mean ~1.0
    ]
    result = apply_bandit(ranking, aggregates, BanditConfig(exploration_weight=0.1))
    assert result.applied is True
    assert result.reason_code == REASON_UCB
    order = [c.model.canonical_id for c in result.ranking.candidates]
    assert order == ["test/b", "test/a"]  # b wins on mean reward + UCB bonus
    assert all(REASON_UCB in c.reason_codes for c in result.ranking.candidates)
    assert result.ranking.policy_version == "pol-test+bandit"


def test_unobserved_arm_goes_first_for_deterministic_exploration() -> None:
    ranking = _ranking("test/a", "test/b", "test/c")
    aggregates = [
        _aggregate("test/a", executions=10, successes=10),
        _aggregate("test/b", executions=10, successes=10),
    ]
    result = apply_bandit(ranking, aggregates, BanditConfig())
    order = [c.model.canonical_id for c in result.ranking.candidates]
    assert order[0] == "test/c"  # classic UCB: unobserved arms first
    assert set(order) == {"test/a", "test/b", "test/c"}


def test_never_selects_arm_outside_input_ranking() -> None:
    """Constraint satisfaction: output is a pure reorder of the input set."""
    ranking = _ranking("test/a", "test/b")
    aggregates = [
        _aggregate("test/rogue", executions=10, successes=10),  # not in ranking
        _aggregate("test/a", executions=10, successes=5),
    ]
    result = apply_bandit(ranking, aggregates, BanditConfig())
    arms_in = {(c.model.canonical_id, c.variant.effort) for c in ranking.candidates}
    arms_out = {(c.model.canonical_id, c.variant.effort) for c in result.ranking.candidates}
    assert arms_out == arms_in
    assert len(result.ranking.candidates) == len(ranking.candidates)


def test_quality_floor_demotes_below_floor_arms_to_tail() -> None:
    ranking = _ranking("test/a", "test/b", "test/c")
    aggregates = [
        _aggregate("test/a", executions=10, successes=2),  # success 0.2 < floor
        _aggregate("test/b", executions=10, successes=10),
        # test/c unobserved
    ]
    config = BanditConfig(quality_floor=0.5, exploration_weight=0.1)
    result = apply_bandit(ranking, aggregates, config)
    order = [c.model.canonical_id for c in result.ranking.candidates]
    assert order == ["test/c", "test/b", "test/a"]


def test_bandit_version_is_deterministic_and_config_sensitive() -> None:
    v1 = bandit_version(BanditConfig())
    v2 = bandit_version(BanditConfig())
    v3 = bandit_version(BanditConfig(exploration_weight=2.0))
    assert v1 == v2
    assert v1 != v3
    assert v1.startswith("ban-")


# --------------------------------------------------------------------------- #
# Server integration: bandit consult + telemetry-failure fallback
# --------------------------------------------------------------------------- #


def _seed(session: Session) -> None:
    rows = [
        ("test/cheap-1", 30.0, (1.0, 2.0)),
        ("test/strong", 90.0, (8.0, 30.0)),
        ("test/no-tools", 50.0, (2.0, 6.0)),
    ]
    for canonical, aa, (in_p, out_p) in rows:
        provider = registry_db.get_or_create_provider(session, "test")
        model = registry_db.upsert_model(session, canonical_id=canonical, tool_calling=True)
        deployment = registry_db.get_or_create_deployment(session, model, provider, "default")
        for effort in ("low", "medium", "high"):
            registry_db.upsert_variant(session, deployment, effort, effort)
        registry_db.upsert_benchmark(session, model, AA, aa, None, "snap-test")
        registry_db.upsert_price(session, deployment, "snap-test", in_p, out_p, None)


@pytest.fixture
def config(tmp_path: Path) -> RouterConfig:
    return load_config(config_path="/nonexistent/router.yaml", data_dir=tmp_path / "data")


@pytest.fixture
def engine(tmp_path: Path):
    eng = registry_db.get_engine(f"sqlite:///{tmp_path / 'router.db'}")
    registry_db.init_schema(eng)
    with registry_db.Session(eng) as session:
        _seed(session)
        session.commit()
    return eng


def _shim(tmp_path: Path) -> telemetry_shim.ShimStore:
    shim = telemetry_shim.ShimStore(f"sqlite:///{tmp_path / 'telemetry.sqlite'}")
    shim.init_schema()
    return shim


def test_route_empty_shim_is_byte_identical_to_no_shim(
    config: RouterConfig, engine, tmp_path: Path
) -> None:
    """Cold start at the API level: bandit leaves decisions untouched."""
    client_shim = TestClient(create_app(config, engine, _shim(tmp_path)))
    client_plain = TestClient(create_app(config, engine))
    body = {"task": "t", "phase": "explore"}
    with_shim = client_shim.post("/route", json=body)
    without_shim = client_plain.post("/route", json=body)
    assert with_shim.status_code == 200
    assert with_shim.content == without_shim.content
    assert "bandit" not in str(with_shim.json()["reason_codes"])


def test_route_falls_back_when_telemetry_fails(
    config: RouterConfig, engine, tmp_path: Path, monkeypatch
) -> None:
    from gentle_ai_model_router.api import server as server_mod

    def _boom(session):
        raise RuntimeError("telemetry exploded")

    monkeypatch.setattr(server_mod, "compute_rewards", _boom)
    client = TestClient(create_app(config, engine, _shim(tmp_path)))
    response = client.post("/route", json={"task": "t", "phase": "explore"})
    assert response.status_code == 200, response.text
    assert response.json()["model"] == "test/strong"  # pre-bandit deterministic pick
    assert "bandit" not in str(response.json()["reason_codes"])


def test_route_applies_bandit_with_reason_code(
    config: RouterConfig, engine, tmp_path: Path
) -> None:
    shim = _shim(tmp_path)
    # One observed execution per explore-phase arm; only test/strong succeeds.
    # Efforts mirror the policy's minimum-sufficient-effort pick per model
    # (strong -> low, cheap-1 -> high, no-tools -> medium at the 0.6 floor).
    with shim.session() as session:
        for idx, (model, effort, success) in enumerate(
            [("test/strong", "low", 1), ("test/cheap-1", "high", 0), ("test/no-tools", "medium", 0)]
        ):
            decision = shim.record_decision(
                session,
                phase="explore",
                selected={"model": model, "effort": effort},
                alternatives=[],
                reason_codes=["cheapest_of_meeting"],
                estimated_tokens=14_000.0,
                estimated_cost=0.05,
                policy_version="pol-test",
            )
            shim.record_execution(
                session,
                {
                    "execution_id": f"e{idx}",
                    "phase": "explore",
                    "model": model,
                    "effort": effort,
                    "total_tokens": 100,
                    "task_success": success,
                    "decision_id": decision.decision_id,
                },
            )
    client = TestClient(create_app(config, engine, shim))
    response = client.post("/route", json={"task": "t", "phase": "explore"})
    assert response.status_code == 200, response.text
    data = response.json()
    assert "bandit:ucb" in data["reason_codes"]
    assert data["model"] == "test/strong"  # the only above-floor, observed arm
    assert data["policy_version"].endswith("+bandit")
