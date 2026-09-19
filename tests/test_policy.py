"""Baseline policy tests: synthetic registry fixtures, no network."""

from __future__ import annotations

import pytest
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from gentle_ai_model_router.cli.main import app
from gentle_ai_model_router.registry import db as registry_db
from gentle_ai_model_router.registry.models import ModelBenchmark
from gentle_ai_model_router.router.config import (
    PhaseConfig,
    RouterConfig,
    load_config,
)
from gentle_ai_model_router.router.decision import CANONICAL_PHASES, TaskContext
from gentle_ai_model_router.router.policy import PolicyError, select_candidate

runner = CliRunner()

# Benchmark keys understood by the fixture helper ("name" or "name:category").
AA = "artificial_analysis_intelligence_index"
ARENA_TEXT = "lmarena_elo:text"


def _engine(tmp_path):
    engine = registry_db.get_engine(f"sqlite:///{tmp_path / 'router.db'}")
    registry_db.init_schema(engine)
    return engine


def _add_candidate(
    session: Session,
    canonical: str,
    benchmarks: dict[str, float],
    price: tuple[float, float] = (2.0, 8.0),
    variants: tuple[str, ...] = ("low", "medium", "high"),
    context_window: int = 128_000,
    tool_calling: bool = True,
) -> None:
    provider = registry_db.get_or_create_provider(session, canonical.split("/")[0])
    model = registry_db.upsert_model(
        session,
        canonical_id=canonical,
        context_window=context_window,
        tool_calling=tool_calling,
    )
    deployment = registry_db.get_or_create_deployment(session, model, provider, "default")
    for effort in variants:
        registry_db.upsert_variant(session, deployment, effort, effort)
    for key, score in benchmarks.items():
        name, _, category = key.partition(":")
        registry_db.upsert_benchmark(
            session, model, name, score, category or None, "snap-test"
        )
    registry_db.upsert_price(session, deployment, "snap-test", price[0], price[1], None)


def _design_config(tmp_path) -> RouterConfig:
    """Config with explicit two-benchmark weights for the design phase."""
    return load_config(
        config_path="/nonexistent/router.yaml",
        data_dir=tmp_path / "data",
    ).model_copy(
        update={
            "phases": {
                "design": PhaseConfig(
                    threshold_quality=0.85,
                    weights={AA: 1.0, ARENA_TEXT: 1.0},
                )
            }
        }
    )


@pytest.fixture
def design_session(tmp_path, design_config):
    engine = _engine(tmp_path)
    with registry_db.Session(engine) as session:
        # Prior normalization: AA norms A=1.0, B=0.667, C=0.0; arena:text
        # norms A=0.75, B=1.0, C=0.0. Priors: A=0.875, B=0.833, C=0.0.
        # B offers no "low" variant, so its minimum sufficient effort always
        # costs more tokens than A/low — A wins the token minimization.
        _add_candidate(session, "test/model-a", {AA: 50.0, ARENA_TEXT: 90.0})
        _add_candidate(
            session, "test/model-b", {AA: 40.0, ARENA_TEXT: 95.0}, variants=("medium", "high")
        )
        _add_candidate(session, "test/model-c", {AA: 20.0, ARENA_TEXT: 75.0})
        session.commit()
        yield session


@pytest.fixture
def design_config(tmp_path) -> RouterConfig:
    return _design_config(tmp_path)


def _quality(prior: float, gain: float) -> float:
    return prior + (0.98 - prior) * gain


def test_lowest_sufficient_effort_beats_higher_effort(design_session, design_config) -> None:
    """A/low (0.917) wins over A/high (0.954) when the 0.85 threshold is met at low."""
    decision = select_candidate(design_session, "design", design_config)
    assert decision.model == "test/model-a"
    assert decision.effort == "low"
    assert decision.quality == pytest.approx(_quality(0.875, 0.4), abs=1e-3)
    assert "meets_threshold:low" in decision.reason_codes
    assert "cheapest_of_meeting" in decision.reason_codes
    # The runner-up (B at its minimum sufficient effort) is offered as an
    # alternative; higher efforts of the winning model are never generated.
    alt_keys = {(a.model, a.effort) for a in decision.alternatives}
    assert ("test/model-b", "medium") in alt_keys


def test_cheaper_model_medium_beats_escalated_high(tmp_path, design_config) -> None:
    """B/medium wins when A cannot meet the threshold at low and only offers high.

    With threshold 0.92: A/low (0.917) fails, A jumps to high (0.954, 26k
    tokens); B meets at medium (0.921, 19k tokens) — fewer tokens win.
    """
    engine = _engine(tmp_path)
    with registry_db.Session(engine) as session:
        _add_candidate(
            session, "test/model-a", {AA: 50.0, ARENA_TEXT: 90.0}, variants=("low", "high")
        )
        _add_candidate(
            session, "test/model-b", {AA: 40.0, ARENA_TEXT: 95.0}, variants=("medium", "high")
        )
        _add_candidate(session, "test/model-c", {AA: 20.0, ARENA_TEXT: 75.0})
        session.commit()
        config = design_config.model_copy(
            update={
                "phases": {
                    "design": PhaseConfig(
                        threshold_quality=0.92,
                        weights={AA: 1.0, ARENA_TEXT: 1.0},
                    )
                }
            }
        )
        decision = select_candidate(session, "design", config)
        assert decision.model == "test/model-b"
        assert decision.effort == "medium"


def test_threshold_unmet_fails_closed(design_session, design_config) -> None:
    config = design_config.model_copy(
        update={
            "phases": {
                "design": PhaseConfig(
                    threshold_quality=0.99, weights={AA: 1.0, ARENA_TEXT: 1.0}
                )
            }
        }
    )
    with pytest.raises(PolicyError, match="no candidate meets threshold"):
        select_candidate(design_session, "design", config)


def test_empty_registry_fails_closed(tmp_path, design_config):
    engine = _engine(tmp_path)
    with registry_db.Session(engine) as session:
        with pytest.raises(PolicyError, match="registry is empty"):
            select_candidate(session, "explore", design_config)


def test_route_cli_empty_registry_exit_2(tmp_path) -> None:
    result = runner.invoke(
        app, ["route", "--phase", "explore", "--data-dir", str(tmp_path / "data")]
    )
    assert result.exit_code == 2
    assert "registry is empty" in result.output


def test_all_11_phases_accepted(tmp_path, design_config):
    engine = _engine(tmp_path)
    with registry_db.Session(engine) as session:
        # High prior so every default threshold is met at low effort.
        _add_candidate(session, "test/model-a", {AA: 50.0})
        session.commit()
        for phase in CANONICAL_PHASES:
            decision = select_candidate(session, phase, design_config)
            assert decision.phase == phase
            assert decision.model == "test/model-a"


def test_unknown_phase_rejected(tmp_path, design_config):
    engine = _engine(tmp_path)
    with registry_db.Session(engine) as session:
        _add_candidate(session, "test/model-a", {AA: 50.0})
        session.commit()
        with pytest.raises(PolicyError, match="unknown phase"):
            select_candidate(session, "bogus", design_config)
        # The sdd- prefix spelling is accepted.
        decision = select_candidate(session, "sdd-explore", design_config)
        assert decision.phase == "explore"


def test_determinism_two_runs_identical(design_session, design_config) -> None:
    first = select_candidate(design_session, "design", design_config)
    second = select_candidate(design_session, "design", design_config)
    assert first == second
    assert first.policy_version == second.policy_version


def test_tool_calling_hard_filter_for_apply_and_verify(tmp_path, design_config):
    engine = _engine(tmp_path)
    with registry_db.Session(engine) as session:
        # Equal benchmark scores -> both priors 1.0 (span 0 normalizes to 0.5
        # per model only when it is the sole contributor; here each has data).
        _add_candidate(session, "test/no-tools", {AA: 50.0}, tool_calling=False)
        _add_candidate(session, "test/with-tools", {AA: 50.0}, tool_calling=True)
        session.commit()
        decision = select_candidate(session, "apply", design_config)
        assert decision.model == "test/with-tools"
        decision = select_candidate(session, "verify", design_config)
        assert decision.model == "test/with-tools"
        # Outside apply/verify the stronger model is allowed.
        decision = select_candidate(session, "explore", design_config)
        assert decision.model == "test/no-tools"


def test_context_window_hard_filter(tmp_path, design_config):
    engine = _engine(tmp_path)
    with registry_db.Session(engine) as session:
        _add_candidate(session, "test/small-ctx", {AA: 50.0}, context_window=8_000)
        _add_candidate(session, "test/big-ctx", {AA: 20.0}, context_window=200_000)
        session.commit()
        context = TaskContext(context_tokens=32_000)
        decision = select_candidate(session, "explore", design_config, context)
        assert decision.model == "test/big-ctx"


def test_missing_benchmark_uses_flat_prior_with_reason(tmp_path, design_config):
    engine = _engine(tmp_path)
    with registry_db.Session(engine) as session:
        # Only one model with data -> span 0 -> normalized 0.5 == flat prior,
        # so the unmeasured model ties on quality and wins the canonical-id
        # tie-break, surfacing its reason code.
        _add_candidate(session, "test/zzz-measured", {AA: 50.0})
        _add_candidate(session, "test/aaa-unmeasured", {})
        session.commit()
        rows = session.query(ModelBenchmark).all()
        assert len(rows) == 1  # sanity: unmeasured truly has no benchmark rows
        decision = select_candidate(session, "explore", design_config)
        assert decision.model == "test/aaa-unmeasured"
        assert "missing_benchmark_data:using_prior" in decision.reason_codes
        assert decision.quality == pytest.approx(0.5 + 0.48 * 0.4, abs=1e-3)


def test_policy_version_tracks_config(design_session, design_config) -> None:
    base = select_candidate(design_session, "design", design_config)
    changed = design_config.model_copy(
        update={
            "phases": {
                "design": PhaseConfig(
                    threshold_quality=0.86, weights={AA: 1.0, ARENA_TEXT: 1.0}
                )
            }
        }
    )
    other = select_candidate(design_session, "design", changed)
    assert base.policy_version != other.policy_version
