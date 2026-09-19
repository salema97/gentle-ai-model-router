"""R4 tests: threshold tuner frontier math, router.yaml applier, bandit config wiring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError
from typer.testing import CliRunner

from gentle_ai_model_router.cli.main import app
from gentle_ai_model_router.integration import telemetry_shim
from gentle_ai_model_router.integration.router_yaml_adapter import (
    RouterYamlError,
    apply_threshold_proposals,
)
from gentle_ai_model_router.registry import db as registry_db
from gentle_ai_model_router.router.bandit import BanditConfig
from gentle_ai_model_router.router.config import (
    DEFAULT_PHASE_THRESHOLDS,
    PhaseConfig,
    RouterConfig,
    load_config,
)
from gentle_ai_model_router.router.reward import RewardAggregate
from gentle_ai_model_router.router.threshold_tune import (
    ThresholdTuneError,
    propose_thresholds,
)

runner = CliRunner()


def _agg(
    phase: str,
    model: str,
    effort: str,
    executions: int,
    successes: int,
    total_tokens: float,
) -> RewardAggregate:
    """Build the RewardAggregate shape aggregate_rewards emits."""
    return RewardAggregate(
        phase=phase,
        model=model,
        effort=effort,
        executions=executions,
        success_rate=successes / executions,
        mean_reward=0.0,
        mean_total_tokens=total_tokens / executions,
        tokens_per_success=total_tokens / successes if successes > 0 else None,
        win_rate=None,
    )


@pytest.fixture
def config(tmp_path: Path) -> RouterConfig:
    return load_config(config_path="/nonexistent/router.yaml", data_dir=tmp_path / "data")


# --------------------------------------------------------------------------- #
# Frontier math
# --------------------------------------------------------------------------- #


def test_cheapest_effort_meeting_floor_wins(config: RouterConfig) -> None:
    """'low' meets the 0.8 floor with 10 executions; 'medium' must not be chosen."""
    arms = [
        _agg("apply", "test/model-a", "low", 10, 9, 9_000),
        _agg("apply", "test/model-a", "medium", 10, 10, 30_000),
    ]
    (proposal,) = propose_thresholds({"apply": arms}, config=config)
    assert proposal.kind == "upgrade"
    assert proposal.selected_effort == "low"
    assert proposal.executions == 10
    assert proposal.success_rate == pytest.approx(0.9)
    assert proposal.tokens_per_success == pytest.approx(9_000 / 9)
    # proposed = effort_quality(0.9, "low", policy) = 0.9 + (0.98-0.9)*0.4
    expected = 0.9 + (config.policy.effort_ceiling - 0.9) * 0.4
    assert proposal.proposed_threshold == pytest.approx(expected)
    assert proposal.current_threshold == DEFAULT_PHASE_THRESHOLDS["apply"]
    assert proposal.proposed_threshold > proposal.current_threshold


def test_lower_effort_meeting_floor_is_a_downgrade(config: RouterConfig) -> None:
    """Current floor above what the cheapest sufficient effort delivers -> downgrade."""
    arms = [_agg("apply", "test/model-a", "low", 10, 9, 9_000)]
    # model_copy(update=...) does not re-validate nested models: build the
    # override with a real PhaseConfig instance.
    high_config = config.model_copy(
        update={"phases": {"apply": PhaseConfig(threshold_quality=0.97)}}
    )
    (proposal,) = propose_thresholds({"apply": arms}, config=high_config)
    assert proposal.kind == "downgrade"
    assert proposal.selected_effort == "low"
    assert proposal.proposed_threshold < proposal.current_threshold


def test_uphold_when_frontier_matches_current(config: RouterConfig) -> None:
    arms = [_agg("apply", "test/model-a", "low", 10, 9, 9_000)]
    proposed = 0.9 + (config.policy.effort_ceiling - 0.9) * 0.4
    exact_config = config.model_copy(
        update={"phases": {"apply": PhaseConfig(threshold_quality=proposed)}}
    )
    (proposal,) = propose_thresholds({"apply": arms}, config=exact_config)
    assert proposal.kind == "uphold"
    assert proposal.proposed_threshold == proposal.current_threshold


def test_insufficient_evidence_never_changes_phase(config: RouterConfig) -> None:
    arms = [_agg("apply", "test/model-a", "low", 2, 2, 2_000)]  # < min_executions (3)
    (proposal,) = propose_thresholds({"apply": arms}, config=config)
    assert proposal.kind == "insufficient_evidence"
    assert proposal.proposed_threshold == proposal.current_threshold
    assert proposal.selected_effort is None
    assert proposal.executions == 0


def test_evidence_but_no_effort_meets_floor_is_uphold(config: RouterConfig) -> None:
    arms = [_agg("apply", "test/model-a", "low", 10, 4, 4_000)]  # 0.4 < 0.8 floor
    (proposal,) = propose_thresholds({"apply": arms}, config=config)
    assert proposal.kind == "uphold"
    assert proposal.proposed_threshold == proposal.current_threshold
    assert "no admissible effort meets" in proposal.note


def test_evidence_bar_comes_from_bandit_config(config: RouterConfig) -> None:
    """bandit.min_executions_before_exploit is the shared evidence standard."""
    arms = [_agg("apply", "test/model-a", "low", 2, 2, 2_000)]
    low_bar = config.model_copy(
        update={"bandit": BanditConfig(min_executions_before_exploit=1)}
    )
    (proposal,) = propose_thresholds({"apply": arms}, config=low_bar)
    assert proposal.kind != "insufficient_evidence"
    assert proposal.selected_effort == "low"


def test_ties_are_deterministic(config: RouterConfig) -> None:
    arms = [
        _agg("spec", "test/model-b", "low", 5, 5, 5_000),
        _agg("apply", "test/model-a", "high", 6, 5, 12_000),
        _agg("apply", "test/model-a", "low", 10, 9, 9_000),
    ]
    grouped = {"sdd-apply": arms[:2], "spec": arms[2:], "apply": []}
    first = propose_thresholds(grouped, config=config)
    second = propose_thresholds(grouped, config=config)
    assert first == second
    # Sorted by normalized phase name; 'sdd-' prefix normalized away.
    assert [p.phase for p in first] == ["apply", "apply", "spec"]


def test_success_floor_picks_higher_effort_when_low_falls_short(
    config: RouterConfig,
) -> None:
    arms = [
        _agg("apply", "test/model-a", "low", 10, 7, 7_000),  # 0.7 < 0.8
        _agg("apply", "test/model-a", "medium", 10, 8, 16_000),  # 0.8 meets
    ]
    (proposal,) = propose_thresholds({"apply": arms}, config=config)
    assert proposal.selected_effort == "medium"
    assert proposal.kind == "upgrade"


def test_pools_arms_across_models_at_same_effort(config: RouterConfig) -> None:
    arms = [
        _agg("apply", "test/model-a", "low", 10, 9, 9_000),
        _agg("apply", "test/model-b", "low", 10, 7, 8_000),  # pooled: 16/20 = 0.8
    ]
    (proposal,) = propose_thresholds({"apply": arms}, config=config)
    assert proposal.selected_effort == "low"
    assert proposal.success_rate == pytest.approx(0.8)
    assert proposal.executions == 20


def test_invalid_success_floor_fails_closed(config: RouterConfig) -> None:
    with pytest.raises(ThresholdTuneError):
        propose_thresholds({"apply": []}, success_floor=0.0, config=config)
    with pytest.raises(ThresholdTuneError):
        propose_thresholds({"apply": []}, success_floor=1.5, config=config)


def test_unknown_effort_vocabulary_fails_closed(config: RouterConfig) -> None:
    arms = [_agg("apply", "test/model-a", "turbo", 10, 9, 9_000)]
    with pytest.raises(ThresholdTuneError, match="turbo"):
        propose_thresholds({"apply": arms}, config=config)


def test_empty_input_returns_no_proposals(config: RouterConfig) -> None:
    assert propose_thresholds({}, config=config) == []


# --------------------------------------------------------------------------- #
# router.yaml applier
# --------------------------------------------------------------------------- #


def _write_router_yaml(tmp_path: Path) -> Path:
    path = tmp_path / "router.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "data_dir": "data",
                "phases": {
                    "apply": {"threshold_quality": 0.75},
                    "spec": {"threshold_quality": 0.8, "weights": {"bench": 1.0}},
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _proposal(phase: str, kind: str, current: float, proposed: float):
    from gentle_ai_model_router.router.threshold_tune import ThresholdProposal

    return ThresholdProposal(
        phase=phase,
        kind=kind,
        current_threshold=current,
        proposed_threshold=proposed,
        selected_effort="low" if kind in ("upgrade", "downgrade") else None,
        executions=10 if kind in ("upgrade", "downgrade") else 0,
        success_rate=0.9 if kind in ("upgrade", "downgrade") else None,
        tokens_per_success=1_000.0 if kind in ("upgrade", "downgrade") else None,
        success_floor=0.8,
        min_executions=3,
        note="test",
    )


def test_apply_writes_backup_first_atomically(tmp_path: Path) -> None:
    path = _write_router_yaml(tmp_path)
    result = apply_threshold_proposals(
        path, [_proposal("apply", "upgrade", 0.75, 0.932)]
    )
    assert result.wrote
    assert result.applied == {"apply": (0.75, 0.932)}
    assert result.backup_path is not None and result.backup_path.is_file()
    assert result.backup_path.name.startswith("router.yaml.router-backup-")
    # Backup holds the pre-write bytes; the live file holds the new value.
    backed_up = yaml.safe_load(result.backup_path.read_text(encoding="utf-8"))
    assert backed_up["phases"]["apply"]["threshold_quality"] == 0.75
    updated = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert updated["phases"]["apply"]["threshold_quality"] == 0.932
    # Untouched sections survive verbatim.
    assert updated["phases"]["spec"] == {"threshold_quality": 0.8, "weights": {"bench": 1.0}}
    assert updated["data_dir"] == "data"
    assert "threshold_quality: 0.75" in result.diff
    assert "threshold_quality: 0.932" in result.diff
    # No tmp files left behind (atomic replace completed).
    assert list(tmp_path.glob(".router.yaml.router-tmp-*")) == []


def test_apply_dry_run_writes_nothing(tmp_path: Path) -> None:
    path = _write_router_yaml(tmp_path)
    before = path.read_text(encoding="utf-8")
    result = apply_threshold_proposals(
        path, [_proposal("apply", "upgrade", 0.75, 0.932)], dry_run=True
    )
    assert not result.wrote
    assert result.applied == {"apply": (0.75, 0.932)}  # would-be changes reported
    assert result.backup_path is None
    assert path.read_text(encoding="utf-8") == before
    assert list(tmp_path.glob("router.yaml.router-backup-*")) == []


def test_apply_skips_non_applyable_kinds(tmp_path: Path) -> None:
    path = _write_router_yaml(tmp_path)
    proposals = [
        _proposal("apply", "insufficient_evidence", 0.75, 0.75),
        _proposal("spec", "uphold", 0.8, 0.8),
    ]
    result = apply_threshold_proposals(path, proposals)
    assert not result.wrote
    assert result.applied == {}
    assert set(result.skipped) == {"apply", "spec"}
    updated = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert updated["phases"]["apply"]["threshold_quality"] == 0.75
    assert updated["phases"]["spec"]["threshold_quality"] == 0.8


def test_apply_refuses_phase_not_in_router_yaml_without_create(tmp_path: Path) -> None:
    path = _write_router_yaml(tmp_path)
    result = apply_threshold_proposals(path, [_proposal("design", "upgrade", 0.85, 0.9)])
    assert not result.wrote
    assert result.applied == {}
    assert "not present" in result.skipped["design"]
    updated = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert "design" not in updated["phases"]


def test_apply_create_adds_missing_phase(tmp_path: Path) -> None:
    path = _write_router_yaml(tmp_path)
    result = apply_threshold_proposals(
        path, [_proposal("design", "upgrade", 0.85, 0.9)], create=True
    )
    assert result.wrote
    assert result.applied == {"design": (0.85, 0.9)}
    updated = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert updated["phases"]["design"] == {"threshold_quality": 0.9}


def test_apply_missing_file_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(RouterYamlError):
        apply_threshold_proposals(
            tmp_path / "router.yaml", [_proposal("apply", "upgrade", 0.75, 0.9)]
        )


# --------------------------------------------------------------------------- #
# BanditConfig wiring (RouterConfig.bandit from router.yaml)
# --------------------------------------------------------------------------- #


def test_router_config_bandit_defaults_match_today() -> None:
    config = RouterConfig()
    assert config.bandit == BanditConfig()
    assert config.bandit.exploration_weight == 1.0
    assert config.bandit.min_executions_before_exploit == 3
    assert config.bandit.quality_floor == 0.5
    assert config.bandit.phases == {}


def test_router_yaml_bandit_section_overrides(tmp_path: Path) -> None:
    path = tmp_path / "router.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "bandit": {
                    "exploration_weight": 2.0,
                    "min_executions_before_exploit": 5,
                    "quality_floor": 0.7,
                    "phases": {"apply": 0.9},
                }
            }
        ),
        encoding="utf-8",
    )
    config = load_config(config_path=path, data_dir=tmp_path / "data")
    assert config.bandit.exploration_weight == 2.0
    assert config.bandit.min_executions_before_exploit == 5
    assert config.bandit.quality_floor == 0.7
    assert config.bandit.phases == {"apply": 0.9}


@pytest.mark.parametrize(
    "bandit_section",
    [
        {"exploration_weight": -1.0},
        {"min_executions_before_exploit": 0},
        {"quality_floor": 0.0},
        {"quality_floor": 1.5},
        {"phases": {"apply": 1.2}},
    ],
)
def test_invalid_bandit_config_fails_closed(tmp_path: Path, bandit_section: dict) -> None:
    path = tmp_path / "router.yaml"
    path.write_text(yaml.safe_dump({"bandit": bandit_section}), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(config_path=path, data_dir=tmp_path / "data")


# --------------------------------------------------------------------------- #
# Server + CLI integration
# --------------------------------------------------------------------------- #


def _seed_registry(engine) -> None:
    from sqlalchemy.orm import Session as OrmSession

    with OrmSession(engine) as session:
        provider = registry_db.get_or_create_provider(session, "test")
        model = registry_db.upsert_model(session, canonical_id="test/strong", tool_calling=True)
        deployment = registry_db.get_or_create_deployment(session, model, provider, "default")
        for effort in ("low", "medium"):
            registry_db.upsert_variant(session, deployment, effort, effort)
        registry_db.upsert_benchmark(
            session, model, "artificial_analysis_intelligence_index", 90.0, None, "snap"
        )
        registry_db.upsert_price(session, deployment, "snap", 1.0, 2.0, None)
        session.commit()


def _seed_shim_with_execution(tmp_path: Path) -> telemetry_shim.ShimStore:
    """One explore decision + one successful execution (total = 1)."""
    store = telemetry_shim.ShimStore(f"sqlite:///{tmp_path / 'telemetry.sqlite'}")
    store.init_schema()
    with store.session() as session:
        decision = store.record_decision(
            session,
            phase="explore",
            selected={"model": "test/strong", "effort": "low"},
            alternatives=[],
            reason_codes=["cheapest_of_meeting"],
            estimated_tokens=1_000.0,
            estimated_cost=0.01,
            policy_version="pol-test",
        )
        store.record_execution(
            session,
            {
                "execution_id": "e1",
                "phase": "explore",
                "model": "test/strong",
                "effort": "low",
                "total_tokens": 1_000,
                "task_success": 1,
                "decision_id": decision.decision_id,
            },
        )
    return store


def test_server_uses_config_derived_bandit_config(tmp_path: Path) -> None:
    """min_executions_before_exploit=1 from router.yaml turns cold start into UCB."""
    from fastapi.testclient import TestClient

    from gentle_ai_model_router.api.server import create_app

    engine = registry_db.get_engine(f"sqlite:///{tmp_path / 'router.db'}")
    registry_db.init_schema(engine)
    _seed_registry(engine)

    def _client(min_executions: int) -> TestClient:
        router_yaml = tmp_path / f"router-{min_executions}.yaml"
        router_yaml.write_text(
            yaml.safe_dump({"bandit": {"min_executions_before_exploit": min_executions}}),
            encoding="utf-8",
        )
        config = load_config(config_path=router_yaml, data_dir=tmp_path / f"data-{min_executions}")
        shim = _seed_shim_with_execution(tmp_path / f"shim-{min_executions}")
        return TestClient(create_app(config, engine, shim))

    # Default bar (3) with a single execution: cold start, no bandit code.
    cold = _client(3).post("/route", json={"task": "t", "phase": "explore"})
    assert cold.status_code == 200, cold.text
    assert "bandit:ucb" not in cold.json()["reason_codes"]
    # Bar lowered to 1 via router.yaml: the same evidence triggers UCB.
    warm = _client(1).post("/route", json={"task": "t", "phase": "explore"})
    assert warm.status_code == 200, warm.text
    assert "bandit:ucb" in warm.json()["reason_codes"]


def _seed_shim_db(tmp_path: Path) -> str:
    store = telemetry_shim.ShimStore(f"sqlite:///{tmp_path / 'telemetry.sqlite'}")
    store.init_schema()
    with store.session() as session:
        decision = store.record_decision(
            session,
            phase="apply",
            selected={"model": "test/model-a", "effort": "low"},
            alternatives=[],
            reason_codes=["cheapest_of_meeting"],
            estimated_tokens=1_000.0,
            estimated_cost=0.01,
            policy_version="pol-test",
        )
        for idx in range(5):
            store.record_execution(
                session,
                {
                    "execution_id": f"e{idx}",
                    "phase": "apply",
                    "model": "test/model-a",
                    "effort": "low",
                    "total_tokens": 1_000,
                    "task_success": 1 if idx < 4 else 0,  # 0.8 pooled success
                    "decision_id": decision.decision_id,
                },
            )
    return f"sqlite:///{tmp_path / 'telemetry.sqlite'}"


def test_thresholds_propose_cli_prints_json(tmp_path: Path) -> None:
    db = _seed_shim_db(tmp_path)
    result = runner.invoke(
        app,
        ["thresholds", "propose", "--db", db, "--data-dir", str(tmp_path / "data")],
    )
    assert result.exit_code == 0, result.output
    proposals = json.loads(result.output)
    (proposal,) = proposals
    assert proposal["phase"] == "apply"
    assert proposal["kind"] == "upgrade"
    assert proposal["selected_effort"] == "low"
    assert proposal["success_rate"] == pytest.approx(0.8)
    assert proposal["current_threshold"] == DEFAULT_PHASE_THRESHOLDS["apply"]


def test_thresholds_apply_cli_dry_run_then_write(tmp_path: Path) -> None:
    db = _seed_shim_db(tmp_path)
    router_yaml = tmp_path / "router.yaml"
    router_yaml.write_text(
        yaml.safe_dump({"phases": {"apply": {"threshold_quality": 0.75}}}), encoding="utf-8"
    )
    base = ["thresholds", "apply", "--db", db, "--config", str(router_yaml),
            "--data-dir", str(tmp_path / "data")]

    dry = runner.invoke(app, [*base, "--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert "dry-run: nothing written" in dry.output
    assert yaml.safe_load(router_yaml.read_text(encoding="utf-8"))["phases"]["apply"][
        "threshold_quality"
    ] == 0.75

    real = runner.invoke(app, base)
    assert real.exit_code == 0, real.output
    assert "applied apply: 0.750 ->" in real.output
    assert "backup:" in real.output
    assert list(tmp_path.glob("router.yaml.router-backup-*"))
    updated = yaml.safe_load(router_yaml.read_text(encoding="utf-8"))
    assert updated["phases"]["apply"]["threshold_quality"] > 0.75


def test_thresholds_apply_cli_refuses_bundled_example(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No router.yaml around -> refuse instead of editing router.yaml.example."""
    db = _seed_shim_db(tmp_path)
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app, ["thresholds", "apply", "--db", db, "--data-dir", str(tmp_path / "data")]
    )
    assert result.exit_code == 2
    # Rich may wrap the line at the console width; assert on the stable prefix.
    assert "refusing to edit the bundled" in result.output
