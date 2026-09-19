"""Promotion workflow tests: candidate -> evaluate -> compare -> promote.

All synthetic: checkpoints are fake dirs with hand-written metrics.json, so
no torch/[train] extra is needed. The only test that touches the real
evaluate flow (``--dataset`` given) is guarded behind the [train] extra and
skips otherwise.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from gentle_ai_model_router.cli.main import app
from gentle_ai_model_router.training import promote as promote_mod

runner = CliRunner()

BASE_PROVENANCE = {
    "dataset": {"name": "router-priors", "version": 1},
    "source_snapshot_ids": ["snap-01"],
    "label_provenance": "bootstrap_prior",
}


def _ranker_metrics(
    tokens_per_success: float,
    success_rate: float,
    mean_quality: float,
    ndcg5: float = 0.8,
    mrr: float = 0.7,
) -> dict:
    return {
        "groups": 10.0,
        "top1_accuracy": 0.6,
        "top3_recall": 0.9,
        "mrr": mrr,
        "ndcg@3": 0.75,
        "ndcg@5": ndcg5,
        "tokens_per_task": 5000.0,
        "tokens_per_success": tokens_per_success,
        "success_rate": success_rate,
        "mean_quality": mean_quality,
        "routing_regret": 0.05,
    }


def _eval_section(ranker: dict) -> dict:
    """Eval doc shaped like training/evaluate.py output, newest split last."""
    test_ranker = dict(ranker, tokens_per_success=ranker["tokens_per_success"] * 1.1)
    return {
        "splits": {
            "test": {"learned_ranker": test_ranker},
            "temporal_test": {"learned_ranker": ranker},
        },
        "chooser_errors": [],
        "label_provenance": "bootstrap_prior",
    }


def make_checkpoint(
    root: Path,
    version: str,
    *,
    tokens_per_success: float = 12_000.0,
    success_rate: float = 0.85,
    mean_quality: float = 0.80,
    with_eval: bool = True,
    provenance: bool = True,
) -> Path:
    """Fake checkpoint dir with hand-written metrics.json (+ config.json)."""
    ckpt = root / "modernbert-router" / version
    ckpt.mkdir(parents=True)
    metrics: dict = {
        "objective": "pairwise",
        "model_name": "tiny-modernbert",
        "train_rows": 42,
        "train_loss": 0.42,
        "git_commit": "abcdef0",
        "created_at": "2026-09-18T00:00:00+00:00",
    }
    if provenance:
        metrics.update(BASE_PROVENANCE)
    if with_eval:
        metrics["eval"] = _eval_section(
            _ranker_metrics(tokens_per_success, success_rate, mean_quality)
        )
    (ckpt / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (ckpt / "config.json").write_text(json.dumps({"objective": "pairwise"}) + "\n")
    return ckpt


# --------------------------------------------------------------------------- #
# Unit: comparison logic
# --------------------------------------------------------------------------- #


def _cand_eval(tps: float, sr: float, mq: float) -> dict:
    return _eval_section(_ranker_metrics(tps, sr, mq))


def test_compare_first_promotion_always_promotes() -> None:
    comparison = promote_mod.compare(_cand_eval(12_000.0, 0.85, 0.80), None)
    assert comparison.decision == "promote"
    assert comparison.promoted_values is None
    assert comparison.split == "temporal_test"  # newest available split


def test_compare_better_primary_guardrails_hold() -> None:
    comparison = promote_mod.compare(
        _cand_eval(11_000.0, 0.85, 0.80),
        _cand_eval(12_000.0, 0.86, 0.81),
    )
    assert comparison.decision == "promote"
    assert all(g.ok for g in comparison.guardrails)


def test_compare_guardrail_regression_beyond_epsilon_keeps() -> None:
    comparison = promote_mod.compare(
        _cand_eval(11_000.0, 0.83, 0.80),  # sr regressed 0.86 -> 0.83 (> eps 0.02)
        _cand_eval(12_000.0, 0.86, 0.81),
        epsilon=0.02,
    )
    assert comparison.decision == "keep"
    assert not all(g.ok for g in comparison.guardrails)


def test_compare_guardrail_regression_within_epsilon_promotes() -> None:
    comparison = promote_mod.compare(
        _cand_eval(11_000.0, 0.845, 0.80),  # sr regressed 0.86 -> 0.845 (within eps)
        _cand_eval(12_000.0, 0.86, 0.81),
        epsilon=0.02,
    )
    assert comparison.decision == "promote"


def test_compare_worse_primary_keeps_even_with_better_guardrails() -> None:
    comparison = promote_mod.compare(
        _cand_eval(13_000.0, 0.95, 0.90),
        _cand_eval(12_000.0, 0.86, 0.81),
    )
    assert comparison.decision == "keep"


def test_compare_determinism_same_inputs_same_decision() -> None:
    cand, prom = _cand_eval(11_000.0, 0.85, 0.80), _cand_eval(12_000.0, 0.86, 0.81)
    first = promote_mod.compare(cand, prom)
    second = promote_mod.compare(json.loads(json.dumps(cand)), json.loads(json.dumps(prom)))
    assert first.decision == second.decision
    assert [g.delta for g in first.guardrails] == [g.delta for g in second.guardrails]
    assert first.reasons == second.reasons


def test_compare_rejects_unknown_metric_and_negative_epsilon() -> None:
    with pytest.raises(promote_mod.PromotionError):
        promote_mod.compare(_cand_eval(1.0, 1.0, 1.0), None, metric="mrr")
    with pytest.raises(promote_mod.PromotionError):
        promote_mod.compare(_cand_eval(1.0, 1.0, 1.0), None, epsilon=-0.1)


def test_missing_provenance_detection() -> None:
    assert promote_mod.missing_provenance({}) == list(promote_mod.PROVENANCE_FIELDS)
    assert promote_mod.missing_provenance(dict(BASE_PROVENANCE)) == []
    assert promote_mod.missing_provenance({"dataset": {"name": "x"}}) == [
        "dataset.version",
        "source_snapshot_ids",
        "label_provenance",
    ]


# --------------------------------------------------------------------------- #
# CLI: end-to-end with synthetic checkpoints
# --------------------------------------------------------------------------- #


def _promote_args(models_dir: Path, candidate: Path, *extra: str) -> list[str]:
    return [
        "promote",
        "--candidate",
        str(candidate),
        "--models-dir",
        str(models_dir),
        *extra,
    ]


def test_cli_first_promotion_writes_record(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    ckpt = make_checkpoint(models_dir, "v1")
    result = runner.invoke(app, _promote_args(models_dir, ckpt))
    assert result.exit_code == 0, result.output
    assert "PROMOTE" in result.output

    record_path = models_dir / "promoted" / "promoted.json"
    assert record_path.is_file()
    record = json.loads(record_path.read_text())
    assert record["promoted_checkpoint"] == str(ckpt)
    assert record["metrics_path"] == str(models_dir / "promoted" / "metrics.json")
    assert record["promoted_by"] == "manual"
    assert record["git_commit"]  # resolved (or "unknown" outside a repo)
    assert record["promoted_at"].endswith("+00:00")
    # metrics copy is self-contained (eval section included)
    metrics_copy = json.loads((models_dir / "promoted" / "metrics.json").read_text())
    assert metrics_copy["eval"]["splits"]["temporal_test"]["learned_ranker"]


def test_cli_better_primary_promotes_and_replaces(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    v1 = make_checkpoint(models_dir, "v1", tokens_per_success=12_000.0)
    assert runner.invoke(app, _promote_args(models_dir, v1)).exit_code == 0
    v2 = make_checkpoint(models_dir, "v2", tokens_per_success=11_000.0)
    result = runner.invoke(app, _promote_args(models_dir, v2))
    assert result.exit_code == 0, result.output
    record = json.loads((models_dir / "promoted" / "promoted.json").read_text())
    assert record["promoted_checkpoint"] == str(v2)


def test_cli_guardrail_regression_keeps_and_keeps_record(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    v1 = make_checkpoint(models_dir, "v1", tokens_per_success=12_000.0, success_rate=0.86)
    assert runner.invoke(app, _promote_args(models_dir, v1)).exit_code == 0
    v2 = make_checkpoint(
        models_dir, "v2", tokens_per_success=11_000.0, success_rate=0.83
    )  # primary better, guardrail beyond default eps 0.02
    result = runner.invoke(app, _promote_args(models_dir, v2))
    assert result.exit_code == 1, result.output
    assert "KEEP" in result.output
    record = json.loads((models_dir / "promoted" / "promoted.json").read_text())
    assert record["promoted_checkpoint"] == str(v1)  # untouched


def test_cli_missing_provenance_refused(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    ckpt = make_checkpoint(models_dir, "v1", provenance=False)
    result = runner.invoke(app, _promote_args(models_dir, ckpt))
    assert result.exit_code == 2, result.output
    assert "provenance" in result.output
    assert not (models_dir / "promoted").exists()


def test_cli_train_only_checkpoint_refused(tmp_path: Path) -> None:
    """A checkpoint with train metrics only must never be promoted."""
    models_dir = tmp_path / "models"
    ckpt = make_checkpoint(models_dir, "v1", with_eval=False)
    result = runner.invoke(app, _promote_args(models_dir, ckpt))
    assert result.exit_code == 2, result.output
    assert "train loss alone" in result.output


def test_cli_dry_run_writes_nothing(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    ckpt = make_checkpoint(models_dir, "v1")
    result = runner.invoke(app, _promote_args(models_dir, ckpt, "--dry-run"))
    assert result.exit_code == 0, result.output
    assert "dry-run" in result.output
    assert not (models_dir / "promoted").exists()


def test_cli_dry_run_keep_also_writes_nothing(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    v1 = make_checkpoint(models_dir, "v1", tokens_per_success=12_000.0, success_rate=0.86)
    assert runner.invoke(app, _promote_args(models_dir, v1)).exit_code == 0
    v2 = make_checkpoint(models_dir, "v2", tokens_per_success=13_000.0)
    result = runner.invoke(app, _promote_args(models_dir, v2, "--dry-run"))
    assert result.exit_code == 1  # keep, even in dry-run
    record = json.loads((models_dir / "promoted" / "promoted.json").read_text())
    assert record["promoted_checkpoint"] == str(v1)


def test_cli_status_none_then_record(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    result = runner.invoke(app, ["promote", "--status", "--models-dir", str(models_dir)])
    assert result.exit_code == 0
    assert "none" in result.output

    ckpt = make_checkpoint(models_dir, "v1")
    assert runner.invoke(app, _promote_args(models_dir, ckpt)).exit_code == 0
    result = runner.invoke(app, ["promote", "--status", "--models-dir", str(models_dir)])
    assert result.exit_code == 0
    assert "promoted router" in result.output  # rich wraps long paths: check the record itself
    assert "manual" in result.output
    current = promote_mod.load_promoted(models_dir)
    assert current is not None
    assert current["record"]["promoted_checkpoint"] == str(ckpt)


def test_cli_usage_errors(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    # --candidate missing
    assert runner.invoke(app, ["promote", "--models-dir", str(models_dir)]).exit_code == 2
    # --status with other actions
    result = runner.invoke(
        app,
        ["promote", "--status", "--candidate", "x", "--models-dir", str(models_dir)],
    )
    assert result.exit_code == 2
    # unknown metric
    ckpt = make_checkpoint(models_dir, "v1")
    result = runner.invoke(
        app, _promote_args(models_dir, ckpt, "--metric", "ndcg@5")
    )
    assert result.exit_code == 2
    assert "unsupported primary metric" in result.output
    # candidate dir missing
    result = runner.invoke(
        app, _promote_args(models_dir, models_dir / "modernbert-router" / "v99")
    )
    assert result.exit_code == 2


def test_cli_promote_with_dataset_requires_train_extra(tmp_path: Path) -> None:
    """The --dataset path loads the checkpoint -> [train] extra; skip without it.

    This is the ONLY test touching the real evaluate flow; without the extra
    it must fail cleanly with exit 2, with it the test is skipped (downloads
    are forbidden in tests).
    """
    import importlib.util

    has_train = (
        importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("transformers") is not None
    )
    if has_train:  # pragma: no cover - depends on local env
        pytest.skip("checkpoint loading would need HF artifacts/downloads")
    models_dir = tmp_path / "models"
    ckpt = make_checkpoint(models_dir, "v1", with_eval=False)
    # Empty-but-valid dataset dir (load_dataset requires examples + pairs files).
    ds_dir = tmp_path / "ds" / "v1"
    ds_dir.mkdir(parents=True)
    (ds_dir / "examples.jsonl").write_text("")
    (ds_dir / "pairs.jsonl").write_text("")
    result = runner.invoke(app, _promote_args(models_dir, ckpt, "--dataset", str(ds_dir)))
    assert result.exit_code == 2, result.output
    assert "[train] extra" in result.output
