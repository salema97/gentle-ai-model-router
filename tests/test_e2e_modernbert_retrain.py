"""End-to-end verification test for ModernBERT retrain on telemetry.

Runs the full Phase 5 loop:
1. Setup synthetic registry + snapshot store with empirical benchmark tasks.
2. Build initial dataset with empirical benchmark observations (PROVENANCE_EMPIRICAL).
3. Seed shim from empirical dataset via seed_empirical_dataset
   (router_version: "empirical-bootstrap").
4. Build dataset with telemetry enabled (telemetry_db), verifying PROVENANCE_TELEMETRY
   rows and mixed label_provenance.
5. Train tiny ModernBERT offline with pointwise objective and telemetry loss weighting.
6. Evaluate checkpoint, verifying mixed provenance and measured token accounting.
7. Promote candidate with --dry-run, verifying decision and output.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from sqlalchemy import select
from typer.testing import CliRunner

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("accelerate")

from tokenizers import Tokenizer  # noqa: E402
from tokenizers.models import WordLevel  # noqa: E402
from tokenizers.pre_tokenizers import Whitespace  # noqa: E402
from transformers import AutoModel, PreTrainedTokenizerFast  # noqa: E402

from gentle_ai_model_router.cli.main import app  # noqa: E402
from gentle_ai_model_router.collector.snapshots import SnapshotStore  # noqa: E402
from gentle_ai_model_router.dataset.builder import (  # noqa: E402
    DatasetBuilderConfig,
    build_examples,
    write_dataset,
)
from gentle_ai_model_router.dataset.schema import (  # noqa: E402
    PROVENANCE_EMPIRICAL,
    PROVENANCE_TELEMETRY,
)
from gentle_ai_model_router.integration import telemetry_shim  # noqa: E402
from gentle_ai_model_router.integration.empirical_seed import (  # noqa: E402
    ROUTER_VERSION_BOOTSTRAP,
    seed_empirical_dataset,
)
from gentle_ai_model_router.registry import db as registry_db  # noqa: E402
from gentle_ai_model_router.router.config import (  # noqa: E402
    PhaseConfig,
    RouterConfig,
    TrainingConfig,
    load_config,
)
from gentle_ai_model_router.training.evaluate import evaluate_dataset  # noqa: E402
from gentle_ai_model_router.training.model import tiny_modernbert_config  # noqa: E402
from gentle_ai_model_router.training.train import train  # noqa: E402

runner = CliRunner()

AA = "artificial_analysis_intelligence_index"
ARENA_TEXT = "lmarena_elo:text"

SNAP_OLD = "snap-2026-01"
SNAP_MID = "snap-2026-02"
SNAP_NEW = "snap-2026-03"
DATES = {
    SNAP_OLD: datetime(2026, 1, 1),
    SNAP_MID: datetime(2026, 2, 1),
    SNAP_NEW: datetime(2026, 3, 1),
}

TRAIN_END = date(2026, 2, 15)
VAL_END = date(2026, 2, 20)


def _setup_registry(tmp_path: Path):
    engine = registry_db.get_engine(f"sqlite:///{tmp_path / 'router.db'}")
    registry_db.init_schema(engine)
    with registry_db.Session(engine) as session:
        for snap in DATES:
            registry_db.record_snapshot(session, "test", snap, DATES[snap], 1)

        def _add_model(canonical: str, bench_snap: str, price_snap: str) -> None:
            provider = registry_db.get_or_create_provider(session, canonical.split("/")[0])
            m = registry_db.upsert_model(session, canonical_id=canonical, tool_calling=True)
            d = registry_db.get_or_create_deployment(session, m, provider, "default")
            for effort in ("low", "high"):
                registry_db.upsert_variant(session, d, effort, effort)
            registry_db.upsert_benchmark(session, m, AA, 50.0, None, bench_snap)
            registry_db.upsert_benchmark(session, m, "lmarena_elo", 90.0, "text", bench_snap)
            registry_db.upsert_price(session, d, price_snap, 2.0, 8.0, None)

        _add_model("test/old-model", SNAP_OLD, SNAP_OLD)
        _add_model("test/mid-model", SNAP_MID, SNAP_OLD)
        session.commit()
    return engine


def _setup_snapshot_store(tmp_path: Path) -> SnapshotStore:
    store = SnapshotStore(tmp_path / "data" / "snapshots")
    store.save(
        "routing-benchmarks",
        data={
            "source": "huggingface",
            "dars": [
                {
                    "query_id": "d1",
                    "prompt": "Optimize raft consensus throughput under packet loss.",
                    "model": "test/old-model",
                    "quality": 0.95,
                    "cost": 0.001,
                    "input_tokens": 1200,
                    "task_name": "raft-opt",
                    "mapped_phase": "design",
                },
                {
                    "query_id": "d2",
                    "prompt": "Evaluate partition recovery in distributed key-value store.",
                    "model": "test/mid-model",
                    "quality": 0.85,
                    "cost": 0.002,
                    "input_tokens": 1200,
                    "task_name": "raft-eval",
                    "mapped_phase": "design",
                },
            ],
        },
        fetched_at=datetime(2026, 1, 15, tzinfo=UTC),
    )
    return store


def _create_offline_base_dir(base_dir: Path) -> Path:
    """Create a minimal offline base model directory for tokenizer and encoder loading."""
    base_dir.mkdir(parents=True, exist_ok=True)
    vocab = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[CLS]": 2,
        "[SEP]": 3,
        "[EOS]": 4,
        "[MASK]": 5,
    }
    raw_tok = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    raw_tok.pre_tokenizer = Whitespace()
    fast_tok = PreTrainedTokenizerFast(
        tokenizer_object=raw_tok,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    fast_tok.save_pretrained(base_dir)

    cfg = tiny_modernbert_config()
    cfg.bos_token_id = None
    model = AutoModel.from_config(cfg)
    model.save_pretrained(base_dir)
    return base_dir


def test_e2e_modernbert_retrain_loop(tmp_path: Path) -> None:
    # ----------------------------------------------------------------------- #
    # 1) Set up synthetic registry with models and benchmark snapshots
    # ----------------------------------------------------------------------- #
    engine = _setup_registry(tmp_path)
    store = _setup_snapshot_store(tmp_path)
    config: RouterConfig = load_config(
        "/nonexistent/router.yaml", data_dir=tmp_path / "data"
    ).model_copy(
        update={
            "phases": {
                "design": PhaseConfig(
                    threshold_quality=0.8, weights={AA: 1.0, ARENA_TEXT: 1.0}
                )
            }
        }
    )
    builder_cfg = DatasetBuilderConfig(
        name="e2e-retrain-dataset",
        phases=["design"],
        task_types=["feature"],
        context_sizes=[10_000],
        train_end=TRAIN_END,
        val_end=VAL_END,
    )

    # ----------------------------------------------------------------------- #
    # 2) Build initial dataset with empirical benchmark observations
    # ----------------------------------------------------------------------- #
    with registry_db.Session(engine) as session:
        ds1 = build_examples(session, config, builder_cfg, store=store)

    emp_examples = [e for e in ds1.examples if e.label_provenance == PROVENANCE_EMPIRICAL]
    assert len(emp_examples) > 0, "Initial dataset must contain empirical benchmark rows"
    ds1_dir = write_dataset(ds1, tmp_path / "data", builder=builder_cfg)
    assert (ds1_dir / "manifest.json").exists()

    # ----------------------------------------------------------------------- #
    # 3) Seed shim from empirical dataset via seed_empirical_dataset
    # ----------------------------------------------------------------------- #
    shim_db_path = tmp_path / "telemetry.sqlite"
    shim_url = f"sqlite:///{shim_db_path}"
    summary = seed_empirical_dataset(ds1_dir, shim_url)
    assert summary["seeded"] > 0
    assert summary["seeded"] == len(emp_examples)

    # Verify seeded telemetry executions carry router_version == "empirical-bootstrap"
    shim_store = telemetry_shim.ShimStore(shim_url)
    with shim_store.session() as s_sess:
        exec_rows = s_sess.execute(select(telemetry_shim.ExecutionRecord)).scalars().all()
        assert len(exec_rows) == summary["seeded"]
        assert all(row.router_version == ROUTER_VERSION_BOOTSTRAP for row in exec_rows)

        # Update one execution row to a post-val date so it lands in the "test" split
        # for evaluation and promotion comparisons.
        exec_rows[0].finished_at = datetime(2026, 3, 1, 12, 0, 0, tzinfo=UTC)
        s_sess.commit()

    # ----------------------------------------------------------------------- #
    # 4) Build dataset with telemetry enabled, verifying PROVENANCE_TELEMETRY
    # ----------------------------------------------------------------------- #
    with registry_db.Session(engine) as session:
        ds2 = build_examples(
            session, config, builder_cfg, store=store, telemetry_db=str(shim_db_path)
        )

    tel_examples = [e for e in ds2.examples if e.label_provenance == PROVENANCE_TELEMETRY]
    assert len(tel_examples) > 0, "Dataset must contain PROVENANCE_TELEMETRY rows"
    # At least one telemetry example in test split (measured tokens during eval)
    assert any(e.split == "test" for e in tel_examples)
    # At least one telemetry example in train split (loss weighting during retrain)
    assert any(e.split == "train" for e in tel_examples)

    ds2_dir = write_dataset(ds2, tmp_path / "data", builder=builder_cfg)
    manifest = json.loads((ds2_dir / "manifest.json").read_text())
    assert "telemetry" in manifest["label_provenance"]
    assert "+" in manifest["label_provenance"]  # mixed provenance string

    # ----------------------------------------------------------------------- #
    # 5) Train tiny ModernBERT offline
    # ----------------------------------------------------------------------- #
    tiny_base_dir = _create_offline_base_dir(tmp_path / "tiny_modernbert_base")
    models_dir = tmp_path / "models"
    training_cfg = TrainingConfig(
        model_name=str(tiny_base_dir),
        output_dir=str(models_dir),
        epochs=1,
        batch_size=2,
        objective="pointwise",
        telemetry_weight=2.0,
        device="cpu",
    )
    checkpoint_dir = train(config, training_cfg, ds2_dir)
    assert checkpoint_dir.is_dir()
    assert (checkpoint_dir / "head.pt").is_file()

    metrics_file = checkpoint_dir / "metrics.json"
    assert metrics_file.is_file()
    metrics = json.loads(metrics_file.read_text())
    assert "telemetry" in metrics["label_provenance"]
    assert "+" in metrics["label_provenance"]
    assert metrics["train_rows"] > 0
    assert metrics["train_loss"] >= 0.0

    # ----------------------------------------------------------------------- #
    # 6) Evaluate checkpoint
    # ----------------------------------------------------------------------- #
    eval_results = evaluate_dataset(ds2_dir, config, checkpoint=checkpoint_dir, device="cpu")
    assert "telemetry" in eval_results["label_provenance"]
    assert "+" in eval_results["label_provenance"]
    assert eval_results["measured_token_rows"] > 0
    assert "test" in eval_results["splits"]
    assert "learned_ranker" in eval_results["splits"]["test"]

    # Embed eval results in checkpoint metrics for the promotion comparison
    metrics["eval"] = eval_results
    metrics_file.write_text(json.dumps(metrics, indent=2) + "\n")

    # ----------------------------------------------------------------------- #
    # 7) Promote with --dry-run
    # ----------------------------------------------------------------------- #
    prom_res = runner.invoke(
        app,
        [
            "promote",
            "--candidate",
            str(checkpoint_dir),
            "--dry-run",
            "--models-dir",
            str(models_dir),
        ],
    )
    assert prom_res.exit_code == 0, prom_res.output
    assert "PROMOTE" in prom_res.output
    assert "dry-run" in prom_res.output
    # Dry-run must not create the promoted.json record
    assert not (models_dir / "promoted" / "promoted.json").exists()
