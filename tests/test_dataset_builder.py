"""Dataset builder tests: synthetic dated registries, no network.

Fixture timeline:
- snap-2026-01-01: test/old-model (benchmarks + price)
- snap-2026-02-01: test/mid-model (benchmarks); mid price row at snap-2026-03
- snap-2026-03-01: test/new-model (benchmarks + price)

With train_end=2026-02-15 / val_end=2026-02-20:
- old-model → train (but dropped by task-overlap anti-leakage)
- mid-model → test (first seen 2026-02-01, example date 2026-03-01)
- new-model → temporal_test (first seen after train_end)
"""

from __future__ import annotations

from datetime import date, datetime

import pytest
from sqlalchemy.orm import Session

from gentle_ai_model_router.dataset.builder import (
    DatasetBuilderConfig,
    DatasetBuildError,
    DatasetLeakageError,
    build_examples,
    load_dataset,
    write_dataset,
)
from gentle_ai_model_router.dataset.schema import PROVENANCE_BOOTSTRAP, PROVENANCE_STATEMENT
from gentle_ai_model_router.registry import db as registry_db
from gentle_ai_model_router.registry.models import Model
from gentle_ai_model_router.router.config import PhaseConfig, RouterConfig, load_config
from gentle_ai_model_router.router.policy import benchmark_priors, effort_quality

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


def _engine(tmp_path):
    engine = registry_db.get_engine(f"sqlite:///{tmp_path / 'router.db'}")
    registry_db.init_schema(engine)
    return engine


def _snap(session: Session, snapshot_id: str) -> None:
    registry_db.record_snapshot(
        session, "test", snapshot_id, DATES[snapshot_id], record_count=1
    )


def _add_model(
    session: Session,
    canonical: str,
    benchmark_snap: str,
    price_snap: str | None,
    benchmarks: dict[str, float],
    price: tuple[float, float] = (2.0, 8.0),
    variants: tuple[str, ...] = ("low", "medium", "high"),
) -> None:
    provider = registry_db.get_or_create_provider(session, canonical.split("/")[0])
    model = registry_db.upsert_model(session, canonical_id=canonical, tool_calling=True)
    deployment = registry_db.get_or_create_deployment(session, model, provider, "default")
    for effort in variants:
        registry_db.upsert_variant(session, deployment, effort, effort)
    for key, score in benchmarks.items():
        name, _, category = key.partition(":")
        registry_db.upsert_benchmark(
            session, model, name, score, category or None, benchmark_snap
        )
    if price_snap is not None:
        registry_db.upsert_price(session, deployment, price_snap, price[0], price[1], None)


@pytest.fixture
def builder_config() -> DatasetBuilderConfig:
    return DatasetBuilderConfig(
        name="test-priors",
        phases=["design"],
        task_types=["feature"],
        context_sizes=[10_000],
        train_end=TRAIN_END,
        val_end=VAL_END,
    )


@pytest.fixture
def dated_session(tmp_path):
    engine = _engine(tmp_path)
    with registry_db.Session(engine) as session:
        for snap in DATES:
            _snap(session, snap)
        _add_model(session, "test/old-model", SNAP_OLD, SNAP_OLD, {AA: 50.0, ARENA_TEXT: 90.0})
        _add_model(
            session,
            "test/mid-model",
            SNAP_MID,
            SNAP_NEW,  # price known only later -> example date 2026-03-01
            {AA: 40.0, ARENA_TEXT: 95.0},
        )
        _add_model(session, "test/new-model", SNAP_NEW, SNAP_NEW, {AA: 45.0, ARENA_TEXT: 85.0})
        session.commit()
        yield session


def _design_config(tmp_path) -> RouterConfig:
    return load_config(
        config_path="/nonexistent/router.yaml",
        data_dir=tmp_path / "data",
    ).model_copy(
        update={
            "phases": {
                "design": PhaseConfig(
                    threshold_quality=0.85, weights={AA: 1.0, ARENA_TEXT: 1.0}
                )
            }
        }
    )


# --------------------------------------------------------------------------- #
# Labels & provenance
# --------------------------------------------------------------------------- #


def test_labels_are_bootstrap_priors(dated_session, builder_config, tmp_path) -> None:
    dataset = build_examples(dated_session, _design_config(tmp_path), builder_config)
    assert dataset.examples, "expected examples"
    assert all(e.label_provenance == PROVENANCE_BOOTSTRAP for e in dataset.examples)
    assert all(p.label_provenance == PROVENANCE_BOOTSTRAP for p in dataset.pairs)
    assert "NOT ground truth" in dataset.label_provenance_statement
    assert PROVENANCE_STATEMENT in dataset.label_provenance_statement


def test_utility_label_matches_policy_quality_model(
    dated_session, builder_config, tmp_path
) -> None:
    """utility == clip(quality - cost_weight * tokens/max_tokens, 0, 1), using
    the SAME prior + quality functions the deterministic policy uses."""
    config = _design_config(tmp_path)
    dataset = build_examples(dated_session, config, builder_config)
    # Recompute priors exactly like the policy does.
    registry_models = dated_session.query(Model).all()
    priors, _ = benchmark_priors(
        dated_session, registry_models, {AA: 1.0, ARENA_TEXT: 1.0}, config.policy.flat_prior
    )
    by_canonical = {m.canonical_id: m for m in registry_models}
    for group_task in {e.task_id for e in dataset.examples}:
        group = [e for e in dataset.examples if e.task_id == group_task]
        max_tokens = max(e.cost_features["est_total_tokens"] for e in group)
        for e in group:
            prior = priors[by_canonical[e.candidate.model].id]
            quality = effort_quality(prior, e.candidate.effort, config.policy)
            assert e.label_quality_estimate == pytest.approx(quality, abs=1e-9)
            penalty = builder_config.cost_weight * e.cost_features["est_total_tokens"] / max_tokens
            expected = max(0.0, min(1.0, quality - penalty))
            assert e.label_utility == pytest.approx(expected, abs=1e-9)


def test_utility_is_normalized_to_unit_interval(dated_session, builder_config, tmp_path) -> None:
    dataset = build_examples(dated_session, _design_config(tmp_path), builder_config)
    for e in dataset.examples:
        assert 0.0 <= e.label_utility <= 1.0


def test_threshold_penalty_penalizes_failing_candidates(
    dated_session, builder_config, tmp_path
) -> None:
    config = _design_config(tmp_path)
    threshold = config.phase_config("design").threshold_quality

    dataset_default = build_examples(dated_session, config, builder_config)

    penalty_value = 0.25
    builder_penalized = builder_config.model_copy(update={"threshold_penalty": penalty_value})
    dataset_penalized = build_examples(dated_session, config, builder_penalized)

    assert dataset_penalized.examples
    failing_count = 0
    passing_count = 0

    for def_ex, pen_ex in zip(dataset_default.examples, dataset_penalized.examples, strict=True):
        assert def_ex.example_id == pen_ex.example_id
        assert def_ex.label_quality_estimate == pen_ex.label_quality_estimate

        if pen_ex.label_quality_estimate < threshold:
            failing_count += 1
            expected_utility = max(0.0, def_ex.label_utility - penalty_value)
            assert pen_ex.label_utility == pytest.approx(expected_utility, abs=1e-9)
            if def_ex.label_utility > 0.0:
                assert pen_ex.label_utility < def_ex.label_utility
        else:
            passing_count += 1
            assert pen_ex.label_utility == pytest.approx(def_ex.label_utility, abs=1e-9)

    assert failing_count > 0, "expected candidates below threshold in test fixture"
    assert passing_count > 0, "expected candidates meeting threshold in test fixture"


def test_hard_threshold_zeros_failing_candidates(
    dated_session, builder_config, tmp_path
) -> None:
    config = _design_config(tmp_path)
    threshold = config.phase_config("design").threshold_quality

    dataset_default = build_examples(dated_session, config, builder_config)

    builder_hard = builder_config.model_copy(update={"hard_threshold": True})
    dataset_hard = build_examples(dated_session, config, builder_hard)

    assert dataset_hard.examples
    failing_count = 0
    passing_count = 0

    for def_ex, hard_ex in zip(dataset_default.examples, dataset_hard.examples, strict=True):
        assert def_ex.example_id == hard_ex.example_id
        assert def_ex.label_quality_estimate == hard_ex.label_quality_estimate

        if hard_ex.label_quality_estimate < threshold:
            failing_count += 1
            assert hard_ex.label_utility == 0.0
            if def_ex.label_utility > 0.0:
                assert hard_ex.label_utility < def_ex.label_utility
        else:
            passing_count += 1
            assert hard_ex.label_utility == pytest.approx(def_ex.label_utility, abs=1e-9)

    assert failing_count > 0, "expected candidates below threshold in test fixture"
    assert passing_count > 0, "expected candidates meeting threshold in test fixture"


# --------------------------------------------------------------------------- #
# Temporal split & anti-leakage
# --------------------------------------------------------------------------- #


def test_temporal_split_and_first_appearance(dated_session, builder_config, tmp_path) -> None:
    dataset = build_examples(dated_session, _design_config(tmp_path), builder_config)
    by_model = {}
    for e in dataset.examples:
        by_model.setdefault(e.candidate.model, set()).add(e.split)

    # old-model rows are train-dated but share task_ids with newer models:
    # anti-leakage drops them (counted, loud).
    assert dataset.dropped_train_task_overlap > 0
    assert "test/old-model" not in by_model

    # mid-model: first seen before train_end, feature date after val_end.
    assert by_model["test/mid-model"] == {"test"}
    mid = next(e for e in dataset.examples if e.candidate.model == "test/mid-model")
    assert mid.snapshot_date == "2026-03-01"
    assert date.fromisoformat(mid.snapshot_date) >= TRAIN_END

    # new-model: first appears after the train cutoff -> temporal_test.
    assert by_model["test/new-model"] == {"temporal_test"}

    # Hard ordering invariant: remaining train rows predate test rows.
    train_dates = [e.snapshot_date for e in dataset.examples if e.split == "train"]
    test_dates = [e.snapshot_date for e in dataset.examples if e.split == "test"]
    if train_dates and test_dates:
        assert max(train_dates) < min(test_dates)

    # No train example shares a task_id with non-train (enforced).
    non_train_tasks = {e.task_id for e in dataset.examples if e.split != "train"}
    assert all(
        e.task_id not in non_train_tasks for e in dataset.examples if e.split == "train"
    )


def test_pairs_inherit_splits_and_respect_temporal(dated_session, builder_config, tmp_path) -> None:
    dataset = build_examples(dated_session, _design_config(tmp_path), builder_config)
    assert dataset.pairs
    for pair in dataset.pairs:
        # Pairs only exist within one group, so both members share split class.
        assert pair.split in {"test", "temporal_test", "train", "validation"}
        if (
            pair.candidate_a.model == "test/new-model"
            or pair.candidate_b.model == "test/new-model"
        ):
            assert pair.split == "temporal_test"


def test_leakage_violation_is_build_error(dated_session, builder_config, tmp_path) -> None:
    builder = builder_config.model_copy(update={"as_of": date(2026, 2, 15)})
    with pytest.raises(DatasetLeakageError, match="anti-leakage violation"):
        build_examples(dated_session, _design_config(tmp_path), builder)


def test_undatable_row_is_build_error(tmp_path, builder_config) -> None:
    engine = _engine(tmp_path)
    with registry_db.Session(engine) as session:
        _snap(session, SNAP_OLD)
        # Benchmark row references a snapshot that was never recorded.
        _add_model(session, "test/ghost", "snap-unknown", None, {AA: 50.0})
        session.commit()
        with pytest.raises(DatasetBuildError, match="unknown snapshot_id"):
            build_examples(session, _design_config(tmp_path), builder_config)


# --------------------------------------------------------------------------- #
# Pairwise generation
# --------------------------------------------------------------------------- #


def test_pair_margin_filtering(tmp_path, builder_config) -> None:
    engine = _engine(tmp_path)
    with registry_db.Session(engine) as session:
        _snap(session, SNAP_OLD)
        # Identical priors + prices -> identical utilities -> no pair passes
        # any positive margin. One variant per model: C(3,2) = 3 pairs at
        # zero margin.
        for name in ("test/m-1", "test/m-2", "test/m-3"):
            _add_model(
                session, name, SNAP_OLD, SNAP_OLD, {AA: 50.0, ARENA_TEXT: 90.0}, variants=("low",)
            )
        session.commit()
        config = _design_config(tmp_path)
        builder = builder_config.model_copy(update={"train_end": date(2026, 6, 1)})
        dataset = build_examples(session, config, builder)
        assert dataset.examples
        assert dataset.pairs == []

        # Zero margin admits ties (deterministic candidate-key ordering).
        builder0 = builder.model_copy(update={"pair_margin": 0.0})
        dataset0 = build_examples(session, config, builder0)
        assert len(dataset0.pairs) == 3  # C(3, 2) within the single group


def test_max_pairs_per_group_cap(tmp_path, builder_config) -> None:
    engine = _engine(tmp_path)
    with registry_db.Session(engine) as session:
        _snap(session, SNAP_OLD)
        # Divergent utilities: 4 models x 3 efforts = 12 candidates per group.
        _add_model(session, "test/p-1", SNAP_OLD, SNAP_OLD, {AA: 90.0}, price=(1.0, 2.0))
        _add_model(session, "test/p-2", SNAP_OLD, SNAP_OLD, {AA: 60.0}, price=(2.0, 6.0))
        _add_model(session, "test/p-3", SNAP_OLD, SNAP_OLD, {AA: 30.0}, price=(3.0, 9.0))
        _add_model(session, "test/p-4", SNAP_OLD, SNAP_OLD, {AA: 10.0}, price=(5.0, 15.0))
        session.commit()
        builder = builder_config.model_copy(
            update={"train_end": date(2026, 6, 1), "max_pairs_per_group": 2}
        )
        dataset = build_examples(session, _design_config(tmp_path), builder)
        per_group: dict[str, int] = {}
        for pair in dataset.pairs:
            per_group[pair.task_id] = per_group.get(pair.task_id, 0) + 1
        assert per_group
        assert all(count <= 2 for count in per_group.values())
        assert dataset.pairs  # margin filter still admits confident pairs
        assert all(p.margin >= builder.pair_margin for p in dataset.pairs)


def test_pair_winner_beats_loser_by_margin(dated_session, builder_config, tmp_path) -> None:
    dataset = build_examples(dated_session, _design_config(tmp_path), builder_config)
    for pair in dataset.pairs:
        assert pair.score_a > pair.score_b
        assert pair.margin >= builder_config.pair_margin


# --------------------------------------------------------------------------- #
# Persistence round-trip
# --------------------------------------------------------------------------- #


def test_write_and_load_roundtrip(dated_session, builder_config, tmp_path) -> None:
    config = _design_config(tmp_path)
    dataset = build_examples(dated_session, config, builder_config)
    out_dir = write_dataset(dataset, tmp_path / "data")
    assert (out_dir / "examples.parquet").exists()
    manifest = json_loads_manifest(out_dir / "manifest.json")
    assert manifest["label_provenance"] == PROVENANCE_BOOTSTRAP
    assert "NOT ground truth" in manifest["label_provenance_statement"]
    assert manifest["split_counts"]
    assert manifest["threshold_penalty"] == 0.0
    assert manifest["hard_threshold"] is False

    reloaded = load_dataset(out_dir)
    assert len(reloaded.examples) == len(dataset.examples)
    assert len(reloaded.pairs) == len(dataset.pairs)
    assert reloaded.version == dataset.version
    assert reloaded.threshold_penalty == 0.0
    assert reloaded.hard_threshold is False
    first = dataset.examples[0]
    again = next(e for e in reloaded.examples if e.example_id == first.example_id)
    assert again.label_utility == pytest.approx(first.label_utility)
    assert again.candidate == first.candidate
    assert again.context_tokens == first.context_tokens


def test_write_and_load_roundtrip_with_threshold_conditioning(
    dated_session, builder_config, tmp_path
) -> None:
    config = _design_config(tmp_path)
    custom_builder = builder_config.model_copy(
        update={"threshold_penalty": 0.2, "hard_threshold": True}
    )
    dataset = build_examples(dated_session, config, custom_builder)
    out_dir = write_dataset(dataset, tmp_path / "data", builder=custom_builder)
    manifest = json_loads_manifest(out_dir / "manifest.json")
    assert manifest["threshold_penalty"] == 0.2
    assert manifest["hard_threshold"] is True

    reloaded = load_dataset(out_dir)
    assert reloaded.threshold_penalty == 0.2
    assert reloaded.hard_threshold is True


def json_loads_manifest(path):
    import json

    return json.loads(path.read_text())
