"""Batched ranker scoring equivalence tests (CPU, tiny random-init model).

The evaluator scores candidates in batches now (padded tokenization + one
forward per batch). These tests pin the equivalence with the old
single-example loop: same scores within 1e-5, same chosen candidate (incl.
the candidate-key tie-break). Guarded with importorskip: no HF downloads.
"""

from __future__ import annotations

from datetime import date, datetime

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from gentle_ai_model_router.dataset.builder import (  # noqa: E402
    DatasetBuilderConfig,
    build_examples,
    write_dataset,
)
from gentle_ai_model_router.registry import db as registry_db  # noqa: E402
from gentle_ai_model_router.router.config import load_config  # noqa: E402
from gentle_ai_model_router.training.evaluate import (  # noqa: E402
    _example_text,
    _ranker_chooser,
    _score_group,
)
from gentle_ai_model_router.training.model import build_model, tiny_modernbert_config  # noqa: E402

AA = "artificial_analysis_intelligence_index"


class _FakeTokenizer:
    """Duck-typed tokenizer: synthetic token ids, batch padding (no HF)."""

    def __call__(self, texts, padding=True, truncation=True, max_length=512, return_tensors=None):
        assert return_tensors == "pt"
        encoded = []
        for text in texts:
            ids = [(hash(text) + i) % 900 + 2 for i in range(8)]  # avoid pad id 0
            encoded.append(ids[:max_length])
        width = max(len(ids) for ids in encoded)
        input_ids = [ids + [0] * (width - len(ids)) for ids in encoded]
        attention = [[1] * len(ids) + [0] * (width - len(ids)) for ids in encoded]
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention),
        }


def _dataset(tmp_path):
    """Real two-candidate groups in the test split (same fixture style as
    test_evaluate.py) so DatasetExample fields are fully realistic.

    Split trick (mirrors tests.test_evaluate._seed_registry): models first
    seen at snap-01 (before train_end) with a LATER price row at snap-03
    (after val_end) -> snapshot_date 2026-03-01 -> split "test".
    """
    engine = registry_db.get_engine(f"sqlite:///{tmp_path / 'router.db'}")
    registry_db.init_schema(engine)
    with registry_db.Session(engine) as session:
        registry_db.record_snapshot(session, "test", "snap-01", datetime(2026, 1, 1), 2)
        registry_db.record_snapshot(session, "test", "snap-03", datetime(2026, 3, 1), 2)
        for canonical, aa in (("test/weak", 30.0), ("test/strong", 90.0)):
            provider = registry_db.get_or_create_provider(session, "test")
            model = registry_db.upsert_model(session, canonical_id=canonical)
            deployment = registry_db.get_or_create_deployment(session, model, provider, "default")
            for effort in ("low", "high"):
                registry_db.upsert_variant(session, deployment, effort, effort)
            registry_db.upsert_benchmark(session, model, AA, aa, None, "snap-01")
            # Price at snap-03 (after val_end) -> examples split "test".
            registry_db.upsert_price(session, deployment, "snap-03", 1.0, 2.0, None)
        session.commit()
    config = load_config(config_path="/nonexistent/router.yaml", data_dir=tmp_path / "data")
    builder = DatasetBuilderConfig(
        name="batch-equivalence",
        phases=["design"],
        task_types=["feature"],
        context_sizes=[10_000],
        train_end=date(2026, 2, 15),
        val_end=date(2026, 2, 20),
        pair_margin=0.0,
    )
    with registry_db.Session(engine) as session:
        dataset = build_examples(session, config, builder)
        write_dataset(dataset, tmp_path / "data")
    return dataset


def _tiny_ranker(dataset):
    torch.manual_seed(0)
    numeric_dim = len(dataset.model_feature_names) + len(dataset.benchmark_feature_names)
    return build_model(tiny_config=tiny_modernbert_config(), numeric_dim=numeric_dim)


def _groups(dataset):
    from gentle_ai_model_router.training.baselines import group_examples

    groups = group_examples([e for e in dataset.examples if e.split == "test"])
    assert groups, "fixture regression: expected non-empty test split"
    return groups


def _single_example_scores(model, tokenizer, dataset, group) -> list[float]:
    """Reference: the OLD loop — one tokenize + one forward per example."""
    scores = []
    for e in group:
        enc = tokenizer(
            [_example_text(dataset, e)],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        numeric = torch.tensor(
            [list(e.model_features) + list(e.benchmark_features)], dtype=torch.float32
        )
        with torch.no_grad():
            scores.append(float(model(enc["input_ids"], enc["attention_mask"], numeric)[0]))
    return scores


@pytest.mark.parametrize("batch_size", [1, 2, 64])
def test_batched_scores_match_single_example(tmp_path, batch_size: int) -> None:
    dataset = _dataset(tmp_path)
    model = _tiny_ranker(dataset)
    tokenizer = _FakeTokenizer()
    model.eval()

    for group in _groups(dataset).values():
        texts = [_example_text(dataset, e) for e in group]
        numeric_rows = [list(e.model_features) + list(e.benchmark_features) for e in group]
        batched = _score_group(
            model,
            tokenizer,
            texts,
            numeric_rows,
            batch_size=batch_size,
            device="cpu",
        )
        single = _single_example_scores(model, tokenizer, dataset, group)
        assert batched == pytest.approx(single, abs=1e-5)


def test_chooser_picks_same_candidate_as_single_example(tmp_path) -> None:
    dataset = _dataset(tmp_path)
    model = _tiny_ranker(dataset)
    tokenizer = _FakeTokenizer()
    model.eval()

    chooser = _ranker_chooser(dataset, model, tokenizer, batch_size=2, device="cpu")
    for group in _groups(dataset).values():
        single_scores = _single_example_scores(model, tokenizer, dataset, group)
        expected = min(
            range(len(group)),
            key=lambda i: (-single_scores[i], group[i].candidate.key),
        )
        assert chooser(group).candidate.key == group[expected].candidate.key


def test_chooser_respects_device_cpu_when_cuda_available(tmp_path) -> None:
    """Explicit --device cpu must win over an available GPU."""
    dataset = _dataset(tmp_path)
    model = _tiny_ranker(dataset)
    tokenizer = _FakeTokenizer()
    chooser = _ranker_chooser(dataset, model, tokenizer, batch_size=4, device="cpu")
    group = next(iter(_groups(dataset).values()))
    assert chooser(group).candidate.key in {e.candidate.key for e in group}
