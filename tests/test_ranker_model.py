"""DeBERTa ranker model tests: tiny random-init config, no HF downloads.

Guarded with pytest.importorskip: the base env has no [train] extra, and
tests must never download from Hugging Face — hence the tiny local config.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from gentle_ai_model_router.training.model import (  # noqa: E402
    build_model,
    collate_examples,
    encode_pair_text,
    tiny_modernbert_config,
)
from gentle_ai_model_router.training.model_pairwise import (  # noqa: E402
    build_pairwise_model,
    pairwise_loss,
)


class _FakeTokenizer:
    """Duck-typed tokenizer: maps text to synthetic token id lists (no HF)."""

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


def _tiny():
    return tiny_modernbert_config()


def test_pointwise_forward_shape() -> None:
    torch.manual_seed(0)
    model = build_model(tiny_config=_tiny(), numeric_dim=6)
    input_ids = torch.randint(2, 1024, (2, 16))
    attention_mask = torch.ones((2, 16), dtype=torch.long)
    numeric = torch.randn(2, 6)
    scores = model(input_ids, attention_mask, numeric)
    assert scores.shape == (2,)


def test_pairwise_forward_shape_and_loss() -> None:
    torch.manual_seed(0)
    model = build_pairwise_model(tiny_config=_tiny(), numeric_dim=6)
    ids_a = torch.randint(2, 1024, (2, 16))
    ids_b = torch.randint(2, 1024, (2, 16))
    mask = torch.ones((2, 16), dtype=torch.long)
    numeric = torch.randn(2, 6)
    s_a, s_b, diff = model(ids_a, mask, numeric, ids_b, mask, numeric)
    assert s_a.shape == (2,) and s_b.shape == (2,) and diff.shape == (2,)
    loss = pairwise_loss(diff)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_overfit_sanity_loss_decreases() -> None:
    """4 examples, 3 optimizer steps, tiny model: MSE must decrease."""
    torch.manual_seed(0)
    model = build_model(tiny_config=_tiny(), numeric_dim=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    input_ids = torch.randint(2, 1024, (4, 16))
    attention_mask = torch.ones((4, 16), dtype=torch.long)
    numeric = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    labels = torch.tensor([0.9, 0.1, 0.8, 0.2])

    with torch.no_grad():
        initial = torch.nn.functional.mse_loss(model(input_ids, attention_mask, numeric), labels)
    for _ in range(3):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(input_ids, attention_mask, numeric), labels)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final = torch.nn.functional.mse_loss(model(input_ids, attention_mask, numeric), labels)
    assert final < initial


def test_collate_examples_with_fake_tokenizer() -> None:
    from gentle_ai_model_router.dataset.schema import CandidateRef, DatasetExample

    example = DatasetExample(
        example_id="ex-1",
        task_id="task-1",
        phase="design",
        task_type="feature",
        context_tokens=10_000,
        task_text="[phase] design",
        candidate=CandidateRef(model="test/m", deployment="default", effort="low"),
        model_features=[1.0, 2.0],
        benchmark_features=[50.0],
        benchmark_feature_names=("aa_index",),
        cost_features={"est_total_tokens": 10_000.0, "est_cost": 0.01},
        label_utility=0.7,
        label_quality_estimate=0.8,
        label_provenance="bootstrap_prior",
        snapshot_date="2026-01-01",
        split="train",
    )
    batch = collate_examples([example], _FakeTokenizer(), ("f1", "f2"), max_length=32)
    assert batch["input_ids"].shape[0] == 1
    assert batch["numeric_features"].shape == (1, 3)  # 2 model + 1 benchmark
    assert batch["labels"].tolist() == pytest.approx([0.7])
    # The features must be visible to the encoder as key=value text.
    text = encode_pair_text(
        example.task_text,
        example.candidate.model,
        example.candidate.deployment,
        example.candidate.effort,
        "aa_index=50",
    )
    assert "[candidate] model=test/m" in text
    assert "effort=low" in text
    assert "[features] aa_index=50" in text
