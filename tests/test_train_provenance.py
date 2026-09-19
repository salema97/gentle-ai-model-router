"""R2/R3 training-side tests: provenance derivation + telemetry loss weighting.

Covers:
- ``derive_label_provenance``: the single shared rule used by the dataset
  manifest, ``train.py`` metrics.json, and ``evaluate.py`` results docs
  (pure-bootstrap / pure-telemetry / mixed datasets).
- ``example_loss_weights`` / ``collate_pointwise`` / ``pointwise_loss``:
  the ``telemetry_weight`` actually reaches the pointwise loss, and
  ``telemetry_weight == 1.0`` stays byte-identical to the unweighted path.
"""

from __future__ import annotations

import pytest

from gentle_ai_model_router.dataset.schema import (
    MODEL_FEATURE_NAMES,
    PROVENANCE_BOOTSTRAP,
    PROVENANCE_TELEMETRY,
    CandidateRef,
    DatasetExample,
    DatasetV1,
    derive_label_provenance,
)
from gentle_ai_model_router.router.config import TrainingConfig
from gentle_ai_model_router.training.train import (
    collate_pointwise,
    example_loss_weights,
    pointwise_loss,
)

AA = "artificial_analysis_intelligence_index"


def _example(idx: int, provenance: str, utility: float = 0.5) -> DatasetExample:
    return DatasetExample(
        example_id=f"ex-{idx}",
        task_id="task-1",
        phase="design",
        task_type="feature",
        context_tokens=10_000,
        task_text="synthetic task",
        candidate=CandidateRef(model=f"test/model-{idx}", deployment="default", effort="high"),
        model_features=[0.0] * 7,
        benchmark_features=[utility],
        benchmark_feature_names=(AA,),
        cost_features={
            "est_input_tokens": 100.0,
            "est_output_tokens": 100.0,
            "est_total_tokens": 200.0,
            "est_cost": 1.0,
        },
        label_utility=utility,
        label_quality_estimate=utility,
        label_provenance=provenance,
        snapshot_date="2026-01-01",
        split="train",
    )


# --------------------------------------------------------------------------- #
# R2: provenance derivation (same rule as DatasetV1.manifest())
# --------------------------------------------------------------------------- #


def test_derive_label_provenance_pure_bootstrap() -> None:
    assert derive_label_provenance([PROVENANCE_BOOTSTRAP, PROVENANCE_BOOTSTRAP, ""]) == (
        PROVENANCE_BOOTSTRAP
    )


def test_derive_label_provenance_pure_telemetry() -> None:
    assert derive_label_provenance([PROVENANCE_TELEMETRY, PROVENANCE_TELEMETRY]) == (
        PROVENANCE_TELEMETRY
    )


def test_derive_label_provenance_mixed_joins_sorted() -> None:
    # Mixed provenance must be explicit, never silently reported as one kind.
    assert derive_label_provenance([PROVENANCE_TELEMETRY, PROVENANCE_BOOTSTRAP]) == (
        "bootstrap_prior+telemetry"
    )


def test_derive_label_provenance_empty_defaults_to_bootstrap() -> None:
    assert derive_label_provenance([]) == PROVENANCE_BOOTSTRAP


def test_manifest_and_derivation_agree_on_mixed_dataset() -> None:
    dataset = DatasetV1(
        name="mixed",
        version=1,
        examples=[
            _example(0, PROVENANCE_BOOTSTRAP),
            _example(1, PROVENANCE_TELEMETRY),
        ],
        benchmark_feature_names=(AA,),
    )
    derived = derive_label_provenance(e.label_provenance for e in dataset.examples)
    assert dataset.manifest()["label_provenance"] == derived == "bootstrap_prior+telemetry"


# --------------------------------------------------------------------------- #
# R3: telemetry_weight config + loss weighting
# --------------------------------------------------------------------------- #


def test_training_config_default_telemetry_weight_is_uniform() -> None:
    assert TrainingConfig().telemetry_weight == 1.0


def test_train_cli_exposes_telemetry_weight_flag() -> None:
    from typer.testing import CliRunner

    from gentle_ai_model_router.cli.main import app

    result = CliRunner().invoke(app, ["train", "--help"])
    assert result.exit_code == 0
    assert "--telemetry-weight" in result.output


def test_example_loss_weights_uniform_returns_none() -> None:
    # weight 1.0: never touch the loss path (byte-identical behavior)...
    rows = [_example(0, PROVENANCE_BOOTSTRAP), _example(1, PROVENANCE_TELEMETRY)]
    assert example_loss_weights(rows, 1.0) is None
    # ...and neither does a != 1.0 weight when NO telemetry row is present.
    rows_boot = [_example(0, PROVENANCE_BOOTSTRAP), _example(1, PROVENANCE_BOOTSTRAP)]
    assert example_loss_weights(rows_boot, 2.0) is None


def test_example_loss_weights_upweights_telemetry_rows() -> None:
    rows = [
        _example(0, PROVENANCE_BOOTSTRAP),
        _example(1, PROVENANCE_TELEMETRY),
        _example(2, PROVENANCE_TELEMETRY),
    ]
    assert example_loss_weights(rows, 2.5) == [1.0, 2.5, 2.5]


def test_collate_pointwise_weight_one_adds_no_loss_weights_key() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    rows = [_example(0, PROVENANCE_BOOTSTRAP), _example(1, PROVENANCE_TELEMETRY)]
    batch = collate_pointwise(rows, _FakeTokenizer(), MODEL_FEATURE_NAMES, 512, 1.0)
    assert "loss_weights" not in batch
    assert batch["labels"].shape == (2,)


def test_collate_pointwise_upweight_adds_loss_weights() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    rows = [_example(0, PROVENANCE_BOOTSTRAP), _example(1, PROVENANCE_TELEMETRY)]
    batch = collate_pointwise(rows, _FakeTokenizer(), MODEL_FEATURE_NAMES, 512, 2.0)
    assert batch["loss_weights"].tolist() == [1.0, 2.0]


def test_pointwise_loss_weight_actually_reaches_the_loss() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")

    outputs = torch.zeros(2)
    labels = torch.ones(2)
    weights = torch.tensor([1.0, 3.0])
    # Plain MSE = 1.0; upweighting the second example 3x gives (1*1 + 1*3)/2.
    assert float(pointwise_loss(outputs, labels)) == 1.0
    assert float(pointwise_loss(outputs, labels, weights)) == 2.0


def test_pointwise_loss_without_weights_is_byte_identical_to_mse() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import torch.nn.functional as F

    torch.manual_seed(0)
    outputs = torch.randn(4)
    labels = torch.randn(4)
    assert pointwise_loss(outputs, labels) == F.mse_loss(outputs, labels)


class _FakeTokenizer:
    """Duck-typed tokenizer: synthetic token ids, batch padding (no HF)."""

    def __call__(self, texts, padding=True, truncation=True, max_length=512, return_tensors=None):
        assert return_tensors == "pt"
        import torch

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
