"""Training loop for the ModernBERT ranker (HF Trainer).

Objectives (docs/training.md):
- ``pointwise``: MSE between the scalar score and the bootstrap utility.
  Simple, calibrated-ish scores; sensitive to label noise scale.
- ``pairwise``: BCE-with-logits on s_a − s_b for preference pairs (label: A
  beats B). More robust to noisy bootstrap labels because only the ORDER
  matters. DEFAULT once pairs exist.
- ``listwise``: documented future work — our dataset builder does generate
  candidate lists per task group, but pairwise is preferred while labels are
  noisy priors; listwise (e.g. RankNet/ListMLE-style softmax over the group)
  will be revisited once telemetry-derived labels exist.

Determinism: seeds fixed for python/numpy/torch; HF Trainer runs with the
given seed. Checkpoints are versioned directories
(``models/modernbert-router/v<N>/``) carrying model + tokenizer + config +
metrics + full provenance (dataset version, feature schema, snapshots, git).
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gentle_ai_model_router.dataset.builder import load_dataset
from gentle_ai_model_router.dataset.schema import (
    FEATURE_SCHEMA_VERSION,
    NORMALIZATION_VERSION,
    PROVENANCE_TELEMETRY,
    derive_label_provenance,
)
from gentle_ai_model_router.router.config import RouterConfig, TrainingConfig

logger = logging.getLogger(__name__)


def example_loss_weights(
    examples: list[Any],
    telemetry_weight: float,
) -> list[float] | None:
    """Per-example pointwise loss weights, or None when uniform.

    ``telemetry_weight`` multiplies the loss contribution of
    ``label_provenance='telemetry'`` examples; everything else keeps weight
    1.0. Returns None when no example would deviate from 1.0 (including
    ``telemetry_weight == 1.0``): callers then use the plain mean MSE, which
    is byte-identical to the pre-weighting behavior.
    """
    if telemetry_weight == 1.0:
        return None
    weights = [
        telemetry_weight if getattr(e, "label_provenance", None) == PROVENANCE_TELEMETRY else 1.0
        for e in examples
    ]
    if all(w == 1.0 for w in weights):
        return None
    return weights


def collate_pointwise(
    rows: list[Any],
    tokenizer: Any,
    model_feature_names: tuple[str, ...],
    max_length: int,
    telemetry_weight: float,
) -> dict[str, Any]:
    """Pointwise batch: tokenized features + labels + optional loss weights.

    A ``loss_weights`` tensor is added only when telemetry weighting is
    active (see :func:`example_loss_weights`); otherwise the batch is exactly
    what :func:`collate_examples` produces.
    """
    from gentle_ai_model_router.training.model import collate_examples

    batch = collate_examples(rows, tokenizer, model_feature_names, max_length)
    weights = example_loss_weights(rows, telemetry_weight)
    if weights is not None:
        import torch

        batch["loss_weights"] = torch.tensor(weights, dtype=torch.float32)
    return batch


def pointwise_loss(outputs: Any, labels: Any, loss_weights: Any = None) -> Any:
    """MSE loss; ``loss_weights`` (when given) re-weights each example.

    With ``loss_weights=None`` this is exactly ``F.mse_loss(outputs, labels)``
    (mean reduction) — the historical unweighted path.
    """
    import torch

    if loss_weights is None:
        return torch.nn.functional.mse_loss(outputs, labels)
    per_example = torch.nn.functional.mse_loss(outputs, labels, reduction="none")
    return (per_example * loss_weights).mean()


def _git_commit() -> str:
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # pragma: no cover
        return "unknown"


def train(
    config: RouterConfig,
    training: TrainingConfig,
    dataset_path: str | Path,
) -> Path:
    """Train the ranker and return the checkpoint directory.

    Requires the ``[train]`` extra (torch/transformers/accelerate); imports
    are lazy so the rest of the package works without them.
    """
    try:
        import numpy as np
        import torch
        from transformers import (
            AutoTokenizer,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "training requires the [train] extra: "
            "pip install 'gentle-ai-model-router[train]'"
        ) from exc

    from gentle_ai_model_router.training.device import resolve_device
    from gentle_ai_model_router.training.model import build_model
    from gentle_ai_model_router.training.model_pairwise import (
        build_pairwise_model,
        collate_pairs,
        pairwise_loss,
    )

    dataset = load_dataset(dataset_path)
    set_seed(training.seed)
    np.random.seed(training.seed)
    torch.manual_seed(training.seed)

    # Resolve the device up front (fail fast on --device cuda without a GPU
    # or an invalid value) and log it clearly; placement happens once the
    # model exists below. Trainer respects an already-placed model.
    device = resolve_device(training.device, torch.cuda.is_available())
    logger.info("training device=%s", device)
    if device.startswith("cuda") and getattr(training, "max_vram_fraction", None) is not None:
        torch.cuda.set_per_process_memory_fraction(training.max_vram_fraction, 0)
        logger.info("capped CUDA memory fraction at %.2f", training.max_vram_fraction)

    numeric_dim = len(dataset.model_feature_names) + len(dataset.benchmark_feature_names)
    tokenizer = AutoTokenizer.from_pretrained(training.model_name)

    class _RowsDataset(torch.utils.data.Dataset):
        def __init__(self, rows: list[Any]) -> None:
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, idx: int) -> Any:
            return self.rows[idx]

    if training.objective == "pointwise":
        train_rows = [e for e in dataset.examples if e.split == "train"]
        model = build_model(training.model_name, numeric_dim=numeric_dim)
        examples_by_key: dict[str, Any] = {}

        def collate(rows: list[Any]) -> dict[str, Any]:
            return collate_pointwise(
                rows,
                tokenizer,
                dataset.model_feature_names,
                training.max_length,
                training.telemetry_weight,
            )

        def compute_loss(
            model_: Any, inputs: dict[str, Any], return_outputs: bool = False, **kwargs: Any
        ) -> Any:
            labels = inputs.pop("labels")
            loss_weights = inputs.pop("loss_weights", None)
            outputs = model_(
                inputs["input_ids"], inputs["attention_mask"], inputs["numeric_features"]
            )
            loss = pointwise_loss(outputs, labels, loss_weights)
            return (loss, outputs) if return_outputs else loss

    elif training.objective == "pairwise":
        from gentle_ai_model_router.training.model_pairwise import precompute_pairwise_cache

        pair_keys = {p.candidate_a.key for p in dataset.pairs} | {
            p.candidate_b.key for p in dataset.pairs
        }
        examples_by_key = {}
        for e in dataset.examples:
            if e.candidate.key in pair_keys and e.candidate.key not in examples_by_key:
                examples_by_key[e.candidate.key] = e
                if len(examples_by_key) == len(pair_keys):
                    break
        cached_encodings = precompute_pairwise_cache(
            examples_by_key, tokenizer, dataset.model_feature_names, training.max_length
        )
        train_rows = [
            p for p in dataset.pairs if p.split == "train"
        ]
        model = build_pairwise_model(training.model_name, numeric_dim=numeric_dim)

        def collate(rows: list[Any]) -> dict[str, Any]:
            return collate_pairs(
                rows,
                examples_by_key,
                tokenizer,
                dataset.model_feature_names,
                training.max_length,
                cached_encodings=cached_encodings,
            )

        def compute_loss(
            model_: Any, inputs: dict[str, Any], return_outputs: bool = False, **kwargs: Any
        ) -> Any:
            _, _, diff = model_(
                inputs["input_ids_a"],
                inputs["attention_mask_a"],
                inputs["numeric_a"],
                inputs["input_ids_b"],
                inputs["attention_mask_b"],
                inputs["numeric_b"],
            )
            loss = pairwise_loss(diff)
            return (loss, diff) if return_outputs else loss

    else:  # pragma: no cover - config validation happens earlier
        raise ValueError(f"unknown objective '{training.objective}'")

    model.to(device)

    if not train_rows:
        raise ValueError(
            f"dataset at {dataset_path} has no '{training.objective}' train rows "
            "(temporal split may have emptied train — check train_end)"
        )

    out_root = Path(training.output_dir)
    existing = [p for p in out_root.glob("v*") if p.is_dir()]
    version = 1 + max((int(p.name[1:]) for p in existing if p.name[1:].isdigit()), default=0)
    out_dir = out_root / f"v{version}"

    args = TrainingArguments(
        output_dir=str(out_dir / "hf"),
        per_device_train_batch_size=training.batch_size,
        learning_rate=training.learning_rate,
        num_train_epochs=training.epochs,
        seed=training.seed,
        data_seed=training.seed,
        logging_steps=10,
        save_strategy="no",  # we save manually with full provenance below
        report_to=[],
        disable_tqdm=training.disable_tqdm,
        bf16=training.bf16 and device.startswith("cuda"),
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )

    # transformers 5.x changed the custom-loss contract: ``compute_loss_func``
    # now receives (outputs, labels), not (model, inputs). The stable override
    # point is the ``compute_loss`` method, so we subclass instead.
    class _LossTrainer(Trainer):
        def __init__(self, *args: Any, loss_fn: Any = None, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._loss_fn = loss_fn

        def compute_loss(
            self,
            model_: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any = None,
        ) -> Any:
            return self._loss_fn(model_, inputs, return_outputs=return_outputs)

    trainer = _LossTrainer(
        model=model,
        args=args,
        train_dataset=_RowsDataset(train_rows),
        data_collator=collate,
        loss_fn=compute_loss,
    )
    logger.info(
        "training start objective=%s rows=%d out=%s",
        training.objective,
        len(train_rows),
        out_dir,
    )
    train_result = trainer.train()

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    metrics = {
        "objective": training.objective,
        "model_name": training.model_name,
        "train_rows": len(train_rows),
        "train_loss": float(train_result.training_loss),
        "epochs": training.epochs,
        "learning_rate": training.learning_rate,
        "seed": training.seed,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "dataset": {"name": dataset.name, "version": dataset.version},
        "source_snapshot_ids": dataset.source_snapshot_ids,
        # Truthful provenance: derived from the dataset itself (same rule as
        # DatasetV1.manifest()), never a hardcoded assumption.
        "label_provenance": derive_label_provenance(
            e.label_provenance for e in dataset.examples
        ),
        # Feature-name recording: serving (router/neural.py) resolves the
        # benchmark feature vector from THESE names, so the served numeric
        # dim always matches the trained dim even for non-15-name datasets.
        "model_feature_names": list(dataset.model_feature_names),
        "benchmark_feature_names": list(dataset.benchmark_feature_names),
        "git_commit": _git_commit(),
        "created_at": datetime.now(UTC).isoformat(),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    run_config = training.model_dump()
    run_config["dataset_path"] = str(dataset_path)
    (out_dir / "training_config.json").write_text(json.dumps(run_config, indent=2) + "\n")
    logger.info("training done: %s", out_dir)
    return out_dir
