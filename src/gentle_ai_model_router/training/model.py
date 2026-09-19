"""DeBERTa-V3 ranker model: text encoder + numeric-feature MLP branch.

This is a RANKER (scalar score per (query, candidate)), not a multiclass
classifier: the objective is to order candidates by expected utility, which
matches the project's objective (min tokens for sufficient success) far
better than fixed classes.

Hybrid input design (deliberate, docs/training.md):
1. TEXT: task_text + candidate text + features serialized as ``key=value``
   pairs, so the encoder can attend over features semantically and generalize
   to unseen feature combinations.
2. NUMERIC: the same feature vector concatenated with the pooled encoder
   output into a small MLP head — exact numeric gradients that the
   discretized text path cannot provide.

Both branches feed one scalar head; the score is comparable across
candidates of the same task group.

Heavy deps (torch/transformers) are imported lazily so the base install and
all core tests stay light. Use :func:`tiny_deberta_config` in tests to build
a random-initialized tiny encoder — no HF downloads, fully offline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    import torch
    from transformers import DebertaV2Config


def _require_train_extra() -> None:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised in base env
        raise RuntimeError(
            "training requires the [train] extra: pip install 'gentle-ai-model-router[train]'"
        ) from exc


def tiny_deberta_config(
    hidden_size: int = 96,
    num_hidden_layers: int = 2,
    num_attention_heads: int = 4,
    intermediate_size: int = 192,
    vocab_size: int = 1024,
    max_position_embeddings: int = 256,
) -> DebertaV2Config:
    """Random-initialized tiny DeBERTa-V2 config for tests/dev smoke runs.

    Deliberately NOT a pretrained checkpoint: tests must never download from
    Hugging Face. Vocab/max positions are small because the smoke tests feed
    synthetic token ids, not real text.
    """
    _require_train_extra()
    from transformers import DebertaV2Config

    return DebertaV2Config(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        max_position_embeddings=max_position_embeddings,
        pooler_hidden_size=hidden_size,
        pooler_num_attention_heads=num_attention_heads,
        relative_attention=True,
        max_relative_positions=64,
        norm_rel_ebd="layer_norm",
        position_buckets=256,
    )


def build_model(
    model_name: str = "microsoft/deberta-v3-base",
    numeric_dim: int = 0,
    tiny_config: DebertaV2Config | None = None,
) -> torch.nn.Module:
    """Pointwise ranker: DeBERTa encoder + numeric MLP branch → scalar score.

    ``tiny_config`` overrides ``model_name`` (random init, offline tests).
    """
    _require_train_extra()
    import torch
    from transformers import AutoModel, DebertaV2Model

    if tiny_config is not None:
        encoder = DebertaV2Model(tiny_config)
        hidden = tiny_config.hidden_size
    else:
        encoder = AutoModel.from_pretrained(model_name)
        hidden = encoder.config.hidden_size

    class _Ranker(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = encoder
            mlp_input = hidden + numeric_dim
            self.head = torch.nn.Sequential(
                torch.nn.Linear(mlp_input, hidden),
                torch.nn.GELU(),
                torch.nn.Linear(hidden, 1),
            )

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            numeric_features: torch.Tensor | None = None,
        ) -> torch.Tensor:
            """Return shape (batch,) scalar scores."""
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            pooled = outputs.last_hidden_state[:, 0]  # [CLS] pooling
            if numeric_features is not None:
                pooled = torch.cat([pooled, numeric_features], dim=-1)
            return self.head(pooled).squeeze(-1)

        def save_pretrained(self, out_dir: str, **kwargs: Any) -> None:
            """Save encoder in HF format (dir root) + MLP head weights (head.pt).

            Keeping the encoder at the root lets ``AutoModel.from_pretrained``
            load it directly (used by the evaluation checkpoint path).
            """
            from pathlib import Path

            out = Path(out_dir)
            out.mkdir(parents=True, exist_ok=True)
            self.encoder.save_pretrained(out, **kwargs)
            torch.save(self.head.state_dict(), out / "head.pt")

    return _Ranker()


def encode_pair_text(
    task_text: str,
    candidate_model: str,
    candidate_deployment: str,
    candidate_effort: str,
    features_text: str,
) -> str:
    """Concatenated input text for the encoder (kept here so train/tests agree)."""
    return (
        f"[task] {task_text}\n"
        f"[candidate] model={candidate_model}; deployment={candidate_deployment}; "
        f"effort={candidate_effort}\n"
        f"[features] {features_text}"
    )


def collate_examples(
    examples: list[Any],
    tokenizer: Any,
    model_feature_names: tuple[str, ...],
    max_length: int = 512,
) -> dict[str, torch.Tensor]:
    """Tokenize pointwise examples into a model batch (numpy-free, deterministic)."""
    _require_train_extra()
    import torch

    from gentle_ai_model_router.dataset.builder import render_features_text

    texts = [
        encode_pair_text(
            e.task_text,
            e.candidate.model,
            e.candidate.deployment,
            e.candidate.effort,
            render_features_text(
                list(model_feature_names) + list(e.benchmark_feature_names),
                list(e.model_features) + list(e.benchmark_features),
            ),
        )
        for e in examples
    ]
    enc = tokenizer(
        texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )
    numeric = torch.tensor(
        [e.model_features + e.benchmark_features for e in examples], dtype=torch.float32
    )
    labels = torch.tensor([e.label_utility for e in examples], dtype=torch.float32)
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "numeric_features": numeric,
        "labels": labels,
    }
