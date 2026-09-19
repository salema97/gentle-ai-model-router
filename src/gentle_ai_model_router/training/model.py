"""ModernBERT ranker model: text encoder + numeric-feature MLP branch.

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
all core tests stay light. Use :func:`tiny_modernbert_config` in tests to build
a random-initialized tiny encoder — no HF downloads, fully offline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

try:
    import torch
    import torch.nn as nn

    _ModuleBase = nn.Module
except ImportError:
    torch = None  # type: ignore[assignment]
    _ModuleBase = object  # type: ignore[misc,assignment]

if TYPE_CHECKING:  # pragma: no cover
    import torch
    from transformers import ModernBertConfig


def _require_train_extra() -> None:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised in base env
        raise RuntimeError(
            "training requires the [train] extra: pip install 'gentle-ai-model-router[train]'"
        ) from exc


def tiny_modernbert_config(
    hidden_size: int = 96,
    num_hidden_layers: int = 2,
    num_attention_heads: int = 4,
    intermediate_size: int = 192,
    vocab_size: int = 1024,
    max_position_embeddings: int = 256,
) -> ModernBertConfig:
    """Random-initialized tiny ModernBERT config for tests/dev smoke runs.

    Deliberately NOT a pretrained checkpoint: tests must never download from
    Hugging Face. Vocab/max positions are small because the smoke tests feed
    synthetic token ids, not real text.
    """
    _require_train_extra()
    from transformers import ModernBertConfig

    return ModernBertConfig(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        max_position_embeddings=max_position_embeddings,
        sliding_window=32,
        global_attn_every_n_layers=2,
        # The real ModernBERT special ids exceed the tiny test vocab;
        # remap them inside the synthetic range (tests feed fake ids).
        pad_token_id=0,
        unk_token_id=1,
        cls_token_id=2,
        sep_token_id=3,
        eos_token_id=4,
        mask_token_id=5,
    )


def build_model(
    model_name: str = "answerdotai/ModernBERT-base",
    numeric_dim: int = 0,
    tiny_config: Any | None = None,
) -> torch.nn.Module:
    """Pointwise ranker: ModernBERT encoder + numeric MLP branch → scalar score.

    ``tiny_config`` overrides ``model_name`` (random init, offline tests).
    Any encoder config with ``hidden_size`` works (resolved via AutoModel).
    """
    _require_train_extra()
    import torch
    from transformers import AutoModel

    if tiny_config is not None:
        encoder = AutoModel.from_config(tiny_config)
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


class ChoiceHead(_ModuleBase):
    """Linear projection for candidate affinity scoring."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        _require_train_extra()
        import torch.nn as nn

        self.linear = nn.Linear(input_dim, 1)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        """Return shape (batch,) candidate affinity scores."""
        return self.linear(pooled).squeeze(-1)


class ScoreHead(_ModuleBase):
    """Linear projection to 4 rubric bins (effort: low=0, medium=1, high=2, max=3).

    Produces softmax probabilities and expected effort score sum(k * p_k).
    """

    def __init__(self, input_dim: int, num_bins: int = 4) -> None:
        super().__init__()
        _require_train_extra()
        import torch.nn as nn

        self.num_bins = num_bins
        self.linear = nn.Linear(input_dim, num_bins)

    def forward(
        self, pooled: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (logits, softmax_probs, expected_score)."""
        import torch

        logits = self.linear(pooled)
        probs = torch.softmax(logits, dim=-1)
        k = torch.arange(self.num_bins, dtype=probs.dtype, device=probs.device)
        expected = torch.sum(probs * k, dim=-1)
        return logits, probs, expected


class NoulHead(_ModuleBase):
    """Linear projection to 1 logit -> sigmoid binary fast success viability P(success)."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        _require_train_extra()
        import torch.nn as nn

        self.linear = nn.Linear(input_dim, 1)

    def forward(self, pooled: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (logit, sigmoid_probability)."""
        import torch

        logit = self.linear(pooled).squeeze(-1)
        prob = torch.sigmoid(logit)
        return logit, prob


@dataclass
class SystemOneOutput:
    """Output container for SystemOneModernBERT multi-task forward pass."""

    choice_logits: torch.Tensor
    score_logits: torch.Tensor
    score_probs: torch.Tensor
    expected_score: torch.Tensor
    noul_logits: torch.Tensor
    noul_prob: torch.Tensor

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


class SystemOneModernBERT(_ModuleBase):
    """Multi-task head over ModernBERT pooled [CLS] embedding.

    Heads:
      - choice_head: linear projection for candidate affinity scoring.
      - score_head: linear projection to 4 rubric bins -> softmax & expected score.
      - noul_head: linear projection to 1 logit -> sigmoid binary fast success viability.
    """

    def __init__(
        self,
        encoder: Any,
        hidden_size: int,
        numeric_dim: int = 0,
    ) -> None:
        super().__init__()
        _require_train_extra()
        self.encoder = encoder
        self.hidden_size = hidden_size
        self.numeric_dim = numeric_dim
        mlp_input = hidden_size + numeric_dim
        self.choice_head = ChoiceHead(mlp_input)
        self.score_head = ScoreHead(mlp_input, num_bins=4)
        self.noul_head = NoulHead(mlp_input)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        numeric_features: torch.Tensor | None = None,
    ) -> SystemOneOutput:
        """Evaluate pooled [CLS] representation through all System One heads."""
        import torch

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        if numeric_features is not None:
            pooled = torch.cat([pooled, numeric_features], dim=-1)
        choice_logits = self.choice_head(pooled)
        score_logits, score_probs, expected_score = self.score_head(pooled)
        noul_logits, noul_prob = self.noul_head(pooled)
        return SystemOneOutput(
            choice_logits=choice_logits,
            score_logits=score_logits,
            score_probs=score_probs,
            expected_score=expected_score,
            noul_logits=noul_logits,
            noul_prob=noul_prob,
        )

    def save_pretrained(self, out_dir: str, **kwargs: Any) -> None:
        """Save encoder in HF format + multi-head weights in system_one_heads.pt."""
        from pathlib import Path

        import torch

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(out, **kwargs)
        torch.save(
            {
                "choice_head": self.choice_head.state_dict(),
                "score_head": self.score_head.state_dict(),
                "noul_head": self.noul_head.state_dict(),
                "numeric_dim": self.numeric_dim,
                "hidden_size": self.hidden_size,
            },
            out / "system_one_heads.pt",
        )


def build_system_one_model(
    model_name: str = "answerdotai/ModernBERT-base",
    numeric_dim: int = 0,
    tiny_config: Any | None = None,
) -> SystemOneModernBERT:
    """Build a SystemOneModernBERT model with Choice, Score, and Noul heads.

    ``tiny_config`` overrides ``model_name`` (random init, offline tests).
    """
    _require_train_extra()
    from transformers import AutoModel

    if tiny_config is not None:
        encoder = AutoModel.from_config(tiny_config)
        hidden = tiny_config.hidden_size
    else:
        encoder = AutoModel.from_pretrained(model_name)
        hidden = encoder.config.hidden_size

    return SystemOneModernBERT(
        encoder=encoder,
        hidden_size=hidden,
        numeric_dim=numeric_dim,
    )


def load_ranker(path: str, numeric_dim: int | None = None) -> tuple[Any, Any]:
    """Rebuild the pointwise ranker wrapper from a checkpoint directory.

    Checkpoints store the encoder in HF format at the dir root plus the MLP
    head weights in ``head.pt``. ``numeric_dim`` is inferred from the head's
    first-layer input width when not given. Returns (ranker, tokenizer).
    """
    _require_train_extra()
    from pathlib import Path

    import torch
    from transformers import AutoModel, AutoTokenizer

    path = Path(path)
    encoder = AutoModel.from_pretrained(path)
    head_sd = torch.load(path / "head.pt", weights_only=True)
    hidden = encoder.config.hidden_size
    if numeric_dim is None:
        numeric_dim = int(head_sd["0.weight"].shape[1]) - hidden

    class _Loaded(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = encoder
            mlp_input = hidden + numeric_dim
            self.head = torch.nn.Sequential(
                torch.nn.Linear(mlp_input, hidden),
                torch.nn.GELU(),
                torch.nn.Linear(hidden, 1),
            )
            self.head.load_state_dict(head_sd)

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            numeric_features: torch.Tensor | None = None,
        ) -> torch.Tensor:
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            pooled = outputs.last_hidden_state[:, 0]
            if numeric_features is not None:
                pooled = torch.cat([pooled, numeric_features], dim=-1)
            return self.head(pooled).squeeze(-1)

    return _Loaded(), AutoTokenizer.from_pretrained(path)


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
