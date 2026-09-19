# gentle-ai-model-router

Learned, phase-aware **model + effort router** for [Gentle AI](https://github.com/Gentleman-Programming/gentle-ai).
For each SDD phase (explore, propose, spec, design, tasks, apply, verify, …),
it picks the `(model, deployment, effort)` candidate that minimizes
`tokens_per_success` subject to per-phase quality floors.

## Why

Gentle AI assigns models per phase through static presets and manual pickers
(see `docs/gentle-ai-integration-research.md`). Quality and cost per model vary
by phase and by repository; a learned router replaces guesswork with
evidence: external benchmarks as priors, your own execution telemetry as the
personalization signal.

## Architecture (target)

```
 collectors (Artificial Analysis, LMArena, local configs, telemetry)
      │
      ▼
 registry (Postgres / SQLite) ──► dataset builder ──► DeBERTa ranker + policy
                                                          │
                                                     FastAPI /route
                                                          │
                                            Gentle AI adapters (OpenCode, Pi, Codex, Claude)
```

Full design: `docs/architecture.md`. Data plan: `docs/data-sources.md`.
Gentle AI integration evidence: `docs/gentle-ai-integration-research.md`.

## Roadmap

| Phase | Milestone | Status |
|---|---|---|
| **0** | Research + scaffold (this repo skeleton, integration research, data-source verification) | ✅ done |
| 1 | Collectors + registry + snapshots (Artificial Analysis, LMArena, local discovery); telemetry probe | ✅ done |
| 2 | Dataset builder + deterministic baseline policy; OpenCode adapter (strongest config surface) | ✅ done |
| 2b | DeBERTa ranker training + offline evaluation harness; escalation ladder (labels = bootstrap priors, NOT ground truth — see docs/training.md) | ✅ done |
| 3 | Telemetry collector shim + bandit/policy loop; Pi + Codex adapters | pending |
| 4 | DeBERTa phase/context ranker retrained on telemetry; threshold tuning | pending |
| 5 | FastAPI `/route` server, provenance reporting, eval harness | pending |

## Status of this repo

Phases 0–2 implemented and tested: collectors + snapshot store + registry
(11k+ arena records, real local candidates), deterministic baseline policy
with escalation ladder, dataset builder with temporal anti-leakage splits,
DeBERTa ranker training/evaluation harness (`[train]` extra), telemetry
shim, and the OpenCode write adapter. **All training labels today are
bootstrap priors from external benchmarks — not ground truth** (loud warning
in `manifest.json`, `docs/training.md`, `docs/evaluation.md`). The Gentle AI
reference repository is **read-only** and unmodified.
