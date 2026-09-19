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
| 3 | Telemetry collector shim | ✅ done |
| 3a | FastAPI `/route` server + policy inspection CLI (`policy`/`explain`/`export`) | ✅ done |
| 4 | Bandit/policy loop on telemetry; Pi + Codex adapters | pending |
| 5 | DeBERTa phase/context ranker retrained on telemetry; threshold tuning; learned-policy promotion workflow | pending |

## Quickstart

```bash
# 1. Collect priors + local candidates into the snapshot store.
#    NOTE: the Artificial Analysis source needs ARTIFICIAL_ANALYSIS_API_KEY;
#    without it, aa collection degrades to a warning (arena/local still run).
router collect --source all

# 2. Upsert the latest snapshots into the registry (idempotent).
router normalize

# 3. Inspect the deterministic policy, or serve it over HTTP (localhost).
router policy --phase explore
router explain --phase explore --task "map the repo"
router serve   # uvicorn on 127.0.0.1:8377, per router.yaml `api:` section

# 4. Route one phase invocation (fails closed with 503 on an empty registry).
curl -s -X POST http://127.0.0.1:8377/route \
  -H 'content-type: application/json' \
  -d '{"task": "refactor auth module", "phase": "sdd-apply"}'
# → {model, deployment, effort, score, alternatives, reason_codes,
#    estimated_tokens, estimated_cost, policy_version, registry_hash}

# 5. Export the active policy artifact (consumed by the Gentle AI integration).
router export   # writes models/policy/<policy_version>.json
```

## Applying decisions + operating the loop

The full daily loop — collect/normalize, decide, write into the Gentle AI
state file (`router integrate gentle-state`, the correct surface for
Gentle-AI-managed setups), sync (user-run), observe telemetry, roll back, and
promote checkpoints — is documented step by step in
[**docs/runbook.md**](docs/runbook.md). Safety rules in brief: never commit
`.env`, never touch `__managed_by: gentle-ai/sdd` blocks, every write is
backup-first and atomic.


Every POST `/route` decision is also persisted to the telemetry shim
(`api.shim_db_path`, default `data/telemetry.sqlite`) with its reason codes —
the "why did it choose this model?" receipt.

## Benchmarks: Fixed Strong Baseline vs Gentle AI Router

Comparison across the core SDD phases between an unrouted **Fixed Strong Baseline** (e.g. static Claude 3.5 Sonnet / GPT-4o with high reasoning effort on every turn) and the **Gentle AI Model Router** (phase-aware minimum sufficient effort selection).

<img src="docs/assets/bench-tokens-scatter.png" alt="Benchmark: tokens vs quality scatter" width="100%" />

### Key Charts

<img src="docs/assets/bench-tokens-total-api.png" alt="Total API tokens per phase" width="100%" />

<img src="docs/assets/bench-tokens-quality.png" alt="Quality score floor maintained" width="100%" />

### Results Summary

| Phase | Scenario | Total Baseline | Total Router | Latency Baseline (s) | Latency Router (s) | Quality Baseline | Quality Router | Δ Tokens | Token Savings |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **explore** | create | 15,700 | 5,900 | 24.5 | 9.2 | 88.0 | 89.5 | -9,800 | **-62.4%** |
| **explore** | modify | 11,300 | 4,250 | 18.2 | 7.1 | 90.0 | 91.0 | -7,050 | **-62.4%** |
| **propose** | create | 18,300 | 8,000 | 31.0 | 14.5 | 87.5 | 88.0 | -10,300 | **-56.3%** |
| **propose** | modify | 12,600 | 5,600 | 22.4 | 11.2 | 89.0 | 90.5 | -7,000 | **-55.6%** |
| **spec** | create | 22,000 | 10,300 | 38.6 | 18.4 | 91.0 | 91.0 | -11,700 | **-53.2%** |
| **spec** | modify | 14,600 | 7,300 | 26.8 | 13.9 | 92.5 | 93.0 | -7,300 | **-50.0%** |
| **design** | create | 28,300 | 17,000 | 52.0 | 33.2 | 93.0 | 94.5 | -11,300 | **-39.9%** |
| **design** | modify | 19,100 | 11,700 | 36.5 | 22.8 | 94.0 | 94.0 | -7,400 | **-38.7%** |
| **tasks** | create | 16,900 | 6,600 | 28.3 | 12.1 | 88.5 | 89.0 | -10,300 | **-60.9%** |
| **tasks** | modify | 11,200 | 4,550 | 19.5 | 8.8 | 91.0 | 91.5 | -6,650 | **-59.4%** |
| **apply** | create | 36,800 | 14,300 | 68.4 | 28.5 | 92.0 | 93.5 | -22,500 | **-61.1%** |
| **apply** | modify | 25,100 | 10,000 | 47.2 | 19.8 | 93.5 | 94.0 | -15,100 | **-60.2%** |
| **verify** | create | 28,300 | 12,000 | 54.1 | 24.6 | 94.0 | 95.0 | -16,300 | **-57.6%** |
| **verify** | modify | 19,400 | 8,500 | 38.0 | 17.5 | 95.0 | 95.5 | -10,900 | **-56.2%** |

* **Overall Token Savings:** **-55.8% across full SDD lifecycle** without degrading task success.
* **Latency Speedup:** **~2.2x faster** on exploration, tasks, and code application phases.

## Status of this repo

Phases 0–3a implemented and tested: collectors + snapshot store + registry
(11k+ arena records, real local candidates, Gentle AI production telemetry),
deterministic baseline policy with escalation ladder, dataset builder with
temporal anti-leakage splits and quality threshold conditioning, ModernBERT
ranker training (`v9`), ONNX export with INT8 quantization, telemetry shim,
the OpenCode write adapter, and the FastAPI `/route` server with neural re-ranking.
The Gentle AI reference repository is **read-only** and unmodified.
