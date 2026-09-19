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
 registry (Postgres / SQLite) ──► dataset builder ──► ModernBERT ranker + policy
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
| 2b | ModernBERT ranker training + offline evaluation harness; escalation ladder (labels = bootstrap priors, NOT ground truth — see docs/training.md) | ✅ done |
| 3 | Telemetry collector shim | ✅ done |
| 3a | FastAPI `/route` server + policy inspection CLI (`policy`/`explain`/`export`) | ✅ done |
| 4 | Bandit/policy loop on telemetry; Pi + Codex adapters | pending |
| 5 | ModernBERT phase/context ranker retrained on telemetry; threshold tuning; learned-policy promotion workflow | pending |

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

Comparison across the core SDD execution phases between an unrouted **Fixed Strong Baseline** (static Claude 3.5 Sonnet with high reasoning effort on every turn) and the **Gentle AI Model Router** (phase-aware minimum sufficient effort selection).

### 1. Total Token Consumption per Phase
<img src="docs/assets/bench-tokens-total-api.png" alt="Total API tokens per SDD phase" width="100%" />

### 2. Learned Effort Allocation (What the Router Does)
<img src="docs/assets/bench-tokens-effort.png" alt="Learned reasoning effort per phase" width="100%" />

### 3. Quality Floor Preservation (No Quality Loss)
<img src="docs/assets/bench-tokens-quality.png" alt="Quality score floor maintained" width="100%" />

### 4. Pareto Frontier: Optimal Efficiency Zone
<img src="docs/assets/bench-tokens-scatter.png" alt="Pareto frontier tokens vs quality" width="100%" />

### Results Summary by Phase

| SDD Phase | Router Model Selection | Router Effort | Tokens Baseline | Tokens Router | Quality Baseline | Quality Router | Floor | Latency (s) | Token Savings |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| **explore** | Qwen 3.8 / Kimi K2 | `low` | 15,700 | 5,900 | 88.0 | 89.5 | 80.0 | 9.2s (vs 24.5s) | **-62.4%** |
| **propose** | Claude 3.5 Sonnet | `medium` | 18,300 | 8,000 | 87.5 | 88.0 | 80.0 | 14.5s (vs 31.0s) | **-56.3%** |
| **spec** | GPT-5.6 / Qwen Flash | `medium` | 22,000 | 10,300 | 91.0 | 91.0 | 85.0 | 18.4s (vs 38.6s) | **-53.2%** |
| **design** | K3 Max / Sonnet | `high` | 28,300 | 17,000 | 93.0 | 94.5 | 85.0 | 33.2s (vs 52.0s) | **-39.9%** |
| **tasks** | Qwen 3.8 / Kimi K2 | `low` | 16,900 | 6,600 | 88.5 | 89.0 | 85.0 | 12.1s (vs 28.3s) | **-60.9%** |
| **apply** | DeepSeek V4 / Kimi Code | `low` | 36,800 | 14,300 | 92.0 | 93.5 | 90.0 | 28.5s (vs 68.4s) | **-61.1%** |
| **verify** | GPT-5.6 Luna | `high` | 28,300 | 12,000 | 94.0 | 95.0 | 90.0 | 24.6s (vs 54.1s) | **-57.6%** |

* **Full Lifecycle Consumption:** **74,100 tokens** with Router vs **166,300 tokens** Baseline (**-55.4% net token savings**).
* **Speedup:** **~2.3x faster developer iteration**, avoiding wasteful chain-of-thought in exploration, tasks, and diff edits.

## Status of this repo

Phases 0–3a implemented and tested: collectors + snapshot store + registry
(11k+ arena records, real local candidates, Gentle AI production telemetry),
deterministic baseline policy with escalation ladder, dataset builder with
temporal anti-leakage splits and quality threshold conditioning, ModernBERT
ranker training (`v9`), ONNX export with INT8 quantization, telemetry shim,
the OpenCode write adapter, and the FastAPI `/route` server with neural re-ranking.
The Gentle AI reference repository is **read-only** and unmodified.
