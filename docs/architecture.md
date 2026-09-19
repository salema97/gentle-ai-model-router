# Target Architecture — Gentle AI Model Router

**Status**: Phases 0–5 IMPLEMENTED (collectors, registry, dataset builder,
deterministic baseline policy + escalation ladder, ModernBERT ranker
training/eval harness `[train]`, ONNX export + quantization, telemetry shim
with HTTP ingestion endpoints `/shim/execution` and `/shim/feedback`,
runtime hook plugins for OpenCode and Pi, plugin installer CLI, OpenCode/Pi/Codex
adapters, bandit learning loop, threshold tuning, and learned ranker promotion).
Claims about Gentle AI internals are backed by
`docs/gentle-ai-integration-research.md` (evidence: file + line in the
reference clone).

## Objective

Learn, per SDD phase, which **(model, deployment, effort)** candidate minimizes
`tokens_per_success` — total tokens consumed until the phase outcome is
accepted — subject to a per-phase quality floor (e.g. verify/apply may not
regress success rate below a configurable threshold).

- **Candidate unit**: `(model, deployment, effort)`.
  - `model`: provider-qualified id, e.g. `anthropic/claude-sonnet-4`.
  - `deployment`: the serving endpoint/latency class (managed API, priority
    tier, local) — pricing/latency differ per deployment even for one model.
    **[PENDING VERIFICATION]** exact deployment taxonomy from Artificial
    Analysis v2 schema.
  - `effort`: reasoning level. Reconciled vocabulary across runtimes:
    OpenCode `variant` / `#variant`; Codex `reasoning_effort`
    (`low|medium|high|xhigh`); Claude frontmatter `effort`
    (`low|medium|high|xhigh|max`); Pi `--thinking`
    (`off|minimal|low|medium|high|xhigh|max`). Mapping table lives in
    `integration/`.

## Data flow

```
                      ┌────────────────────────────────────────────┐
                      │              DATA SOURCES                  │
                      │  Artificial Analysis API v2 (external      │
                      │  benchmark/pricing/latency = PRIOR)        │
                      │  LMArena leaderboard-dataset (HF,          │
                      │  category quality = PRIOR)                 │
                      │  Local configs (OpenCode/Pi/Codex/Claude   │
                      │  = available candidates)                   │
                      │  Execution telemetry (tokens, success,     │
                      │  per phase = PERSONALIZATION signal)       │
                      └───────┬───────────────────────┬────────────┘
                              │                       │
                    ┌─────────▼─────────┐   ┌─────────▼──────────┐
                    │     collector/    │   │  collector/        │
                    │  artificial_      │   │  telemetry_        │
                    │  analysis.py      │   │  collector.py      │
                    └─────────┬─────────┘   └─────────┬──────────┘
                              │                       │
                              ▼                       ▼
                    ┌──────────────────────────────────────────┐
                    │            registry/                     │
                    │  Normalized model registry (Postgres;    │
                    │  SQLite fallback for dev): capabilities, │
                    │  pricing, latency, arena scores per      │
                    │  category, availability per environment  │
                    └─────────┬────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────────────────────────────┐
                    │           dataset/                       │
                    │  Dataset builder: joins priors with      │
                    │  telemetry → (phase, features, candidate,│
                    │  tokens, success) training rows;         │
                    │  snapshot versioned artifacts            │
                    └─────────┬────────────────────────────────┘
                              │
              ┌───────────────┼────────────────────┐
              ▼               ▼                    ▼
    ┌──────────────┐ ┌───────────────┐  ┌──────────────────┐
    │  training/   │ │   training/   │  │     router/      │
    │  modernbert_ │ │  policy.py    │  │  ranker.py       │
    │  ranker.py   │ │  constrained  │  │  (loads ranker + │
    │ (phase→      │ │  bandit over  │  │  policy, applies │
    │  candidate   │ │  candidates,  │  │  per-phase       │
    │  scoring)    │ │  minimize     │  │  thresholds)     │
    └──────┬───────┘ │  tokens_per_  │  └────────┬─────────┘
           │         │  success s.t. │           │
           │         │  quality floor│           │
           │         └───────┬───────┘           │
           │                 │                   │
           └─────────┬───────┴───────────────────┘
                     ▼
            ┌──────────────────┐
            │  FastAPI /route  │  POST {phase, context} →
            │  api/server.py   │  {(model, deployment, effort)}
            │  (✅ Phase 3a)   │  + provenance (why this pick)
            └────────┬─────────┘
                    │
        ┌───────────┼───────────────┬──────────────┐
        ▼           ▼               ▼              ▼
  ┌──────────┐ ┌─────────┐  ┌────────────┐ ┌────────────┐
  │ OpenCode │ │   Pi    │  │   Codex    │ │   Claude   │
  │ adapter  │ │ adapter │  │  adapter   │ │  adapter   │
  │(opencode.│ │(.pi/     │  │(carriles + │ │(agent      │ │
  │json /    │ │gentle-ai/│  │per-phase   │ │frontmatter; │
  │profiles) │ │models.   │  │table,       │ │weakest      │
  │          │ │json)     │  │fork_turns)  │ │surface —    │
  │ strongest│ │          │  │            │ │sync         │
  │ surface  │ │          │  │            │ │overwrites   │
  └──────────┘ └─────────┘  └────────────┘ └────────────┘
```

## Components (scaffold only — no implementation in Phase 0)

| Module | Status | Responsibility | Delivered as |
|---|---|---|---|
| `collector/` | ✅ | Pull + snapshot external priors (Artificial Analysis, LMArena, local discovery); never call upstream without a cache | `collector/artificial_analysis.py`, `collector/arena.py`, `collector/local_discovery.py`, `collector/snapshots.py` |
| `registry/` | ✅ | Schema + CRUD for models, deployments, capabilities, prices, scores, availability | `registry/models.py`, `registry/db.py` (idempotent upserts), `registry/normalize.py`; SQLite fallback via env, Postgres-compatible |
| `registry/fingerprint.py` | ✅ | Cheap deterministic registry fingerprint (`count:max_id` per table, sha256) for API determinism + policy-cache invalidation | `registry_fingerprint(engine)` |
| `dataset/` | ✅ | Join priors into training rows (labels = bootstrap priors); temporal anti-leakage splits; dataset versioning | `dataset/builder.py`, versioned exports under `data/datasets/` |
| `training/` | ✅ | ModernBERT phase/context ranker training + offline evaluation + ONNX export with quantization + promotion workflow | `training/train.py`, `training/evaluate.py`, `training/onnx_export.py`, `training/promote.py` |
| `router/` | ✅ | Deterministic prior-weighted policy: min-sufficient-effort selection, per-phase quality floors, hard filters, escalation ladder; full ranking exposed for inspection | `router/policy.py` (`rank_candidates`/`select_candidate`), `router/decision.py`, `router/escalation.py`, `router/config.py` |
| `api/` | ✅ | FastAPI `/route` server (per-request decisions, provenance, shim decision logging, 422/503 semantics) + telemetry ingestion `/shim/execution` and `/shim/feedback` + cached `/policy` + `/health` | `api/server.py`, `api/schemas.py`; `router serve` launches uvicorn on localhost |
| `integration/` | ✅ | Gentle AI runtime adapters (OpenCode, Pi, Codex) + telemetry shim + runtime hook plugins (OpenCode, Pi) and plugin installer | `integration/opencode_adapter.py`, `integration/pi_adapter.py`, `integration/codex_adapter.py`, `integration/telemetry_shim.py`, `integration/plugin_installer.py`, `plugins/` |
| `cli/` | ✅ | `router collect|normalize|route|policy|explain|export|serve|build-dataset|train|evaluate|integrate|shim` | `cli/main.py` (Typer) |
| `tests/` | ✅ | Unit + CLI + API contract tests (99+ before Phase 3a; 119 + Phase 3a suite after) | `tests/` |

## Key design decisions (to validate in Phase 1)

1. **External benchmarks are priors; telemetry is the personalization
   signal.** Cold start per (phase, environment) = prior-weighted policy;
   telemetry shifts the bandit's posterior. Prevents the classic failure of
   arena-to-workload mismatch.
2. **Policy constrained, not free-form**: `argmin tokens_per_success` subject
   to `P(success | phase, candidate) ≥ threshold[phase]`, with an escalation
   ladder (cheap → strong) when the constraint is at risk. Matches the repo's
   own carril philosophy (`sdd-strong/sdd-mid/sdd-cheap`,
   `internal/model/codex_model.go:287-298`).
3. **Candidate feasibility filter first**: tool-call support is already a hard
   requirement in Gentle AI (`FilterModelsForSDD`, `internal/opencode/models.go:62-77`).
4. **Escalation policy** in `router.yaml`: max escalations per phase run,
   cooldown, and "never downgrade below floor" rules.
5. **Provenance on every pick** (ranker score, prior vs telemetry weight,
   constraint check) — the repo's RDD culture demands receipts, and debugging
   a wrong model pick without provenance is hopeless.

## Explicitly out of scope (Phase 0)

Collectors, registry code, training code, API server, and any modification to
the Gentle AI repository.
