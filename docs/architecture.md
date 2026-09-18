# Target Architecture — Gentle AI Model Router

**Status**: DESIGN. Phase 0 skeleton only; nothing here is implemented yet.
Claims about Gentle AI internals are backed by
`docs/gentle-ai-integration-research.md` (evidence: file + line in the
reference clone). Items marked **[PENDING VERIFICATION]** depend on external
systems and must be re-validated before Phase 1 implementation.

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
    │  deberta_    │ │  policy.py    │  │  ranker.py       │
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
           │  integration/    │  {(model, deployment, effort)}
           │  server.py       │  + provenance (why this pick)
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

| Module | Responsibility | Phase 1+ deliverable |
|---|---|---|
| `collector/` | Pull + snapshot external priors (Artificial Analysis, LMArena); never call upstream without a cache | Snapshot tables in Postgres + raw JSON under `data/` |
| `registry/` | Schema + migrations + CRUD for models, deployments, capabilities, prices, scores, availability | Alembic/SQLModel schema; sqlite fallback via env |
| `dataset/` | Join telemetry × priors into training rows; dataset versioning | Versioned parquet exports keyed by registry snapshot id |
| `training/` | DeBERTa-based phase/context encoder + ranker; constrained policy | Fine-tune `microsoft/deberta-v3-base` **[PENDING VERIFICATION: exact checkpoint + GPU budget]**; fallback: GBM on tabular features first |
| `router/` | Serve ranked candidates, apply phase thresholds + escalation | Deterministic policy before learned ranker is trustworthy |
| `integration/` | Gentle AI runtime adapters (write opencode.json / models.json / codex tables), FastAPI `/route` server | Adapters per §6 of the research doc |
| `cli/` | `router collect|build|train|serve|apply` commands | Typer-based CLI |
| `tests/` | Unit + contract tests (config-surface writers against golden files) | — |

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
