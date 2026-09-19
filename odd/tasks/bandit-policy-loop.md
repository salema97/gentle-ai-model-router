# Feature: Bandit policy loop on telemetry + Pi/Codex adapters

## Objective

Implement roadmap phase 4: a constrained contextual bandit over (model, deployment,
effort) arms that consumes telemetry-shim outcomes and minimizes tokens-per-success
subject to per-phase quality floors, plus Pi and Codex integration adapters. The
candidate arm universe is the 7 verified frontier models from the canonical benchmark
suite:

1. Google Gemini 3.8 Flash
2. Meta Muse Spark 1.3
3. GLM-5.3
4. Claude Fable 5.1 (Thinking)
5. Kimi K3
6. DeepSeek-V4.1-Flash
7. OpenAI GPT 5.6 Sol

## Problem

The deterministic policy (`router/policy.py`) selects minimum-sufficient effort from
static benchmark priors. Nothing consumes real execution outcomes, so the policy can
never learn. The telemetry shim schema exists (`integration/telemetry_shim.py`) with
`task_success` / `quality_score` columns, but populating and consuming them is
documented as DESIGN/pending. Pi and Codex have no write adapters (OpenCode and
gentle-state only).

## Why

Roadmap phase 4 is the only pending milestone. The bandit closes the loop:
decision → execution → reward → better decision. `training/__init__.py` already names
the design: "a constrained bandit over (model, deployment, effort) candidates that
minimizes tokens_per_success subject to per-phase quality floors".

## Scope

Authorized: new modules `integration/outcome.py`, `router/reward.py`, `router/bandit.py`,
`integration/pi_adapter.py`, Codex state-fields extension, dataset telemetry bridge,
CLI commands, tests, docs updates.

NOT in scope: direct `~/.codex/*.config.toml` writing (weakest surface, overwritten
by `gentle-ai sync` — defer and document as unsupported); live hook plugins
(message.updated / turn_context attachments); phase 5 retraining.

## Constraints

- Fail closed everywhere: typed errors (`PolicyError`, `AdapterError`), cold start
  falls back to deterministic `rank_candidates` ordering.
- Determinism: seeded exploration only, sha256 policy versioning, `bandit:*`
  reason_codes on every bandit-influenced decision.
- Backup-first atomic writes: `.<name>.router-backup-<UTC ts>` + tmp + `os.replace`.
- Never touch `__managed_by: gentle-ai/sdd` blocks; refuse unrecognized shapes.
- Arm space per phase is restricted to the 7 frontier models above when present in
  the registry; bandit exploration happens only inside the threshold-meeting set.
- Tests: one `tests/test_<module>.py` per new module; `uv run pytest` must pass;
  `uv run ruff check` clean.
- TDD mode: not configured in this project — ordinary functional checks (pytest
  per task), no strict RED/GREEN cycle.

## Delivery

Forecast: ~1,100 authored changed lines across 4 work units → over the ~400-line
heuristic per task but each task is one coherent module. Strategy: `ask-on-risk`
(default). Before the first push/PR decision, ask once for chain strategy if needed.
Commit per work unit on a feature branch.

## Tasks

- [x] **T1** — Outcome rubric: `integration/outcome.py` ✅ 939d62b (20 tests)
- [x] **T2** — Reward computation: `router/reward.py` ✅ 939d62b (9 tests)
- [x] **T3** — Bandit core: `router/bandit.py` ✅ 939d62b (12 tests; cold start
  byte-identical, deterministic UCB, floor demotion)
- [x] **T4** — Server + CLI integration ✅ 939d62b (6 CLI tests; route fails closed
  to prior ranking on telemetry errors; `router bandit report/update`,
  `router feedback`)
- [x] **T5** — Dataset bridge ✅ bd25615 + 928f382 (11 tests; PROVENANCE_TELEMETRY
  examples, actual token labels, schema v2 gated on emitted telemetry rows,
  v1 backward-load test; verified by independent verifier after high-risk assess)
- [x] **T6** — Pi adapter ✅ 427f48c (31 tests; models.json chain, object-form
  writes, atomic+backup+rollback, `router integrate pi`)
- [x] **T7** — Codex adapter ✅ ee2b534 (50 tests; state-file route, underscore
  keys, carriles, total effort map, `router integrate codex`)
- [ ] **T8** — Docs: update `README.md` roadmap status, `docs/runbook.md`,
  `docs/telemetry-shim.md` status lines for phase 4 components.

## Acceptance criteria

- `uv run pytest` green (new tests included), `uv run ruff check` clean.
- Bandit decisions carry `bandit:*` reason_codes and join back to rewards via
  `decision_id`.
- With an empty shim DB, the bandit path is byte-identical to today's deterministic
  behavior (cold-start fallback verified by test).
- Pi and Codex adapters pass the same write-safety properties as the OpenCode
  adapter (backup, rollback, atomic, refuse unknown shapes).

## Progress log

- 2026-09-19 — T7 done in ee2b534 (Codex adapter). Suite 348 passed; assess
  medium; spot check 50/50. Next: T5 dataset bridge.
- 2026-09-19 — T5 done in bd25615; assess high → independent verifier found
  no blockers, 1 moderate (version string stamped on v1-only manifests) fixed
  inline in 928f382 + v1 backward-load test added. Suite 359 passed; 1
  pre-existing failure (test_fresh_cache_short_circuits, httpx MockTransport
  API drift, fails on base too). Branch pushed to origin. Next: T8 docs.
