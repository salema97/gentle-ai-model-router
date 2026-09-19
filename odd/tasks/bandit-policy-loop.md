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

- [ ] **T1** — Outcome rubric: `integration/outcome.py` scoring `PHASE_SIGNALS`
  (tests, tool_errors, escalation_count, latency) into `task_success` (0/1) and
  `quality_score` (0–100). Pure functions, per-phase rubric table.
- [ ] **T2** — Reward computation: `router/reward.py` joining `decisions` ↔
  `executions` on `decision_id`; reward = f(success, total_tokens, cost, latency);
  per-(phase, model, effort) aggregates; win-rate vs alternatives.
- [ ] **T3** — Bandit core: `router/bandit.py` constrained UCB-style bandit over
  (model, deployment, effort) arms within the threshold-meeting set; cold-start
  fallback to deterministic ranking; `bandit:*` reason_codes; versioned like
  `policy_version`.
- [ ] **T4** — Server + CLI integration: `api/server.py` route consults bandit after
  `rank_candidates`; new `router bandit update|report` CLI; `router feedback`
  outcome-ingest plumbing.
- [ ] **T5** — Dataset bridge: `dataset/builder.py` reads shim executions and emits
  `label_provenance=PROVENANCE_TELEMETRY` examples (schema version bump if actual
  cost/tokens/latency fields are added).
- [ ] **T6** — Pi adapter: `integration/pi_adapter.py` (pure JSON round trip),
  `models.json` resolution chain, `sdd-<phase>` → `{model, thinking}`, atomic
  writes + backup + rollback + read_assignments; CLI `router integrate pi`.
- [ ] **T7** — Codex adapter (state-file route): extend gentle-state-style writing
  to `CodexModelAssignments` / `CodexPhaseModelAssignments` /
  `CodexCarrilModelAssignments`; effort-subset mapping (low|medium|high|xhigh);
  CLI `router integrate codex`; document direct-TOML route as unsupported.
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

- 2026-09-19 — Feature document created; scope mapped via exploration agent
  (policy/telemetry/adapter/dataset surface + gap analysis). Next: T1+T2+T3+T4
  (bandit core, one writer), then T6, T7, T5, T8.
