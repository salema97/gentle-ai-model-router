# Feature: Runtime hook plugins feeding the shim (OpenCode + Pi)

## Objective

Implement roadmap phase 4a: runtime hook plugins for OpenCode and Pi that
automatically capture execution telemetry (tokens, latency, tool calls, phase,
model, effort, outcome) and stream it directly into the router's telemetry shim,
closing the autonomous feedback loop without manual CSV/JSONL ingestion.

## Problem

Currently, the telemetry shim (`integration/telemetry_shim.py`) and reward/bandit
pipeline (`router/reward.py`, `router/bandit.py`) operate on data ingested via CLI
(`router shim ingest` / `router feedback`) or synthetic bootstrap seeds (`router
shim seed`). In real agent runs on OpenCode and Pi, execution events are not
captured at runtime because the hook plugins are pending design and the FastAPI
server (`api/server.py`) has no HTTP ingestion endpoint for live execution
payloads.

## Why

Without automated hook instrumentation:
1. Telemetry accumulation requires manual export or external logging scripts.
2. The bandit policy loop and ModernBERT retrain cannot adapt continuously to
   live developer workloads.
3. Hook plugins close the loop: Agent Runs → Hook Intercepts Event → HTTP POST /
   Local Spool → Shim Store → Bandit Rewards / Retrain Dataset.

## Scope

- **H1** — API Ingestion Endpoints: Add `POST /shim/execution` and `POST
  /shim/feedback` to `api/server.py` with wire schemas in `api/schemas.py`.
  Accepts execution records, validates fields, and records to `ShimStore`
  idempotently (fails closed on invalid schemas, 503 if shim not configured).
- **H2** — OpenCode Hook Plugin: Standalone, zero-dependency plugin
  (`plugins/opencode/router_telemetry.js` + `.ts` definition) subscribing to
  `message.updated` (incremental tokens, latency) and `SubagentStop` (phase
  boundary, tools called, outcome). Emits `ExecutionRecord` JSON payloads to
  the local router server (`http://127.0.0.1:8377/shim/execution`) with
  non-blocking error-safe dispatch and local spooling fallback.
- **H3** — Pi Hook Plugin: Zero-dependency hook (`plugins/pi/router_telemetry.js`
  + `.ts` definition) intercepting `turn_context` / review completion events to
  capture review phase assignments, effort, tokens, and decisions.
- **H4** — Plugin Management CLI: CLI commands `router integrate plugins install`
  and `router integrate plugins status` supporting `--opencode` and `--pi`,
  with backup-first atomic writes, `--dry-run`, and verification.
- **H5** — Verification Tests: Unit tests for the API ingestion endpoints,
  payload contract verification against the shim schema, spooling mechanics, and
  CLI plugin installation.
- **H6** — Documentation: Update `README.md` (Phase 4a ✅), `docs/architecture.md`,
  `docs/telemetry-shim.md`, and `docs/runbook.md`.

## Constraints

- **Zero-crash guarantee**: Hook errors must NEVER crash the agent host runtime
  (OpenCode or Pi). All network and file calls inside hooks are wrapped with
  safe timeouts and silent/logged error guards.
- **Zero external npm dependencies**: Plugins must run natively in Node.js /
  runtime host environments using standard APIs (`fetch`, `fs`, `path`).
- **Idempotency**: Every event carries a deterministic `execution_id` so duplicate
  network deliveries never double-count tokens or rewards.
- **Strict typing & validation**: FastAPI endpoints validate inputs via Pydantic;
  Python side enforces typed errors and schema compliance.
- **Tests & Quality**: `uv run ruff check` clean, `uv run pytest` 100% green.

## Tasks

- [x] **H1** — API Ingestion Endpoints (`POST /shim/execution` and `POST /shim/feedback`
  in `api/server.py` + schemas in `api/schemas.py`).
- [x] **H2** — OpenCode Telemetry Plugin (`plugins/opencode/router_telemetry.js` + `.ts`).
- [x] **H3** — Pi Telemetry Plugin (`plugins/pi/router_telemetry.js` + `.ts`).
- [x] **H4** — CLI Plugin Installer (`router integrate plugins install/status` in
  `cli/main.py` and `integration/plugin_installer.py`).
- [x] **H5** — Contract & Integration Tests (`tests/test_api_telemetry.py`,
  `tests/test_runtime_hook_contracts.py`, `tests/test_plugin_installer.py`).
- [x] **H6** — Documentation Sync (`README.md`, `docs/architecture.md`,
  `docs/telemetry-shim.md`, `docs/runbook.md`).

## Acceptance criteria

- `POST /shim/execution` correctly upserts rows into `telemetry.sqlite` and returns
  HTTP 200 with recorded status.
- Plugins emit payloads that strictly validate against `ExecutionRecord` schema.
- Hook failure does not throw uncaught errors when the router server is offline.
- `router integrate plugins install` writes plugins atomically with backup to the
  target plugin directories.
- Full test suite green (`uv run pytest`), lint clean (`uv run ruff check`).
