# Telemetry Shim — router-owned instrumentation

**Status**: skeleton (Phase 1b). Store + ingestion live; hook plugins and the
per-phase scoring rubric are DESIGN/pending.

## Why not Gentle AI's telemetry store

`docs/telemetry-probe.md` §1.5 (with `file:line` evidence into the Gentle AI
reference clone, commit `b626a1f`) concludes the collector store can only be a
secondary reference, never the primary learning source:

| Needed for learning | Gentle AI store |
|---|---|
| Session/task/repo id | Absent by design (privacy-scrubbed, `runtime.go:201,220-251`) |
| Exact model id (incl. local providers) | Degraded to `custom` / `opencode:custom` |
| Deterministic phase→task linkage | Only `agent_class` |
| Per-response latency | Aggregate `sum_ms` only (`runtime.go:106-112`) |
| Effective effort on OpenCode | Always `unavailable` (`runtime_opencode.go:117`) |
| Tool calls, test results, task outcome | NOT FOUND |
| Actual cost | NOT FOUND |

The wire schema rejects unknown fields (`runtime.go:410-428`), so the shim is
a **parallel path**: same hook surfaces, own store. Nothing here impersonates
`__managed_by: gentle-ai/sdd` keys, and `gentle-ai sync` regeneration does not
affect this store.

## Schema (`data/telemetry.sqlite`)

- `decisions`: `decision_id` PK, `phase`, `selected` JSON, `alternatives` JSON,
  `reason_codes` JSON, `estimated_tokens`, `estimated_cost`, `policy_version`,
  `created_at`. One row per `router route` invocation (policy decisions).
- `executions`: `execution_id` UNIQUE (upsert key), `session_id`, `project_id`,
  `phase`, `task_type`, `model`, `deployment`, `effort`, `started_at`,
  `finished_at`, token counters (`input/output/reasoning/cached/total`),
  `latency_ms`, `tool_calls`, `tool_errors`, `tests_passed`, `tests_failed`,
  `task_success` (0/1, NULL = unknown), `quality_score`, `escalation_count`,
  `repo_features` JSON, `router_version`, `decision_id` FK → `decisions`.

## Ingestion today

- API: `ShimStore.record_execution(session, payload)` (idempotent upsert on
  `execution_id`), `record_decision(...)`, `tokens_per_success(session, phase)`.
- CLI: `router shim ingest < executions.jsonl` — one JSON object per line;
  malformed lines are skipped and counted, duplicates update.

## Planned hook attachment (DESIGN/pending)

- **OpenCode**: plugin subscribing to `message.updated` (per-response tokens,
  latency) and `SubagentStop` (phase boundary, outcome). Emits the same
  JSON-lines shape the shim ingests; transport later via local POST to the
  Phase-2 FastAPI server or direct append.
- **Pi**: `turn_context` hook for effort/model correlation on review roles.
- No Gentle AI plugin is written by this phase; the shim only documents the
  attachment points.

## Privacy stance

Local-only SQLite. No prompt content, no diffs, no transcripts — tokens,
counters, durations, and phase outcomes only.

## Outcome population (DESIGN/pending)

`task_success` / `quality_score` will be populated from phase-specific signals
(`PHASE_SIGNALS` in `integration/telemetry_shim.py` documents the expected
fields per phase): tests for `apply`/`verify` (tests_passed/failed, build
green), findings quality for `explore` (count + corroborated ratio), spec
coverage for `spec`, etc. The per-phase scoring rubric — exact formulas and
thresholds — is DESIGN/pending and must be specified before the dataset
builder consumes these columns.
