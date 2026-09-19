# Telemetry & Discovery Probe — Gentle AI Reference Repo

**Status**: Phase 1 research. Evidence from read-only clone of
`Gentleman-Programming/gentle-ai` (branch `main`, commit `b626a1f`). Every claim
cites `file:line`. Marks **NOT FOUND** where absent.

---

## 1. Telemetry: what Gentle AI captures today

There are **two distinct telemetry systems**:

### 1.1 Public anonymous telemetry (counters)

`internal/telemetry/event.go`, client `gentle-ai telemetry`, collector binary
`cmd/gentle-telemetry`. Only: install_id, version, OS/arch, installed
agents/components, rdd_enabled, and aggregate counters since last flush:
`syncs`, `sdd_phase_runs`, `reviews_approved`, `reviews_correction`,
`reviews_escalated` (`internal/telemetrycollector/event.go:75-98`). **No tokens,
no models, no latency.**

### 1.2 Runtime telemetry (per agent message) — the signal that matters

Each `RuntimeRow` (`internal/telemetry/runtime.go:35-52`) carries:

- `model {provider, id}` + `model_evidence` (`selected|response|unknown`)
  (`:31-37`). **Privacy-scrubbed**: ids not matching `runtimeModelIDPattern`
  become `{custom,custom}` (`runtime.go:201,220-251`); OpenCode-local private
  ids degrade to `{opencode, custom}`.
- `agent_kind` (`orchestrator|built_in|custom|unknown`) and `agent_class`
  (closed vocabulary: `sdd-init`…`sdd-onboard`, `jd-*`, `review-*`, `:18`).
- `selected_effort` / `effective_effort` — closed vocabulary
  `off|minimal|low|medium|high|xhigh|max|not_selected|unknown|custom|unavailable|unsupported`
  (`:19`). OpenCode always reports `unavailable`
  (`runtime_opencode.go:117`); Codex extracts it from the transcript
  (`runtime_codex.go:295-311`).
- Tokens: `input_tokens, output_tokens, cache_read_tokens,
  cache_creation_tokens, reasoning_tokens, total_tokens`, each as
  `{reported, unavailable, unsupported, sum}` (`:44-61`).
- `launches`, `responses` counts; `error_category`
  (`none|unknown|auth|output_length|aborted|api|rate_limit|server`, `:356`).
- `duration {kind: request|message, measured_count, sum_ms}` — **aggregate sum
  only, never per-response latency** (`:106-112`).

**Tool calls: NOT FOUND** in telemetry (tool_call exists only as a model
catalog capability, `internal/opencode/models.go:47`).
**Test results: NOT FOUND** in telemetry. `sdd-task-result` classifies a phase
result but only returns `{"status":"ok"}` or a handoff error; it persists
nothing (`internal/cli/sdd_task_result.go:67-71`). Verify reports are markdown
files, not structured data.

### 1.3 Persistence and phase/task linkage

- **Client persists nothing**: `runtime_state.go:1-4`, `runtime_flush.go:1-4`.
  The `telemetry-runtime.ts` plugin fires
  `gentle-ai telemetry runtime <host> --json` per event, payload via stdin,
  single best-effort POST to `/v1/runtime-events` (`runtime_send.go:53`,
  3s timeout, no retry, HTTPS required `:50`).
- **Collector** (separate self-hosted `gentle-telemetry` binary): SQLite at
  `/var/lib/gentle-telemetry/events.sqlite` by default
  (`cmd/gentle-telemetry/main.go:28`). Tables `runtime_deliveries(delivery_id,
  received_at, canonical_payload)` and `runtime_rows(delivery_id, ordinal,
  row_json)` (`runtime_storage.go:66-77`); can also run Prometheus metrics
  mode (`main.go:56-71`).
- **Linkage**: aggregation keys are `agent_class + agent_kind + model +
  effort` (`metrics.go:139-148`). **No session id, no task id** — hooks receive
  `session_id`/`transcript_path` but discard them (privacy,
  `runtime_claude_test.go:20`). `delivery_id` is a one-way hash of message id
  for dedupe only (`runtime_event.go:13-18`).

### 1.4 Task outcome schema

None in telemetry (only transport/API `error_category`). The native review
lifecycle has outcome vocabularies — finding outcomes
`corroborated|refuted|inconclusive|info` (`reviewtransaction/transaction.go:69-75`),
transaction states `reviewing|validating|correction_required|approved|escalated`
(`compact.go:21`) — persisted under
`<git-common-dir>/gentle-ai/review-transactions/v1|v2/<lineage>/`
(`store.go:163`). These are review outcomes; they do **not** link model →
result.

### 1.5 Conclusion: can the router consume the store as-is?

**Partially — a router-owned shim is mandatory.** The collector SQLite store is
readable read-only (`runtime_rows.row_json`) and its row shape is a good
reference, but for telemetry-driven learning it lacks:

| Needed field | Status in Gentle AI store |
|---|---|
| Session/task id, repo id | Absent by design (privacy-scrubbed) |
| Exact model id (incl. local provider ids) | Degraded to `custom` / `opencode:custom` |
| Deterministic phase→task→change linkage | Only `agent_class` |
| Per-response latency | Aggregate `sum_ms` only |
| Effective effort on OpenCode | Always `unavailable` |
| Tool calls, test results, diff stats, task outcome | NOT FOUND |
| Actual cost ($) | NOT FOUND (static catalog pricing only, `models.go:31-34`) |

The wire schema **rejects unknown fields** (`runtimeExactRows`,
`runtime.go:410-428`), so fields cannot be added without breaking the
contract. The router shim must be a parallel path: same hook mechanism
(`message.updated` / `SubagentStop` / `turn_context` or transcript JSONL
parsing) with its own store.

---

## 2. Discovery surfaces (for the local discovery collector)

### 2.1 OpenCode config

- Global default `~/.config/opencode/opencode.json`; XDG-aware via
  `$XDG_CONFIG_HOME` (`internal/opencode/models.go:9-28`).
- Layered resolution: project dirs up to repo root (where `.git` exists), then
  global; **JSONC beats JSON within each dir**; absolute `OPENCODE_CONFIG_DIR`
  prepends and displaces global (`config.go:41-82,152-173`). Explicit
  limitation: does not emulate remote config, substitutions, plugins, or
  `OPENCODE_CONFIG` (`config.go:32-36`).
- Legacy shape: `agent.<name> = {model: "provider/model", variant: "<effort>"}`
  (`config.go:248-281`; `variant` doubles as effort when no `#` in spec,
  `:275-277`).
- Native v2 shape: `agents.<name>.model = "provider/model#variant"` or object
  `{providerID, model, variant}` (`internal/model/model_reference.go:8-38`,
  `config_v2.go:142-166`). `sdd-orchestrator` is renamed to
  `gentle-orchestrator` (`config.go:253-254`).
- Catalog variants: `providers.<id>.models.<modelID>.variants = [{"id":"high"},…]`
  (`config_v2.go:93-105`); real fixtures in `config_v2_test.go:70`,
  `catalog_test.go:22,73-75`.
- **Correction**: no `reasoningEffort`/`model_reasoning_effort` key exists in
  OpenCode config; effort travels as `variant` / `#variant` suffix.
  `reasoning_effort` appears only in Codex transcripts.

### 2.2 model-variants plugin cache

- **V1**: `~/.gentle-ai/cache/model-variants.json`, written atomically
  (tmp+rename); shape `{ "<providerId>": { "<modelId>": ["variantKey", …] } }`
  (`plugins/model-variants.ts:33-70`).
- **V2**: one file per location at
  `~/.gentle-ai/cache/opencode-v2/<sha256(directory+workspaceID)>.json`, same
  array shape (`plugins-v2/model-variants.ts:8-33`).
- Consumer: TUI picker (`model_picker.go:420-433`); missing cache = the effort
  step is silently skipped. **Trap**: the picker deletes efforts it cannot
  validate against the variants cache (`:429-433`).

### 2.3 Pi

- Owner: gentle-pi (external npm package); Go reads it only for review roles
  (`docs/pi.md:119-121`, `review_routing.go:15-17`).
- Resolution: `GENTLE_PI_CONFIG_HOME/models.json` → `~/.pi/gentle-ai/models.json`
  → `<repo>/.pi/gentle-ai/models.json`; no merge (`review_routing.go:19-35`).
- Shape: map `<agentName> → "provider/model"` or
  `{"model": "…", "thinking": "off|minimal|low|medium|high|xhigh|max"}`
  (`review_routing.go:42-84`); unknown keys fail closed (`:80-82`). Applied as
  `--model`/`--thinking` argv. Phase assignments live as `sdd-<phase>` keys in
  the same map.

### 2.4 Gentle AI state file

- Path `~/.gentle-ai/state.json` (`state.go:15-16,167-170`).
- Assignment fields (`state.go:59-97`): `claude_model_assignments`,
  `claude_phase_assignments` (phase → `{model, effort}`), `kiro_model_assignments`,
  Codex fields (`codexModelAssignments`, `codexOrchestratorAssignment`,
  `codexCarrilModelAssignments`, `codexPhaseModelAssignments`),
  `model_assignments` (OpenCode: agent → `{provider_id, model_id, effort}`).

### 2.5 CLI dump of effective model config

**NOT FOUND.** Command dispatch (`app.go:82-148,282-315`) has no models
list/status/dump. Closest: `sdd-status` (SDD change state, not models),
`bench-model-picker` (bench fixture). Real catalog discovery happens by
**executing `opencode models --verbose` as a subprocess and parsing stdout**
(`catalog.go:53-64,128-194`).

---

## 3. Implications for the router

1. **Telemetry**: the collector store is insufficient as the primary learning
   source. The router needs its own instrumentation shim (same hooks /
   transcript parsing, own store). Gentle AI's row shape is reusable as a
   reference, and `agent_class` is the canonical phase vocabulary to align to.
2. **Outcome/quality**: the only existing outcome schema is the review
   lifecycle's (`corroborated|refuted|…`, `approved|correction_required|
   escalated`) under the git common dir. Readable as per-change quality signal,
   but it does not link model → result; the router must build that link by
   correlating phase/time with its own observations.
3. **Discovery collector** is 100% read-only viable over: OpenCode configs
   (layered, JSONC>JSON, `OPENCODE_CONFIG_DIR`), both model-variants caches,
   Pi `models.json` (3-level resolution), and `~/.gentle-ai/state.json`.
4. **Write path traps**: `gentle-ai sync` regenerates agents from `state.json`
   (overwrites external edits); the OpenCode TUI deletes efforts not in the
   variants cache. The router must write via documented surfaces (profile
   strategy `external-single-active`, or Pi `models.json`), never by
   impersonating `__managed_by: gentle-ai/sdd` keys.
