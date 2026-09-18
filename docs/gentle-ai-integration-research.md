# Gentle AI Integration Research — Phase-Aware Model+Effort Router

**Status**: Phase 0 research. Evidence collected from a read-only clone of
`Gentleman-Programming/gentle-ai` (branch `main`, commit `b626a1f`, shallow
depth 50). Every claim cites file paths and line numbers relative to the repo
root. Where a claim could not be verified, it is marked **NOT FOUND** or
**UNVERIFIED**.

**Reference repo layout used throughout**: `_refs/gentle-ai/` is abbreviated as
`REF`.

---

## 1. Architecture overview of Gentle AI

Gentle AI is a Go CLI (module `github.com/gentleman-programming/gentle-ai/v3`)
that installs and synchronizes a "gentleman" harness into AI coding agents
(Claude Code, OpenCode, Codex, Pi, Cursor, Windsurf, Gemini, Kimi, Kiro, Qwen,
Antigravity, Hermes, GGA).

- Entry point: `cmd/gentle-ai/main.go:13-20` → `internal/app/app.go` command
  dispatch. Subcommands include `install`, `sync`, `restore`, `doctor`,
  `uninstall`, `telemetry`, `review` (a large native bounded-review lifecycle),
  and `sdd-*` helpers (`sdd-status`, `sdd-continue`, `sdd-attempt`,
  `sdd-archive-compose`, `sdd-task-result`, `sdd-preflight-hook`)
  (`internal/app/app.go:84-142`, second dispatch block `:283-311`).
- A second binary `cmd/gentle-telemetry` exists for telemetry collection.
- Interactive model/profile configuration happens in a Bubbletea TUI under
  `internal/tui/` (e.g. `internal/tui/model_test.go:331-340` exercises
  `ProfileDraft.PhaseAssignments` with `{ProviderID, ModelID, Effort}`).
- Agent-facing assets (orchestrator prompts, sub-agent definitions, slash
  commands, OpenCode plugins) are embedded from `internal/assets/<agent>/`.
- User choices persist in a state file; the struct is `internal/state/state.go`.
  Model-relevant fields (`internal/state/state.go:60-99`):
  - `ClaudeModelAssignments`, `ClaudePhaseAssignments` (model+effort),
    `KiroModelAssignments`, `CodexModelAssignments` (per-phase effort),
    `CodexOrchestratorAssignment`, `CodexCarrilModelAssignments`,
    `CodexPhaseModelAssignments`, and `ModelAssignments` (OpenCode
    provider/model pairs).

## 2. SDD orchestrator: where phases are defined and launched

### 2.1 Canonical phase list

The ordered SDD phase sub-agent names are defined once in
`internal/opencode/models.go:89-103` (`SDDPhases()`):

```
sdd-init, sdd-explore, sdd-research, sdd-propose, sdd-spec,
sdd-design, sdd-tasks, sdd-apply, sdd-verify, sdd-archive, sdd-onboard
```

(11 phases, not 7: the user's scope named 7; `sdd-init`, `sdd-research`,
`sdd-archive`, `sdd-onboard` also exist.) Additional configurable agent
families: Judgment Day `jd-judge-a`, `jd-judge-b`, `jd-fix-agent`
(`models.go:109-115`) and native review lenses + refuter/validator
(`models.go:117-136`). `ConfigurableAgentPhases()` unions all of them
(`models.go:143-148`).

### 2.2 Per-runtime launch mechanics (each verified)

**OpenCode** — markdown-driven orchestration:
- Slash commands in `internal/assets/opencode/commands/sdd-*.md` carry
  frontmatter `agent: gentle-orchestrator` and instruct the orchestrator to
  launch the hidden `sdd-<phase>` sub-agent after gates pass
  (`commands/sdd-explore.md:1-40`).
- The orchestrator prompt `internal/assets/opencode/sdd-orchestrator.md` reads
  model assignments **from `opencode.json` at session start**
  (`sdd-orchestrator.md:308-319`): `agent.gentle-orchestrator.model` and
  `agent.sdd-<phase>.model` are "authoritative".
- Sub-agent entries come from the overlay
  `internal/assets/opencode/sdd-overlay-multi.json` (keys: `gentle-orchestrator`,
  `general`, `explore`, `sdd-init` … `sdd-onboard`, `jd-*`, `review-*`, each
  with `__managed_by: gentle-ai/sdd`, `mode`, `prompt` file reference).
- The same file list is documented in `docs/opencode-profiles.md:188-190`:
  each named profile generates 11 agent entries (1 orchestrator + 10 sub-agents;
  note: this doc predates `sdd-research`, so "10" vs the current 11 phases in
  `SDDPhases()` is a minor doc drift).

**Claude Code** — frontmatter-driven:
- Phase agents are `internal/assets/claude/agents/sdd-*.md` with YAML frontmatter
  `model: {{CLAUDE_MODEL}}` and `{{CLAUDE_EFFORT_FRONTMATTER}}`
  (`agents/sdd-explore.md:1-8`, `agents/sdd-apply.md:1-8`). The orchestrator
  (`assets/claude/sdd-orchestrator.md`, `sdd-orchestrator-workflow.md`) launches
  them as Task sub-agents; each phase agent reads its skill from
  `~/.claude/skills/sdd-<phase>/SKILL.md`.

**Codex** — argv-driven:
- `internal/assets/codex/sdd-orchestrator.md:165-181` instructs
  `spawn_agent(task_name="sdd_design", message=<prompt>, model="<assigned-model>",
  reasoning_effort="<assigned-effort>", fork_turns="none")`.
  Phase → `task_name` uses underscores; `fork_turns: "none"` is **required** or
  the model/effort override is silently rejected (`:277`, echoed in
  `testdata/golden/sdd-codex-agentsmd.golden:276-277`).
- Whole-session tiers are TOML profiles `~/.codex/{sdd-strong,sdd-mid,sdd-cheap}.config.toml`
  with `model` + `model_reasoning_effort` (`internal/agents/codex/profiles.go:69-99`);
  these do NOT apply to spawned sub-agents (`sdd-orchestrator.md:190`).

**Pi** — external package:
- `gentle-pi` (separate repo, npm package) owns Pi runtime: SDD agents
  `.pi/agents/sdd-*.md`, chains `.pi/chains/sdd-*.chain.md` (`docs/pi.md:152-163`).
- Per-phase model assignment is gentle-pi's `/gentle:models` modal; saved to
  `.pi/gentle-ai/models.json`, applied to `.pi/agents/*.md` and `.pi/settings.json`
  (`docs/pi.md:119-150`).
- The Go side only **reads** gentle-pi's saved assignments, and only for native
  review roles (`review-refuter`, `review-validator`), never writes them
  (`internal/agents/pi/review_routing.go:15-17`, `docs/pi-provider-routing.md:1-21`).

### 2.3 Where a model/effort decision could be injected per phase

All launch paths resolve the assignment **statically from config at session/sync
time**, never at launch time per task. The resolution surfaces are:

| Runtime | Assignment surface | Writer in repo |
|---|---|---|
| OpenCode | `opencode.json` `agent.<name>.model` + `.variant`; v2 `agents.<name>.model = "provider/model#variant"` | `internal/components/sdd/inject.go:3439-3472`, `profiles.go:191,241-253,334` |
| Claude | agent frontmatter `model:`/`effort:` rendered from presets | `internal/model/claude_model.go:132-210` (presets), render via `internal/components/sdd/inject.go` |
| Codex | `~/.codex/*.config.toml` + per-phase table in orchestrator markdown | `internal/agents/codex/profiles.go`, `internal/components/engram/inject.go:475-488` |
| Pi | `.pi/gentle-ai/models.json` (owned by gentle-pi) | external; Go reads only |

An external router can therefore inject decisions by **writing the same config
artifacts** (out-of-process) without modifying Gentle AI at all — the configs
are the documented integration surface. In-process injection would hook the
preset resolution (`internal/model/*_model.go`) before inject/sync.

## 3. How Gentle AI configures models today

- **Presets** (static Go tables, not learned):
  - Claude: `ClaudeModelPresetBalanced/Performance/Economy/Diversity`,
    per-phase alias map, e.g. `sdd-explore: sonnet` …
    (`internal/model/claude_model.go:132-210`; test evidence
    `claude_model_test.go:104-114`).
  - Codex: preset matrix `low-cost | recommended | powerful` mapping 3
    carriles (`sdd-strong/sdd-mid/sdd-cheap`) → model + effort
    (`internal/model/codex_model.go:166-193`); efforts
    `low|medium|high|xhigh` (`codex_model.go:136-141`). Phase→carril grouping
    ("Sol reasons, Terra writes, Luna transcribes") at `codex_model.go:287-298`:
    strong = explore/propose/design/verify/judges; mid = apply/fix;
    cheap = spec/tasks/archive/onboard.
  - OpenCode: `ModelAssignments` map + generated profiles
    (`docs/opencode-profiles.md`).
- **Persistence**: `internal/state/state.go:60-99` (fields listed in §1).
- **TUI**: model picker with effort levels filtered by cached variants
  (`internal/tui/model_test.go:331-488`: invalid known efforts are cleared,
  efforts kept when variant data is unknown).
- **Profiles**: `docs/opencode-profiles.md:45-107` — generated multi-profile
  mode (`sdd-orchestrator-{name}` + suffixed phase agents, Tab-switched) vs
  `external-single-active` compatibility mode for external profile managers
  (`:109-145`). CLI: `gentle-ai sync --profile name:provider/model`,
  `--profile-phase name:phase:provider/model` (`:83-107`).

## 4. OpenCode integration (verified against repo)

- Settings path: `~/.config/opencode/opencode.json` (XDG-aware)
  (`internal/opencode/models.go:9-28`).
- Layered config resolution: global → ancestor dirs up to repo root → project;
  JSONC over JSON; `OPENCODE_CONFIG_DIR` override
  (`internal/opencode/config.go:41-82,152-173`). The snapshot reader does NOT
  emulate remote config, substitutions, plugins, or `OPENCODE_CONFIG`
  (`config.go:32-36`) — a router must not rely on those either.
- **Assignment shape**: legacy `agent.<name> = {model: "provider/model", variant: "<effort>"}`
  (`config.go:248-281`); native v2 `agents.<name> = {model: "provider/model#variant"}`
  parsed by `internal/model/model_reference.go:8-23` (`SplitModelSpec` accepts
  `/` or `:` separators; `#` separates the variant). Core type:
  `internal/model/model_assignment.go:7-11` — `{ProviderID, ModelID, Effort}`,
  `Effort == ""` means provider default.
- **model-variants plugin**: CONFIRMED. `internal/assets/opencode/plugins/model-variants.ts`
  (and `plugins-v2/`) — on OpenCode startup it calls `client.provider.list()`,
  extracts `models.*.variants` keys, and writes
  `~/.gentle-ai/cache/model-variants.json` (atomic tmp+rename)
  (`model-variants.ts:33-70`). Documented in `docs/opencode-profiles.md:57-69`.
  The TUI consumes this cache for the effort picker; missing cache = skip
  effort step silently.
- **reasoningEffort in OpenCode config: CORRECTED.** No `reasoningEffort` key
  exists in OpenCode handling. Effort travels as the `variant` field (legacy),
  `#variant` selector suffix (v2), or `models.*.variants` catalog entries.
  `reasoning_effort`/`model_reasoning_effort` appear only in Codex contexts
  (§2.2) and telemetry parsing (`internal/telemetry/runtime_codex.go:306`).

## 5. Pi (gentle-pi) integration

- Model assignment ownership is explicitly external: "Pi model assignment
  belongs to `gentle-pi`, not the Gentle AI installer" (`docs/pi.md:119-121`).
- Config: `.pi/gentle-ai/models.json`; resolution order
  `GENTLE_PI_CONFIG_HOME` → `~/.pi/gentle-ai/models.json` → project
  `.pi/gentle-ai/models.json`; entries are a model string or `{model, thinking}`;
  applied as `--model`/`--thinking` argv pairs; "Go has no model registry"
  (`docs/pi-provider-routing.md:5-17`, `internal/agents/pi/review_routing.go:17-35`).
- Phase-differentiated guidance CONFIRMED (`docs/pi.md:129-135`): exploration /
  proposal / archive → fast+cheap; spec / design / tasks → strong reasoning;
  apply → strong coding with reliable tool use; verify/review → strong
  fresh-context model. Pi SDD agents are "shown first" in the modal
  (`docs/pi.md:127`).
- **thinkingLevelMap: NOT FOUND.** No symbol or config named `thinkingLevelMap`
  exists anywhere in the repo (full-text search). The allowed `--thinking`
  levels are exactly `off, minimal, low, medium, high, xhigh, max` — CONFIRMED
  twice: `internal/agents/pi/review_routing.go:75-79` (validation switch) and
  `docs/pi-provider-routing.md:11`. Unknown values fail closed
  (`review_routing.go:77-79`).

## 6. Minimal, safest integration point for an external router

**Recommendation: out-of-process router that writes the existing config
surfaces; zero modifications to Gentle AI in Phase 1.**

Evidence for feasibility:
1. OpenCode reads assignments from `opencode.json` at session start
   (`sdd-orchestrator.md:312-316`) and `gentle-ai sync` supports
   `--sdd-profile-strategy external-single-active` precisely so external tools
   own runtime profile activation without fighting the generated overlay
   (`docs/opencode-profiles.md:109-145`).
2. Codex takes model+effort as plain argv on `spawn_agent`
   (`codex/sdd-orchestrator.md:181`), driven by a markdown table that sync
   regenerates — an external writer can regenerate the same table inputs only
   via state, so Codex is the weakest out-of-process surface.
3. Pi already delegates assignment authority to an external package
   (`docs/pi.md:121`), and its config is plain JSON keyed by agent name.
4. Claude phase agents are files with two frontmatter keys — trivially writable,
   but `gentle-ai sync` re-renders them from persisted presets, so an external
   router must either update `internal/state` equivalent data or re-apply after
   sync.

**What must stay untouched**: the native review lifecycle (`internal/cli/review_*`),
transport admission/consent machinery, telemetry privacy scrubbing
(`docs/telemetry.md`), and the `__managed_by: gentle-ai/sdd` ownership markers
(`internal/opencode/config.go:309-318`) — external writers should use the
`external-single-active` strategy or separate profile names rather than
impersonating managed keys.

**Practical minimal point (phase → model+effort resolution)**: a resolver run
**before** `gentle-ai sync`/OpenCode session start that emits:
- OpenCode: `opencode.json` agent entries (or profile files) — strongest surface;
- Pi: `.pi/gentle-ai/models.json` entries for `sdd-*` keys;
- Claude/Codex: best handled by feeding choices through the TUI/state, or
  re-applying file edits post-sync (flagged as the riskiest path).

## 7. Files modified vs. new (if integrating in-process later)

New files (router side, this project): all under
`gentle-ai-model-router/` (see its `docs/architecture.md`).

If Gentle AI itself ever hosts the hook (NOT in scope now), candidate touch
points found by inspection:
- `internal/model/presets.go` / `internal/model/{claude,codex,kiro}_model.go`
  (preset tables) — modified.
- `internal/components/sdd/inject.go`, `profiles.go`, `opencode_v2.go` —
  modified (assignment emission).
- `internal/state/state.go` — new persisted fields for router overrides.
- `internal/tui/` model picker — modified (show router-suggested defaults).
- Compatibility risks: `sync` regenerates agents from persisted state and
  would overwrite external edits; layered-config diagnostics
  (`config.go:78-80`) warn that edits may not be effective; stale-effort
  clearing in the TUI (`model_test.go:426-488`) drops efforts it cannot
  validate against the variants cache — a router writing exotic effort values
  into `variant` would see them erased on the next picker pass; Codex
  `fork_turns: "none"` omission silently ignores overrides; Pi validation is
  deliberately stricter than gentle-pi normalization and fails closed
  (`pi-provider-routing.md:13-15`).

## 8. Phase → runtime detection metadata at launch time

A router never needs to guess the phase; deterministic identifiers exist:

| Runtime | Phase signal at launch | Evidence |
|---|---|---|
| OpenCode | Slash command name (`/sdd-explore` …) → hidden sub-agent key `sdd-<phase>`; named profiles suffix `-{profile}` | `commands/sdd-explore.md:1-6`, `docs/opencode-profiles.md:157,188-190` |
| Claude | Task `subagent_type` = `sdd-<phase>`; agent frontmatter `name:`/`model:`/`effort:` | `assets/claude/agents/sdd-*.md:1-8` |
| Codex | `spawn_agent(task_name="sdd_<phase>")` (underscore transport identifier, hyphenated canonical) | `codex/sdd-orchestrator.md:166,276` |
| Pi | Chain step names `.pi/chains/sdd-*.chain.md`; agent keys in `models.json` | `docs/pi.md:152-161` |

Contextual metadata available for a *learned* router beyond the phase label:
repo identity, task size heuristics the orchestrator already computes
(file-count rules `opencode/sdd-orchestrator.md:84-89`, review workload
forecast `:293`), and post-hoc telemetry: runtime token usage per
model/effort/phase is already parsed from transcripts
(`internal/telemetrycollector/metrics.go:24,202`, `docs/telemetry.md:355-380`,
`internal/telemetry/runtime_codex.go`) — the personalization signal this
project targets. Note the public telemetry is privacy-scrubbed and anonymous;
local telemetry collectors (`cmd/gentle-telemetry`, `internal/telemetry*`) are
the model for on-premise execution telemetry.

## 9. Verification of user claims

| Claim | Verdict | Evidence |
|---|---|---|
| sdd-* phase agents exist with per-phase model assignment | **CONFIRMED** (11 phases, not 7) | `internal/opencode/models.go:89-103`; Claude `claude_model.go:132-210`; Codex `codex_model.go:166-298`; OpenCode `opencode-profiles.md:188-190` |
| OpenCode profiles generated per phase with cached model-variants | **CONFIRMED** | `docs/opencode-profiles.md:11-12,57-69,188-190`; `internal/assets/opencode/plugins/model-variants.ts:33-70` |
| Pi distinguishes exploration/proposal/design/apply/verify assignments | **CONFIRMED** | `docs/pi.md:127-135` |
| thinkingLevelMap with levels off/minimal/low/medium/high/xhigh/max | **CORRECTED**: levels CONFIRMED exactly; `thinkingLevelMap` symbol **NOT FOUND** | `internal/agents/pi/review_routing.go:75-79`; `docs/pi-provider-routing.md:11`; full-text search empty |
| OpenCode custom variants with reasoningEffort | **CORRECTED**: variants CONFIRMED, but the key is `variant` / `#variant` selector, not `reasoningEffort` (that is Codex-only) | `internal/model/model_reference.go:14-23`; `internal/components/sdd/inject.go:3439-3472`; `internal/model/model_assignment.go:7-11` |

**Additional correction**: user scope lists 7 phases (explore, propose, spec,
design, tasks, apply, verify); the repo defines 11 SDD phases plus JD and
review agent families that also carry per-agent model config
(`models.go:89-148`).
