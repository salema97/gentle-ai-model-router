# Operational Runbook — gentle-ai-model-router

End-to-end daily operation of the phase-aware model+effort router, from data
collection to writing decisions back into Gentle AI. Companion to
`docs/architecture.md` (system design) and `docs/gentle-ai-integration-research.md`
(verified Gentle AI integration evidence).

**Golden rules (apply to every step):**

- NEVER commit `.env` (it holds `ARTIFICIAL_ANALYSIS_API_KEY`); it is
  gitignored — keep it that way.
- NEVER touch `__managed_by: gentle-ai/sdd` blocks in `opencode.json`; they are
  owned by `gentle-ai sync`. The router refuses to modify them by design.
- EVERY write makes a backup first (`<path>.router-backup-<ts>`) and is atomic
  (tmp file + rename). Verify the backup exists before trusting a write.

---

## 1. Daily: collect + normalize (priors → registry)

```bash
router collect --source all   # aa + arena + local discovery → snapshot store
router normalize              # upsert latest snapshots into the registry
```

- The `aa` source needs `ARTIFICIAL_ANALYSIS_API_KEY` in `.env`; without it,
  collection degrades to a warning and serves the cached snapshot if one exists
  (**AA quota note**: the collector refuses to burn quota when a fresh snapshot
  is already stored — pass `--force` only when you intentionally want a
  re-fetch).
- `router collect --source local` is fully read-only over your machine's
  configs; it is the candidate source for models you actually have access to.

## 2. Decide: inspect the policy before writing anything

```bash
router policy --phase explore          # per-phase recommended config + top-3
router explain --phase apply --task "refactor auth module"
router route --phase apply --json      # single decision as JSON
```

All three are read-only. `explain` shows the full candidate ladder with quality
estimates and cost — decide HERE, not at write time. The policy version and
reason codes are your audit trail for why a model was chosen.

## 3. Write: apply the decision to Gentle AI

**Two write surfaces — pick the right one:**

### 3a. `router integrate gentle-state` — PREFERRED (Gentle-AI-managed setups)

If your `opencode.json` agent blocks carry `__managed_by: gentle-ai/sdd`
(the norm after `gentle-ai install`), the state file is the only correct
surface, because `gentle-ai sync` **regenerates agent configs from the state
file** — direct `opencode.json` edits would be overwritten (and the opencode
adapter correctly refuses to touch managed blocks anyway).

```bash
router integrate gentle-state \
  --phase apply --model anthropic/claude-sonnet-4 --effort high \
  --verify
```

- Writes `model_assignments["sdd-apply"] = {provider_id, model_id, effort}`
  into `~/.gentle-ai/state.json` (override with `--state PATH`; use that in
  any test/sandbox context — never experiment against the real state file).
- `--verify` re-reads the file and confirms the assignment round-trips.
- Refuses (exit 2) on: missing file (use `--create` to initialize with
  `{model_assignments: {…}}` only), malformed JSON, or a pre-existing entry
  whose shape is not a recognizable model assignment (defensive: your data is
  never silently destroyed).
- Dry-run first when unsure: add `--dry-run` to see the unified diff.

### 3b. `router integrate opencode` — ONLY for unmanaged configs

If your config has NO `__managed_by: gentle-ai/sdd` blocks (Gentle AI not
managing this opencode install), write `agent["sdd-<phase>"].model/variant`
directly:

```bash
router integrate opencode --phase apply --model anthropic/claude-sonnet-4 --effort high
```

The adapter clamps effort DOWN to the closest variant supported by the
model-variants cache (never up) and refuses `__managed_by` blocks with exit 2.

## 4. Render: sync (user-run) + verification

The router NEVER runs sync itself. After a `gentle-state` write:

```bash
gentle-ai sync --sdd-profile-strategy external-single-active
```

- `external-single-active` is the profile strategy Gentle AI provides
  precisely so external tools own runtime profile activation without fighting
  the generated overlay (docs/opencode-profiles.md, cited in
  `docs/gentle-ai-integration-research.md` §6.1).
- **Warning**: sync regenerates opencode agent configs from the state file;
  hand-edited agent blocks in `opencode.json` will be overwritten.
- Verify after sync:

```bash
router integrate status               # opencode config view
router integrate gentle-state status  # state file view (both must agree)
router integrate pi status            # gentle-pi models.json view
router integrate codex status         # codex state-file assignments view
```

## 5. Observe: telemetry + cost-per-success

```bash
# ingest execution JSON-lines (from the shim hook) into the telemetry store
router shim ingest < executions.jsonl

# or ingest AND score outcomes in one pass (rubric -> task_success/quality_score)
router feedback < executions.jsonl

# refresh rewards from the store and print the bandit version
router bandit update

# per-(phase, model, effort) reward aggregates the bandit is exploiting
router bandit report [--phase sdd-apply]

# tokens_per_success is the primary promotion metric (lower is better);
# it is reported by `router evaluate` per split/router.
```

`/route` decisions are already bandit-consulted in-process: when scored
rewards exist for the phase, candidates inside the deterministic
threshold-meeting set are reordered by constrained UCB (`bandit:ucb` reason
code); with no rewards the decision is byte-identical to the deterministic
policy (`bandit:cold_start_fallback`).

Every `POST /route` decision is persisted with reason codes — the "why this
model?" receipt. Correlate these with outcomes before trusting any learned
checkpoint (labels from priors are bootstrap-quality, not ground truth).

## 6. Rollback + full deactivation

```bash
# undo the last write (restores the latest .router-backup-<ts>)
router integrate gentle-state rollback            # or: router integrate rollback

# full deactivation — return Gentle AI to normal behavior:
# 1) restore the state file from its pre-router backup (command above), or
#    remove the router-written keys from model_assignments manually;
# 2) re-run `gentle-ai sync` yourself so agent configs regenerate WITHOUT
#    the router's assignments;
# 3) the router leaves no other persistent state in Gentle AI — once the
#    assignments are gone and sync re-rendered, behavior is stock.
# 4) stale backups (*.router-backup-*) can be deleted once verified.
```

## 7. Promote a learned checkpoint (train → evaluate → promote)

```bash
router build_dataset --name v1 --train-end 2026-09-01
router train --dataset data/datasets/v1
router evaluate --dataset data/datasets/v1 --checkpoint models/modernbert-router/v1
router promote --candidate models/modernbert-router/v1 --dry-run   # decide first
router promote --candidate models/modernbert-router/v1
```

Promotion is NEVER automatic: the candidate must beat the active router on
`tokens_per_success` (primary, lower is better) within epsilon guardrails on
ranking metrics. A failed comparison exits 1 and keeps the current router.

---

## Quick reference: which command owns which file

| File | Writer | Reader |
|---|---|---|
| `~/.gentle-ai/state.json` (`model_assignments`) | `router integrate gentle-state` | `gentle-ai sync` |
| `~/.config/opencode/opencode.json` (unmanaged blocks only) | `router integrate opencode` | OpenCode session start |
| `~/.pi/gentle-ai/models.json` (or `GENTLE_PI_CONFIG_HOME`) | `router integrate pi` | gentle-pi review routing |
| `~/.gentle-ai/state.json` (`CodexModelAssignments`, `CodexPhaseModelAssignments`) | `router integrate codex` | `gentle-ai sync` → Codex TOML profiles |
| `__managed_by: gentle-ai/sdd` blocks | **nobody external — `gentle-ai sync` only** | OpenCode session start |
