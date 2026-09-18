# Data Source Plan

**Status**: Phase 0. Verified facts cite how they were checked (2026-09-18);
anything not directly verified is marked **UNVERIFIED**.

## 1. Artificial Analysis API v2 — external benchmark/pricing/latency PRIOR

- Endpoints: `https://artificialanalysis.ai/api/v2/language/models` and
  `.../language/models/free`. Both return **HTTP 401 without a key** (verified
  live via `curl`, 2026-09-18), which confirms the paths exist and require
  auth. Base host is `artificialanalysis.ai`, **not**
  `api.artificialanalysis.ai` (that host 404s — verified).
- **UNVERIFIED**: the documented free-tier limit of ~100 requests / 24 h.
  Treat as true and design for it: response **caching/snapshots are
  mandatory**, not optional. Snapshot every response verbatim (raw JSON under
  `data/snapshots/artificial-analysis/<date>/`) before normalization, and load
  registry rows from snapshots when the quota is exhausted or the key absent.
- Key via env var: `ARTIFICIAL_ANALYSIS_API_KEY` (see `.env.example`).
- Expected fields (to confirm against first real response in Phase 1):
  pricing per 1M input/output tokens, context window, latency percentiles,
  throughput, provider/deployment identity, intelligence index.
  **[PENDING VERIFICATION: exact v2 JSON schema.]**

## 2. LMArena leaderboard — category quality PRIOR

- Verified via Hugging Face API (2026-09-18): dataset `lmarena-ai/leaderboard-dataset`
  is public, non-gated, parquet, CC-BY-4.0, ~37.8k downloads, last modified
  2026-09-16.
- **Actual configs/splits** (verified from the dataset card — this corrects
  the original plan): configs =
  `agent, agent_bash_recovery_steps, agent_praise_complaint, agent_steerability,
  agent_tool_hallucination, document, document_style_control, image_edit,
  image_to_video, search, search_factuality, search_style_control, text,
  text_factuality, text_style_control, text_to_image, text_to_video,
  vision, vision_style_control, webdev, video_edit`;
  each with splits `latest` and `full`.
- **CORRECTED**: the plan listed "math" as a priority category — there is
  **no `math` config** in this dataset. Closest verified alternatives:
  `text` (overall), `search_factuality`, `agent*`. If math-specific signal is
  needed, consider a separate source (e.g. AIME/MATH benchmark repos) —
  **UNVERIFIED, deferred**.
- Router-relevant priority (verified to exist): `text`, `webdev`, `agent`,
  `search`. Phase mapping hypothesis (to validate with telemetry):
  explore↔text/search, spec/design↔text, apply↔webdev+agent,
  verify↔agent_tool_hallucination/search_factuality. **Hypothesis, not fact.**
- Also available: `lmarena-ai/arena-human-preference-100k` and
  `webdev-arena-preference-10k` (raw preference data; heavier; Phase 2
  candidate for pairwise training signal).

## 3. Local environment discovery — available candidates

Read-only inspection of what the user can actually run (drives the feasibility
filter):

- OpenCode: `~/.config/opencode/opencode.json` (and `.jsonc`, layered,
  `OPENCODE_CONFIG_DIR`) — providers, models, `capabilities.tools`, `variants`
  (verified: `internal/opencode/config.go`, `config_v2.go` in the reference
  repo). Variant cache: `~/.gentle-ai/cache/model-variants.json` written by
  the bundled model-variants plugin.
- Pi: `GENTLE_PI_CONFIG_HOME` or `~/.pi/gentle-ai/models.json` (verified
  resolution order in `internal/agents/pi/review_routing.go:17-35`).
- Codex: `~/.codex/{sdd-strong,sdd-mid,sdd-cheap}.config.toml`.
- Claude Code: `~/.claude/agents/sdd-*.md` frontmatter (`model:`, `effort:`).
- Gentle AI persisted state: model assignment maps (see research doc §1/§3) —
  **read-only**; never write Gentle AI state from the router in Phase 1.

## 4. Real execution telemetry — PERSONALIZATION signal

The differentiating dataset: per phase run, `(candidate, effort, tokens,
latency, success)` from actual Gentle AI usage.

- Prior art in the reference repo: runtime telemetry parses transcripts and
  extracts per-model/effort token usage (`internal/telemetrycollector/metrics.go`,
  `internal/telemetry/runtime_codex.go`, `docs/telemetry.md:355-380`); tokens
  split into input/cached/output/reasoning. **UNVERIFIED** whether a local,
  non-anonymous per-phase telemetry store exists in a consumable form — Phase 1
  must probe `cmd/gentle-telemetry` and the runtime collectors, else we ship a
  small local collector shim that wraps the same transcript formats.
- `success` label: phase accepted by the orchestrator / review verdict /
  human edit distance — **definition TBD in Phase 1**; start with orchestrator
  acceptance + verify outcome.
- Privacy: local-first storage (router's own DB); no anonymous upload path.
  Telemetry quota discipline: only (phase, candidate, effort, token counts,
  outcome) — no prompts, no diffs (mirrors the repo's scrubbing posture).

## Policy

- External benchmark data = **prior**; real telemetry = **personalization
  signal**. Phase thresholds (quality floors) are configured per phase in
  `router.yaml`, not learned in Phase 1.
