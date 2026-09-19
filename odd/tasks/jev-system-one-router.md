# Feature: Jev System One Calibrated Decision Routing

## Objective

Incorporate TypeSafe Jev's "System One" non-autoregressive, calibrated decision
logic into `gentle-ai-model-router`. Transform the router from outputting
uncalibrated heuristic scores into outputting typed, statistically calibrated
decision primitives (`Choice`, `Score`, `Noul`) over ModernBERT.

## Problem

Currently, `RouteResponse` outputs heuristic scores (`score = prior + (ceiling - prior) * gain[effort]`)
or uncalibrated scalar outputs from ModernBERT. These scores:
1. Do not represent true probability distributions or confidence levels ($P \in [0, 1]$).
2. Cannot tell downstream agents (like Gentle AI's SDD orchestrator) whether the router is
   statistically confident or guessing between close alternatives.
3. Lack typed decision primitives:
   - No explicit effort rubric expectation (`Score`: $\mathbb{E}[\text{effort}] = \sum k \cdot p_k$).
   - No direct escalation gate probability (`Noul`: $P(\text{fast\_success})$).
   - No normalized probability distribution over candidate arms (`Choice`).

## Why

Following TypeSafe Jev's System One principles:
- **Non-autoregressive & fast**: Inferences run in a single forward pass ($\le 20$ ms on ModernBERT).
- **Calibrated probabilities**: Downstream executors can gate autonomous execution on confidence
  thresholds (e.g., execute autonomously if `confidence >= 0.85`, otherwise escalate or flag for human review).
- **Zero hallucinations**: The output space is algebraically constrained by the candidate arms and rubric levels.

## Scope

Authorized:
- Wire contract extension in `api/schemas.py` and `router/decision.py` adding `confidence`, `probabilities`, and `system_one` metadata while preserving full backward compatibility.
- Multi-head ModernBERT architecture in `training/model.py` supporting `ChoiceHead`, `ScoreHead`, and `NoulHead`.
- System One decision engine in `router/system_one.py` computing non-autoregressive calibrated evaluations.
- Integration in `api/server.py` `/route` endpoint.
- Calibration metrics (ECE, Brier score) in `router/calibrate.py`.
- Unit and integration tests covering the new primitives and contracts.
- Architectural documentation in `docs/system-one-decisions.md`.

NOT in scope:
- Replacing ModernBERT with causal decoder LLMs (Qwen/Llama).
- Altering the deterministic cold-start baseline fallback.

## Tasks

- [x] **J1** — Schemas & Wire Contracts: Add `SystemOneMeta`, `probabilities`, and `confidence` to `RouteResponse` and `Decision` in `api/schemas.py` and `router/decision.py`.
- [x] **J2** — Multi-Head ModernBERT: Implement `SystemOneModernBERT` with `ChoiceHead`, `ScoreHead` (ordered rubric expectation), and `NoulHead` (binary viability) in `training/model.py`.
- [x] **J3** — Calibration & Inference Engine: Build `router/system_one.py` to evaluate candidate arms and task context with temperature-scaled softmax and expected score computation.
- [x] **J4** — Fast API `/route` Integration: Connect the System One engine to `api/server.py`, populating calibrated fields with graceful fallback.
- [x] **J5** — Verification & Tests: Write unit/integration tests in `tests/test_system_one_router.py` verifying distribution sum to 1.0, score expectation, noul bounds, and backward compatibility.
- [x] **J6** — Documentation: Add `docs/system-one-decisions.md` explaining the System One calibrated decision architecture.

## Acceptance Criteria

- `RouteResponse` returns `confidence` in $[0, 1]$ and `probabilities` summing to $1.0 \pm 1e-4$.
- `system_one` metadata returns calibrated `effort_score` (rubric expectation) and `noul_fast_success_probability`.
- All existing 459 tests continue to pass without regressions.
- `uv run ruff check` and `uv run pytest` pass cleanly.
