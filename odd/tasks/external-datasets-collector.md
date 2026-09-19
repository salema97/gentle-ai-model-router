# Feature: Public Ground-Truth Datasets Collector (RouterBench, RouteLLM, SWE-Traces)

## Objective

Incorporate public ground-truth datasets from the AI/SWE research ecosystem into
`gentle-ai-model-router` to eliminate reliance on synthetic bootstrap priors and
harden the System One calibrated decision engine with real empirical execution
evidence.

## Problem

Currently, `data/datasets/empirical-v1` relies almost entirely on bootstrap
priors derived from external benchmarks (Artificial Analysis, LMArena). While
valuable for cold-start ordering, these priors:
1. Do not contain per-task ground-truth execution correctness labels.
2. Lack real agent trajectories (tool calls, diffs, unit test pass/fail).
3. Do not capture pairwise model routing trade-offs across cost-performance
   Pareto frontiers.

## Why

Public datasets created specifically for LLM routing and SWE agents provide:
- **RouterBench (`withmartian/routerbench`)**: 30,000+ prompts evaluated across 11
  LLMs with exact cost, latency, and correctness labels.
- **RouteLLM (`lm-sys/RouteLLM`)**: Preference and threshold datasets for
  routing between lightweight and frontier models (directly calibrates `Noul` and `Choice`).
- **SWE-Traces & Aider Benchmarks**: Real coding trajectories with repo context,
  file diffs, token consumption, and test suite outcomes mapped to SDD phases.

## Scope

Authorized:
- **C1** — RouterBench Collector (`collector/routerbench.py`): Fetch, parse, map to SDD phases, and save to snapshot store (`source: routerbench`).
- **C2** — RouteLLM Collector (`collector/routellm.py`): Fetch, parse win/loss comparisons between economy and frontier models, save to snapshot store (`source: routellm`).
- **C3** — SWE-Traces Collector (`collector/swe_traces.py`): Fetch, parse software engineering trajectories and test outcomes, save to snapshot store (`source: swe-traces`).
- **C4** — CLI Integration: Add sources to `router collect` in `cli/main.py` and `router/config.py` with graceful offline fallback and `--source all` support.
- **C5** — Dataset Bridge & Schema: Ensure `dataset/builder.py` and `telemetry_bridge.py` consume the new snapshots with `ground_truth_traces` provenance.
- **C6** — Unit & Integration Tests: Add comprehensive offline tests in `tests/test_external_collectors.py` verifying parsing, phase mapping, and snapshot storage.
- **C7** — Documentation: Document external data sources in `docs/data-sources.md` and `README.md`.

## Constraints

- **Zero network flakiness in CI/tests**: All unit tests must run fully offline using synthetic/mock payloads without requiring Hugging Face or GitHub credentials.
- **Graceful degradation**: Live collection failures (e.g. rate limits, missing network) log warnings and fallback cleanly to existing snapshots.
- **Deterministic provenance**: Datasets built from these snapshots must record `label_provenance: "ground_truth_traces"`.
- `uv run ruff check` clean, `uv run pytest` 100% green.

## Tasks

- [x] **D1** — RouterBench Collector (`collector/routerbench.py`).
- [x] **D2** — RouteLLM Collector (`collector/routellm.py`).
- [x] **D3** — SWE-Traces Collector (`collector/swe_traces.py`).
- [x] **D4** — CLI & Config Integration (`cli/main.py` & `router/config.py`).
- [x] **D5** — Dataset Builder Integration (`dataset/builder.py`).
- [x] **D6** — Tests & Verification (`tests/test_external_collectors.py`).
- [x] **D7** — Documentation Update (`docs/data-sources.md`).
