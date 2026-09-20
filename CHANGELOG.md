# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-09-19

### Added
- **TypeSafe Jev System One Calibrated Decision Routing (Phase 6)**:
  - Multi-task non-autoregressive decision heads over ModernBERT (`ChoiceHead`, `ScoreHead`, `NoulHead`).
  - Calibrated probability distributions over model candidates via temperature scaling.
  - Expected reasoning effort scoring ($\sum k \cdot p_k$) across discrete rubric bins.
  - Fast-success viability estimator (`Noul`) for escalation prevention.
  - Calibration metrics including Brier score and Expected Calibration Error (ECE).
  - FastAPI `/route` integration surfacing calibrated confidence and System One metadata.
- **Public Ground-Truth Dataset Ingestion**:
  - `RouterBenchCollector` ingesting RoutingCompendium benchmark tasks with cost/latency/win rates.
  - `RouteLLMCollector` ingesting pairwise model preference battles.
  - `SWETracesCollector` ingesting SWE-bench Lite trajectory execution traces.
  - Provenance tagging (`label_provenance='ground_truth_traces'`) linking empirical evidence to SDD phases.
- **ModernBERT Ranker Optimization & Blackwell Acceleration**:
  - Checkpoint `models/modernbert-router/v14` with 100.0% pairwise ranking accuracy across 4,791 preference pairs.
  - Native `bf16` precision support utilizing 5th-Gen Tensor Cores on NVIDIA Blackwell (RTX 50-series) and Ada GPUs.
  - Gradient accumulation (`--gradient-accumulation-steps`) and memory fraction bounding (`--max-vram-fraction`) capping peak VRAM usage to ~3.3 GB.
  - Unique candidate pre-tokenization cache (`precompute_pairwise_cache`) removing per-step CPU tokenization bottlenecks (~30.4 samples/sec throughput).
  - Single-pass inference latency of 19.68 ms on GPU and 59.84 ms on CPU.
- **Constrained Bandit Policy & Telemetry Feedback (Phase 4 & 4a)**:
  - Telemetry shim database recording model execution telemetry and business outcomes.
  - Constrained UCB bandit loop over quality-floor meeting arms.
  - Runtime hook plugins for OpenCode (`message.updated`/`SubagentStop`) and Pi (`turn_context`).
- **Visual Benchmarking Suite**:
  - Automated sketch-styled charts for total token consumption, effort distribution, quality retention, Pareto frontier, and execution time speedup.

### Changed
- **Dataset Builder Optimization**:
  - Replaced $O(N^2)$ Cartesian pair expansion with matched candidate filtering for empirical benchmark tasks.
  - Windowed search for preference pair generation ($O(K)$).
  - Chunked streaming Parquet writing (10,000 rows/batch) preventing out-of-memory errors on large datasets.
