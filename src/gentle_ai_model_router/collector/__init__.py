"""Collectors: pull and snapshot external data sources.

Phase 1 will implement:
- artificial_analysis: Artificial Analysis API v2 client with mandatory
  response snapshots and quota-aware caching.
- lmarena: Hugging Face ``lmarena-ai/leaderboard-dataset`` loader
  (configs: text, webdev, agent, search; splits: latest/full).
- telemetry_collector: local execution telemetry ingestion
  (tokens, latency, success per phase and candidate).
- local_discovery: read-only inspection of OpenCode/Pi/Codex/Claude
  configurations to enumerate available candidates.

Design constraints: never call an upstream source without writing a verbatim
snapshot first; never write to Gentle AI state or managed config markers.
"""
