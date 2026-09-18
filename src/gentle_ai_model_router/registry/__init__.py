"""Registry: normalized model/deployment/candidate storage.

Phase 1 will implement the relational schema (Postgres in production,
SQLite fallback for development) holding:
- models and deployments (candidate units are (model, deployment, effort));
- capabilities (e.g. tool-call support, required by SDD phases);
- pricing, context limits, latency percentiles;
- LMArena category scores and Artificial Analysis priors;
- availability per local environment (from collector.local_discovery).

All registry rows must be traceable to a snapshot id (provenance).
"""
