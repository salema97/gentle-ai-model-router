"""CLI: operator-facing commands.

Phase 1+ will expose (Typer):
- ``collect``  — run data-source collectors (snapshots mandatory);
- ``build``    — build a versioned dataset from registry + telemetry;
- ``train``    — fit the policy/ranker;
- ``serve``    — start the FastAPI /route server;
- ``apply``    — write router decisions via the Gentle AI adapters.

All commands read ``router.yaml`` and log with provenance enabled.
"""
