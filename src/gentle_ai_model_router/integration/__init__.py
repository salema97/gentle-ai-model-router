"""Integration: Gentle AI runtime adapters and the /route API server.

Adapters (Phase 2+) write decisions to the config surfaces Gentle AI already
reads (never modifying the Gentle AI repo itself):
- OpenCode (strongest surface): ``opencode.json`` agent entries
  (``model`` + ``variant`` / v2 ``provider/model#variant``) or profile files
  under ``external-single-active`` strategy;
- Pi: ``.pi/gentle-ai/models.json`` entries for sdd-* agent keys
  ({model, thinking});
- Codex: carril profile TOMLs plus per-phase table inputs
  (spawn_agent requires fork_turns="none" for overrides);
- Claude: agent frontmatter model/effort (weakest surface: sync re-renders
  from persisted presets, so re-apply after sync).

The FastAPI server (Phase 5) exposes POST /route returning the chosen
(model, deployment, effort) plus provenance.
"""
