"""Integration: Gentle AI runtime adapters and the /route API server.

Adapters (Phase 2+) write decisions to the config surfaces Gentle AI already
reads (never modifying the Gentle AI repo itself):
- Gentle AI state file (PREFERRED for managed setups): model_assignments in
  ~/.gentle-ai/state.json ({provider_id, model_id, effort} per agent) —
  gentle-ai sync regenerates agent configs from it; the profile strategy
  external-single-active is the compatible mode for external tooling;
- OpenCode (unmanaged configs only): ``opencode.json`` agent entries
  (``model`` + ``variant`` / v2 ``provider/model#variant``) or profile files
  under ``external-single-active`` strategy; refuses ``__managed_by`` blocks;
- Pi: ``.pi/gentle-ai/models.json`` entries for sdd-* agent keys
  ({model, thinking});
- Codex: carril profile TOMLs plus per-phase table inputs
  (spawn_agent requires fork_turns="none" for overrides);
- Claude: agent frontmatter model/effort (weakest surface: sync re-renders
  from persisted presets, so re-apply after sync).

The FastAPI server (Phase 5) exposes POST /route returning the chosen
(model, deployment, effort) plus provenance.
"""
