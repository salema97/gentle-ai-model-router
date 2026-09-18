"""Test suite for gentle-ai-model-router.

Phase 1+ will include:
- unit tests per module;
- contract tests for adapter writers against golden config files
  (opencode.json shape, Pi models.json shape, Codex TOML shape) derived from
  the verified Gentle AI reference repo;
- quota/caching tests for collectors (no uncached upstream call);
- policy tests asserting the quality-floor constraint can never be violated.
"""
