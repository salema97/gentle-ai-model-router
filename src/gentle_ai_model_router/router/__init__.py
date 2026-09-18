"""Router: resolve a phase + context to a ranked candidate list.

Implements the core objective:

    argmin_{model, deployment, effort} tokens_per_success
    subject to P(success | phase, candidate) >= phase_thresholds[phase]

Phase 1+ will implement:
- feasibility filtering (tool-call support, availability, effort vocabulary
  per runtime);
- escalation ladder handling with cooldowns and per-run limits;
- provenance records for every decision.

Effort vocabulary differs per runtime and is reconciled here:
OpenCode ``variant``/``#variant``, Codex ``reasoning_effort``
(low|medium|high|xhigh), Claude frontmatter ``effort``
(low|medium|high|xhigh|max), Pi ``--thinking``
(off|minimal|low|medium|high|xhigh|max).
"""
