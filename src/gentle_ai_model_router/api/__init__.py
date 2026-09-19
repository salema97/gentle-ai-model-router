"""FastAPI server surface for the deterministic routing policy.

Public entry point: :func:`create_app`. Local-first: binds localhost by
default, no auth (the CLI `serve` command is the intended launcher).
"""

from gentle_ai_model_router.api.server import create_app

__all__ = ["create_app"]
