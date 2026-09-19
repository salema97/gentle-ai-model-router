"""Logging setup for collectors and the CLI.

Structured JSON logging is enabled with ``ROUTER_LOG_JSON=1`` (one JSON object
per record, suitable for log aggregation). Every collect run must log source,
snapshot id, record count, duration, and errors.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import UTC, datetime
from typing import Any


class JsonFormatter(logging.Formatter):
    """Render log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key.startswith("_") or key in _RESERVED_ATTRS:
                continue
            try:
                json.dumps(value)
                payload[key] = value
            except (TypeError, ValueError):  # pragma: no cover - defensive
                payload[key] = str(value)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


_RESERVED_ATTRS = frozenset(
    {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "taskName",
    }
)


def setup_logging(level: str | None = None, force_json: bool | None = None) -> None:
    """Configure root logging for the router process (idempotent).

    Args:
        level: Log level; defaults to ``ROUTER_LOG_LEVEL`` then ``INFO``.
        force_json: Force JSON output; defaults to ``ROUTER_LOG_JSON=1``.
    """
    root = logging.getLogger()
    if root.handlers and getattr(root, "_router_configured", False):
        return
    root.handlers.clear()
    json_mode = force_json if force_json is not None else os.environ.get("ROUTER_LOG_JSON") == "1"
    log_level = (level or os.environ.get("ROUTER_LOG_LEVEL") or "INFO").upper()
    handler = logging.StreamHandler(sys.stderr)
    if json_mode:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
    root.addHandler(handler)
    root.setLevel(log_level)
    root._router_configured = True  # type: ignore[attr-defined]


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, ensuring default setup happened at least once."""
    setup_logging()
    return logging.getLogger(name)
