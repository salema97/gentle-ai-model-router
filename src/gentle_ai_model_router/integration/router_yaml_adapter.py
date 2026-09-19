"""router.yaml write adapter: apply threshold proposals explicitly.

The tuner (``router/threshold_tune.py``) is pure; THIS module is its only
write path, mirroring the calibrate.py split (read-only decision support vs
explicit apply). It edits exactly one key per phase —
``phases.<name>.threshold_quality`` — and nothing else in the file survives
modified (a pure YAML round trip).

Safety rules (fail closed, never silently degraded):

- Only proposals with kind ``upgrade``/``downgrade`` (i.e. sufficient
  evidence behind them) are applied; ``uphold`` and ``insufficient_evidence``
  proposals are reported as skipped and never touch the file.
- A phase absent from router.yaml is refused unless ``create=True``.
- Writes are backup-first (``<path>.router-backup-<ts>``) and atomic
  (tmp file + ``os.replace``), same convention as the other write adapters.
"""

from __future__ import annotations

import difflib
import os
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from gentle_ai_model_router.router.threshold_tune import ThresholdProposal

APPLYABLE_KINDS = ("upgrade", "downgrade")


class RouterYamlError(Exception):
    """Fatal router.yaml write failure (missing file, malformed YAML, bad shape)."""


@dataclass(frozen=True)
class ThresholdApplyResult:
    """Outcome of one apply run (provenance included for the CLI output)."""

    path: Path
    applied: dict[str, tuple[float, float]]  # phase -> (old threshold, new threshold)
    skipped: dict[str, str]  # phase -> human-readable skip reason
    diff: str
    backup_path: Path | None
    wrote: bool  # False in dry-run mode


def _load_router_yaml(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RouterYamlError(f"cannot read {path}: {exc}") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise RouterYamlError(f"malformed YAML in {path}: {exc}") from exc
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise RouterYamlError(f"{path} root is not a mapping (got {type(data).__name__})")
    return data


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.router-tmp-{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def apply_threshold_proposals(
    path: Path,
    proposals: list[ThresholdProposal],
    *,
    create: bool = False,
    dry_run: bool = False,
    backup: bool = True,
) -> ThresholdApplyResult:
    """Apply ``upgrade``/``downgrade`` proposals to ``phases.<name>.threshold_quality``.

    ``uphold``/``insufficient_evidence`` proposals are skipped with a reason.
    Phases missing from the file are refused unless ``create=True`` (they are
    then added as ``phases.<name>.threshold_quality: <proposed>``). The write
    is backup-first and atomic; ``dry_run=True`` computes the diff without
    touching the file. Raises :class:`RouterYamlError` when the file is
    missing or malformed.
    """
    path = Path(path)
    if not path.is_file():
        raise RouterYamlError(f"router.yaml does not exist: {path} (pass --config)")

    original = path.read_text(encoding="utf-8")
    data = _load_router_yaml(path)
    phases_section = data.get("phases")
    if phases_section is None:
        phases_section = {}
        data["phases"] = phases_section
    if not isinstance(phases_section, dict):
        raise RouterYamlError(
            f"{path}: 'phases' section is not a mapping (got {type(phases_section).__name__})"
        )

    applied: dict[str, tuple[float, float]] = {}
    skipped: dict[str, str] = {}
    for proposal in sorted(proposals, key=lambda p: p.phase):
        if proposal.kind not in APPLYABLE_KINDS:
            skipped[proposal.phase] = f"kind={proposal.kind}: {proposal.note}"
            continue
        name = proposal.phase.removeprefix("sdd-")
        entry = phases_section.get(name)
        if entry is None:
            if not create:
                skipped[proposal.phase] = (
                    "phase not present in router.yaml (pass --create to add it)"
                )
                continue
            phases_section[name] = {"threshold_quality": proposal.proposed_threshold}
            applied[name] = (proposal.current_threshold, proposal.proposed_threshold)
            continue
        if not isinstance(entry, dict):
            raise RouterYamlError(
                f"{path}: phases.{name} is not a mapping "
                f"(got {type(entry).__name__}); refusing to overwrite"
            )
        old = entry.get("threshold_quality", proposal.current_threshold)
        if not isinstance(old, (int, float)):
            raise RouterYamlError(
                f"{path}: phases.{name}.threshold_quality is not a number "
                f"(got {old!r}); refusing to overwrite"
            )
        entry["threshold_quality"] = proposal.proposed_threshold
        applied[name] = (float(old), proposal.proposed_threshold)

    new_text = yaml.safe_dump(data, sort_keys=False, default_flow_style=False)
    diff = "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            fromfile=str(path),
            tofile=str(path),
        )
    )
    if dry_run or new_text == original:
        return ThresholdApplyResult(
            path=path,
            applied=applied,
            skipped=skipped,
            diff=diff,
            backup_path=None,
            wrote=False,
        )

    backup_path: Path | None = None
    if backup and original:
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%f")
        backup_path = path.with_name(f"{path.name}.router-backup-{ts}")
        shutil.copy2(path, backup_path)
    _atomic_write(path, new_text)
    return ThresholdApplyResult(
        path=path,
        applied=applied,
        skipped=skipped,
        diff=diff,
        backup_path=backup_path,
        wrote=True,
    )
