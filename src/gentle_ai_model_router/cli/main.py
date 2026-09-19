"""``router`` CLI: collect, normalize, snapshots, registry.

Exit codes: 0 = success (including graceful degradation with warnings),
1 = hard failure, 2 = usage error (typer default).
"""

from __future__ import annotations

import json

import httpx
import typer
from rich.console import Console
from rich.table import Table

from gentle_ai_model_router.collector.arena import LMArenaCollector
from gentle_ai_model_router.collector.artificial_analysis import (
    ArtificialAnalysisCollector,
    QuotaExceededError,
)
from gentle_ai_model_router.collector.local_discovery import (
    DiscoveryResult,
    collect_local_candidates,
)
from gentle_ai_model_router.collector.logging_conf import setup_logging
from gentle_ai_model_router.collector.snapshots import SnapshotRecord, SnapshotStore
from gentle_ai_model_router.registry import db as registry_db
from gentle_ai_model_router.router.config import RouterConfig, load_config

app = typer.Typer(
    name="router",
    help="Learned phase-aware model+effort router for Gentle AI SDD phases.",
    no_args_is_help=True,
)
snapshots_app = typer.Typer(help="Snapshot inspection.", no_args_is_help=True)
registry_app = typer.Typer(help="Registry inspection.", no_args_is_help=True)
app.add_typer(snapshots_app, name="snapshots")
app.add_typer(registry_app, name="registry")

console = Console()
err_console = Console(stderr=True)


def _load_ctx(
    config_path: str | None, data_dir: str | None
) -> tuple[RouterConfig, SnapshotStore]:
    config = load_config(config_path=config_path, data_dir=data_dir)
    setup_logging(config.logging.level)
    return config, SnapshotStore(config.snapshot_root)


def _print_record(record: SnapshotRecord, served_from_cache: bool = False) -> None:
    table = Table(title=f"snapshot {record.snapshot_id}")
    table.add_row("source", record.source)
    table.add_row("snapshot_id", record.snapshot_id)
    table.add_row("fetched_at", record.fetched_at.isoformat())
    table.add_row("path", str(record.path))
    table.add_row("record_count", str(record.record_count))
    table.add_row("errors", json.dumps(record.errors))
    if served_from_cache:
        table.add_row("served_from_cache", "true")
    console.print(table)


def _collect_aa(
    config: RouterConfig, store: SnapshotStore, force: bool, max_age_hours: int | None
) -> SnapshotRecord:
    collector = ArtificialAnalysisCollector(config.data_sources.artificial_analysis, store)
    try:
        return collector.collect(force=force, max_age_hours=max_age_hours)
    except QuotaExceededError as exc:
        err_console.print(f"[yellow]warning: {exc}[/yellow]")
        cached = store.latest_record("artificial-analysis")
        if cached is not None:
            err_console.print(f"[yellow]serving stale cache: {cached.snapshot_id}[/yellow]")
            return cached
        raise typer.Exit(code=1) from exc
    except httpx.HTTPError as exc:
        err_console.print(f"[yellow]warning: AA upstream failure: {exc}[/yellow]")
        cached = store.latest_record("artificial-analysis")
        if cached is not None:
            err_console.print(f"[yellow]serving stale cache: {cached.snapshot_id}[/yellow]")
            return cached
        err_console.print("[red]error: no cached snapshot available[/red]")
        raise typer.Exit(code=1) from exc
    finally:
        collector.close()


def _collect_arena(config: RouterConfig, store: SnapshotStore) -> SnapshotRecord:
    collector = LMArenaCollector(config.data_sources.lmarena, store)
    try:
        return collector.collect()
    except Exception as exc:
        err_console.print(f"[red]error: arena collection failed: {exc}[/red]")
        raise typer.Exit(code=1) from exc


def _print_local_result(result: DiscoveryResult) -> None:
    table = Table(title="local candidates")
    table.add_column("model")
    table.add_column("provider")
    table.add_column("efforts")
    table.add_column("provenance")
    for candidate in result.candidates:
        prov = "; ".join(f"{p.file} :: {p.json_path}" for p in candidate.provenance)
        table.add_row(
            candidate.model,
            candidate.provider or "-",
            ",".join(candidate.efforts) or "-",
            prov,
        )
    console.print(table)
    for note in result.notes:
        err_console.print(f"[dim]note: {note}[/dim]")


@app.command()
def collect(
    source: str = typer.Option(
        ..., "--source", help="Data source: aa | arena | local | all."
    ),
    config_path: str | None = typer.Option(None, "--config", help="Path to router.yaml."),
    data_dir: str | None = typer.Option(None, "--data-dir", help="Override data directory."),
    force: bool = typer.Option(False, "--force", help="Bypass cache and quota refusal."),
    max_age_hours: int | None = typer.Option(
        None, "--max-age-hours", help="Cache TTL override for aa."
    ),
) -> None:
    """Collect from a data source into the snapshot store."""
    valid = {"aa", "arena", "local", "all"}
    if source not in valid:
        err_console.print(
            f"[red]error: unknown source '{source}' (expected one of {sorted(valid)})[/red]"
        )
        raise typer.Exit(code=2)
    config, store = _load_ctx(config_path, data_dir)
    if source in {"aa", "all"}:
        record = _collect_aa(config, store, force, max_age_hours)
        _print_record(record)
        if record.errors:
            err_console.print(
                f"[yellow]warning: {len(record.errors)} error(s) recorded in snapshot meta[/yellow]"
            )
    if source in {"arena", "all"}:
        record = _collect_arena(config, store)
        _print_record(record)
    if source in {"local", "all"}:
        result = collect_local_candidates(config.data_sources.local_discovery)
        _print_local_result(result)
        err_console.print(f"[green]local discovery: {len(result.candidates)} candidate(s)[/green]")


@app.command()
def normalize(
    config_path: str | None = typer.Option(None, "--config", help="Path to router.yaml."),
    data_dir: str | None = typer.Option(None, "--data-dir", help="Override data directory."),
    source: str = typer.Option("all", "--source", help="Snapshots to apply: aa | arena | all."),
) -> None:
    """Load latest snapshots and upsert them into the registry."""
    valid = {"aa", "arena", "all"}
    if source not in valid:
        err_console.print(f"[red]error: unknown source '{source}'[/red]")
        raise typer.Exit(code=2)
    config, store = _load_ctx(config_path, data_dir)
    engine, _effective_url = registry_db.get_engine_with_fallback(
        config.database_url, config.sqlite_fallback_url
    )
    registry_db.init_schema(engine)

    applied: dict[str, dict[str, int]] = {}
    with registry_db.Session(engine) as session:
        if source in {"aa", "all"}:
            doc = store.latest("artificial-analysis")
            if doc is None:
                err_console.print(
                    "[yellow]warning: no artificial-analysis snapshot to normalize[/yellow]"
                )
            else:
                applied["artificial-analysis"] = registry_db.apply_aa_snapshot(session, doc)
        if source in {"arena", "all"}:
            doc = store.latest("lmarena")
            if doc is None:
                err_console.print("[yellow]warning: no lmarena snapshot to normalize[/yellow]")
            else:
                applied["lmarena"] = registry_db.apply_arena_snapshot(session, doc)
        session.commit()

    table = Table(title="normalize results")
    table.add_column("source")
    table.add_column("counts")
    for name, counts in applied.items():
        table.add_row(name, json.dumps(counts))
    console.print(table)


@snapshots_app.command("list")
def snapshots_list(
    source: str = typer.Option(..., "--source", help="Snapshot source name."),
    data_dir: str | None = typer.Option(None, "--data-dir", help="Override data directory."),
    config_path: str | None = typer.Option(None, "--config", help="Path to router.yaml."),
) -> None:
    """List stored snapshots for a source."""
    _, store = _load_ctx(config_path, data_dir)
    entries = store.index(source)
    if not entries:
        err_console.print(f"[yellow]no snapshots for source '{source}'[/yellow]")
        return
    table = Table(title=f"snapshots: {source}")
    for key in ("snapshot_id", "fetched_at", "record_count", "path"):
        table.add_column(key)
    for entry in entries:
        table.add_row(
            str(entry.get("snapshot_id", "")),
            str(entry.get("fetched_at", "")),
            str(entry.get("record_count", "")),
            str(entry.get("path", "")),
        )
    console.print(table)


@registry_app.command("stats")
def registry_stats(
    config_path: str | None = typer.Option(None, "--config", help="Path to router.yaml."),
    data_dir: str | None = typer.Option(None, "--data-dir", help="Override data directory."),
) -> None:
    """Show registry table row counts."""
    config, _ = _load_ctx(config_path, data_dir)
    engine, effective_url = registry_db.get_engine_with_fallback(
        config.database_url, config.sqlite_fallback_url
    )
    registry_db.init_schema(engine)
    stats = registry_db.registry_stats(engine)
    table = Table(title=f"registry stats ({effective_url})")
    table.add_column("table")
    table.add_column("rows", justify="right")
    for name, count in stats.items():
        table.add_row(name, str(count))
    console.print(table)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
