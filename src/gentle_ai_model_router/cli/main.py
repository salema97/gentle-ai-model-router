"""``router`` CLI: collect, normalize, snapshots, registry.

Exit codes: 0 = success (including graceful degradation with warnings),
1 = hard failure, 2 = usage error (typer default).
"""

from __future__ import annotations

import json
import sys
from contextlib import nullcontext as _nullcontext
from pathlib import Path
from typing import Annotated, Any

import httpx
import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from gentle_ai_model_router.collector.arena import LMArenaCollector
from gentle_ai_model_router.collector.artificial_analysis import (
    ArtificialAnalysisCollector,
    QuotaExceededError,
)
from gentle_ai_model_router.collector.gentle_telemetry import (
    GentleTelemetryCollector,
)
from gentle_ai_model_router.collector.local_discovery import (
    DiscoveryResult,
    collect_local_candidates,
)
from gentle_ai_model_router.collector.logging_conf import setup_logging
from gentle_ai_model_router.collector.routellm import RouteLLMCollector
from gentle_ai_model_router.collector.routerbench import RouterBenchCollector
from gentle_ai_model_router.collector.routing_benchmarks import (
    RoutingBenchmarksCollector,
)
from gentle_ai_model_router.collector.snapshots import SnapshotRecord, SnapshotStore
from gentle_ai_model_router.collector.swe_traces import SWETracesCollector
from gentle_ai_model_router.integration import codex_adapter as cx
from gentle_ai_model_router.integration import gentle_state_adapter as gs
from gentle_ai_model_router.integration import opencode_adapter as oa
from gentle_ai_model_router.integration import pi_adapter as pi
from gentle_ai_model_router.integration import telemetry_shim
from gentle_ai_model_router.integration.outcome import apply_outcomes
from gentle_ai_model_router.registry import db as registry_db
from gentle_ai_model_router.registry.normalize import Effort
from gentle_ai_model_router.router.bandit import bandit_version
from gentle_ai_model_router.router.config import RouterConfig, load_config
from gentle_ai_model_router.router.decision import CANONICAL_PHASES, TaskContext
from gentle_ai_model_router.router.policy import (
    CandidateRanking,
    PolicyError,
    rank_candidates,
    select_candidate,
)
from gentle_ai_model_router.router.reward import RewardError, aggregate_rewards, compute_rewards

app = typer.Typer(
    name="router",
    help="Learned phase-aware model+effort router for Gentle AI SDD phases.",
    no_args_is_help=True,
)
snapshots_app = typer.Typer(help="Snapshot inspection.", no_args_is_help=True)
registry_app = typer.Typer(help="Registry inspection.", no_args_is_help=True)
integrate_app = typer.Typer(help="Write decisions into runtime configs.", no_args_is_help=True)
gentle_state_app = typer.Typer(
    help="Write model_assignments into the Gentle AI state file "
    "(~/.gentle-ai/state.json) — the correct surface for Gentle-AI-managed setups.",
    no_args_is_help=True,
    invoke_without_command=True,
)
shim_app = typer.Typer(help="Telemetry shim ingestion.", no_args_is_help=True)
bandit_app = typer.Typer(help="Bandit reward loop: report + outcome update.", no_args_is_help=True)
thresholds_app = typer.Typer(
    help="Telemetry-driven threshold tuning: propose + explicit router.yaml apply.",
    no_args_is_help=True,
)
app.add_typer(snapshots_app, name="snapshots")
app.add_typer(registry_app, name="registry")
app.add_typer(integrate_app, name="integrate")
app.add_typer(shim_app, name="shim")
app.add_typer(bandit_app, name="bandit")
app.add_typer(thresholds_app, name="thresholds")
integrate_app.add_typer(gentle_state_app, name="gentle-state")
pi_app = typer.Typer(
    help="Write sdd-<phase> model assignments into Pi's models.json "
    "(~/.pi/gentle-ai/models.json) — the file gentle-pi's /gentle:models modal owns.",
    no_args_is_help=True,
    invoke_without_command=True,
)
integrate_app.add_typer(pi_app, name="pi")
codex_app = typer.Typer(
    help="Write sdd_<phase> model assignments into the Codex blocks of the "
    "Gentle AI state file (~/.gentle-ai/state.json) — the only safe Codex "
    "write surface (gentle-ai sync regenerates the per-phase table from it).",
    no_args_is_help=True,
    invoke_without_command=True,
)
integrate_app.add_typer(codex_app, name="codex")
plugins_app = typer.Typer(
    help="Manage runtime hook plugins for OpenCode and Pi.",
    no_args_is_help=True,
)
integrate_app.add_typer(plugins_app, name="plugins")

console = Console()
err_console = Console(stderr=True)
# Data tables (bandit report) need deterministic width under pipes/tests:
# a plain Console collapses to 80 cols and crops numeric cells.
bandit_console = Console(width=140)


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


def _collect_telemetry(config: RouterConfig, store: SnapshotStore) -> SnapshotRecord:
    collector = GentleTelemetryCollector(config.data_sources.gentle_telemetry, store)
    try:
        return collector.collect()
    except Exception as exc:
        err_console.print(f"[red]error: telemetry collection failed: {exc}[/red]")
        raise typer.Exit(code=1) from exc
    finally:
        collector.close()


def _collect_benchmarks(config: RouterConfig, store: SnapshotStore) -> SnapshotRecord:
    collector = RoutingBenchmarksCollector(config.data_sources.routing_benchmarks, store)
    try:
        return collector.collect()
    except Exception as exc:
        err_console.print(f"[red]error: benchmarks collection failed: {exc}[/red]")
        raise typer.Exit(code=1) from exc
    finally:
        collector.close()


def _collect_routerbench(
    config: RouterConfig,
    store: SnapshotStore,
    force: bool = False,
    max_age_hours: int | None = None,
) -> SnapshotRecord:
    collector = RouterBenchCollector(config.data_sources.routerbench, store)
    try:
        return collector.collect(force=force, max_age_hours=max_age_hours)
    except Exception as exc:
        err_console.print(f"[red]error: routerbench collection failed: {exc}[/red]")
        raise typer.Exit(code=1) from exc
    finally:
        collector.close()


def _collect_routellm(
    config: RouterConfig,
    store: SnapshotStore,
    force: bool = False,
    max_age_hours: int | None = None,
) -> SnapshotRecord:
    collector = RouteLLMCollector(config.data_sources.routellm, store)
    try:
        return collector.collect(force=force, max_age_hours=max_age_hours)
    except Exception as exc:
        err_console.print(f"[red]error: routellm collection failed: {exc}[/red]")
        raise typer.Exit(code=1) from exc
    finally:
        collector.close()


def _collect_swe_traces(
    config: RouterConfig,
    store: SnapshotStore,
    force: bool = False,
    max_age_hours: int | None = None,
) -> SnapshotRecord:
    collector = SWETracesCollector(config.data_sources.swe_traces, store)
    try:
        return collector.collect(force=force, max_age_hours=max_age_hours)
    except Exception as exc:
        err_console.print(f"[red]error: swe-traces collection failed: {exc}[/red]")
        raise typer.Exit(code=1) from exc
    finally:
        collector.close()


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


def _warn_all_continue(src: str) -> None:
    err_console.print(
        f"[yellow]warning: {src} collection failed; continuing with remaining sources[/yellow]"
    )


def _warn_snapshot_errors(record: SnapshotRecord) -> None:
    if record.errors:
        err_console.print(
            f"[yellow]warning: {len(record.errors)} error(s) recorded in snapshot meta[/yellow]"
        )


@app.command()
def collect(
    source: str = typer.Option(
        ...,
        "--source",
        help=(
            "Data source: aa | arena | local | telemetry | benchmarks | "
            "routerbench | routellm | swe-traces | all."
        ),
    ),
    config_path: str | None = typer.Option(None, "--config", help="Path to router.yaml."),
    data_dir: str | None = typer.Option(None, "--data-dir", help="Override data directory."),
    force: bool = typer.Option(False, "--force", help="Bypass cache and quota refusal."),
    max_age_hours: int | None = typer.Option(
        None, "--max-age-hours", help="Cache TTL override for aa."
    ),
) -> None:
    """Collect from a data source into the snapshot store."""
    valid = {
        "aa",
        "arena",
        "local",
        "telemetry",
        "benchmarks",
        "routerbench",
        "routellm",
        "swe-traces",
        "all",
    }
    if source not in valid:
        err_console.print(
            f"[red]error: unknown source '{source}' (expected one of {sorted(valid)})[/red]"
        )
        raise typer.Exit(code=2)
    config, store = _load_ctx(config_path, data_dir)
    if source in {"aa", "all"}:
        try:
            record = _collect_aa(config, store, force, max_age_hours)
            _print_record(record)
            _warn_snapshot_errors(record)
        except (typer.Exit, Exception):
            if source != "all":
                raise
            _warn_all_continue("aa")
    if source in {"arena", "all"}:
        try:
            record = _collect_arena(config, store)
            _print_record(record)
        except (typer.Exit, Exception):
            if source != "all":
                raise
            _warn_all_continue("arena")
    if source in {"telemetry", "all"}:
        try:
            record = _collect_telemetry(config, store)
            _print_record(record)
            _warn_snapshot_errors(record)
        except (typer.Exit, Exception):
            if source != "all":
                raise
            _warn_all_continue("telemetry")
    if source in {"benchmarks", "all"}:
        try:
            record = _collect_benchmarks(config, store)
            _print_record(record)
            _warn_snapshot_errors(record)
        except (typer.Exit, Exception):
            if source != "all":
                raise
            _warn_all_continue("benchmarks")
    if source in {"routerbench", "all"}:
        try:
            record = _collect_routerbench(config, store, force=force, max_age_hours=max_age_hours)
            _print_record(record)
            _warn_snapshot_errors(record)
        except (typer.Exit, Exception):
            if source != "all":
                raise
            _warn_all_continue("routerbench")
    if source in {"routellm", "all"}:
        try:
            record = _collect_routellm(config, store, force=force, max_age_hours=max_age_hours)
            _print_record(record)
            _warn_snapshot_errors(record)
        except (typer.Exit, Exception):
            if source != "all":
                raise
            _warn_all_continue("routellm")
    if source in {"swe-traces", "all"}:
        try:
            record = _collect_swe_traces(config, store, force=force, max_age_hours=max_age_hours)
            _print_record(record)
            _warn_snapshot_errors(record)
        except (typer.Exit, Exception):
            if source != "all":
                raise
            _warn_all_continue("swe-traces")
    if source in {"local", "all"}:
        result = collect_local_candidates(config.data_sources.local_discovery)
        _print_local_result(result)
        err_console.print(f"[green]local discovery: {len(result.candidates)} candidate(s)[/green]")


@app.command()
def normalize(
    config_path: str | None = typer.Option(None, "--config", help="Path to router.yaml."),
    data_dir: str | None = typer.Option(None, "--data-dir", help="Override data directory."),
    source: str = typer.Option(
        "all",
        "--source",
        help="Snapshots to apply: aa | arena | local | telemetry | benchmarks | all.",
    ),
) -> None:
    """Load latest snapshots and upsert them into the registry."""
    valid = {"aa", "arena", "local", "telemetry", "benchmarks", "all"}
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
        if source in {"telemetry", "all"}:
            doc = store.latest("gentle-telemetry")
            if doc is None:
                err_console.print(
                    "[yellow]warning: no gentle-telemetry snapshot to normalize[/yellow]"
                )
            else:
                applied["gentle-telemetry"] = registry_db.apply_telemetry_snapshot(session, doc)
        if source in {"benchmarks", "all"}:
            doc = store.latest("routing-benchmarks")
            if doc is None:
                err_console.print(
                    "[yellow]warning: no routing-benchmarks snapshot to normalize[/yellow]"
                )
            else:
                applied["routing-benchmarks"] = (
                    registry_db.apply_routing_benchmarks_snapshot(session, doc)
                )
        if source in {"local", "all"}:
            local_result = collect_local_candidates(config.data_sources.local_discovery)
            applied["local"] = registry_db.apply_local_candidates(session, local_result.candidates)
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


# --------------------------------------------------------------------------- #
# Phase 1b: deterministic policy + OpenCode write adapter + telemetry shim
# --------------------------------------------------------------------------- #


@app.command()
def route(
    phase: str = typer.Option(..., "--phase", help="SDD phase (one of the 11 canonical phases)."),
    config_path: str | None = typer.Option(None, "--config", help="Path to router.yaml."),
    data_dir: str | None = typer.Option(None, "--data-dir", help="Override data directory."),
    context_tokens: int | None = typer.Option(None, "--context-tokens", help="Task context size."),
    task_type: str | None = typer.Option(None, "--task-type", help="Task type hint."),
    as_json: bool = typer.Option(False, "--json", help="Print the decision as JSON."),
) -> None:
    """Select (model, deployment, effort) for a phase with the baseline policy."""
    config, _ = _load_ctx(config_path, data_dir)
    engine, _effective = registry_db.get_engine_with_fallback(
        config.database_url, config.sqlite_fallback_url
    )
    registry_db.init_schema(engine)
    try:
        with registry_db.Session(engine) as session:
            decision = select_candidate(
                session,
                phase,
                config,
                TaskContext(task_type=task_type, context_tokens=context_tokens),
            )
    except PolicyError as exc:
        err_console.print(f"[red]error: {exc}[/red]")
        raise typer.Exit(code=2) from exc
    if as_json:
        console.print(json.dumps(decision.to_dict(), indent=2))
        return
    table = Table(title=f"route: {decision.phase}")
    table.add_row("model", decision.model)
    table.add_row("provider", decision.provider)
    table.add_row("deployment", decision.deployment)
    table.add_row("effort", decision.effort)
    table.add_row("score", str(decision.score))
    table.add_row("quality", str(decision.quality))
    table.add_row("estimated_tokens", str(decision.estimated_tokens))
    table.add_row("estimated_cost", f"${decision.estimated_cost:.6f}")
    table.add_row("policy_version", decision.policy_version)
    table.add_row("reason_codes", ", ".join(decision.reason_codes))
    if decision.alternatives:
        table.add_row(
            "alternatives",
            "\n".join(
                f"{a.model}#{a.effort} ({a.estimated_tokens:.0f} tok)"
                for a in decision.alternatives
            ),
        )
    console.print(table)


def _open_registry(config: RouterConfig):
    engine, _effective = registry_db.get_engine_with_fallback(
        config.database_url, config.sqlite_fallback_url
    )
    registry_db.init_schema(engine)
    return engine


# --------------------------------------------------------------------------- #
# Phase 3a: FastAPI server + policy inspection CLI
# --------------------------------------------------------------------------- #


def _resolve_serve_ranker(ranker: str | None, models_dir: Path) -> Any | None:
    """Build the OnnxRanker for serve: explicit --ranker wins, else the record.

    Without --ranker the promoted checkpoint is resolved from
    ``models/promoted/promoted.json`` (artifact_kind=checkpoint) → its dir →
    model.quant.onnx preferred, model.onnx fallback. Returns None when no
    checkpoint is promoted. Raises PromotionError when the record points to
    a missing artifact (the serve command fails closed on that).
    """
    from gentle_ai_model_router.training import promote as promote_mod
    from gentle_ai_model_router.training.onnx_export import OnnxRanker

    if ranker is not None:
        ranker_p = Path(ranker)
        if ranker_p.is_file():
            return OnnxRanker(ranker_p.parent, model_path=ranker_p)
        return OnnxRanker(ranker_p)
    onnx_path = promote_mod.resolve_promoted_onnx(models_dir)
    if onnx_path is None:
        return None
    return OnnxRanker(onnx_path.parent, model_path=onnx_path)


@app.command()
def serve(
    host: str | None = typer.Option(None, "--host", help="Bind host (default: config api.host)."),
    port: int | None = typer.Option(None, "--port", help="Bind port (default: config api.port)."),
    config_path: str | None = typer.Option(None, "--config", help="Path to router.yaml."),
    data_dir: str | None = typer.Option(None, "--data-dir", help="Override data directory."),
    ranker: str | None = typer.Option(
        None, "--ranker", help="Path to ONNX ranker directory or model file."
    ),
    models_dir: str = typer.Option(
        "models", "--models-dir", help="Models root that holds the promoted/ record."
    ),
) -> None:
    """Serve the routing API over uvicorn (localhost by default, local-first)."""
    import uvicorn

    from gentle_ai_model_router.api.server import create_app
    from gentle_ai_model_router.training import promote as promote_mod

    config, _ = _load_ctx(config_path, data_dir)
    engine = _open_registry(config)
    shim_url = config.api.shim_db_path or config.telemetry_url
    if "://" not in shim_url:
        shim_url = f"sqlite:///{shim_url}"  # plain SQLite path → SQLAlchemy URL
    shim_store = telemetry_shim.ShimStore(shim_url)
    shim_store.init_schema()

    try:
        ranker_instance = _resolve_serve_ranker(ranker, Path(models_dir))
    except promote_mod.PromotionError as exc:
        # Fail closed: the promotion record points to a missing artifact.
        err_console.print(f"[red]error: {escape(str(exc))}[/red]")
        raise typer.Exit(code=2) from exc
    except Exception as exc:
        err_console.print(
            f"[yellow]warning: could not auto-load promoted ONNX ranker: {exc}[/yellow]"
        )
        ranker_instance = None

    api_app = create_app(config, engine, shim_store, ranker=ranker_instance)
    bind_host = host or config.api.host
    bind_port = port or config.api.port
    console.print(f"serving gentle-ai-model-router on http://{bind_host}:{bind_port}")
    uvicorn.run(api_app, host=bind_host, port=bind_port)


@app.command()
def policy(
    phase: str | None = typer.Option(
        None, "--phase", help="One SDD phase (default: all 11 canonical phases)."
    ),
    config_path: str | None = typer.Option(None, "--config", help="Path to router.yaml."),
    data_dir: str | None = typer.Option(None, "--data-dir", help="Override data directory."),
) -> None:
    """Show the active per-phase policy: recommended config + top-3 alternatives."""
    config, _ = _load_ctx(config_path, data_dir)
    engine = _open_registry(config)
    phases = [phase] if phase else list(CANONICAL_PHASES)
    try:
        with registry_db.Session(engine) as session:
            for ph in phases:
                ranking = rank_candidates(session, ph, config)
                _print_policy_table(ranking)
    except PolicyError as exc:
        err_console.print(f"[red]error: {exc}[/red]")
        raise typer.Exit(code=2) from exc


def _print_policy_table(ranking: CandidateRanking) -> None:
    """Render one phase ranking: selected row + top-3 alternatives."""
    top = ranking.candidates[: 1 + 3]
    table = Table(
        title=f"policy: {ranking.phase} "
        f"(threshold={ranking.threshold}, candidates={len(ranking.candidates)})"
    )
    table.add_column("pick", justify="right")
    for column in ("model", "deployment", "effort", "score", "est tokens", "est cost"):
        numeric = column in {"score", "est tokens", "est cost"}
        table.add_column(column, justify="right" if numeric else "left")
    for idx, c in enumerate(top):
        table.add_row(
            "*" if idx == 0 else str(idx),
            c.model.canonical_id,
            c.deployment.deployment_ref,
            c.variant.effort,
            f"{c.score:.4f}",
            f"{c.estimated_tokens:.0f}",
            f"${c.estimated_cost:.6f}",
        )
    console.print(table)


@app.command()
def calibrate_thresholds(
    phase: str | None = typer.Option(
        None, "--phase", help="One SDD phase (default: all 11 canonical phases)."
    ),
    config_path: str | None = typer.Option(None, "--config", help="Path to router.yaml."),
    data_dir: str | None = typer.Option(None, "--data-dir", help="Override data directory."),
) -> None:
    """Suggest per-phase threshold_quality values from registry priors (read-only).

    Decision support ONLY: this command never modifies router.yaml. For each
    phase it shows the current threshold, the p25/p50/p75 of the achievable
    quality distribution (computed with the policy's own effort_quality +
    benchmark priors), the fraction of candidates whose best effort meets the
    current threshold, and a suggested threshold = min(max(current, p50), p90).
    Priors are bootstrap-quality (external benchmark scores, not measured task
    success) — treat the suggestion as a starting point, not ground truth.
    Always exits 0.
    """
    from gentle_ai_model_router.router.calibrate import calibrate_thresholds as calibrate

    config, _ = _load_ctx(config_path, data_dir)
    engine = _open_registry(config)
    try:
        with registry_db.Session(engine) as session:
            rows = calibrate(session, config, phase)
    except PolicyError as exc:
        err_console.print(f"[red]error: {exc}[/red]")
        return  # informational command: always exit 0

    table = Table(title="threshold calibration (read-only; nothing written)")
    for column in (
        "phase",
        "current",
        "p25",
        "p50",
        "p75",
        "meet %",
        "suggested",
    ):
        numeric = column not in {"phase"}
        table.add_column(column, justify="right" if numeric else "left")
    for row in rows:
        table.add_row(
            row.phase,
            f"{row.threshold:.3f}",
            f"{row.p25:.3f}",
            f"{row.p50:.3f}",
            f"{row.p75:.3f}",
            f"{row.fraction_meeting:.1%}",
            f"{row.suggested:.3f}",
        )
    console.print(table)
    err_console.print(
        "[yellow]suggested = min(max(current, p50), p90); priors are bootstrap-quality —[/yellow]"
    )
    err_console.print("[yellow]verify against real telemetry before adopting.[/yellow]")


@app.command()
def explain(
    phase: str = typer.Option(..., "--phase", help="SDD phase to explain."),
    task: str | None = typer.Option(
        None, "--task", help="Task description (echoed for context; not scored yet)."
    ),
    as_json: bool = typer.Option(False, "--json", help="Print the Decision as JSON."),
    config_path: str | None = typer.Option(None, "--config", help="Path to router.yaml."),
    data_dir: str | None = typer.Option(None, "--data-dir", help="Override data directory."),
) -> None:
    """Explain the ranking for a phase: full candidate ladder + selected config."""
    from gentle_ai_model_router.router.policy import normalize_phase

    config, _ = _load_ctx(config_path, data_dir)
    engine = _open_registry(config)
    try:
        phase_name = normalize_phase(phase)
    except PolicyError as exc:
        err_console.print(f"[red]error: {exc}[/red]")
        raise typer.Exit(code=2) from exc

    try:
        with registry_db.Session(engine) as session:
            decision = select_candidate(session, phase_name, config)
            if as_json:
                console.print(json.dumps(decision.to_dict(), indent=2))
                return
            ranking = rank_candidates(session, phase_name, config)
    except PolicyError as exc:
        err_console.print(f"[red]error: {exc}[/red]")
        raise typer.Exit(code=2) from exc

    summary = Table(title=f"explain: {phase_name}")
    summary.add_row("phase (canonical)", ranking.phase)
    if task:
        summary.add_row("task", task)
    summary.add_row("registry (model,deployment) pairs", str(ranking.pairs_considered))
    summary.add_row("threshold_quality", str(ranking.threshold))
    summary.add_row("policy_version", ranking.policy_version)
    if decision.confidence is not None:
        summary.add_row("confidence (calibrated)", f"{decision.confidence:.2%}")
    if decision.system_one:
        eff = decision.system_one.get("effort_score")
        if eff is not None:
            summary.add_row(escape("effort expected (E[k])"), f"{eff:.2f}")
        noul = decision.system_one.get("noul_fast_success")
        if noul is not None:
            summary.add_row("P(fast_success)", f"{noul:.2%}")
    console.print(summary)

    winner = ranking.candidates[0]
    table = Table(title=f"candidate ranking (top {min(10, len(ranking.candidates))})")
    table.add_column("pick", justify="right")
    for column in ("model", "deployment", "effort", "quality", "est tokens", "est cost", "score"):
        table.add_column(
            column,
            justify="right" if column in {"quality", "est tokens", "est cost", "score"} else "left",
        )
    for idx, c in enumerate(ranking.candidates[:10]):
        table.add_row(
            "*" if idx == 0 else str(idx),
            c.model.canonical_id,
            c.deployment.deployment_ref,
            c.variant.effort,
            f"{c.quality:.4f}",
            f"{c.estimated_tokens:.0f}",
            f"${c.estimated_cost:.6f}",
            f"{c.score:.4f}",
        )
    console.print(table)

    selected = Table(title="selected configuration")
    selected.add_row("model", winner.model.canonical_id)
    selected.add_row("provider", winner.provider.registry_key)
    selected.add_row("deployment", winner.deployment.deployment_ref)
    selected.add_row("effort", winner.variant.effort)
    selected.add_row("estimated_tokens", f"{winner.estimated_tokens:.0f}")
    selected.add_row("estimated_cost", f"${winner.estimated_cost:.6f}")
    selected.add_row("reason_codes", ", ".join(winner.reason_codes + ("cheapest_of_meeting",)))
    console.print(selected)


@app.command()
def export(
    output: str | None = typer.Option(
        None, "--output", help="Target file (default: models/policy/<policy_version>.json)."
    ),
    config_path: str | None = typer.Option(None, "--config", help="Path to router.yaml."),
    data_dir: str | None = typer.Option(None, "--data-dir", help="Override data directory."),
) -> None:
    """Export the active policy as JSON (the Gentle AI integration artifact)."""
    from datetime import UTC, datetime

    from gentle_ai_model_router.registry.fingerprint import registry_fingerprint
    from gentle_ai_model_router.router.policy import full_policy_version, rank_candidates

    config, _ = _load_ctx(config_path, data_dir)
    engine = _open_registry(config)
    try:
        with registry_db.Session(engine) as session:
            phases: dict[str, dict] = {}
            for ph in CANONICAL_PHASES:
                ranking = rank_candidates(session, ph, config)
                phase_cfg = config.phase_config(ph)
                winner = ranking.candidates[0]
                phases[ph] = {
                    "policy_version": ranking.policy_version,
                    "thresholds": {"threshold_quality": ranking.threshold},
                    "weights": phase_cfg.weights,
                    "selected": {
                        "model": winner.model.canonical_id,
                        "provider": winner.provider.registry_key,
                        "deployment": winner.deployment.deployment_ref,
                        "effort": winner.variant.effort,
                        "score": winner.score,
                        "quality": winner.quality,
                        "estimated_tokens": winner.estimated_tokens,
                        "estimated_cost": winner.estimated_cost,
                        "reason_codes": list(winner.reason_codes) + ["cheapest_of_meeting"],
                    },
                    "alternatives": [
                        {
                            "model": c.model.canonical_id,
                            "provider": c.provider.registry_key,
                            "deployment": c.deployment.deployment_ref,
                            "effort": c.variant.effort,
                            "score": c.score,
                            "quality": c.quality,
                            "estimated_tokens": c.estimated_tokens,
                            "estimated_cost": c.estimated_cost,
                        }
                        for c in ranking.candidates[1 : config.policy.top_k]
                    ],
                }
    except PolicyError as exc:
        err_console.print(f"[red]error: {exc}[/red]")
        raise typer.Exit(code=2) from exc

    policy_version = full_policy_version(config)
    payload = {
        "exported_at": datetime.now(UTC).isoformat(),
        "policy_version": policy_version,
        "registry_hash": registry_fingerprint(engine),
        "phases": phases,
    }
    out_path = Path(output) if output else Path("models/policy") / f"{policy_version}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    console.print(f"[green]exported policy {policy_version} -> {out_path}[/green]")


def _resolve_variants(config: RouterConfig) -> dict[tuple[str, str], list[str]]:
    return oa.load_variants_cache(
        Path(config.integrate.variants_cache_v1).expanduser(),
        Path(config.integrate.variants_cache_v2_dir).expanduser(),
    )


@integrate_app.command("opencode")
def integrate_opencode(
    phase: str = typer.Option(..., "--phase", help="SDD phase."),
    model: str = typer.Option(..., "--model", help="provider/model."),
    effort: str = typer.Option(..., "--effort", help="Reasoning effort level."),
    config_path: str | None = typer.Option(None, "--config", help="Path to router.yaml."),
    target: str | None = typer.Option(
        None, "--config-target", help="OpenCode config path (default: resolved global)."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the diff, write nothing."),
    create: bool = typer.Option(False, "--create", help="Create the config file if missing."),
    data_dir: str | None = typer.Option(None, "--data-dir", help="Override data directory."),
) -> None:
    """Write agent['sdd-<phase>'].model/variant into an OpenCode config."""
    config, _ = _load_ctx(config_path, data_dir)
    name = phase.removeprefix("sdd-")
    if name not in CANONICAL_PHASES:
        err_console.print(
            f"[red]error: unknown phase '{phase}' "
            f"(expected one of: {', '.join(CANONICAL_PHASES)})[/red]"
        )
        raise typer.Exit(code=2)
    if effort not in {level.value for level in Effort}:
        err_console.print(
            f"[red]error: unknown effort '{effort}' (expected one of: "
            f"{', '.join(level.value for level in Effort)})[/red]"
        )
        raise typer.Exit(code=2)
    path = oa.resolve_global_config(target)
    variants = _resolve_variants(config)
    try:
        result = oa.apply_decision(
            path, name, model, effort, variants, create=create, dry_run=dry_run,
            backup=config.integrate.backup,
        )
    except oa.AdapterError as exc:
        err_console.print(f"[red]error: {exc}[/red]")
        raise typer.Exit(code=2) from exc
    for code in result.reason_codes:
        err_console.print(f"[yellow]reason: {code}[/yellow]")
    if result.diff:
        console.print(result.diff, highlight=False)
    if result.wrote:
        console.print(f"[green]wrote {result.path}[/green]")
        if result.backup_path:
            console.print(f"[dim]backup: {result.backup_path}[/dim]")
    else:
        console.print("[dim]dry-run: nothing written[/dim]")


@integrate_app.command("rollback")
def integrate_rollback(
    target: str | None = typer.Option(
        None, "--config-target", help="OpenCode config path (default: resolved global)."
    ),
) -> None:
    """Restore the latest router backup of an OpenCode config."""
    path = oa.resolve_global_config(target)
    restored = oa.rollback(path)
    if restored is None:
        err_console.print(f"[yellow]no backup found for {path}[/yellow]")
        raise typer.Exit(code=1)
    console.print(f"[green]restored {path} from {restored}[/green]")


@integrate_app.command("status")
def integrate_status(
    target: str | None = typer.Option(
        None, "--config-target", help="OpenCode config path (default: resolved global)."
    ),
) -> None:
    """Show current per-phase assignments from the effective OpenCode config."""
    path = oa.resolve_global_config(target)
    assignments = oa.read_assignments(path)
    table = Table(title=f"opencode assignments ({path})")
    table.add_column("agent")
    table.add_column("model")
    table.add_column("variant")
    table.add_column("managed")
    for agent_key in [f"sdd-{p}" for p in CANONICAL_PHASES] + ["gentle-orchestrator"]:
        entry = assignments.get(agent_key)
        if entry is None:
            continue
        table.add_row(
            agent_key,
            str(entry.get("model") or "-"),
            str(entry.get("variant") or "-"),
            "gentle-ai/sdd" if entry.get("managed") else "-",
        )
    console.print(table)


@plugins_app.command("install")
def plugins_install(
    opencode: bool = typer.Option(False, "--opencode", help="Install OpenCode hook plugin."),
    pi: bool = typer.Option(False, "--pi", help="Install Pi hook plugin."),
    target_dir: str | None = typer.Option(
        None, "--target-dir", help="Target installation directory override."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Report planned changes without writing."
    ),
) -> None:
    """Install runtime hook plugins for OpenCode and/or Pi."""
    from gentle_ai_model_router.integration.plugin_installer import (
        install_opencode_plugin,
        install_pi_plugin,
    )

    install_all = not opencode and not pi
    targets: list[str] = []
    if opencode or install_all:
        targets.append("opencode")
    if pi or install_all:
        targets.append("pi")

    for target in targets:
        if target == "opencode":
            res = install_opencode_plugin(target_dir=target_dir, dry_run=dry_run)
        else:
            res = install_pi_plugin(target_dir=target_dir, dry_run=dry_run)

        console.print(f"[bold]{res.plugin_name} plugin[/bold] -> {res.target_dir}")
        for f in res.files:
            prefix = "[yellow][dry-run][/yellow] " if dry_run else ""
            if f.status == "unchanged":
                console.print(f"  {prefix}[dim]{f.target_path.name}: unchanged[/dim]")
            elif f.status in ("created", "dry-run"):
                console.print(f"  {prefix}[green]{f.target_path.name}: installed[/green]")
            elif f.status == "updated":
                console.print(f"  {prefix}[green]{f.target_path.name}: updated[/green]")
                if f.backup_path:
                    console.print(f"  {prefix}[dim]backup: {f.backup_path}[/dim]")


@plugins_app.command("status")
def plugins_status(
    opencode_dir: str | None = typer.Option(
        None, "--opencode-dir", help="OpenCode plugin directory override."
    ),
    pi_dir: str | None = typer.Option(
        None, "--pi-dir", help="Pi plugin directory override."
    ),
) -> None:
    """Check installation and version status of runtime hook plugins."""
    from gentle_ai_model_router.integration.plugin_installer import plugin_status

    statuses = plugin_status(opencode_dir=opencode_dir, pi_dir=pi_dir)
    table = Table(title="Runtime hook plugins status")
    table.add_column("Plugin", style="bold")
    table.add_column("Status")
    table.add_column("Target Directory")
    table.add_column("Files Present")
    table.add_column("Backups")
    table.add_column("Version")

    for _, info in statuses.items():
        status_str = (
            "[green]installed[/green]" if info.installed else "[yellow]not installed[/yellow]"
        )
        table.add_row(
            info.name,
            status_str,
            str(info.target_dir),
            ", ".join(info.files_present) if info.files_present else "[dim]none[/dim]",
            str(info.backup_count),
            info.router_version or "[dim]-[/dim]",
        )

    console.print(table)



def _print_gentle_state_followup() -> None:
    console.print("next step — run it yourself (the router NEVER runs sync):")
    console.print("  gentle-ai sync --sdd-profile-strategy external-single-active")
    err_console.print(
        "[yellow]WARNING: gentle-ai sync REGENERATES the opencode agent configs from[/yellow]"
    )
    err_console.print(
        "[yellow]this state file ('model_assignments'); hand-edited agent blocks in[/yellow]"
    )
    err_console.print(
        "[yellow]opencode.json will be overwritten. 'external-single-active' is the[/yellow]"
    )
    err_console.print(
        "[yellow]profile strategy designed for external tooling "
        "(docs/opencode-profiles.md).[/yellow]"
    )
    err_console.print(
        "[yellow]For configs with NO __managed_by: gentle-ai/sdd blocks (unmanaged),[/yellow]"
    )
    err_console.print("[yellow]use 'router integrate opencode' instead.[/yellow]")


@gentle_state_app.callback()
def integrate_gentle_state(
    ctx: typer.Context,
    phase: str | None = typer.Option(None, "--phase", help="SDD phase."),
    model: str | None = typer.Option(None, "--model", help="provider/model."),
    effort: str | None = typer.Option(None, "--effort", help="Reasoning effort level."),
    state: str | None = typer.Option(
        None, "--state", help="Gentle AI state file path (default: ~/.gentle-ai/state.json)."
    ),
    config_path: str | None = typer.Option(None, "--config", help="Path to router.yaml."),
    data_dir: str | None = typer.Option(None, "--data-dir", help="Override data directory."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the diff, write nothing."),
    create: bool = typer.Option(False, "--create", help="Create the state file if missing."),
    verify: bool = typer.Option(
        False, "--verify", help="Re-read the state file and confirm the assignment round-trips."
    ),
) -> None:
    """Set model_assignments["sdd-<phase>"] in the Gentle AI state file.

    Preferred write surface for Gentle-AI-managed setups: gentle-ai sync
    regenerates opencode agent configs FROM this file. Refuses malformed JSON,
    unrecognized existing entries, and (without --create) missing files.
    """
    if ctx.invoked_subcommand is not None:
        return  # `gentle-state rollback ...` — handled by its own command
    required = (("--phase", phase), ("--model", model), ("--effort", effort))
    missing = [name for name, value in required if value is None]
    if missing:
        err_console.print(f"[red]error: missing required option(s): {', '.join(missing)}[/red]")
        raise typer.Exit(code=2)
    name = phase.removeprefix("sdd-")
    if name not in CANONICAL_PHASES:
        err_console.print(
            f"[red]error: unknown phase '{phase}' "
            f"(expected one of: {', '.join(CANONICAL_PHASES)})[/red]"
        )
        raise typer.Exit(code=2)
    if dry_run and verify:
        err_console.print("[red]error: --verify requires a real write (not --dry-run)[/red]")
        raise typer.Exit(code=2)
    config, _ = _load_ctx(config_path, data_dir)
    path = gs.resolve_state_path(state)
    try:
        result = gs.apply_assignment(
            path, name, model, effort, create=create, dry_run=dry_run,
            backup=config.integrate.backup,
        )
    except gs.AdapterError as exc:
        err_console.print(f"[red]error: {exc}[/red]")
        raise typer.Exit(code=2) from exc
    if result.diff:
        console.print(result.diff, highlight=False)
    if result.wrote:
        console.print(f"[green]wrote {result.path}[/green]")
        console.print(
            f'{result.key_used}["{result.agent_key}"] = '
            f'{{"provider_id": {json.dumps(result.provider_id)}, '
            f'"model_id": {json.dumps(result.model_id)}, '
            f'"effort": {json.dumps(result.effort)}}}'
        )
        if result.backup_path:
            console.print(f"[dim]backup: {result.backup_path}[/dim]")
        _print_gentle_state_followup()
    else:
        console.print("[dim]dry-run: nothing written[/dim]")
    if verify:
        ok, observed = gs.verify_assignment(path, name, model, effort)
        if ok:
            console.print(
                f"[green]verify OK: {result.agent_key} -> "
                f"{model}#{effort} round-trips in {path}[/green]"
            )
        else:
            err_console.print(
                f"[red]verify MISMATCH: expected {model}#{effort}, "
                f"observed {observed!r}[/red]"
            )
            raise typer.Exit(code=1)


@gentle_state_app.command("rollback")
def integrate_gentle_state_rollback(
    state: str | None = typer.Option(
        None, "--state", help="Gentle AI state file path (default: ~/.gentle-ai/state.json)."
    ),
) -> None:
    """Restore the latest router backup of the Gentle AI state file."""
    path = gs.resolve_state_path(state)
    restored = gs.rollback(path)
    if restored is None:
        err_console.print(f"[yellow]no backup found for {path}[/yellow]")
        raise typer.Exit(code=1)
    console.print(f"[green]restored {path} from {restored}[/green]")


@gentle_state_app.command("status")
def integrate_gentle_state_status(
    state: str | None = typer.Option(
        None, "--state", help="Gentle AI state file path (default: ~/.gentle-ai/state.json)."
    ),
) -> None:
    """Show current per-phase assignments from the Gentle AI state file."""
    path = gs.resolve_state_path(state)
    table = Table(title=f"gentle-state assignments ({path})")
    table.add_column("agent")
    table.add_column("provider_id")
    table.add_column("model_id")
    table.add_column("effort")
    for ph in CANONICAL_PHASES:
        entry = gs.read_assignment(path, ph)
        if entry is None:
            continue
        table.add_row(
            f"sdd-{ph}",
            str(entry.get("provider_id") or "-"),
            str(entry.get("model_id") or "-"),
            str(entry.get("effort") or "-"),
        )
    console.print(table)


def _print_pi_followup() -> None:
    err_console.print(
        "[yellow]NOTE: Pi model assignment belongs to gentle-pi, not the Gentle AI[/yellow]"
    )
    err_console.print(
        "[yellow]installer (docs/pi.md:119-121). This file is rewritten by the[/yellow]"
    )
    err_console.print(
        "[yellow]/gentle:models modal; a router write can be overwritten by the[/yellow]"
    )
    err_console.print(
        "[yellow]next modal save. Entries outside sdd-<phase> keys are preserved.[/yellow]"
    )


@pi_app.callback()
def integrate_pi(
    ctx: typer.Context,
    phase: str | None = typer.Option(None, "--phase", help="SDD phase."),
    model: str | None = typer.Option(None, "--model", help="provider/model."),
    effort: str | None = typer.Option(None, "--effort", help="Thinking effort level."),
    models: str | None = typer.Option(
        None,
        "--models",
        help="Pi models.json path (default: $GENTLE_PI_CONFIG_HOME/gentle-ai/models.json "
        "or ~/.pi/gentle-ai/models.json).",
    ),
    config_path: str | None = typer.Option(None, "--config", help="Path to router.yaml."),
    data_dir: str | None = typer.Option(None, "--data-dir", help="Override data directory."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the diff, write nothing."),
    create: bool = typer.Option(False, "--create", help="Create the models file if missing."),
    verify: bool = typer.Option(
        False, "--verify", help="Re-read the models file and confirm the assignment round-trips."
    ),
) -> None:
    """Set models.json["sdd-<phase>"] = {"model", "thinking"} for Pi (gentle-pi).

    The file is owned by gentle-pi and applied as --model/--thinking argv
    pairs. Bare-string entries are upgraded to the object form; unknown keys
    and unrecognized entry shapes are refused (fail closed), never clobbered.
    """
    if ctx.invoked_subcommand is not None:
        return  # `pi rollback ...` / `pi status ...` — handled by their commands
    required = (("--phase", phase), ("--model", model), ("--effort", effort))
    missing = [name for name, value in required if value is None]
    if missing:
        err_console.print(f"[red]error: missing required option(s): {', '.join(missing)}[/red]")
        raise typer.Exit(code=2)
    name = phase.removeprefix("sdd-")
    if name not in CANONICAL_PHASES:
        err_console.print(
            f"[red]error: unknown phase '{phase}' "
            f"(expected one of: {', '.join(CANONICAL_PHASES)})[/red]"
        )
        raise typer.Exit(code=2)
    if dry_run and verify:
        err_console.print("[red]error: --verify requires a real write (not --dry-run)[/red]")
        raise typer.Exit(code=2)
    config, _ = _load_ctx(config_path, data_dir)
    path = pi.resolve_models_path(models)
    try:
        result = pi.apply_assignment(
            path, name, model, effort, create=create, dry_run=dry_run,
            backup=config.integrate.backup,
        )
    except pi.AdapterError as exc:
        err_console.print(f"[red]error: {exc}[/red]")
        raise typer.Exit(code=2) from exc
    if result.diff:
        console.print(result.diff, highlight=False)
    if result.wrote:
        console.print(f"[green]wrote {result.path}[/green]")
        console.print(
            f'{result.agent_key} = {{"model": {json.dumps(result.model)}, '
            f'"thinking": {json.dumps(result.thinking)}}}'
        )
        if result.backup_path:
            console.print(f"[dim]backup: {result.backup_path}[/dim]")
        _print_pi_followup()
    else:
        console.print("[dim]dry-run: nothing written[/dim]")
    if verify:
        ok, observed = pi.verify_assignment(path, name, model, effort)
        if ok:
            console.print(
                f"[green]verify OK: {result.agent_key} -> "
                f"{model}#{effort} round-trips in {path}[/green]"
            )
        else:
            err_console.print(
                f"[red]verify MISMATCH: expected {model}#{effort}, "
                f"observed {observed!r}[/red]"
            )
            raise typer.Exit(code=1)


@pi_app.command("rollback")
def integrate_pi_rollback(
    models: str | None = typer.Option(
        None,
        "--models",
        help="Pi models.json path (default: $GENTLE_PI_CONFIG_HOME/gentle-ai/models.json "
        "or ~/.pi/gentle-ai/models.json).",
    ),
) -> None:
    """Restore the latest router backup of Pi's models file."""
    path = pi.resolve_models_path(models)
    restored = pi.rollback(path)
    if restored is None:
        err_console.print(f"[yellow]no backup found for {path}[/yellow]")
        raise typer.Exit(code=1)
    console.print(f"[green]restored {path} from {restored}[/green]")


@pi_app.command("status")
def integrate_pi_status(
    models: str | None = typer.Option(
        None,
        "--models",
        help="Pi models.json path (default: $GENTLE_PI_CONFIG_HOME/gentle-ai/models.json "
        "or ~/.pi/gentle-ai/models.json).",
    ),
) -> None:
    """Show current per-phase assignments from Pi's models file."""
    path = pi.resolve_models_path(models)
    assignments = pi.read_assignments(path)
    table = Table(title=f"pi assignments ({path})")
    table.add_column("agent")
    table.add_column("model")
    table.add_column("thinking")
    for ph in CANONICAL_PHASES:
        entry = assignments.get(f"sdd-{ph}")
        if entry is None:
            continue
        table.add_row(
            f"sdd-{ph}",
            str(entry.get("model") or "-"),
            str(entry.get("thinking") or "-"),
        )
    console.print(table)


def _print_codex_followup() -> None:
    err_console.print(
        "[yellow]NOTE: direct ~/.codex/*.config.toml writing is intentionally[/yellow]"
    )
    err_console.print(
        "[yellow]unsupported: gentle-ai sync regenerates those profiles, so any[/yellow]"
    )
    err_console.print(
        "[yellow]hand edit is overwritten; whole-session tier profiles also do NOT[/yellow]"
    )
    err_console.print(
        "[yellow]reach spawned sub-agents (sdd-orchestrator.md:190). The state file[/yellow]"
    )
    err_console.print(
        "[yellow]is the only durable Codex write surface (research doc §6).[/yellow]"
    )


@codex_app.callback()
def integrate_codex(
    ctx: typer.Context,
    phase: str | None = typer.Option(None, "--phase", help="SDD phase."),
    model: str | None = typer.Option(None, "--model", help="provider/model."),
    effort: str | None = typer.Option(None, "--effort", help="Reasoning effort level."),
    carril: str | None = typer.Option(
        None,
        "--carril",
        help="Also write the Codex carril assignment (strong/mid/cheap).",
    ),
    state_path: str | None = typer.Option(
        None,
        "--state-path",
        help="Gentle AI state file path (default: ~/.gentle-ai/state.json).",
    ),
    config_path: str | None = typer.Option(None, "--config", help="Path to router.yaml."),
    data_dir: str | None = typer.Option(None, "--data-dir", help="Override data directory."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the diff, write nothing."),
    create: bool = typer.Option(False, "--create", help="Create the state file if missing."),
    verify: bool = typer.Option(
        False, "--verify", help="Re-read the state file and confirm the assignment round-trips."
    ),
) -> None:
    """Set CodexPhaseModelAssignments["sdd_<phase>"] in the Gentle AI state file.

    Codex phase agents are addressed with underscore transport identifiers
    (spawn_agent task_name="sdd_<phase>"), so keys use underscores. Effort is
    clamped into the Codex vocabulary (low|medium|high|xhigh); clamps are
    reported as codex_effort_mapped:* reason codes. Direct ~/.codex TOML
    writing is intentionally unsupported (overwritten by gentle-ai sync).
    """
    if ctx.invoked_subcommand is not None:
        return  # `codex rollback ...` / `codex status ...` — handled by their commands
    required = (("--phase", phase), ("--model", model), ("--effort", effort))
    missing = [name for name, value in required if value is None]
    if missing:
        err_console.print(f"[red]error: missing required option(s): {', '.join(missing)}[/red]")
        raise typer.Exit(code=2)
    name = phase.removeprefix("sdd-")
    if name not in CANONICAL_PHASES:
        err_console.print(
            f"[red]error: unknown phase '{phase}' "
            f"(expected one of: {', '.join(CANONICAL_PHASES)})[/red]"
        )
        raise typer.Exit(code=2)
    if carril is not None and carril.removeprefix("sdd-") not in cx.CARRILES:
        err_console.print(
            f"[red]error: unknown carril '{carril}' "
            f"(expected one of: {', '.join(cx.CARRILES)})[/red]"
        )
        raise typer.Exit(code=2)
    if dry_run and verify:
        err_console.print("[red]error: --verify requires a real write (not --dry-run)[/red]")
        raise typer.Exit(code=2)
    config, _ = _load_ctx(config_path, data_dir)
    path = cx.resolve_state_path(state_path)
    try:
        result = cx.apply_assignment(
            path, name, model, effort, carril=carril, create=create, dry_run=dry_run,
            backup=config.integrate.backup,
        )
    except cx.AdapterError as exc:
        err_console.print(f"[red]error: {exc}[/red]")
        raise typer.Exit(code=2) from exc
    for code in result.reason_codes:
        err_console.print(f"[yellow]reason: {code}[/yellow]")
    if result.diff:
        console.print(result.diff, highlight=False)
    if result.wrote:
        console.print(f"[green]wrote {result.path}[/green]")
        console.print(
            f'{result.phase_key} = {{"provider_id": {json.dumps(result.provider_id)}, '
            f'"model_id": {json.dumps(result.model_id)}, '
            f'"effort": {json.dumps(result.codex_effort)}}}'
        )
        if result.backup_path:
            console.print(f"[dim]backup: {result.backup_path}[/dim]")
        _print_codex_followup()
    else:
        console.print("[dim]dry-run: nothing written[/dim]")
    if verify:
        ok, observed = cx.verify_assignment(path, name, model, effort)
        if ok:
            console.print(
                f"[green]verify OK: {result.phase_key} -> "
                f"{model}#{effort} round-trips in {path}[/green]"
            )
        else:
            err_console.print(
                f"[red]verify MISMATCH: expected {model}#{effort}, "
                f"observed {observed!r}[/red]"
            )
            raise typer.Exit(code=1)


@codex_app.command("rollback")
def integrate_codex_rollback(
    state_path: str | None = typer.Option(
        None,
        "--state-path",
        help="Gentle AI state file path (default: ~/.gentle-ai/state.json).",
    ),
) -> None:
    """Restore the latest router backup of the Gentle AI state file."""
    path = cx.resolve_state_path(state_path)
    restored = cx.rollback(path)
    if restored is None:
        err_console.print(f"[yellow]no backup found for {path}[/yellow]")
        raise typer.Exit(code=1)
    console.print(f"[green]restored {path} from {restored}[/green]")


@codex_app.command("status")
def integrate_codex_status(
    state_path: str | None = typer.Option(
        None,
        "--state-path",
        help="Gentle AI state file path (default: ~/.gentle-ai/state.json).",
    ),
) -> None:
    """Show current Codex phase + carril assignments from the state file."""
    path = cx.resolve_state_path(state_path)
    assignments = cx.read_assignments(path)
    table = Table(title=f"codex assignments ({path})")
    table.add_column("key")
    table.add_column("provider")
    table.add_column("model")
    table.add_column("effort")
    for ph in CANONICAL_PHASES:
        key = cx.phase_key(ph)
        entry = assignments["phases"].get(key)
        if entry is None:
            continue
        table.add_row(
            key,
            str(entry.get("provider_id") or "-"),
            str(entry.get("model_id") or "-"),
            str(entry.get("effort") or "-"),
        )
    for carril in cx.CARRILES:
        key = cx.carril_key(carril)
        entry = assignments["carriles"].get(key)
        if entry is None:
            continue
        table.add_row(
            key,
            str(entry.get("provider_id") or "-"),
            str(entry.get("model_id") or "-"),
            str(entry.get("effort") or "-"),
        )
    console.print(table)


@shim_app.command("ingest")
def shim_ingest(
    config_path: str | None = typer.Option(None, "--config", help="Path to router.yaml."),
    data_dir: str | None = typer.Option(None, "--data-dir", help="Override data directory."),
    db: str | None = typer.Option(None, "--db", help="Telemetry SQLite path override."),
) -> None:
    """Ingest execution JSON-lines from stdin into the telemetry store."""
    config, _ = _load_ctx(config_path, data_dir)
    store = telemetry_shim.ShimStore(db or config.telemetry_url)
    store.init_schema()
    with store.session() as session:
        counts = store.ingest_jsonl(session, sys.stdin)
    console.print(json.dumps(counts))


@shim_app.command("seed")
def shim_seed(
    dataset: str = typer.Option(..., "--dataset", help="Written dataset directory to seed from."),
    db: str | None = typer.Option(None, "--db", help="Telemetry SQLite path override."),
    config_path: str | None = typer.Option(None, "--config", help="Path to router.yaml."),
    data_dir: str | None = typer.Option(None, "--data-dir", help="Override data directory."),
) -> None:
    """Seed the telemetry shim from empirical benchmark observations in a dataset.

    Converts empirical-provenance examples into scored, bootstrap-stamped
    (decision, execution) rows so the telemetry bridge has measured outcomes
    to learn from. Idempotent: deterministic ids make re-runs no-ops. Prints
    a machine-readable JSON summary {seeded, skipped, total}.
    """
    from gentle_ai_model_router.integration.empirical_seed import (
        EmpiricalSeedError,
        seed_empirical_dataset,
    )

    config, _ = _load_ctx(config_path, data_dir)
    try:
        summary = seed_empirical_dataset(dataset, db or config.telemetry_url)
    except EmpiricalSeedError as exc:
        err_console.print(f"[red]error: {exc}[/red]")
        raise typer.Exit(code=2) from exc
    # Plain print: the JSON payload must survive piping unwrapped/unmangled.
    print(json.dumps(summary))


# --------------------------------------------------------------------------- #
# Phase 4: bandit policy loop — outcomes, rewards, bandit inspection
# --------------------------------------------------------------------------- #


def _open_shim(config: RouterConfig, db: str | None) -> telemetry_shim.ShimStore:
    url = db or config.telemetry_url
    if "://" not in url:
        url = f"sqlite:///{url}"  # plain SQLite path → SQLAlchemy URL
    store = telemetry_shim.ShimStore(url)
    store.init_schema()
    return store


@bandit_app.command("report")
def bandit_report(
    phase: str | None = typer.Option(
        None, "--phase", help="Filter aggregates to one SDD phase (default: all)."
    ),
    db: str | None = typer.Option(None, "--db", help="Telemetry SQLite path override."),
    config_path: str | None = typer.Option(None, "--config", help="Path to router.yaml."),
    data_dir: str | None = typer.Option(None, "--data-dir", help="Override data directory."),
) -> None:
    """Show per-(phase, model, effort) reward aggregates from the telemetry store."""
    config, _ = _load_ctx(config_path, data_dir)
    store = _open_shim(config, db)
    try:
        with store.session() as session:
            aggregates = aggregate_rewards(compute_rewards(session))
    except RewardError as exc:
        err_console.print(f"[red]error: {exc}[/red]")
        raise typer.Exit(code=2) from exc
    if phase is not None:
        name = phase.removeprefix("sdd-")
        aggregates = [a for a in aggregates if a.phase == name]
    if not aggregates:
        console.print("no reward data (cold start: the bandit falls back to rank_candidates)")
        return
    table = Table(title=f"bandit reward aggregates ({bandit_version(config.bandit)})")
    for column in (
        "phase", "model", "effort", "executions", "success", "mean reward",
        "mean tokens", "tokens/success", "win rate",
    ):
        numeric = column in {"executions", "success", "mean reward", "mean tokens",
                             "tokens/success", "win rate"}
        table.add_column(
            column,
            justify="right" if numeric else "left",
            no_wrap=column in {"model", "effort", "phase"},
        )
    for agg in aggregates:
        table.add_row(
            agg.phase,
            agg.model,
            agg.effort,
            str(agg.executions),
            f"{agg.success_rate:.3f}",
            f"{agg.mean_reward:.4f}",
            f"{agg.mean_total_tokens:.0f}",
            f"{agg.tokens_per_success:.0f}" if agg.tokens_per_success is not None else "-",
            f"{agg.win_rate:.3f}" if agg.win_rate is not None else "-",
        )
    bandit_console.print(table)


@bandit_app.command("update")
def bandit_update(
    phase: str | None = typer.Option(
        None, "--phase", help="Restrict outcome scoring to one SDD phase."
    ),
    db: str | None = typer.Option(None, "--db", help="Telemetry SQLite path override."),
    config_path: str | None = typer.Option(None, "--config", help="Path to router.yaml."),
    data_dir: str | None = typer.Option(None, "--data-dir", help="Override data directory."),
) -> None:
    """Recompute NULL execution outcomes via the rubric; print the bandit version.

    Idempotent: executions already carrying task_success are never touched.
    """
    config, _ = _load_ctx(config_path, data_dir)
    store = _open_shim(config, db)
    with store.session() as session:
        counts = apply_outcomes(session, phase=phase)
    table = Table(title="bandit update")
    table.add_row("outcomes_scored", str(counts["scored"]))
    table.add_row("bandit_version", bandit_version(config.bandit))
    console.print(table)


@app.command()
def feedback(
    db: str | None = typer.Option(None, "--db", help="Telemetry SQLite path override."),
    config_path: str | None = typer.Option(None, "--config", help="Path to router.yaml."),
    data_dir: str | None = typer.Option(None, "--data-dir", help="Override data directory."),
) -> None:
    """Ingest execution JSON-lines from stdin and score their outcomes.

    Feeds the bandit loop: ingest (idempotent upsert on execution_id) then
    rubric-scoring of any rows still missing task_success/quality_score.
    """
    config, _ = _load_ctx(config_path, data_dir)
    store = _open_shim(config, db)
    with store.session() as session:
        ingest_counts = store.ingest_jsonl(session, sys.stdin)
        outcome_counts = apply_outcomes(session)
    # Plain print: the JSON payload must survive piping unwrapped/unmangled
    # (rich would word-wrap it at the console width and parse markup).
    print(
        json.dumps(
            {
                "ingested": ingest_counts,
                "outcomes": outcome_counts,
                "bandit_version": bandit_version(config.bandit),
            }
        )
    )


# --------------------------------------------------------------------------- #
# Phase 5: telemetry-driven threshold tuning (pure tuner + explicit applier)
# --------------------------------------------------------------------------- #


def _threshold_proposals(
    config: RouterConfig,
    db: str | None,
    phase: str | None,
    success_floor: float,
) -> list:
    """Compute tuner proposals from the shim store (shared by propose/apply)."""
    from gentle_ai_model_router.router.policy import normalize_phase
    from gentle_ai_model_router.router.threshold_tune import propose_thresholds

    store = _open_shim(config, db)
    try:
        with store.session() as session:
            aggregates = aggregate_rewards(compute_rewards(session))
    except RewardError as exc:
        err_console.print(f"[red]error: {exc}[/red]")
        raise typer.Exit(code=2) from exc

    grouped: dict[str, list] = {}
    for agg in aggregates:
        grouped.setdefault(agg.phase.removeprefix("sdd-"), []).append(agg)
    if phase is not None:
        try:
            name = normalize_phase(phase)
        except PolicyError as exc:
            err_console.print(f"[red]error: {exc}[/red]")
            raise typer.Exit(code=2) from exc
        grouped = {name: grouped.get(name, [])}
    return propose_thresholds(grouped, success_floor=success_floor, config=config)


@thresholds_app.command("propose")
def thresholds_propose(
    phase: str | None = typer.Option(
        None, "--phase", help="One SDD phase (default: every phase with telemetry)."
    ),
    success_floor: float = typer.Option(
        0.8, "--success-floor", help="Minimum pooled success rate to admit an effort level."
    ),
    db: str | None = typer.Option(None, "--db", help="Telemetry SQLite path override."),
    config_path: str | None = typer.Option(None, "--config", help="Path to router.yaml."),
    data_dir: str | None = typer.Option(None, "--data-dir", help="Override data directory."),
) -> None:
    """Propose per-phase threshold_quality values from measured rewards (read-only).

    For each phase, the cheapest effort level whose pooled success rate meets
    --success-floor with at least bandit.min_executions_before_exploit
    executions defines the proposal; the proposed value is the policy's own
    quality curve applied to that measured success rate. Phases below the
    evidence bar are reported as insufficient_evidence and never changed.
    Prints the proposals as JSON; writes nothing.
    """
    from dataclasses import asdict

    from gentle_ai_model_router.router.threshold_tune import ThresholdTuneError

    config, _ = _load_ctx(config_path, data_dir)
    try:
        proposals = _threshold_proposals(config, db, phase, success_floor)
    except ThresholdTuneError as exc:
        err_console.print(f"[red]error: {exc}[/red]")
        raise typer.Exit(code=2) from exc
    # Plain print: the JSON payload must survive piping unwrapped/unmangled.
    print(json.dumps([asdict(p) for p in proposals], indent=2))


def _resolve_router_yaml(config_path: str | None) -> Path:
    """The real router.yaml to edit — never the bundled example fallback."""
    from gentle_ai_model_router.router.config import find_config_file

    path = find_config_file(config_path)
    if path is None or path.name == "router.yaml.example":
        err_console.print(
            "[red]error: no router.yaml found (pass --config <path>); "
            "refusing to edit the bundled example[/red]"
        )
        raise typer.Exit(code=2)
    return path


@thresholds_app.command("apply")
def thresholds_apply(
    phase: str | None = typer.Option(
        None, "--phase", help="One SDD phase (default: every phase with proposals)."
    ),
    success_floor: float = typer.Option(
        0.8, "--success-floor", help="Minimum pooled success rate to admit an effort level."
    ),
    db: str | None = typer.Option(None, "--db", help="Telemetry SQLite path override."),
    config_path: str | None = typer.Option(None, "--config", help="Path to router.yaml."),
    data_dir: str | None = typer.Option(None, "--data-dir", help="Override data directory."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the diff, write nothing."),
    create: bool = typer.Option(
        False, "--create", help="Add phases missing from router.yaml instead of refusing."
    ),
) -> None:
    """Apply upgrade/downgrade threshold proposals to router.yaml (explicit write).

    Edits ONLY ``phases.<name>.threshold_quality`` for phases whose proposal
    carries sufficient evidence (kind=upgrade/downgrade); uphold and
    insufficient_evidence proposals are reported as skipped and never touch
    the file. Phases absent from router.yaml are refused unless --create.
    The write is backup-first (router.yaml.router-backup-<ts>) and atomic
    (tmp + os.replace). Always propose with ``thresholds propose`` first.
    """
    from gentle_ai_model_router.integration.router_yaml_adapter import (
        RouterYamlError,
        apply_threshold_proposals,
    )
    from gentle_ai_model_router.router.threshold_tune import ThresholdTuneError

    config, _ = _load_ctx(config_path, data_dir)
    try:
        proposals = _threshold_proposals(config, db, phase, success_floor)
    except ThresholdTuneError as exc:
        err_console.print(f"[red]error: {exc}[/red]")
        raise typer.Exit(code=2) from exc

    path = _resolve_router_yaml(config_path)
    try:
        result = apply_threshold_proposals(path, proposals, create=create, dry_run=dry_run)
    except RouterYamlError as exc:
        err_console.print(f"[red]error: {exc}[/red]")
        raise typer.Exit(code=2) from exc

    if result.diff:
        console.print(result.diff, highlight=False)
    for name, (old, new) in result.applied.items():
        console.print(f"[green]applied {name}: {old:.3f} -> {new:.3f}[/green]")
    for name, reason in result.skipped.items():
        err_console.print(f"[yellow]skipped {name}: {reason}[/yellow]")
    if result.wrote:
        console.print(f"[green]wrote {result.path}[/green]")
        if result.backup_path:
            console.print(f"[dim]backup: {result.backup_path}[/dim]")
    else:
        console.print("[dim]dry-run: nothing written[/dim]")


# --------------------------------------------------------------------------- #
# Phase 2: dataset builder + ModernBERT ranker + offline evaluation + escalation
# --------------------------------------------------------------------------- #


@app.command()
def build_dataset(
    name: str = typer.Option(..., "--name", help="Dataset name."),
    train_end: str = typer.Option(
        ..., "--train-end", help="Train cutoff (ISO date); earlier = train."
    ),
    val_end: str | None = typer.Option(
        None, "--val-end", help="Validation cutoff (ISO date); earlier = validation."
    ),
    as_of: str | None = typer.Option(
        None, "--as-of", help="Knowledge cutoff (ISO date); newer rows = build error."
    ),
    pair_margin: float | None = typer.Option(
        None, "--pair-margin", help="Pairwise margin threshold (default 0.05)."
    ),
    max_pairs: int = typer.Option(50, "--max-pairs", help="Cap pairs per group."),
    threshold_penalty: float = typer.Option(
        0.0,
        "--threshold-penalty",
        help="Penalty applied to candidates below phase threshold_quality.",
    ),
    hard_threshold: bool = typer.Option(
        False,
        "--hard-threshold",
        help="Zero out utility for candidates below phase threshold_quality.",
    ),
    cost_weight: float | None = typer.Option(
        None, "--cost-weight", help="Weight of token cost penalty (default 0.5)."
    ),
    phase: Annotated[
        list[str] | None,
        typer.Option("--phase", help="Phase to include (repeatable). Default: all."),
    ] = None,
    include_empirical: bool = typer.Option(
        True, "--empirical/--no-empirical", help="Include empirical benchmark prompts in dataset."
    ),
    max_empirical_tasks: int = typer.Option(
        30, "--max-empirical-tasks", help="Max empirical tasks per phase."
    ),
    telemetry_db: str | None = typer.Option(
        None,
        "--telemetry-db",
        help="Enable the telemetry bridge: shim SQLite DB path (an empty value "
        "falls back to the config telemetry_url). Off by default.",
    ),
    config_path: str | None = typer.Option(None, "--config", help="Path to router.yaml."),
    data_dir: str | None = typer.Option(None, "--data-dir", help="Override data directory."),
) -> None:
    """Build a DatasetV1 from registry priors (labels are bootstrap priors!)."""
    from datetime import date

    from gentle_ai_model_router.dataset.builder import (
        DatasetBuilderConfig,
        DatasetBuildError,
        build_examples,
        write_dataset,
    )

    config, store = _load_ctx(config_path, data_dir)
    builder_kwargs: dict = {
        "name": name,
        "train_end": date.fromisoformat(train_end),
        "val_end": date.fromisoformat(val_end) if val_end else None,
        "as_of": date.fromisoformat(as_of) if as_of else None,
        "max_pairs_per_group": max_pairs,
        "phases": phase or list(CANONICAL_PHASES),
        "threshold_penalty": threshold_penalty,
        "hard_threshold": hard_threshold,
        "include_empirical_benchmarks": include_empirical,
        "max_empirical_tasks_per_phase": max_empirical_tasks,
    }
    if pair_margin is not None:
        builder_kwargs["pair_margin"] = pair_margin
    if cost_weight is not None:
        builder_kwargs["cost_weight"] = cost_weight
    try:
        builder = DatasetBuilderConfig(**builder_kwargs)
    except ValueError as exc:
        err_console.print(f"[red]error: invalid date: {exc}[/red]")
        raise typer.Exit(code=2) from exc

    engine, _effective = registry_db.get_engine_with_fallback(
        config.database_url, config.sqlite_fallback_url
    )
    registry_db.init_schema(engine)
    try:
        with registry_db.Session(engine) as session:
            dataset = build_examples(
                session, config, builder, store=store, telemetry_db=telemetry_db
            )
            out_dir = write_dataset(dataset, config.data_dir, builder=builder)
    except DatasetBuildError as exc:
        err_console.print(f"[red]error: {exc}[/red]")
        raise typer.Exit(code=2) from exc

    manifest = json.loads((out_dir / "manifest.json").read_text())
    table = Table(title=f"dataset {dataset.name} v{dataset.version}")
    table.add_row("path", str(out_dir))
    table.add_row("examples", str(manifest["example_count"]))
    table.add_row("pairs", str(manifest["pair_count"]))
    table.add_row("split_counts", json.dumps(manifest["split_counts"]))
    table.add_row("pair_split_counts", json.dumps(manifest["pair_split_counts"]))
    if telemetry_db is not None:
        stats = manifest.get("telemetry", {})
        table.add_row("telemetry", json.dumps(stats))
        if stats.get("skipped"):
            err_console.print(
                f"[yellow]WARNING: skipped {stats['skipped']} invalid telemetry "
                "row(s) — see build log.[/yellow]"
            )
        if stats.get("emitted"):
            err_console.print(
                f"[dim]telemetry: {stats['emitted']} measured row(s) "
                f"(skipped {stats.get('skipped', 0)} invalid).[/dim]"
            )
    console.print(table)
    err_console.print("[yellow]WARNING: labels are bootstrap priors, not ground truth —[/yellow]")
    err_console.print("[yellow]see manifest.json label_provenance_statement.[/yellow]")


@app.command()
def train(
    dataset_path: str = typer.Option(..., "--dataset", help="Dataset version directory."),
    objective: str | None = typer.Option(None, "--objective", help="pointwise | pairwise."),
    model_name: str | None = typer.Option(
        None, "--model-name", help="HF encoder model (default answerdotai/ModernBERT-base)."
    ),
    output_dir: str | None = typer.Option(None, "--output-dir", help="Checkpoint root."),
    epochs: float | None = typer.Option(None, "--epochs"),
    batch_size: int | None = typer.Option(None, "--batch-size"),
    gradient_accumulation_steps: int | None = typer.Option(
        None, "--gradient-accumulation-steps", help="Number of update steps to accumulate."
    ),
    seed: int | None = typer.Option(None, "--seed"),
    device: str | None = typer.Option(
        None, "--device", help="auto | cpu | cuda (default: auto = cuda if available)."
    ),
    telemetry_weight: float | None = typer.Option(
        None,
        "--telemetry-weight",
        help=(
            "Pointwise only: loss-weight multiplier for label_provenance='telemetry' "
            "examples (default: 1.0 = uniform, byte-identical to unweighted)."
        ),
    ),
    bf16: bool | None = typer.Option(
        None, "--bf16/--no-bf16", help="Enable bfloat16 mixed precision."
    ),
    max_vram_fraction: float | None = typer.Option(
        None, "--max-vram-fraction", help="Cap CUDA memory allocation fraction (e.g. 0.5)."
    ),
    progress: bool | None = typer.Option(
        None, "--progress/--no-progress", help="Show live progress bar."
    ),
    config_path: str | None = typer.Option(None, "--config", help="Path to router.yaml."),
    data_dir: str | None = typer.Option(None, "--data-dir", help="Override data directory."),
) -> None:
    """Train the ModernBERT ranker (requires: pip install .[train])."""
    from gentle_ai_model_router.training.train import train as run_training

    config, _ = _load_ctx(config_path, data_dir)
    overrides: dict = {}
    for key, value in (
        ("objective", objective),
        ("model_name", model_name),
        ("output_dir", output_dir),
        ("epochs", epochs),
        ("batch_size", batch_size),
        ("gradient_accumulation_steps", gradient_accumulation_steps),
        ("seed", seed),
        ("device", device),
        ("telemetry_weight", telemetry_weight),
        ("bf16", bf16),
        ("max_vram_fraction", max_vram_fraction),
        ("disable_tqdm", not progress if progress is not None else None),
    ):
        if value is not None:
            overrides[key] = value
    training_cfg = config.training.model_copy(update=overrides)
    if training_cfg.telemetry_weight <= 0.0:
        err_console.print(
            f"[red]error: --telemetry-weight must be > 0 "
            f"(got {training_cfg.telemetry_weight})[/red]"
        )
        raise typer.Exit(code=2)
    if training_cfg.objective not in {"pointwise", "pairwise"}:
        err_console.print(
            f"[red]error: unknown objective '{training_cfg.objective}' "
            "(expected pointwise | pairwise)[/red]"
        )
        raise typer.Exit(code=2)
    try:
        out_dir = run_training(config, training_cfg, dataset_path)
    except (RuntimeError, ValueError) as exc:
        err_console.print(f"[red]error: {escape(str(exc))}[/red]")
        raise typer.Exit(code=2) from exc
    metrics = json.loads((out_dir / "metrics.json").read_text())
    table = Table(title=f"checkpoint {out_dir}")
    table.add_row("objective", metrics["objective"])
    table.add_row("train_rows", str(metrics["train_rows"]))
    table.add_row("train_loss", f"{metrics['train_loss']:.6f}")
    table.add_row("label_provenance", metrics["label_provenance"])
    console.print(table)
    err_console.print(
        "[yellow]WARNING: trained on bootstrap PRIOR labels — not ground truth.[/yellow]"
    )


@app.command()
def evaluate(
    dataset_path: str = typer.Option(..., "--dataset", help="Dataset version directory."),
    checkpoint: str | None = typer.Option(None, "--checkpoint", help="Ranker checkpoint dir."),
    baselines_only: bool = typer.Option(
        False, "--evaluate-baselines-only", help="No checkpoint; reference routers only."
    ),
    output: str | None = typer.Option(None, "--output", help="Write metrics.json here."),
    batch_size: int | None = typer.Option(
        None, "--batch-size", help="Ranker scoring batch size (default: evaluation.batch_size)."
    ),
    device: str | None = typer.Option(
        None, "--device", help="auto | cpu | cuda (default: auto = cuda if available)."
    ),
    config_path: str | None = typer.Option(None, "--config", help="Path to router.yaml."),
    data_dir: str | None = typer.Option(None, "--data-dir", help="Override data directory."),
) -> None:
    """Offline evaluation: ranking + business metrics per split."""
    from gentle_ai_model_router.training.evaluate import evaluate_dataset

    config, _ = _load_ctx(config_path, data_dir)
    # The baseline-policy reference needs the registry in every mode — without
    # it the most important reference silently disappears from the comparison.
    engine, _eff = registry_db.get_engine_with_fallback(
        config.database_url, config.sqlite_fallback_url
    )
    registry_db.init_schema(engine)
    try:
        session_ctx = registry_db.Session(engine) if engine is not None else _nullcontext()
        with session_ctx as session:
            results = evaluate_dataset(
                dataset_path,
                config,
                session=session,
                checkpoint=None if baselines_only else checkpoint,
                output_path=output,
                batch_size=batch_size,
                device=device,
            )
    except (RuntimeError, ValueError) as exc:
        err_console.print(f"[red]error: {escape(str(exc))}[/red]")
        raise typer.Exit(code=2) from exc

    for split, routers in results["splits"].items():
        table = Table(title=f"split: {split}")
        table.add_column("router")
        for key in ("top1_accuracy", "top3_recall", "mrr", "ndcg@3", "ndcg@5",
                    "tokens_per_task", "tokens_per_success", "success_rate",
                    "routing_regret"):
            table.add_column(key, justify="right")
        for router, metrics in routers.items():
            table.add_row(
                router,
                f"{metrics['top1_accuracy']:.3f}",
                f"{metrics['top3_recall']:.3f}",
                f"{metrics['mrr']:.3f}",
                f"{metrics['ndcg@3']:.3f}",
                f"{metrics['ndcg@5']:.3f}",
                f"{metrics['tokens_per_task']:.0f}",
                f"{metrics['tokens_per_success']:.0f}",
                f"{metrics['success_rate']:.3f}",
                f"{metrics['routing_regret']:.4f}",
            )
        console.print(table)
    chooser_errors = results.get("chooser_errors") or []
    if chooser_errors:
        err_console.print(
            f"[yellow]{len(chooser_errors)} chooser error(s): see metrics json[/yellow]"
        )
    err_console.print(
        "[yellow]Caveat: labels are bootstrap priors — compare routers relative to"
        " each other only.[/yellow]"
    )


# --------------------------------------------------------------------------- #
# Model promotion: candidate -> evaluate -> compare -> promote (never
# auto-replace the active router; see docs/training.md "Promotion workflow")
# --------------------------------------------------------------------------- #


def _print_promotion_comparison(comparison: Any, *, dry_run: bool) -> None:
    from gentle_ai_model_router.training.promote import INFO_METRICS

    table = Table(
        title=(
            f"promotion comparison (split: {comparison.split}) — "
            f"primary {comparison.metric}, lower is better"
        )
    )
    table.add_column("metric")
    table.add_column("promoted", justify="right")
    table.add_column("candidate", justify="right")
    table.add_column("delta", justify="right")
    table.add_column("note")
    promoted_values = comparison.promoted_values or {}
    promoted_label = "- (first promotion)" if comparison.promoted_values is None else None

    def _row(name: str, note: str) -> None:
        prom = promoted_values.get(name)
        cand = comparison.candidate_values.get(name)
        table.add_row(
            name,
            f"{prom:.4f}" if prom is not None else (promoted_label or "missing"),
            f"{cand:.4f}" if cand is not None else "missing",
            f"{cand - prom:+.4f}" if (prom is not None and cand is not None) else "-",
            note,
        )

    primary_note = "primary (lower better)"
    if comparison.promoted_values is not None:
        delta = (
            comparison.candidate_values[comparison.metric]
            - comparison.promoted_values[comparison.metric]
        )
        primary_note += " — improved" if delta < 0 else " — NOT improved"
    _row(comparison.metric, primary_note)
    for guardrail in comparison.guardrails:
        _row(
            guardrail.metric,
            f"guardrail (eps abs) — {'holds' if guardrail.ok else 'REGRESSED'}",
        )
    table.caption = (
        "epsilon = absolute guardrail tolerance; ranking metrics "
        f"({', '.join(INFO_METRICS)}) are informational"
    )
    for name in INFO_METRICS:
        _row(name, "informational")
    console.print(table)
    for reason in comparison.reasons:
        err_console.print(f"[dim]{escape(reason)}[/dim]")
    decision = comparison.decision.upper()
    if dry_run:
        console.print(f"[bold]decision: {decision}[/bold] (dry-run — nothing written)")
    else:
        console.print(f"[bold]decision: {decision}[/bold]")


@app.command()
def promote(
    candidate: str | None = typer.Option(
        None, "--candidate", help="Candidate checkpoint dir (models/modernbert-router/v<N>)."
    ),
    dataset: str | None = typer.Option(
        None, "--dataset", help="Re-evaluate the candidate on this dataset before comparing."
    ),
    metric: str = typer.Option(
        "tokens_per_success", "--metric", help="Primary metric (lower is better)."
    ),
    epsilon: float = typer.Option(
        0.02, "--epsilon", help="Max absolute guardrail regression allowed (default 0.02)."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Decide and print, write nothing."),
    status: bool = typer.Option(False, "--status", help="Print the current promoted record."),
    thresholds_path: str | None = typer.Option(
        None,
        "--thresholds",
        help=(
            "Promote a thresholds artifact instead of a checkpoint: JSON file with "
            "threshold-tune proposals (the `router thresholds propose` output, a list "
            "or {'proposals': [...], 'bandit_version': ...})."
        ),
    ),
    models_dir: str = typer.Option(
        "models", "--models-dir", help="Models root that holds the promoted/ record."
    ),
    config_path: str | None = typer.Option(None, "--config", help="Path to router.yaml."),
    data_dir: str | None = typer.Option(None, "--data-dir", help="Override data directory."),
) -> None:
    """Promote an artifact: a checkpoint (if it beats the active router on
    eval metrics) or an evidence-backed thresholds payload.

    NEVER auto-replaces the active record: candidate -> evaluate -> compare ->
    promote (checkpoints), or proposals -> evidence-filter -> promote
    (thresholds, no metric comparison — evidence is recorded). Exit codes:
    0 = promoted (or dry-run), 1 = comparison says keep, 2 = usage/validation
    errors.
    """
    from gentle_ai_model_router.training import promote as promote_mod

    models_root = Path(models_dir)
    if status:
        if candidate is not None or dataset is not None or thresholds_path is not None:
            err_console.print("[red]error: --status takes no other action options[/red]")
            raise typer.Exit(code=2)
        current = promote_mod.load_promoted(models_root)
        if current is None:
            console.print("none")
            return
        _print_promotion_status(current, models_root)
        return

    if thresholds_path is not None:
        if candidate is not None or dataset is not None:
            err_console.print(
                "[red]error: --thresholds is mutually exclusive with --candidate/--dataset[/red]"
            )
            raise typer.Exit(code=2)
        _promote_thresholds_cli(promote_mod, models_root, thresholds_path, dry_run)
        return

    if candidate is None:
        err_console.print(
            "[red]error: --candidate or --thresholds is required (or use --status)[/red]"
        )
        raise typer.Exit(code=2)

    config = None
    if dataset is not None:
        config, _ = _load_ctx(config_path, data_dir)
    try:
        outcome = promote_mod.promote_checkpoint(
            candidate,
            models_dir=models_root,
            dataset=dataset,
            config=config,
            metric=metric,
            epsilon=epsilon,
            dry_run=dry_run,
        )
    except promote_mod.PromotionError as exc:
        err_console.print(f"[red]error: {escape(str(exc))}[/red]")
        raise typer.Exit(code=2) from exc
    except (RuntimeError, ValueError) as exc:
        err_console.print(f"[red]error: {escape(str(exc))}[/red]")
        raise typer.Exit(code=2) from exc

    _print_promotion_comparison(outcome.comparison, dry_run=dry_run)
    if outcome.comparison.decision == "keep":
        err_console.print("[yellow]current router stays active[/yellow]")
        raise typer.Exit(code=1)
    if dry_run:
        return
    assert outcome.record is not None  # wrote=True implies a record
    console.print(
        f"[green]promoted {outcome.record['promoted_checkpoint']} "
        f"-> {outcome.record_path}[/green]"
    )


def _print_promotion_status(current: Any, models_root: Path) -> None:
    """--status output for the active record, per artifact kind."""
    from gentle_ai_model_router.training import promote as promote_mod

    record = current["record"]
    kind = promote_mod.artifact_kind_of(record)
    if kind == promote_mod.ARTIFACT_KIND_THRESHOLDS:
        table = Table(title=f"promoted thresholds ({promote_mod.promoted_dir(models_root)})")
        table.add_column("phase")
        table.add_column("threshold", justify="right")
        table.add_column("kind")
        table.add_column("executions", justify="right")
        table.add_column("success_rate", justify="right")
        for phase, value in record.get("thresholds", {}).items():
            ev = record.get("evidence", {}).get(phase, {})
            success_rate = ev.get("success_rate")
            table.add_row(
                phase,
                f"{value:.3f}",
                str(ev.get("kind", "-")),
                str(ev.get("executions", 0)),
                f"{success_rate:.3f}" if success_rate is not None else "-",
            )
        table.caption = (
            f"source: {record.get('source', '-')} | promoted_at: "
            f"{record.get('promoted_at', '-')} | promoted_by: {record.get('promoted_by', '-')}"
        )
        console.print(table)
        return
    table = Table(title=f"promoted router ({promote_mod.promoted_dir(models_root)})")
    for key in (
        "artifact_kind",
        "promoted_checkpoint",
        "metrics_path",
        "promotion_reason",
        "promoted_at",
        "git_commit",
        "promoted_by",
    ):
        table.add_row(key, str(record.get(key, "-")))
    console.print(table)


def _promote_thresholds_cli(
    promote_mod: Any, models_root: Path, thresholds_path: str, dry_run: bool
) -> None:
    """Thresholds promotion from a proposals JSON file (`router thresholds propose` output).

    The file holds threshold-tune proposals (a list, or an object with
    ``proposals`` + optional ``bandit_version``); only upgrade/downgrade
    kinds promote. No metric comparison — evidence is recorded, never
    auto-replaced, atomic write, dry-run first.
    """
    import json

    path = Path(thresholds_path)
    if not path.is_file():
        err_console.print(f"[red]error: thresholds proposals file not found: {path}[/red]")
        raise typer.Exit(code=2)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        err_console.print(f"[red]error: invalid thresholds JSON in {path}: {exc}[/red]")
        raise typer.Exit(code=2) from exc
    if isinstance(doc, dict):
        proposals = doc.get("proposals", [])
        source = (
            f"bandit_version:{doc['bandit_version']}" if doc.get("bandit_version") else str(path)
        )
    elif isinstance(doc, list):
        proposals = doc
        source = str(path)
    else:
        err_console.print(
            f"[red]error: thresholds JSON must be a proposals list or an object "
            f"with 'proposals', got {type(doc).__name__}[/red]"
        )
        raise typer.Exit(code=2)

    payload, evidence = promote_mod.thresholds_payload_from_proposals(proposals)
    if not payload:
        err_console.print(
            "[yellow]no upgrade/downgrade proposals — nothing to promote[/yellow]"
        )
        raise typer.Exit(code=1)
    try:
        outcome = promote_mod.promote_checkpoint(
            None,
            models_dir=models_root,
            thresholds=payload,
            evidence=proposals,
            source=source,
            dry_run=dry_run,
        )
    except promote_mod.PromotionError as exc:
        err_console.print(f"[red]error: {escape(str(exc))}[/red]")
        raise typer.Exit(code=2) from exc

    if dry_run:
        console.print(
            f"[bold]decision: PROMOTE[/bold] (dry-run — nothing written) "
            f"thresholds for {len(payload)} phase(s) from {source}"
        )
        for phase, value in payload.items():
            console.print(f"  {phase}: {value:.3f}")
        return
    assert outcome.record is not None  # wrote=True implies a record
    console.print(
        f"[green]promoted thresholds ({len(payload)} phases) from {source} "
        f"-> {outcome.record_path}[/green]"
    )


def _format_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes / (1024 * 1024):.1f} MB"


@app.command("export-onnx")
@app.command("export_onnx", hidden=True)
def export_onnx(
    checkpoint: str = typer.Option(..., "--checkpoint", help="Path to checkpoint directory."),
    output: str | None = typer.Option(None, "--output", help="Output directory for ONNX models."),
    quantize: bool = typer.Option(
        True, "--quantize/--no-quantize", help="Also produce INT8 quantized model."
    ),
) -> None:
    """Export a trained ranker checkpoint to ONNX with optional INT8 dynamic quantization."""
    from gentle_ai_model_router.training.onnx_export import export_onnx as do_export

    try:
        model_path, quant_path = do_export(
            checkpoint=checkpoint,
            output_dir=output,
            quantize=quantize,
        )
    except Exception as exc:
        err_console.print(f"[red]error: {escape(str(exc))}[/red]")
        raise typer.Exit(code=1) from exc

    table = Table(title="ONNX Export Summary")
    table.add_column("Artifact", style="bold")
    table.add_column("Path")
    table.add_column("Size", justify="right")

    table.add_row("FP32 Model", str(model_path), _format_size(model_path.stat().st_size))
    if quant_path is not None and quant_path.exists():
        table.add_row(
            "INT8 Quantized Model", str(quant_path), _format_size(quant_path.stat().st_size)
        )

    console.print(table)


if __name__ == "__main__":
    main()
