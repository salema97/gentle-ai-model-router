"""FastAPI app factory for the deterministic routing policy.

Local-first server: no auth, localhost binding by default. Every decision is
computed per-request over the live registry session (so results never go
stale within a process), logged to the telemetry shim store when one is
wired, and fingerprinted with ``registry_hash`` + ``policy_version`` so two
identical requests against unchanged state yield byte-identical responses.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy.engine import Engine

from gentle_ai_model_router.api.schemas import (
    AlternativeOut,
    BatchExecutionIngestResponse,
    ExecutionIn,
    ExecutionIngestResponse,
    RouteRequest,
    RouteResponse,
    SystemOneMeta,
)
from gentle_ai_model_router.integration import telemetry_shim
from gentle_ai_model_router.integration.outcome import (
    OutcomeRubricError,
    score_execution,
)
from gentle_ai_model_router.registry import db as registry_db
from gentle_ai_model_router.registry.fingerprint import registry_fingerprint
from gentle_ai_model_router.router.analytics import compute_roi
from gentle_ai_model_router.router.bandit import (
    REASON_UCB,
    apply_bandit,
)
from gentle_ai_model_router.router.config import RouterConfig
from gentle_ai_model_router.router.decision import (
    CANONICAL_PHASES,
    Alternative,
    Decision,
    TaskContext,
)
from gentle_ai_model_router.router.neural import neural_rerank
from gentle_ai_model_router.router.policy import (
    PolicyError,
    full_policy_version,
    normalize_phase,
    rank_candidates,
)
from gentle_ai_model_router.router.reward import aggregate_rewards, compute_rewards
from gentle_ai_model_router.router.system_one import evaluate_system_one

logger = logging.getLogger(__name__)


def _decision_to_response(decision: Decision, registry_hash: str) -> RouteResponse:
    """Map a policy Decision to the wire contract (registry_hash attached)."""
    system_one_meta = None
    if decision.system_one is not None:
        system_one_meta = SystemOneMeta(**decision.system_one)

    return RouteResponse(
        model=decision.model,
        deployment=decision.deployment,
        effort=decision.effort,
        score=decision.score,
        alternatives=[
            AlternativeOut(model=a.model, deployment=a.deployment, effort=a.effort, score=a.score)
            for a in decision.alternatives
        ],
        reason_codes=list(decision.reason_codes),
        estimated_tokens=decision.estimated_tokens,
        estimated_cost=decision.estimated_cost,
        policy_version=decision.policy_version,
        registry_hash=registry_hash,
        confidence=decision.confidence,
        probabilities=decision.probabilities,
        system_one=system_one_meta,
    )


def _record_decision(
    shim_store: telemetry_shim.ShimStore | None,
    request: RouteRequest,
    decision: Decision,
    candidate_count: int,
) -> None:
    """Persist + log one routing decision (best effort, never blocks routing)."""
    logger.info(
        "route_decision task=%r phase=%s candidate_count=%s model=%s effort=%s "
        "score=%s alternatives=%s estimated_cost=%s policy_version=%s "
        "reason_codes=%s",
        request.task,
        decision.phase,
        candidate_count,
        decision.model,
        decision.effort,
        decision.score,
        len(decision.alternatives),
        decision.estimated_cost,
        decision.policy_version,
        list(decision.reason_codes),
    )
    if shim_store is None:
        return
    try:
        with shim_store.session() as session:
            shim_store.record_decision(
                session,
                phase=decision.phase,
                selected={
                    "model": decision.model,
                    "provider": decision.provider,
                    "deployment": decision.deployment,
                    "effort": decision.effort,
                    "score": decision.score,
                    "task": request.task,
                    "task_type": request.task_type,
                    "candidate_count": candidate_count,
                },
                alternatives=[
                    {
                        "model": a.model,
                        "provider": a.provider,
                        "deployment": a.deployment,
                        "effort": a.effort,
                        "score": a.score,
                        "quality": a.quality,
                        "estimated_tokens": a.estimated_tokens,
                    }
                    for a in decision.alternatives
                ],
                reason_codes=list(decision.reason_codes),
                estimated_tokens=decision.estimated_tokens,
                estimated_cost=decision.estimated_cost,
                policy_version=decision.policy_version,
            )
    except Exception as exc:  # logging must never break routing
        logger.warning("decision_log_failed error=%s", exc)


def _ranked_policy_payload(config: RouterConfig, engine: Engine) -> dict[str, Any]:
    """Full per-phase ranked policy over the current registry (cached)."""
    phases: dict[str, Any] = {}
    with registry_db.Session(engine) as session:
        for phase in CANONICAL_PHASES:
            try:
                ranking = rank_candidates(session, phase, config)
            except PolicyError as exc:
                phases[phase] = {"error": str(exc)}
                continue
            winner = ranking.candidates[0]
            phases[phase] = {
                "threshold": ranking.threshold,
                "weights": config.phase_config(phase).weights,
                "policy_version": ranking.policy_version,
                "candidate_count": len(ranking.candidates),
                "selected": {
                    "model": winner.model.canonical_id,
                    "provider": winner.provider.registry_key,
                    "deployment": winner.deployment.deployment_ref,
                    "effort": winner.variant.effort,
                    "score": winner.score,
                    "quality": winner.quality,
                    "estimated_tokens": winner.estimated_tokens,
                    "estimated_cost": winner.estimated_cost,
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
    return phases


def create_app(
    config: RouterConfig,
    engine: Engine,
    shim_store: telemetry_shim.ShimStore | None = None,
    ranker: Any | None = None,
) -> FastAPI:
    """Build the FastAPI app over a registry engine + router config.

    ``shim_store`` is optional: when None, decisions are structured-logged
    only. The per-phase policy payload (``GET /policy``) is computed once per
    ``registry_hash`` and invalidated automatically when the fingerprint
    changes.
    """
    app = FastAPI(title="gentle-ai-model-router", version="0.3.0")
    app.state.ranker = ranker
    # registry_hash -> full policy payload. Bounded: fingerprints are 16-hex,
    # and old entries are dropped whenever the hash rotates.
    policy_cache: dict[str, dict[str, Any]] = {}
    app.state.policy_cache = policy_cache

    def _current_hash() -> str:
        return registry_fingerprint(engine)

    @app.post("/route", response_model=RouteResponse)
    def route(request: RouteRequest) -> RouteResponse:
        """Select (model, deployment, effort) for one phase invocation."""
        try:
            phase = normalize_phase(request.phase)
        except PolicyError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        try:
            efforts = request.validate_efforts()
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        context = TaskContext(
            task_type=request.task_type,
            context_tokens=request.context_tokens,
            repo_features=request.repo_features or {},
        )
        try:
            with registry_db.Session(engine) as session:
                ranking = rank_candidates(
                    session,
                    phase,
                    config,
                    context,
                    allowed_models=set(request.available_models)
                    if request.available_models is not None
                    else None,
                    allowed_efforts=set(efforts) if efforts else None,
                )
                if app.state.ranker is not None and request.task:
                    ranking = neural_rerank(
                        session, ranking, app.state.ranker, request.task, config
                    )

                if shim_store is not None:
                    # The bandit consults shim rewards OUTSIDE the registry
                    # session and must never break routing: any telemetry
                    # failure falls back to the pre-bandit ranking (same
                    # convention as _record_decision).
                    try:
                        with shim_store.session() as shim_session:
                            bandit_aggregates = aggregate_rewards(
                                compute_rewards(shim_session)
                            )
                        bandit_result = apply_bandit(
                            ranking, bandit_aggregates, config.bandit
                        )
                        if bandit_result.applied:
                            ranking = bandit_result.ranking
                    except Exception as exc:  # bandit must never break routing
                        logger.warning("bandit_consult_failed error=%s", exc)

                winner = ranking.candidates[0]
                reasons = list(winner.reason_codes)
                if (
                    "neural_ranker:onnx" not in reasons
                    and REASON_UCB not in reasons
                    and "cheapest_of_meeting" not in reasons
                ):
                    reasons.append("cheapest_of_meeting")

                sys1 = evaluate_system_one(ranking, context, app.state.ranker)

                decision = Decision(
                    phase=ranking.phase,
                    model=winner.model.canonical_id,
                    provider=winner.provider.registry_key,
                    deployment=winner.deployment.deployment_ref,
                    effort=winner.variant.effort,
                    score=winner.score,
                    quality=winner.quality,
                    alternatives=tuple(
                        Alternative(
                            model=c.model.canonical_id,
                            provider=c.provider.registry_key,
                            deployment=c.deployment.deployment_ref,
                            effort=c.variant.effort,
                            score=c.score,
                            quality=c.quality,
                            estimated_tokens=c.estimated_tokens,
                        )
                        for c in ranking.candidates[1 : config.policy.top_k]
                    ),
                    reason_codes=tuple(reasons),
                    estimated_tokens=winner.estimated_tokens,
                    estimated_cost=round(winner.estimated_cost, 6),
                    policy_version=ranking.policy_version,
                    confidence=sys1.confidence,
                    probabilities=sys1.probabilities,
                    system_one=sys1.to_dict(),
                )
                candidate_count = len(ranking.candidates)
        except PolicyError as exc:
            # Fail closed, same semantics as the CLI: empty registry / no
            # candidate after hard filters / threshold unreachable → 503.
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        response = _decision_to_response(decision, _current_hash())
        _record_decision(shim_store, request, decision, candidate_count)
        return response

    @app.get("/health")
    def health() -> dict[str, str]:
        """Liveness + the provenance hashes of the served state."""
        return {
            "status": "ok",
            "registry_hash": _current_hash(),
            "policy_version": full_policy_version(config),
            "ranker": "onnx" if app.state.ranker is not None else "none",
        }

    @app.get("/policy")
    def policy() -> dict[str, Any]:
        """Full per-phase ranked policy; cached per registry_hash."""
        reg_hash = _current_hash()
        if reg_hash not in policy_cache:
            policy_cache.clear()
            policy_cache[reg_hash] = _ranked_policy_payload(config, engine)
        return {
            "registry_hash": reg_hash,
            "policy_version": full_policy_version(config),
            "phases": policy_cache[reg_hash],
        }

    @app.post("/shim/execution", response_model=ExecutionIngestResponse)
    def ingest_execution(
        payload: ExecutionIn,
        apply_rubric: bool = True,
    ) -> ExecutionIngestResponse:
        """Ingest a single runtime execution record into the telemetry shim store."""
        if shim_store is None:
            raise HTTPException(
                status_code=503, detail="telemetry shim store is not configured"
            )
        data = payload.model_dump(exclude_unset=False)
        try:
            data["phase"] = normalize_phase(payload.phase)
        except PolicyError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        if apply_rubric and (payload.task_success is None or payload.quality_score is None):
            try:
                lat = float(payload.latency_ms) if payload.latency_ms is not None else None
                outcome = score_execution(
                    data["phase"],
                    tests_passed=payload.tests_passed,
                    tests_failed=payload.tests_failed,
                    tool_errors=payload.tool_errors,
                    escalation_count=payload.escalation_count,
                    latency_ms=lat,
                    task_success=payload.task_success,
                    quality_score=payload.quality_score,
                )
                data["task_success"] = outcome.task_success
                data["quality_score"] = outcome.quality_score
            except (OutcomeRubricError, ValueError) as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc

        with shim_store.session() as session:
            record, created = shim_store.record_execution(session, data)
            exec_id = record.execution_id
            t_succ = record.task_success
            q_score = record.quality_score

        return ExecutionIngestResponse(
            status="ok",
            execution_id=exec_id,
            created=created,
            task_success=t_succ,
            quality_score=q_score,
        )

    @app.post("/shim/executions", response_model=BatchExecutionIngestResponse)
    def ingest_executions_batch(
        payloads: list[ExecutionIn],
        apply_rubric: bool = True,
    ) -> BatchExecutionIngestResponse:
        """Ingest a batch of runtime execution records in a single transaction."""
        if shim_store is None:
            raise HTTPException(
                status_code=503, detail="telemetry shim store is not configured"
            )
        created_count = 0
        updated_count = 0
        execution_ids: list[str] = []
        with shim_store.session() as session:
            for item in payloads:
                data = item.model_dump(exclude_unset=False)
                try:
                    data["phase"] = normalize_phase(item.phase)
                except PolicyError as exc:
                    raise HTTPException(status_code=422, detail=str(exc)) from exc

                if apply_rubric and (item.task_success is None or item.quality_score is None):
                    try:
                        lat = float(item.latency_ms) if item.latency_ms is not None else None
                        outcome = score_execution(
                            data["phase"],
                            tests_passed=item.tests_passed,
                            tests_failed=item.tests_failed,
                            tool_errors=item.tool_errors,
                            escalation_count=item.escalation_count,
                            latency_ms=lat,
                            task_success=item.task_success,
                            quality_score=item.quality_score,
                        )
                        data["task_success"] = outcome.task_success
                        data["quality_score"] = outcome.quality_score
                    except (OutcomeRubricError, ValueError) as exc:
                        raise HTTPException(status_code=422, detail=str(exc)) from exc

                record, created = shim_store.record_execution(session, data)
                if created:
                    created_count += 1
                else:
                    updated_count += 1
                execution_ids.append(record.execution_id)

        return BatchExecutionIngestResponse(
            status="ok",
            total=len(payloads),
            created=created_count,
            updated=updated_count,
            execution_ids=execution_ids,
        )

    @app.post("/shim/feedback", response_model=ExecutionIngestResponse)
    def ingest_feedback(
        payload: ExecutionIn,
    ) -> ExecutionIngestResponse:
        """Ingest execution telemetry and re-evaluate task_success/quality_score via rubric."""
        if shim_store is None:
            raise HTTPException(
                status_code=503, detail="telemetry shim store is not configured"
            )
        data = payload.model_dump(exclude_unset=False)
        try:
            data["phase"] = normalize_phase(payload.phase)
        except PolicyError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        try:
            lat = float(payload.latency_ms) if payload.latency_ms is not None else None
            outcome = score_execution(
                data["phase"],
                tests_passed=payload.tests_passed,
                tests_failed=payload.tests_failed,
                tool_errors=payload.tool_errors,
                escalation_count=payload.escalation_count,
                latency_ms=lat,
            )
            data["task_success"] = outcome.task_success
            data["quality_score"] = outcome.quality_score
        except (OutcomeRubricError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        with shim_store.session() as session:
            record, created = shim_store.record_execution(session, data)
            exec_id = record.execution_id
            t_succ = record.task_success
            q_score = record.quality_score

        return ExecutionIngestResponse(
            status="ok",
            execution_id=exec_id,
            created=created,
            task_success=t_succ,
            quality_score=q_score,
        )

    @app.get("/analytics/roi")
    def get_roi_analytics(limit: int = 10000) -> dict[str, Any]:
        """Compute live ROI, cost savings, and phase metrics from telemetry."""
        if shim_store is None:
            raise HTTPException(
                status_code=503, detail="telemetry shim store is not configured"
            )
        with shim_store.session() as session:
            roi = compute_roi(session, limit=limit)
            return roi.to_dict()

    @app.get("/dashboard", response_class=HTMLResponse)
    def get_dashboard() -> HTMLResponse:
        """Render the web business ROI and telemetry dashboard."""
        tpl_path = Path(__file__).resolve().parent.parent / "templates" / "dashboard.html"
        if not tpl_path.is_file():
            raise HTTPException(status_code=404, detail="dashboard template not found")
        html_content = tpl_path.read_text(encoding="utf-8")
        return HTMLResponse(content=html_content, status_code=200)

    return app
