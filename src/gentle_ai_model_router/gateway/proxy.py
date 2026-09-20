"""OpenAI-compatible gateway proxy routing and upstream dispatcher."""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from datetime import UTC, datetime
from typing import Any

import httpx
from fastapi import HTTPException
from fastapi.responses import Response, StreamingResponse
from sqlalchemy import select
from sqlalchemy.engine import Engine

from gentle_ai_model_router.gateway.schemas import (
    ChatCompletionRequest,
    ChatMessage,
    ModelCard,
    ModelListResponse,
)
from gentle_ai_model_router.integration import telemetry_shim
from gentle_ai_model_router.registry import db as registry_db
from gentle_ai_model_router.registry.models import Model
from gentle_ai_model_router.router.bandit import REASON_UCB, apply_bandit
from gentle_ai_model_router.router.config import RouterConfig
from gentle_ai_model_router.router.decision import Alternative, Decision, TaskContext
from gentle_ai_model_router.router.neural import neural_rerank
from gentle_ai_model_router.router.policy import (
    PolicyError,
    normalize_phase,
    rank_candidates,
)
from gentle_ai_model_router.router.reward import aggregate_rewards, compute_rewards
from gentle_ai_model_router.router.system_one import evaluate_system_one

logger = logging.getLogger(__name__)

SUPPORTED_SDD_PHASES = ("explore", "propose", "spec", "design", "tasks", "apply", "verify")
_PHASE_PATTERN_SDD = re.compile(
    r"\bsdd[_-](explore|propose|spec|design|tasks|apply|verify)\b",
    re.IGNORECASE,
)
_PHASE_PATTERN_AGENT = re.compile(
    r"\b(?:phase|agent|mode)\s*[:=]\s*['\"]?(?:sdd[_-])?(explore|propose|spec|design|tasks|apply|verify)\b",
    re.IGNORECASE,
)


def _get_message_text(msg: ChatMessage) -> str:
    """Extract string content from a ChatMessage, handling multimodals and text lists."""
    if isinstance(msg.content, str):
        return msg.content
    if isinstance(msg.content, list):
        parts: list[str] = []
        for item in msg.content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            elif isinstance(item, str):
                parts.append(item)
        return " ".join(parts)
    return ""


def extract_task_and_phase(messages: list[ChatMessage]) -> tuple[str, str, int]:
    """Extract prompt text, detect SDD phase, and estimate token count from messages.

    Detects SDD phases ('explore', 'propose', 'spec', 'design', 'tasks',
    'apply', 'verify'), defaulting to 'explore'.
    Returns (task_text, phase, estimated_tokens).
    """
    if not messages:
        return "", "explore", 0

    detected_phase: str | None = None
    task_text = ""

    # Reverse iterate to prioritize the most recent phase markers and user prompt
    for msg in reversed(messages):
        text = _get_message_text(msg)
        if not detected_phase:
            # Check explicit message name first
            if msg.name:
                cleaned_name = (
                    msg.name.removeprefix("sdd-").removeprefix("sdd_").lower().strip()
                )
                if cleaned_name in SUPPORTED_SDD_PHASES:
                    detected_phase = cleaned_name

            # Check sdd-<phase> pattern in content
            if not detected_phase and text:
                m_sdd = _PHASE_PATTERN_SDD.search(text)
                if m_sdd:
                    detected_phase = m_sdd.group(1).lower()

            # Check phase: / agent: pattern in content
            if not detected_phase and text:
                m_agent = _PHASE_PATTERN_AGENT.search(text)
                if m_agent:
                    detected_phase = m_agent.group(1).lower()

        # Capture task prompt from the last user message
        if not task_text and msg.role == "user" and text.strip():
            task_text = text.strip()

    # Fallback to the last non-empty message content if no user message found
    if not task_text:
        for msg in reversed(messages):
            t = _get_message_text(msg).strip()
            if t:
                task_text = t
                break

    phase = detected_phase if detected_phase in SUPPORTED_SDD_PHASES else "explore"

    # Estimate tokens: ~4 chars per token across all messages + minimal overhead
    total_chars = sum(len(_get_message_text(m)) for m in messages)
    estimated_tokens = max(1, total_chars // 4) if total_chars > 0 else 0

    return task_text, phase, estimated_tokens


def resolve_upstream(
    decision: Decision,
    requested_model: str,
    config: RouterConfig,
    auth_header: str | None = None,
) -> tuple[str, str, str]:
    """Resolve target upstream endpoint URL, winning model ID, and API credential.

    Returns (target_url, target_model, api_key).
    - Upstream Kimi (api.kimi.ai) maps to 'kimi-for-coding'
      ('kimi-for-coding-highspeed' if effort == 'low').
    - Upstream OpenRouter (openrouter.ai) maps to decision.model.
    - Specific requested models (not 'auto' or 'gentle-router/auto') pass through.
    """
    raw_base = config.gateway.upstream_base_url.rstrip("/")
    if raw_base.endswith("/chat/completions"):
        target_url = raw_base
    else:
        target_url = f"{raw_base}/chat/completions"

    is_auto = requested_model.lower().strip() in ("auto", "gentle-router/auto", "")

    if not is_auto:
        target_model = requested_model
    else:
        lower_url = raw_base.lower()
        if "api.kimi.ai" in lower_url:
            if decision.effort == "low":
                target_model = "kimi-for-coding-highspeed"
            else:
                target_model = "kimi-for-coding"
        elif "openrouter.ai" in lower_url:
            target_model = decision.model
        else:
            target_model = decision.model

    # Resolve API credential: config > environment variables > incoming Authorization header
    api_key = config.gateway.upstream_api_key
    if not api_key:
        api_key = (
            os.environ.get("UPSTREAM_API_KEY")
            or os.environ.get("GATEWAY_API_KEY")
            or os.environ.get("KIMI_API_KEY")
            or os.environ.get("OPENROUTER_API_KEY")
        )
    if not api_key and auth_header:
        token = auth_header.strip()
        if token.lower().startswith("bearer "):
            token = token[7:].strip()
        api_key = token
    if not api_key:
        api_key = ""

    return target_url, target_model, api_key


def list_gateway_models(engine: Engine) -> ModelListResponse:
    """Return OpenAI-compatible model list containing auto meta-models and registered models."""
    cards = [
        ModelCard(id="auto"),
        ModelCard(id="gentle-router/auto"),
    ]
    seen_ids = {"auto", "gentle-router/auto"}

    with registry_db.Session(engine) as session:
        models = session.scalars(select(Model).order_by(Model.canonical_id)).all()
        for m in models:
            if m.canonical_id not in seen_ids:
                cards.append(ModelCard(id=m.canonical_id))
                seen_ids.add(m.canonical_id)

    return ModelListResponse(object="list", data=cards)


def _serialize_message(msg: ChatMessage) -> dict[str, Any]:
    """Serialize a ChatMessage into an OpenAI-compatible wire payload."""
    data = msg.model_dump(exclude_none=True)
    if "content" not in data and not msg.tool_calls:
        data["content"] = ""
    return data


async def handle_chat_completion(
    request: ChatCompletionRequest,
    config: RouterConfig,
    engine: Engine,
    shim_store: telemetry_shim.ShimStore | None = None,
    ranker: Any | None = None,
    auth_header: str | None = None,
) -> Response:
    """Process an OpenAI-compatible chat completion request with dynamic model routing."""
    task, phase, estimated_tokens = extract_task_and_phase(request.messages)

    try:
        norm_phase = normalize_phase(phase)
    except PolicyError:
        norm_phase = "explore"

    context = TaskContext(
        task_type=None,
        context_tokens=estimated_tokens,
        repo_features={},
    )

    try:
        with registry_db.Session(engine) as session:
            ranking = rank_candidates(
                session,
                norm_phase,
                config,
                context,
            )
            if ranker is not None and task:
                ranking = neural_rerank(session, ranking, ranker, task, config)

            if shim_store is not None:
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
                except Exception as exc:
                    logger.warning("gateway bandit consult failed: %s", exc)

            winner = ranking.candidates[0]
            reasons = list(winner.reason_codes)
            if (
                "neural_ranker:onnx" not in reasons
                and REASON_UCB not in reasons
                and "cheapest_of_meeting" not in reasons
            ):
                reasons.append("cheapest_of_meeting")

            sys1 = evaluate_system_one(ranking, context, ranker)

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
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # Log & record decision telemetry
    decision_record_id: str | None = None
    if shim_store is not None:
        try:
            with shim_store.session() as shim_session:
                dec_rec = shim_store.record_decision(
                    shim_session,
                    phase=decision.phase,
                    selected={
                        "model": decision.model,
                        "provider": decision.provider,
                        "deployment": decision.deployment,
                        "effort": decision.effort,
                        "score": decision.score,
                        "task": task,
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
                decision_record_id = dec_rec.decision_id
        except Exception as exc:
            logger.warning("gateway decision log failed: %s", exc)

    target_url, target_model, api_key = resolve_upstream(
        decision, request.model, config, auth_header
    )

    forward_payload: dict[str, Any] = {
        "model": target_model,
        "messages": [_serialize_message(m) for m in request.messages],
        "stream": request.stream,
    }
    if request.temperature is not None:
        forward_payload["temperature"] = request.temperature
    if request.top_p is not None:
        forward_payload["top_p"] = request.top_p
    if request.max_tokens is not None:
        forward_payload["max_tokens"] = request.max_tokens
    if request.tools is not None:
        forward_payload["tools"] = request.tools
    if request.tool_choice is not None:
        forward_payload["tool_choice"] = request.tool_choice

    if request.model_extra:
        for k, v in request.model_extra.items():
            if k not in forward_payload and v is not None:
                forward_payload[k] = v

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    start_time = time.perf_counter()
    started_at = datetime.now(UTC)

    # ------------------------------------------------------------------
    # Non-streaming forward
    # ------------------------------------------------------------------
    if not request.stream:
        try:
            async with httpx.AsyncClient(timeout=config.gateway.timeout_seconds) as client:
                resp = await client.post(target_url, json=forward_payload, headers=headers)
        except httpx.TimeoutException as exc:
            raise HTTPException(status_code=504, detail="Upstream request timed out") from exc
        except httpx.RequestError as exc:
            raise HTTPException(
                status_code=502, detail=f"Upstream gateway connection error: {exc}"
            ) from exc

        latency_ms = int((time.perf_counter() - start_time) * 1000)
        finished_at = datetime.now(UTC)
        exec_id = uuid.uuid4().hex

        input_tokens = estimated_tokens
        output_tokens = 0
        total_tokens = 0
        reasoning_tokens = 0
        cached_tokens = 0
        tool_calls_count = 0

        if resp.status_code == 200:
            try:
                resp_json = resp.json()
                if resp_json.get("id"):
                    exec_id = str(resp_json["id"])
                usage = resp_json.get("usage") or {}
                if usage:
                    input_tokens = (
                        usage.get("prompt_tokens")
                        or usage.get("input_tokens")
                        or input_tokens
                    )
                    output_tokens = (
                        usage.get("completion_tokens") or usage.get("output_tokens") or 0
                    )
                    total_tokens = usage.get("total_tokens") or (input_tokens + output_tokens)
                    p_details = usage.get("prompt_tokens_details") or {}
                    c_details = usage.get("completion_tokens_details") or {}
                    cached_tokens = p_details.get("cached_tokens") or 0
                    reasoning_tokens = c_details.get("reasoning_tokens") or 0
                for choice in resp_json.get("choices") or []:
                    msg = choice.get("message") or {}
                    tc = msg.get("tool_calls")
                    if tc and isinstance(tc, list):
                        tool_calls_count += len(tc)
            except Exception as exc:
                logger.debug("failed to parse upstream usage: %s", exc)

        if total_tokens == 0:
            total_tokens = input_tokens + output_tokens

        if shim_store is not None:
            try:
                with shim_store.session() as shim_session:
                    shim_store.record_execution(
                        shim_session,
                        {
                            "execution_id": exec_id,
                            "phase": phase,
                            "model": target_model,
                            "deployment": decision.deployment,
                            "effort": decision.effort,
                            "latency_ms": latency_ms,
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "total_tokens": total_tokens,
                            "reasoning_tokens": reasoning_tokens,
                            "cached_tokens": cached_tokens,
                            "tool_calls": tool_calls_count,
                            "task_success": 1 if resp.status_code < 400 else 0,
                            "started_at": started_at,
                            "finished_at": finished_at,
                            "router_version": decision.policy_version,
                            "decision_id": decision_record_id,
                        },
                    )
            except Exception as exc:
                logger.warning("gateway shim record failed: %s", exc)

        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type", "application/json"),
        )

    # ------------------------------------------------------------------
    # Streaming SSE forward
    # ------------------------------------------------------------------
    client = httpx.AsyncClient(timeout=config.gateway.timeout_seconds)
    try:
        req = client.build_request("POST", target_url, json=forward_payload, headers=headers)
        upstream_resp = await client.send(req, stream=True)
    except httpx.TimeoutException as exc:
        await client.aclose()
        raise HTTPException(status_code=504, detail="Upstream request timed out") from exc
    except httpx.RequestError as exc:
        await client.aclose()
        raise HTTPException(
            status_code=502, detail=f"Upstream gateway connection error: {exc}"
        ) from exc

    if upstream_resp.status_code >= 400:
        err_bytes = await upstream_resp.aread()
        await upstream_resp.aclose()
        await client.aclose()
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        finished_at = datetime.now(UTC)

        if shim_store is not None:
            try:
                with shim_store.session() as shim_session:
                    shim_store.record_execution(
                        shim_session,
                        {
                            "execution_id": uuid.uuid4().hex,
                            "phase": phase,
                            "model": target_model,
                            "deployment": decision.deployment,
                            "effort": decision.effort,
                            "latency_ms": latency_ms,
                            "input_tokens": estimated_tokens,
                            "output_tokens": 0,
                            "total_tokens": estimated_tokens,
                            "task_success": 0,
                            "started_at": started_at,
                            "finished_at": finished_at,
                            "router_version": decision.policy_version,
                            "decision_id": decision_record_id,
                        },
                    )
            except Exception as exc:
                logger.warning("gateway shim record failed: %s", exc)

        return Response(
            content=err_bytes,
            status_code=upstream_resp.status_code,
            media_type=upstream_resp.headers.get("content-type", "application/json"),
        )

    async def stream_generator():
        stream_id: str | None = None
        input_toks = estimated_tokens
        output_toks = 0
        tot_toks = 0
        reas_toks = 0
        cache_toks = 0
        tool_cnt = 0
        accumulated_chunks: list[str] = []

        try:
            async for chunk in upstream_resp.aiter_bytes():
                yield chunk
                try:
                    chunk_text = chunk.decode("utf-8", errors="ignore")
                    for line in chunk_text.splitlines():
                        line = line.strip()
                        if line.startswith("data:"):
                            data_part = line[5:].strip()
                            if data_part == "[DONE]":
                                continue
                            if data_part.startswith("{"):
                                parsed = json.loads(data_part)
                                if not stream_id and parsed.get("id"):
                                    stream_id = parsed.get("id")
                                usage = parsed.get("usage")
                                if usage:
                                    input_toks = usage.get("prompt_tokens") or input_toks
                                    output_toks = usage.get("completion_tokens") or output_toks
                                    tot_toks = usage.get("total_tokens") or tot_toks
                                    p_details = usage.get("prompt_tokens_details") or {}
                                    c_details = usage.get("completion_tokens_details") or {}
                                    cache_toks = p_details.get("cached_tokens") or cache_toks
                                    reas_toks = c_details.get("reasoning_tokens") or reas_toks
                                for choice in parsed.get("choices") or []:
                                    delta = choice.get("delta") or {}
                                    if delta.get("content"):
                                        accumulated_chunks.append(str(delta["content"]))
                                    tc = delta.get("tool_calls")
                                    if tc and isinstance(tc, list):
                                        tool_cnt += len(tc)
                except Exception:
                    pass
        finally:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            finished_at = datetime.now(UTC)
            await upstream_resp.aclose()
            await client.aclose()

            if output_toks == 0 and accumulated_chunks:
                full_out = "".join(accumulated_chunks)
                output_toks = max(1, len(full_out) // 4)
            if tot_toks == 0:
                tot_toks = input_toks + output_toks

            exec_id = stream_id or uuid.uuid4().hex
            if shim_store is not None:
                try:
                    with shim_store.session() as shim_session:
                        shim_store.record_execution(
                            shim_session,
                            {
                                "execution_id": exec_id,
                                "phase": phase,
                                "model": target_model,
                                "deployment": decision.deployment,
                                "effort": decision.effort,
                                "latency_ms": latency_ms,
                                "input_tokens": input_toks,
                                "output_tokens": output_toks,
                                "total_tokens": tot_toks,
                                "reasoning_tokens": reas_toks,
                                "cached_tokens": cache_toks,
                                "tool_calls": tool_cnt,
                                "task_success": 1,
                                "started_at": started_at,
                                "finished_at": finished_at,
                                "router_version": decision.policy_version,
                                "decision_id": decision_record_id,
                            },
                        )
                except Exception as exc:
                    logger.warning("gateway shim record streaming failed: %s", exc)

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
