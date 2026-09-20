"""Analytics and ROI engine for Gentle AI Model Router.

Calculates counterfactual baseline savings (what if all queries ran on frontier models
like Claude 3.5 Sonnet / GPT-4o?) vs. actual model routing cost, latency savings,
and reliability metrics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from gentle_ai_model_router.integration.telemetry_shim import ExecutionRecord

# Frontier baseline pricing (Claude 3.5 Sonnet / GPT-4o standard tier)
FRONTIER_INPUT_PRICE_PER_M = 3.00   # $3.00 per 1M input tokens
FRONTIER_OUTPUT_PRICE_PER_M = 15.00 # $15.00 per 1M output tokens
FRONTIER_AVG_LATENCY_MS = 2200.0    # 2.2s typical frontier round-trip


@dataclass(frozen=True)
class PhaseRoiStat:
    phase: str
    executions: int
    total_tokens: int
    actual_cost_usd: float
    baseline_cost_usd: float
    saved_usd: float
    avg_latency_ms: float
    success_rate: float


@dataclass(frozen=True)
class ModelUsageStat:
    model: str
    executions: int
    total_tokens: int
    actual_cost_usd: float
    avg_latency_ms: float


@dataclass(frozen=True)
class RoiSummary:
    total_executions: int
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    actual_cost_usd: float
    baseline_cost_usd: float
    total_saved_usd: float
    savings_percentage: float
    avg_latency_ms: float
    latency_saved_ms: float
    overall_success_rate: float
    phases: list[PhaseRoiStat] = field(default_factory=list)
    top_models: list[ModelUsageStat] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_roi(session: Session, limit: int = 10000) -> RoiSummary:
    """Compute comprehensive ROI metrics over recorded telemetry executions."""
    records: list[ExecutionRecord] = list(
        session.scalars(
            select(ExecutionRecord)
            .order_by(ExecutionRecord.id.desc())
            .limit(limit)
        ).all()
    )

    if not records:
        return RoiSummary(
            total_executions=0,
            total_input_tokens=0,
            total_output_tokens=0,
            total_tokens=0,
            actual_cost_usd=0.0,
            baseline_cost_usd=0.0,
            total_saved_usd=0.0,
            savings_percentage=0.0,
            avg_latency_ms=0.0,
            latency_saved_ms=0.0,
            overall_success_rate=1.0,
        )

    tot_in = 0
    tot_out = 0
    tot_act_cost = 0.0
    tot_base_cost = 0.0
    latencies: list[float] = []
    successes: list[float] = []

    phase_map: dict[str, list[ExecutionRecord]] = {}
    model_map: dict[str, list[ExecutionRecord]] = {}

    for r in records:
        in_tok = r.input_tokens or 0
        out_tok = r.output_tokens or 0
        tot_in += in_tok
        tot_out += out_tok

        base_cost = (in_tok * FRONTIER_INPUT_PRICE_PER_M / 1_000_000.0) + (
            out_tok * FRONTIER_OUTPUT_PRICE_PER_M / 1_000_000.0
        )
        tot_base_cost += base_cost

        # Approximate actual cost based on model tier or default discount
        model_name = (r.model or "unknown").lower()
        if any(f in model_name for f in ("flash", "mini", "haiku", "small", "8b")):
            cost_factor = 0.05
        elif any(f in model_name for f in ("deepseek", "qwen", "cerebras", "llama", "local")):
            cost_factor = 0.08
        elif any(f in model_name for f in ("sonnet", "opus", "gpt-4o", "pro")):
            cost_factor = 0.85
        else:
            cost_factor = 0.15

        act_cost = base_cost * cost_factor
        tot_act_cost += act_cost

        if r.latency_ms is not None and r.latency_ms > 0:
            latencies.append(float(r.latency_ms))

        if r.task_success is not None:
            successes.append(float(r.task_success))

        p = r.phase or "unknown"
        phase_map.setdefault(p, []).append(r)

        m = r.model or "unknown"
        model_map.setdefault(m, []).append(r)

    total_saved = max(0.0, tot_base_cost - tot_act_cost)
    savings_pct = (total_saved / tot_base_cost * 100.0) if tot_base_cost > 0 else 0.0
    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
    lat_saved = max(0.0, FRONTIER_AVG_LATENCY_MS - avg_lat)
    overall_success = sum(successes) / len(successes) if successes else 1.0

    # Phase breakdown
    phase_stats: list[PhaseRoiStat] = []
    for phase_name, p_records in phase_map.items():
        p_in = sum(x.input_tokens or 0 for x in p_records)
        p_out = sum(x.output_tokens or 0 for x in p_records)
        p_base = (p_in * FRONTIER_INPUT_PRICE_PER_M / 1_000_000.0) + (
            p_out * FRONTIER_OUTPUT_PRICE_PER_M / 1_000_000.0
        )
        p_lats = [float(x.latency_ms) for x in p_records if x.latency_ms]
        p_avg_lat = sum(p_lats) / len(p_lats) if p_lats else 0.0
        p_succ = [float(x.task_success) for x in p_records if x.task_success is not None]
        p_avg_succ = sum(p_succ) / len(p_succ) if p_succ else 1.0

        p_act = p_base * 0.15
        phase_stats.append(
            PhaseRoiStat(
                phase=phase_name,
                executions=len(p_records),
                total_tokens=p_in + p_out,
                actual_cost_usd=round(p_act, 4),
                baseline_cost_usd=round(p_base, 4),
                saved_usd=round(max(0.0, p_base - p_act), 4),
                avg_latency_ms=round(p_avg_lat, 1),
                success_rate=round(p_avg_succ, 3),
            )
        )

    # Top models
    model_stats: list[ModelUsageStat] = []
    for mod_name, m_records in model_map.items():
        m_in = sum(x.input_tokens or 0 for x in m_records)
        m_out = sum(x.output_tokens or 0 for x in m_records)
        m_lats = [float(x.latency_ms) for x in m_records if x.latency_ms]
        m_avg_lat = sum(m_lats) / len(m_lats) if m_lats else 0.0
        m_base = (m_in * FRONTIER_INPUT_PRICE_PER_M / 1_000_000.0) + (
            m_out * FRONTIER_OUTPUT_PRICE_PER_M / 1_000_000.0
        )
        model_stats.append(
            ModelUsageStat(
                model=mod_name,
                executions=len(m_records),
                total_tokens=m_in + m_out,
                actual_cost_usd=round(m_base * 0.15, 4),
                avg_latency_ms=round(m_avg_lat, 1),
            )
        )

    model_stats.sort(key=lambda s: s.executions, reverse=True)

    return RoiSummary(
        total_executions=len(records),
        total_input_tokens=tot_in,
        total_output_tokens=tot_out,
        total_tokens=tot_in + tot_out,
        actual_cost_usd=round(tot_act_cost, 4),
        baseline_cost_usd=round(tot_base_cost, 4),
        total_saved_usd=round(total_saved, 4),
        savings_percentage=round(savings_pct, 1),
        avg_latency_ms=round(avg_lat, 1),
        latency_saved_ms=round(lat_saved, 1),
        overall_success_rate=round(overall_success, 3),
        phases=phase_stats,
        top_models=model_stats[:10],
    )
