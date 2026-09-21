# Feature: OpenAI-Compatible Gateway Proxy

## Objective

Expose standard OpenAI-compatible endpoints (`/v1/models` and `/v1/chat/completions`) on `gentle-ai-model-router` so any standard client or IDE (specifically OpenCode via `@ai-sdk/openai-compatible`) can route requests dynamically through `gentle-router/auto` with zero plugin hacks.

## Problem

1. OpenCode resolves provider, model, and auth credentials *before* executing runtime hooks (`chat.params` cannot change the provider/model on the fly).
2. OpenCode's free tier (`muse-spark-1.3-contributor-free`) rejects subagents (`agent=explore mode=subagent`) with provider errors.
3. Users need an OpenAI-compatible Gateway where selecting `gentle-router/auto` automatically:
   - Evaluates SDD phase and task context.
   - Executes the learned neural ranking + System One calibrated decision.
   - Proxies the completion to the winning target provider (Moonshot Kimi, OpenRouter, etc.).
   - Transparently streams SSE chunks back to the editor.
   - Ingests token usage, latency, and outcome telemetry into the database.

## Scope

Authorized:
- Gateway schemas in `src/gentle_ai_model_router/gateway/schemas.py`.
- Proxy router and SSE streaming pipeline in `src/gentle_ai_model_router/gateway/proxy.py`.
- Gateway config in `src/gentle_ai_model_router/router/config.py`.
- Gateway routes mounted in `src/gentle_ai_model_router/api/server.py` (`/v1/models`, `/v1/chat/completions`, `/models`, `/chat/completions`).
- Unit and integration tests in `tests/test_proxy_gateway.py`.
- OpenCode provider configuration in `~/.config/opencode/opencode.jsonc`.

NOT in scope:
- Re-architecting the deterministic policy or registry schemas.

## Tasks

- [x] **G1** — Gateway Schemas: Define OpenAI-compatible request, response, chunk, and model schemas in `gateway/schemas.py`.
- [x] **G2** — Proxy Routing & Upstream Dispatcher: Build `gateway/proxy.py` with phase inference, upstream resolution, streaming SSE forwarder, and telemetry recorder.
- [x] **G3** — Configuration: Add `GatewayConfig` to `router/config.py` with environment variable overrides (`UPSTREAM_BASE_URL`, `UPSTREAM_API_KEY`, etc.).
- [x] **G4** — Server Integration: Mount `/v1/models` and `/v1/chat/completions` in `api/server.py`.
- [x] **G5** — Verification & Test Suite: Write comprehensive tests in `tests/test_proxy_gateway.py` covering models listing, streaming SSE, non-streaming, telemetry ingestion, and error handling.
- [x] **G6** — OpenCode Provider Setup: Configure `gentle-router` in `~/.config/opencode/opencode.jsonc`.
- [x] **G7** — Quota-Exhaustion & Error Fallback: Detect quota, balance, rate limit (402, 403, 429) and upstream failure (5xx), implement in-memory cooldown/circuit breaker, and automatically fall back to alternative candidates or secondary upstream in `gateway/proxy.py`.
- [x] **G8** — Fallback Configuration & Tests: Add `fallback_upstream_url`, `fallback_upstream_key`, `fallback_model` to `GatewayConfig` and verify auto-fallback in `tests/test_proxy_gateway.py`.
