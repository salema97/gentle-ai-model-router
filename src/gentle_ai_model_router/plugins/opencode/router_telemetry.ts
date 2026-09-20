/**
 * OpenCode Runtime Telemetry Hook Plugin (Native TypeScript).
 *
 * Captures live execution metrics (tokens, latency, tool errors) from OpenCode
 * agent turns and streams them to the gentle-ai-model-router telemetry endpoint
 * (https://router.salema.dev/shim/execution).
 */

import type { Plugin } from "@opencode-ai/plugin";

const DEFAULT_ENDPOINT = "https://router.salema.dev/shim/execution";
const DEFAULT_TIMEOUT_MS = 2500;
const ROUTER_VERSION = "opencode-hook-v1";

export const routerTelemetryPlugin: Plugin = async () => {
  return {
    event: async ({ event }: { event: any }) => {
      try {
        if (!event) return;
        const type = event.type;
        const properties = event.properties || {};

        if (type === "message.updated") {
          const info = properties.info || {};
          // Only process completed assistant messages
          if (info.role === "assistant" && Number.isSafeInteger(info.time?.completed)) {
            const rawAgent = info.agent || info.mode || "explore";
            const phase = String(rawAgent).replace(/^sdd-/, "");
            const model =
              info.providerID && info.modelID
                ? `${info.providerID}/${info.modelID}`
                : info.modelID || "unknown";
            const latency =
              info.time.completed && info.time.created
                ? info.time.completed - info.time.created
                : undefined;
            const tokens = info.tokens || {};

            const inTok = Number(tokens.input || 0);
            const outTok = Number(tokens.output || 0);
            const hasError = Boolean(info.error);

            const payload = {
              execution_id: `exec-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`,
              phase: phase,
              model: model,
              input_tokens: inTok,
              output_tokens: outTok,
              reasoning_tokens: Number(tokens.reasoning || 0),
              cached_tokens: Number(tokens.cache?.read || 0),
              total_tokens: inTok + outTok,
              latency_ms: latency,
              tool_calls: 0,
              tool_errors: hasError ? 1 : 0,
              tests_passed: null,
              tests_failed: null,
              task_success: hasError ? 0 : 1,
              quality_score: hasError ? 0.0 : 1.0,
              router_version: ROUTER_VERSION,
            };

            const endpoint = process.env.ROUTER_SHIM_ENDPOINT || DEFAULT_ENDPOINT;
            fetch(endpoint, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify(payload),
              signal: AbortSignal.timeout(DEFAULT_TIMEOUT_MS),
            }).catch(() => {});
          }
        }
      } catch (err) {
        // Zero-crash guarantee
      }
    },
  };
};

export default routerTelemetryPlugin;
