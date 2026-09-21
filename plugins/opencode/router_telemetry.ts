/**
 * OpenCode Runtime Model Router & Telemetry Plugin (Native TypeScript).
 *
 * 1. Active Routing Dispatch:
 *    Intercepts subagent task dispatch in `tool.execute.before`, queries
 *    local ModernBERT ONNX router daemon at `http://127.0.0.1:8000/route`
 *    (with fallback to cloud if offline), and binds optimal calibrated
 *    effort, model, and System One decision parameters to the subagent.
 *
 * 2. Static Policy Ingestion:
 *    On startup, hooks `config` to ingest policy from local/cloud `/policy`.
 *
 * 3. Execution Outcome Telemetry:
 *    Hooks `event` on completed assistant turns to stream real-time tokens,
 *    latency, tool errors, and task success to `/shim/execution`.
 */

import type { Plugin, Config } from "@opencode-ai/plugin";
import { appendFileSync } from "fs";
import { spawn } from "child_process";

const LOCAL_ROUTER_BASE = process.env.ROUTER_BASE_URL || "http://127.0.0.1:8000";
const CLOUD_ROUTER_BASE = "https://router.salema.dev";
const ROUTER_VERSION = "opencode-hook-v2-active";

const VALID_PHASES = new Set([
  "init", "explore", "research", "propose", "spec", "design", "tasks", "apply", "verify", "archive", "onboard"
]);

const CANDIDATE_MODELS = [
  "opencode/nemotron-3.5-lightning-free",
  "opencode/mimo-v2.5-free",
  "opencode/nemotron-3-ultra-free",
  "muse-spark-1.3-contributor-free",
  "kimi-for-coding",
];

const ROUTER_TO_OPENCODE_MODEL: Record<string, string> = {
  "opencode/nemotron-3.5-lightning-free": "opencode/nemotron-3.5-lightning-free",
  "opencode/mimo-v2.5-free": "opencode/mimo-v2.5-free",
  "opencode/nemotron-3-ultra-free": "opencode/nemotron-3-ultra-free",
  "muse-spark-1.3-contributor-free": "opencode/muse-spark-1.3-contributor-free",
  "kimi-for-coding": "kimi-code-plan-global/kimi-for-coding",
};

function normalizePhase(rawAgent: unknown): string {
  const cleaned = String(rawAgent || "explore").replace(/^sdd-/, "").toLowerCase();
  return VALID_PHASES.has(cleaned) ? cleaned : "explore";
}

async function fetchRouter(path: string, options: RequestInit = {}): Promise<Response | null> {
  // 1. Try local ModernBERT server first (fastest, 100% offline)
  try {
    const res = await fetch(`${LOCAL_ROUTER_BASE}${path}`, {
      ...options,
      signal: AbortSignal.timeout(1000),
    });
    if (res && res.ok) return res;
  } catch {}

  // 2. Fallback to cloud if local daemon is not running
  try {
    const res = await fetch(`${CLOUD_ROUTER_BASE}${path}`, {
      ...options,
      signal: AbortSignal.timeout(2000),
    });
    if (res && res.ok) return res;
  } catch {}

  return null;
}

export const routerTelemetryPlugin: Plugin = async () => {
  return {
    config: async (config: Config) => {
      try {
        const res = await fetchRouter("/policy");
        if (res && res.ok) {
          const policy = await res.json();
          const phases = policy?.phases || {};
          if (config.agent) {
            for (const [phaseKey, phaseData] of Object.entries(phases)) {
              const agentName = `sdd-${phaseKey}`;
              const agent = config.agent[agentName] || config.agent[phaseKey];
              if (agent && (phaseData as any)?.selected?.effort) {
                (agent as any).options = {
                  ...(agent as any).options,
                  router_recommended_effort: (phaseData as any).selected.effort,
                  router_policy_version: (phaseData as any).policy_version,
                };
              }
            }
          }
        }
      } catch {
        // Non-blocking initialization
      }
    },

    "tool.execute.before": async (input, output) => {
      try {
        if (input.tool !== "task") return;
        const subagent = output.args?.subagent_type || output.args?.agent || output.args?.subagent || "explore";
        const phase = normalizePhase(subagent);
        const taskText =
          output.args.description ||
          (typeof output.args.prompt === "string"
            ? output.args.prompt.slice(0, 300).trim()
            : `Execute ${phase}`);

        const res = await fetchRouter("/route", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            task: taskText,
            phase: phase,
            available_models: CANDIDATE_MODELS,
          }),
        });

        if (res && res.ok) {
          const decision = await res.json();
          const effort = decision.effort || "medium";
          const score = typeof decision.score === "number" ? decision.score.toFixed(2) : "1.00";
          const noul = decision.system_one?.noul_fast_success;
          const viability = typeof noul === "number" ? `${(noul * 100).toFixed(0)}%` : "optimal";
          const model = decision.model || "";
          const opencodeModel = ROUTER_TO_OPENCODE_MODEL[model] || model;

          // Log active routing decision
          try {
            appendFileSync(
              "/tmp/router_decisions.log",
              `[DECISION] time=${new Date().toISOString()} phase=${phase} model=${model} opencode_model=${opencodeModel} effort=${effort} viability=${viability}\n`
            );
          } catch {}

          // Inject calibrated decision directive into the subagent prompt
          const directive = `\n<!-- gentle-ai:router-decision -->\n[Model Router System One]: Evaluated calibrated dispatch for phase "${phase}". Recommended effort: "${effort}" (Score: ${score}, Viability: ${viability}). Model target: "${opencodeModel}".\n<!-- /gentle-ai:router-decision -->\n`;
          if (typeof output.args.prompt === "string") {
            output.args.prompt = `${directive}\n${output.args.prompt}`;
          }

          if (output.args && typeof output.args === "object" && opencodeModel && opencodeModel !== "auto") {
            output.args.model = opencodeModel;
          }
        }
      } catch {
        // Zero-crash guarantee
      }
    },

    event: async ({ event }: { event: any }) => {
      try {
        if (!event || event.type !== "message.updated") return;
        const properties = event.properties || {};
        const info = properties.info || {};

        // Only process completed assistant messages
        if (info.role === "assistant" && Number.isSafeInteger(info.time?.completed)) {
          const phase = normalizePhase(info.agent || info.mode);
          let model =
            info.providerID && info.modelID
              ? `${info.providerID}/${info.modelID}`
              : info.modelID || "unknown";
          if (model.includes("muse-spark")) {
            model = "meta/muse-spark-1.3";
          }
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

          const endpoint =
            process.env.ROUTER_SHIM_ENDPOINT || `${LOCAL_ROUTER_BASE}/shim/execution`;
          const child = spawn(
            "curl",
            [
              "-s",
              "-m",
              "5",
              "-X",
              "POST",
              endpoint,
              "-H",
              "Content-Type: application/json",
              "-d",
              JSON.stringify(payload),
            ],
            { detached: true, stdio: "ignore" }
          );
          child.unref();
        }
      } catch {
        // Zero-crash guarantee
      }
    },
  };
};

export default routerTelemetryPlugin;
