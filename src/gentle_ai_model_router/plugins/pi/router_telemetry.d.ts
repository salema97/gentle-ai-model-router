/**
 * Type declarations for the Pi runtime telemetry plugin.
 */

export interface PiTelemetryOptions {
  endpoint?: string;
  timeoutMs?: number;
  spoolFile?: string;
}

export interface ExecutionRecordPayload {
  execution_id: string;
  session_id: string | null;
  project_id: string | null;
  phase: string;
  task_type: string | null;
  model: string | null;
  deployment: string | null;
  effort: string | null;
  started_at: string | null;
  finished_at: string | null;
  input_tokens: number;
  output_tokens: number;
  reasoning_tokens: number;
  cached_tokens: number;
  total_tokens: number;
  latency_ms: number | null;
  tool_calls: number;
  tool_errors: number;
  tests_passed: number | null;
  tests_failed: number | null;
  task_success: number | null;
  quality_score: number | null;
  escalation_count: number;
  repo_features: Record<string, unknown> | null;
  router_version: string;
  decision_id: string | null;
}

export interface PiPlugin {
  name: string;
  version: string;
  hooks: {
    turn_context: (context: any) => void;
    review_complete: (event?: any, options?: PiTelemetryOptions) => ExecutionRecordPayload | null;
  };
  onTurnContext: (context: any) => void;
  onReviewComplete: (event?: any, options?: PiTelemetryOptions) => ExecutionRecordPayload | null;
  dispatchExecution: (payload: ExecutionRecordPayload, options?: PiTelemetryOptions) => Promise<void>;
  appendToSpool: (payload: ExecutionRecordPayload, spoolPath?: string) => void;
  postJson: (targetUrl: string, payload: object, timeoutMs?: number) => Promise<{ status: number }>;
  mapReviewOutcome: (decision?: string) => { task_success: number | null; quality_score: number | null; escalation_count: number };
  createExecutionPayload: (event?: any, accumulated?: any) => ExecutionRecordPayload;
}

declare const plugin: PiPlugin;
export default plugin;
