/**
 * Type declarations for the OpenCode runtime telemetry plugin.
 */

export interface OpenCodeTelemetryOptions {
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

export interface OpenCodePlugin {
  name: string;
  version: string;
  hooks: {
    'message.updated': (event: any) => void;
    SubagentStop: (event?: any, options?: OpenCodeTelemetryOptions) => ExecutionRecordPayload | null;
  };
  onMessageUpdated: (event: any) => void;
  onSubagentStop: (event?: any, options?: OpenCodeTelemetryOptions) => ExecutionRecordPayload | null;
  dispatchExecution: (payload: ExecutionRecordPayload, options?: OpenCodeTelemetryOptions) => Promise<void>;
  appendToSpool: (payload: ExecutionRecordPayload, spoolPath?: string) => void;
  postJson: (targetUrl: string, payload: object, timeoutMs?: number) => Promise<{ status: number }>;
  createExecutionPayload: (event?: any, accumulated?: any) => ExecutionRecordPayload;
}

declare const plugin: OpenCodePlugin;
export default plugin;
