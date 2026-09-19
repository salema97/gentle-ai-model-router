/**
 * Pi (gentle-pi) Runtime Telemetry Hook Plugin.
 *
 * Intercepts Pi's `turn_context` and review completion events to capture
 * model assignments, thinking effort levels, token metrics, and review outcomes,
 * streaming them to the gentle-ai-model-router telemetry shim store
 * (http://127.0.0.1:8377/shim/execution).
 *
 * Design constraints:
 * - Zero external dependencies: Pure standard Node.js APIs (`http`, `https`, `fs`, `path`, `crypto`).
 * - Zero crash guarantee: Every hook call is wrapped in safe guards; host runtime never throws.
 * - Non-blocking: HTTP requests have a 1500ms timeout.
 * - Spool fallback: If server is unreachable, appends to data/telemetry-spool.jsonl.
 */

'use strict';

const http = require('http');
const https = require('https');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const DEFAULT_ENDPOINT = 'http://127.0.0.1:8377/shim/execution';
const DEFAULT_TIMEOUT_MS = 1500;
const ROUTER_VERSION = 'pi-hook-v1';

// In-memory store for active review turns
const turnStore = new Map();

/**
 * Retrieve or initialize turn record.
 * @param {string} turnId
 * @returns {object}
 */
function getTurnRecord(turnId) {
  const key = turnId || 'default';
  if (!turnStore.has(key)) {
    turnStore.set(key, {
      turn_id: key,
      phase: 'verify',
      model: null,
      effort: null,
      input_tokens: 0,
      output_tokens: 0,
      reasoning_tokens: 0,
      cached_tokens: 0,
      total_tokens: 0,
      latency_ms: null,
      tool_calls: 0,
      tool_errors: 0,
      escalation_count: 0,
      started_at: new Date().toISOString(),
    });
  }
  return turnStore.get(key);
}

/**
 * Post JSON payload over HTTP/HTTPS with timeout.
 * @param {string} targetUrl
 * @param {object} payload
 * @param {number} timeoutMs
 * @returns {Promise<{ status: number }>}
 */
function postJson(targetUrl, payload, timeoutMs = DEFAULT_TIMEOUT_MS) {
  return new Promise((resolve, reject) => {
    try {
      const url = new URL(targetUrl);
      const body = JSON.stringify(payload);
      const client = url.protocol === 'https:' ? https : http;
      const req = client.request(
        url,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Content-Length': Buffer.byteLength(body),
          },
          timeout: timeoutMs,
        },
        (res) => {
          res.resume();
          if (res.statusCode >= 200 && res.statusCode < 300) {
            resolve({ status: res.statusCode });
          } else {
            reject(new Error(`HTTP ${res.statusCode}`));
          }
        }
      );
      req.on('timeout', () => {
        req.destroy(new Error('Request timed out'));
      });
      req.on('error', (err) => {
        reject(err);
      });
      req.write(body);
      req.end();
    } catch (err) {
      reject(err);
    }
  });
}

/**
 * Append execution payload to local spool file on disk.
 * @param {object} payload
 * @param {string} [spoolPath]
 */
function appendToSpool(payload, spoolPath) {
  try {
    const line = JSON.stringify(payload) + '\n';
    const resolvedPath =
      spoolPath ||
      process.env.ROUTER_SPOOL_FILE ||
      path.join(process.cwd(), 'data', 'telemetry-spool.jsonl');
    const dir = path.dirname(resolvedPath);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
    fs.appendFileSync(resolvedPath, line, { encoding: 'utf-8' });
  } catch (err) {
    // Fail-safe: zero crash guarantee
  }
}

/**
 * Dispatches execution payload via HTTP with fallback to local spool.
 * @param {object} payload
 * @param {object} [options]
 * @returns {Promise<void>}
 */
async function dispatchExecution(payload, options = {}) {
  const endpoint = options.endpoint || process.env.ROUTER_SHIM_ENDPOINT || DEFAULT_ENDPOINT;
  const timeoutMs = options.timeoutMs || DEFAULT_TIMEOUT_MS;
  const spoolFile = options.spoolFile || process.env.ROUTER_SPOOL_FILE;

  try {
    await postJson(endpoint, payload, timeoutMs);
  } catch (err) {
    appendToSpool(payload, spoolFile);
  }
}

/**
 * Map review outcome decisions to (task_success, quality_score, escalation_count).
 * @param {string} [decision]
 * @returns {{ task_success: number|null, quality_score: number|null, escalation_count: number }}
 */
function mapReviewOutcome(decision) {
  if (!decision) {
    return { task_success: null, quality_score: null, escalation_count: 0 };
  }
  const norm = String(decision).toLowerCase().trim();
  switch (norm) {
    case 'approved':
      return { task_success: 1, quality_score: 100.0, escalation_count: 0 };
    case 'correction_required':
      return { task_success: 0, quality_score: 40.0, escalation_count: 0 };
    case 'escalated':
      return { task_success: 0, quality_score: 20.0, escalation_count: 1 };
    case 'info':
    case 'inconclusive':
      return { task_success: 1, quality_score: 75.0, escalation_count: 0 };
    default:
      return { task_success: null, quality_score: null, escalation_count: 0 };
  }
}

/**
 * Compose a valid ExecutionRecord payload from Pi turn data.
 * @param {object} event
 * @param {object} [accumulated]
 * @returns {object}
 */
function createExecutionPayload(event = {}, accumulated = {}) {
  const rawPhase = event.phase || accumulated.phase || 'verify';
  const phase = String(rawPhase).replace(/^sdd-/, '');
  const sessionId = event.session_id || event.turn_id || accumulated.turn_id || null;

  const executionId =
    event.execution_id ||
    (typeof crypto.randomUUID === 'function'
      ? crypto.randomUUID()
      : `pi-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`);

  const outcomeMapping = mapReviewOutcome(event.decision || event.outcome);
  const taskSuccess =
    event.task_success != null ? Number(event.task_success) : outcomeMapping.task_success;
  const qualityScore =
    event.quality_score != null ? Number(event.quality_score) : outcomeMapping.quality_score;
  const escalationCount =
    event.escalation_count != null
      ? Number(event.escalation_count)
      : (accumulated.escalation_count || outcomeMapping.escalation_count);

  const inTok = Number(accumulated.input_tokens || event.input_tokens || 0);
  const outTok = Number(accumulated.output_tokens || event.output_tokens || 0);
  const reasTok = Number(accumulated.reasoning_tokens || event.reasoning_tokens || 0);
  const cacheTok = Number(accumulated.cached_tokens || event.cached_tokens || 0);
  let totTok = Number(accumulated.total_tokens || event.total_tokens || 0);
  if (totTok === 0 && (inTok > 0 || outTok > 0)) {
    totTok = inTok + outTok;
  }

  const latMs =
    event.latency_ms != null
      ? Number(event.latency_ms)
      : (accumulated.latency_ms != null ? accumulated.latency_ms : null);

  const toolCalls =
    typeof event.tool_calls === 'number'
      ? event.tool_calls
      : (accumulated.tool_calls || 0);

  const toolErrors =
    typeof event.tool_errors === 'number'
      ? event.tool_errors
      : (accumulated.tool_errors || 0);

  return {
    execution_id: executionId,
    session_id: sessionId,
    project_id: event.project_id || accumulated.project_id || null,
    phase: phase,
    task_type: event.task_type || 'review',
    model: event.model || accumulated.model || null,
    deployment: event.deployment || accumulated.deployment || null,
    effort: event.effort || event.thinking || accumulated.effort || null,
    started_at: accumulated.started_at || new Date().toISOString(),
    finished_at: new Date().toISOString(),
    input_tokens: inTok,
    output_tokens: outTok,
    reasoning_tokens: reasTok,
    cached_tokens: cacheTok,
    total_tokens: totTok,
    latency_ms: latMs,
    tool_calls: toolCalls,
    tool_errors: toolErrors,
    tests_passed: event.tests_passed != null ? Number(event.tests_passed) : null,
    tests_failed: event.tests_failed != null ? Number(event.tests_failed) : null,
    task_success: taskSuccess,
    quality_score: qualityScore,
    escalation_count: escalationCount,
    repo_features: event.repo_features || null,
    router_version: ROUTER_VERSION,
    decision_id: event.decision_id || null,
  };
}

/**
 * Pi hook: turn_context
 * Intercepts turn context containing model, thinking level, and review role.
 * @param {object} context
 */
function onTurnContext(context = {}) {
  try {
    if (!context) return;
    const turnId = context.turn_id || context.session_id || context.id || 'default';
    const record = getTurnRecord(turnId);

    if (context.phase) record.phase = String(context.phase).replace(/^sdd-/, '');
    if (context.model) record.model = context.model;
    if (context.deployment) record.deployment = context.deployment;
    if (context.effort || context.thinking) record.effort = context.effort || context.thinking;

    const tokens = context.tokens || context.usage || {};
    if (tokens.input_tokens || tokens.input) record.input_tokens += Number(tokens.input_tokens || tokens.input);
    if (tokens.output_tokens || tokens.output) record.output_tokens += Number(tokens.output_tokens || tokens.output);
    if (tokens.reasoning_tokens || tokens.reasoning) record.reasoning_tokens += Number(tokens.reasoning_tokens || tokens.reasoning);
    if (tokens.cached_tokens || tokens.cached) record.cached_tokens += Number(tokens.cached_tokens || tokens.cached);
    if (tokens.total_tokens || tokens.total) record.total_tokens += Number(tokens.total_tokens || tokens.total);

    if (context.latency_ms != null) {
      record.latency_ms = (record.latency_ms || 0) + Number(context.latency_ms);
    }
  } catch (err) {
    // Zero-crash guarantee
  }
}

/**
 * Pi hook: review completion.
 * Dispatches completed review execution record with outcome to the router shim.
 * @param {object} event
 * @param {object} [options]
 * @returns {object|null} The emitted ExecutionRecord payload
 */
function onReviewComplete(event = {}, options = {}) {
  try {
    const turnId = event.turn_id || event.session_id || event.id || 'default';
    const accumulated = turnStore.get(turnId) || {};
    const payload = createExecutionPayload(event, accumulated);

    // Clean up in-memory turn store
    turnStore.delete(turnId);

    // Asynchronous non-blocking dispatch
    dispatchExecution(payload, options).catch(() => {});

    return payload;
  } catch (err) {
    // Zero-crash guarantee
    return null;
  }
}

const plugin = {
  name: 'pi-router-telemetry',
  version: '1.0.0',
  hooks: {
    turn_context: onTurnContext,
    review_complete: onReviewComplete,
  },
  onTurnContext,
  onReviewComplete,
  dispatchExecution,
  appendToSpool,
  postJson,
  mapReviewOutcome,
  createExecutionPayload,
  getTurnRecord,
  _turnStore: turnStore,
};

module.exports = plugin;
module.exports.default = plugin;
