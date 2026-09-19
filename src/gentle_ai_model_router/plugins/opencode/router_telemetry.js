/**
 * OpenCode Runtime Telemetry Hook Plugin.
 *
 * Captures live execution metrics (tokens, latency, tool errors, test results)
 * from OpenCode agent invocations and streams them to the gentle-ai-model-router
 * telemetry shim store (http://127.0.0.1:8377/shim/execution).
 *
 * Design constraints:
 * - Zero external dependencies: Node.js standard libraries only (`http`, `https`, `fs`, `path`, `crypto`).
 * - Zero crash guarantee: Every hook call is wrapped in safe guards; host runtime never throws.
 * - Non-blocking: HTTP requests have a 1500ms timeout.
 * - Spool fallback: If server is offline or unreachable, events append to data/telemetry-spool.jsonl.
 */

'use strict';

const http = require('http');
const https = require('https');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const DEFAULT_ENDPOINT = 'http://127.0.0.1:8377/shim/execution';
const DEFAULT_TIMEOUT_MS = 1500;
const ROUTER_VERSION = 'opencode-hook-v1';

// In-memory state tracking per session/subagent
const sessionStore = new Map();

/**
 * Retrieve or initialize a session tracking record.
 * @param {string} sessionId
 * @returns {object}
 */
function getSessionRecord(sessionId) {
  const key = sessionId || 'default';
  if (!sessionStore.has(key)) {
    sessionStore.set(key, {
      session_id: key,
      input_tokens: 0,
      output_tokens: 0,
      reasoning_tokens: 0,
      cached_tokens: 0,
      total_tokens: 0,
      latency_ms: 0,
      tool_calls: 0,
      tool_errors: 0,
      tests_passed: null,
      tests_failed: null,
      started_at: new Date().toISOString(),
      model: null,
      deployment: null,
      effort: null,
    });
  }
  return sessionStore.get(key);
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
 * Compose a valid ExecutionRecord payload from event data and accumulated metrics.
 * @param {object} event
 * @param {object} [accumulated]
 * @returns {object}
 */
function createExecutionPayload(event = {}, accumulated = {}) {
  const rawPhase = event.phase || event.agent || accumulated.phase || 'explore';
  const phase = String(rawPhase).replace(/^sdd-/, '');
  const sessionId = event.session_id || event.sessionId || accumulated.session_id || null;

  const executionId =
    event.execution_id ||
    (typeof crypto.randomUUID === 'function'
      ? crypto.randomUUID()
      : `exec-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`);

  const inTok = Number(accumulated.input_tokens || event.input_tokens || 0);
  const outTok = Number(accumulated.output_tokens || event.output_tokens || 0);
  const reasTok = Number(accumulated.reasoning_tokens || event.reasoning_tokens || 0);
  const cacheTok = Number(accumulated.cached_tokens || event.cached_tokens || 0);
  let totTok = Number(accumulated.total_tokens || event.total_tokens || 0);
  if (totTok === 0 && (inTok > 0 || outTok > 0)) {
    totTok = inTok + outTok;
  }

  const latMs =
    accumulated.latency_ms > 0
      ? accumulated.latency_ms
      : (event.latency_ms != null ? Number(event.latency_ms) : null);

  const toolCalls =
    typeof event.tool_calls === 'number'
      ? event.tool_calls
      : (Array.isArray(event.tool_calls) ? event.tool_calls.length : (accumulated.tool_calls || 0));

  const toolErrors =
    typeof event.tool_errors === 'number'
      ? event.tool_errors
      : (Array.isArray(event.tool_errors) ? event.tool_errors.length : (accumulated.tool_errors || 0));

  return {
    execution_id: executionId,
    session_id: sessionId,
    project_id: event.project_id || accumulated.project_id || null,
    phase: phase,
    task_type: event.task_type || accumulated.task_type || null,
    model: event.model || accumulated.model || null,
    deployment: event.deployment || accumulated.deployment || null,
    effort: event.effort || event.variant || accumulated.effort || null,
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
    task_success: event.task_success != null ? Number(event.task_success) : null,
    quality_score: event.quality_score != null ? Number(event.quality_score) : null,
    escalation_count: event.escalation_count != null ? Number(event.escalation_count) : 0,
    repo_features: event.repo_features || accumulated.repo_features || null,
    router_version: ROUTER_VERSION,
    decision_id: event.decision_id || accumulated.decision_id || null,
  };
}

/**
 * OpenCode hook: message.updated
 * Accumulates incremental token counts, latency, and model metadata.
 * @param {object} event
 */
function onMessageUpdated(event) {
  try {
    if (!event) return;
    const sessionId =
      event.session_id ||
      event.sessionId ||
      (event.message && (event.message.session_id || event.message.sessionId));
    const record = getSessionRecord(sessionId);

    const msg = event.message || event;
    if (msg.model && !record.model) record.model = msg.model;
    if (msg.deployment && !record.deployment) record.deployment = msg.deployment;
    if (msg.effort && !record.effort) record.effort = msg.effort;

    const usage = event.usage || msg.usage || event.tokens || msg.tokens || {};
    const inTok = usage.input_tokens || usage.prompt_tokens || usage.input || 0;
    const outTok = usage.output_tokens || usage.completion_tokens || usage.output || 0;
    const reasTok = usage.reasoning_tokens || usage.reasoning || 0;
    const cacheTok = usage.cached_tokens || usage.cache_read_tokens || usage.cached || 0;
    const totTok = usage.total_tokens || usage.total || (Number(inTok) + Number(outTok));

    record.input_tokens += Number(inTok) || 0;
    record.output_tokens += Number(outTok) || 0;
    record.reasoning_tokens += Number(reasTok) || 0;
    record.cached_tokens += Number(cacheTok) || 0;
    record.total_tokens += Number(totTok) || 0;

    const lat = event.latency_ms || event.duration_ms || msg.latency_ms || msg.duration_ms;
    if (lat != null && !isNaN(lat)) {
      record.latency_ms += Math.round(Number(lat));
    }
  } catch (err) {
    // Zero-crash guarantee
  }
}

/**
 * OpenCode hook: SubagentStop (or agent completion).
 * Gathers accumulated phase execution data and dispatches to router shim.
 * @param {object} event
 * @param {object} [options]
 * @returns {object|null} The emitted ExecutionRecord payload
 */
function onSubagentStop(event = {}, options = {}) {
  try {
    const sessionId =
      event.session_id ||
      event.sessionId ||
      (event.subagent && event.subagent.id) ||
      'default';

    const accumulated = sessionStore.get(sessionId) || {};
    const payload = createExecutionPayload(event, accumulated);

    // Clean up session from in-memory store
    sessionStore.delete(sessionId);

    // Asynchronous non-blocking dispatch
    dispatchExecution(payload, options).catch(() => {});

    return payload;
  } catch (err) {
    // Zero-crash guarantee
    return null;
  }
}

const plugin = {
  name: 'opencode-router-telemetry',
  version: '1.0.0',
  hooks: {
    'message.updated': onMessageUpdated,
    SubagentStop: onSubagentStop,
  },
  onMessageUpdated,
  onSubagentStop,
  dispatchExecution,
  appendToSpool,
  postJson,
  createExecutionPayload,
  getSessionRecord,
  _sessionStore: sessionStore,
};

module.exports = plugin;
module.exports.default = plugin;
