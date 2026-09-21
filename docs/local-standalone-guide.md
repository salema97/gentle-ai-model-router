# Local Standalone Architecture & Native OpenCode Hook

The **Gentle AI Model Router** runs as a fast, 100% local daemon on your development workstation. It requires zero cloud infrastructure, no remote gateway proxies, and no dummy models like `auto` in OpenCode.

---

## Architectural Topology

```
┌─────────────────────────────────────────────────────────────┐
│                      Workstation User                       │
└──────────────────────────────┬──────────────────────────────┘
                               │ Prompts & Tasks
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                    OpenCode Orchestrator                    │
│   (runs native model: e.g. kimi-for-coding or muse-spark)   │
└──────────────────────────────┬──────────────────────────────┘
                               │ Delegating subagent task
                               ▼
┌─────────────────────────────────────────────────────────────┐
│    OpenCode Native Plugin: plugins/opencode/router_telemetry│
│   • Intercepts `tool.execute.before` (task)                 │
│   • Passes currently available models in OpenCode           │
└──────────────────────────────┬──────────────────────────────┘
                               │ POST http://127.0.0.1:8000/route
                               │ (<10 ms offline)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│      Gentle AI Model Router Local Daemon (Port 8000)        │
│   • Registry: data/router.db (SQLite)                       │
│   • Ranker: ModernBERT v14 INT8 ONNX (model.quant.onnx)     │
│   • System One: Effort expectation + Noul fast viability    │
└──────────────────────────────┬──────────────────────────────┘
                               │ Returns winner model & effort
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                Subagent Execution in OpenCode               │
│   • Injected with optimal model for specific phase          │
│   • Spools execution tokens & latency to /shim/execution   │
└─────────────────────────────────────────────────────────────┘
```

---

## 1. Local Configuration

The local daemon is configured via [`router.yaml`](../router.yaml):

```yaml
data_dir: "data"

registry:
  sqlite_filename: "router.db"

api:
  host: "127.0.0.1"
  port: 8000
  route_path: "/route"
```

The promoted model is linked via [`models/promoted/promoted.json`](../models/promoted/promoted.json) pointing to `models/deberta-router/v14/model.quant.onnx`.

---

## 2. Running the Daemon via Systemd

A systemd user unit automatically manages the lifecycle:

```ini
# ~/.config/systemd/user/gentle-router.service
[Unit]
Description=Gentle AI Model Router (ModernBERT Local ONNX)
After=network.target

[Service]
Type=simple
WorkingDirectory=%h/repositorios/gentle-ai-model-router
ExecStart=%h/.local/bin/uv run router serve --port 8000
Restart=always
RestartSec=3
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=default.target
```

Commands:
```bash
# Start and enable on boot
systemctl --user daemon-reload
systemctl --user enable --now gentle-router.service

# View status & logs
systemctl --user status gentle-router.service
journalctl --user -u gentle-router.service -f
```

---

## 3. Native OpenCode Plugin

The native plugin is located at [`plugins/opencode/router_telemetry.ts`](../plugins/opencode/router_telemetry.ts).

### How it operates:
1. **Filtering by Real Models:** OpenCode transmits its real available models (`available_models: ["opencode/nemotron-3.5-lightning-free", "kimi-for-coding", ...]`).
2. **Phase-Calibrated Ranking:** ModernBERT cross-encodes the phase and task against candidates.
   - For `explore` / simple tasks: allocates low reasoning effort and fast cost-effective models.
   - For `design` / complex tasks: allocates high reasoning effort and deeper reasoning models.
3. **Subagent Injection:** Injects `output.args.model = winnerModel` dynamically into the subagent dispatch.
4. **Execution Telemetry:** On `message.updated`, the plugin reports tokens, latency, and success back to local SQLite (`data/telemetry.sqlite`).

---

## 4. Inspection & Diagnostics

```bash
# Verify daemon health
curl -s http://127.0.0.1:8000/health

# Inspect real-time decisions
cat /tmp/router_decisions.log

# View telemetry execution stats
uv run python -c "
from gentle_ai_model_router.integration.telemetry_shim import ShimStore, DecisionRecord, ExecutionRecord
s = ShimStore('sqlite:///data/telemetry.sqlite')
with s.session() as sess:
    print('Decisions recorded:', sess.query(DecisionRecord).count())
    print('Executions recorded:', sess.query(ExecutionRecord).count())
"
```
