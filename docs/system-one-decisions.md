# System One: Calibrated Decision Routing

Architectural guide to TypeSafe Jev's **System One** non-autoregressive, calibrated decision routing in `gentle-ai-model-router`.

---

## 1. Overview & Principles

Standard model routers typically produce heuristic scalar scores (such as prior quality minus cost/latency penalties) or uncalibrated neural scores. While sufficient for deterministic ranking, these outputs present key limitations for autonomous multi-agent orchestrators:

1. **Uncalibrated Confidence**: Raw scores do not represent true statistical probabilities ($P \in [0, 1]$). An orchestrator cannot distinguish whether the router is 99% certain or coin-flipping between close candidates.
2. **Lack of Ordered Rubric Expectations**: Downstream execution pipelines lack a continuous expectation of reasoning effort required for a task.
3. **Missing Escalation Gates**: Orchestrators cannot gauge whether a task will succeed cleanly on the fast path or requires immediate escalation.

Following TypeSafe Jev's System One principles, the router outputs typed, mathematically calibrated decision primitives over ModernBERT:

- **Non-Autoregressive & Fast**: Inferences execute in a single forward pass ($\le 20$ ms on ModernBERT / ONNX).
- **Calibrated Probabilities**: Candidate probabilities sum strictly to $1.0 \pm 1e-4$, allowing executors to gate autonomous execution on calibrated confidence thresholds (e.g. `confidence >= 0.85`).
- **Zero Hallucination**: The output decision space is algebraically constrained by the registered candidate arms and defined rubric levels.

---

## 2. Calibrated Decision Primitives

System One evaluates each routing request into three fundamental primitives:

### Choice (Candidate Arm Distribution)
Given candidate scores $s_1, \dots, s_N$ and temperature $T > 0$, candidate arm probabilities are computed via temperature-scaled softmax:

$$P(\text{arm}_i) = \frac{\exp((s_i - \max_j s_j) / T)}{\sum_{k=1}^N \exp((s_k - \max_j s_j) / T)}$$

The router confidence is the peak probability in the distribution:

$$\text{confidence} = \max_i P(\text{arm}_i) \in [0, 1]$$

### Score (Ordered Rubric Expectation)
Effort is modeled across 4 ordered rubric bins:
- `low` ($k = 0$)
- `medium` ($k = 1$)
- `high` ($k = 2$)
- `max` ($k = 3$)

The expected effort score represents the mathematical expectation across the rubric distribution:

$$\mathbb{E}[\text{effort}] = \sum_{k=0}^3 k \cdot p_k \in [0.0, 3.0]$$

### Noul (Fast-Success Viability)
Noul models the probability that the winning model candidate will complete the task successfully on the first pass without requiring escalation:

$$\text{noul\_fast\_success} = P(\text{fast\_success}) \in [0.0, 1.0]$$

---

## 3. Multi-Head ModernBERT Architecture

The neural representation is built on ModernBERT with a shared pooled representation feeding three dedicated heads:

```
                  ┌───────────────────────────────┐
                  │    Task Text + Candidate Text  │
                  └───────────────┬───────────────┘
                                  ▼
                  ┌───────────────────────────────┐
                  │       ModernBERT Encoder      │
                  └───────────────┬───────────────┘
                                  ▼ [CLS] pooled (hidden_size)
                         + numeric_features (dim)
                                  │
         ┌────────────────────────┼────────────────────────┐
         ▼                        ▼                        ▼
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│   ChoiceHead     │    │    ScoreHead     │    │    NoulHead      │
│ (Linear -> 1)    │    │ (Linear -> 4)    │    │ (Linear -> 1)    │
└────────┬─────────┘    └────────┬─────────┘    └────────┬─────────┘
         ▼                       ▼                       ▼
  Affinity Score         Rubric Probs + E[k]       P(fast_success)
```

### Module Specifications (`training/model.py`)
- `ChoiceHead`: Linear projection `hidden + numeric_dim -> 1` producing candidate affinity logit.
- `ScoreHead`: Linear projection `hidden + numeric_dim -> 4` producing logits over the effort rubric bins, returning logits, softmax probabilities, and expected score $\sum k \cdot p_k$.
- `NoulHead`: Linear projection `hidden + numeric_dim -> 1` producing logit and sigmoid viability probability $\sigma(z) \in [0, 1]$.
- `SystemOneModernBERT`: Encapsulates encoder and all three heads with unified `forward()` and `save_pretrained()`.
- `build_system_one_model`: Factory function supporting `tiny_modernbert_config()` for offline testing without Hugging Face downloads.

---

## 4. Calibration Assessment & Metrics (`router/calibrate.py`)

To guarantee calibration quality, the router provides standard statistical calibration metrics:

### Brier Score
Measures mean squared error between predicted probabilities and binary outcomes:

$$\text{BS} = \frac{1}{N} \sum_{i=1}^N (p_i - y_i)^2$$

### Expected Calibration Error (ECE)
Partitions predicted probabilities into $M$ equal-width bins $B_1, \dots, B_M \subseteq [0, 1]$:

$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

Where:
- $\text{conf}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} p_i$
- $\text{acc}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} y_i$

---

## 5. Wire Contracts & Schema Compatibility

All System One extensions preserve **100% backward compatibility** with existing clients.

### Pydantic Schemas (`api/schemas.py`)
```python
class SystemOneMeta(BaseModel):
    effort_score: float | None = None
    effort_probabilities: dict[str, float] | None = None
    noul_fast_success: float | None = None
    calibrated: bool = False

    @property
    def noul_fast_success_probability(self) -> float | None:
        return self.noul_fast_success

class RouteResponse(BaseModel):
    # Existing baseline fields
    model: str
    deployment: str
    effort: str
    score: float
    alternatives: list[AlternativeOut]
    reason_codes: list[str]
    estimated_tokens: float
    estimated_cost: float
    policy_version: str
    registry_hash: str

    # Calibrated System One fields
    confidence: float | None = None
    probabilities: dict[str, float] | None = None
    system_one: SystemOneMeta | None = None
```

### JSON Response Example
```json
{
  "model": "anthropic/claude-sonnet-4",
  "deployment": "default",
  "effort": "low",
  "score": 0.917,
  "alternatives": [
    {
      "model": "openai/gpt-4o",
      "deployment": "default",
      "effort": "medium",
      "score": 0.884
    }
  ],
  "reason_codes": ["meets_threshold:low", "cheapest_of_meeting"],
  "estimated_tokens": 12000.0,
  "estimated_cost": 0.036,
  "policy_version": "pol-8f3b...",
  "registry_hash": "reg-4a1c...",
  "confidence": 0.7421,
  "probabilities": {
    "anthropic/claude-sonnet-4": 0.7421,
    "openai/gpt-4o": 0.2579
  },
  "system_one": {
    "effort_score": 0.284,
    "effort_probabilities": {
      "low": 0.7538,
      "medium": 0.1682,
      "high": 0.0375,
      "max": 0.0084
    },
    "noul_fast_success": 0.917,
    "calibrated": true
  }
}
```

---

## 6. Downstream Agent Orchestration

Gentle AI downstream orchestrators (such as SDD workflows) consume these primitives for autonomous branching:

1. **Autonomous Execution Gating**: If `response.confidence >= 0.85`, execute immediately with selected model. If `confidence < 0.60`, flag for multi-model verification or human approval.
2. **Dynamic Reasoning Allocation**: If `system_one.effort_score > 1.5`, preemptively allocate higher token ceilings and reasoning budget.
3. **Escalation Prediction**: If `system_one.noul_fast_success < 0.50`, prepare an escalation worker model in parallel rather than waiting for downstream test failures.
