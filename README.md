# Baar-Core

**Semantic routing + a hard financial kill-switch for LLM agents.**

Never get surprised by another OpenAI or Anthropic bill.

[![PyPI version](https://badge.fury.io/py/baar-core.svg)](https://badge.fury.io/py/baar-core)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Scientific Validation](https://img.shields.io/badge/Validated-MMLU%20%7C%20GSM8K%20%7C%20HumanEval-success)](https://github.com/orvi2014/Baar-Core/blob/main/RESEARCH.md)

```bash
pip install baar-core
```

`baar-core` is the PyPI package name. **Baar-Core** is the project.

---

## Why Baar-Core?

Production LLM agents have a dangerous habit:

- Simple queries still get sent to expensive models.
- One runaway loop turns your **$0.10** budget into **$8+** overnight.
- The invoice lands before you know which step burned the budget.

**Most routers optimize averages. Baar-Core ships a hard Zero-Call Financial Kill-Switch:** enforce a strict USD cap, score complexity, route cheap vs capable — and if the next safe call would exceed what’s left, **reject locally** before a single provider request. **$0 spent. Zero network calls.**

### What you get

- **Smart semantic routing** — Easy work → cheap model; hard work → capable model.
- **Budget-constrained downgrade** — If the big model would break the budget, fall back to the small one so the turn can still finish.
- **True zero-call kill-switch** — Even the cheap model unaffordable? **Fail fast** — no completion call, no surprise line item.

No surprise invoices. Stronger stance against runaway and adversarial “denial of wallet” patterns. Quality where it matters (reasoning, coding, agents) because hard tasks still reach the capable tier when the budget allows.

---

## How it works

```mermaid
graph TD
    A[User task] --> B{Semantic complexity router}
    B -- Low complexity --> C[Cheap model]
    B -- High complexity --> D{Budget check}
    D -- Affordable --> E[Capable model]
    D -- Too expensive --> F[Downgrade to cheap]
    C --> G[Spend tracking]
    E --> G
    F --> G
    G --> H[Response]
```

1. **Complexity scoring** — Fast signal for cheap vs expensive route.
2. **Budget-aware choice** — Remaining budget checked before committing to the expensive path.
3. **Local rejection** — Exhausted or unsafe to call? Stop **before** the wire.

---

## Benchmarks (hard subset, mock mode)

Difficult prompts only; **20 prompts per dataset**, `baar-bench --dataset all --mock --limit 20`. On this slice Baar-Core tracks **always-big** on quality and cost.

| Dataset | Strategy | Accuracy | Approx. cost | vs always-big |
| :--- | :--- | :---: | :---: | :---: |
| **MMLU** | Always big | 100% | ~$0.19 | — |
| **MMLU** | **Baar-Core** | **100%** | **~$0.19** | **~0%** |
| **GSM8K** | Always big | 100% | ~$0.19 | — |
| **GSM8K** | **Baar-Core** | **100%** | **~$0.19** | **~0%** |
| **HumanEval** | Always big | 100% | ~$0.19 | — |
| **HumanEval** | **Baar-Core** | **100%** | **~$0.19** | **~0%** |

**Real-world savings** show up on mixed traffic — chat, extraction, formatting — where many turns stay on the cheap model.

```bash
baar-bench --dataset all --mock
```

---

## Quick start

```python
from baar import BAARRouter

router = BAARRouter(budget=0.10)
print(router.chat("What is the capital of France?"))          # → usually cheap model
print(router.chat("Write an optimized CUDA matmul kernel."))  # → capable model if affordable

# Kill-switch: budget too low for any safe call → blocked before the API
tight = BAARRouter(budget=0.00001)
try:
    tight.chat("Any prompt")
except RuntimeError as e:
    print("Blocked safely:", e)  # zero completion calls, $0 spent
```

Works with any LiteLLM-supported provider (OpenAI, Anthropic, Groq, Together, Ollama, OpenRouter, …).

---

## Resilience

```bash
baar-stress
```

Adversarial-style checks (complexity games, tight budget). Baar-Core is designed with **OWASP LLM Top 10** style risks in mind — including unbounded consumption. Details: [RESEARCH.md](https://github.com/orvi2014/Baar-Core/blob/main/RESEARCH.md).

---

## Telemetry Summary CLI

If you enable `telemetry_jsonl_path` on `BAARRouter`, summarize logs with:

```bash
baar-telemetry path/to/telemetry.jsonl
```

This prints reject rate, failover rate, total spend, and per-model spend distribution.

---

## Configuration

Default **`complexity_threshold=0.80`** routes more traffic to the cheap model than `0.65` did; the effective threshold also **rises with budget utilization** so BIG is harder to justify as spend accumulates. Tighten or loosen with `complexity_threshold` if your workload skews very easy or very hard.

```python
router = BAARRouter(
    budget=0.10,
    small_model="gpt-4o-mini",
    big_model="gpt-4o",
    complexity_threshold=0.80,
)
```

---

## License & research

**MIT** — [LICENSE](https://github.com/orvi2014/Baar-Core/blob/main/LICENSE).

Architecture, validation notes, and security mapping: [RESEARCH.md](https://github.com/orvi2014/Baar-Core/blob/main/RESEARCH.md).
