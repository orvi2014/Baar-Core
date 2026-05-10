# Baar-Core

Pre-flight runtime governance for LLM agents. Enforces hard execution limits locally — before requests reach the provider.

**Stop LLM API calls before they happen. Not after.**

[![CI](https://github.com/orvi2014/Baar-Core/actions/workflows/ci.yml/badge.svg)](https://github.com/orvi2014/Baar-Core/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/baar-core.svg)](https://badge.fury.io/py/baar-core)
[![PyPI Downloads](https://img.shields.io/pypi/dm/baar-core)](https://pypi.org/project/baar-core/)
[![GitHub Stars](https://img.shields.io/github/stars/orvi2014/Baar-Core)](https://github.com/orvi2014/Baar-Core/stargazers)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

```bash
pip install baar-core
```


> I left an agent loop running overnight.
> Woke up to a **$47 bill** — 20,000 GPT-4o tokens answering "what time is it?" queries.
>
> Baar-Core would have stopped it at **$0.10**. Before the first overage call.
> No network request made. **$0 spent.**

```python
from baar import BAARRouter

router = BAARRouter(budget=0.10)          # hard cap: $0.10 total
router.chat("What time is it?")           # → cheap model, ~$0.0001
router.chat("Write a CUDA matmul kernel") # → capable model if budget allows
# budget exhausted → raises BudgetExhausted, zero API calls made
```

> 84–94% cost reduction in benchmarks — [see benchmarks](#benchmarks)


## Built for

- Teams deploying autonomous agents in production
- SaaS founders giving users LLM access with per-user quotas
- Anyone who has had (or fears) a runaway agent bill


## The problem with every other solution

Most cost tools **track spend after the fact.** You get an alert when the bill is already large.

LiteLLM's budget manager, Portkey rate limits, provider spend alerts — they all tell you what happened. They don't stop it mid-flight.

Helicone observes the disaster. Portkey rate-limits after the fact. **Baar prevents it.**

**Baar-Core is a local kill-switch.** Before each call, it estimates the cost. If the remaining budget is too low, it raises an exception **locally** — no DNS lookup, no TCP connection, no token consumed. The call never leaves your machine.


## How it works

```
User task
    │
    ▼
┌─────────────────────────────────┐
│  Pre-flight budget check        │ ← if estimated cost > remaining budget
│  (local, zero network)          │   raise BudgetExhausted immediately
└────────────┬────────────────────┘
             │ affordable
             ▼
┌─────────────────────────────────┐
│  Semantic complexity router     │ ← cheap LLM scores complexity 0.0–1.0
│  (gpt-4o-mini, ~$0.000015/call) │
└────────────┬────────────────────┘
             │
      ┌──────┴───────┐
      │              │
   simple         complex
      │              │
      ▼              ▼
 Cheap model    Budget check
 (fast, $)      ├─ affordable → Capable model ($$$)
                └─ too close  → Downgrade to cheap model ($)
```

1. **Pre-flight check** — Estimates cost locally before any network call. Kills the request if it would overshoot.
2. **Semantic routing** — A fast, cheap model scores task complexity. Not keyword matching — actual semantic understanding.
3. **Budget-aware downgrade** — Running low? Hard tasks automatically fall back to the cheaper model so the turn still completes.


## Quick start

```python
from baar import BAARRouter, BudgetExhausted

# Basic usage
router = BAARRouter(budget=0.10)
reply = router.chat("Explain recursion with a Python example")
print(reply)
print(f"Spent: ${router.spent:.5f} / Remaining: ${router.remaining:.5f}")

# Multi-step with a report
log = router.run([
    "What is 42 * 17?",
    "Translate 'good morning' to Japanese",
    "Design a distributed rate-limiter for 100k RPS — include trade-offs",
    "Convert 72°F to Celsius",
])
log.print_report()

# Async
import asyncio
async def main():
    router = BAARRouter(budget=0.05)
    reply = await router.achat("Summarize the CAP theorem")
    print(reply)

asyncio.run(main())

# Kill-switch in action
router = BAARRouter(budget=0.00001)
try:
    router.chat("Any prompt at all")
except BudgetExhausted as e:
    print(f"Blocked before API call. Remaining: ${e.remaining:.6f}")
    # Zero network calls made. $0 spent.
```

Works with any [LiteLLM-supported provider](https://docs.litellm.ai/docs/providers): OpenAI, Anthropic, Groq, Together, Ollama, OpenRouter, Azure, and more.


## LangChain & LangGraph integration

```bash
pip install baar-core[langchain]
```

**Pattern 1 — add a kill-switch to an existing chain** (zero refactoring):

```python
from baar import BAARRouter
from baar.integrations.langchain import BaarCallbackHandler
from langchain_openai import ChatOpenAI

router  = BAARRouter(budget=0.05)
handler = BaarCallbackHandler(router)

llm   = ChatOpenAI(model="gpt-4o", callbacks=[handler])
chain = prompt | llm | StrOutputParser()
# BudgetExhausted is raised before the API call when budget is too low
```

**Pattern 2 — BaarChatModel as a drop-in ChatModel** (full routing + downgrade):

```python
from baar.integrations.langchain import BaarChatModel

router = BAARRouter(budget=0.10, small_model="gpt-4o-mini", big_model="gpt-4o")
llm    = BaarChatModel(router=router)

chain = prompt | llm | StrOutputParser()   # works anywhere ChatOpenAI would
```

**Pattern 3 — LangGraph agent with budget protection**:

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(model=llm, tools=[...])
# every tool-call iteration is budget-checked before the LLM is invoked
# BudgetExhausted stops the loop locally — no dangling API calls
```

Full example: [langchain_guardrail.py](examples/langchain_guardrail.py)


## Real-world examples

| Example | Use case |
|---|---|
| [langchain_guardrail.py](examples/langchain_guardrail.py) | LangChain callback handler, BaarChatModel, LangGraph agent |
| [fastapi_per_user_budget.py](examples/fastapi_per_user_budget.py) | SaaS: per-user $0.10 quota with SQLite persistence |
| [agent_loop.py](examples/agent_loop.py) | Autonomous agent loop with graceful budget stop |
| [streaming.py](examples/streaming.py) | Streaming responses with live budget tracking |
| [multi_tenant.py](examples/multi_tenant.py) | Concurrent multi-user budget isolation, quota report |
| [basic_usage.py](examples/basic_usage.py) | Getting started |


## Multi-tenant & per-user budgets

Give each user their own hard quota — enforced locally, persisted across restarts, no proxy or external service required.

```python
from baar import BAARRouter, BudgetExhausted
from baar.core.stores import SQLiteBudgetStore

def router_for(user_id: str) -> BAARRouter:
    return BAARRouter(
        budget=0.10,
        store=SQLiteBudgetStore("budgets.db", namespace=user_id),
    )

# Alice and Bob share one database file — budgets are fully isolated
alice = router_for("alice")
bob   = router_for("bob")

alice.chat("Summarise this document")  # deducted from Alice's $0.10 only
bob.chat("Translate to French")        # Bob's quota untouched
```

Namespace isolation is enforced at the store level — one user exhausting their quota cannot affect another's. Concurrent writes across threads **and across OS processes** (gunicorn/uvicorn workers) are safe: `SQLiteBudgetStore` uses WAL mode and an exclusive transaction for the check-and-reserve step, eliminating the TOCTOU race that affects most multi-tenant budget implementations.

**SaaS pattern — return HTTP 402 when a user's quota is exhausted:**

```python
@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        reply = await router_for(req.user_id).achat(req.message)
    except BudgetExhausted as e:
        raise HTTPException(status_code=402, detail={
            "error": "quota_exhausted",
            "remaining_usd": e.remaining,
        })
    return {"reply": reply}
```

No billing service. No proxy. One SQLite file. Full example: [fastapi_per_user_budget.py](examples/fastapi_per_user_budget.py)

**Time-windowed quotas** — scope budgets to a day or month by encoding the period in the namespace:

```python
import datetime
period = datetime.date.today().strftime("%Y-%m")   # "2026-05"
store  = SQLiteBudgetStore("budgets.db", namespace=f"user:{user_id}:{period}")
# quota resets automatically next month — no cron job needed
```

**Single-process scripts** — JSON file store, no SQLite needed:

```python
from baar.core.stores import FileBudgetStore

router = BAARRouter(budget=1.00, store=FileBudgetStore("my_budget.json"))
```


## Benchmarks

### Routing cost benchmark — mock mode

Mock mode runs the full routing pipeline with simulated completions to measure **routing overhead and cost allocation** without spending real money. Use this to tune thresholds before a live run.

> **Note:** Accuracy figures in mock mode reflect simulated task responses, not real model capability. Use live mode for accuracy measurement. The cost figures and routing split percentages are the meaningful outputs here.

```bash
baar-bench --dataset all --limit 200 --budget 10 --mock \
  --complexity-threshold 0.80 --coding-threshold 0.75 --seed 42
```

| Dataset | Strategy | % routed to cheap | Total cost | Savings vs always-big |
| :--- | :--- | :---: | :---: | :---: |
| **MMLU** | Always big | 0% | $1.0005 | — |
| **MMLU** | **Baar-Core** | **81%** | **$0.157** | **84.3% cheaper** |
| **GSM8K** | Always big | 0% | $1.0005 | — |
| **GSM8K** | **Baar-Core** | **87%** | **$0.129** | **87.1% cheaper** |
| **HumanEval** | Always big | 0% | $1.0005 | — |
| **HumanEval** | **Baar-Core** | **39%** | **$0.614** | **38.6% cheaper** |

HumanEval routes fewer tasks to the cheap tier because coding questions score high complexity — the router correctly identifies them as hard.

### Live benchmark — real API calls (10 tasks per dataset)

```bash
baar-bench --dataset all --limit 10 --budget 2 \
  --complexity-threshold 0.80 --coding-threshold 0.75 --seed 42
```

| Dataset | Strategy | Total cost | Savings vs always-big |
| :--- | :--- | :---: | :---: |
| **MMLU** | Always big | $0.002337 | — |
| **MMLU** | **Baar-Core** | **$0.000137** | **94.1% cheaper** |
| **GSM8K** | Always big | $0.027615 | — |
| **GSM8K** | **Baar-Core** | **$0.002097** | **92.4% cheaper** |
| **HumanEval** | Always big | $0.032125 | — |
| **HumanEval** | **Baar-Core** | **$0.002743** | **91.5% cheaper** |

Run it yourself: `pip install baar-core datasets` then `baar-bench --limit 10 --mock` (free) or add your API key for live results.


## vs. alternatives

| | **Baar-Core** | RouteLLM | LiteLLM | Portkey |
|---|:---:|:---:|:---:|:---:|
| Hard local kill-switch (zero network calls) | ✅ | ❌ | ❌ | ❌ |
| Prevents Denial-of-Wallet (OWASP LLM10:2025) | ✅ | ❌ | ❌ | ❌ |
| Works fully offline | ✅ | ❌ | ❌ | ❌ |
| Per-user namespaced budgets (no proxy) | ✅ SQLite/File | ❌ | ❌ (proxy required) | ❌ (cloud only) |
| Cross-process TOCTOU-safe reservations | ✅ | ❌ | ❌ | N/A |
| Semantic complexity routing | ✅ | ✅ | ✅ | ✅ |
| No proxy / no server required | ✅ | ✅ | ❌ | ❌ |
| Open source (MIT) | ✅ | ✅ | ✅ | ❌ |

The key difference: every alternative routes and tracks. Baar-Core **prevents** — the exception is raised before a single byte leaves your machine.


## Security

Baar-Core maps to [OWASP LLM10:2025 — Unbounded Consumption](https://owasp.org/www-project-top-10-for-llm-applications-2/). The pre-flight kill-switch is a direct mitigation for Denial-of-Wallet attacks: even if an adversary crafts a prompt designed to trigger expensive model calls, the local budget cap catches it before any provider request is made.

Details: [RESEARCH.md](https://github.com/orvi2014/Baar-Core/blob/main/RESEARCH.md)


## Configuration

```python
router = BAARRouter(
    budget=0.10,                    # hard cap in USD
    small_model="gpt-4o-mini",      # cheap tier (any LiteLLM model)
    big_model="gpt-4o",             # capable tier
    complexity_threshold=0.80,      # 0.0–1.0: higher = more traffic to cheap model
    min_cost_threshold=0.0001,      # kill-switch floor — reject if any call costs more
    routing_task_char_limit=500,    # chars sent to routing LLM (head+mid+tail sample)
    use_llm_router=True,            # False = rule-based heuristic only (no routing cost)
    small_fallback_models=["gpt-4o-mini-2024-07-18"],  # failover chain
    big_fallback_models=["gpt-4o-2024-08-06"],
    telemetry_jsonl_path="telemetry.jsonl",  # optional audit log
)
```

**Budget pressure** — as spend approaches the cap, the effective complexity threshold rises automatically. The big model becomes harder to justify as you run low, so more traffic shifts to cheap naturally.

**Telemetry** — inspect spend, routing splits, and reject rates:

```bash
baar-telemetry telemetry.jsonl
```

**Resilience testing** — adversarial scenarios (complexity games, tight budgets, padding attacks):

```bash
baar-stress
```


## License

**MIT** — [LICENSE](https://github.com/orvi2014/Baar-Core/blob/main/LICENSE)
