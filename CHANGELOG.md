# Changelog

All notable changes to baar-core are documented here.

## [Unreleased]

### Added
- **Output-overrun modes** (`on_output_overrun`) — control how the *output* side of a
  call is bounded. Settable on the router and per call (`chat`, `achat`, `stream_chat`,
  `astream_chat`):
  - `"estimate"` (default) — previous behaviour. Reserves the expected output; never
    cuts responses off.
  - `"preflight"` — reserves the worst-case output (`max_output_tokens`, else the model's
    real max output, capped by the room left in the context window). Responses are
    never cut off and the budget is never overshot; calls that can't cover the worst
    case downgrade to the small model or are blocked locally.
  - `"truncate"` — opt-in. Derives `max_tokens` from the remaining budget. The big model
    is used only if its full response fits; otherwise Baar downgrades first and only the
    small model's response can be cut off.
- **`"calibrated"` output mode** — reserves the empirical (1 − β) quantile of logged
  output lengths per (model, step kind) instead of the mean or the worst case. Never
  cuts responses off. Configure with `output_quantiles=QuantileTable(...)` and
  `calibration_beta=` (a float or a `CalibrationResult`); without a usable quantile, or
  when calibration certified nothing (`beta=None`), it behaves exactly like `"preflight"`.
- `baar.calibration` — `QuantileTable` (build from logged calls, JSON save/load),
  `ltt_select(betas, run_fn, n, eps, delta)` (fixed-sequence Learn-then-Test: picks β so
  that P(run failure rate ≤ ε) ≥ 1 − δ from i.i.d. runs), `calibrate_from_failures`, and
  `CalibrationResult`. Exact binomial p-values, no new dependencies.
- `max_output_tokens` (router and per call) — your own output limit, forwarded to the
  provider as `max_tokens`. `min_useful_tokens` (default 64) — `"truncate"` blocks
  locally when fewer tokens are affordable.
- `OutputLimitUnavailable` (subclass of `ValueError`) — raised before any network call
  when `"preflight"` has no known output limit for the small model, or the prompt fills
  the context window. The big model downgrades instead of raising.
- `StepResult.truncated_by_budget` — `True` only when Baar's budget-derived limit cut a
  response off (never for your own `max_output_tokens`). Counted in
  `RoutingLog.summary()["truncated_by_budget_steps"]`.
- Vercel proxy and LangChain `BaarChatModel` forward a request's own `max_tokens` as
  `max_output_tokens`.

### Added (tool calling)
- `BAARRouter.chat` / `achat` accept `tools=` and `tool_choice=`, and new
  `complete()` / `acomplete()` return the provider's full response (tool calls,
  finish reason) under the same budget guardrails. Streaming methods do not take tools yet.

### Fixed
- **Pre-flight prompt counts ignored tool schemas.** Providers bill tool/function
  definitions as prompt tokens, but `token_counter` had no way to include them, and the
  LangChain `BaarCallbackHandler` flattened messages to plain text and never saw tools
  bound with `bind_tools()`. Measured on 5,487 real OpenAI calls with one small tool,
  the estimate was short in 94% of calls (37 tokens on the first step). `token_counter`
  now takes `tools=`, and the callback counts the full message list (per-message
  overhead, assistant tool calls, tool-call ids) plus bound tool schemas from
  `invocation_params`. With tools included the estimate matched OpenAI's billed prompt
  tokens to within one token. The heuristic fallback no longer crashes on messages with
  `content=None`.
- **Tool calling did not work through the OpenAI-compatible proxy** (Vercel and
  Hermes integrations). The proxy dropped `tools`/`tool_choice` from requests, rejected
  assistant messages with `content: null` (every tool-call turn) and multi-part content
  with a 422, and could only return plain text. It now forwards tools (legacy
  `functions`/`function_call` are mapped onto them), keeps `tool_calls`,
  `tool_call_id` and unknown message fields, counts tool schemas in the pre-flight
  estimate, and returns `tool_calls` with the real `finish_reason`. Streaming requests
  with tools run the call under the same guardrails and replay it as standard OpenAI
  chunks. Verified live against OpenAI: tool call, tool result, final answer.

- **Streaming under-recorded spend.** `stream_chat` / `astream_chat` counted only the
  visible text, so hidden reasoning tokens (o-series, GPT-5, Claude and Gemini thinking)
  were never recorded. Streams now request provider usage via
  `stream_options={"include_usage": True}` where LiteLLM supports it, and also read the
  totals LiteLLM attaches to the final chunk (Anthropic). If a stream reports no usage,
  Baar falls back to counting visible text and warns once per model.
- **Failover could overshoot the budget.** The reservation covered only the primary
  model; a pricier fallback ran unreserved. Each fallback is now covered at its own
  price — the reservation is topped up, or the fallback is skipped if that can't be
  afforded.

### Changed
- A pricier fallback that the budget can't cover is now skipped instead of called.
- Fallback models without LiteLLM pricing now emit the same "no pricing data" warning as
  primary models, since their cost is now estimated.

## [0.7.2] — 2026-05-20

### Fixed
- **`baar_guard` negative `cost_per_call` bug** — `_deduct` used a Python truthiness
  check (`if cost_per_call:`) that evaluates to `True` for negative values. Passing a
  negative `cost_per_call` would call `add_spent` with a negative amount, driving the
  store's `spent` counter below zero and allowing `remaining` to exceed `total_budget`,
  silently breaking budget integrity. Fixed by requiring `cost_per_call is not None and
  cost_per_call > 0`. Two regression tests added.

## [0.7.1] — 2026-05-19

### Security
- **CVE-2026-33634** — exclude `litellm==1.82.7` and `litellm==1.82.8` from install-time
  dependency resolution (`!=1.82.7,!=1.82.8`). Both versions contain a supply chain
  compromise. The existing runtime block in `BAARRouter` is preserved as defence-in-depth,
  but the install constraint ensures the compromised package is never downloaded.

## [0.7.0] — 2026-05-19

### Added
- **Policy engine** (`baar.core.policy`) — declarative governance rules evaluated before
  every LLM call. Rules match on any context key or system fact (model, utilization,
  domain, complexity) and produce `block`, `force_small`, `force_big`, or `allow` actions.
  First-match wins; raises `PolicyViolation` (with a `facts` dict) on block.
- **Time-windowed budgets** (`BudgetWindow`) — scope spend caps to `HOURLY`, `DAILY`, or
  `MONTHLY` windows via `BAARRouter(window=BudgetWindow.DAILY)`. Backed by
  `WindowedBudgetStore`, which auto-resets spend each period while preserving historical
  data. Works with all three store backends (Memory, File, SQLite).
- **Alert hooks** (`Alert`) — fire a callback when utilization crosses a threshold.
  `once=True` (default) fires only the first time per window; `once=False` fires on every
  check while above threshold. Alerts re-arm automatically on window rollover and on
  `BudgetTracker.reset()`.
- `BudgetTracker.reset()` — resets spend to zero and re-arms all `once=True` alerts.
- New exports on `baar`: `BudgetWindow`, `Alert`, `WindowedBudgetStore`, `Policy`,
  `PolicyAction`, `Rule`, `PolicyViolation`.

### Changed
- `BAARRouter` accepts three new keyword args: `policy`, `window`, `alerts`.
- Policy-forced model changes set `forced_by_budget=False` and prefix the decision reason
  with `[POLICY FORCE_SMALL]` / `[POLICY FORCE_BIG]` for clear audit trails.
- When a `force_big` policy rule is overridden by budget constraint, a `UserWarning` is
  emitted rather than silently downgrading.
- `PolicyViolation` is treated like `TaskRejected` in `run()`/`arun()` — the task is
  skipped without incrementing the consecutive-error counter.
- `record_manual()` emits a `UserWarning` instead of silently recording $0 when pricing
  data is unavailable for the model.

## [0.6.0] — 2026-05-19

### Added
- **Hermes Agent integration** (`baar.integrations.hermes`) — budget-aware wrapper for
  NousResearch Hermes-style tool-calling agents.
- **CrewAI integration** (`baar.integrations.crewai`) — `BaarCrewCallback` enforces spend
  caps across CrewAI task runs; raises `BudgetExhausted` before runaway crews overshoot.

## [0.5.1] — 2026-05-12

### Added
- **MCP server example** (`examples/mcp_server.py`) — exposes `get_budget_status`, `web_search`, and `run_python` as budget-guarded MCP tools for Claude Code and Cursor. Register via `~/.claude/claude.json`; all three tools are budget-capped and call-limited via `@baar_guard`.
- Test suite for MCP server (`tests/test_mcp_server.py`, 18 tests) — mocks `mcp` and `firecrawl` at import time; no real network calls needed.
- README: new "MCP server" section with setup snippet and `claude.json` registration example.

## [0.5.0] — 2026-05-12

### Added
- **BaarMiddleware** — wraps any compiled LangGraph graph with hard execution limits (`max_steps`, `max_budget`, `recursion_limit`). Raises `StepLimitExceeded` or `BudgetExhausted` before runaway loops can accumulate cost.
- **@baar_guard** — decorator for any sync/async callable. Enforces `max_calls`, `min_budget`, and `cost_per_call` per tool invocation.
- **Vercel integration** (`baar.integrations.vercel`) — drop-in FastAPI proxy that enforces budget before forwarding to OpenAI/Anthropic; ready for one-click Vercel deployment.
- **LangChain guardrail** (`baar.integrations.langchain`) — `BaarCallbackHandler` and `BaarGuardrailChain` for budget-aware LangChain chains.
- New examples: `tool_guard.py`, `langgraph_middleware.py`, `langchain_guardrail.py`, `openai_tool_guard.py`, `claude_tool_guard.py`, `vercel_server.py`.
- CI matrix now includes Python 3.11; coverage reported via Codecov.

### Changed
- `Development Status` classifier: `3 - Alpha` → `5 - Production/Stable`.
- Project homepage updated to `https://baar.robatdasorvi.com`.
- `pyproject.toml` keywords expanded: added `langgraph`, `agent budget`.

## [0.4.1] — 2025-04-18

### Fixed
- Unknown model pricing now falls back to `$0/token` instead of raising `KeyError`.

## [0.4.0] — 2025-04-15

### Added
- LangChain integration (`baar.integrations.langchain`) initial release.
- Async router support (`aroute`).
- Pluggable budget stores with atomic reservation semantics.

## [0.3.0] — 2025-03-20

### Added
- Pluggable budget stores (Redis, SQLite, in-memory).
- Atomic budget reservations to prevent over-spend under concurrency.

## [0.2.5] — 2025-03-05

### Changed
- Improved routing calibration; benchmark docs refreshed.

## [0.2.4] — 2025-02-28

### Fixed
- Remove false-rejection affordability re-check on fallback models.

## [0.2.2] — 2025-02-20

### Added
- Configurable `min_cost` threshold with dynamic guard.

### Fixed
- Pre-flight budget kill-switch messaging edge cases.
- Suppress litellm debug output and provider hints in logs.

## [0.2.0] — 2025-02-10

### Added
- Initial public release.
- `BAARRouter` with heuristic and LLM-based routing.
- Hard budget cap with `BudgetExhausted` exception.
- LLM routing fallback on API error.
