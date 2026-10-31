# Changelog

All notable changes to baar-core are documented here.

## [Unreleased]

### Added
- **`on_output_overrun="continue"`** — the answer is generated in segments of
  `segment_tokens` (default 2000). Each segment reserves only its own cost (unlike
  `"truncate"`, which can reserve the whole remaining budget), and a segment that stops at
  its limit is continued with the text so far. The answer ends when the model finishes,
  `max_output_tokens` (the total) is reached, `max_segments` (default 8) is hit, or the
  budget can't pay for another useful segment — then the text so far is returned and the
  last step is flagged `truncated_by_budget`. A value-gate rejection or policy block on a
  later segment also returns the text so far. Works on all four entry points; each segment
  is its own step in the log.
- **OverThink benchmark** (`benchmarks/overthink_eval.py`) — a deterministic agent run under
  inflated hidden reasoning (OverThink, arXiv 2502.02542) against a simulated reasoning
  model, comparing post-hoc alerts and each output mode on overshoot, cut-off answers,
  tasks done and budget used. Seeds 1–30: alerts and `estimate` overshoot; `preflight`,
  `preflight` + `thinking_budget` and `truncate` never do; `preflight` never cuts an answer.
- **`overrun_risk`** (default 0) — in `"preflight"`, reserve the split-conformal
  (1 − risk) quantile of the output lengths each model actually produced, instead of its
  full max output. `max_tokens` is unchanged, so responses are never cut off; at most about
  `overrun_risk` of calls can run past the reservation. Until a model has 20 completed
  calls (`OutputLengths(min_samples=20)`), the worst case is reserved. Lengths are learned
  per model from completed calls (streams stopped early don't count) in a 500-call window;
  `BAARRouter(output_lengths=OutputLengths())` shares them between routers, and
  `router.output_lengths` exposes them.
- **`thinking_budget`** (router and per call, default off, minimum 1024) — caps a reasoning
  model's hidden thinking without cutting the visible answer. On token-budget providers
  (Claude, Gemini, DeepSeek) Baar sends `thinking={"type": "enabled", "budget_tokens": N}`
  and `max_tokens` = answer limit + N, so `max_output_tokens` is the answer's own limit and
  `preflight` reserves answer + thinking instead of the model's full max output. On
  effort-level providers (OpenAI o-series, GPT-5) it maps to `reasoning_effort`
  (advisory; the reservation is unchanged). Other models ignore it.

## [0.9.0] — 2026-10-31

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
- `max_output_tokens` (router and per call) — your own output limit, forwarded to the
  provider as `max_tokens`. `min_useful_tokens` (default 64) — `"truncate"` blocks
  locally when fewer tokens are affordable.
- `OutputLimitUnavailable` (subclass of `ValueError`) — raised before any network call
  when `"preflight"` has no known output limit for the small model, or the prompt fills
  the context window. The big model downgrades instead of raising.
- `StepResult.truncated_by_budget` — `True` only when Baar's budget-derived limit cut a
  response off (never for your own `max_output_tokens`). Counted in
  `RoutingLog.summary()["truncated_by_budget_steps"]`.
- `max_call_share` (default 1.0) — in `"truncate"` mode, the largest share of the
  remaining budget one call may reserve, so concurrent calls near the cap aren't all
  blocked by the first one.
- Vercel proxy, LangChain `BaarChatModel` and CrewAI `BaarCrewLLM` forward the request's
  or LLM's own `max_tokens` as `max_output_tokens`. The Vercel proxy also accepts
  `max_completion_tokens` (used by newer OpenAI SDKs; it wins over `max_tokens`) and
  returns `400 invalid_request_error` for a non-positive limit instead of a 500.
- `prompt_margin` (default 0.2) — `"preflight"` and `"truncate"` price the prompt at the
  estimate plus 20%, since Baar's token count is approximate for Claude, Gemini, images
  and tool schemas. The guarantee is "never overshoots, apart from a prompt undercount
  larger than the margin". `prompt_margin=0` turns it off. The context-window room
  still uses the raw estimate.
- `StepResult.finish_reason` — the provider's finish reason (`"stop"`, `"length"`, …).
  The Vercel proxy now returns it, so clients see `"length"` when a reply was cut off
  instead of always `"stop"`, and maps `OutputLimitUnavailable` to `400
  invalid_request_error` (an error chunk when streaming).
- `loop_nudge(event)` — the note text, for frameworks that add it to the prompt themselves.
- `LoopDetector.peek(tool, args)` — the stop `before()` would raise, without raising or
  reporting it.
- Tool-call arguments given as a JSON string compare equal to the same arguments as a dict.

### Fixed
- **Streaming under-recorded spend.** `stream_chat` / `astream_chat` counted only the
  visible text, so hidden reasoning tokens (o-series, GPT-5, Claude and Gemini thinking)
  were never recorded. Streams now request provider usage via
  `stream_options={"include_usage": True}` where LiteLLM supports it, and also read the
  totals LiteLLM attaches to the final chunk (Anthropic). If a stream reports no usage —
  including LiteLLM's placeholder `(0, 0)` when the provider sent none — Baar falls back
  to counting visible text and warns once per model.
- **Stopping a stream early.** When the caller stops reading, Baar now closes the
  provider stream so the request ends, and in `"preflight"` / `"truncate"` records the
  reserved output (the provider may have generated more than was read) instead of only
  the visible text.
- **`max_output_tokens` above the model's real max** was sent as-is and rejected by the
  provider. It is now capped at the model's max output when LiteLLM knows it.
- **`"truncate"` ignored tiered pricing.** The output price was read at prompt length 0,
  so models whose rates rise with prompt length (e.g. Gemini 2.5 Pro above 200k tokens)
  were planned at the cheaper rate and the call was then rejected. The price now uses
  the real prompt length.
- **The value gate's downshift check ignored the output mode.** Value is still compared
  with the *expected* cost (a worst-case reservation shouldn't make a cheap task look
  unprofitable), but the affordability check after a downshift now uses the mode's
  planned output.
- **Failover could overshoot the budget.** The reservation covered only the primary
  model; a pricier fallback ran unreserved. Each fallback is now covered at its own
  price — the reservation is topped up, or the fallback is skipped if that can't be
  afforded.
- **`baar_guard` negative `cost_per_call` bug** — `_deduct` used a Python truthiness
  check (`if cost_per_call:`) that evaluates to `True` for negative values. A negative
  `cost_per_call` called `add_spent` with a negative amount, driving the store's `spent`
  counter down and letting `remaining` exceed `total_budget`, silently breaking budget
  integrity. Fixed by requiring `cost_per_call is not None and cost_per_call > 0`.
  First released as 0.7.2 but never merged to `main`; this brings it to the 0.8.x line.
  Two regression tests added.

### Changed
- A pricier fallback that the budget can't cover is now skipped instead of called.
- Fallback models without LiteLLM pricing now emit the same "no pricing data" warning as
  primary models, since their cost is now estimated.
- **Loop detection stops earlier and tells the model first** (ported from noburn SDK 0.6.1
  so the two match again):
  - `LoopDetector(max_repeats=...)` default **4 → 2**: the 3rd identical call (same
    arguments, same result) is refused. Error and ping-pong defaults are unchanged.
  - The `on_loop` warning now fires on the step after which the next identical one is
    refused — the first repeat by default — instead of one occurrence before that.
  - `baar_guard` appends a note to a **string** tool result when the detector warns, so the
    model sees it in its next turn and can change course. `loop_nudge=False` turns this off;
    non-string results are never changed. In noburn's live benchmark, models changed course
    in 19 of 20 warned runs; stopping without the note did worse than no detection.
- **Waiting on a job is not a loop.** `LoopDetector(in_progress=...)`: short results
  (≤ 200 characters) saying running, pending, queued, in progress, processing, booting,
  starting, initializing, waiting or not ready never count. Pass a compiled pattern to
  change the words, or `False` to count them. On `benchmarks/loop_eval.py` the default
  detector's false stops drop from 14.3% to 0% without `exempt_tools`, and the median wasted
  calls from 3 to 2, on every seed from 1 to 50.

## [0.8.0] — 2026-10-30

### Added
- **Loop detection** (`baar.core.loop`) — `LoopDetector` watches an agent run and
  refuses the call that would continue a loop *before it runs*, raising
  `LoopDetected` (with `pattern`, `tool_name`, `count`). Each step is reduced to
  (tool, normalized arguments) → hash of the result or error, so repetition that
  makes progress is never counted. Patterns: `repeat` (same call, same result,
  `max_repeats=4`), `error` (same call, same error in a row, `max_error_repeats=3`;
  a success resets it), `ping_pong` (two steps alternating, `max_ping_pong=3`
  cycles). `on_loop` gets a `LoopEvent` warning one step before a stop.
  Options: `exempt_tools`, `ignore_args`, `normalize_result`, `warn`; `tighten()`
  after history summarization; `reset()` per run. Thread-safe.
- `baar_guard(loop_detector=...)` — feeds each tool call's arguments and result
  (or error) to the detector; a refused call is not made and does not count
  against `max_calls`.
- `benchmarks/loop_eval.py` — 180 labeled deterministic traces (75 loops, 105
  healthy look-alikes, including a flaky API failing on and off) comparing the
  detector with call-hash counting, `max_calls` and a run budget. Detector +
  `exempt_tools` + budget: 100% of loops stopped, 0% of healthy runs stopped,
  median 3 wasted calls, for every seed from 1 to 50.
- New exports on `baar`: `LoopDetector`, `LoopDetected`, `LoopEvent`.

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
