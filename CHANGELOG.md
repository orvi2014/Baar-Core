# Changelog

All notable changes to baar-core are documented here.

## [Unreleased]

### Fixed
- **`baar_guard` negative `cost_per_call` bug** — `_deduct` used a Python truthiness
  check (`if cost_per_call:`) that evaluates to `True` for negative values. A negative
  `cost_per_call` called `add_spent` with a negative amount, driving the store's `spent`
  counter down and letting `remaining` exceed `total_budget`, silently breaking budget
  integrity. Fixed by requiring `cost_per_call is not None and cost_per_call > 0`.
  First released as 0.7.2 but never merged to `main`; this brings it to the 0.8.x line.
  Two regression tests added.

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
