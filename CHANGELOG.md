# Changelog

All notable changes to baar-core are documented here.

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
