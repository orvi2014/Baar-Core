# Changelog

All notable changes to baar-core are documented here.

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
