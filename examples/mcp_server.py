"""
examples/mcp_server.py — Baar-Core MCP server for Claude Code & Cursor.

Exposes 3 budget-guarded tools to any MCP client:
  • get_budget_status  — check remaining budget and call counts
  • web_search         — DuckDuckGo search (max 20 calls, $0.001/call)
  • run_python         — sandboxed subprocess exec (max 10 calls, needs $0.005)

Setup:
    pip install baar-core mcp httpx

Register with Claude Code — add to ~/.claude/claude.json:
    {
      "mcpServers": {
        "baar": {
          "command": "python3",
          "args": ["/Users/orvi/Baar-Core/examples/mcp_server.py"],
          "env": { "BAAR_BUDGET": "0.10" }
        }
      }
    }

Then restart Claude Code — it will show the 3 tools automatically.
"""

import os
import json
import asyncio
import subprocess
from mcp.server.fastmcp import FastMCP
from firecrawl import FirecrawlApp as _FC

from baar import BAARRouter, BudgetExhausted
from baar.integrations.tools import baar_guard, ToolCallLimitExceeded

# ── Budget setup ──────────────────────────────────────────────────────────────

BUDGET = float(os.environ.get("BAAR_BUDGET", "0.10"))
router = BAARRouter(budget=BUDGET, use_llm_router=False)
_firecrawl = _FC(api_key=os.environ.get("FIRECRAWL_API_KEY"))

# ── Guarded tools ─────────────────────────────────────────────────────────────

@baar_guard(router=router, max_calls=20, cost_per_call=0.001)
async def _web_search(query: str) -> str:
    results = _firecrawl.search(query, limit=5)
    items = (results or {}).get("data", {}).get("web", [])
    if not items:
        return "No results found."
    lines = [f"{r.get('title','')} — {r.get('url','')}\n{r.get('description','')[:200]}" for r in items]
    return "\n\n".join(lines)[:800]


@baar_guard(router=router, max_calls=10, min_budget=0.005)
async def _run_python(code: str) -> str:
    result = subprocess.run(
        ["python3", "-c", code],
        capture_output=True, text=True, timeout=10,
    )
    output = result.stdout or result.stderr or "(no output)"
    return output[:1000]


# ── MCP server ────────────────────────────────────────────────────────────────

mcp = FastMCP("baar-core")


@mcp.tool()
def get_budget_status() -> str:
    """Return remaining Baar-Core budget and per-tool call counts."""
    return json.dumps({
        "budget_usd":        round(router.budget, 4),
        "remaining_usd":     round(router.remaining, 4),
        "spent_usd":         round(router.budget - router.remaining, 6),
        "web_search_calls":  _web_search._call_count,
        "run_python_calls":  _run_python._call_count,
        "web_search_limit":  20,
        "run_python_limit":  10,
    }, indent=2)


@mcp.tool()
async def web_search(query: str) -> str:
    """
    Search the web via DuckDuckGo instant answers.
    Hard limits: max 20 calls per session, $0.001 deducted per call.
    Raises if budget or call limit is exhausted.
    """
    try:
        return await _web_search(query)
    except ToolCallLimitExceeded as e:
        return f"❌ web_search limit reached ({e.calls}/{e.max_calls} calls used)."
    except BudgetExhausted as e:
        return f"❌ Budget exhausted. Remaining: ${e.remaining:.6f}"


@mcp.tool()
async def run_python(code: str) -> str:
    """
    Execute Python code in a subprocess and return stdout/stderr.
    Hard limits: max 10 calls per session, requires $0.005 remaining budget.
    """
    try:
        return await _run_python(code)
    except ToolCallLimitExceeded as e:
        return f"❌ run_python limit reached ({e.calls}/{e.max_calls} calls used)."
    except BudgetExhausted as e:
        return f"❌ Budget exhausted. Remaining: ${e.remaining:.6f}"


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Baar-Core MCP server starting (budget: ${BUDGET:.2f})", flush=True)
    mcp.run()
