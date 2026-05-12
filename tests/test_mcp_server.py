"""
tests/test_mcp_server.py — Unit tests for examples/mcp_server.py.

Mocks mcp.server.fastmcp and firecrawl at import time so no real network
calls or MCP server startup is needed.
"""

import importlib
import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── helpers ───────────────────────────────────────────────────────────────────

MCP_SERVER_PATH = Path(__file__).parent.parent / "examples" / "mcp_server.py"


def _load_mcp_module():
    """Import examples/mcp_server.py with mcp and firecrawl stubbed out."""
    # Build minimal mcp stub
    mcp_stub = types.ModuleType("mcp")
    mcp_server_stub = types.ModuleType("mcp.server")
    mcp_fastmcp_stub = types.ModuleType("mcp.server.fastmcp")

    class _FakeMCP:
        def __init__(self, name):
            self.name = name
            self._tools = {}

        def tool(self):
            def decorator(fn):
                self._tools[fn.__name__] = fn
                return fn
            return decorator

        def run(self):
            pass

    mcp_fastmcp_stub.FastMCP = _FakeMCP
    mcp_stub.server = mcp_server_stub
    mcp_server_stub.fastmcp = mcp_fastmcp_stub

    # Build minimal firecrawl stub
    firecrawl_stub = types.ModuleType("firecrawl")

    class _FakeFirecrawl:
        def __init__(self, api_key=None):
            self.api_key = api_key
            self.search = MagicMock(return_value={})

    firecrawl_stub.FirecrawlApp = _FakeFirecrawl

    # Inject stubs before importing
    sys.modules.setdefault("mcp", mcp_stub)
    sys.modules.setdefault("mcp.server", mcp_server_stub)
    sys.modules.setdefault("mcp.server.fastmcp", mcp_fastmcp_stub)
    sys.modules.setdefault("firecrawl", firecrawl_stub)

    # Force fresh load each time (remove cached module if present)
    mod_name = "examples.mcp_server"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    # Also clear by file path key if loaded that way
    for key in list(sys.modules):
        mod = sys.modules[key]
        if hasattr(mod, "__file__") and mod.__file__ and "mcp_server" in str(mod.__file__):
            del sys.modules[key]
            break

    spec = importlib.util.spec_from_file_location("examples.mcp_server", MCP_SERVER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def mcp_mod():
    return _load_mcp_module()


# ── get_budget_status ─────────────────────────────────────────────────────────

class TestGetBudgetStatus:
    def test_returns_valid_json(self, mcp_mod):
        result = mcp_mod.get_budget_status()
        data = json.loads(result)
        assert "budget_usd" in data
        assert "remaining_usd" in data
        assert "spent_usd" in data

    def test_budget_matches_router(self, mcp_mod):
        data = json.loads(mcp_mod.get_budget_status())
        assert data["budget_usd"] == round(mcp_mod.router.budget, 4)

    def test_call_counts_present(self, mcp_mod):
        data = json.loads(mcp_mod.get_budget_status())
        assert "web_search_calls" in data
        assert "run_python_calls" in data
        assert data["web_search_limit"] == 20
        assert data["run_python_limit"] == 10

    def test_spent_plus_remaining_equals_budget(self, mcp_mod):
        data = json.loads(mcp_mod.get_budget_status())
        assert abs(data["spent_usd"] + data["remaining_usd"] - data["budget_usd"]) < 1e-9

    def test_initial_spent_is_zero(self, mcp_mod):
        data = json.loads(mcp_mod.get_budget_status())
        assert data["spent_usd"] == 0.0

    def test_initial_call_counts_are_zero(self, mcp_mod):
        data = json.loads(mcp_mod.get_budget_status())
        assert data["web_search_calls"] == 0
        assert data["run_python_calls"] == 0


# ── web_search ────────────────────────────────────────────────────────────────

class TestWebSearch:
    @pytest.mark.asyncio
    async def test_returns_result_string(self, mcp_mod):
        async def _fake_search(query):
            return "Result: some content"

        with patch.object(mcp_mod, "_web_search", new=AsyncMock(return_value="Result: some content")):
            result = await mcp_mod.web_search("test query")
        assert "Result" in result

    @pytest.mark.asyncio
    async def test_tool_call_limit_exceeded_returns_message(self, mcp_mod):
        from baar.integrations.tools import ToolCallLimitExceeded

        exc = ToolCallLimitExceeded(calls=20, max_calls=20, tool_name="_web_search")
        with patch.object(mcp_mod, "_web_search", new=AsyncMock(side_effect=exc)):
            result = await mcp_mod.web_search("anything")
        assert "limit reached" in result
        assert "20/20" in result

    @pytest.mark.asyncio
    async def test_budget_exhausted_returns_message(self, mcp_mod):
        from baar import BudgetExhausted

        exc = BudgetExhausted("over budget", remaining=0.0)
        with patch.object(mcp_mod, "_web_search", new=AsyncMock(side_effect=exc)):
            result = await mcp_mod.web_search("anything")
        assert "Budget exhausted" in result
        assert "0.000000" in result


# ── run_python ────────────────────────────────────────────────────────────────

class TestRunPython:
    @pytest.mark.asyncio
    async def test_returns_output(self, mcp_mod):
        with patch.object(mcp_mod, "_run_python", new=AsyncMock(return_value="42\n")):
            result = await mcp_mod.run_python("print(42)")
        assert "42" in result

    @pytest.mark.asyncio
    async def test_tool_call_limit_exceeded_returns_message(self, mcp_mod):
        from baar.integrations.tools import ToolCallLimitExceeded

        exc = ToolCallLimitExceeded(calls=10, max_calls=10, tool_name="_run_python")
        with patch.object(mcp_mod, "_run_python", new=AsyncMock(side_effect=exc)):
            result = await mcp_mod.run_python("print(1)")
        assert "limit reached" in result
        assert "10/10" in result

    @pytest.mark.asyncio
    async def test_budget_exhausted_returns_message(self, mcp_mod):
        from baar import BudgetExhausted

        exc = BudgetExhausted("over budget", remaining=0.001)
        with patch.object(mcp_mod, "_run_python", new=AsyncMock(side_effect=exc)):
            result = await mcp_mod.run_python("print(1)")
        assert "Budget exhausted" in result

    @pytest.mark.asyncio
    async def test_executes_real_subprocess(self, mcp_mod):
        """Integration-style: actually calls subprocess.run via _run_python internals."""
        import subprocess

        fake_result = MagicMock()
        fake_result.stdout = "hello\n"
        fake_result.stderr = ""

        with patch("subprocess.run", return_value=fake_result):
            # Reset call count so guard doesn't block
            mcp_mod._run_python._call_count = 0
            result = await mcp_mod._run_python("print('hello')")
        assert "hello" in result


# ── baar_guard integration ────────────────────────────────────────────────────

class TestBaarGuardIntegration:
    def test_web_search_has_call_count_attr(self, mcp_mod):
        assert hasattr(mcp_mod._web_search, "_call_count")

    def test_run_python_has_call_count_attr(self, mcp_mod):
        assert hasattr(mcp_mod._run_python, "_call_count")

    def test_router_uses_no_llm_routing(self, mcp_mod):
        assert mcp_mod.router._config.use_llm_router is False

    def test_router_budget_from_env(self, monkeypatch):
        monkeypatch.setenv("BAAR_BUDGET", "0.25")
        mod = _load_mcp_module()
        assert mod.router.budget == pytest.approx(0.25)

    def test_router_budget_default(self, monkeypatch):
        monkeypatch.delenv("BAAR_BUDGET", raising=False)
        mod = _load_mcp_module()
        assert mod.router.budget == pytest.approx(0.10)
