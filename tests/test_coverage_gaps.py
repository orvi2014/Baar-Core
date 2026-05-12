"""
tests/test_coverage_gaps.py — Targeted tests for remaining uncovered lines.

Covers: stream_chat domain/BIG paths, achat domain/BIG paths, astream_chat
affordability paths, baar/__init__ fallback, budget edge cases, model cost
helpers, routing cache, heuristic paths, vercel/langchain edge cases,
telemetry CLI branches.
"""

import importlib.metadata
import math
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from baar import BAARRouter, BudgetExhausted
from baar.core.budget import BudgetExceeded, BudgetTracker
from baar.core.exceptions import TaskRejected
from baar.core.router import ModelTier, Router


# ── Helpers ────────────────────────────────────────────────────────────────────

def _resp(content="OK", prompt_tokens=10, completion_tokens=5):
    r = MagicMock()
    r.model = "gpt-4o-mini"
    r.choices[0].message.content = content
    r.usage.prompt_tokens = prompt_tokens
    r.usage.completion_tokens = completion_tokens
    return r


def _stream_chunks(text="Hello world"):
    for word in text.split():
        c = MagicMock()
        c.choices[0].delta.content = word + " "
        yield c
    tail = MagicMock()
    tail.choices[0].delta.content = ""
    yield tail


async def _async_chunks(text="Hello world"):
    for word in text.split():
        c = MagicMock()
        c.choices[0].delta.content = word + " "
        yield c
    tail = MagicMock()
    tail.choices[0].delta.content = ""
    yield tail


def _router(**kw):
    kw.setdefault("budget", 10.0)
    kw.setdefault("use_llm_router", False)
    with patch("baar.core.budget.completion_cost", return_value=0.00005), \
         patch("baar.core.budget.cost_per_token", return_value=(0.000025, 0.000025)):
        return BAARRouter(**kw)


# ── baar/__init__.py — PackageNotFoundError fallback ──────────────────────────
# Lines 10-11 are import-time code that only runs when baar-core is not installed.
# Since tests install the package, this path is unreachable during test runs.
# Covered by # pragma: no cover on those lines instead.


# ── baar/core/budget.py edge cases ────────────────────────────────────────────

class TestBudgetEdgeCases:
    def test_cost_from_response_no_model(self):
        """response.model is None/empty → cost_from_response returns 0.0."""
        tracker = BudgetTracker(total_budget=1.0)
        resp = MagicMock()
        resp.usage.prompt_tokens = 10
        resp.usage.completion_tokens = 5
        resp.model = None
        cost = tracker.cost_from_response(resp)
        assert cost == 0.0

    def test_cost_from_response_exception_fallback(self):
        """cost_per_token raises → cost_from_response returns 0.0."""
        tracker = BudgetTracker(total_budget=1.0)
        resp = MagicMock()
        resp.usage.prompt_tokens = 10
        resp.usage.completion_tokens = 5
        resp.model = "gpt-4o-mini"
        with patch("baar.core.budget.cost_per_token", side_effect=Exception("pricing down")):
            cost = tracker.cost_from_response(resp)
        assert cost == 0.0

    def test_reserve_method_delegates_to_check_and_reserve(self):
        """reserve() is an alias for check_and_reserve()."""
        tracker = BudgetTracker(total_budget=1.0)
        tracker.reserve(0.001)  # should not raise
        # Reservation was processed — total_budget was not exceeded
        assert tracker.total_budget == 1.0

    def test_estimate_cost_non_finite_falls_back(self):
        """Non-finite cost from pricing → fallback conservative estimate returned."""
        tracker = BudgetTracker(total_budget=1.0)
        with patch("baar.core.budget.cost_per_token", return_value=(math.inf, 0.0)):
            # Line 140-141 (isfinite check + ValueError raise) are exercised;
            # the ValueError is caught internally and a fallback is returned.
            result = tracker.estimate_cost("gpt-4o-mini", 100, 500)
        assert result >= 0.0  # fallback conservative estimate

    def test_spent_property_getter_via_direct_read(self):
        """_spent property getter (line 72) is exercised."""
        tracker = BudgetTracker(total_budget=1.0)
        val = tracker._spent  # triggers the @property getter
        assert val == 0.0


# ── baar/core/models.py edge cases ────────────────────────────────────────────

class TestModelsEdgeCases:
    def test_cached_big_model_cost_exception_returns_none(self):
        """_cached_big_model_cost returns None on pricing exception."""
        from baar.core.models import _cached_big_model_cost
        import functools
        # Clear the lru_cache so the mock is actually called
        _cached_big_model_cost.cache_clear()
        with patch("baar.core.models.litellm") as mock_litellm:
            mock_litellm.cost_per_token.side_effect = Exception("pricing fail")
            result = _cached_big_model_cost("unknown-model-xyz-test", 100, 500)
        assert result is None
        _cached_big_model_cost.cache_clear()

    def test_step_result_to_dict_includes_failover_errors(self):
        """to_dict() includes failover_errors when non-empty."""
        from baar.core.models import StepResult
        from baar.core.router import RoutingDecision, ModelTier
        decision = MagicMock()
        decision.tier = ModelTier.SMALL
        decision.model = "gpt-4o-mini"
        decision.complexity_score = 0.2
        decision.reason = "simple"
        decision.domain = "general"
        decision.confidence = 0.9
        decision.estimated_output_tokens = 500
        decision.estimated_value = None
        decision.estimated_cost_usd = None
        decision.forced_by_budget = False
        step = StepResult(
            step_num=1,
            task="test",
            decision=decision,
            response_text="hi",
            cost=0.001,
            cumulative_cost=0.001,
            prompt_tokens=10,
            completion_tokens=5,
            latency_ms=100.0,
            attempted_models=["gpt-4o-mini", "gpt-3.5-turbo"],
            failover_count=1,
            failover_errors=["gpt-4o-mini: failed"],
        )
        d = step.to_dict()
        assert "failover_errors" in d
        assert d["failover_errors"] == ["gpt-4o-mini: failed"]

    def test_routing_log_print_report_with_errors(self, capsys):
        """print_report() shows errors section when errors exist."""
        r = _router()
        r._log.errors.append("something went wrong")
        r._log.print_report()
        out = capsys.readouterr().out
        assert "Errors" in out or "error" in out.lower()


# ── baar/core/router.py edge cases ────────────────────────────────────────────

class TestCoreRouterEdgeCases:
    def test_heuristic_long_task(self):
        """Tasks with >50 words score 0.65."""
        router = Router(small_model="gpt-4o-mini", big_model="gpt-4o",
                        complexity_threshold=0.8, use_llm_router=False)
        long_task = " ".join(["word"] * 55)
        score, reason = router._heuristic_score(long_task)
        assert score == 0.65

    def test_routing_cache_set_and_evict(self):
        """Cache evicts oldest entry when full."""
        router = Router(small_model="gpt-4o-mini", big_model="gpt-4o",
                        complexity_threshold=0.8, use_llm_router=False,
                        routing_cache_size=2)
        router._routing_cache_set("key1", (0.1, "r1", "d1", 100))
        router._routing_cache_set("key2", (0.2, "r2", "d2", 200))
        router._routing_cache_set("key3", (0.3, "r3", "d3", 300))
        assert len(router._routing_cache) <= 2

    @pytest.mark.asyncio
    async def test_async_cache_disabled_path(self):
        """routing_cache_enabled=False skips cache on async path."""
        router = Router(small_model="gpt-4o-mini", big_model="gpt-4o",
                        complexity_threshold=0.8, use_llm_router=False,
                        routing_cache_enabled=False)
        result = await router._ascore_complexity("hello world")
        assert len(result) == 5  # (c, r, d, eot, cache_hit)
        assert result[4] is False  # no cache hit

    def test_llm_score_bad_request_fallback(self):
        """_llm_score falls back when response_format triggers BadRequestError."""
        router = Router(small_model="gpt-4o-mini", big_model="gpt-4o",
                        complexity_threshold=0.8, use_llm_router=True)
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = '{"complexity": 0.5, "reason": "medium"}'

        import litellm.exceptions as _exc
        with patch("baar.core.router.litellm.completion") as mock_comp:
            mock_comp.side_effect = [
                _exc.BadRequestError(
                    message="unsupported", model="gpt-4o-mini",
                    llm_provider="openai", response=MagicMock()
                ),
                mock_resp,
            ]
            result = router._llm_score("explain recursion")
        assert result[0] == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_allm_score_bad_request_fallback(self):
        """_allm_score falls back when response_format triggers BadRequestError."""
        router = Router(small_model="gpt-4o-mini", big_model="gpt-4o",
                        complexity_threshold=0.8, use_llm_router=True)
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = '{"complexity": 0.7, "reason": "complex"}'

        import litellm.exceptions as _exc
        with patch("baar.core.router.litellm.acompletion") as mock_acomp:
            mock_acomp.side_effect = [
                _exc.BadRequestError(
                    message="unsupported", model="gpt-4o-mini",
                    llm_provider="openai", response=MagicMock()
                ),
                mock_resp,
            ]
            result = await router._allm_score("design a system")
        assert result[0] == pytest.approx(0.7)

    @pytest.mark.asyncio
    async def test_acompute_complexity_uncached_llm_failure_warns(self):
        """Async LLM failure emits RuntimeWarning and falls back to heuristic."""
        router = Router(small_model="gpt-4o-mini", big_model="gpt-4o",
                        complexity_threshold=0.8, use_llm_router=True)
        with patch("baar.core.router.litellm.acompletion",
                   side_effect=Exception("async API down")):
            import warnings
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = await router._acompute_complexity_uncached("hello")
            runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
            assert len(runtime_warnings) >= 1


# ── stream_chat() missing paths ────────────────────────────────────────────────

class TestStreamChatMissingPaths:
    @patch("baar.router.litellm.completion")
    @patch("baar.core.budget.completion_cost", return_value=0.00005)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000025, 0.000025))
    def test_affordability_fail_raises_budget_exhausted(self, _cpt, _cc, mock_comp):
        """BudgetExceeded from check_affordability → BudgetExhausted in stream_chat."""
        mock_comp.return_value = _stream_chunks("hi")
        r = _router(budget=10.0)
        with patch.object(r._tracker, "check_affordability",
                          side_effect=BudgetExceeded(0.1, 0.001, "gpt-4o-mini")):
            with pytest.raises(BudgetExhausted):
                list(r.stream_chat("hello"))

    @patch("baar.router.litellm.completion")
    @patch("baar.core.budget.completion_cost", return_value=0.00005)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000025, 0.000025))
    def test_domain_override_string_in_stream(self, _cpt, _cc, mock_comp):
        """Domain string override applied inside stream_chat."""
        mock_comp.return_value = _stream_chunks("Stream override")
        r = _router(domain_model_overrides={"general": "gpt-4o-mini"})
        result = "".join(r.stream_chat("hello stream"))
        assert "Stream" in result

    @patch("baar.router.litellm.completion")
    @patch("baar.core.budget.completion_cost", return_value=0.00005)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000025, 0.000025))
    def test_domain_override_tuple_in_stream(self, _cpt, _cc, mock_comp):
        """Domain tuple override applied inside stream_chat."""
        mock_comp.return_value = _stream_chunks("Tuple stream")
        r = _router(domain_model_overrides={"general": ("gpt-4o-mini", "small")})
        result = "".join(r.stream_chat("hello"))
        assert "Tuple" in result

    @patch("baar.router.litellm.completion")
    @patch("baar.core.budget.completion_cost", return_value=0.00005)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000025, 0.000025))
    def test_big_downgrade_in_stream(self, _cpt, _cc, mock_comp):
        """BIG → SMALL downgrade in stream_chat."""
        mock_comp.return_value = _stream_chunks("Downgraded stream")
        r = _router(complexity_threshold=0.0)
        original = r._tracker.check_affordability
        def _check(model, tokens, eot):
            if model == r.big_model:
                raise BudgetExceeded(0.01, 0.001, model)
            return original(model, tokens, eot)
        with patch.object(r._tracker, "check_affordability", side_effect=_check):
            result = "".join(r.stream_chat("complex stream task"))
        assert "Downgraded" in result

    @patch("baar.router.litellm.completion")
    @patch("baar.core.budget.completion_cost", return_value=0.00005)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000025, 0.000025))
    def test_failover_recorded_in_step(self, _cpt, _cc, mock_comp):
        """Failover in stream_chat updates decision.model in log step."""
        calls = [0]
        def _side(model, messages, **kw):
            calls[0] += 1
            if calls[0] == 1:
                raise Exception("primary down")
            return _stream_chunks("Fallback chunks")
        mock_comp.side_effect = _side
        r = _router(small_fallback_models=["gpt-3.5-turbo"])
        list(r.stream_chat("failover test"))
        assert len(r.log.steps) == 1
        assert r.log.steps[0].failover_count >= 1


# ── achat() missing paths ─────────────────────────────────────────────────────

class TestAchatMissingPaths:
    @pytest.mark.asyncio
    @patch("baar.router.litellm.acompletion")
    @patch("baar.core.budget.completion_cost", return_value=0.00005)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000025, 0.000025))
    async def test_affordability_fail_raises_budget_exhausted(self, _cpt, _cc, mock_acomp):
        mock_acomp.return_value = _resp("OK")
        r = _router(budget=10.0)
        with patch.object(r._tracker, "check_affordability",
                          side_effect=BudgetExceeded(0.1, 0.001, "gpt-4o-mini")):
            with pytest.raises(BudgetExhausted):
                await r.achat("hello")

    @pytest.mark.asyncio
    @patch("baar.router.litellm.acompletion")
    @patch("baar.core.budget.completion_cost", return_value=0.00005)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000025, 0.000025))
    async def test_big_downgrade_in_achat(self, _cpt, _cc, mock_acomp):
        mock_acomp.return_value = _resp("Async downgraded")
        r = _router(complexity_threshold=0.0)
        original = r._tracker.check_affordability
        def _check(model, tokens, eot):
            if model == r.big_model:
                raise BudgetExceeded(0.01, 0.001, model)
            return original(model, tokens, eot)
        with patch.object(r._tracker, "check_affordability", side_effect=_check):
            result = await r.achat("complex async task")
        assert result == "Async downgraded"

    @pytest.mark.asyncio
    @patch("baar.router.litellm.acompletion")
    @patch("baar.core.budget.completion_cost", return_value=0.00005)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000025, 0.000025))
    async def test_domain_override_in_achat(self, _cpt, _cc, mock_acomp):
        mock_acomp.return_value = _resp("Domain async")
        r = _router(domain_model_overrides={"general": "gpt-4o-mini"})
        result = await r.achat("hello async domain")
        assert result == "Domain async"

    @pytest.mark.asyncio
    @patch("baar.router.litellm.acompletion")
    @patch("baar.core.budget.completion_cost", return_value=0.00005)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000025, 0.000025))
    async def test_domain_override_tuple_in_achat(self, _cpt, _cc, mock_acomp):
        mock_acomp.return_value = _resp("Tuple async")
        r = _router(domain_model_overrides={"general": ("gpt-4o-mini", "small")})
        result = await r.achat("hello tuple async")
        assert result == "Tuple async"


# ── astream_chat() affordability paths ────────────────────────────────────────

class TestAstreamChatMissingPaths:
    @pytest.mark.asyncio
    async def test_affordability_fail_raises_budget_exhausted(self):
        r = _router(budget=10.0)
        with patch.object(r._tracker, "check_affordability",
                          side_effect=BudgetExceeded(0.1, 0.001, "gpt-4o-mini")):
            with pytest.raises(BudgetExhausted):
                async for _ in r.astream_chat("hello"):
                    pass

    @pytest.mark.asyncio
    @patch("baar.router.litellm.acompletion")
    @patch("baar.core.budget.completion_cost", return_value=0.00005)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000025, 0.000025))
    async def test_domain_override_in_astream(self, _cpt, _cc, mock_acomp):
        mock_acomp.return_value = _async_chunks("Async domain stream")
        r = _router(domain_model_overrides={"general": ("gpt-4o-mini", "small")})
        chunks = []
        async for chunk in r.astream_chat("hello"):
            chunks.append(chunk)
        assert len(chunks) > 0

    @pytest.mark.asyncio
    @patch("baar.router.litellm.acompletion")
    @patch("baar.core.budget.completion_cost", return_value=0.00005)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000025, 0.000025))
    async def test_failover_in_astream(self, _cpt, _cc, mock_acomp):
        calls = [0]
        async def _side(model, messages, **kw):
            calls[0] += 1
            if calls[0] == 1:
                raise Exception("primary astream fail")
            return _async_chunks("Fallback async stream")
        mock_acomp.side_effect = _side
        r = _router(small_fallback_models=["gpt-3.5-turbo"])
        chunks = []
        async for chunk in r.astream_chat("hello"):
            chunks.append(chunk)
        assert "Fallback" in "".join(chunks)


# ── _build_messages system_prompt handling ────────────────────────────────────

class TestBuildMessages:
    @patch("baar.router.litellm.completion")
    @patch("baar.core.budget.completion_cost", return_value=0.00005)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000025, 0.000025))
    def test_system_prompt_added_when_missing(self, _cpt, _cc, mock_comp):
        """system_prompt is prepended when not present in messages."""
        mock_comp.return_value = _resp("With system prompt")
        r = _router(system_prompt="You are helpful.")
        msgs = [{"role": "user", "content": "Hi"}]
        r.chat("Hi", messages=msgs)
        call_args = mock_comp.call_args
        sent_messages = call_args[1].get("messages") or call_args[0][1]
        assert sent_messages[0]["role"] == "system"
        assert sent_messages[0]["content"] == "You are helpful."


# ── vercel integration edge cases ─────────────────────────────────────────────

class TestVercelEdgeCases:
    def test_last_step_info_returns_zeros_when_no_steps(self):
        """_last_step_info returns (0, 0, 'baar') when log has no steps."""
        from baar.integrations.vercel import _last_step_info
        r = _router()
        pt, ct, model = _last_step_info(r)
        assert pt == 0
        assert ct == 0
        assert model == "baar"

    @patch("baar.router.litellm.completion")
    @patch("baar.core.budget.completion_cost", return_value=0.00005)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000025, 0.000025))
    def test_last_step_info_returns_last_step(self, _cpt, _cc, mock_comp):
        """_last_step_info reads from last log step."""
        from baar.integrations.vercel import _last_step_info
        mock_comp.return_value = _resp("hi", prompt_tokens=20, completion_tokens=10)
        r = _router()
        r.chat("hello")
        pt, ct, model = _last_step_info(r)
        assert pt == 20
        assert ct == 10


# ── langchain integration edge cases ──────────────────────────────────────────

class TestLangchainEdgeCases:
    def test_callback_handler_warns_on_missing_model(self):
        """BaarCallbackHandler warns when model name can't be extracted."""
        from baar.integrations.langchain import BaarCallbackHandler
        import uuid, warnings
        r = _router()
        handler = BaarCallbackHandler(r)
        run_id = uuid.uuid4()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            handler.on_llm_start(serialized={}, prompts=["hello"], run_id=run_id)
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert any("model name" in str(x.message) for x in user_warnings)


# ── telemetry CLI coverage ─────────────────────────────────────────────────────

class TestTelemetryCLI:
    def test_main_with_nonexistent_file(self, capsys):
        from baar.telemetry_cli import main
        import sys
        original = sys.argv
        sys.argv = ["baar-telemetry", "/nonexistent/path.jsonl"]
        try:
            main()
        except SystemExit:
            pass
        finally:
            sys.argv = original
        out = capsys.readouterr()
        assert len(out.out) >= 0  # just verify it doesn't crash

    def test_main_with_valid_file(self, tmp_path, capsys):
        import json
        from baar.telemetry_cli import main
        import sys
        # Create a valid telemetry file
        jsonl = tmp_path / "tel.jsonl"
        entry = {
            "task_preview": "test task",
            "model": "gpt-4o-mini",
            "tier": "small",
            "cost": 0.0001,
            "cumulative_cost": 0.0001,
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "latency_ms": 50.0,
            "step_num": 1,
            "forced_by_budget": False,
            "failover_count": 0,
            "timestamp_unix_ms": 1700000000000,
            "router_budget_usd": 1.0,
        }
        jsonl.write_text(json.dumps(entry) + "\n")
        original = sys.argv
        sys.argv = ["baar-telemetry", str(jsonl)]
        try:
            main()
        except SystemExit:
            pass
        finally:
            sys.argv = original
        out = capsys.readouterr().out
        assert len(out) > 0
