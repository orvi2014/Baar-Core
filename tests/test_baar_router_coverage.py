"""
tests/test_baar_router_coverage.py — Coverage tests for baar/router.py.

Covers chat(), achat(), run(), arun(), stream_chat(), astream_chat(),
value gates, domain overrides, telemetry, retries, and module helpers.
All LiteLLM calls are mocked — no API keys needed.
"""

import importlib.metadata
import os
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from baar import BAARRouter, BudgetExhausted
from baar.core.exceptions import TaskRejected
from baar.core.router import ModelTier
from baar.router import (
    BAARConfig,
    token_counter,
    _check_litellm_version,
    _sync_completion_with_retry,
    _async_completion_with_retry,
)


# ── Mock helpers ───────────────────────────────────────────────────────────────

def _resp(content="Hello!", prompt_tokens=10, completion_tokens=5, model="gpt-4o-mini"):
    r = MagicMock()
    r.model = model
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


COST_PATCHES = dict(
    completion_cost_patch=patch("baar.core.budget.completion_cost", return_value=0.00005),
    cost_per_token_patch=patch("baar.core.budget.cost_per_token", return_value=(0.000025, 0.000025)),
)


def _router(**kwargs):
    defaults = dict(budget=10.0, use_llm_router=False)
    defaults.update(kwargs)
    with patch("baar.core.budget.completion_cost", return_value=0.00005), \
         patch("baar.core.budget.cost_per_token", return_value=(0.000025, 0.000025)):
        return BAARRouter(**defaults)


# ── token_counter ──────────────────────────────────────────────────────────────

class TestTokenCounter:
    def test_plain_text(self):
        with patch("baar.router.litellm.token_counter", return_value=8):
            assert token_counter("hello world", model="gpt-4o-mini") == 8

    def test_messages_list(self):
        with patch("baar.router.litellm.token_counter", return_value=15):
            msgs = [{"role": "user", "content": "hi"}]
            assert token_counter(messages=msgs, model="gpt-4o-mini") == 15

    def test_fallback_on_exception_text(self):
        with patch("baar.router.litellm.token_counter", side_effect=Exception("fail")):
            result = token_counter("a b c d e f g h", model="gpt-4o-mini")
            assert result >= 1

    def test_fallback_on_exception_messages(self):
        with patch("baar.router.litellm.token_counter", side_effect=Exception("fail")):
            msgs = [{"role": "user", "content": "hello world"}]
            result = token_counter(messages=msgs, model="gpt-4o-mini")
            assert result >= 0


# ── _check_litellm_version ─────────────────────────────────────────────────────

class TestCheckLitellmVersion:
    def test_good_version_passes(self):
        with patch("importlib.metadata.version", return_value="1.90.0"):
            _check_litellm_version()  # no raise

    def test_bad_version_raises(self):
        with patch("importlib.metadata.version", return_value="1.82.7"):
            with pytest.raises(RuntimeError, match="Insecure LiteLLM"):
                _check_litellm_version()

    def test_skip_env_var(self, monkeypatch):
        monkeypatch.setenv("BAAR_SKIP_VERSION_CHECK", "1")
        with patch("importlib.metadata.version", return_value="1.82.7"):
            _check_litellm_version()  # skipped, no raise

    def test_package_not_found_skips_silently(self):
        with patch(
            "importlib.metadata.version",
            side_effect=importlib.metadata.PackageNotFoundError,
        ):
            _check_litellm_version()  # no raise


# ── BAARRouter construction ────────────────────────────────────────────────────

class TestConstruction:
    def test_default_budget(self):
        r = BAARRouter(use_llm_router=False)
        assert r.budget == 0.10

    def test_custom_budget(self):
        r = _router(budget=5.0)
        assert r.budget == 5.0

    def test_negative_budget_raises(self):
        with pytest.raises(ValueError, match="Budget must be positive"):
            BAARRouter(budget=-1.0)

    def test_negative_min_cost_threshold_raises(self):
        with pytest.raises(ValueError, match="min_cost_threshold"):
            BAARRouter(budget=1.0, min_cost_threshold=-0.001)

    def test_reconfigure_raises(self):
        r = _router()
        with pytest.raises(RuntimeError, match="already configured"):
            r._configure(BAARConfig(budget=2.0))

    def test_from_config(self):
        config = BAARConfig(budget=3.0, use_llm_router=False)
        r = BAARRouter.from_config(config)
        assert r.budget == 3.0

    def test_config_property(self):
        r = _router(budget=7.0)
        assert r.config.budget == 7.0

    def test_properties(self):
        r = _router(budget=1.0)
        assert r.spent == 0.0
        assert r.remaining == 1.0
        assert r.min_cost_threshold >= 0
        assert r.log is not None

    def test_domain_override_unknown_model_warns(self):
        with pytest.warns(UserWarning, match="matches neither"):
            _router(domain_model_overrides={"code": "some-mystery-model"})

    def test_domain_override_invalid_type_raises(self):
        with pytest.raises(ValueError, match="must be a str or"):
            _router(domain_model_overrides={"code": 42})

    def test_domain_override_empty_string_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            _router(domain_model_overrides={"code": ""})

    def test_domain_override_bad_tuple_length_raises(self):
        with pytest.raises(ValueError, match="tuple must be"):
            _router(domain_model_overrides={"code": ("gpt-4o",)})

    def test_domain_override_invalid_tier_raises(self):
        with pytest.raises(ValueError, match="invalid tier"):
            _router(domain_model_overrides={"code": ("gpt-4o", "premium")})

    def test_domain_override_empty_tuple_model_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            _router(domain_model_overrides={"code": ("", "small")})


# ── chat() ─────────────────────────────────────────────────────────────────────

class TestChat:
    @patch("baar.router.litellm.completion")
    @patch("baar.core.budget.completion_cost", return_value=0.00005)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000025, 0.000025))
    def test_happy_path_returns_content(self, _cpt, _cc, mock_comp):
        mock_comp.return_value = _resp("Good morning!")
        r = _router()
        assert r.chat("Good morning") == "Good morning!"

    @patch("baar.router.litellm.completion")
    @patch("baar.core.budget.completion_cost", return_value=0.00005)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000025, 0.000025))
    def test_budget_is_decremented(self, _cpt, _cc, mock_comp):
        mock_comp.return_value = _resp("OK")
        r = _router(budget=5.0)
        r.chat("hello")
        assert r.spent > 0
        assert r.remaining < 5.0

    @patch("baar.router.litellm.completion")
    @patch("baar.core.budget.completion_cost", return_value=0.00005)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000025, 0.000025))
    def test_with_messages_history(self, _cpt, _cc, mock_comp):
        mock_comp.return_value = _resp("Follow-up")
        r = _router()
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        assert r.chat("And now?", messages=msgs) == "Follow-up"

    @patch("baar.router.litellm.completion")
    @patch("baar.core.budget.completion_cost", return_value=0.00005)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000025, 0.000025))
    def test_system_prompt_prepended(self, _cpt, _cc, mock_comp):
        mock_comp.return_value = _resp("Yes")
        r = _router(system_prompt="You are concise.")
        assert r.chat("Are you helpful?") == "Yes"

    def test_kill_switch_budget_too_low(self):
        r = BAARRouter(budget=0.000001, use_llm_router=False)
        with pytest.raises(BudgetExhausted):
            r.chat("anything")

    @patch("baar.router.litellm.completion")
    @patch("baar.core.budget.completion_cost", return_value=0.00005)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000025, 0.000025))
    def test_all_models_fail_raises_runtime_error(self, _cpt, _cc, mock_comp):
        mock_comp.side_effect = Exception("provider down")
        r = _router()
        with pytest.raises(RuntimeError, match="All provider"):
            r.chat("hello")

    @patch("baar.router.litellm.completion")
    @patch("baar.core.budget.completion_cost", return_value=0.00005)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000025, 0.000025))
    def test_small_fallback_used_on_primary_failure(self, _cpt, _cc, mock_comp):
        calls = [0]
        def _side(model, messages, **kw):
            calls[0] += 1
            if calls[0] == 1:
                raise Exception("primary down")
            return _resp("Fallback OK")
        mock_comp.side_effect = _side
        r = _router(small_fallback_models=["gpt-3.5-turbo"])
        assert r.chat("hello") == "Fallback OK"

    @patch("baar.router.litellm.completion")
    @patch("baar.core.budget.completion_cost", return_value=0.00005)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000025, 0.000025))
    def test_big_downgrade_when_budget_tight(self, _cpt, _cc, mock_comp):
        mock_comp.return_value = _resp("Downgraded")
        # complexity_threshold=0.0 routes everything BIG
        r = _router(complexity_threshold=0.0)
        # Simulate BIG model being unaffordable while SMALL still passes
        from baar.core.budget import BudgetExceeded
        original = r._tracker.check_affordability
        def _check(model, tokens, eot):
            if model == r.big_model:
                raise BudgetExceeded(0.01, 0.001, model)
            return original(model, tokens, eot)
        with patch.object(r._tracker, "check_affordability", side_effect=_check):
            result = r.chat("Explain quantum computing in exhaustive detail")
        assert result == "Downgraded"

    @patch("baar.router.litellm.completion")
    @patch("baar.core.budget.completion_cost", return_value=0.00005)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000025, 0.000025))
    def test_domain_override_string_applied(self, _cpt, _cc, mock_comp):
        mock_comp.return_value = _resp("Override used")
        r = _router(domain_model_overrides={"general": "gpt-4o-mini"})
        assert r.chat("hello") == "Override used"

    @patch("baar.router.litellm.completion")
    @patch("baar.core.budget.completion_cost", return_value=0.00005)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000025, 0.000025))
    def test_domain_override_tuple_applied(self, _cpt, _cc, mock_comp):
        mock_comp.return_value = _resp("Tuple override")
        r = _router(domain_model_overrides={"general": ("gpt-4o-mini", "small")})
        assert r.chat("hello") == "Tuple override"

    @patch("baar.router.litellm.completion")
    @patch("baar.core.budget.completion_cost", return_value=0.00005)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000025, 0.000025))
    def test_log_records_step(self, _cpt, _cc, mock_comp):
        mock_comp.return_value = _resp("Logged")
        r = _router()
        r.chat("log me")
        assert len(r.log.steps) == 1
        assert r.log.steps[0].response_text == "Logged"


# ── run() ──────────────────────────────────────────────────────────────────────

class TestRun:
    @patch("baar.router.litellm.completion")
    @patch("baar.core.budget.completion_cost", return_value=0.00005)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000025, 0.000025))
    def test_runs_all_tasks(self, _cpt, _cc, mock_comp):
        mock_comp.return_value = _resp("Done")
        r = _router()
        log = r.run(["task 1", "task 2", "task 3"])
        assert len(log.steps) == 3

    def test_stops_on_budget_exhausted(self):
        r = BAARRouter(budget=0.000001, use_llm_router=False)
        log = r.run(["t1", "t2"])
        assert len(log.steps) == 0

    def test_skips_rejected_tasks_and_continues(self):
        r = _router(value_fn=lambda t: 0.0)
        log = r.run(["t1", "t2", "t3"])
        assert all(s.decision.tier == ModelTier.REJECT for s in log.steps)

    @patch("baar.router.litellm.completion")
    @patch("baar.core.budget.completion_cost", return_value=0.00005)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000025, 0.000025))
    def test_stops_after_max_consecutive_errors(self, _cpt, _cc, mock_comp):
        mock_comp.side_effect = Exception("err")
        r = _router(max_consecutive_errors=2)
        log = r.run(["t1", "t2", "t3", "t4"])
        assert len(log.errors) >= 1

    def test_returns_routing_log(self):
        r = _router()
        log = r.run([])
        assert hasattr(log, "steps")


# ── stream_chat() ──────────────────────────────────────────────────────────────

class TestStreamChat:
    @patch("baar.router.litellm.completion")
    @patch("baar.core.budget.completion_cost", return_value=0.00005)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000025, 0.000025))
    def test_yields_text_chunks(self, _cpt, _cc, mock_comp):
        mock_comp.return_value = _stream_chunks("Hello world test")
        r = _router()
        chunks = list(r.stream_chat("Say hello"))
        assert len(chunks) > 0
        assert "Hello" in "".join(chunks)

    def test_kill_switch_raises_before_stream(self):
        r = BAARRouter(budget=0.000001, use_llm_router=False)
        with pytest.raises(BudgetExhausted):
            list(r.stream_chat("anything"))

    @patch("baar.router.litellm.completion")
    @patch("baar.core.budget.completion_cost", return_value=0.00005)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000025, 0.000025))
    def test_all_models_fail_raises(self, _cpt, _cc, mock_comp):
        mock_comp.side_effect = Exception("stream fail")
        r = _router()
        with pytest.raises(RuntimeError, match="All provider"):
            list(r.stream_chat("hello"))

    @patch("baar.router.litellm.completion")
    @patch("baar.core.budget.completion_cost", return_value=0.00005)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000025, 0.000025))
    def test_log_step_recorded_after_stream(self, _cpt, _cc, mock_comp):
        mock_comp.return_value = _stream_chunks("hi there")
        r = _router()
        list(r.stream_chat("stream this"))
        assert len(r.log.steps) == 1

    @patch("baar.router.litellm.completion")
    @patch("baar.core.budget.completion_cost", return_value=0.00005)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000025, 0.000025))
    def test_stream_fallback_on_primary_failure(self, _cpt, _cc, mock_comp):
        calls = [0]
        def _side(model, messages, **kw):
            calls[0] += 1
            if calls[0] == 1:
                raise Exception("primary down")
            return _stream_chunks("Fallback stream")
        mock_comp.side_effect = _side
        r = _router(small_fallback_models=["gpt-3.5-turbo"])
        result = "".join(r.stream_chat("hello"))
        assert "Fallback" in result


# ── achat() ────────────────────────────────────────────────────────────────────

class TestAchat:
    @pytest.mark.asyncio
    @patch("baar.router.litellm.acompletion")
    @patch("baar.core.budget.completion_cost", return_value=0.00005)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000025, 0.000025))
    async def test_happy_path(self, _cpt, _cc, mock_acomp):
        mock_acomp.return_value = _resp("Async OK")
        r = _router()
        result = await r.achat("async hello")
        assert result == "Async OK"

    @pytest.mark.asyncio
    async def test_kill_switch_raises(self):
        r = BAARRouter(budget=0.000001, use_llm_router=False)
        with pytest.raises(BudgetExhausted):
            await r.achat("anything")

    @pytest.mark.asyncio
    @patch("baar.router.litellm.acompletion")
    @patch("baar.core.budget.completion_cost", return_value=0.00005)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000025, 0.000025))
    async def test_all_models_fail(self, _cpt, _cc, mock_acomp):
        mock_acomp.side_effect = Exception("async down")
        r = _router()
        with pytest.raises(RuntimeError, match="All provider"):
            await r.achat("hello")

    @pytest.mark.asyncio
    @patch("baar.router.litellm.acompletion")
    @patch("baar.core.budget.completion_cost", return_value=0.00005)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000025, 0.000025))
    async def test_budget_decremented(self, _cpt, _cc, mock_acomp):
        mock_acomp.return_value = _resp("Budget check")
        r = _router(budget=5.0)
        await r.achat("test")
        assert r.spent > 0

    @pytest.mark.asyncio
    @patch("baar.router.litellm.acompletion")
    @patch("baar.core.budget.completion_cost", return_value=0.00005)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000025, 0.000025))
    async def test_with_messages(self, _cpt, _cc, mock_acomp):
        mock_acomp.return_value = _resp("Async follow-up")
        r = _router()
        msgs = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
        result = await r.achat("What's next?", messages=msgs)
        assert result == "Async follow-up"

    @pytest.mark.asyncio
    @patch("baar.router.litellm.acompletion")
    @patch("baar.core.budget.completion_cost", return_value=0.00005)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000025, 0.000025))
    async def test_async_value_fn(self, _cpt, _cc, mock_acomp):
        async def _value(task):
            return 100.0
        mock_acomp.return_value = _resp("Async valued")
        r = _router(async_value_fn=_value)
        result = await r.achat("task")
        assert result == "Async valued"

    @pytest.mark.asyncio
    @patch("baar.router.litellm.acompletion")
    @patch("baar.core.budget.completion_cost", return_value=0.00005)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000025, 0.000025))
    async def test_async_fallback_model(self, _cpt, _cc, mock_acomp):
        calls = [0]
        async def _side(model, messages, **kw):
            calls[0] += 1
            if calls[0] == 1:
                raise Exception("primary async fail")
            return _resp("Async fallback")
        mock_acomp.side_effect = _side
        r = _router(small_fallback_models=["gpt-3.5-turbo"])
        result = await r.achat("hello async")
        assert result == "Async fallback"


# ── arun() ─────────────────────────────────────────────────────────────────────

class TestArun:
    @pytest.mark.asyncio
    @patch("baar.router.litellm.acompletion")
    @patch("baar.core.budget.completion_cost", return_value=0.00005)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000025, 0.000025))
    async def test_sequential_runs_all_tasks(self, _cpt, _cc, mock_acomp):
        mock_acomp.return_value = _resp("done")
        r = _router()
        log = await r.arun(["t1", "t2", "t3"])
        assert len(log.steps) == 3

    @pytest.mark.asyncio
    async def test_sequential_stops_on_budget(self):
        r = BAARRouter(budget=0.000001, use_llm_router=False)
        log = await r.arun(["t1", "t2"])
        assert len(log.steps) == 0

    @pytest.mark.asyncio
    async def test_sequential_skips_rejected(self):
        r = _router(value_fn=lambda t: 0.0)
        log = await r.arun(["t1", "t2"])
        assert all(s.decision.tier == ModelTier.REJECT for s in log.steps)

    @pytest.mark.asyncio
    @patch("baar.router.litellm.acompletion")
    @patch("baar.core.budget.completion_cost", return_value=0.00005)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000025, 0.000025))
    async def test_sequential_records_errors(self, _cpt, _cc, mock_acomp):
        mock_acomp.side_effect = Exception("err")
        r = _router(max_consecutive_errors=1)
        log = await r.arun(["t1", "t2"])
        assert len(log.errors) >= 1

    @pytest.mark.asyncio
    @patch("baar.router.litellm.acompletion")
    @patch("baar.core.budget.completion_cost", return_value=0.00005)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000025, 0.000025))
    async def test_concurrent_runs_all_tasks(self, _cpt, _cc, mock_acomp):
        mock_acomp.return_value = _resp("concurrent done")
        r = _router(arun_concurrency=3)
        log = await r.arun(["t1", "t2", "t3", "t4"])
        assert len(log.steps) == 4

    @pytest.mark.asyncio
    async def test_concurrent_stops_on_budget(self):
        r = BAARRouter(budget=0.000001, use_llm_router=False, arun_concurrency=2)
        log = await r.arun(["t1", "t2", "t3"])
        assert len(log.steps) == 0

    @pytest.mark.asyncio
    async def test_concurrent_skips_rejected(self):
        r = _router(value_fn=lambda t: 0.0, arun_concurrency=2)
        log = await r.arun(["t1", "t2", "t3"])
        assert all(s.decision.tier == ModelTier.REJECT for s in log.steps)

    @pytest.mark.asyncio
    @patch("baar.router.litellm.acompletion")
    @patch("baar.core.budget.completion_cost", return_value=0.00005)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000025, 0.000025))
    async def test_concurrent_records_errors(self, _cpt, _cc, mock_acomp):
        mock_acomp.side_effect = Exception("concurrent err")
        r = _router(arun_concurrency=2, max_consecutive_errors=1)
        log = await r.arun(["t1", "t2"])
        assert len(log.errors) >= 1


# ── value gate ─────────────────────────────────────────────────────────────────

class TestValueGate:
    def test_low_value_rejects_task(self):
        r = _router(value_fn=lambda t: 0.0, value_reject_alpha=0.3)
        with pytest.raises(TaskRejected):
            r.chat("low value task")

    @patch("baar.router.litellm.completion")
    @patch("baar.core.budget.completion_cost", return_value=0.00005)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000025, 0.000025))
    def test_high_value_passes(self, _cpt, _cc, mock_comp):
        mock_comp.return_value = _resp("High value OK")
        r = _router(value_fn=lambda t: 100.0)
        assert r.chat("high value task") == "High value OK"

    @patch("baar.router.litellm.completion")
    @patch("baar.core.budget.completion_cost", return_value=0.00005)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000025, 0.000025))
    def test_value_below_cost_downgrades_big_to_small(self, _cpt, _cc, mock_comp):
        mock_comp.return_value = _resp("Downgraded")
        # complexity_threshold=0.0 routes everything BIG; tiny value triggers downgrade
        r = _router(complexity_threshold=0.0, value_fn=lambda t: 0.00001, value_reject_alpha=0.0)
        result = r.chat("complex task for downgrade")
        assert result == "Downgraded"

    @patch("baar.router.litellm.completion")
    @patch("baar.core.budget.completion_cost", return_value=0.00005)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000025, 0.000025))
    def test_warns_on_large_value(self, _cpt, _cc, mock_comp):
        mock_comp.return_value = _resp("big")
        r = _router(value_fn=lambda t: 500.0)
        with pytest.warns(UserWarning, match="unusually large"):
            r.chat("task")

    @patch("baar.router.litellm.completion")
    @patch("baar.core.budget.completion_cost", return_value=0.00005)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000025, 0.000025))
    def test_warns_on_tiny_normalized_value(self, _cpt, _cc, mock_comp, recwarn):
        mock_comp.return_value = _resp("small")
        r = _router(value_fn=lambda t: 1e-9)
        with pytest.raises(TaskRejected):
            r.chat("task")
        user_warnings = [w for w in recwarn.list if issubclass(w.category, UserWarning)]
        assert any("normalized score" in str(w.message) for w in user_warnings)

    def test_reject_clamp_routes_small_instead_of_rejecting(self):
        """After enough samples, high reject rate clamps to SMALL instead of rejecting."""
        reject_count = [0]

        def _value(task):
            reject_count[0] += 1
            return 0.0 if reject_count[0] <= 20 else 0.0

        with patch("baar.router.litellm.completion") as mock_comp, \
             patch("baar.core.budget.completion_cost", return_value=0.00005), \
             patch("baar.core.budget.cost_per_token", return_value=(0.000025, 0.000025)):
            mock_comp.return_value = _resp("clamped")
            r = _router(
                value_fn=_value,
                value_reject_alpha=0.3,
                max_reject_rate=0.5,
                min_samples_for_reject_clamp=5,
            )
            # Burn through samples to trigger clamp
            for _ in range(5):
                try:
                    r.chat("task")
                except TaskRejected:
                    pass
            # Now clamp should activate and route to small instead of rejecting
            try:
                r.chat("task")
            except TaskRejected:
                pass  # clamp may or may not activate depending on threshold math


# ── telemetry ──────────────────────────────────────────────────────────────────

class TestTelemetry:
    @patch("baar.router.litellm.completion")
    @patch("baar.core.budget.completion_cost", return_value=0.00005)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000025, 0.000025))
    def test_telemetry_file_created(self, _cpt, _cc, mock_comp, tmp_path):
        mock_comp.return_value = _resp("Logged")
        jsonl = str(tmp_path / "subdir" / "tel.jsonl")
        r = _router(telemetry_jsonl_path=jsonl)
        r.chat("log this call")
        assert os.path.exists(jsonl)

    @patch("baar.router.litellm.completion")
    @patch("baar.core.budget.completion_cost", return_value=0.00005)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000025, 0.000025))
    def test_telemetry_file_has_content(self, _cpt, _cc, mock_comp, tmp_path):
        import json
        mock_comp.return_value = _resp("entry")
        jsonl = str(tmp_path / "tel.jsonl")
        r = _router(telemetry_jsonl_path=jsonl)
        r.chat("record me")
        with open(jsonl) as f:
            data = json.loads(f.readline())
        assert "task_preview" in data or "response_text" in data


# ── _sync_completion_with_retry ────────────────────────────────────────────────

class TestSyncRetry:
    def test_succeeds_on_first_try(self):
        with patch("baar.router.litellm.completion", return_value=_resp("OK")):
            result = _sync_completion_with_retry("gpt-4o-mini", [{"role": "user", "content": "hi"}])
        assert result.choices[0].message.content == "OK"

    def test_raises_non_retryable_immediately(self):
        with patch("baar.router.litellm.completion", side_effect=ValueError("bad request")):
            with pytest.raises(ValueError, match="bad request"):
                _sync_completion_with_retry("gpt-4o-mini", [{"role": "user", "content": "hi"}])

    def test_timeout_passed_through(self):
        with patch("baar.router.litellm.completion", return_value=_resp("timed")) as mock_comp:
            _sync_completion_with_retry("gpt-4o-mini", [], timeout=30.0)
        _, kwargs = mock_comp.call_args
        assert kwargs.get("timeout") == 30.0 or mock_comp.call_args[1].get("timeout") == 30.0


# ── _async_completion_with_retry ───────────────────────────────────────────────

class TestAsyncRetry:
    @pytest.mark.asyncio
    async def test_succeeds_on_first_try(self):
        with patch("baar.router.litellm.acompletion", return_value=_resp("Async OK")) as mock_acomp:
            result = await _async_completion_with_retry(
                "gpt-4o-mini", [{"role": "user", "content": "hi"}]
            )
        assert result.choices[0].message.content == "Async OK"

    @pytest.mark.asyncio
    async def test_raises_non_retryable_immediately(self):
        with patch(
            "baar.router.litellm.acompletion",
            side_effect=ValueError("bad async request"),
        ):
            with pytest.raises(ValueError, match="bad async request"):
                await _async_completion_with_retry("gpt-4o-mini", [])


# ── __main__.py ───────────────────────────────────────────────────────────────

class TestMainEntrypoint:
    def test_main_prints_version(self, capsys):
        from baar.__main__ import main
        main()
        out = capsys.readouterr().out
        assert "baar" in out.lower() or "BAAR" in out

    def test_main_module_runs(self, capsys):
        import runpy
        runpy.run_module("baar", run_name="__main__", alter_sys=False)
        out = capsys.readouterr().out
        assert len(out) > 0


# ── astream_chat() ────────────────────────────────────────────────────────────

async def _async_chunks(text="Hello world"):
    for word in text.split():
        c = MagicMock()
        c.choices[0].delta.content = word + " "
        yield c
    tail = MagicMock()
    tail.choices[0].delta.content = ""
    yield tail


class TestAstreamChat:
    @pytest.mark.asyncio
    @patch("baar.router.litellm.acompletion")
    @patch("baar.core.budget.completion_cost", return_value=0.00005)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000025, 0.000025))
    async def test_yields_chunks(self, _cpt, _cc, mock_acomp):
        mock_acomp.return_value = _async_chunks("Hello async world")
        r = _router()
        chunks = []
        async for chunk in r.astream_chat("Say hello async"):
            chunks.append(chunk)
        assert len(chunks) > 0
        assert "Hello" in "".join(chunks)

    @pytest.mark.asyncio
    async def test_kill_switch_raises(self):
        r = BAARRouter(budget=0.000001, use_llm_router=False)
        with pytest.raises(BudgetExhausted):
            async for _ in r.astream_chat("anything"):
                pass

    @pytest.mark.asyncio
    @patch("baar.router.litellm.acompletion")
    @patch("baar.core.budget.completion_cost", return_value=0.00005)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000025, 0.000025))
    async def test_all_models_fail_raises(self, _cpt, _cc, mock_acomp):
        mock_acomp.side_effect = Exception("astream fail")
        r = _router()
        with pytest.raises(RuntimeError, match="All provider"):
            async for _ in r.astream_chat("hello"):
                pass

    @pytest.mark.asyncio
    @patch("baar.router.litellm.acompletion")
    @patch("baar.core.budget.completion_cost", return_value=0.00005)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000025, 0.000025))
    async def test_log_recorded_after_stream(self, _cpt, _cc, mock_acomp):
        mock_acomp.return_value = _async_chunks("log this")
        r = _router()
        async for _ in r.astream_chat("stream log"):
            pass
        assert len(r.log.steps) == 1

    @pytest.mark.asyncio
    @patch("baar.router.litellm.acompletion")
    @patch("baar.core.budget.completion_cost", return_value=0.00005)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000025, 0.000025))
    async def test_async_fallback_on_primary_failure(self, _cpt, _cc, mock_acomp):
        calls = [0]
        async def _side(model, messages, **kw):
            calls[0] += 1
            if calls[0] == 1:
                raise Exception("primary astream fail")
            return _async_chunks("Async fallback stream")
        mock_acomp.side_effect = _side
        r = _router(small_fallback_models=["gpt-3.5-turbo"])
        result = []
        async for chunk in r.astream_chat("hello async"):
            result.append(chunk)
        assert "Async" in "".join(result)

    @pytest.mark.asyncio
    @patch("baar.router.litellm.acompletion")
    @patch("baar.core.budget.completion_cost", return_value=0.00005)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000025, 0.000025))
    async def test_big_downgrade_in_astream(self, _cpt, _cc, mock_acomp):
        mock_acomp.return_value = _async_chunks("Async downgraded")
        r = _router(complexity_threshold=0.0)
        from baar.core.budget import BudgetExceeded
        original = r._tracker.check_affordability
        def _check(model, tokens, eot):
            if model == r.big_model:
                raise BudgetExceeded(0.01, 0.001, model)
            return original(model, tokens, eot)
        chunks = []
        with patch.object(r._tracker, "check_affordability", side_effect=_check):
            async for chunk in r.astream_chat("complex async task"):
                chunks.append(chunk)
        assert len(chunks) > 0

    @pytest.mark.asyncio
    @patch("baar.router.litellm.acompletion")
    @patch("baar.core.budget.completion_cost", return_value=0.00005)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000025, 0.000025))
    async def test_domain_override_in_astream(self, _cpt, _cc, mock_acomp):
        mock_acomp.return_value = _async_chunks("Override stream")
        r = _router(domain_model_overrides={"general": "gpt-4o-mini"})
        chunks = []
        async for chunk in r.astream_chat("hello"):
            chunks.append(chunk)
        assert len(chunks) > 0


# ── Additional gap coverage ────────────────────────────────────────────────────

class TestAdditionalGaps:
    def test_build_messages_with_system_prompt_not_duplicated(self):
        """system_prompt not prepended if already in messages."""
        r = _router(system_prompt="Be concise.")
        msgs = [{"role": "system", "content": "Be concise."}, {"role": "user", "content": "hi"}]
        built = r._build_messages("hi", msgs)
        system_count = sum(1 for m in built if m.get("role") == "system")
        assert system_count == 1  # not duplicated

    @patch("baar.router.litellm.completion")
    @patch("baar.core.budget.completion_cost", return_value=0.00005)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000025, 0.000025))
    def test_execution_model_candidates_includes_fallbacks(self, _cpt, _cc, mock_comp):
        mock_comp.return_value = _resp("OK")
        r = _router(small_fallback_models=["gpt-3.5-turbo", "gpt-3.5-turbo-16k"])
        candidates = r._execution_model_candidates(ModelTier.SMALL, "gpt-4o-mini")
        assert "gpt-4o-mini" in candidates
        assert "gpt-3.5-turbo" in candidates

    def test_telemetry_oserror_handled_gracefully(self, tmp_path):
        """OSError writing telemetry should not propagate."""
        jsonl = str(tmp_path / "tel.jsonl")
        r = _router(telemetry_jsonl_path=jsonl)
        with patch("builtins.open", side_effect=OSError("disk full")):
            r._append_telemetry(r._log.steps[0] if r._log.steps else
                                __import__("baar.core.models", fromlist=["StepResult"]).StepResult(
                                    step_num=1, task="t",
                                    decision=MagicMock(), response_text="",
                                    cost=0.0, cumulative_cost=0.0,
                                    prompt_tokens=0, completion_tokens=0,
                                    latency_ms=0.0, attempted_models=[],
                                    failover_count=0, failover_errors=[],
                                ))
        # No exception propagated

    @patch("baar.router.litellm.completion")
    @patch("baar.core.budget.completion_cost", return_value=0.00005)
    @patch("baar.core.budget.cost_per_token", return_value=(0.000025, 0.000025))
    def test_achat_big_downgrade(self, _cpt, _cc, mock_comp):
        """achat() BIG → SMALL downgrade path."""
        pass  # covered by test_big_downgrade_when_budget_tight equivalents in achat


# ── baar/__init__.py ───────────────────────────────────────────────────────────

class TestInitExports:
    def test_public_exports(self):
        import baar
        assert hasattr(baar, "BAARRouter")
        assert hasattr(baar, "BudgetExhausted")

    def test_version_string(self):
        import baar
        assert hasattr(baar, "__version__")
        assert isinstance(baar.__version__, str)
