"""
tests/test_output_overrun.py — output-overrun modes (PR 1a).

Covers how Baar bounds the *output* side of a call:
  - "estimate"  (default) — today's behaviour; never cuts responses off.
  - "preflight" — reserves the worst-case output before the call; never cuts
                  responses off and never overshoots.
  - "truncate"  — opt-in; derives max_tokens from the remaining budget.

All pricing is faked with linear per-token rates so worst-case maths is exact.
Runs without API keys.
"""

import math
import warnings
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from baar import BAARConfig, BAARRouter, BudgetExceeded, BudgetExhausted, Policy, Rule
from baar.core import caps
from baar.core.budget import BudgetTracker
from baar.core.caps import OutputLimitUnavailable, OutputPlan, plan_output


# ── Fake pricing ──────────────────────────────────────────────────────────────

IN_PRICE = {"gpt-4o-mini": 1.5e-7, "gpt-4o": 2.5e-6, "cheap-fallback": 1e-7}
OUT_PRICE = {"gpt-4o-mini": 6e-7, "gpt-4o": 1e-5, "cheap-fallback": 4e-7}
MAX_OUT = {"gpt-4o-mini": 16384, "gpt-4o": 16384, "cheap-fallback": 8192, "tiered-model": 65536}
CONTEXT = {"gpt-4o-mini": 128000, "gpt-4o": 128000, "cheap-fallback": 32000, "tiered-model": 1_000_000}

# Gemini-2.5-Pro-style tiered pricing: both rates rise once the prompt passes 200k tokens.
TIER_AT = 200_000
TIERED_IN = (1.25e-6, 2.5e-6)
TIERED_OUT = (1e-5, 1.5e-5)
PROMPT_TOKENS = 100


def _fake_cost_per_token(model, prompt_tokens=0, completion_tokens=0, **_):
    if model == "tiered-model":
        tier = 1 if prompt_tokens > TIER_AT else 0
        return prompt_tokens * TIERED_IN[tier], completion_tokens * TIERED_OUT[tier]
    if model not in IN_PRICE:
        raise Exception(f"no pricing for {model}")
    return prompt_tokens * IN_PRICE[model], completion_tokens * OUT_PRICE[model]


def _fake_completion_cost(completion_response=None, **_):
    u = completion_response.usage
    i, o = _fake_cost_per_token(completion_response.model, u.prompt_tokens, u.completion_tokens)
    return i + o


def cost(model, completion_tokens, prompt_tokens=PROMPT_TOKENS):
    i, o = _fake_cost_per_token(model, prompt_tokens, completion_tokens)
    return i + o


@pytest.fixture
def priced():
    with patch("baar.core.budget.cost_per_token", side_effect=_fake_cost_per_token), \
         patch("baar.core.budget.completion_cost", side_effect=_fake_completion_cost), \
         patch("baar.router.token_counter", return_value=PROMPT_TOKENS), \
         patch("baar.core.caps.model_max_output_tokens", side_effect=MAX_OUT.get), \
         patch("baar.core.caps.model_context_window", side_effect=CONTEXT.get), \
         patch("baar.core.caps.supports_stream_usage", return_value=True):
        yield


def make_response(model="gpt-4o-mini", completion_tokens=30, finish_reason="stop", content="ok"):
    resp = MagicMock()
    resp.model = model
    resp.choices[0].message.content = content
    resp.choices[0].finish_reason = finish_reason
    resp.usage.prompt_tokens = PROMPT_TOKENS
    resp.usage.completion_tokens = completion_tokens
    return resp


def make_chunks(texts, finish_reason="stop", usage=None):
    """Stream chunks; ``usage`` adds an OpenAI-style final chunk with no choices."""
    chunks = []
    for i, t in enumerate(texts):
        fr = finish_reason if i == len(texts) - 1 else None
        chunks.append(SimpleNamespace(choices=[SimpleNamespace(
            delta=SimpleNamespace(content=t), finish_reason=fr,
        )]))
    if usage is not None:
        prompt_tokens, completion_tokens = usage
        chunks.append(SimpleNamespace(choices=[], usage=SimpleNamespace(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
        )))
    return chunks


def completion_echo(completion_tokens_from_max=False, finish_reason="stop"):
    """side_effect for litellm.completion that echoes the requested model."""
    def _call(**kwargs):
        ct = kwargs.get("max_tokens", 30) if completion_tokens_from_max else 30
        return make_response(model=kwargs["model"], completion_tokens=ct, finish_reason=finish_reason)
    return _call


FORCE_BIG = Policy(rules=[Rule(when={}, then="force_big")])


def make_router(**kwargs):
    kwargs.setdefault("use_llm_router", False)
    return BAARRouter(**kwargs)


# ── caps.model_max_output_tokens ──────────────────────────────────────────────

class TestModelMaxOutputTokens:
    def test_reads_max_output_tokens_from_litellm(self):
        with patch("baar.core.caps.litellm.get_model_info",
                   return_value={"max_output_tokens": 8192, "max_tokens": 4096}):
            assert caps.model_max_output_tokens("some-model") == 8192

    def test_falls_back_to_max_tokens(self):
        with patch("baar.core.caps.litellm.get_model_info", return_value={"max_tokens": 4096}):
            assert caps.model_max_output_tokens("some-model") == 4096

    def test_unknown_model_returns_none(self):
        with patch("baar.core.caps.litellm.get_model_info", side_effect=Exception("unknown")):
            assert caps.model_max_output_tokens("mystery") is None

    def test_missing_fields_return_none(self):
        with patch("baar.core.caps.litellm.get_model_info", return_value={}):
            assert caps.model_max_output_tokens("some-model") is None


# ── BudgetTracker.output_price_per_token ──────────────────────────────────────

class TestOutputPricePerToken:
    def test_returns_per_token_output_price(self, priced):
        t = BudgetTracker(total_budget=1.0)
        assert t.output_price_per_token("gpt-4o") == pytest.approx(1e-5)

    def test_unknown_model_returns_none(self, priced):
        t = BudgetTracker(total_budget=1.0)
        assert t.output_price_per_token("mystery") is None


# ── plan_output (pure planning logic) ─────────────────────────────────────────

def _plan(mode, model="gpt-4o-mini", remaining=1.0, eot=500, **kw):
    return plan_output(
        mode,
        model=model,
        prompt_tokens=PROMPT_TOKENS,
        expected_output_tokens=eot,
        remaining=remaining,
        tracker=BudgetTracker(total_budget=10.0),
        **kw,
    )


class TestPlanOutput:
    def test_estimate_without_limit_uses_eot_and_sends_no_max_tokens(self, priced):
        p = _plan("estimate", eot=500)
        assert isinstance(p, OutputPlan)
        assert p.completion_tokens == 500
        assert p.max_tokens is None
        assert p.budget_limited is False

    def test_estimate_with_user_limit_caps_eot_and_sends_limit(self, priced):
        p = _plan("estimate", eot=500, max_output_tokens=200)
        assert p.completion_tokens == 200
        assert p.max_tokens == 200

    def test_estimate_user_limit_above_eot_keeps_eot_for_estimate(self, priced):
        p = _plan("estimate", eot=500, max_output_tokens=2000)
        assert p.completion_tokens == 500
        assert p.max_tokens == 2000

    def test_preflight_uses_model_max_as_worst_case(self, priced):
        p = _plan("preflight")
        assert p.completion_tokens == MAX_OUT["gpt-4o-mini"]
        assert p.max_tokens == MAX_OUT["gpt-4o-mini"]
        assert p.budget_limited is False

    def test_preflight_uses_user_limit_when_set(self, priced):
        p = _plan("preflight", max_output_tokens=1000)
        assert p.completion_tokens == 1000
        assert p.max_tokens == 1000

    def test_preflight_unknown_limit_raises_clear_error(self, priced):
        with pytest.raises(OutputLimitUnavailable, match="max_output_tokens"):
            _plan("preflight", model="mystery")

    def test_output_limit_unavailable_is_a_value_error(self):
        assert issubclass(OutputLimitUnavailable, ValueError)

    def test_truncate_derives_max_tokens_from_remaining(self, priced):
        model = "gpt-4o-mini"
        remaining = cost(model, 300) + OUT_PRICE[model] / 2
        p = _plan("truncate", model=model, remaining=remaining)
        assert p.completion_tokens == 300
        assert p.max_tokens == 300
        assert p.budget_limited is True
        assert cost(model, p.max_tokens) <= remaining

    def test_truncate_plenty_of_budget_uses_model_max(self, priced):
        p = _plan("truncate", remaining=100.0)
        assert p.max_tokens == MAX_OUT["gpt-4o-mini"]
        assert p.budget_limited is False

    def test_truncate_respects_user_limit(self, priced):
        p = _plan("truncate", remaining=100.0, max_output_tokens=777)
        assert p.max_tokens == 777
        assert p.budget_limited is False

    def test_truncate_below_min_useful_raises_budget_exceeded(self, priced):
        model = "gpt-4o-mini"
        remaining = cost(model, 63)
        with pytest.raises(BudgetExceeded):
            _plan("truncate", model=model, remaining=remaining, min_useful_tokens=64)

    def test_truncate_unknown_price_falls_back_to_limit(self, priced):
        p = _plan("truncate", model="mystery", remaining=1.0, max_output_tokens=400)
        assert p.max_tokens == 400
        assert p.budget_limited is False

    def test_unknown_mode_raises(self, priced):
        with pytest.raises(ValueError, match="on_output_overrun"):
            _plan("explode")


# ── Configuration ─────────────────────────────────────────────────────────────

class TestConfig:
    def test_default_mode_is_estimate(self):
        router = make_router()
        assert router.config.on_output_overrun == "estimate"
        assert router.config.max_output_tokens is None
        assert router.config.min_useful_tokens == 64

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError, match="on_output_overrun"):
            make_router(on_output_overrun="explode")

    @pytest.mark.parametrize("bad", [0, -5])
    def test_invalid_max_output_tokens_rejected(self, bad):
        with pytest.raises(ValueError, match="max_output_tokens"):
            make_router(max_output_tokens=bad)

    def test_invalid_min_useful_tokens_rejected(self):
        with pytest.raises(ValueError, match="min_useful_tokens"):
            make_router(min_useful_tokens=0)

    def test_from_config_carries_fields(self):
        router = BAARRouter.from_config(BAARConfig(
            use_llm_router=False, on_output_overrun="preflight", max_output_tokens=900,
        ))
        assert router.config.on_output_overrun == "preflight"
        assert router.config.max_output_tokens == 900

    def test_invalid_per_call_mode_rejected(self, priced):
        router = make_router(budget=1.0)
        with patch("baar.router.litellm.completion") as mock_completion:
            with pytest.raises(ValueError, match="on_output_overrun"):
                router.chat("hi", on_output_overrun="explode")
            mock_completion.assert_not_called()


# ── Default mode stays backward compatible ────────────────────────────────────

class TestEstimateMode:
    def test_default_sends_no_max_tokens(self, priced):
        router = make_router(budget=1.0)
        with patch("baar.router.litellm.completion", side_effect=completion_echo()) as m:
            router.chat("hello")
        assert "max_tokens" not in m.call_args.kwargs

    def test_user_limit_is_forwarded(self, priced):
        router = make_router(budget=1.0, max_output_tokens=200)
        with patch("baar.router.litellm.completion", side_effect=completion_echo()) as m:
            router.chat("hello")
        assert m.call_args.kwargs["max_tokens"] == 200

    def test_user_limit_cutoff_is_not_flagged_as_budget_truncation(self, priced):
        router = make_router(budget=1.0, max_output_tokens=200)
        with patch("baar.router.litellm.completion",
                   side_effect=completion_echo(True, finish_reason="length")):
            router.chat("hello")
        assert router.log.steps[-1].truncated_by_budget is False


# ── preflight: never cut off, never overshoot ─────────────────────────────────

class TestPreflightMode:
    def test_big_unaffordable_worst_case_downgrades_to_small(self, priced):
        # big worst case ≈ $0.164, small worst case ≈ $0.0098
        router = make_router(budget=0.05, on_output_overrun="preflight", policy=FORCE_BIG)
        with pytest.warns(UserWarning, match="force_big"):
            with patch("baar.router.litellm.completion", side_effect=completion_echo()) as m:
                router.chat("hello")
        assert m.call_args.kwargs["model"] == "gpt-4o-mini"
        assert m.call_args.kwargs["max_tokens"] == MAX_OUT["gpt-4o-mini"]

    def test_small_unaffordable_worst_case_blocks_locally(self, priced):
        router = make_router(budget=0.005, on_output_overrun="preflight")
        with patch("baar.router.litellm.completion") as m:
            with pytest.raises(BudgetExhausted):
                router.chat("hello")
            m.assert_not_called()

    def test_worst_case_response_never_overshoots(self, priced):
        router = make_router(budget=0.011, on_output_overrun="preflight")
        with patch("baar.router.litellm.completion",
                   side_effect=completion_echo(True, finish_reason="length")):
            router.chat("hello")
        assert router.spent <= 0.011
        assert router.log.steps[-1].completion_tokens == MAX_OUT["gpt-4o-mini"]

    def test_user_limit_makes_big_affordable(self, priced):
        router = make_router(budget=0.05, on_output_overrun="preflight",
                             max_output_tokens=500, policy=FORCE_BIG)
        with patch("baar.router.litellm.completion", side_effect=completion_echo()) as m:
            router.chat("hello")
        assert m.call_args.kwargs["model"] == "gpt-4o"
        assert m.call_args.kwargs["max_tokens"] == 500

    def test_user_limit_cutoff_is_not_budget_truncation(self, priced):
        router = make_router(budget=1.0, on_output_overrun="preflight", max_output_tokens=500)
        with patch("baar.router.litellm.completion",
                   side_effect=completion_echo(True, finish_reason="length")):
            router.chat("hello")
        assert router.log.steps[-1].truncated_by_budget is False

    def test_per_call_override(self, priced):
        router = make_router(budget=0.005)  # default estimate
        with patch("baar.router.litellm.completion", side_effect=completion_echo()) as m:
            router.chat("hello")
            assert m.call_count == 1
            with pytest.raises(BudgetExhausted):
                router.chat("hello", on_output_overrun="preflight")
            assert m.call_count == 1

    def test_per_call_max_output_tokens_override(self, priced):
        router = make_router(budget=0.005, on_output_overrun="preflight")
        with patch("baar.router.litellm.completion", side_effect=completion_echo()) as m:
            router.chat("hello", max_output_tokens=1000)
        assert m.call_args.kwargs["max_tokens"] == 1000

    def test_unknown_model_limit_without_user_limit_raises(self, priced):
        router = make_router(budget=1.0, on_output_overrun="preflight",
                             small_model="mystery-small")
        with patch("baar.core.budget.cost_per_token", return_value=(1e-9, 1e-9)):
            with patch("baar.router.litellm.completion") as m:
                with pytest.raises(ValueError, match="max_output_tokens"):
                    router.chat("hello")
                m.assert_not_called()


# ── truncate: opt-in, budget-derived max_tokens ───────────────────────────────

class TestTruncateMode:
    def test_max_tokens_derived_from_budget(self, priced):
        router = make_router(budget=0.002, on_output_overrun="truncate")
        with patch("baar.router.litellm.completion", side_effect=completion_echo()) as m:
            router.chat("hello")
        sent = m.call_args.kwargs["max_tokens"]
        assert 64 <= sent < MAX_OUT["gpt-4o-mini"]
        assert cost("gpt-4o-mini", sent) <= 0.002

    def test_budget_cutoff_is_flagged(self, priced):
        router = make_router(budget=0.002, on_output_overrun="truncate")
        with patch("baar.router.litellm.completion",
                   side_effect=completion_echo(True, finish_reason="length")):
            router.chat("hello")
        step = router.log.steps[-1]
        assert step.truncated_by_budget is True
        assert step.to_dict()["truncated_by_budget"] is True
        assert router.log.summary()["truncated_by_budget_steps"] == 1

    def test_natural_stop_not_flagged(self, priced):
        router = make_router(budget=0.002, on_output_overrun="truncate")
        with patch("baar.router.litellm.completion", side_effect=completion_echo()):
            router.chat("hello")
        step = router.log.steps[-1]
        assert step.truncated_by_budget is False
        assert "truncated_by_budget" not in step.to_dict()

    def test_full_length_response_never_overshoots(self, priced):
        router = make_router(budget=0.002, on_output_overrun="truncate")
        with patch("baar.router.litellm.completion",
                   side_effect=completion_echo(True, finish_reason="length")):
            router.chat("hello")
        assert router.spent <= 0.002

    def test_below_min_useful_blocks_locally(self, priced):
        # 1000 tokens on small ≈ $0.000615 > $0.0005 budget
        router = make_router(budget=0.0005, on_output_overrun="truncate",
                             min_useful_tokens=1000, min_cost_threshold=0.0)
        with patch("baar.router.litellm.completion") as m:
            with pytest.raises(BudgetExhausted):
                router.chat("hello")
            m.assert_not_called()


# ── Failover: each candidate is covered at its own price ──────────────────────

class TestFailoverReservation:
    def test_unaffordable_fallback_is_skipped_not_called(self, priced):
        router = make_router(budget=0.05, on_output_overrun="preflight",
                             small_fallback_models=["gpt-4o"])

        def _call(**kwargs):
            raise ConnectionError("primary down")

        with patch("baar.router.litellm.completion", side_effect=_call) as m:
            with pytest.raises(RuntimeError, match="unaffordable"):
                router.chat("hello")
        called = [c.kwargs["model"] for c in m.call_args_list]
        assert "gpt-4o" not in called
        assert router.spent == pytest.approx(0.0)

    def test_pricier_fallback_reservation_is_topped_up_then_released(self, priced):
        router = make_router(budget=1.0, small_fallback_models=["gpt-4o"])
        eot = router._router.decide(task="hello", budget_utilization=0.0).estimated_output_tokens
        held_during_fallback = []

        def _call(**kwargs):
            if kwargs["model"] == "gpt-4o-mini":
                raise ConnectionError("primary down")
            held_during_fallback.append(router.spent)
            return make_response(model=kwargs["model"], completion_tokens=30)

        with patch("baar.router.litellm.completion", side_effect=_call) as m:
            router.chat("hello")
        assert m.call_args.kwargs["model"] == "gpt-4o"
        # While the fallback ran, the reservation covered it at gpt-4o prices…
        assert held_during_fallback[0] == pytest.approx(cost("gpt-4o", eot))
        # …and afterwards only the real cost remains.
        assert router.spent == pytest.approx(cost("gpt-4o", 30))

    def test_estimate_mode_unaffordable_fallback_is_skipped(self, priced):
        router = make_router(budget=1.0, small_fallback_models=["gpt-4o"])
        eot = router._router.decide(task="hello", budget_utilization=0.0).estimated_output_tokens
        budget = (cost("gpt-4o-mini", eot) + cost("gpt-4o", eot)) / 2
        router = make_router(budget=budget, small_fallback_models=["gpt-4o"], min_cost_threshold=0.0)

        def _call(**kwargs):
            raise ConnectionError("primary down")

        with patch("baar.router.litellm.completion", side_effect=_call) as m:
            with pytest.raises(RuntimeError, match="unaffordable"):
                router.chat("hello")
        assert [c.kwargs["model"] for c in m.call_args_list] == ["gpt-4o-mini"]
        assert router.spent == pytest.approx(0.0)

    def test_fallback_gets_its_own_max_tokens(self, priced):
        router = make_router(budget=1.0, on_output_overrun="preflight",
                             small_fallback_models=["cheap-fallback"])

        def _call(**kwargs):
            if kwargs["model"] == "gpt-4o-mini":
                raise ConnectionError("primary down")
            return make_response(model=kwargs["model"], completion_tokens=30)

        with patch("baar.router.litellm.completion", side_effect=_call) as m:
            router.chat("hello")
        assert m.call_args.kwargs["model"] == "cheap-fallback"
        assert m.call_args.kwargs["max_tokens"] == MAX_OUT["cheap-fallback"]


# ── Async and streaming paths ─────────────────────────────────────────────────

class TestOtherCallPaths:
    @pytest.mark.asyncio
    async def test_achat_preflight_sends_max_tokens(self, priced):
        router = make_router(budget=1.0, on_output_overrun="preflight")
        with patch("baar.router.litellm.acompletion", new_callable=AsyncMock,
                   return_value=make_response()) as m:
            await router.achat("hello")
        assert m.call_args.kwargs["max_tokens"] == MAX_OUT["gpt-4o-mini"]

    @pytest.mark.asyncio
    async def test_achat_preflight_blocks_locally(self, priced):
        router = make_router(budget=0.005, on_output_overrun="preflight")
        with patch("baar.router.litellm.acompletion", new_callable=AsyncMock) as m:
            with pytest.raises(BudgetExhausted):
                await router.achat("hello")
            m.assert_not_called()

    @pytest.mark.asyncio
    async def test_achat_truncate_flags_cutoff(self, priced):
        router = make_router(budget=0.002, on_output_overrun="truncate")
        with patch("baar.router.litellm.acompletion", new_callable=AsyncMock,
                   return_value=make_response(finish_reason="length")):
            await router.achat("hello")
        assert router.log.steps[-1].truncated_by_budget is True

    def test_stream_chat_truncate_sends_max_tokens_and_flags(self, priced):
        router = make_router(budget=0.002, on_output_overrun="truncate")
        with patch("baar.router.litellm.completion",
                   return_value=iter(make_chunks(["a", "b"], finish_reason="length",
                                                 usage=(100, 30)))) as m:
            assert "".join(router.stream_chat("hello")) == "ab"
        assert m.call_args.kwargs["stream"] is True
        assert m.call_args.kwargs["max_tokens"] < MAX_OUT["gpt-4o-mini"]
        assert router.log.steps[-1].truncated_by_budget is True

    def test_stream_chat_preflight_blocks_locally(self, priced):
        router = make_router(budget=0.005, on_output_overrun="preflight")
        with patch("baar.router.litellm.completion") as m:
            with pytest.raises(BudgetExhausted):
                list(router.stream_chat("hello"))
            m.assert_not_called()

    @pytest.mark.asyncio
    async def test_astream_chat_preflight_sends_max_tokens(self, priced):
        router = make_router(budget=1.0, on_output_overrun="preflight")

        async def _agen():
            for c in make_chunks(["x", "y"], usage=(100, 30)):
                yield c

        with patch("baar.router.litellm.acompletion", new_callable=AsyncMock,
                   return_value=_agen()) as m:
            out = [t async for t in router.astream_chat("hello")]
        assert "".join(out) == "xy"
        assert m.call_args.kwargs["max_tokens"] == MAX_OUT["gpt-4o-mini"]
        assert router.log.steps[-1].truncated_by_budget is False


# ── Integrations forward a request's own max_tokens ───────────────────────────

class TestIntegrationPassThrough:
    def test_vercel_forwards_max_tokens(self):
        pytest.importorskip("fastapi", reason="fastapi not installed")
        from fastapi.testclient import TestClient
        from baar.integrations.vercel import create_app

        router = make_router(budget=1.0)
        client = TestClient(create_app(router), raise_server_exceptions=False)
        with patch.object(router, "achat", new_callable=AsyncMock, return_value="hi") as m:
            resp = client.post("/v1/chat/completions", json={
                "model": "baar",
                "messages": [{"role": "user", "content": "Say hello"}],
                "max_tokens": 321,
            })
        assert resp.status_code == 200
        assert m.call_args.kwargs["max_output_tokens"] == 321

    def test_vercel_without_max_tokens_sends_no_limit(self):
        pytest.importorskip("fastapi", reason="fastapi not installed")
        from fastapi.testclient import TestClient
        from baar.integrations.vercel import create_app

        router = make_router(budget=1.0)
        client = TestClient(create_app(router), raise_server_exceptions=False)
        with patch.object(router, "achat", new_callable=AsyncMock, return_value="hi") as m:
            client.post("/v1/chat/completions", json={
                "model": "baar",
                "messages": [{"role": "user", "content": "Say hello"}],
            })
        assert "max_output_tokens" not in m.call_args.kwargs

    def test_langchain_forwards_bound_max_tokens(self):
        pytest.importorskip("langchain_core", reason="langchain-core not installed")
        from langchain_core.messages import HumanMessage
        from baar.integrations.langchain import BaarChatModel

        router = make_router(budget=1.0)
        llm = BaarChatModel(router=router)
        with patch.object(router, "chat", return_value="4") as m:
            llm._generate([HumanMessage(content="2+2?")], max_tokens=150)
        assert m.call_args.kwargs["max_output_tokens"] == 150

    @pytest.mark.asyncio
    async def test_langchain_async_forwards_bound_max_tokens(self):
        pytest.importorskip("langchain_core", reason="langchain-core not installed")
        from langchain_core.messages import HumanMessage
        from baar.integrations.langchain import BaarChatModel

        router = make_router(budget=1.0)
        llm = BaarChatModel(router=router)
        with patch.object(router, "achat", new_callable=AsyncMock, return_value="4") as m:
            await llm._agenerate([HumanMessage(content="2+2?")], max_tokens=150)
        assert m.call_args.kwargs["max_output_tokens"] == 150


# ── Review fix #1: output must fit the context window ─────────────────────────

class TestContextWindow:
    def test_model_context_window_reads_max_input_tokens(self):
        with patch("baar.core.caps.litellm.get_model_info", return_value={"max_input_tokens": 128000}):
            assert caps.model_context_window("some-model") == 128000

    def test_model_context_window_unknown_returns_none(self):
        with patch("baar.core.caps.litellm.get_model_info", side_effect=Exception("unknown")):
            assert caps.model_context_window("mystery") is None

    def test_preflight_limit_shrinks_to_room_left(self, priced):
        p = plan_output("preflight", model="gpt-4o", prompt_tokens=120000, expected_output_tokens=500,
                        remaining=10.0, tracker=BudgetTracker(total_budget=10.0))
        assert p.max_tokens == 8000
        assert p.completion_tokens == 8000

    def test_preflight_full_context_raises(self, priced):
        with pytest.raises(OutputLimitUnavailable, match="context window"):
            plan_output("preflight", model="gpt-4o", prompt_tokens=128000, expected_output_tokens=500,
                        remaining=10.0, tracker=BudgetTracker(total_budget=10.0))

    def test_truncate_limit_shrinks_to_room_left(self, priced):
        p = plan_output("truncate", model="gpt-4o", prompt_tokens=120000, expected_output_tokens=500,
                        remaining=10.0, tracker=BudgetTracker(total_budget=10.0))
        assert p.max_tokens == 8000
        assert p.budget_limited is False

    def test_estimate_mode_is_unchanged_by_context(self, priced):
        p = plan_output("estimate", model="gpt-4o", prompt_tokens=127990, expected_output_tokens=500,
                        remaining=10.0, tracker=BudgetTracker(total_budget=10.0))
        assert p.max_tokens is None

    def test_unknown_context_window_does_not_limit(self, priced):
        with patch("baar.core.caps.model_context_window", return_value=None):
            p = _plan("preflight")
        assert p.max_tokens == MAX_OUT["gpt-4o-mini"]

    def test_router_preflight_long_prompt_sends_room_left(self, priced):
        router = make_router(budget=1.0, on_output_overrun="preflight")
        with patch("baar.router.token_counter", return_value=120000), \
             patch("baar.router.litellm.completion", side_effect=completion_echo()) as m:
            router.chat("hello")
        assert m.call_args.kwargs["max_tokens"] == 8000

    def test_router_preflight_full_context_blocks_locally(self, priced):
        router = make_router(budget=10.0, on_output_overrun="preflight")
        with patch("baar.router.token_counter", return_value=128000), \
             patch("baar.router.litellm.completion") as m:
            with pytest.raises(OutputLimitUnavailable):
                router.chat("hello")
            m.assert_not_called()


# ── Review fix #2: streaming records provider-reported usage ──────────────────

class TestStreamingUsage:
    def test_supports_stream_usage_reads_litellm_params(self):
        with patch("baar.core.caps.litellm.get_supported_openai_params",
                   return_value=["max_tokens", "stream_options"]):
            assert caps.supports_stream_usage("gpt-4o") is True
        with patch("baar.core.caps.litellm.get_supported_openai_params", return_value=["max_tokens"]):
            assert caps.supports_stream_usage("other") is False
        with patch("baar.core.caps.litellm.get_supported_openai_params", side_effect=Exception("x")):
            assert caps.supports_stream_usage("mystery") is False

    def test_stream_requests_usage_when_supported(self, priced):
        router = make_router(budget=1.0)
        with patch("baar.router.litellm.completion",
                   return_value=iter(make_chunks(["a"], usage=(100, 30)))) as m:
            list(router.stream_chat("hello"))
        assert m.call_args.kwargs["stream_options"] == {"include_usage": True}

    def test_stream_does_not_request_usage_when_unsupported(self, priced):
        router = make_router(budget=1.0)
        with patch("baar.core.caps.supports_stream_usage", return_value=False), \
             patch("baar.router.litellm.completion", return_value=iter(make_chunks(["a"]))) as m, \
             pytest.warns(UserWarning, match="usage"):
            list(router.stream_chat("hello"))
        assert "stream_options" not in m.call_args.kwargs

    def test_stream_records_hidden_reasoning_tokens_from_usage(self, priced):
        # Visible text is tiny, but the provider bills 5000 completion tokens (reasoning).
        router = make_router(budget=1.0)
        with patch("baar.router.litellm.completion",
                   return_value=iter(make_chunks(["ok"], usage=(100, 5000)))):
            assert "".join(router.stream_chat("hello")) == "ok"
        step = router.log.steps[-1]
        assert step.completion_tokens == 5000
        assert router.spent == pytest.approx(cost("gpt-4o-mini", 5000))

    def test_stream_usage_on_last_content_chunk_is_used(self, priced):
        chunks = make_chunks(["a", "b"])
        chunks[-1].usage = SimpleNamespace(prompt_tokens=120, completion_tokens=900)
        router = make_router(budget=1.0)
        with patch("baar.router.litellm.completion", return_value=iter(chunks)):
            list(router.stream_chat("hello"))
        step = router.log.steps[-1]
        assert (step.prompt_tokens, step.completion_tokens) == (120, 900)

    def test_stream_without_usage_falls_back_and_warns(self, priced):
        router = make_router(budget=1.0)
        with patch("baar.router.litellm.completion", return_value=iter(make_chunks(["a"]))), \
             pytest.warns(UserWarning, match="usage"):
            list(router.stream_chat("hello"))
        assert router.log.steps[-1].completion_tokens == PROMPT_TOKENS  # token_counter fallback

    def test_stream_reads_litellm_hidden_usage_without_stream_options(self, priced):
        # Claude/Gemini: LiteLLM strips chunk.usage but attaches the provider's
        # real totals to the last chunk's _hidden_params["usage"].
        chunks = make_chunks(["hi"])
        chunks[-1]._hidden_params = {"usage": SimpleNamespace(prompt_tokens=100, completion_tokens=2500)}
        router = make_router(budget=1.0)
        with patch("baar.core.caps.supports_stream_usage", return_value=False), \
             patch("baar.router.litellm.completion", return_value=iter(chunks)) as m:
            with warnings.catch_warnings():
                warnings.simplefilter("error", UserWarning)
                list(router.stream_chat("hello"))
        assert "stream_options" not in m.call_args.kwargs
        assert router.log.steps[-1].completion_tokens == 2500
        assert router.spent == pytest.approx(cost("gpt-4o-mini", 2500))

    def test_stream_usage_with_zero_prompt_tokens_keeps_estimate(self, priced):
        # Anthropic's final message_delta can report 0 input tokens; LiteLLM keeps the latest value.
        chunks = make_chunks(["hi"])
        chunks[-1]._hidden_params = {"usage": SimpleNamespace(prompt_tokens=0, completion_tokens=700)}
        router = make_router(budget=1.0)
        with patch("baar.router.litellm.completion", return_value=iter(chunks)):
            list(router.stream_chat("hello"))
        step = router.log.steps[-1]
        assert (step.prompt_tokens, step.completion_tokens) == (PROMPT_TOKENS, 700)

    @pytest.mark.asyncio
    async def test_astream_records_usage(self, priced):
        router = make_router(budget=1.0)

        async def _agen():
            for c in make_chunks(["x"], usage=(100, 4000)):
                yield c

        with patch("baar.router.litellm.acompletion", new_callable=AsyncMock, return_value=_agen()) as m:
            out = [t async for t in router.astream_chat("hello")]
        assert out == ["x"]
        assert m.call_args.kwargs["stream_options"] == {"include_usage": True}
        assert router.log.steps[-1].completion_tokens == 4000
        assert router.spent == pytest.approx(cost("gpt-4o-mini", 4000))


# ── Review fix #3: unknown output limit downgrades instead of crashing ────────

class TestUnknownLimitDowngrade:
    def test_preflight_big_with_unknown_limit_downgrades(self, priced):
        router = make_router(budget=1.0, on_output_overrun="preflight",
                             big_model="mystery-big", policy=FORCE_BIG)
        with patch("baar.router.litellm.completion", side_effect=completion_echo()) as m, \
             pytest.warns(UserWarning, match="force_big"):
            router.chat("hello")
        assert m.call_args.kwargs["model"] == "gpt-4o-mini"


# ── Review fix #4: truncate downgrades before cutting the big model off ───────

class TestTruncateDowngradesFirst:
    def test_big_that_would_be_cut_off_downgrades_to_small(self, priced):
        router = make_router(budget=0.002, on_output_overrun="truncate", policy=FORCE_BIG)
        with patch("baar.router.litellm.completion", side_effect=completion_echo()) as m, \
             pytest.warns(UserWarning, match="force_big"):
            router.chat("hello")
        assert m.call_args.kwargs["model"] == "gpt-4o-mini"

    def test_big_that_fits_uncut_stays_big(self, priced):
        router = make_router(budget=1.0, on_output_overrun="truncate", policy=FORCE_BIG)
        with patch("baar.router.litellm.completion", side_effect=completion_echo()) as m:
            router.chat("hello")
        assert m.call_args.kwargs["model"] == "gpt-4o"
        assert m.call_args.kwargs["max_tokens"] == MAX_OUT["gpt-4o"]
        assert router.log.steps[-1].truncated_by_budget is False

    def test_big_with_user_limit_that_fits_stays_big(self, priced):
        router = make_router(budget=0.05, on_output_overrun="truncate",
                             max_output_tokens=500, policy=FORCE_BIG)
        with patch("baar.router.litellm.completion", side_effect=completion_echo()) as m:
            router.chat("hello")
        assert m.call_args.kwargs["model"] == "gpt-4o"
        assert m.call_args.kwargs["max_tokens"] == 500


# ── Review fix #5: tiered pricing uses the real prompt length ─────────────────

class TestTieredPricing:
    def test_output_price_uses_prompt_tier(self, priced):
        t = BudgetTracker(total_budget=1.0)
        assert t.output_price_per_token("tiered-model") == pytest.approx(TIERED_OUT[0])
        assert t.output_price_per_token("tiered-model", prompt_tokens=250_000) == pytest.approx(TIERED_OUT[1])

    def test_truncate_long_prompt_fits_remaining(self, priced):
        prompt = 250_000
        remaining = cost("tiered-model", 300, prompt_tokens=prompt) + TIERED_OUT[1] / 2
        p = plan_output("truncate", model="tiered-model", prompt_tokens=prompt, expected_output_tokens=500,
                        remaining=remaining, tracker=BudgetTracker(total_budget=10.0))
        assert p.max_tokens == 300
        assert cost("tiered-model", p.max_tokens, prompt_tokens=prompt) <= remaining


# ── Review fix #6: optional cap on one truncate call's share of the budget ────

class TestMaxCallShare:
    def test_default_is_whole_remaining(self):
        assert make_router().config.max_call_share == 1.0

    @pytest.mark.parametrize("bad", [0, -0.1, 1.5])
    def test_invalid_share_rejected(self, bad):
        with pytest.raises(ValueError, match="max_call_share"):
            make_router(max_call_share=bad)

    def test_plan_limits_reservation_to_share(self, priced):
        remaining = 0.002
        p = plan_output("truncate", model="gpt-4o-mini", prompt_tokens=PROMPT_TOKENS,
                        expected_output_tokens=500, remaining=remaining,
                        tracker=BudgetTracker(total_budget=10.0), max_call_share=0.5)
        assert cost("gpt-4o-mini", p.max_tokens) <= remaining * 0.5
        assert p.budget_limited is True

    def test_router_truncate_leaves_room_for_concurrent_calls(self, priced):
        router = make_router(budget=0.002, on_output_overrun="truncate", max_call_share=0.5)
        with patch("baar.router.litellm.completion", side_effect=completion_echo()) as m:
            router.chat("hello")
        assert cost("gpt-4o-mini", m.call_args.kwargs["max_tokens"]) <= 0.001

    def test_share_does_not_affect_preflight(self, priced):
        router = make_router(budget=1.0, on_output_overrun="preflight", max_call_share=0.1)
        with patch("baar.router.litellm.completion", side_effect=completion_echo()) as m:
            router.chat("hello")
        assert m.call_args.kwargs["max_tokens"] == MAX_OUT["gpt-4o-mini"]


# ── Review fix #7: integrations forward output limits ─────────────────────────

class TestIntegrationLimitsFollowUp:
    def _vercel_client(self, router):
        pytest.importorskip("fastapi", reason="fastapi not installed")
        from fastapi.testclient import TestClient
        from baar.integrations.vercel import create_app
        return TestClient(create_app(router), raise_server_exceptions=False)

    def _post(self, client, **extra):
        return client.post("/v1/chat/completions", json={
            "model": "baar", "messages": [{"role": "user", "content": "Say hello"}], **extra,
        })

    def test_vercel_forwards_max_completion_tokens(self):
        router = make_router(budget=1.0)
        client = self._vercel_client(router)
        with patch.object(router, "achat", new_callable=AsyncMock, return_value="hi") as m:
            assert self._post(client, max_completion_tokens=222).status_code == 200
        assert m.call_args.kwargs["max_output_tokens"] == 222

    def test_vercel_max_completion_tokens_wins_over_max_tokens(self):
        router = make_router(budget=1.0)
        client = self._vercel_client(router)
        with patch.object(router, "achat", new_callable=AsyncMock, return_value="hi") as m:
            self._post(client, max_tokens=100, max_completion_tokens=222)
        assert m.call_args.kwargs["max_output_tokens"] == 222

    @pytest.mark.parametrize("field", ["max_tokens", "max_completion_tokens"])
    def test_vercel_invalid_limit_is_400(self, field):
        router = make_router(budget=1.0)
        client = self._vercel_client(router)
        with patch.object(router, "achat", new_callable=AsyncMock, return_value="hi") as m:
            resp = self._post(client, **{field: 0})
        assert resp.status_code == 400
        assert "invalid_request_error" in resp.text
        m.assert_not_called()

    def test_vercel_invalid_limit_is_400_when_streaming(self):
        router = make_router(budget=1.0)
        client = self._vercel_client(router)
        assert self._post(client, stream=True, max_tokens=-1).status_code == 400

    def _crew_llm_class(self):
        import sys
        if "crewai" not in sys.modules:
            class _FakeCrewAILLM:
                def __init__(self, **kwargs):
                    for k, v in kwargs.items():
                        setattr(self, k, v)
            sys.modules["crewai"] = MagicMock(LLM=_FakeCrewAILLM)
        from baar.integrations.crewai import BaarCrewLLM
        return BaarCrewLLM

    def test_crewai_forwards_max_tokens(self):
        BaarCrewLLM = self._crew_llm_class()
        router = make_router(budget=1.0)
        llm = BaarCrewLLM(router=router, max_tokens=300)
        with patch.object(router, "chat", return_value="ok") as m:
            llm.call([{"role": "user", "content": "hi"}])
        assert m.call_args.kwargs["max_output_tokens"] == 300

    def test_crewai_without_max_tokens_sends_no_limit(self):
        BaarCrewLLM = self._crew_llm_class()
        router = make_router(budget=1.0)
        llm = BaarCrewLLM(router=router)
        with patch.object(router, "chat", return_value="ok") as m:
            llm.call([{"role": "user", "content": "hi"}])
        assert "max_output_tokens" not in m.call_args.kwargs

    @pytest.mark.asyncio
    async def test_crewai_acall_forwards_max_tokens(self):
        BaarCrewLLM = self._crew_llm_class()
        router = make_router(budget=1.0)
        llm = BaarCrewLLM(router=router, max_tokens=300)
        with patch.object(router, "achat", new_callable=AsyncMock, return_value="ok") as m:
            await llm.acall([{"role": "user", "content": "hi"}])
        assert m.call_args.kwargs["max_output_tokens"] == 300


# ── Review fix #6 (round 2): value is compared with the expected cost ─────────

class TestValueGateUsesExpectedCost:
    def _eot(self, router):
        return router._router.decide(task="hello", budget_utilization=0.0).estimated_output_tokens

    def test_value_gate_estimate_is_the_expected_cost(self, priced):
        router = make_router(budget=1.0, on_output_overrun="preflight", value_fn=lambda t: 0.5)
        eot = self._eot(router)
        with patch("baar.router.litellm.completion", side_effect=completion_echo()):
            router.chat("hello")
        decision = router.log.steps[-1].decision
        assert decision.estimated_cost_usd == pytest.approx(cost("gpt-4o-mini", eot))

    def test_cheap_task_is_not_rejected_because_preflight_is_on(self, priced):
        router = make_router(budget=1.0, on_output_overrun="preflight")
        eot = self._eot(router)
        expected, worst = cost("gpt-4o-mini", eot), cost("gpt-4o-mini", MAX_OUT["gpt-4o-mini"])
        value = (expected + worst) / 2  # worth more than expected, less than worst case
        router = make_router(budget=1.0, on_output_overrun="preflight", value_fn=lambda t: value)
        with patch("baar.router.litellm.completion", side_effect=completion_echo()) as m:
            router.chat("hello")
        assert m.call_count == 1

    def test_expected_cost_respects_user_limit(self, priced):
        router = make_router(budget=1.0, value_fn=lambda t: 0.5, max_output_tokens=10)
        with patch("baar.router.litellm.completion", side_effect=completion_echo()):
            router.chat("hello")
        assert router.log.steps[-1].decision.estimated_cost_usd == pytest.approx(cost("gpt-4o-mini", 10))


# ── Round 2 #1: a stream with no usage is not recorded as 0 tokens ────────────

class TestZeroUsageIsNoUsage:
    def test_zero_pair_is_not_usage(self):
        from baar.router import _reported_usage
        chunk = SimpleNamespace(choices=[], _hidden_params={"usage": SimpleNamespace(prompt_tokens=0, completion_tokens=0)})
        assert _reported_usage(chunk) is None

    def test_real_litellm_stream_without_usage_falls_back_and_warns(self, priced):
        import litellm as _litellm
        real = _litellm.completion

        def _call(**kwargs):
            return real(**kwargs, mock_response="one two three four five")

        router = make_router(budget=1.0)
        with patch("baar.core.caps.supports_stream_usage", return_value=False), \
             patch("baar.router.litellm.completion", side_effect=_call), \
             pytest.warns(UserWarning, match="usage"):
            assert "".join(router.stream_chat("hello")) == "one two three four five"
        step = router.log.steps[-1]
        assert step.completion_tokens == PROMPT_TOKENS  # patched token_counter, not 0
        assert router.spent > cost("gpt-4o-mini", 0)

    def test_real_litellm_stream_with_include_usage_records_it(self, priced):
        import litellm as _litellm
        real = _litellm.completion

        def _call(**kwargs):
            return real(**kwargs, mock_response="one two three four five")

        router = make_router(budget=1.0)
        with patch("baar.router.litellm.completion", side_effect=_call):
            with warnings.catch_warnings():
                warnings.simplefilter("error", UserWarning)
                list(router.stream_chat("hello"))
        assert router.log.steps[-1].completion_tokens == 5

    def test_zero_completion_with_text_is_not_trusted(self, priced):
        chunks = make_chunks(["visible answer"])
        chunks[-1].usage = SimpleNamespace(prompt_tokens=100, completion_tokens=0)
        router = make_router(budget=1.0)
        with patch("baar.router.litellm.completion", return_value=iter(chunks)), \
             pytest.warns(UserWarning, match="usage"):
            list(router.stream_chat("hello"))
        assert router.log.steps[-1].completion_tokens == PROMPT_TOKENS


# ── Round 2 #2: stopping a stream early closes it and stays safe ──────────────

class _Upstream:
    """A provider stream that records whether it was closed."""

    def __init__(self, chunks):
        self._it = iter(chunks)
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._it)

    def close(self):
        self.closed = True


class _AsyncUpstream:
    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)

    async def aclose(self):
        self.closed = True


class TestEarlyStreamExit:
    def _chunks(self):
        return make_chunks(["a", "b", "c", "d"], usage=(100, 9000))

    def test_sync_early_exit_closes_upstream(self, priced):
        upstream = _Upstream(self._chunks())
        router = make_router(budget=1.0)
        with patch("baar.router.litellm.completion", return_value=upstream):
            gen = router.stream_chat("hello")
            assert next(gen) == "a"
            gen.close()
        assert upstream.closed is True

    def test_preflight_early_exit_records_the_reservation(self, priced):
        upstream = _Upstream(self._chunks())
        router = make_router(budget=1.0, on_output_overrun="preflight")
        with patch("baar.router.litellm.completion", return_value=upstream):
            gen = router.stream_chat("hello")
            next(gen)
            gen.close()
        step = router.log.steps[-1]
        assert step.completion_tokens == MAX_OUT["gpt-4o-mini"]
        assert router.spent == pytest.approx(cost("gpt-4o-mini", MAX_OUT["gpt-4o-mini"]))

    def test_truncate_early_exit_records_the_reservation(self, priced):
        upstream = _Upstream(self._chunks())
        router = make_router(budget=0.002, on_output_overrun="truncate")
        with patch("baar.router.litellm.completion", return_value=upstream) as m:
            gen = router.stream_chat("hello")
            next(gen)
            gen.close()
        sent = m.call_args.kwargs["max_tokens"]
        assert router.log.steps[-1].completion_tokens == sent
        assert router.spent <= 0.002

    def test_estimate_early_exit_records_visible_text(self, priced):
        upstream = _Upstream(self._chunks())
        router = make_router(budget=1.0)
        with patch("baar.router.litellm.completion", return_value=upstream):
            gen = router.stream_chat("hello")
            next(gen)
            gen.close()
        assert router.log.steps[-1].completion_tokens == PROMPT_TOKENS  # token_counter fallback

    def test_full_stream_does_not_use_the_early_exit_path(self, priced):
        upstream = _Upstream(self._chunks())
        router = make_router(budget=1.0, on_output_overrun="preflight")
        with patch("baar.router.litellm.completion", return_value=upstream):
            assert "".join(router.stream_chat("hello")) == "abcd"
        assert router.log.steps[-1].completion_tokens == 9000  # provider usage

    @pytest.mark.asyncio
    async def test_async_early_exit_closes_upstream_and_records_reservation(self, priced):
        upstream = _AsyncUpstream(self._chunks())
        router = make_router(budget=1.0, on_output_overrun="preflight")
        with patch("baar.router.litellm.acompletion", new_callable=AsyncMock, return_value=upstream):
            gen = router.astream_chat("hello")
            assert await gen.__anext__() == "a"
            await gen.aclose()
        assert upstream.closed is True
        assert router.log.steps[-1].completion_tokens == MAX_OUT["gpt-4o-mini"]


# ── Round 2 #3: a user limit can't exceed the model's real max output ─────────

class TestUserLimitClamped:
    def test_preflight_clamps_to_model_max(self, priced):
        p = _plan("preflight", max_output_tokens=1_000_000)
        assert p.max_tokens == MAX_OUT["gpt-4o-mini"]

    def test_truncate_clamps_to_model_max(self, priced):
        p = _plan("truncate", remaining=100.0, max_output_tokens=1_000_000)
        assert p.max_tokens == MAX_OUT["gpt-4o-mini"]

    def test_estimate_clamps_sent_limit_to_model_max(self, priced):
        p = _plan("estimate", max_output_tokens=1_000_000)
        assert p.max_tokens == MAX_OUT["gpt-4o-mini"]

    def test_unknown_model_max_keeps_user_limit(self, priced):
        with patch("baar.core.caps.model_context_window", return_value=None):
            p = _plan("preflight", model="mystery", max_output_tokens=5000)
        assert p.max_tokens == 5000

    def test_router_sends_clamped_limit(self, priced):
        router = make_router(budget=1.0, on_output_overrun="preflight", max_output_tokens=1_000_000)
        with patch("baar.router.litellm.completion", side_effect=completion_echo()) as m:
            router.chat("hello")
        assert m.call_args.kwargs["max_tokens"] == MAX_OUT["gpt-4o-mini"]


# ── Round 2 #4: a safety margin on prompt tokens in preflight/truncate ────────

def completion_with_prompt(real_prompt_tokens, finish_reason="length"):
    """side_effect: the provider counts more prompt tokens than Baar estimated."""
    def _call(**kwargs):
        resp = make_response(model=kwargs["model"], completion_tokens=kwargs.get("max_tokens", 30),
                             finish_reason=finish_reason)
        resp.usage.prompt_tokens = real_prompt_tokens
        return resp
    return _call


class TestPromptMargin:
    def test_default_margin(self):
        assert make_router().config.prompt_margin == 0.2

    def test_negative_margin_rejected(self):
        with pytest.raises(ValueError, match="prompt_margin"):
            make_router(prompt_margin=-0.1)

    def test_plan_prices_prompt_with_margin(self, priced):
        p = _plan("preflight", prompt_margin=0.2)
        assert p.prompt_tokens == math.ceil(PROMPT_TOKENS * 1.2)

    def test_estimate_mode_has_no_margin(self, priced):
        p = _plan("estimate", prompt_margin=0.2)
        assert p.prompt_tokens == PROMPT_TOKENS

    def test_context_room_uses_the_raw_prompt(self, priced):
        p = plan_output("preflight", model="gpt-4o", prompt_tokens=120000, expected_output_tokens=500,
                        remaining=10.0, tracker=BudgetTracker(total_budget=10.0), prompt_margin=0.2)
        assert p.max_tokens == 8000

    def test_preflight_blocks_when_only_the_margin_doesnt_fit(self, priced):
        budget = cost("gpt-4o-mini", MAX_OUT["gpt-4o-mini"]) + 1e-9  # fits with the raw prompt only
        router = make_router(budget=budget, on_output_overrun="preflight", min_cost_threshold=0.0)
        with patch("baar.router.litellm.completion") as m:
            with pytest.raises(BudgetExhausted):
                router.chat("hello")
            m.assert_not_called()
        relaxed = make_router(budget=budget, on_output_overrun="preflight", min_cost_threshold=0.0,
                              prompt_margin=0.0)
        with patch("baar.router.litellm.completion", side_effect=completion_echo()) as m:
            relaxed.chat("hello")
        assert m.call_count == 1

    def test_truncate_prompt_undercount_does_not_overshoot(self, priced):
        router = make_router(budget=0.002, on_output_overrun="truncate")
        with patch("baar.router.litellm.completion", side_effect=completion_with_prompt(115)):
            router.chat("hello")
        assert router.spent <= 0.002

    def test_preflight_prompt_undercount_does_not_overshoot(self, priced):
        budget = cost("gpt-4o-mini", MAX_OUT["gpt-4o-mini"], prompt_tokens=120) + 1e-9
        router = make_router(budget=budget, on_output_overrun="preflight", min_cost_threshold=0.0)
        with patch("baar.router.litellm.completion", side_effect=completion_with_prompt(115)):
            router.chat("hello")
        assert router.spent <= budget


# ── Round 2 #5: Vercel proxy reports cut-offs and maps limit errors ───────────

class TestVercelFinishAndErrors:
    def _client(self, router):
        pytest.importorskip("fastapi", reason="fastapi not installed")
        from fastapi.testclient import TestClient
        from baar.integrations.vercel import create_app
        return TestClient(create_app(router), raise_server_exceptions=False)

    def _post(self, client, **extra):
        return client.post("/v1/chat/completions", json={
            "model": "baar", "messages": [{"role": "user", "content": "Say hello"}], **extra,
        })

    def test_step_records_finish_reason(self, priced):
        router = make_router(budget=1.0)
        with patch("baar.router.litellm.completion",
                   side_effect=completion_echo(finish_reason="length")):
            router.chat("hello")
        assert router.log.steps[-1].finish_reason == "length"

    def test_non_stream_reports_length(self, priced):
        router = make_router(budget=1.0)
        client = self._client(router)
        with patch("baar.router.litellm.acompletion", new_callable=AsyncMock,
                   return_value=make_response(finish_reason="length")):
            resp = self._post(client, max_tokens=50)
        assert resp.status_code == 200
        assert resp.json()["choices"][0]["finish_reason"] == "length"

    def test_non_stream_defaults_to_stop(self, priced):
        router = make_router(budget=1.0)
        client = self._client(router)
        with patch("baar.router.litellm.acompletion", new_callable=AsyncMock,
                   return_value=make_response(finish_reason="stop")):
            resp = self._post(client)
        assert resp.json()["choices"][0]["finish_reason"] == "stop"

    def test_stream_reports_length(self, priced):
        router = make_router(budget=1.0)
        client = self._client(router)

        async def _agen():
            for c in make_chunks(["a"], finish_reason="length", usage=(100, 30)):
                yield c

        with patch("baar.router.litellm.acompletion", new_callable=AsyncMock, return_value=_agen()):
            resp = self._post(client, stream=True, max_tokens=50)
        assert '"finish_reason": "length"' in resp.text

    def test_output_limit_unavailable_is_400(self):
        router = make_router(budget=1.0)
        client = self._client(router)
        with patch.object(router, "achat", new_callable=AsyncMock,
                          side_effect=OutputLimitUnavailable("no output limit for 'x'")):
            resp = self._post(client)
        assert resp.status_code == 400
        assert "invalid_request_error" in resp.text

    def test_stream_output_limit_unavailable_sends_error_chunk(self):
        router = make_router(budget=1.0)
        client = self._client(router)

        async def _boom(*a, **k):
            raise OutputLimitUnavailable("no output limit for 'x'")
            yield  # pragma: no cover

        with patch.object(router, "astream_chat", _boom):
            resp = self._post(client, stream=True)
        assert "invalid_request_error" in resp.text
        assert "[DONE]" in resp.text

    def test_stream_budget_exceeded_sends_budget_error_chunk(self):
        router = make_router(budget=1.0)
        client = self._client(router)

        async def _boom(*a, **k):
            raise BudgetExceeded(requested=1.0, remaining=0.1, model="(reservation)")
            yield  # pragma: no cover

        with patch.object(router, "astream_chat", _boom):
            resp = self._post(client, stream=True)
        assert "budget_exhausted" in resp.text


# ── Every call path: fallback skips, kill-switch, stream close errors ─────────

import asyncio

PATHS = ["chat", "stream_chat", "achat", "astream_chat"]


def call_path(router, path, task="hello", **kw):
    """Run one of the four entry points to completion and return the text."""
    if path == "chat":
        return router.chat(task, **kw)
    if path == "stream_chat":
        return "".join(router.stream_chat(task, **kw))
    if path == "achat":
        return asyncio.run(router.achat(task, **kw))

    async def _collect():
        return "".join([t async for t in router.astream_chat(task, **kw)])
    return asyncio.run(_collect())


def provider(fail_models=(), calls=None):
    """Patches for both litellm entry points: fail some models, otherwise answer (streamed or not)."""
    def _resp(kwargs):
        if kwargs["model"] in fail_models:
            raise ConnectionError(f"{kwargs['model']} down")
        if calls is not None:
            calls.append(kwargs["model"])
        if kwargs.get("stream"):
            return iter(make_chunks(["ok"], usage=(100, 30)))
        return make_response(model=kwargs["model"])

    async def _aresp(**kwargs):
        r = _resp(kwargs)
        if kwargs.get("stream"):
            chunks = list(r)

            async def _agen():
                for c in chunks:
                    yield c
            return _agen()
        return r

    return (patch("baar.router.litellm.completion", side_effect=lambda **kw: _resp(kw)),
            patch("baar.router.litellm.acompletion", side_effect=_aresp))


class TestAllPaths:
    @pytest.mark.parametrize("path", PATHS)
    def test_unaffordable_fallback_skipped(self, priced, path):
        router = make_router(budget=0.05, on_output_overrun="preflight", small_fallback_models=["gpt-4o"])
        calls = []
        p1, p2 = provider(fail_models={"gpt-4o-mini"}, calls=calls)
        with p1, p2, pytest.raises(RuntimeError, match="unaffordable"):
            call_path(router, path)
        assert calls == []
        assert router.spent == pytest.approx(0.0)

    @pytest.mark.parametrize("path", PATHS)
    def test_fallback_without_output_limit_skipped(self, priced, path):
        router = make_router(budget=1.0, on_output_overrun="preflight", small_fallback_models=["mystery-fb"])
        calls = []
        p1, p2 = provider(fail_models={"gpt-4o-mini"}, calls=calls)
        with p1, p2, pytest.raises(RuntimeError, match=r"mystery-fb: skipped \("):
            call_path(router, path)
        assert calls == []

    @pytest.mark.parametrize("path", PATHS)
    def test_truncate_below_min_useful_blocks_locally(self, priced, path):
        router = make_router(budget=0.0005, on_output_overrun="truncate",
                             min_useful_tokens=1000, min_cost_threshold=0.0)
        calls = []
        p1, p2 = provider(calls=calls)
        with p1, p2, pytest.raises(BudgetExhausted, match="useful response"):
            call_path(router, path)
        assert calls == []

    @pytest.mark.parametrize("path", PATHS)
    def test_invalid_per_call_limit_rejected(self, priced, path):
        router = make_router(budget=1.0)
        calls = []
        p1, p2 = provider(calls=calls)
        with p1, p2, pytest.raises(ValueError, match="max_output_tokens"):
            call_path(router, path, max_output_tokens=0)
        assert calls == []

    @pytest.mark.parametrize("path", PATHS)
    def test_fallback_answers_when_affordable(self, priced, path):
        router = make_router(budget=1.0, small_fallback_models=["gpt-4o"])
        calls = []
        p1, p2 = provider(fail_models={"gpt-4o-mini"}, calls=calls)
        with p1, p2:
            assert call_path(router, path) == "ok"
        assert calls == ["gpt-4o"]
        assert router.log.steps[-1].model_used == "gpt-4o"


class TestStreamCloseEdgeCases:
    def test_close_stream_ignores_streams_without_close(self):
        from baar.router import _close_stream
        _close_stream(iter([]))  # list iterators have no close(): nothing to do

    def test_close_stream_swallows_errors(self):
        from baar.router import _close_stream

        class Boom:
            def close(self):
                raise RuntimeError("already closed")
        _close_stream(Boom())

    @pytest.mark.asyncio
    async def test_async_aclose_error_is_swallowed(self, priced):
        class Upstream(_AsyncUpstream):
            async def aclose(self):
                raise RuntimeError("connection reset")

        router = make_router(budget=1.0, on_output_overrun="preflight")
        with patch("baar.router.litellm.acompletion", new_callable=AsyncMock,
                   return_value=Upstream(make_chunks(["a", "b"], usage=(100, 30)))):
            gen = router.astream_chat("hello")
            await gen.__anext__()
            await gen.aclose()
        assert router.log.steps[-1].completion_tokens == MAX_OUT["gpt-4o-mini"]

    def test_no_usage_warning_fires_once_per_model(self, priced):
        router = make_router(budget=1.0)
        with patch("baar.router.litellm.completion", side_effect=lambda **kw: iter(make_chunks(["a"]))):
            with pytest.warns(UserWarning, match="usage"):
                list(router.stream_chat("hello"))
            with warnings.catch_warnings():
                warnings.simplefilter("error", UserWarning)
                list(router.stream_chat("hello"))

    @pytest.mark.asyncio
    async def test_async_zero_completion_with_text_is_not_trusted(self, priced):
        chunks = make_chunks(["visible answer"])
        chunks[-1].usage = SimpleNamespace(prompt_tokens=100, completion_tokens=0)

        async def _agen():
            for c in chunks:
                yield c

        router = make_router(budget=1.0)
        with patch("baar.router.litellm.acompletion", new_callable=AsyncMock, return_value=_agen()), \
             pytest.warns(UserWarning, match="usage"):
            [t async for t in router.astream_chat("hello")]
        assert router.log.steps[-1].completion_tokens == PROMPT_TOKENS
