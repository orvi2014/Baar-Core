"""
tests/test_overrun_risk.py — overrun_risk: reserve a learned output length, not the worst case.

In "preflight", overrun_risk=r reserves the split-conformal (1 - r) quantile of
the output lengths a model actually produced, instead of its full max output.
max_tokens is unchanged, so a response is never cut off; at most about r of
calls can run past the reservation. Until a model has enough history, the
worst case is reserved.
"""

import asyncio
import math
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from baar import BAARRouter, BudgetExhausted
from baar.core.lengths import OutputLengths

IN_PRICE = {"gpt-4o-mini": 1.5e-7, "gpt-4o": 2.5e-6}
OUT_PRICE = {"gpt-4o-mini": 6e-7, "gpt-4o": 1e-5}
MAX_OUT = {"gpt-4o-mini": 16384, "gpt-4o": 16384}
CONTEXT = {"gpt-4o-mini": 128000, "gpt-4o": 128000}
PROMPT = 100


def _cost_per_token(model, prompt_tokens=0, completion_tokens=0, **_):
    return prompt_tokens * IN_PRICE[model], completion_tokens * OUT_PRICE[model]


def cost(model, completion_tokens, prompt_tokens=PROMPT):
    i, o = _cost_per_token(model, prompt_tokens, completion_tokens)
    return i + o


def _completion_cost(completion_response=None, **_):
    u = completion_response.usage
    return cost(completion_response.model, u.completion_tokens, u.prompt_tokens)


@pytest.fixture
def priced():
    with patch("baar.core.budget.cost_per_token", side_effect=_cost_per_token), \
         patch("baar.core.budget.completion_cost", side_effect=_completion_cost), \
         patch("baar.router.token_counter", return_value=PROMPT), \
         patch("baar.core.caps.model_max_output_tokens", side_effect=MAX_OUT.get), \
         patch("baar.core.caps.model_context_window", side_effect=CONTEXT.get), \
         patch("baar.core.caps.supports_stream_usage", return_value=True):
        yield


def response(model, completion_tokens):
    r = MagicMock()
    r.model = model
    r.choices[0].message.content = "ok"
    r.choices[0].finish_reason = "stop"
    r.usage.prompt_tokens = PROMPT
    r.usage.completion_tokens = completion_tokens
    return r


def router(**kw):
    kw.setdefault("use_llm_router", False)
    kw.setdefault("prompt_margin", 0.0)
    return BAARRouter(**kw)


def warm_up(r, n=20, tokens=200, model="gpt-4o-mini"):
    with patch("baar.router.litellm.completion", side_effect=lambda **kw: response(kw["model"], tokens)):
        for _ in range(n):
            r.chat("hello")


# ── OutputLengths ─────────────────────────────────────────────────────────────

class TestOutputLengths:
    def test_no_prediction_before_min_samples(self):
        lengths = OutputLengths(min_samples=20)
        for _ in range(19):
            lengths.observe("m", 100)
        assert lengths.quantile("m", 0.05) is None
        lengths.observe("m", 100)
        assert lengths.quantile("m", 0.05) == 100

    def test_split_conformal_index(self):
        lengths = OutputLengths(min_samples=20)
        for n in range(1, 101):
            lengths.observe("m", n)
        # ceil((n + 1) * (1 - r)) - 1 with n = 100, r = 0.05 → index 95 → value 96.
        assert lengths.quantile("m", 0.05) == 96

    def test_too_little_history_for_the_risk_falls_back(self):
        lengths = OutputLengths(min_samples=5)
        for n in range(1, 11):
            lengths.observe("m", n)
        # ceil(11 * 0.99) - 1 = 10 > last index 9: the sample can't support r = 0.01.
        assert lengths.quantile("m", 0.01) is None

    def test_per_model(self):
        lengths = OutputLengths(min_samples=1)
        lengths.observe("a", 10)
        assert lengths.quantile("b", 0.5) is None

    def test_window_keeps_recent_lengths(self):
        lengths = OutputLengths(min_samples=1, window=10)
        for _ in range(10):
            lengths.observe("m", 5000)
        for _ in range(10):
            lengths.observe("m", 50)
        assert lengths.quantile("m", 0.1) == 50

    @pytest.mark.parametrize("bad", [0, -1])
    def test_invalid_settings(self, bad):
        with pytest.raises(ValueError):
            OutputLengths(min_samples=bad)
        with pytest.raises(ValueError):
            OutputLengths(window=bad)

    def test_risk_bounds(self):
        lengths = OutputLengths(min_samples=1)
        lengths.observe("m", 1)
        assert lengths.quantile("m", 0) is None  # r = 0 means worst case
        with pytest.raises(ValueError):
            lengths.quantile("m", 1.0)

    def test_negative_or_non_int_lengths_ignored(self):
        lengths = OutputLengths(min_samples=1)
        lengths.observe("m", -5)
        lengths.observe("m", "x")
        assert lengths.quantile("m", 0.5) is None

    def test_thread_safe(self):
        lengths = OutputLengths(min_samples=1, window=10_000)

        def add():
            for _ in range(1000):
                lengths.observe("m", 7)
        threads = [threading.Thread(target=add) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert lengths.count("m") == 8000


# ── router ────────────────────────────────────────────────────────────────────

class TestRouterOverrunRisk:
    def test_default_is_worst_case(self):
        assert router().config.overrun_risk == 0.0

    @pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5])
    def test_invalid_risk_rejected(self, bad):
        with pytest.raises(ValueError, match="overrun_risk"):
            router(overrun_risk=bad)

    def test_reserves_worst_case_until_warmed_up_then_the_quantile(self, priced):
        r = router(budget=10.0, on_output_overrun="preflight", overrun_risk=0.05)
        held = []

        def _call(**kw):
            held.append(r.spent)
            return response(kw["model"], 200)

        with patch("baar.router.litellm.completion", side_effect=_call):
            for _ in range(21):
                r.chat("hello")
        first_reservation = held[0]
        assert first_reservation == pytest.approx(cost("gpt-4o-mini", MAX_OUT["gpt-4o-mini"]))
        spent_before_21st = 20 * cost("gpt-4o-mini", 200)
        assert held[20] - spent_before_21st == pytest.approx(cost("gpt-4o-mini", 200))

    def test_max_tokens_is_still_the_full_limit(self, priced):
        r = router(budget=10.0, on_output_overrun="preflight", overrun_risk=0.05)
        warm_up(r)
        with patch("baar.router.litellm.completion",
                   side_effect=lambda **kw: response(kw["model"], 200)) as m:
            r.chat("hello")
        assert m.call_args.kwargs["max_tokens"] == MAX_OUT["gpt-4o-mini"]

    def test_learned_reservation_makes_a_small_budget_usable(self, priced):
        r = router(budget=10.0, on_output_overrun="preflight", overrun_risk=0.05)
        warm_up(r)
        # Move the learned history into a router with a small budget.
        small = router(budget=0.005, on_output_overrun="preflight", overrun_risk=0.05,
                       output_lengths=r.output_lengths)
        with patch("baar.router.litellm.completion",
                   side_effect=lambda **kw: response(kw["model"], 200)) as m:
            small.chat("hello")  # worst case ($0.0098) wouldn't fit; the quantile does
        assert m.call_count == 1
        cold = router(budget=0.005, on_output_overrun="preflight", overrun_risk=0.05)
        with patch("baar.router.litellm.completion") as m:
            with pytest.raises(BudgetExhausted):
                cold.chat("hello")
            m.assert_not_called()

    def test_risk_zero_ignores_history(self, priced):
        r = router(budget=10.0, on_output_overrun="preflight")
        warm_up(r)
        held = []

        def _call(**kw):
            held.append(r.spent)
            return response(kw["model"], 200)

        before = r.spent
        with patch("baar.router.litellm.completion", side_effect=_call):
            r.chat("hello")
        assert held[0] - before == pytest.approx(cost("gpt-4o-mini", MAX_OUT["gpt-4o-mini"]))

    def test_other_modes_are_unaffected(self, priced):
        r = router(budget=10.0, overrun_risk=0.05)  # estimate mode
        warm_up(r)
        with patch("baar.router.litellm.completion",
                   side_effect=lambda **kw: response(kw["model"], 200)) as m:
            r.chat("hello")
        assert "max_tokens" not in m.call_args.kwargs

    def test_every_completed_call_is_observed(self, priced):
        r = router(budget=10.0)
        warm_up(r, n=3, tokens=321)
        assert r.output_lengths.count("gpt-4o-mini") == 3

    def test_stream_completed_is_observed_and_early_exit_is_not(self, priced):
        def chunks():
            return iter([SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=t), finish_reason=None)]) for t in "abc"]
                + [SimpleNamespace(choices=[], usage=SimpleNamespace(prompt_tokens=PROMPT, completion_tokens=444))])

        r = router(budget=10.0, on_output_overrun="preflight")
        with patch("baar.router.litellm.completion", side_effect=lambda **kw: chunks()):
            list(r.stream_chat("hello"))
            gen = r.stream_chat("hello")
            next(gen)
            gen.close()
        assert r.output_lengths.count("gpt-4o-mini") == 1

    def test_async_paths_observe(self, priced):
        r = router(budget=10.0)

        async def _run():
            await r.achat("hello")

            async def _agen():
                yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="x"), finish_reason="stop")],
                                      usage=SimpleNamespace(prompt_tokens=PROMPT, completion_tokens=50))
            with patch("baar.router.litellm.acompletion", new_callable=AsyncMock, return_value=_agen()):
                [t async for t in r.astream_chat("hello")]

        with patch("baar.router.litellm.acompletion", new_callable=AsyncMock,
                   return_value=response("gpt-4o-mini", 77)):
            asyncio.run(_run())
        assert r.output_lengths.count("gpt-4o-mini") == 2

    def test_shared_history_across_routers(self, priced):
        shared = OutputLengths()
        a = router(budget=10.0, output_lengths=shared)
        b = router(budget=10.0, output_lengths=shared)
        warm_up(a, n=2)
        warm_up(b, n=3)
        assert shared.count("gpt-4o-mini") == 5
