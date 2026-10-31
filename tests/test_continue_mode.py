"""
tests/test_continue_mode.py — on_output_overrun="continue".

The answer is generated in segments of `segment_tokens`. Each segment reserves
only its own cost; when one stops at its limit, Baar asks the model to continue
where it stopped. The answer is cut off only when the budget can't pay for
another useful segment — then the text so far is returned and the last step is
flagged truncated_by_budget. Runs without API keys.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from baar import BAARRouter, BudgetExhausted, TaskRejected
from baar.core import caps

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


class Writer:
    """A model whose full answer is `total` tokens, written `piece` per segment as text."""

    def __init__(self, total, piece_text=None):
        self.total = total
        self.written = 0
        self.calls = []

    def _next(self, kwargs):
        self.calls.append(kwargs)
        limit = kwargs.get("max_tokens") or 10**9
        n = min(limit, self.total - self.written)
        self.written += n
        finish = "stop" if self.written >= self.total else "length"
        return n, finish, f"[{n}]"

    def completion(self, **kwargs):
        n, finish, text = self._next(kwargs)
        if kwargs.get("stream"):
            return iter([SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=text), finish_reason=finish)],
                usage=SimpleNamespace(prompt_tokens=PROMPT, completion_tokens=n))])
        r = MagicMock()
        r.model = kwargs["model"]
        r.choices[0].message.content = text
        r.choices[0].finish_reason = finish
        r.usage.prompt_tokens = PROMPT
        r.usage.completion_tokens = n
        return r

    async def acompletion(self, **kwargs):
        r = self.completion(**kwargs)
        if kwargs.get("stream"):
            chunks = list(r)

            async def _agen():
                for c in chunks:
                    yield c
            return _agen()
        return r


def router(**kw):
    kw.setdefault("use_llm_router", False)
    kw.setdefault("prompt_margin", 0.0)
    kw.setdefault("on_output_overrun", "continue")
    return BAARRouter(**kw)


def patched(writer):
    return (patch("baar.router.litellm.completion", side_effect=writer.completion),
            patch("baar.router.litellm.acompletion", side_effect=writer.acompletion))


# ── configuration ─────────────────────────────────────────────────────────────

class TestConfig:
    def test_continue_is_a_mode(self):
        caps.validate_mode("continue")

    def test_defaults(self):
        r = router()
        assert (r.config.segment_tokens, r.config.max_segments) == (2000, 8)

    @pytest.mark.parametrize("kw", [{"segment_tokens": 10}, {"max_segments": 0}])
    def test_invalid_settings(self, kw):
        with pytest.raises(ValueError):
            router(**kw)

    def test_plan_output_treats_continue_like_truncate(self, priced):
        from baar.core.budget import BudgetTracker
        a = caps.plan_output("continue", model="gpt-4o-mini", prompt_tokens=PROMPT, expected_output_tokens=500,
                             remaining=0.002, tracker=BudgetTracker(total_budget=1.0))
        b = caps.plan_output("truncate", model="gpt-4o-mini", prompt_tokens=PROMPT, expected_output_tokens=500,
                             remaining=0.002, tracker=BudgetTracker(total_budget=1.0))
        assert a == b


# ── segments ──────────────────────────────────────────────────────────────────

class TestSegments:
    def test_long_answer_is_continued_until_done(self, priced):
        w = Writer(total=5000)
        p1, p2 = patched(w)
        r = router(budget=1.0)
        with p1, p2:
            out = r.chat("write a long report")
        assert out == "[2000][2000][1000]"
        assert [c["max_tokens"] for c in w.calls] == [2000, 2000, 2000]
        assert len(r.log.steps) == 3
        assert not any(s.truncated_by_budget for s in r.log.steps)

    def test_continuation_sends_the_text_so_far(self, priced):
        w = Writer(total=3000)
        p1, p2 = patched(w)
        r = router(budget=1.0)
        with p1, p2:
            r.chat("write a long report")
        second = w.calls[1]["messages"]
        assert second[-2] == {"role": "assistant", "content": "[2000]"}
        assert second[-1]["role"] == "user" and "continue" in second[-1]["content"].lower()

    def test_short_answer_is_one_call(self, priced):
        w = Writer(total=300)
        p1, p2 = patched(w)
        r = router(budget=1.0)
        with p1, p2:
            assert r.chat("hi") == "[300]"
        assert len(w.calls) == 1

    def test_stops_at_max_segments(self, priced):
        w = Writer(total=100_000)
        p1, p2 = patched(w)
        r = router(budget=1.0, max_segments=3)
        with p1, p2:
            r.chat("go")
        assert len(w.calls) == 3

    def test_user_limit_caps_the_total(self, priced):
        w = Writer(total=10_000)
        p1, p2 = patched(w)
        r = router(budget=1.0, max_output_tokens=3000)
        with p1, p2:
            assert r.chat("go") == "[2000][1000]"
        assert [c["max_tokens"] for c in w.calls] == [2000, 1000]

    def test_each_segment_reserves_only_its_own_cost(self, priced):
        w = Writer(total=4000)
        held = []
        r = router(budget=1.0)

        def _call(**kw):
            held.append(r.spent)
            return w.completion(**kw)

        with patch("baar.router.litellm.completion", side_effect=_call):
            r.chat("go")
        assert held[0] == pytest.approx(cost("gpt-4o-mini", 2000))

    def test_budget_running_out_returns_text_so_far_and_flags_it(self, priced):
        # Enough for ~2.5 segments.
        budget = cost("gpt-4o-mini", 2000) * 2 + cost("gpt-4o-mini", 1000)
        w = Writer(total=100_000)
        p1, p2 = patched(w)
        r = router(budget=budget, min_cost_threshold=0.0)
        with p1, p2:
            out = r.chat("go")
        assert out.startswith("[2000][2000]")
        assert r.log.steps[-1].truncated_by_budget is True
        assert r.spent <= budget

    def test_no_budget_for_the_first_segment_raises(self, priced):
        w = Writer(total=5000)
        p1, p2 = patched(w)
        r = router(budget=cost("gpt-4o-mini", 10), min_cost_threshold=0.0)
        with p1, p2:
            with pytest.raises(BudgetExhausted):
                r.chat("go")
        assert w.calls == []

    def test_value_gate_rejecting_a_later_segment_returns_text_so_far(self, priced):
        w = Writer(total=5000)
        p1, p2 = patched(w)
        values = iter([1.0, 0.0, 0.0])
        r = router(budget=1.0, value_fn=lambda t: next(values), max_reject_rate=1.0,
                   value_reject_alpha=1.0)
        with p1, p2:
            out = r.chat("go")
        assert out == "[2000]"

    def test_never_overshoots(self, priced):
        budget = 0.004
        w = Writer(total=100_000)
        p1, p2 = patched(w)
        r = router(budget=budget, min_cost_threshold=0.0)
        with p1, p2:
            r.chat("go")
        assert r.spent <= budget


# ── other entry points ────────────────────────────────────────────────────────

class TestOtherPaths:
    def test_stream(self, priced):
        w = Writer(total=4500)
        p1, p2 = patched(w)
        r = router(budget=1.0)
        with p1, p2:
            assert "".join(r.stream_chat("go")) == "[2000][2000][500]"

    def test_achat(self, priced):
        w = Writer(total=4500)
        p1, p2 = patched(w)
        r = router(budget=1.0)
        with p1, p2:
            assert asyncio.run(r.achat("go")) == "[2000][2000][500]"

    def test_astream(self, priced):
        w = Writer(total=4500)
        p1, p2 = patched(w)
        r = router(budget=1.0)

        async def _collect():
            return "".join([t async for t in r.astream_chat("go")])
        with p1, p2:
            assert asyncio.run(_collect()) == "[2000][2000][500]"

    def test_stream_budget_running_out(self, priced):
        budget = cost("gpt-4o-mini", 2000) + cost("gpt-4o-mini", 500)
        w = Writer(total=100_000)
        p1, p2 = patched(w)
        r = router(budget=budget, min_cost_threshold=0.0)
        with p1, p2:
            out = "".join(r.stream_chat("go"))
        assert out.startswith("[2000]")
        assert r.log.steps[-1].truncated_by_budget is True

    def test_per_call_mode(self, priced):
        w = Writer(total=4000)
        p1, p2 = patched(w)
        r = router(budget=1.0, on_output_overrun="estimate")
        with p1, p2:
            assert r.chat("go", on_output_overrun="continue") == "[2000][2000]"


class TestEarlyExit:
    def test_stopping_a_continue_stream_closes_the_provider_stream(self, priced):
        closed = []

        class Upstream:
            def __init__(self, chunks):
                self._it = iter(chunks)

            def __iter__(self):
                return self

            def __next__(self):
                return next(self._it)

            def close(self):
                closed.append(True)

        chunk = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="a"), finish_reason=None)])
        r = router(budget=1.0)
        with patch("baar.router.litellm.completion", side_effect=lambda **kw: Upstream([chunk, chunk])):
            gen = r.stream_chat("go")
            next(gen)
            gen.close()
        assert closed == [True]

    def test_stopping_an_async_continue_stream_closes_the_provider_stream(self, priced):
        closed = []

        class Upstream:
            def __init__(self):
                self.n = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                self.n += 1
                if self.n > 3:
                    raise StopAsyncIteration
                return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="a"), finish_reason=None)])

            async def aclose(self):
                closed.append(True)

        async def _acompletion(**kw):
            return Upstream()

        r = router(budget=1.0)

        async def _run():
            gen = r.astream_chat("go")
            await gen.__anext__()
            await gen.aclose()

        with patch("baar.router.litellm.acompletion", side_effect=_acompletion):
            asyncio.run(_run())
        assert closed == [True]


class TestAllPathsStops:
    BUDGET_FOR_TWO_AND_A_HALF = None

    def _budget(self):
        return cost("gpt-4o-mini", 2000) * 2 + cost("gpt-4o-mini", 1000)

    def test_achat_budget_running_out(self, priced):
        w = Writer(total=100_000)
        p1, p2 = patched(w)
        r = router(budget=self._budget(), min_cost_threshold=0.0)
        with p1, p2:
            out = asyncio.run(r.achat("go"))
        assert out.startswith("[2000][2000]")
        assert r.log.steps[-1].truncated_by_budget is True

    def test_achat_no_budget_for_first_segment_raises(self, priced):
        w = Writer(total=5000)
        p1, p2 = patched(w)
        r = router(budget=cost("gpt-4o-mini", 10), min_cost_threshold=0.0)
        with p1, p2, pytest.raises(BudgetExhausted):
            asyncio.run(r.achat("go"))

    def test_stream_no_budget_for_first_segment_raises(self, priced):
        w = Writer(total=5000)
        p1, p2 = patched(w)
        r = router(budget=cost("gpt-4o-mini", 10), min_cost_threshold=0.0)
        with p1, p2, pytest.raises(BudgetExhausted):
            list(r.stream_chat("go"))

    def test_astream_budget_running_out_and_no_budget(self, priced):
        async def _collect(r):
            return "".join([t async for t in r.astream_chat("go")])

        w = Writer(total=100_000)
        p1, p2 = patched(w)
        r = router(budget=self._budget(), min_cost_threshold=0.0)
        with p1, p2:
            assert asyncio.run(_collect(r)).startswith("[2000][2000]")
        assert r.log.steps[-1].truncated_by_budget is True
        broke = router(budget=cost("gpt-4o-mini", 10), min_cost_threshold=0.0)
        with p1, p2, pytest.raises(BudgetExhausted):
            asyncio.run(_collect(broke))

    @pytest.mark.parametrize("path", ["stream", "astream"])
    def test_segment_whose_recording_failed_ends_the_answer(self, priced, path):
        w = Writer(total=100_000)
        p1, p2 = patched(w)
        r = router(budget=1.0)

        async def _collect():
            return "".join([t async for t in r.astream_chat("go")])

        with p1, p2, patch.object(r._tracker, "record_manual", side_effect=RuntimeError("disk full")):
            out = "".join(r.stream_chat("go")) if path == "stream" else asyncio.run(_collect())
        assert out == "[2000]"
        assert any("post-stream recording failed" in e for e in r.log.errors)

    def test_policy_violation_mid_answer_returns_text_so_far_unflagged(self, priced):
        from baar import Policy, Rule
        w = Writer(total=5000)
        p1, p2 = patched(w)
        # Blocks once the run has used any budget, i.e. from the second segment on.
        r = router(budget=1.0, policy=Policy(rules=[Rule(when={"utilization": "> 0"}, then="block")]))
        with p1, p2:
            assert r.chat("go") == "[2000]"
        assert r.log.steps[-1].truncated_by_budget is False


def test_full_segment_then_no_money_flags_the_answer(priced):
    # Two full segments fit; a third couldn't pay for even min_useful_tokens.
    budget = cost("gpt-4o-mini", 2000) * 2 + cost("gpt-4o-mini", 20)
    w = Writer(total=100_000)
    p1, p2 = patched(w)
    r = router(budget=budget, min_cost_threshold=0.0)
    with p1, p2:
        assert r.chat("go") == "[2000][2000]"
    last = r.log.steps[-1]
    assert last.completion_tokens == 2000 and last.finish_reason == "length"
    assert last.truncated_by_budget is True  # flagged by the segment loop, not by truncate
