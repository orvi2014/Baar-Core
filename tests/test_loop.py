"""
tests/test_loop.py — LoopDetector and baar_guard(loop_detector=...) unit tests

Runs entirely without API keys. Every test is deterministic and fast.
"""

import asyncio

import pytest

from baar import BAARRouter, LoopDetected, LoopDetector, LoopEvent
from baar.integrations.tools import baar_guard


def run(detector: LoopDetector, tool: str, args, result=None, error=None):
    """One step: before() then after(). Returns the warning, if any."""
    detector.before(tool, args)
    return detector.after(tool, args, result=result, error=error)


# ── repeat ────────────────────────────────────────────────────────────────────

class TestRepeat:
    def test_refuses_the_call_after_max_repeats_identical_results(self):
        d = LoopDetector(max_repeats=3)
        for _ in range(3):
            run(d, "search", {"q": "x"}, result="nothing")
        with pytest.raises(LoopDetected) as exc:
            d.before("search", {"q": "x"})
        assert exc.value.pattern == "repeat"
        assert exc.value.count == 3
        assert exc.value.tool_name == "search"

    def test_same_call_with_a_new_result_is_not_a_loop(self):
        d = LoopDetector(max_repeats=3)
        for status in ["queued", "running", "running", "done"]:
            run(d, "status", {"job": 1}, result=status)
        d.before("status", {"job": 1})  # no result came back 3 times

    def test_different_arguments_never_add_up(self):
        d = LoopDetector(max_repeats=2)
        for page in range(10):
            run(d, "list", {"cursor": page}, result="[]")
        d.before("list", {"cursor": 10})

    def test_argument_order_and_whitespace_do_not_matter(self):
        d = LoopDetector(max_repeats=2)
        run(d, "search", {"q": "hello  world", "n": 5}, result="r")
        run(d, "search", {"n": 5, "q": " hello world "}, result="r")
        with pytest.raises(LoopDetected):
            d.before("search", {"q": "hello world", "n": 5})

    def test_ignore_args_drops_volatile_fields_at_any_depth(self):
        d = LoopDetector(max_repeats=2, ignore_args=["request_id"])
        run(d, "fetch", {"url": "u", "meta": {"request_id": 1}}, result="r")
        run(d, "fetch", {"url": "u", "meta": {"request_id": 2}}, result="r")
        with pytest.raises(LoopDetected):
            d.before("fetch", {"url": "u", "meta": {"request_id": 3}})

    def test_normalize_result_strips_volatile_output(self):
        d = LoopDetector(max_repeats=2, normalize_result=lambda r: r["items"])
        run(d, "fetch", {"u": 1}, result={"items": [], "at": 1})
        run(d, "fetch", {"u": 1}, result={"items": [], "at": 2})
        with pytest.raises(LoopDetected):
            d.before("fetch", {"u": 1})

    def test_exempt_tools_are_never_counted(self):
        d = LoopDetector(max_repeats=1, exempt_tools=["poll"])
        for _ in range(10):
            run(d, "poll", {}, result="running")
        assert d.steps == 0


# ── error ─────────────────────────────────────────────────────────────────────

class TestError:
    def test_refuses_after_the_same_error_max_error_repeats_times(self):
        d = LoopDetector(max_error_repeats=3)
        for _ in range(3):
            run(d, "deploy", {"env": "prod"}, error=ValueError("bad   config"))
        with pytest.raises(LoopDetected) as exc:
            d.before("deploy", {"env": "prod"})
        assert exc.value.pattern == "error"

    def test_a_different_error_is_a_different_outcome(self):
        d = LoopDetector(max_error_repeats=2)
        run(d, "deploy", {}, error=ValueError("a"))
        run(d, "deploy", {}, error=ValueError("b"))
        d.before("deploy", {})

    def test_a_success_in_between_resets_the_error_count(self):
        d = LoopDetector(max_error_repeats=2)
        run(d, "api", {}, error=TimeoutError("429"))
        run(d, "api", {}, result="ok")
        run(d, "api", {}, error=TimeoutError("429"))
        d.before("api", {})  # flaky, not stuck
        run(d, "api", {}, error=TimeoutError("429"))  # second in a row…
        with pytest.raises(LoopDetected):
            d.before("api", {})  # …so the third attempt is refused


# ── ping-pong ─────────────────────────────────────────────────────────────────

class TestPingPong:
    def test_refuses_the_step_that_would_continue_an_alternation(self):
        d = LoopDetector(max_ping_pong=3, max_repeats=10)
        for _ in range(3):
            run(d, "read", {"f": "a"}, result="x")
            run(d, "write", {"f": "a"}, result="ok")
        with pytest.raises(LoopDetected) as exc:
            d.before("read", {"f": "a"})
        assert exc.value.pattern == "ping_pong"
        assert exc.value.count == 3
        # Breaking the alternation is allowed.
        d.before("test", {})

    def test_alternation_with_changing_results_is_progress(self):
        d = LoopDetector(max_ping_pong=2, max_repeats=10)
        for i in range(5):
            run(d, "read", {"f": "a"}, result=f"v{i}")
            run(d, "write", {"f": "a"}, result="ok")
        d.before("read", {"f": "a"})


# ── warnings, tighten, reset ──────────────────────────────────────────────────

class TestEvents:
    def test_warns_once_on_the_step_after_which_the_next_is_refused(self):
        seen: list[LoopEvent] = []
        d = LoopDetector(max_repeats=3, on_loop=seen.append)
        run(d, "s", {}, result="r")
        assert run(d, "s", {}, result="r") is None
        assert run(d, "s", {}, result="r").action == "warn"
        with pytest.raises(LoopDetected):
            d.before("s", {})
        assert [e.action for e in seen] == ["warn", "stop"]

    def test_a_first_error_is_not_a_warning(self):
        d = LoopDetector(max_error_repeats=2)
        assert run(d, "x", {}, error=ValueError("e")) is None

    def test_warn_false_only_stops(self):
        seen: list[LoopEvent] = []
        d = LoopDetector(max_repeats=2, warn=False, on_loop=seen.append)
        run(d, "s", {}, result="r")
        run(d, "s", {}, result="r")
        with pytest.raises(LoopDetected):
            d.before("s", {})
        assert [e.action for e in seen] == ["stop"]

    def test_a_broken_callback_cannot_break_the_run(self):
        def boom(_):
            raise RuntimeError("callback bug")
        d = LoopDetector(max_repeats=2, on_loop=boom)
        run(d, "s", {}, result="r")
        run(d, "s", {}, result="r")
        with pytest.raises(LoopDetected):
            d.before("s", {})

    def test_tighten_lowers_thresholds_but_not_below_one(self):
        d = LoopDetector(max_repeats=2, max_error_repeats=1, max_ping_pong=3)
        d.tighten()
        assert (d.max_repeats, d.max_error_repeats, d.max_ping_pong) == (1, 1, 2)
        run(d, "s", {}, result="r")
        with pytest.raises(LoopDetected):
            d.before("s", {})

    def test_reset_forgets_the_run(self):
        d = LoopDetector(max_repeats=1)
        run(d, "s", {}, result="r")
        d.reset()
        d.before("s", {})
        assert d.steps == 0 and d.events == []

    def test_thresholds_must_be_positive(self):
        with pytest.raises(ValueError):
            LoopDetector(max_repeats=0)


# ── concurrency ───────────────────────────────────────────────────────────────

class TestConcurrency:
    def test_counts_stay_exact_across_threads(self):
        import threading

        d = LoopDetector(max_repeats=800, warn=False)
        barrier = threading.Barrier(8)

        def worker():
            barrier.wait()
            for _ in range(100):
                d.after("s", {}, result="r")

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert d.steps == 800
        with pytest.raises(LoopDetected) as exc:
            d.before("s", {})
        assert exc.value.count == 800


# ── baar_guard integration ────────────────────────────────────────────────────

def make_router() -> BAARRouter:
    return BAARRouter(budget=1.0, use_llm_router=False)


class TestGuard:
    def test_refused_call_is_never_made(self):
        calls = []
        loops = LoopDetector(max_repeats=2)

        @baar_guard(router=make_router(), loop_detector=loops)
        def search(query: str) -> str:
            calls.append(query)
            return "no results"

        search("x")
        search(query="x")  # keyword or positional: same call
        with pytest.raises(LoopDetected):
            search("x")
        assert calls == ["x", "x"]

    def test_errors_are_reported_and_reraised(self):
        loops = LoopDetector(max_error_repeats=2)

        @baar_guard(router=make_router(), loop_detector=loops)
        def flaky(n: int) -> int:
            raise ConnectionError("refused")

        for _ in range(2):
            with pytest.raises(ConnectionError):
                flaky(1)
        with pytest.raises(LoopDetected):
            flaky(1)

    def test_one_detector_spans_tools(self):
        loops = LoopDetector(max_ping_pong=2, max_repeats=10)
        router = make_router()

        @baar_guard(router=router, loop_detector=loops)
        def read() -> str:
            return "same"

        @baar_guard(router=router, loop_detector=loops)
        def write() -> str:
            return "ok"

        for _ in range(2):
            read()
            write()
        with pytest.raises(LoopDetected):
            read()

    def test_async_tools(self):
        loops = LoopDetector(max_repeats=1)

        @baar_guard(router=make_router(), loop_detector=loops)
        async def fetch(url: str) -> str:
            return "page"

        async def go():
            await fetch("u")
            with pytest.raises(LoopDetected):
                await fetch("u")

        asyncio.run(go())

    def test_refused_call_does_not_count_against_max_calls(self):
        loops = LoopDetector(max_repeats=1)

        @baar_guard(router=make_router(), max_calls=5, loop_detector=loops)
        def t(x: int) -> int:
            return x

        t(1)
        with pytest.raises(LoopDetected):
            t(1)
        assert t._call_count == 1

    def test_without_a_detector_nothing_changes(self):
        @baar_guard(router=make_router())
        def t() -> str:
            return "same"

        for _ in range(20):
            t()
