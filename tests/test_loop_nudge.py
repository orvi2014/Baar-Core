"""
tests/test_loop_nudge.py — loop detection catches up with noburn SDK 0.6.1.

  1. Earlier stop: the 3rd identical call is refused (max_repeats 4 -> 2).
  2. The model is told: the warning comes on the first repeat, and baar_guard
     adds a note to a string tool result so the model sees it in its next turn.
  3. Waiting is not looping: short "running / pending / queued …" results
     don't count, so polling a job isn't stopped.
"""

import pytest

from baar import BAARRouter, LoopDetected, LoopDetector, loop_nudge
from baar.integrations.tools import baar_guard
from baar.core.loop import IN_PROGRESS, LoopEvent


def run(d, tool, args, result=None, error=None):
    """One step: before() then after(). Returns the warning, if any."""
    d.before(tool, args)
    if error is not None:
        return d.after(tool, args, error=error)
    return d.after(tool, args, result=result)


def make_router(budget=1.0):
    return BAARRouter(budget=budget, use_llm_router=False)


# ── 1. Earlier stop ───────────────────────────────────────────────────────────

class TestDefaults:
    def test_defaults_refuse_the_third_identical_call(self):
        d = LoopDetector()
        run(d, "search", {"q": "x"}, result="none")
        run(d, "search", {"q": "x"}, result="none")
        with pytest.raises(LoopDetected):
            d.before("search", {"q": "x"})

    def test_error_default_unchanged(self):
        d = LoopDetector()
        for _ in range(3):
            run(d, "api", {}, error=RuntimeError("500"))
        with pytest.raises(LoopDetected) as exc:
            d.before("api", {})
        assert exc.value.pattern == "error"

    def test_default_values(self):
        d = LoopDetector()
        assert (d.max_repeats, d.max_error_repeats, d.max_ping_pong) == (2, 3, 3)

    def test_explicit_max_repeats_still_respected(self):
        d = LoopDetector(max_repeats=4)
        for _ in range(4):
            run(d, "s", {}, result="r")
        with pytest.raises(LoopDetected):
            d.before("s", {})


# ── 2. The warning comes when the next identical step would be refused ────────

class TestWarningTiming:
    def test_warns_on_the_first_repeat_by_default(self):
        seen = []
        d = LoopDetector(on_loop=lambda e: seen.append(e.action))
        assert run(d, "s", {}, result="r") is None
        warn = run(d, "s", {}, result="r")
        assert (warn.action, warn.pattern, warn.count) == ("warn", "repeat", 2)
        with pytest.raises(LoopDetected):
            d.before("s", {})
        assert seen == ["warn", "stop"]

    def test_warns_on_the_step_after_which_the_next_is_refused(self):
        d = LoopDetector(max_repeats=3)
        assert run(d, "s", {}, result="r") is None
        assert run(d, "s", {}, result="r") is None
        assert run(d, "s", {}, result="r").action == "warn"
        with pytest.raises(LoopDetected):
            d.before("s", {})

    def test_error_warning_when_next_failure_would_be_refused(self):
        d = LoopDetector()
        assert run(d, "api", {}, error=RuntimeError("500")) is None
        assert run(d, "api", {}, error=RuntimeError("500")) is None
        warn = run(d, "api", {}, error=RuntimeError("500"))
        assert (warn.action, warn.pattern, warn.count) == ("warn", "error", 3)

    def test_ping_pong_warning_when_next_step_would_be_refused(self):
        d = LoopDetector(max_ping_pong=2, max_repeats=10)
        events = []
        for _ in range(2):
            events.append(run(d, "a", {}, result="1"))
            events.append(run(d, "b", {}, result="2"))
        warns = [e for e in events if e is not None]
        assert [(e.action, e.pattern) for e in warns] == [("warn", "ping_pong")]
        assert events[-1] is warns[0]
        with pytest.raises(LoopDetected):
            d.before("a", {})

    def test_warns_only_once_per_loop(self):
        seen = []
        d = LoopDetector(max_repeats=2, on_loop=seen.append)
        run(d, "s", {}, result="r")
        run(d, "s", {}, result="r")
        run(d, "t", {}, result="other")
        assert [e.action for e in seen] == ["warn"]


# ── 3. Waiting is not looping ─────────────────────────────────────────────────

class TestInProgress:
    @pytest.mark.parametrize("result", [
        "booting…", "running", {"status": "running"}, "Job is still in progress",
        "queued", "PENDING", {"state": "processing"}, "starting up", "initializing",
        "waiting for worker", "not ready", "not_ready", "in-progress",
    ])
    def test_in_progress_results_never_count(self, result):
        d = LoopDetector()
        for _ in range(5):
            assert run(d, "job_status", {"id": 1}, result=result) is None
        d.before("job_status", {"id": 1})  # not refused

    def test_progress_to_done_is_fine(self):
        d = LoopDetector()
        for r in ["booting…", {"status": "running"}, {"status": "running"}, "Job is still in progress"]:
            assert run(d, "job_status", {"id": 1}, result=r) is None
        d.before("job_status", {"id": 1})

    def test_in_progress_false_counts_them(self):
        d = LoopDetector(in_progress=False)
        run(d, "job_status", {"id": 1}, result={"status": "running"})
        run(d, "job_status", {"id": 1}, result={"status": "running"})
        with pytest.raises(LoopDetected):
            d.before("job_status", {"id": 1})

    def test_custom_pattern(self):
        import re
        d = LoopDetector(in_progress=re.compile(r"\bbaking\b"))
        for _ in range(4):
            run(d, "oven", {}, result="still baking")
        d.before("oven", {})
        # The default words no longer apply with a custom pattern.
        run(d, "job", {}, result="running")
        run(d, "job", {}, result="running")
        with pytest.raises(LoopDetected):
            d.before("job", {})

    def test_long_results_still_count(self):
        # A big page that happens to mention "running" is a real result, not a status.
        page = "Results: " + "running shoes, " * 30
        assert len(page) > 200
        d = LoopDetector()
        run(d, "search", {"q": "shoes"}, result=page)
        run(d, "search", {"q": "shoes"}, result=page)
        with pytest.raises(LoopDetected):
            d.before("search", {"q": "shoes"})

    def test_word_boundaries(self):
        # "overrunning" and "unqueued" aren't statuses.
        assert not IN_PROGRESS.search("overrunning")
        assert not IN_PROGRESS.search("unqueuedish")
        d = LoopDetector()
        run(d, "s", {}, result="overrunning")
        run(d, "s", {}, result="overrunning")
        with pytest.raises(LoopDetected):
            d.before("s", {})

    def test_errors_mentioning_pending_still_count(self):
        d = LoopDetector()
        for _ in range(3):
            run(d, "api", {}, error=RuntimeError("request pending timeout"))
        with pytest.raises(LoopDetected):
            d.before("api", {})

    def test_unserializable_result_counts_normally(self):
        class Opaque:
            def __repr__(self):
                return "<opaque>"
        d = LoopDetector()
        o = Opaque()
        run(d, "s", {}, result=o)
        run(d, "s", {}, result=o)
        with pytest.raises(LoopDetected):
            d.before("s", {})


# ── Parity helpers ported from noburn ─────────────────────────────────────────

class TestParity:
    def test_json_string_arguments_compare_as_values(self):
        d = LoopDetector()
        run(d, "search", '{"q": "x", "n": 1}', result="none")
        run(d, "search", {"n": 1, "q": "x"}, result="none")
        with pytest.raises(LoopDetected):
            d.before("search", '{"q":"x","n":1}')

    def test_non_json_string_arguments_compare_as_strings(self):
        d = LoopDetector()
        run(d, "shell", "ls -la", result="same")
        run(d, "shell", "ls  -la", result="same")  # whitespace is normalized
        with pytest.raises(LoopDetected):
            d.before("shell", "ls -la")
        d.before("shell", "ls -l")  # a different command is fine

    def test_peek_ignores_exempt_tools(self):
        d = LoopDetector(exempt_tools=["poll"])
        assert d.peek("poll", {}) is None

    def test_result_that_cannot_be_serialized_is_not_in_progress(self):
        circular = {}
        circular["self"] = circular
        assert LoopDetector()._is_in_progress(circular) is False

    def test_peek_reports_without_raising_or_firing(self):
        seen = []
        d = LoopDetector(on_loop=seen.append)
        run(d, "s", {}, result="r")
        assert d.peek("s", {}) is None
        run(d, "s", {}, result="r")
        seen.clear()
        event = d.peek("s", {})
        assert (event.action, event.pattern) == ("stop", "repeat")
        assert seen == []
        assert d.peek("s", {"other": 1}) is None


# ── loop_nudge text ───────────────────────────────────────────────────────────

class TestLoopNudge:
    @pytest.mark.parametrize("pattern,phrase", [
        ("repeat", "same result 2 times"),
        ("error", "failed the same way 2 times"),
        ("ping_pong", "alternated"),
    ])
    def test_text(self, pattern, phrase):
        note = loop_nudge(LoopEvent("warn", pattern, "search", 2, 1))
        assert note.startswith("[baar]")
        assert "'search'" in note and phrase in note
        assert "refused" in note
        assert "different" in note  # tells the model what to do instead


# ── baar_guard tells the model through the tool result ────────────────────────

class TestGuardNudge:
    def test_string_result_gets_the_note_on_the_first_repeat(self):
        loops = LoopDetector()
        router = make_router()

        @baar_guard(router=router, loop_detector=loops)
        def search(q):
            return "No results"

        assert search("Acme revenue") == "No results"
        second = search("Acme revenue")
        assert second.startswith("No results")
        assert "[baar]" in second and "'search'" in second
        with pytest.raises(LoopDetected):
            search("Acme revenue")

    def test_note_only_once(self):
        loops = LoopDetector(max_repeats=3)
        router = make_router()

        @baar_guard(router=router, loop_detector=loops)
        def search(q):
            return "No results"

        outs = [search("x") for _ in range(3)]
        assert ["[baar]" in o for o in outs] == [False, False, True]

    def test_non_string_result_is_returned_unchanged(self):
        seen = []
        loops_cb = LoopDetector(on_loop=seen.append)
        router = make_router()

        @baar_guard(router=router, loop_detector=loops_cb)
        def lookup(k):
            return {"value": None}

        assert lookup("a") == {"value": None}
        assert lookup("a") == {"value": None}
        assert [e.action for e in seen] == ["warn"]  # still reported to on_loop

    def test_loop_nudge_false_leaves_results_alone(self):
        loops = LoopDetector()
        router = make_router()

        @baar_guard(router=router, loop_detector=loops, loop_nudge=False)
        def search(q):
            return "No results"

        assert search("x") == "No results"
        assert search("x") == "No results"
        with pytest.raises(LoopDetected):
            search("x")

    def test_changed_call_after_the_note_goes_through(self):
        loops = LoopDetector()
        router = make_router()

        @baar_guard(router=router, loop_detector=loops)
        def search(q):
            return "No results"

        search("x")
        search("x")
        assert search("y") == "No results"

    def test_polling_a_job_is_not_stopped_or_nudged(self):
        loops = LoopDetector()
        router = make_router()
        statuses = iter(["running"] * 6 + ["done"])

        @baar_guard(router=router, loop_detector=loops)
        def job_status(job_id):
            return next(statuses)

        outs = [job_status("j1") for _ in range(7)]
        assert outs == ["running"] * 6 + ["done"]

    @pytest.mark.asyncio
    async def test_async_tool_gets_the_note(self):
        loops = LoopDetector()
        router = make_router()

        @baar_guard(router=router, loop_detector=loops)
        async def search(q):
            return "No results"

        await search("x")
        assert "[baar]" in await search("x")
        with pytest.raises(LoopDetected):
            await search("x")

    def test_error_warning_does_not_change_the_exception(self):
        seen = []
        loops = LoopDetector(on_loop=seen.append)
        router = make_router()

        @baar_guard(router=router, loop_detector=loops)
        def deploy():
            raise ValueError("missing config")

        for _ in range(3):
            with pytest.raises(ValueError, match="^missing config$"):
                deploy()
        assert [e.action for e in seen] == ["warn"]
        with pytest.raises(LoopDetected):
            deploy()

    def test_refusal_message_says_the_call_was_not_run(self):
        loops = LoopDetector()
        router = make_router()

        @baar_guard(router=router, loop_detector=loops)
        def search(q):
            return "No results"

        search("x")
        search("x")
        with pytest.raises(LoopDetected, match="not made"):
            search("x")
