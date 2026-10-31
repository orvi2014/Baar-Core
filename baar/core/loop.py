"""
baar/core/loop.py — Loop detection for agent runs.

A loop is an agent repeating a call that already returned the same thing
(DeLoop, arXiv 2610.19376). Each step is reduced to a signature:

    (tool name, normalized arguments) → hash of the result, or of the error

and three patterns are watched across the run (default thresholds: 2 repeats,
3 errors, 3 alternating cycles — the repeat limit is lower than OpenHands'
StuckDetector because the agent is warned first):

  repeat     the same call returned the same result `max_repeats` times
  error      the same call failed the same way `max_error_repeats` times in a
             row (a success in between resets it: flaky is not stuck)
  ping_pong  two steps alternating for `max_ping_pong` full cycles

Detection is pre-call: once a pattern is complete, the next identical call is
refused with LoopDetected *before it runs*, so the call that would have been
wasted is never paid for. On the step after which the next identical one would
be refused (the first repeat, by default), `on_loop` gets a warning, and
baar_guard adds a note to a string tool result so the model can change course.
In a live benchmark, telling the model mattered most: it changed course in 19 of
20 runs, and stopping without telling it did worse than no detection at all.

Including the result is what keeps healthy repetition apart from a loop: the
same status check returning "running" then "done", or the same search with a
new cursor, never builds up a count. Short results saying the work isn't done
yet ("running", "pending", "queued", …) never count, so waiting on a job is not
a loop (`in_progress`). Other tools that legitimately return the same thing many
times belong in `exempt_tools`.

Usage::

    from baar import BAARRouter, LoopDetector
    from baar.integrations.tools import baar_guard

    router = BAARRouter(budget=0.50)
    loops = LoopDetector(on_loop=lambda e: print("loop warning:", e))

    @baar_guard(router=router, loop_detector=loops)
    def search_web(query: str) -> str: ...

Or drive it directly from any framework::

    loops.before("search_web", {"query": q})        # raises LoopDetected
    loops.after("search_web", {"query": q}, result=r)  # or error=exc
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Pattern, Union


class LoopDetected(RuntimeError):
    """
    Raised before a call that would continue a detected loop. Nothing was run.

    `pattern` is 'repeat', 'error' or 'ping_pong'; `count` is how many times
    the pattern already happened; `tool_name` is the refused call's tool.
    """

    def __init__(self, event: "LoopEvent"):
        self.event = event
        self.pattern = event.pattern
        self.tool_name = event.tool_name
        self.count = event.count
        super().__init__(
            f"Loop detected ({event.pattern}): '{event.tool_name}' would repeat a call that "
            f"already {_describe(event)}. The call was not made."
        )


def _describe(event: "LoopEvent") -> str:
    if event.pattern == "error":
        return f"failed the same way {event.count} times"
    if event.pattern == "ping_pong":
        return f"alternated with another step for {event.count} cycles"
    return f"returned the same result {event.count} times"


@dataclass(frozen=True)
class LoopEvent:
    """What on_loop receives: a warning, or the stop that is about to be raised."""

    action: str  # 'warn' | 'stop'
    pattern: str  # 'repeat' | 'error' | 'ping_pong'
    tool_name: str
    count: int
    step: int  # 0-based index of the step this event is about


@dataclass
class _Step:
    call: str  # hash of (tool, normalized args)
    outcome: str  # hash of the result, or "error:" + hash of the error
    tool_name: str = field(compare=False)


def _normalize(value: Any, ignore: frozenset[str]) -> Any:
    if isinstance(value, dict):
        return {
            str(k): _normalize(v, ignore)
            for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
            if str(k) not in ignore
        }
    if isinstance(value, (list, tuple)):
        return [_normalize(v, ignore) for v in value]
    if isinstance(value, str):
        return " ".join(value.split())
    return value


def _parse_args(arguments: Any) -> Any:
    """Tool-call arguments often arrive as a JSON string; compare them as values."""
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except ValueError:
            return arguments
    return arguments


def _digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, default=repr, separators=(",", ":"))
    return hashlib.blake2b(raw.encode(), digest_size=12).hexdigest()


# Default for `in_progress`: a status that says the work isn't finished yet.
IN_PROGRESS = re.compile(
    r"\b(running|pending|queued|in[ _-]?progress|processing|booting|starting|initializing|waiting|not[ _-]ready)\b",
    re.IGNORECASE,
)


class LoopDetector:
    """
    Watches one agent run for loops. Thread-safe; share one instance across
    every tool in the run, and create a new one (or call reset()) per run.

    Args:
        max_repeats:       identical (call, result) pairs allowed; the next is refused.
                           Default 2: the 3rd identical call is refused.
        max_error_repeats: identical (call, error) pairs allowed; the next is refused.
        max_ping_pong:     full A→B cycles allowed before the next step is refused.
        warn:              fire an on_loop 'warn' event on the step after which
                           the next identical one is refused.
        exempt_tools:      tool names never counted (e.g. status polls).
        in_progress:       results that mean "not done yet" never count, so
                           waiting on a job is not a loop. Default: a short
                           result (<= 200 characters as JSON) saying running,
                           pending, queued, in progress, processing, booting,
                           starting, initializing, waiting or not ready. A
                           compiled pattern changes it; False counts them.
        ignore_args:       argument names dropped before comparing (timestamps,
                           request ids), at any depth.
        normalize_result:  map a result before it is hashed, e.g. to drop a
                           timestamp it carries. Default: the result itself.
        on_loop:           callback for every LoopEvent; exceptions in it are
                           swallowed so a broken callback can't break the run.
    """

    def __init__(
        self,
        *,
        max_repeats: int = 2,
        max_error_repeats: int = 3,
        max_ping_pong: int = 3,
        warn: bool = True,
        exempt_tools: Iterable[str] = (),
        in_progress: Union[Pattern[str], bool, None] = None,
        ignore_args: Iterable[str] = (),
        normalize_result: Optional[Callable[[Any], Any]] = None,
        on_loop: Optional[Callable[[LoopEvent], None]] = None,
    ) -> None:
        for name, v in (
            ("max_repeats", max_repeats),
            ("max_error_repeats", max_error_repeats),
            ("max_ping_pong", max_ping_pong),
        ):
            if v < 1:
                raise ValueError(f"{name} must be at least 1")
        self.max_repeats = max_repeats
        self.max_error_repeats = max_error_repeats
        self.max_ping_pong = max_ping_pong
        self.warn = warn
        self.exempt_tools = frozenset(exempt_tools)
        self._in_progress = IN_PROGRESS if in_progress is None or in_progress is True else in_progress
        self._ignore = frozenset(ignore_args)
        self._normalize_result = normalize_result
        self._on_loop = on_loop
        self._lock = threading.Lock()
        self._steps: list[_Step] = []
        self._outcomes: Counter[tuple[str, str]] = Counter()
        self._last_outcome: dict[str, str] = {}
        self._error_streak: dict[str, int] = {}
        self._warned: set[Any] = set()
        self.events: list[LoopEvent] = []

    # ── public API ────────────────────────────────────────────────────────────

    def peek(self, tool_name: str, arguments: Any = None) -> Optional[LoopEvent]:
        """The stop before() would raise for this call, without raising or reporting it."""
        if tool_name in self.exempt_tools:
            return None
        call = self._call_key(tool_name, arguments)
        with self._lock:
            return self._stop_for(call, tool_name)

    def before(self, tool_name: str, arguments: Any = None) -> None:
        """
        Call before running a tool. Raises LoopDetected (after an on_loop
        'stop' event) when the call would continue a detected loop.
        """
        if tool_name in self.exempt_tools:
            return
        call = self._call_key(tool_name, arguments)
        with self._lock:
            event = self._stop_for(call, tool_name)
            if event is not None:
                self.events.append(event)
        if event is not None:
            self._emit(event)
            raise LoopDetected(event)

    def after(
        self,
        tool_name: str,
        arguments: Any = None,
        *,
        result: Any = None,
        error: Optional[BaseException] = None,
    ) -> Optional[LoopEvent]:
        """
        Call after a tool returned (`result`) or raised (`error`). Returns the
        warning it emitted, if any.
        """
        if tool_name in self.exempt_tools or (error is None and self._is_in_progress(result)):
            return None
        call = self._call_key(tool_name, arguments)
        if error is not None:
            outcome = "error:" + _digest([type(error).__name__, " ".join(str(error).split())])
        else:
            value = self._normalize_result(result) if self._normalize_result else result
            outcome = _digest(_normalize(value, frozenset()))
        with self._lock:
            self._steps.append(_Step(call, outcome, tool_name))
            self._outcomes[(call, outcome)] += 1
            if outcome.startswith("error:") and self._last_outcome.get(call) == outcome:
                self._error_streak[call] = self._error_streak.get(call, 0) + 1
            else:
                self._error_streak[call] = 1 if outcome.startswith("error:") else 0
            self._last_outcome[call] = outcome
            event = self._warning_for(call, outcome, tool_name) if self.warn else None
            if event is not None:
                self.events.append(event)
        if event is not None:
            self._emit(event)
        return event

    def tighten(self) -> None:
        """
        Lower every threshold by one (never below 1). Call after compressing or
        summarizing the agent's history: DeLoop finds that roughly doubles looping.
        """
        with self._lock:
            self.max_repeats = max(1, self.max_repeats - 1)
            self.max_error_repeats = max(1, self.max_error_repeats - 1)
            self.max_ping_pong = max(1, self.max_ping_pong - 1)

    def reset(self) -> None:
        """Forget the run (thresholds are kept)."""
        with self._lock:
            self._steps.clear()
            self._outcomes.clear()
            self._last_outcome.clear()
            self._error_streak.clear()
            self._warned.clear()
            self.events.clear()

    @property
    def steps(self) -> int:
        return len(self._steps)

    # ── internals ─────────────────────────────────────────────────────────────

    def _is_in_progress(self, result: Any) -> bool:
        if not self._in_progress:
            return False
        if isinstance(result, str):
            text = result
        else:
            try:
                text = json.dumps(result, default=str)
            except Exception:
                return False
        return len(text) <= 200 and bool(self._in_progress.search(text))

    def _call_key(self, tool_name: str, arguments: Any) -> str:
        return _digest([tool_name, _normalize(_parse_args(arguments), self._ignore)])

    def _limit(self, outcome: str) -> tuple[str, int]:
        if outcome.startswith("error:"):
            return "error", self.max_error_repeats
        return "repeat", self.max_repeats

    def _count(self, call: str, outcome: str) -> int:
        """Errors count in a row; identical results count across the run."""
        if outcome.startswith("error:"):
            return self._error_streak.get(call, 0)
        return self._outcomes[(call, outcome)]

    def _stop_for(self, call: str, tool_name: str) -> Optional[LoopEvent]:
        step = len(self._steps)
        last = self._last_outcome.get(call)
        if last is not None:
            pattern, limit = self._limit(last)
            count = self._count(call, last)
            if count >= limit:
                return LoopEvent("stop", pattern, tool_name, count, step)
        cycles = self._ping_pong_cycles(call)
        if cycles >= self.max_ping_pong:
            return LoopEvent("stop", "ping_pong", tool_name, cycles, step)
        return None

    def _warning_for(self, call: str, outcome: str, tool_name: str) -> Optional[LoopEvent]:
        step = len(self._steps) - 1
        pattern, limit = self._limit(outcome)
        count = self._count(call, outcome)
        # When the next identical step is refused, and only once something
        # actually repeated (a first error is not a warning sign).
        if count >= 2 and count == limit and (pattern, call, outcome) not in self._warned:
            self._warned.add((pattern, call, outcome))
            return LoopEvent("warn", pattern, tool_name, count, step)
        nxt = self._ping_pong_peek()
        if nxt is not None:
            cycles = self._ping_pong_cycles(nxt)
            pair = ("ping_pong", frozenset(
                ((self._steps[-1].call, self._steps[-1].outcome), (self._steps[-2].call, self._steps[-2].outcome))
            ))
            if cycles >= 1 and cycles == self.max_ping_pong and pair not in self._warned:
                self._warned.add(pair)
                return LoopEvent("warn", "ping_pong", tool_name, cycles, step)
        return None

    def _ping_pong_peek(self) -> Optional[str]:
        """The call that would continue an A,B,A,B tail, or None."""
        s = self._steps
        if len(s) < 2:
            return None
        a, b = s[-2], s[-1]
        if (a.call, a.outcome) == (b.call, b.outcome):
            return None
        return a.call

    def _ping_pong_cycles(self, next_call: str) -> int:
        """
        Full A→B cycles at the end of the run, counted only when `next_call`
        is A (the step that would continue the alternation).
        """
        s = self._steps
        if len(s) < 2:
            return 0
        a, b = (s[-2].call, s[-2].outcome), (s[-1].call, s[-1].outcome)
        if a == b or next_call != a[0]:
            return 0
        n = 0
        i = len(s) - 1
        expect = b
        while i >= 0 and (s[i].call, s[i].outcome) == expect:
            n += 1
            expect = a if expect == b else b
            i -= 1
        return n // 2

    def _emit(self, event: LoopEvent) -> None:
        if self._on_loop is None:
            return
        try:
            self._on_loop(event)
        except Exception:
            pass


def loop_nudge(event: LoopEvent) -> str:
    """The note for the model when its next step would be refused."""
    if event.pattern == "error":
        did = f"has failed the same way {event.count} times in a row"
    elif event.pattern == "ping_pong":
        did = f"and another step have alternated with the same results for {event.count} cycles"
    else:
        did = f"has returned the same result {event.count} times for the same arguments"
    return (
        f"[baar] The tool '{event.tool_name}' {did}. Calling it again the same way will be refused "
        "and this run stopped. Change the arguments, use a different tool, or answer with what you already have."
    )
