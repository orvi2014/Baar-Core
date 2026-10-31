"""
baar/core/awareness.py — letting the agent see its budget, and wrap up before it ends.

A kill-switch alone stops an agent mid-task with nothing to show for the money
spent. BATS (arXiv 2511.17006) finds that adding a short budget status block to
each step makes agents spend better, with no training. Two pieces:

  BudgetStatus  what's been spent and what's left, plus calls left at the
                current average cost — `router.budget_status()`, or added to
                each call's last user message with `budget_awareness=True`.
  WindDown      once the budget is nearly used, the next call is the final one:
                it carries an instruction to wrap up, and later calls raise
                BudgetExhausted with the final answer in `.partial_result`.

Notes go at the end of the last user message, not into the system prompt, so a
cached prompt prefix (system prompt and history) is left intact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

DEFAULT_WRAP_UP = (
    "[budget] This is your final call: the budget is nearly used. Wrap up now — give your "
    "best complete answer with what you already have, and don't start new work."
)


@dataclass(frozen=True)
class BudgetStatus:
    """A snapshot of a router's budget."""

    total: float
    spent: float
    remaining: float
    utilization: float
    calls: int  # completed model calls
    avg_call_cost: Optional[float]  # None until a call has completed
    calls_left: Optional[int]  # remaining / avg_call_cost, rounded down
    winding_down: bool  # the final call has been made or is being made

    def note(self) -> str:
        """The status line added to a call with budget_awareness=True."""
        text = f"[budget] ${self.spent:.4f} of ${self.total:.2f} used ({self.utilization:.0%})"
        if self.calls_left is not None:
            text += f" · ~{self.calls_left} calls left at the current rate"
        return text


@dataclass(frozen=True)
class WindDown:
    """
    Make one final, wrap-up call before the budget runs out.

    Args:
        at:          utilization (0–1] at which the next call becomes the final one.
        reserve:     also wind down once the remaining budget falls to this share
                     of the total. It triggers the final call; it does not fence
                     the money off from earlier calls (use "preflight" to make sure
                     the final call can't overshoot).
        instruction: text added to the final call's last user message.
    """

    at: float = 0.9
    reserve: float = 0.05
    instruction: str = DEFAULT_WRAP_UP

    def __post_init__(self) -> None:
        if not 0 < self.at <= 1:
            raise ValueError("WindDown.at must be in (0, 1]")
        if not 0 <= self.reserve < 1:
            raise ValueError("WindDown.reserve must be in [0, 1)")

    def due(self, status: BudgetStatus) -> bool:
        return status.utilization >= self.at or status.remaining <= self.reserve * status.total


def add_note(messages: List[dict], text: str) -> List[dict]:
    """
    A copy of ``messages`` with ``text`` appended to the last user message (as a
    text part for multi-part content), or as a new user message if there is none.
    """
    out = list(messages)
    for i in range(len(out) - 1, -1, -1):
        if out[i].get("role") != "user":
            continue
        message = dict(out[i])
        content: Any = message.get("content")
        if isinstance(content, list):
            message["content"] = list(content) + [{"type": "text", "text": text}]
        else:
            message["content"] = f"{content}\n\n{text}" if content else text
        out[i] = message
        return out
    out.append({"role": "user", "content": text})
    return out
