"""
baar/core/caps.py — output-side budget planning.

Decides how many completion tokens a call is planned (and reserved) for, and
which ``max_tokens`` value — if any — is sent to the provider.

Modes (``on_output_overrun``):
  - "estimate"  — reserve the router's expected output. Never cuts responses
                  off; a long response can overshoot the budget.
  - "preflight" — reserve the worst-case output (user limit, else the model's
                  real max output). Never cuts responses off, never overshoots;
                  calls that cannot cover the worst case are downgraded or
                  blocked before any network request.
  - "truncate"  — opt-in. Derive ``max_tokens`` from the remaining budget, so
                  a long response is cut off instead of overshooting.
"""

import math
from dataclasses import dataclass, field, replace
from typing import Optional

import litellm

from baar.core.budget import BudgetExceeded, BudgetTracker

OUTPUT_OVERRUN_MODES = ("estimate", "preflight", "truncate")

# Anthropic's minimum thinking budget; also used as the floor for every provider.
MIN_THINKING_BUDGET = 1024


class OutputLimitUnavailable(ValueError):
    """
    No usable output limit for a model: LiteLLM doesn't know its max output
    (and no max_output_tokens was given), or the prompt already fills its
    context window. The router downgrades the big model on this; for the
    small model it is raised to the caller before any network call.
    """


@dataclass(frozen=True)
class OutputPlan:
    """
    completion_tokens: tokens used for affordability checks and reservation.
    max_tokens:        value sent to the provider, or None to send nothing.
    budget_limited:    True when max_tokens was lowered by the remaining budget
                       (only in "truncate" mode) — a "length" finish then means
                       Baar cut the response off.
    prompt_tokens:     prompt tokens the input is priced at: the estimate plus
                       the safety margin in "preflight" / "truncate".
    params:            extra provider parameters for the call, e.g. a thinking
                       budget or a reasoning effort.
    """
    completion_tokens: int
    max_tokens: Optional[int]
    budget_limited: bool = False
    prompt_tokens: int = 0
    params: dict = field(default_factory=dict)


def validate_mode(mode: str) -> None:
    if mode not in OUTPUT_OVERRUN_MODES:
        raise ValueError(
            f"on_output_overrun must be one of {list(OUTPUT_OVERRUN_MODES)}, got '{mode}'"
        )


def model_max_output_tokens(model: str) -> Optional[int]:
    """The model's real maximum output tokens from LiteLLM's model info, or None if unknown."""
    try:
        info = litellm.get_model_info(model)
    except Exception:
        return None
    value = info.get("max_output_tokens") or info.get("max_tokens")
    return int(value) if value else None


def model_context_window(model: str) -> Optional[int]:
    """The model's context window (max input tokens) from LiteLLM's model info, or None if unknown."""
    try:
        info = litellm.get_model_info(model)
    except Exception:
        return None
    value = info.get("max_input_tokens")
    return int(value) if value else None


def supports_stream_usage(model: str) -> bool:
    """True if LiteLLM accepts stream_options for ``model`` (so the stream can report real usage)."""
    try:
        params = litellm.get_supported_openai_params(model=model) or []
    except Exception:
        return False
    return "stream_options" in params


def thinking_style(model: str) -> Optional[str]:
    """
    How ``model`` takes a reasoning limit through LiteLLM: "budget" (a token
    budget — Anthropic, Gemini, DeepSeek), "effort" (a low/medium/high level —
    OpenAI o-series, GPT-5), or None (not a reasoning model, or unknown).
    """
    try:
        params = litellm.get_supported_openai_params(model=model) or []
        reasoning = bool(litellm.supports_reasoning(model=model))
    except Exception:
        return None
    if not reasoning:
        return None
    if "thinking" in params:
        return "budget"
    if "reasoning_effort" in params:
        return "effort"
    return None


def reasoning_effort_for(thinking_budget: int) -> str:
    """The effort level closest to a token budget, for providers without token budgets."""
    if thinking_budget < 4096:
        return "low"
    if thinking_budget < 16384:
        return "medium"
    return "high"


def priced_prompt_tokens(mode: str, prompt_tokens: int, prompt_margin: float) -> int:
    """
    Prompt tokens to price the input at. Baar's count is an estimate (tiktoken
    for every provider; images and tool schemas are approximate), so the modes
    that promise no overshoot add ``prompt_margin`` on top. "estimate" doesn't.
    """
    if mode == "estimate" or prompt_margin <= 0:
        return prompt_tokens
    return math.ceil(prompt_tokens * (1 + prompt_margin))


def _clamped_limit(model: str, max_output_tokens: Optional[int]) -> Optional[int]:
    """The user's limit, never above the model's real max output; the model max if no user limit."""
    model_max = model_max_output_tokens(model)
    if max_output_tokens is None:
        return model_max
    return max_output_tokens if model_max is None else min(max_output_tokens, model_max)


def output_limit(model: str, prompt_tokens: int, max_output_tokens: Optional[int]) -> Optional[int]:
    """
    Largest response that can be requested: the user's limit (never above the
    model's max output), else the model's max output, further capped by the room
    left in the context window. None if no limit is known.

    Raises:
        OutputLimitUnavailable: the prompt already fills the context window.
    """
    limit = _clamped_limit(model, max_output_tokens)
    window = model_context_window(model)
    if window is not None:
        room = window - prompt_tokens
        if room <= 0:
            raise OutputLimitUnavailable(
                f"Prompt ({prompt_tokens} tokens) fills the {window}-token context window "
                f"of model '{model}'; there is no room for a response."
            )
        limit = room if limit is None else min(limit, room)
    return limit


def _answer_limit_with_thinking(
    model: str, prompt_tokens: int, max_output_tokens: Optional[int], thinking_budget: int
) -> int:
    """
    The visible answer's limit when ``thinking_budget`` tokens of thinking share
    the output: the user's limit (capped by the model), but never more than what
    the model's max output leaves after the thinking.
    """
    total = output_limit(model, prompt_tokens, None)
    if total is None:
        raise OutputLimitUnavailable(
            f"thinking_budget needs the model's max output tokens, but LiteLLM has none for '{model}'."
        )
    if thinking_budget >= total:
        raise OutputLimitUnavailable(
            f"thinking_budget={thinking_budget} leaves no room for an answer within the "
            f"{total} output tokens available on '{model}'."
        )
    answer = output_limit(model, prompt_tokens, max_output_tokens)
    return min(answer if answer is not None else total, total - thinking_budget)


def uncut_output_tokens(
    model: str,
    prompt_tokens: int,
    expected_output_tokens: int,
    max_output_tokens: Optional[int],
    thinking_budget: Optional[int] = None,
) -> int:
    """Completion tokens for a response the budget does not cut off: the output limit, else the expected output."""
    if thinking_budget and thinking_style(model) == "budget":
        return _answer_limit_with_thinking(model, prompt_tokens, max_output_tokens, thinking_budget) + thinking_budget
    limit = output_limit(model, prompt_tokens, max_output_tokens)
    return limit if limit is not None else expected_output_tokens


def plan_output(
    mode: str,
    *,
    model: str,
    prompt_tokens: int,
    expected_output_tokens: int,
    remaining: float,
    tracker: BudgetTracker,
    max_output_tokens: Optional[int] = None,
    min_useful_tokens: int = 64,
    max_call_share: float = 1.0,
    prompt_margin: float = 0.0,
    thinking_budget: Optional[int] = None,
) -> OutputPlan:
    """
    Plan the output side of one call to ``model``.

    ``max_call_share`` (0–1] limits how much of ``remaining`` one "truncate" call
    may reserve, leaving room for concurrent calls near the cap. ``prompt_margin``
    is added to the prompt estimate when pricing input in "preflight" / "truncate";
    the context-window room is computed from the raw estimate.

    ``thinking_budget`` caps hidden reasoning. On token-budget providers the
    visible answer keeps its own limit (``max_output_tokens``) and the thinking
    is added on top; on effort-level providers it becomes a reasoning effort and
    the plan is otherwise unchanged; other models ignore it.

    Raises:
        ValueError:             unknown mode.
        OutputLimitUnavailable: "preflight" with no known output limit, or the
                                prompt fills the context window.
        BudgetExceeded:         "truncate" and fewer than ``min_useful_tokens`` are affordable.
    """
    validate_mode(mode)
    style = thinking_style(model) if thinking_budget else None
    if style == "budget":
        return _plan_with_thinking(
            mode, model=model, prompt_tokens=prompt_tokens, expected_output_tokens=expected_output_tokens,
            remaining=remaining, tracker=tracker, max_output_tokens=max_output_tokens,
            min_useful_tokens=min_useful_tokens, max_call_share=max_call_share,
            prompt_margin=prompt_margin, thinking_budget=thinking_budget,
        )
    plan = _plan_answer_only(
        mode, model=model, prompt_tokens=prompt_tokens, expected_output_tokens=expected_output_tokens,
        remaining=remaining, tracker=tracker, max_output_tokens=max_output_tokens,
        min_useful_tokens=min_useful_tokens, max_call_share=max_call_share, prompt_margin=prompt_margin,
    )
    if style == "effort":
        return replace(plan, params={"reasoning_effort": reasoning_effort_for(thinking_budget)})
    return plan


def _plan_with_thinking(
    mode: str,
    *,
    model: str,
    prompt_tokens: int,
    expected_output_tokens: int,
    remaining: float,
    tracker: BudgetTracker,
    max_output_tokens: Optional[int],
    min_useful_tokens: int,
    max_call_share: float,
    prompt_margin: float,
    thinking_budget: int,
) -> OutputPlan:
    """plan_output for token-budget providers: max_tokens covers thinking + answer."""
    priced = priced_prompt_tokens(mode, prompt_tokens, prompt_margin)
    params = {"thinking": {"type": "enabled", "budget_tokens": thinking_budget}}
    answer = _answer_limit_with_thinking(model, prompt_tokens, max_output_tokens, thinking_budget)
    full = answer + thinking_budget

    if mode == "estimate":
        return OutputPlan(min(expected_output_tokens, answer) + thinking_budget, full,
                          prompt_tokens=priced, params=params)
    if mode == "preflight":
        return OutputPlan(full, full, prompt_tokens=priced, params=params)

    # mode == "truncate": the thinking budget is kept; only the answer is shortened.
    price = tracker.output_price_per_token(model, priced)
    if not price:
        return OutputPlan(full, full, prompt_tokens=priced, params=params)
    allowance = remaining * max_call_share
    input_cost = tracker.estimate_cost(model, priced, 0)
    affordable = math.floor((allowance - input_cost) / price) if allowance > input_cost else 0
    if affordable - thinking_budget < min_useful_tokens:
        raise BudgetExceeded(
            requested=input_cost + (thinking_budget + min_useful_tokens) * price,
            remaining=allowance,
            model=model,
        )
    if full <= affordable:
        return OutputPlan(full, full, prompt_tokens=priced, params=params)
    return OutputPlan(affordable, affordable, budget_limited=True, prompt_tokens=priced, params=params)


def _plan_answer_only(
    mode: str,
    *,
    model: str,
    prompt_tokens: int,
    expected_output_tokens: int,
    remaining: float,
    tracker: BudgetTracker,
    max_output_tokens: Optional[int],
    min_useful_tokens: int,
    max_call_share: float,
    prompt_margin: float,
) -> OutputPlan:
    priced = priced_prompt_tokens(mode, prompt_tokens, prompt_margin)

    if mode == "estimate":
        if max_output_tokens is None:
            return OutputPlan(expected_output_tokens, None, prompt_tokens=priced)
        sent = _clamped_limit(model, max_output_tokens)
        return OutputPlan(min(expected_output_tokens, sent), sent, prompt_tokens=priced)

    if mode == "preflight":
        # Only the model's own max output counts as a worst case here — a context
        # window alone (when max output is unknown) is not a meaningful bound.
        if max_output_tokens is None and model_max_output_tokens(model) is None:
            raise OutputLimitUnavailable(
                f"on_output_overrun='preflight' needs an output limit, but LiteLLM has no "
                f"max output tokens for model '{model}'. Pass max_output_tokens=..."
            )
        limit = output_limit(model, prompt_tokens, max_output_tokens)
        return OutputPlan(limit, limit, prompt_tokens=priced)

    limit = output_limit(model, prompt_tokens, max_output_tokens)

    # mode == "truncate"
    price = tracker.output_price_per_token(model, priced)
    if not price:
        # No output pricing: the budget cannot bound the length, so fall back
        # to the plain limit (or the expected output if there is none).
        tokens = limit if limit is not None else expected_output_tokens
        return OutputPlan(tokens, tokens, prompt_tokens=priced)

    allowance = remaining * max_call_share
    input_cost = tracker.estimate_cost(model, priced, 0)
    affordable = math.floor((allowance - input_cost) / price) if allowance > input_cost else 0
    if affordable < min_useful_tokens:
        raise BudgetExceeded(
            requested=input_cost + min_useful_tokens * price,
            remaining=allowance,
            model=model,
        )
    if limit is not None and limit <= affordable:
        return OutputPlan(limit, limit, prompt_tokens=priced)
    return OutputPlan(affordable, affordable, budget_limited=True, prompt_tokens=priced)
