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
from dataclasses import dataclass
from typing import Optional

import litellm

from baar.core.budget import BudgetExceeded, BudgetTracker

OUTPUT_OVERRUN_MODES = ("estimate", "preflight", "truncate")


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
    """
    completion_tokens: int
    max_tokens: Optional[int]
    budget_limited: bool = False
    prompt_tokens: int = 0


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


def uncut_output_tokens(
    model: str, prompt_tokens: int, expected_output_tokens: int, max_output_tokens: Optional[int]
) -> int:
    """Completion tokens for a response the budget does not cut off: the output limit, else the expected output."""
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
) -> OutputPlan:
    """
    Plan the output side of one call to ``model``.

    ``max_call_share`` (0–1] limits how much of ``remaining`` one "truncate" call
    may reserve, leaving room for concurrent calls near the cap. ``prompt_margin``
    is added to the prompt estimate when pricing input in "preflight" / "truncate";
    the context-window room is computed from the raw estimate.

    Raises:
        ValueError:             unknown mode.
        OutputLimitUnavailable: "preflight" with no known output limit, or the
                                prompt fills the context window.
        BudgetExceeded:         "truncate" and fewer than ``min_useful_tokens`` are affordable.
    """
    validate_mode(mode)
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
