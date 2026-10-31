"""
baar/router.py — Main BAARRouter orchestration class.
Handles the end-to-end flow: routing, budget constraints, and execution.
"""

import asyncio
import dataclasses
import importlib.metadata
import json
import os
import random
import threading
import time
import warnings
import weakref
import litellm
from dataclasses import dataclass, replace
from typing import AsyncGenerator, AsyncIterator, Callable, Iterator, List, Optional, Any

from baar.core.router import Router, ModelTier
from baar.core import caps
from baar.core.lengths import OutputLengths
from baar.core.awareness import BudgetStatus, WindDown, add_note, DEFAULT_WRAP_UP
from baar.core.cache import cached_tokens
from baar.core.budget import BudgetTracker, BudgetExceeded, BudgetWindow, Alert
from baar.core.exceptions import TaskRejected, BudgetExhausted, PolicyViolation
from baar.core.models import StepResult, RoutingLog
from baar.core.policy import Policy, PolicyAction
from baar.core.stores import BudgetStore

# ── Module-level litellm config (runs once on import) ─────────────────────────
litellm.set_verbose = False
litellm.suppress_debug_info = True

# ── Module-level retryable exception tuple ─────────────────────────────────────
try:
    import litellm.exceptions as _litellm_exc
    _LITELLM_RETRYABLE: tuple = (
        _litellm_exc.RateLimitError,
        _litellm_exc.ServiceUnavailableError,
        _litellm_exc.Timeout,
        _litellm_exc.APIConnectionError,
    )
except (ImportError, AttributeError):  # pragma: no cover
    _LITELLM_RETRYABLE = ()


# ── Per-path telemetry write locks ────────────────────────────────────────────
# WeakValueDictionary so lock objects are GC'd when no router references them.
_telemetry_locks: "weakref.WeakValueDictionary[str, threading.Lock]" = weakref.WeakValueDictionary()
_telemetry_locks_guard = threading.Lock()


def _telemetry_lock(path: str) -> threading.Lock:
    with _telemetry_locks_guard:
        lock = _telemetry_locks.get(path)
        if lock is None:
            lock = threading.Lock()
            _telemetry_locks[path] = lock
        return lock


def _is_retryable(exc: Exception) -> bool:
    """
    Return True for transient provider errors that warrant retry + backoff
    before failing over to the next model candidate.
    Non-retryable errors (auth, bad request, model not found) fail immediately.
    """
    return bool(_LITELLM_RETRYABLE) and isinstance(exc, _LITELLM_RETRYABLE)


def token_counter(text: str = "", model: str = "gpt-4o", messages=None) -> int:
    """
    Estimate token count for a task or message list.
    Used for pre-flight budget checks (BCD).

    Args:
        text:     Plain text to count tokens for (used when messages is None).
        model:    LiteLLM model name for tokenizer selection.
        messages: Optional list of message dicts. When provided, counts tokens
                  for the entire message list instead of plain text.
    """
    try:
        if messages is not None:
            return litellm.token_counter(model=model, messages=messages)
        return litellm.token_counter(model=model, text=text)
    except Exception:
        if messages is not None:
            return sum(len(m.get("content", "")) // 4 for m in messages)
        return max(1, len(text) // 4)


# ── Version guard ──────────────────────────────────────────────────────────────

# Add to this frozenset as CVEs are disclosed; this is easier to audit than
# inline string comparisons scattered through the code.
_LITELLM_KNOWN_BAD_VERSIONS: frozenset[str] = frozenset({"1.82.7", "1.82.8"})


def _check_litellm_version() -> None:
    if os.environ.get("BAAR_SKIP_VERSION_CHECK"):
        return
    try:
        v = importlib.metadata.version("litellm")
    except importlib.metadata.PackageNotFoundError:
        return
    if v in _LITELLM_KNOWN_BAD_VERSIONS:
        raise RuntimeError(
            f"Insecure LiteLLM version detected: {v}. "
            "Upgrade to 1.82.9+ or downgrade below 1.82.7. "
            f"Known bad versions: {sorted(_LITELLM_KNOWN_BAD_VERSIONS)}"
        )


# ── Retry helpers ──────────────────────────────────────────────────────────────

def _sync_completion_with_retry(
    model: str,
    messages: list,
    max_retries: int = 2,
    stream: bool = False,
    timeout: Optional[float] = None,
    max_tokens: Optional[int] = None,
    stream_options: Optional[dict] = None,
    extra: Optional[dict] = None,
) -> object:
    """Synchronous LiteLLM completion with exponential backoff on transient errors."""
    delay = 1.0
    kwargs = dict(model=model, messages=messages, stream=stream)
    if timeout is not None:
        kwargs["timeout"] = timeout
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if stream_options is not None:
        kwargs["stream_options"] = stream_options
    if extra:
        kwargs.update(extra)
    for attempt in range(max_retries + 1):
        try:
            return litellm.completion(**kwargs)
        except Exception as exc:
            if attempt < max_retries and _is_retryable(exc):  # pragma: no cover
                time.sleep(delay * (1.0 + random.uniform(-0.1, 0.3)))  # pragma: no cover
                delay = min(delay * 2, 30.0)  # pragma: no cover
            else:
                raise


async def _async_completion_with_retry(
    model: str,
    messages: list,
    max_retries: int = 2,
    stream: bool = False,
    timeout: Optional[float] = None,
    max_tokens: Optional[int] = None,
    stream_options: Optional[dict] = None,
    extra: Optional[dict] = None,
) -> object:
    """Async LiteLLM completion with exponential backoff on transient errors."""
    delay = 1.0
    kwargs = dict(model=model, messages=messages, stream=stream)
    if timeout is not None:
        kwargs["timeout"] = timeout
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if stream_options is not None:
        kwargs["stream_options"] = stream_options
    if extra:
        kwargs.update(extra)
    for attempt in range(max_retries + 1):
        try:
            return await litellm.acompletion(**kwargs)
        except Exception as exc:
            if attempt < max_retries and _is_retryable(exc):  # pragma: no cover
                await asyncio.sleep(delay * (1.0 + random.uniform(-0.1, 0.3)))  # pragma: no cover
                delay = min(delay * 2, 30.0)  # pragma: no cover
            else:
                raise


def _stream_usage_options(model: str) -> Optional[dict]:
    """Ask the provider to report real usage at the end of a stream, where LiteLLM supports it."""
    return {"include_usage": True} if caps.supports_stream_usage(model) else None


def _reported_usage(chunk) -> Optional[tuple]:
    """
    (prompt_tokens, completion_tokens, cached_tokens) reported in a stream chunk, or None.

    Reads ``chunk.usage`` (sent with stream_options include_usage) and falls back
    to ``chunk._hidden_params["usage"]``, where LiteLLM puts the provider's real
    totals on the last chunk when stream_options isn't sent (e.g. Anthropic).
    """
    hidden = getattr(chunk, "_hidden_params", None)
    candidates = [getattr(chunk, "usage", None)]
    if isinstance(hidden, dict):
        candidates.append(hidden.get("usage"))
    for usage in candidates:
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
            # LiteLLM attaches (0, 0) to the last chunk when the provider sent no usage.
            if prompt_tokens == 0 and completion_tokens == 0:
                continue
            return prompt_tokens, completion_tokens, cached_tokens(usage)
    return None


_CONTINUE_PROMPT = "Continue exactly where you stopped. Don't repeat anything you already wrote."

# What ends a "continue" answer early: the text so far is returned instead.
_SEGMENT_STOPS = (BudgetExhausted, BudgetExceeded, TaskRejected, PolicyViolation)


def _flag_budget_cut(step: StepResult, reason: BaseException) -> None:
    """The answer stopped because the budget couldn't pay for another segment."""
    if isinstance(reason, (BudgetExhausted, BudgetExceeded)) and step.finish_reason == "length":
        step.truncated_by_budget = True


def _validate_thinking_budget(value: Optional[int]) -> None:
    if value is not None and value < caps.MIN_THINKING_BUDGET:
        raise ValueError(
            f"thinking_budget must be at least {caps.MIN_THINKING_BUDGET} tokens (provider minimum) or None"
        )


def _close_stream(stream) -> None:
    """Close a provider stream the caller stopped reading, so the request ends."""
    close = getattr(stream, "close", None)
    if close is None:
        return
    try:
        close()
    except Exception:
        pass


def _finish_reason(value) -> Optional[str]:
    return value if isinstance(value, str) else None


# ── BAARConfig ─────────────────────────────────────────────────────────────────

@dataclass
class BAARConfig:
    """
    Full configuration for BAARRouter.
    Use BAARRouter.from_config(BAARConfig(...)) for cleaner call sites.
    """
    budget: float = 0.10
    small_model: str = "gpt-4o-mini"
    big_model: str = "gpt-4o"
    complexity_threshold: float = 0.80
    use_llm_router: bool = True
    system_prompt: Optional[str] = None
    value_fn: Optional[Callable[[str], float]] = None
    async_value_fn: Optional[Callable] = None  # async def(task: str) -> float
    value_reject_alpha: float = 0.3
    max_reject_rate: float = 0.5
    min_samples_for_reject_clamp: int = 20
    small_fallback_models: Optional[List[str]] = None
    big_fallback_models: Optional[List[str]] = None
    telemetry_jsonl_path: Optional[str] = None
    routing_cache_size: int = 256
    routing_cache_enabled: bool = True
    small_exploration_rate: float = 0.0
    min_cost_threshold: float = 0.0001
    routing_task_char_limit: int = 500
    store: Optional[Any] = None  # Optional[BudgetStore]
    expected_output_tokens: int = 500
    routing_cache: Optional[dict] = None
    max_log_steps: Optional[int] = None
    domain_model_overrides: Optional[dict] = None  # {"code": "gpt-4o"} etc.
    call_timeout: Optional[float] = None
    routing_timeout: Optional[float] = None
    max_consecutive_errors: int = 1
    arun_concurrency: int = 1
    policy: Optional[Policy] = None
    window: Optional[BudgetWindow] = None
    alerts: Optional[List[Alert]] = None
    on_output_overrun: str = "estimate"  # "estimate" | "preflight" | "truncate" — see baar.core.caps
    max_output_tokens: Optional[int] = None
    min_useful_tokens: int = 64  # "truncate" blocks locally below this many affordable tokens
    max_call_share: float = 1.0  # "truncate": largest share of the remaining budget one call may reserve
    prompt_margin: float = 0.2  # "preflight"/"truncate": extra share of the prompt estimate priced in
    thinking_budget: Optional[int] = None  # cap on hidden reasoning tokens (reasoning models only)
    overrun_risk: float = 0.0  # "preflight": reserve the learned (1 - risk) output length, not the worst case
    output_lengths: Optional[Any] = None  # Optional[OutputLengths]; share to pool learned lengths
    segment_tokens: int = 2000  # "continue": output tokens per segment
    max_segments: int = 8  # "continue": most segments for one answer
    budget_awareness: Any = False  # True, or Callable[[BudgetStatus], str]: add a status line to each call
    wind_down: Optional[WindDown] = None  # make one final, wrap-up call before the budget runs out

    @classmethod
    def anthropic(cls, **kwargs: Any) -> "BAARConfig":
        """
        Preset for Anthropic Claude.

        Small: claude-haiku-4-5-20251001  (fast, cheap)
        Big:   claude-sonnet-4-6          (capable)

        Usage::

            router = BAARRouter.from_config(BAARConfig.anthropic(budget=0.10))
        """
        kwargs.setdefault("small_model", "claude-haiku-4-5-20251001")
        kwargs.setdefault("big_model", "claude-sonnet-4-6")
        return cls(**kwargs)

    @classmethod
    def openai(cls, **kwargs: Any) -> "BAARConfig":
        """
        Preset for OpenAI (identical to the default, provided for symmetry).

        Small: gpt-4o-mini  (fast, cheap)
        Big:   gpt-4o       (capable)

        Usage::

            router = BAARRouter.from_config(BAARConfig.openai(budget=0.10))
        """
        kwargs.setdefault("small_model", "gpt-4o-mini")
        kwargs.setdefault("big_model", "gpt-4o")
        return cls(**kwargs)


# ── BAARRouter ─────────────────────────────────────────────────────────────────

class BAARRouter:
    """
    Budget-Aware Agentic Router (BAAR).

    The main high-level API for the library. Wraps the semantic router
    and budget tracker into a single easy-to-use interface.
    """

    def __init__(
        self,
        budget: float = 0.10,
        small_model: str = "gpt-4o-mini",
        big_model: str = "gpt-4o",
        complexity_threshold: float = 0.80,
        use_llm_router: bool = True,
        system_prompt: Optional[str] = None,
        value_fn: Optional[Callable[[str], float]] = None,
        async_value_fn: Optional[Callable] = None,
        value_reject_alpha: float = 0.3,
        max_reject_rate: float = 0.5,
        min_samples_for_reject_clamp: int = 20,
        small_fallback_models: Optional[List[str]] = None,
        big_fallback_models: Optional[List[str]] = None,
        telemetry_jsonl_path: Optional[str] = None,
        routing_cache_size: int = 256,
        routing_cache_enabled: bool = True,
        small_exploration_rate: float = 0.0,
        min_cost_threshold: float = 0.0001,
        routing_task_char_limit: int = 500,
        store: Optional[BudgetStore] = None,
        expected_output_tokens: int = 500,
        routing_cache: Optional[dict] = None,
        max_log_steps: Optional[int] = None,
        domain_model_overrides: Optional[dict] = None,
        call_timeout: Optional[float] = None,
        routing_timeout: Optional[float] = None,
        max_consecutive_errors: int = 1,
        arun_concurrency: int = 1,
        policy: Optional[Policy] = None,
        window: Optional[BudgetWindow] = None,
        alerts: Optional[List[Alert]] = None,
        on_output_overrun: str = "estimate",
        max_output_tokens: Optional[int] = None,
        min_useful_tokens: int = 64,
        max_call_share: float = 1.0,
        prompt_margin: float = 0.2,
        thinking_budget: Optional[int] = None,
        overrun_risk: float = 0.0,
        output_lengths: Optional[OutputLengths] = None,
        segment_tokens: int = 2000,
        max_segments: int = 8,
        budget_awareness: Any = False,
        wind_down: Optional[WindDown] = None,
    ):
        _check_litellm_version()
        self._configure(BAARConfig(
            budget=budget,
            small_model=small_model,
            big_model=big_model,
            complexity_threshold=complexity_threshold,
            use_llm_router=use_llm_router,
            system_prompt=system_prompt,
            value_fn=value_fn,
            async_value_fn=async_value_fn,
            value_reject_alpha=value_reject_alpha,
            max_reject_rate=max_reject_rate,
            min_samples_for_reject_clamp=min_samples_for_reject_clamp,
            small_fallback_models=small_fallback_models,
            big_fallback_models=big_fallback_models,
            telemetry_jsonl_path=telemetry_jsonl_path,
            routing_cache_size=routing_cache_size,
            routing_cache_enabled=routing_cache_enabled,
            small_exploration_rate=small_exploration_rate,
            min_cost_threshold=min_cost_threshold,
            routing_task_char_limit=routing_task_char_limit,
            store=store,
            expected_output_tokens=expected_output_tokens,
            routing_cache=routing_cache,
            max_log_steps=max_log_steps,
            domain_model_overrides=domain_model_overrides,
            call_timeout=call_timeout,
            routing_timeout=routing_timeout,
            max_consecutive_errors=max_consecutive_errors,
            arun_concurrency=arun_concurrency,
            policy=policy,
            window=window,
            alerts=alerts,
            on_output_overrun=on_output_overrun,
            max_output_tokens=max_output_tokens,
            min_useful_tokens=min_useful_tokens,
            max_call_share=max_call_share,
            prompt_margin=prompt_margin,
            thinking_budget=thinking_budget,
            overrun_risk=overrun_risk,
            output_lengths=output_lengths,
            segment_tokens=segment_tokens,
            max_segments=max_segments,
            budget_awareness=budget_awareness,
            wind_down=wind_down,
        ))

    def _configure(self, config: BAARConfig) -> None:
        """Single-source initialization from a BAARConfig. Called by __init__ and from_config."""
        if hasattr(self, '_config'):
            raise RuntimeError("BAARRouter is already configured. Create a new instance to change configuration.")
        if config.budget <= 0:
            raise ValueError("Budget must be positive")
        if config.min_cost_threshold < 0:
            raise ValueError("min_cost_threshold must be non-negative")
        caps.validate_mode(config.on_output_overrun)
        if config.max_output_tokens is not None and config.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be a positive integer or None")
        if config.min_useful_tokens < 1:
            raise ValueError("min_useful_tokens must be a positive integer")
        if not 0 < config.max_call_share <= 1:
            raise ValueError("max_call_share must be in (0, 1]")
        if config.prompt_margin < 0:
            raise ValueError("prompt_margin must be non-negative")
        _validate_thinking_budget(config.thinking_budget)
        if not 0 <= config.overrun_risk < 1:
            raise ValueError("overrun_risk must be in [0, 1)")
        if config.segment_tokens < config.min_useful_tokens:
            raise ValueError("segment_tokens must be at least min_useful_tokens")
        if config.max_segments < 1:
            raise ValueError("max_segments must be at least 1")

        self.small_model = config.small_model
        self.big_model = config.big_model
        self.system_prompt = config.system_prompt
        self._min_cost_threshold = float(config.min_cost_threshold)
        self._expected_output_tokens = max(1, int(config.expected_output_tokens))
        self._on_output_overrun = config.on_output_overrun
        self._max_output_tokens = config.max_output_tokens
        self._min_useful_tokens = int(config.min_useful_tokens)
        self._max_call_share = float(config.max_call_share)
        self._prompt_margin = float(config.prompt_margin)
        self._thinking_budget = config.thinking_budget
        self._overrun_risk = float(config.overrun_risk)
        self._segment_tokens = int(config.segment_tokens)
        self._max_segments = int(config.max_segments)
        self._budget_awareness = config.budget_awareness
        self._wind_down: Optional[WindDown] = config.wind_down
        self._final_lock = threading.Lock()
        self._final_started = False
        self._final_answer: Optional[str] = None
        self._final_mark = 0.0  # spend when the final call was claimed
        self._output_lengths: OutputLengths = (
            config.output_lengths if config.output_lengths is not None else OutputLengths()
        )
        self._no_stream_usage_warned: set = set()
        self._unpriced_warned: set = set()
        self._domain_model_overrides = config.domain_model_overrides or {}
        for _domain, _override in self._domain_model_overrides.items():
            if isinstance(_override, tuple):
                if len(_override) != 2:
                    raise ValueError(
                        f"domain_model_overrides['{_domain}'] tuple must be (model, tier), got length {len(_override)}"
                    )
                if not _override[0]:
                    raise ValueError(
                        f"domain_model_overrides['{_domain}'] model name cannot be empty"
                    )
                try:
                    ModelTier(_override[1])
                except ValueError:
                    raise ValueError(
                        f"domain_model_overrides['{_domain}']: invalid tier '{_override[1]}', "
                        f"must be one of {[t.value for t in ModelTier]}"
                    )
            elif not isinstance(_override, str):
                raise ValueError(
                    f"domain_model_overrides['{_domain}'] must be a str or (model, tier) tuple, "
                    f"got {type(_override).__name__}"
                )
            elif not _override:
                raise ValueError(
                    f"domain_model_overrides['{_domain}'] model name cannot be empty"
                )
            elif _override != config.small_model and _override != config.big_model:
                warnings.warn(
                    f"domain_model_overrides['{_domain}']: '{_override}' matches neither "
                    f"small_model ('{config.small_model}') nor big_model ('{config.big_model}'). "
                    f"It will be treated as SMALL tier and use small fallback models. "
                    f"Use a tuple override ('{_override}', 'big') to assign BIG tier.",
                    UserWarning,
                    stacklevel=2,
                )
        self._call_timeout = config.call_timeout
        self._max_consecutive_errors = max(1, int(config.max_consecutive_errors))
        self._arun_concurrency = max(1, int(config.arun_concurrency))
        self._step_counter: int = 0

        self._policy: Optional[Policy] = config.policy
        self._tracker = BudgetTracker(
            total_budget=config.budget,
            store=config.store,
            window=config.window,
            alerts=config.alerts,
        )
        self._router = Router(
            small_model=config.small_model,
            big_model=config.big_model,
            complexity_threshold=config.complexity_threshold,
            use_llm_router=config.use_llm_router,
            routing_cache_size=config.routing_cache_size,
            routing_cache_enabled=config.routing_cache_enabled,
            small_exploration_rate=config.small_exploration_rate,
            routing_task_char_limit=config.routing_task_char_limit,
            routing_cache=config.routing_cache,
            routing_timeout=config.routing_timeout,
        )
        self._log = RoutingLog(
            budget=config.budget,
            small_model=config.small_model,
            big_model=config.big_model,
            max_steps=config.max_log_steps,
        )
        self._value_fn = config.value_fn
        self._async_value_fn = config.async_value_fn
        self._value_reject_alpha = max(0.0, float(config.value_reject_alpha))
        self._max_reject_rate = max(0.0, min(1.0, float(config.max_reject_rate)))
        self._min_samples_for_reject_clamp = max(1, int(config.min_samples_for_reject_clamp))
        self._value_gate_samples = 0
        self._value_gate_rejects = 0
        self._value_gate_lock = threading.Lock()
        self._sync_budget_lock = threading.Lock()
        self._small_fallback_models = list(config.small_fallback_models or [])
        self._big_fallback_models = list(config.big_fallback_models or [])
        self._telemetry_jsonl_path = config.telemetry_jsonl_path
        self._budget_async_lock = asyncio.Lock()
        self._step_counter_lock = threading.Lock()
        self._config: BAARConfig = config

    @classmethod
    def from_config(cls, config: BAARConfig) -> "BAARRouter":
        """Construct a BAARRouter from a BAARConfig instance."""
        _check_litellm_version()
        instance = cls.__new__(cls)
        instance._configure(config)
        return instance

    @classmethod
    def anthropic(cls, **kwargs: Any) -> "BAARRouter":
        """
        Shortcut for Anthropic Claude routing.

        Equivalent to ``BAARRouter.from_config(BAARConfig.anthropic(**kwargs))``.

        Usage::

            router = BAARRouter.anthropic(budget=0.10)
            reply  = router.chat("Summarise this article...")
        """
        return cls.from_config(BAARConfig.anthropic(**kwargs))

    @classmethod
    def openai(cls, **kwargs: Any) -> "BAARRouter":
        """
        Shortcut for OpenAI routing (same defaults as ``BAARRouter(...)``).

        Equivalent to ``BAARRouter.from_config(BAARConfig.openai(**kwargs))``.

        Usage::

            router = BAARRouter.openai(budget=0.10)
        """
        return cls.from_config(BAARConfig.openai(**kwargs))

    # ── Public properties ──────────────────────────────────────────────────────

    @property
    def spent(self) -> float:
        return self._tracker.spent

    @property
    def remaining(self) -> float:
        return self._tracker.remaining

    @property
    def budget(self) -> float:
        return self._tracker.total_budget

    @property
    def log(self) -> RoutingLog:
        return self._log

    @property
    def complexity_threshold(self) -> float:
        return self._router.complexity_threshold

    @property
    def min_cost_threshold(self) -> float:
        return self._min_cost_threshold

    @property
    def output_lengths(self) -> OutputLengths:
        """Output lengths learned from completed calls (used by overrun_risk)."""
        return self._output_lengths

    @property
    def config(self) -> BAARConfig:
        return self._config

    # ── Sync execution ─────────────────────────────────────────────────────────

    def chat(
        self,
        task: str,
        *,
        messages: Optional[List[dict]] = None,
        context: Optional[dict] = None,
        on_output_overrun: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        thinking_budget: Optional[int] = None,
    ) -> str:
        """
        Execute a single routed chat call with hard financial guardrails.

        With ``on_output_overrun="continue"`` the answer is generated in segments
        of ``segment_tokens`` (see _chat_continue). Otherwise it is one call.
        """
        if self._resolve_mode(on_output_overrun) == "continue":
            return self._chat_continue(task, messages, context, max_output_tokens, thinking_budget)
        return self._chat_step(
            task, messages=messages, context=context, on_output_overrun=on_output_overrun,
            max_output_tokens=max_output_tokens, thinking_budget=thinking_budget,
        ).response_text

    def _chat_step(
        self,
        task: str,
        *,
        messages: Optional[List[dict]] = None,
        context: Optional[dict] = None,
        on_output_overrun: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        thinking_budget: Optional[int] = None,
    ) -> StepResult:
        """
        Execute a single routed chat call with hard financial guardrails.

        Args:
            task:     The user's request (also used as routing signal).
            messages: Optional full conversation history. When provided, the LLM
                      call uses this history directly (with system_prompt prepended
                      if not already present). The routing decision still uses
                      `task` as the complexity signal — pass the latest user
                      message as `task` for accurate routing.
        """
        # Build messages first so token counting includes full history
        msgs, final = self._prepare_messages(task, messages)

        # Cache token counts — tiktoken is O(n); same text must not be re-tokenized
        # once per check. Lazy dict so fallback models are only counted if used.
        _tc_cache: dict[str, int] = {}

        def tc(model: str) -> int:
            if model not in _tc_cache:
                _tc_cache[model] = token_counter(messages=msgs, model=model)
            return _tc_cache[model]

        mode, limit, thinking = self._resolve_output_opts(on_output_overrun, max_output_tokens, thinking_budget)
        eot = self._expected_output_tokens

        # ── 0. Hard Kill-Switch Pre-check ─────────────────────────────────────
        prompt_tokens_est = tc(self.small_model)
        try:
            small_plan = self._plan_output(self.small_model, prompt_tokens_est, eot, mode, limit, thinking)
        except BudgetExceeded as e:
            raise BudgetExhausted(
                "Kill-switch activated: "
                f"budget too low (${self.remaining:.6f}) for a useful response "
                f"on {self.small_model}. Request rejected locally with zero network calls.",
                remaining=self.remaining,
            ) from e
        estimated_small_floor = self._tracker.estimate_cost(
            self.small_model, small_plan.prompt_tokens, small_plan.completion_tokens
        )
        effective_min_cost_threshold = max(self._min_cost_threshold, estimated_small_floor)

        if self.remaining < effective_min_cost_threshold:
            raise BudgetExhausted(
                "Kill-switch activated: "
                f"budget too low (${self.remaining:.6f}). "
                f"Effective preflight floor is ${effective_min_cost_threshold:.6f} "
                f"(configured floor ${self._min_cost_threshold:.6f}). "
                "Request rejected locally with zero network calls.",
                remaining=self.remaining,
            )

        # ── 1. Cheapest-call affordability check ──────────────────────────────
        try:
            self._tracker.check_affordability(
                self.small_model, small_plan.prompt_tokens, small_plan.completion_tokens
            )
        except BudgetExceeded as e:
            raise BudgetExhausted(
                "Kill-switch activated: "
                f"budget too low (${self.remaining:.6f}) for the cheapest safe call "
                f"on {self.small_model}. Request rejected locally with zero network calls.",
                remaining=self.remaining,
            ) from e

        # ── 2. Routing Decision ────────────────────────────────────────────────
        decision = self._router.decide(
            task=task,
            budget_utilization=self._tracker.utilization,
        )
        # Use the router's output token estimate for subsequent affordability checks.
        eot = decision.estimated_output_tokens

        # ── 2b. Apply domain model overrides ─────────────────────────────────
        if self._domain_model_overrides and decision.domain in self._domain_model_overrides:
            _override = self._domain_model_overrides[decision.domain]
            if isinstance(_override, tuple):
                override_model, _tier_str = _override
                override_tier = ModelTier(_tier_str)
            else:
                override_model = _override
                override_tier = ModelTier.BIG if override_model == self.big_model else ModelTier.SMALL
            decision = replace(decision, model=override_model, tier=override_tier)

        # ── 2c. Policy evaluation ─────────────────────────────────────────────
        if self._policy is not None:
            decision = self._apply_policy(decision, context)
            if "[POLICY WRAP_UP]" in decision.reason and not final:
                final = self._claim_final()
                msgs = add_note(msgs, self._wrap_up_instruction())
                _tc_cache.clear()  # the prompt changed

        # ── 3. BIG affordability check — downgrade if necessary ───────────────
        if decision.tier == ModelTier.BIG:
            try:
                # The big model must fit its full response; "truncate" only ever
                # cuts off the small model after a downgrade.
                self._check_output_affordable(
                    self.big_model, tc(self.big_model), eot, mode, limit, thinking, uncut=True
                )
            except (BudgetExceeded, caps.OutputLimitUnavailable) as e:
                if "[POLICY FORCE_BIG]" in decision.reason:
                    warnings.warn(
                        f"Policy requested force_big but budget is insufficient "
                        f"(${self._tracker.remaining:.6f} remaining); downgrading to small model.",
                        UserWarning,
                        stacklevel=2,
                    )
                decision = self._router.force_small(decision, str(e))

        # Final pre-flight check for the chosen model
        model_to_use = decision.model
        self._check_output_affordable(model_to_use, tc(model_to_use), eot, mode, limit, thinking)

        # ── 3b. Value gate ────────────────────────────────────────────────────
        decision, model_to_use = self._apply_value_gate(task, decision, model_to_use, tc, 
            self._expected_tokens(tc, eot, limit, thinking), self._affordability_check(tc, eot, mode, limit, thinking),
        )

        # ── 4. Execution with concurrency-safe budget reservation ─────────────
        # Initialise locals before reservation so a KeyboardInterrupt in the
        # gap between check_and_reserve and the try: cannot leak the reservation
        # into a persistent store (mirrors the structure of achat()).
        start_time = time.perf_counter()
        response = None
        attempted_models: List[str] = []
        failover_errors: List[str] = []
        plan = self._plan_output(model_to_use, tc(model_to_use), eot, mode, limit, thinking)
        reserved = self._tracker.estimate_cost(model_to_use, plan.prompt_tokens, plan.completion_tokens)
        with self._sync_budget_lock:
            self._reserve(reserved)
        with self._step_counter_lock:
            self._step_counter += 1
            step_num = self._step_counter

        try:
            for idx, candidate in enumerate(self._execution_model_candidates(decision.tier, model_to_use)):
                if idx > 0:
                    # A fallback may cost more than the primary: top up the reservation
                    # at the candidate's own price, or skip it if that can't be covered.
                    try:
                        plan, reserved = self._cover_fallback(candidate, tc, eot, mode, limit, thinking, reserved)
                    except BudgetExceeded as e:
                        failover_errors.append(f"{candidate}: skipped, unaffordable ({e})")
                        continue
                    except ValueError as e:
                        failover_errors.append(f"{candidate}: skipped ({e})")
                        continue
                attempted_models.append(candidate)
                # Do NOT re-check affordability here: we already have a reservation
                # that covers this call. A redundant check against the already-reduced
                # remaining would reject valid fallbacks at low budget (same fix as achat).
                try:
                    response = _sync_completion_with_retry(
                        candidate, msgs, timeout=self._call_timeout, max_tokens=plan.max_tokens,
                        extra=plan.params,
                    )
                    model_to_use = candidate
                    break
                except Exception as e:
                    failover_errors.append(
                        f"{candidate}: completion failed ({type(e).__name__}: {e})"
                    )

            if response is None:
                failures = "; ".join(failover_errors) or "no candidate model succeeded"
                raise RuntimeError(f"All provider/model attempts failed: {failures}")

            if model_to_use != decision.model:
                decision = replace(
                    decision,
                    model=model_to_use,
                    reason=f"{decision.reason} [FAILOVER→{model_to_use}]",
                )

            latency_ms = (time.perf_counter() - start_time) * 1000

            # ── 5. Recording ──────────────────────────────────────────────────
            record = self._tracker.record(response, model=model_to_use)
        finally:
            self._tracker.cancel_reservation(reserved)

        step_result = StepResult(
            step_num=step_num,
            task=task,
            decision=decision,
            response_text=response.choices[0].message.content or "",
            cost=record.cost,
            cumulative_cost=record.cumulative_cost,
            prompt_tokens=record.prompt_tokens,
            completion_tokens=record.completion_tokens,
            latency_ms=latency_ms,
            attempted_models=attempted_models,
            failover_count=max(0, len(attempted_models) - 1),
            failover_errors=failover_errors,
            truncated_by_budget=plan.budget_limited and response.choices[0].finish_reason == "length",
            finish_reason=_finish_reason(response.choices[0].finish_reason),
            cached_prompt_tokens=cached_tokens(response.usage),
        )
        self._log.add(step_result)
        self._append_telemetry(step_result)
        self._output_lengths.observe(step_result.model_used, step_result.completion_tokens)
        if final:
            self._final_answer = step_result.response_text
        return step_result

    def stream_chat(
        self,
        task: str,
        *,
        messages: Optional[List[dict]] = None,
        context: Optional[dict] = None,
        on_output_overrun: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        thinking_budget: Optional[int] = None,
    ) -> Iterator[str]:
        """
        Streaming version of chat() — yields text chunks as they arrive. With
        on_output_overrun="continue", segments are streamed one after another.
        """
        if self._resolve_mode(on_output_overrun) == "continue":
            yield from self._stream_continue(task, messages, context, max_output_tokens, thinking_budget)
            return
        yield from self._stream_step(
            task, messages=messages, context=context, on_output_overrun=on_output_overrun,
            max_output_tokens=max_output_tokens, thinking_budget=thinking_budget,
        )

    def _stream_step(
        self,
        task: str,
        *,
        messages: Optional[List[dict]] = None,
        context: Optional[dict] = None,
        on_output_overrun: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        thinking_budget: Optional[int] = None,
        step_sink: Optional[list] = None,
    ) -> Iterator[str]:
        """
        Streaming version of chat() — yields text chunks as they arrive.

        Budget guardrails are identical to chat(): preflight checks run before
        the first token, and the reservation is always cleaned up via try/finally.
        Token counts are accumulated during streaming and recorded afterward.
        """
        # Build messages first so token counting includes full history
        msgs, final = self._prepare_messages(task, messages)

        _tc_cache: dict[str, int] = {}

        def tc(model: str) -> int:
            if model not in _tc_cache:
                _tc_cache[model] = token_counter(messages=msgs, model=model)
            return _tc_cache[model]

        mode, limit, thinking = self._resolve_output_opts(on_output_overrun, max_output_tokens, thinking_budget)
        eot = self._expected_output_tokens

        # ── 0. Kill-switch pre-check ───────────────────────────────────────────
        prompt_tokens_est = tc(self.small_model)
        try:
            small_plan = self._plan_output(self.small_model, prompt_tokens_est, eot, mode, limit, thinking)
        except BudgetExceeded as e:
            raise BudgetExhausted(
                "Kill-switch activated: "
                f"budget too low (${self.remaining:.6f}) for a useful response "
                f"on {self.small_model}. Request rejected locally with zero network calls.",
                remaining=self.remaining,
            ) from e
        estimated_small_floor = self._tracker.estimate_cost(
            self.small_model, small_plan.prompt_tokens, small_plan.completion_tokens
        )
        effective_min_cost_threshold = max(self._min_cost_threshold, estimated_small_floor)

        if self.remaining < effective_min_cost_threshold:
            raise BudgetExhausted(
                "Kill-switch activated: "
                f"budget too low (${self.remaining:.6f}). "
                f"Effective preflight floor is ${effective_min_cost_threshold:.6f} "
                f"(configured floor ${self._min_cost_threshold:.6f}). "
                "Request rejected locally with zero network calls.",
                remaining=self.remaining,
            )

        try:
            self._tracker.check_affordability(
                self.small_model, small_plan.prompt_tokens, small_plan.completion_tokens
            )
        except BudgetExceeded as e:
            raise BudgetExhausted(
                "Kill-switch activated: "
                f"budget too low (${self.remaining:.6f}) for the cheapest safe call "
                f"on {self.small_model}. Request rejected locally with zero network calls.",
                remaining=self.remaining,
            ) from e

        # ── 1. Routing ─────────────────────────────────────────────────────────
        decision = self._router.decide(
            task=task,
            budget_utilization=self._tracker.utilization,
        )
        eot = decision.estimated_output_tokens

        # ── 1b. Apply domain model overrides ──────────────────────────────────
        if self._domain_model_overrides and decision.domain in self._domain_model_overrides:
            _override = self._domain_model_overrides[decision.domain]
            if isinstance(_override, tuple):
                override_model, _tier_str = _override
                override_tier = ModelTier(_tier_str)
            else:
                override_model = _override
                override_tier = ModelTier.BIG if override_model == self.big_model else ModelTier.SMALL
            decision = replace(decision, model=override_model, tier=override_tier)

        # ── 1c. Policy evaluation ──────────────────────────────────────────────
        if self._policy is not None:
            decision = self._apply_policy(decision, context)
            if "[POLICY WRAP_UP]" in decision.reason and not final:
                final = self._claim_final()
                msgs = add_note(msgs, self._wrap_up_instruction())
                _tc_cache.clear()  # the prompt changed

        if decision.tier == ModelTier.BIG:
            try:
                # The big model must fit its full response; "truncate" only ever
                # cuts off the small model after a downgrade.
                self._check_output_affordable(
                    self.big_model, tc(self.big_model), eot, mode, limit, thinking, uncut=True
                )
            except (BudgetExceeded, caps.OutputLimitUnavailable) as e:
                if "[POLICY FORCE_BIG]" in decision.reason:
                    warnings.warn(
                        f"Policy requested force_big but budget is insufficient "
                        f"(${self._tracker.remaining:.6f} remaining); downgrading to small model.",
                        UserWarning,
                        stacklevel=2,
                    )
                decision = self._router.force_small(decision, str(e))

        model_to_use = decision.model
        self._check_output_affordable(model_to_use, tc(model_to_use), eot, mode, limit, thinking)
        decision, model_to_use = self._apply_value_gate(task, decision, model_to_use, tc, 
            self._expected_tokens(tc, eot, limit, thinking), self._affordability_check(tc, eot, mode, limit, thinking),
        )

        # ── 2. Reserve budget atomically, then find a working stream ──────────
        # Locals are initialised before reservation so a KeyboardInterrupt in
        # the gap cannot leak the reservation into a persistent store.
        attempted_models: List[str] = []
        failover_errors: List[str] = []
        response_stream = None
        model_to_use_final = model_to_use
        plan = self._plan_output(model_to_use, tc(model_to_use), eot, mode, limit, thinking)
        reserved = self._tracker.estimate_cost(model_to_use, plan.prompt_tokens, plan.completion_tokens)
        with self._sync_budget_lock:
            self._reserve(reserved)
        with self._step_counter_lock:
            self._step_counter += 1
            step_num = self._step_counter

        try:
            for idx, candidate in enumerate(self._execution_model_candidates(decision.tier, model_to_use)):
                if idx > 0:
                    # A fallback may cost more than the primary: top up the reservation
                    # at the candidate's own price, or skip it if that can't be covered.
                    try:
                        plan, reserved = self._cover_fallback(candidate, tc, eot, mode, limit, thinking, reserved)
                    except BudgetExceeded as e:
                        failover_errors.append(f"{candidate}: skipped, unaffordable ({e})")
                        continue
                    except ValueError as e:
                        failover_errors.append(f"{candidate}: skipped ({e})")
                        continue
                attempted_models.append(candidate)
                # Do NOT re-check affordability here: reservation already covers this call.
                try:
                    response_stream = _sync_completion_with_retry(
                        candidate, msgs, stream=True, timeout=self._call_timeout,
                        max_tokens=plan.max_tokens,
                        extra=plan.params,
                        stream_options=_stream_usage_options(candidate),
                    )
                    model_to_use_final = candidate
                    break
                except Exception as e:
                    failover_errors.append(
                        f"{candidate}: completion failed ({type(e).__name__}: {e})"
                    )
        except BaseException:  # pragma: no cover
            if response_stream is None:  # pragma: no cover
                self._tracker.cancel_reservation(reserved)  # pragma: no cover
            raise  # pragma: no cover

        if response_stream is None:
            self._tracker.cancel_reservation(reserved)
            failures = "; ".join(failover_errors) or "no candidate model succeeded"
            raise RuntimeError(f"All provider/model attempts failed: {failures}")

        # ── 3. Stream, then record ─────────────────────────────────────────────
        parts: list[str] = []
        start_time = time.perf_counter()
        prompt_tokens_for_record = 0  # set inside try; default protects finally if interrupted early
        finish_reason: Optional[str] = None
        usage: Optional[tuple] = None
        cached_for_record = 0
        completed = False

        try:
            if model_to_use_final != decision.model:
                decision = replace(
                    decision,
                    model=model_to_use_final,
                    reason=f"{decision.reason} [FAILOVER→{model_to_use_final}]",
                )
            prompt_tokens_for_record = tc(model_to_use_final)
            for chunk in response_stream:
                usage = _reported_usage(chunk) or usage
                if not chunk.choices:
                    continue  # usage-only final chunk (OpenAI include_usage)
                delta = chunk.choices[0].delta.content or ""
                finish_reason = getattr(chunk.choices[0], "finish_reason", None) or finish_reason
                parts.append(delta)
                yield delta
            completed = True
        finally:
            if not completed:
                _close_stream(response_stream)  # stop the provider generating (and billing)
            full_text = "".join(parts)
            latency_ms = (time.perf_counter() - start_time) * 1000
            if usage is not None and usage[1] == 0 and full_text:
                usage = None  # 0 completion tokens with text produced isn't real usage
            if usage is not None:
                # Provider-reported usage includes hidden reasoning tokens.
                reported_prompt, completion_tokens, cached_for_record = usage
                # Anthropic's final event can report 0 input tokens; keep our count then.
                if reported_prompt > 0:
                    prompt_tokens_for_record = reported_prompt
            else:
                completion_tokens = token_counter(full_text, model=model_to_use_final)
                if completed:
                    self._warn_no_stream_usage(model_to_use_final)
                elif mode != "estimate":
                    # Stopped early: the provider may have generated (and billed) more
                    # than the visible text, so record what was reserved for it.
                    completion_tokens = max(completion_tokens, plan.completion_tokens)
            try:
                # Record actual cost BEFORE cancelling the reservation so that
                # concurrent threads never see a transient under-count window
                # (spent would drop then rise, allowing over-allocation).
                record = self._tracker.record_manual(
                    model_to_use_final, prompt_tokens_for_record, completion_tokens,
                    cached_tokens=cached_for_record,
                )
                step_result = StepResult(
                    step_num=step_num,
                    task=task,
                    decision=decision,
                    response_text=full_text,
                    cost=record.cost,
                    cumulative_cost=record.cumulative_cost,
                    prompt_tokens=record.prompt_tokens,
                    completion_tokens=record.completion_tokens,
                    latency_ms=latency_ms,
                    attempted_models=attempted_models,
                    failover_count=max(0, len(attempted_models) - 1),
                    failover_errors=failover_errors,
                    truncated_by_budget=plan.budget_limited and finish_reason == "length",
                    finish_reason=_finish_reason(finish_reason),
                    cached_prompt_tokens=cached_for_record,
                )
                self._log.add(step_result)
                self._append_telemetry(step_result)
                if completed:  # an early exit records the reservation, not a real length
                    self._output_lengths.observe(step_result.model_used, step_result.completion_tokens)
                if step_sink is not None:
                    step_sink.append(step_result)
                if final and completed:
                    self._final_answer = full_text
            except Exception as _e:
                self._log.errors.append(f"post-stream recording failed: {type(_e).__name__}: {_e}")
            finally:
                self._tracker.cancel_reservation(reserved)

    def run(self, tasks: List[str], *, context: Optional[dict] = None) -> RoutingLog:
        """
        Execute a series of tasks sequentially.
        Stops early if the hard budget cap is hit or max_consecutive_errors reached.
        Unexpected errors are recorded in log.errors rather than silently dropped.
        """
        consecutive_errors = 0
        for task in tasks:
            try:
                self.chat(task, context=context)
                consecutive_errors = 0
            except (BudgetExceeded, BudgetExhausted):
                break
            except (TaskRejected, PolicyViolation):
                consecutive_errors = 0
                continue
            except Exception as e:
                self._log.errors.append(f"{type(e).__name__}: {e}")
                consecutive_errors += 1
                if consecutive_errors >= self._max_consecutive_errors:
                    break
        return self._log

    # ── Async execution ────────────────────────────────────────────────────────

    async def achat(
        self,
        task: str,
        *,
        messages: Optional[List[dict]] = None,
        context: Optional[dict] = None,
        on_output_overrun: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        thinking_budget: Optional[int] = None,
    ) -> str:
        """Async chat() — same guardrails, including on_output_overrun="continue"."""
        if self._resolve_mode(on_output_overrun) == "continue":
            return await self._achat_continue(task, messages, context, max_output_tokens, thinking_budget)
        step = await self._achat_step(
            task, messages=messages, context=context, on_output_overrun=on_output_overrun,
            max_output_tokens=max_output_tokens, thinking_budget=thinking_budget,
        )
        return step.response_text

    async def _achat_step(
        self,
        task: str,
        *,
        messages: Optional[List[dict]] = None,
        context: Optional[dict] = None,
        on_output_overrun: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        thinking_budget: Optional[int] = None,
    ) -> StepResult:
        """
        Async version of chat() — identical hard financial guardrails,
        non-blocking for asyncio / FastAPI / aiohttp applications.

        Concurrency safety: a per-instance asyncio.Lock atomically reserves the
        estimated cost before the await, preventing concurrent callers from both
        passing the affordability check and together overshooting the budget.
        """
        # Build messages first so token counting includes full history
        msgs, final = self._prepare_messages(task, messages)

        _tc_cache: dict[str, int] = {}

        def tc(model: str) -> int:
            if model not in _tc_cache:
                _tc_cache[model] = token_counter(messages=msgs, model=model)
            return _tc_cache[model]

        mode, limit, thinking = self._resolve_output_opts(on_output_overrun, max_output_tokens, thinking_budget)
        eot = self._expected_output_tokens

        # ── 0. Hard Kill-Switch Pre-check ─────────────────────────────────────
        prompt_tokens_est = tc(self.small_model)
        try:
            small_plan = self._plan_output(self.small_model, prompt_tokens_est, eot, mode, limit, thinking)
        except BudgetExceeded as e:
            raise BudgetExhausted(
                "Kill-switch activated: "
                f"budget too low (${self.remaining:.6f}) for a useful response "
                f"on {self.small_model}. Request rejected locally with zero network calls.",
                remaining=self.remaining,
            ) from e
        estimated_small_floor = self._tracker.estimate_cost(
            self.small_model, small_plan.prompt_tokens, small_plan.completion_tokens
        )
        effective_min_cost_threshold = max(self._min_cost_threshold, estimated_small_floor)

        if self.remaining < effective_min_cost_threshold:
            raise BudgetExhausted(
                "Kill-switch activated: "
                f"budget too low (${self.remaining:.6f}). "
                f"Effective preflight floor is ${effective_min_cost_threshold:.6f} "
                f"(configured floor ${self._min_cost_threshold:.6f}). "
                "Request rejected locally with zero network calls.",
                remaining=self.remaining,
            )

        # ── 1. Cheapest-call affordability check ──────────────────────────────
        try:
            self._tracker.check_affordability(
                self.small_model, small_plan.prompt_tokens, small_plan.completion_tokens
            )
        except BudgetExceeded as e:
            raise BudgetExhausted(
                "Kill-switch activated: "
                f"budget too low (${self.remaining:.6f}) for the cheapest safe call "
                f"on {self.small_model}. Request rejected locally with zero network calls.",
                remaining=self.remaining,
            ) from e

        # ── 2. Routing Decision (async — doesn't block event loop) ────────────
        decision = await self._router.adecide(
            task=task,
            budget_utilization=self._tracker.utilization,
        )
        # Use the router's output token estimate for subsequent affordability checks.
        eot = decision.estimated_output_tokens

        # ── 2b. Apply domain model overrides ─────────────────────────────────
        if self._domain_model_overrides and decision.domain in self._domain_model_overrides:
            _override = self._domain_model_overrides[decision.domain]
            if isinstance(_override, tuple):
                override_model, _tier_str = _override
                override_tier = ModelTier(_tier_str)
            else:
                override_model = _override
                override_tier = ModelTier.BIG if override_model == self.big_model else ModelTier.SMALL
            decision = replace(decision, model=override_model, tier=override_tier)

        # ── 2c. Policy evaluation ─────────────────────────────────────────────
        if self._policy is not None:
            decision = self._apply_policy(decision, context)
            if "[POLICY WRAP_UP]" in decision.reason and not final:
                final = self._claim_final()
                msgs = add_note(msgs, self._wrap_up_instruction())
                _tc_cache.clear()  # the prompt changed

        # ── 3. BIG affordability check — downgrade if necessary ───────────────
        if decision.tier == ModelTier.BIG:
            try:
                # The big model must fit its full response; "truncate" only ever
                # cuts off the small model after a downgrade.
                self._check_output_affordable(
                    self.big_model, tc(self.big_model), eot, mode, limit, thinking, uncut=True
                )
            except (BudgetExceeded, caps.OutputLimitUnavailable) as e:
                if "[POLICY FORCE_BIG]" in decision.reason:
                    warnings.warn(
                        f"Policy requested force_big but budget is insufficient "
                        f"(${self._tracker.remaining:.6f} remaining); downgrading to small model.",
                        UserWarning,
                        stacklevel=2,
                    )
                decision = self._router.force_small(decision, str(e))

        model_to_use = decision.model
        self._check_output_affordable(model_to_use, tc(model_to_use), eot, mode, limit, thinking)

        # ── 3b. Value gate (same logic as chat(), fully ported) ───────────────
        decision, model_to_use = await self._aapply_value_gate(task, decision, model_to_use, tc, 
            self._expected_tokens(tc, eot, limit, thinking), self._affordability_check(tc, eot, mode, limit, thinking),
        )

        # ── 4. Execution with concurrency-safe budget reservation ─────────────
        start_time = time.perf_counter()
        response = None
        attempted_models: List[str] = []
        failover_errors: List[str] = []

        # Atomically check + reserve to prevent concurrent overspend.
        async with self._budget_async_lock:
            plan = self._plan_output(model_to_use, tc(model_to_use), eot, mode, limit, thinking)
            reserved = self._tracker.estimate_cost(model_to_use, plan.prompt_tokens, plan.completion_tokens)
            self._reserve(reserved)
        with self._step_counter_lock:
            self._step_counter += 1
            step_num = self._step_counter

        # Use try/finally (not except) so BaseException (KeyboardInterrupt, CancelledError etc.)
        # also releases the reservation — prevents permanent budget under-reporting in persistent stores.
        try:
            for idx, candidate in enumerate(self._execution_model_candidates(decision.tier, model_to_use)):
                if idx > 0:
                    # A fallback may cost more than the primary: top up the reservation
                    # at the candidate's own price, or skip it if that can't be covered.
                    try:
                        plan, reserved = self._cover_fallback(candidate, tc, eot, mode, limit, thinking, reserved)
                    except BudgetExceeded as e:
                        failover_errors.append(f"{candidate}: skipped, unaffordable ({e})")
                        continue
                    except ValueError as e:
                        failover_errors.append(f"{candidate}: skipped ({e})")
                        continue
                attempted_models.append(candidate)
                # Do NOT re-check affordability here: we already have a reservation
                # that covers this call. A redundant check against the already-reduced
                # remaining would reject the primary candidate at low budget (BUG FIX).
                try:
                    response = await _async_completion_with_retry(
                        candidate, msgs, timeout=self._call_timeout, max_tokens=plan.max_tokens,
                        extra=plan.params,
                    )
                    model_to_use = candidate
                    break
                except Exception as e:
                    failover_errors.append(
                        f"{candidate}: completion failed ({type(e).__name__}: {e})"
                    )

            if response is None:
                failures = "; ".join(failover_errors) or "no candidate model succeeded"
                raise RuntimeError(f"All provider/model attempts failed: {failures}")

            if model_to_use != decision.model:
                decision = replace(
                    decision,
                    model=model_to_use,
                    reason=f"{decision.reason} [FAILOVER→{model_to_use}]",
                )

            latency_ms = (time.perf_counter() - start_time) * 1000

            # Record actual cost, then cancel the reservation.
            record = self._tracker.record(response, model=model_to_use)
        finally:
            self._tracker.cancel_reservation(reserved)

        step_result = StepResult(
            step_num=step_num,
            task=task,
            decision=decision,
            response_text=response.choices[0].message.content or "",
            cost=record.cost,
            cumulative_cost=record.cumulative_cost,
            prompt_tokens=record.prompt_tokens,
            completion_tokens=record.completion_tokens,
            latency_ms=latency_ms,
            attempted_models=attempted_models,
            failover_count=max(0, len(attempted_models) - 1),
            failover_errors=failover_errors,
            truncated_by_budget=plan.budget_limited and response.choices[0].finish_reason == "length",
            finish_reason=_finish_reason(response.choices[0].finish_reason),
            cached_prompt_tokens=cached_tokens(response.usage),
        )
        self._log.add(step_result)
        self._append_telemetry(step_result)
        self._output_lengths.observe(step_result.model_used, step_result.completion_tokens)
        if final:
            self._final_answer = step_result.response_text
        return step_result

    async def astream_chat(
        self,
        task: str,
        *,
        messages: Optional[List[dict]] = None,
        context: Optional[dict] = None,
        on_output_overrun: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        thinking_budget: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        """Async streaming chat() — same guardrails, including on_output_overrun="continue"."""
        if self._resolve_mode(on_output_overrun) == "continue":
            segments = self._astream_continue(task, messages, context, max_output_tokens, thinking_budget)
            try:
                async for delta in segments:
                    yield delta
            finally:
                await segments.aclose()
            return
        inner = self._astream_step(
            task, messages=messages, context=context, on_output_overrun=on_output_overrun,
            max_output_tokens=max_output_tokens, thinking_budget=thinking_budget,
        )
        try:
            async for delta in inner:
                yield delta
        finally:
            # Unlike `yield from`, `async for` doesn't pass a close on to the inner
            # generator: close it now so an early exit ends the provider stream.
            await inner.aclose()

    async def _astream_step(
        self,
        task: str,
        *,
        messages: Optional[List[dict]] = None,
        context: Optional[dict] = None,
        on_output_overrun: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        thinking_budget: Optional[int] = None,
        step_sink: Optional[list] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Async streaming version of achat() — yields text chunks as they arrive.

        Budget guardrails are identical to achat(). Reservation is held during
        streaming and cleaned up via try/finally.
        """
        # Build messages first so token counting includes full history
        msgs, final = self._prepare_messages(task, messages)

        _tc_cache: dict[str, int] = {}

        def tc(model: str) -> int:
            if model not in _tc_cache:
                _tc_cache[model] = token_counter(messages=msgs, model=model)
            return _tc_cache[model]

        mode, limit, thinking = self._resolve_output_opts(on_output_overrun, max_output_tokens, thinking_budget)
        eot = self._expected_output_tokens
        prompt_tokens_est = tc(self.small_model)
        try:
            small_plan = self._plan_output(self.small_model, prompt_tokens_est, eot, mode, limit, thinking)
        except BudgetExceeded as e:
            raise BudgetExhausted(
                "Kill-switch activated: "
                f"budget too low (${self.remaining:.6f}) for a useful response "
                f"on {self.small_model}. Request rejected locally with zero network calls.",
                remaining=self.remaining,
            ) from e
        estimated_small_floor = self._tracker.estimate_cost(
            self.small_model, small_plan.prompt_tokens, small_plan.completion_tokens
        )
        effective_min_cost_threshold = max(self._min_cost_threshold, estimated_small_floor)

        if self.remaining < effective_min_cost_threshold:
            raise BudgetExhausted(
                "Kill-switch activated: "
                f"budget too low (${self.remaining:.6f}). "
                f"Effective preflight floor is ${effective_min_cost_threshold:.6f} "
                f"(configured floor ${self._min_cost_threshold:.6f}). "
                "Request rejected locally with zero network calls.",
                remaining=self.remaining,
            )

        try:
            self._tracker.check_affordability(
                self.small_model, small_plan.prompt_tokens, small_plan.completion_tokens
            )
        except BudgetExceeded as e:
            raise BudgetExhausted(
                "Kill-switch activated: "
                f"budget too low (${self.remaining:.6f}) for the cheapest safe call "
                f"on {self.small_model}. Request rejected locally with zero network calls.",
                remaining=self.remaining,
            ) from e

        decision = await self._router.adecide(
            task=task,
            budget_utilization=self._tracker.utilization,
        )
        eot = decision.estimated_output_tokens

        # ── Apply domain model overrides ───────────────────────────────────────
        if self._domain_model_overrides and decision.domain in self._domain_model_overrides:
            _override = self._domain_model_overrides[decision.domain]
            if isinstance(_override, tuple):
                override_model, _tier_str = _override
                override_tier = ModelTier(_tier_str)
            else:
                override_model = _override
                override_tier = ModelTier.BIG if override_model == self.big_model else ModelTier.SMALL
            decision = replace(decision, model=override_model, tier=override_tier)

        # ── Policy evaluation ──────────────────────────────────────────────────
        if self._policy is not None:
            decision = self._apply_policy(decision, context)
            if "[POLICY WRAP_UP]" in decision.reason and not final:
                final = self._claim_final()
                msgs = add_note(msgs, self._wrap_up_instruction())
                _tc_cache.clear()  # the prompt changed

        if decision.tier == ModelTier.BIG:
            try:
                # The big model must fit its full response; "truncate" only ever
                # cuts off the small model after a downgrade.
                self._check_output_affordable(
                    self.big_model, tc(self.big_model), eot, mode, limit, thinking, uncut=True
                )
            except (BudgetExceeded, caps.OutputLimitUnavailable) as e:
                if "[POLICY FORCE_BIG]" in decision.reason:
                    warnings.warn(
                        f"Policy requested force_big but budget is insufficient "
                        f"(${self._tracker.remaining:.6f} remaining); downgrading to small model.",
                        UserWarning,
                        stacklevel=2,
                    )
                decision = self._router.force_small(decision, str(e))

        model_to_use = decision.model
        self._check_output_affordable(model_to_use, tc(model_to_use), eot, mode, limit, thinking)
        decision, model_to_use = await self._aapply_value_gate(task, decision, model_to_use, tc, 
            self._expected_tokens(tc, eot, limit, thinking), self._affordability_check(tc, eot, mode, limit, thinking),
        )

        # ── Reserve budget atomically, then find a working stream ─────────────
        # Reservation precedes the API call so concurrent coroutines cannot both
        # pass the affordability check and both open streams (wasted API calls).
        async with self._budget_async_lock:
            plan = self._plan_output(model_to_use, tc(model_to_use), eot, mode, limit, thinking)
            reserved = self._tracker.estimate_cost(model_to_use, plan.prompt_tokens, plan.completion_tokens)
            self._reserve(reserved)
        with self._step_counter_lock:
            self._step_counter += 1
            step_num = self._step_counter

        attempted_models: List[str] = []
        failover_errors: List[str] = []
        response_stream = None
        model_to_use_final = model_to_use

        try:
            for idx, candidate in enumerate(self._execution_model_candidates(decision.tier, model_to_use)):
                if idx > 0:
                    # A fallback may cost more than the primary: top up the reservation
                    # at the candidate's own price, or skip it if that can't be covered.
                    try:
                        plan, reserved = self._cover_fallback(candidate, tc, eot, mode, limit, thinking, reserved)
                    except BudgetExceeded as e:
                        failover_errors.append(f"{candidate}: skipped, unaffordable ({e})")
                        continue
                    except ValueError as e:
                        failover_errors.append(f"{candidate}: skipped ({e})")
                        continue
                attempted_models.append(candidate)
                # Do NOT re-check affordability here: reservation already covers this call.
                try:
                    response_stream = await _async_completion_with_retry(
                        candidate, msgs, stream=True, timeout=self._call_timeout,
                        max_tokens=plan.max_tokens,
                        extra=plan.params,
                        stream_options=_stream_usage_options(candidate),
                    )
                    model_to_use_final = candidate
                    break
                except Exception as e:
                    failover_errors.append(
                        f"{candidate}: completion failed ({type(e).__name__}: {e})"
                    )
        except BaseException:  # pragma: no cover
            if response_stream is None:  # pragma: no cover
                self._tracker.cancel_reservation(reserved)  # pragma: no cover
            raise  # pragma: no cover

        if response_stream is None:
            self._tracker.cancel_reservation(reserved)
            failures = "; ".join(failover_errors) or "no candidate model succeeded"
            raise RuntimeError(f"All provider/model attempts failed: {failures}")

        parts: list[str] = []
        start_time = time.perf_counter()
        prompt_tokens_for_record = 0  # set inside try; default protects finally if interrupted early
        finish_reason: Optional[str] = None
        usage: Optional[tuple] = None
        cached_for_record = 0
        completed = False

        try:
            if model_to_use_final != decision.model:
                decision = replace(
                    decision,
                    model=model_to_use_final,
                    reason=f"{decision.reason} [FAILOVER→{model_to_use_final}]",
                )
            prompt_tokens_for_record = tc(model_to_use_final)
            async for chunk in response_stream:
                usage = _reported_usage(chunk) or usage
                if not chunk.choices:
                    continue  # usage-only final chunk (OpenAI include_usage)
                delta = chunk.choices[0].delta.content or ""
                finish_reason = getattr(chunk.choices[0], "finish_reason", None) or finish_reason
                parts.append(delta)
                yield delta
            completed = True
        finally:
            if not completed:
                # Stop the provider generating (and billing) after an early exit.
                aclose = getattr(response_stream, "aclose", None)
                if aclose is not None:
                    try:
                        await aclose()
                    except Exception:
                        pass
            full_text = "".join(parts)
            latency_ms = (time.perf_counter() - start_time) * 1000
            if usage is not None and usage[1] == 0 and full_text:
                usage = None  # 0 completion tokens with text produced isn't real usage
            if usage is not None:
                # Provider-reported usage includes hidden reasoning tokens.
                reported_prompt, completion_tokens, cached_for_record = usage
                # Anthropic's final event can report 0 input tokens; keep our count then.
                if reported_prompt > 0:
                    prompt_tokens_for_record = reported_prompt
            else:
                completion_tokens = token_counter(full_text, model=model_to_use_final)
                if completed:
                    self._warn_no_stream_usage(model_to_use_final)
                elif mode != "estimate":
                    # Stopped early: the provider may have generated (and billed) more
                    # than the visible text, so record what was reserved for it.
                    completion_tokens = max(completion_tokens, plan.completion_tokens)
            try:
                record = self._tracker.record_manual(
                    model_to_use_final, prompt_tokens_for_record, completion_tokens,
                    cached_tokens=cached_for_record,
                )
                step_result = StepResult(
                    step_num=step_num,
                    task=task,
                    decision=decision,
                    response_text=full_text,
                    cost=record.cost,
                    cumulative_cost=record.cumulative_cost,
                    prompt_tokens=record.prompt_tokens,
                    completion_tokens=record.completion_tokens,
                    latency_ms=latency_ms,
                    attempted_models=attempted_models,
                    failover_count=max(0, len(attempted_models) - 1),
                    failover_errors=failover_errors,
                    truncated_by_budget=plan.budget_limited and finish_reason == "length",
                    finish_reason=_finish_reason(finish_reason),
                    cached_prompt_tokens=cached_for_record,
                )
                self._log.add(step_result)
                self._append_telemetry(step_result)
                if completed:  # an early exit records the reservation, not a real length
                    self._output_lengths.observe(step_result.model_used, step_result.completion_tokens)
                if step_sink is not None:
                    step_sink.append(step_result)
                if final and completed:
                    self._final_answer = full_text
            except Exception as _e:
                self._log.errors.append(f"post-stream recording failed: {type(_e).__name__}: {_e}")
            finally:
                self._tracker.cancel_reservation(reserved)

    async def arun(self, tasks: List[str], *, context: Optional[dict] = None) -> RoutingLog:
        """
        Async task execution. Stops early if the hard budget cap is hit.
        When arun_concurrency > 1, tasks run concurrently (bounded by semaphore).
        Unexpected errors are recorded in log.errors rather than silently dropped.

        Note: max_consecutive_errors applies strictly to sequential mode only.
        In concurrent mode any interleaved success resets err_count, so the
        counter reflects best-effort error tracking rather than a hard stop.
        """
        if self._arun_concurrency <= 1:
            consecutive_errors = 0
            for task in tasks:
                try:
                    await self.achat(task, context=context)
                    consecutive_errors = 0
                except (BudgetExceeded, BudgetExhausted):
                    break
                except (TaskRejected, PolicyViolation):
                    consecutive_errors = 0
                    continue
                except Exception as e:
                    self._log.errors.append(f"{type(e).__name__}: {e}")
                    consecutive_errors += 1
                    if consecutive_errors >= self._max_consecutive_errors:
                        break
        else:
            sem = asyncio.Semaphore(self._arun_concurrency)
            budget_stop = asyncio.Event()
            # err_count uses a list so the closure can mutate it.
            # No lock needed: asyncio is cooperative and there is no await
            # between the read and write of err_count[0] within _one.
            err_count = [0]

            async def _one(task: str) -> None:
                if budget_stop.is_set():
                    return
                if err_count[0] >= self._max_consecutive_errors:
                    return
                async with sem:
                    # Re-check after acquiring semaphore: another coroutine may have
                    # updated err_count or budget_stop while this one was waiting.
                    if budget_stop.is_set():  # pragma: no cover
                        return  # pragma: no cover
                    if err_count[0] >= self._max_consecutive_errors:  # pragma: no cover
                        return  # pragma: no cover
                    try:
                        await self.achat(task, context=context)
                        err_count[0] = 0
                    except (BudgetExceeded, BudgetExhausted):
                        budget_stop.set()
                    except (TaskRejected, PolicyViolation):
                        err_count[0] = 0
                    except Exception as e:
                        self._log.errors.append(f"{type(e).__name__}: {e}")
                        err_count[0] += 1

            await asyncio.gather(*[_one(t) for t in tasks])
        return self._log

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _apply_policy(self, decision, context: Optional[dict]) -> object:
        """Evaluate the policy and mutate the routing decision accordingly."""
        # System facts always win over user-supplied context keys.
        facts = {
            **(context or {}),
            "model": decision.model,
            "utilization": self._tracker.utilization,
            "domain": decision.domain,
            "complexity": decision.complexity_score,
        }
        action = self._policy.evaluate(facts)
        if action == PolicyAction.BLOCK:
            raise PolicyViolation(
                f"Policy blocked call to '{decision.model}'",
                facts=facts,
            )
        if action == PolicyAction.FORCE_SMALL:
            # Use replace() directly so forced_by_budget stays False —
            # this is a governance decision, not a budget constraint.
            return replace(
                decision,
                tier=ModelTier.SMALL,
                model=self.small_model,
                reason=f"[POLICY FORCE_SMALL] {decision.reason}",
                forced_by_budget=False,
            )
        if action == PolicyAction.WRAP_UP:
            return replace(decision, reason=f"[POLICY WRAP_UP] {decision.reason}")
        if action == PolicyAction.FORCE_BIG:
            return replace(
                decision,
                tier=ModelTier.BIG,
                model=self.big_model,
                reason=f"[POLICY FORCE_BIG] {decision.reason}",
            )
        return decision  # ALLOW or no match

    # ── budget awareness and wind-down ────────────────────────────────────────

    def budget_status(self) -> BudgetStatus:
        """What's been spent, what's left, and roughly how many calls that buys."""
        total, spent, remaining = self._tracker.total_budget, self._tracker.spent, self._tracker.remaining
        calls = self._log.call_count
        avg = self._log.total_cost / calls if calls and self._log.total_cost > 0 else None
        return BudgetStatus(
            total=total,
            spent=spent,
            remaining=remaining,
            utilization=self._tracker.utilization,
            calls=calls,
            avg_call_cost=avg,
            calls_left=int(remaining // avg) if avg else None,
            winding_down=self._winding_down(),
        )

    def _winding_down(self) -> bool:
        """The final call was claimed, and the budget hasn't been reset since."""
        return self._final_started and self._tracker.spent >= self._final_mark

    def _wrap_up_instruction(self) -> str:
        return self._wind_down.instruction if self._wind_down is not None else DEFAULT_WRAP_UP

    def _claim_final(self) -> bool:
        """Make this call the final one, or raise if the final call was already made."""
        with self._final_lock:
            if self._final_started:
                raise BudgetExhausted(
                    "Wind-down: the final call was already made. Its answer is in .partial_result; "
                    "no network call was made.",
                    remaining=self.remaining,
                    partial_result=self._final_answer,
                )
            self._final_started = True
            # Above zero, so a reset is noticed even when the final call came first.
            self._final_mark = max(self._tracker.spent, 1e-12)
        return True

    def _prepare_messages(self, task: str, messages: Optional[List[dict]]) -> tuple:
        """
        The call's messages, with the budget status line and/or the wrap-up
        instruction added. Returns (messages, is_final_call).
        """
        final = False
        if self._final_started and not self._winding_down():
            with self._final_lock:  # the budget was reset (or a new window started)
                self._final_started, self._final_answer = False, None
        if self._final_started:
            self._claim_final()  # raises: the final call was already made
        if self._wind_down is not None and self._wind_down.due(self.budget_status()):
            final = self._claim_final()
        msgs = self._build_messages(task, messages)
        notes = []
        if self._budget_awareness:
            status = self.budget_status()
            notes.append(self._budget_awareness(status) if callable(self._budget_awareness) else status.note())
        if final:
            notes.append(self._wrap_up_instruction())
        if notes:
            msgs = add_note(msgs, "\n".join(notes))
        return msgs, final

    def _build_messages(self, task: str, messages: Optional[List[dict]] = None) -> list:
        if messages is not None and not messages:
            raise ValueError("messages must not be empty when provided")
        if messages is not None:
            # Caller provided full history — prepend system prompt if not already present.
            if self.system_prompt and messages[0].get("role") != "system":
                return [{"role": "system", "content": self.system_prompt}] + list(messages)
            return list(messages)
        base = []
        if self.system_prompt:
            base.append({"role": "system", "content": self.system_prompt})
        base.append({"role": "user", "content": task})
        return base

    def _run_value_gate(
        self,
        task: str,
        decision,
        model_to_use: str,
        tc,
        expected_tokens: Callable[[str], int],
        afford: Callable[[str], None],
        estimated_value: float,
    ) -> tuple:
        """
        Shared value gate logic called by both sync and async paths.
        Value is compared with the *expected* cost (``expected_tokens(model)``):
        the worst case a mode reserves is for affordability, not for deciding
        whether a task is worth running. ``afford(model)`` raises BudgetExceeded
        if a downshifted model can't cover its planned output.
        Returns (decision, model_to_use). Raises TaskRejected if the task is rejected.
        """
        prompt_tokens = tc(model_to_use)
        estimated_cost = self._tracker.estimate_cost(model_to_use, prompt_tokens, expected_tokens(model_to_use))

        # Warn on early samples if value scale looks wrong for USD.
        if self._value_gate_samples < 5:
            if estimated_value > 100:
                warnings.warn(
                    f"value_fn returned {estimated_value:.4f}, which is unusually large for "
                    "a USD cost estimate. If returning a 0-100 score, convert to USD first.",
                    UserWarning,
                    stacklevel=4,
                )
            elif estimated_value > 0 and estimated_value < 1e-6:
                warnings.warn(
                    f"value_fn returned {estimated_value:.2e}, which looks like a 0-1 "
                    "normalized score rather than a USD amount. The value gate compares "
                    "value to estimated LLM cost in USD — a 0-1 score will almost always "
                    "be rejected. Convert to USD before using BAAR's value gate.",
                    UserWarning,
                    stacklevel=4,
                )

        decision = replace(
            decision,
            estimated_value=round(estimated_value, 8),
            estimated_cost_usd=round(estimated_cost, 8),
        )
        with self._value_gate_lock:
            self._value_gate_samples += 1
            _samples_snap = self._value_gate_samples
            _rejects_snap = self._value_gate_rejects
        reject_bar = estimated_cost * self._value_reject_alpha

        if estimated_value < reject_bar:
            projected_ratio = (_rejects_snap + 1) / max(1, _samples_snap)
            clamp_active = (
                _samples_snap >= self._min_samples_for_reject_clamp
                and projected_ratio > self._max_reject_rate
            )

            if clamp_active:
                decision = replace(
                    decision,
                    tier=ModelTier.SMALL,
                    model=self.small_model,
                    forced_by_budget=False,
                    reason=(
                        f"[REJECT CLAMP→SMALL] estimated_value {estimated_value:.8f} "
                        f"< reject_bar {reject_bar:.8f} "
                        f"(projected reject_rate={projected_ratio:.2f} > {self._max_reject_rate:.2f})"
                    ),
                    estimated_value=round(estimated_value, 8),
                    estimated_cost_usd=round(estimated_cost, 8),
                )
                model_to_use = self.small_model
                afford(model_to_use)
            else:
                with self._value_gate_lock:
                    self._value_gate_rejects += 1
                with self._step_counter_lock:
                    self._step_counter += 1
                    step_num = self._step_counter
                reject_decision = replace(
                    decision,
                    tier=ModelTier.REJECT,
                    model="",
                    reason=(
                        f"[VALUE REJECT] estimated_value {estimated_value:.8f} "
                        f"< reject_bar {reject_bar:.8f} (alpha={self._value_reject_alpha:.2f})"
                    ),
                )
                step_result = StepResult(
                    step_num=step_num,
                    task=task,
                    decision=reject_decision,
                    response_text="",
                    cost=0.0,
                    cumulative_cost=self._tracker.spent,
                    prompt_tokens=0,
                    completion_tokens=0,
                    latency_ms=0.0,
                    attempted_models=[],
                    failover_count=0,
                    failover_errors=[],
                )
                self._log.add(step_result)
                self._append_telemetry(step_result)
                raise TaskRejected(
                    reject_decision.reason,
                    estimated_value=estimated_value,
                    estimated_cost_usd=estimated_cost,
                    task=task,
                )

        elif estimated_value < estimated_cost:
            if decision.tier == ModelTier.BIG:
                decision = self._router.force_small(
                    decision,
                    f"value {estimated_value:.8f} < cost {estimated_cost:.8f}; downshift to small",
                )
                decision = replace(
                    decision,
                    estimated_value=round(estimated_value, 8),
                    estimated_cost_usd=round(estimated_cost, 8),
                )
                model_to_use = self.small_model
                afford(model_to_use)

        return decision, model_to_use

    def _apply_value_gate(
        self,
        task: str,
        decision,
        model_to_use: str,
        tc,
        expected_tokens: Callable[[str], int],
        afford: Callable[[str], None],
    ) -> tuple:
        """
        Apply the value gate if value_fn is configured (sync path).
        Returns (decision, model_to_use). Raises TaskRejected if the task is rejected.
        """
        if self._value_fn is None:
            return decision, model_to_use
        estimated_value = float(self._value_fn(task))
        return self._run_value_gate(task, decision, model_to_use, tc, expected_tokens, afford, estimated_value)

    async def _aapply_value_gate(
        self,
        task: str,
        decision,
        model_to_use: str,
        tc,
        expected_tokens: Callable[[str], int],
        afford: Callable[[str], None],
    ) -> tuple:
        """
        Apply the value gate if value_fn or async_value_fn is configured (async path).
        Uses async_value_fn if set, otherwise falls back to value_fn.
        Returns (decision, model_to_use). Raises TaskRejected if the task is rejected.
        """
        if self._async_value_fn is not None:
            estimated_value = float(await self._async_value_fn(task))
        elif self._value_fn is not None:
            estimated_value = float(self._value_fn(task))
        else:
            return decision, model_to_use
        return self._run_value_gate(task, decision, model_to_use, tc, expected_tokens, afford, estimated_value)

    def _resolve_mode(self, on_output_overrun: Optional[str]) -> str:
        mode = on_output_overrun if on_output_overrun is not None else self._on_output_overrun
        caps.validate_mode(mode)
        return mode

    # ── "continue": an answer in segments ─────────────────────────────────────
    # Each segment is a "truncate" call capped at segment_tokens, so it reserves
    # only its own cost. A segment that stops at its limit is continued with the
    # text so far; the answer ends when the model finishes, the user's
    # max_output_tokens is reached, max_segments is hit, or the budget can't pay
    # for another useful segment (then the last step is flagged truncated_by_budget).

    def _segment_limit(self, limit: Optional[int], used: int) -> int:
        return self._segment_tokens if limit is None else min(self._segment_tokens, limit - used)

    @staticmethod
    def _segment_messages(task: str, messages: Optional[List[dict]], so_far: str) -> Optional[List[dict]]:
        if not so_far:
            return messages
        base = list(messages) if messages is not None else [{"role": "user", "content": task}]
        return base + [{"role": "assistant", "content": so_far}, {"role": "user", "content": _CONTINUE_PROMPT}]

    @staticmethod
    def _segment_done(step: StepResult, limit: Optional[int], used: int) -> bool:
        return step.finish_reason != "length" or (limit is not None and used >= limit)

    def _chat_continue(self, task, messages, context, max_output_tokens, thinking_budget) -> str:
        _, limit, thinking = self._resolve_output_opts("continue", max_output_tokens, thinking_budget)
        parts: List[str] = []
        used, last = 0, None
        for _ in range(self._max_segments):
            try:
                last = self._chat_step(
                    task, messages=self._segment_messages(task, messages, "".join(parts)), context=context,
                    on_output_overrun="truncate", max_output_tokens=self._segment_limit(limit, used),
                    thinking_budget=thinking,
                )
            except _SEGMENT_STOPS as e:
                if last is None:
                    raise
                _flag_budget_cut(last, e)
                break
            parts.append(last.response_text)
            used += last.completion_tokens
            if self._segment_done(last, limit, used):
                break
        return "".join(parts)

    async def _achat_continue(self, task, messages, context, max_output_tokens, thinking_budget) -> str:
        _, limit, thinking = self._resolve_output_opts("continue", max_output_tokens, thinking_budget)
        parts: List[str] = []
        used, last = 0, None
        for _ in range(self._max_segments):
            try:
                last = await self._achat_step(
                    task, messages=self._segment_messages(task, messages, "".join(parts)), context=context,
                    on_output_overrun="truncate", max_output_tokens=self._segment_limit(limit, used),
                    thinking_budget=thinking,
                )
            except _SEGMENT_STOPS as e:
                if last is None:
                    raise
                _flag_budget_cut(last, e)
                break
            parts.append(last.response_text)
            used += last.completion_tokens
            if self._segment_done(last, limit, used):
                break
        return "".join(parts)

    def _stream_continue(self, task, messages, context, max_output_tokens, thinking_budget) -> Iterator[str]:
        _, limit, thinking = self._resolve_output_opts("continue", max_output_tokens, thinking_budget)
        parts: List[str] = []
        used, last = 0, None
        for _ in range(self._max_segments):
            sink: list = []
            try:
                for delta in self._stream_step(
                    task, messages=self._segment_messages(task, messages, "".join(parts)), context=context,
                    on_output_overrun="truncate", max_output_tokens=self._segment_limit(limit, used),
                    thinking_budget=thinking, step_sink=sink,
                ):
                    parts.append(delta)
                    yield delta
            except _SEGMENT_STOPS as e:
                if last is None:
                    raise
                _flag_budget_cut(last, e)
                return
            if not sink:
                return
            last = sink[0]
            used += last.completion_tokens
            if self._segment_done(last, limit, used):
                return

    async def _astream_continue(self, task, messages, context, max_output_tokens, thinking_budget):
        _, limit, thinking = self._resolve_output_opts("continue", max_output_tokens, thinking_budget)
        parts: List[str] = []
        used, last = 0, None
        for _ in range(self._max_segments):
            sink: list = []
            segment = self._astream_step(
                task, messages=self._segment_messages(task, messages, "".join(parts)), context=context,
                on_output_overrun="truncate", max_output_tokens=self._segment_limit(limit, used),
                thinking_budget=thinking, step_sink=sink,
            )
            try:
                async for delta in segment:
                    parts.append(delta)
                    yield delta
            except _SEGMENT_STOPS as e:
                if last is None:
                    raise
                _flag_budget_cut(last, e)
                return
            finally:
                await segment.aclose()
            if not sink:
                return
            last = sink[0]
            used += last.completion_tokens
            if self._segment_done(last, limit, used):
                return

    def _resolve_output_opts(
        self,
        on_output_overrun: Optional[str],
        max_output_tokens: Optional[int],
        thinking_budget: Optional[int] = None,
    ) -> tuple:
        """Per-call overrides win over router config. Returns (mode, max_output_tokens, thinking_budget)."""
        mode = on_output_overrun if on_output_overrun is not None else self._on_output_overrun
        caps.validate_mode(mode)
        limit = max_output_tokens if max_output_tokens is not None else self._max_output_tokens
        if limit is not None and limit < 1:
            raise ValueError("max_output_tokens must be a positive integer or None")
        thinking = thinking_budget if thinking_budget is not None else self._thinking_budget
        _validate_thinking_budget(thinking)
        return mode, limit, thinking

    def _reserve(self, amount: float) -> None:
        """Reserve ``amount``; losing it to a concurrent call is a kill-switch refusal like any other."""
        try:
            self._tracker.check_and_reserve(amount)
        except BudgetExceeded as e:
            raise BudgetExhausted(
                "Kill-switch activated: a concurrent call reserved the remaining budget first "
                f"(${self.remaining:.6f} left). Request rejected locally with zero network calls.",
                remaining=self.remaining,
            ) from e

    def _warn_if_unpriced(self, model: str, mode: str) -> None:
        """preflight / truncate / continue can't bound spend on a model LiteLLM has no prices for."""
        if mode == "estimate" or model in self._unpriced_warned:
            return
        if self._tracker.output_price_per_token(model) is not None:
            return
        self._unpriced_warned.add(model)
        warnings.warn(
            f"on_output_overrun='{mode}' can't enforce the budget for '{model}': LiteLLM has no pricing "
            "for it, so its calls are reserved at a placeholder cost. Register real prices with "
            "litellm.register_model() to make the guarantee hold.",
            UserWarning,
            stacklevel=4,
        )

    def _plan_output(
        self,
        model: str,
        prompt_tokens: int,
        eot: int,
        mode: str,
        limit: Optional[int],
        thinking: Optional[int] = None,
        remaining: Optional[float] = None,
    ) -> caps.OutputPlan:
        self._warn_if_unpriced(model, mode)
        return caps.plan_output(
            mode,
            model=model,
            prompt_tokens=prompt_tokens,
            expected_output_tokens=eot,
            remaining=self._tracker.remaining if remaining is None else remaining,
            tracker=self._tracker,
            max_output_tokens=limit,
            min_useful_tokens=self._min_useful_tokens,
            max_call_share=self._max_call_share,
            prompt_margin=self._prompt_margin,
            thinking_budget=thinking,
            predicted_tokens=(
                self._output_lengths.quantile(model, self._overrun_risk)
                if mode == "preflight" and self._overrun_risk > 0 else None
            ),
        )

    def _expected_tokens(
        self, tc, eot: int, limit: Optional[int], thinking: Optional[int] = None
    ) -> Callable[[str], int]:
        """model -> expected completion tokens (the router's estimate, capped by the user's limit)."""
        return lambda model: self._plan_output(model, tc(model), eot, "estimate", limit, thinking).completion_tokens

    def _affordability_check(
        self, tc, eot: int, mode: str, limit: Optional[int], thinking: Optional[int] = None
    ) -> Callable[[str], None]:
        """model -> raises BudgetExceeded if ``model`` can't cover its planned output."""
        return lambda model: self._check_output_affordable(model, tc(model), eot, mode, limit, thinking)

    def _check_output_affordable(
        self,
        model: str,
        prompt_tokens: int,
        eot: int,
        mode: str,
        limit: Optional[int],
        thinking: Optional[int] = None,
        uncut: bool = False,
    ) -> None:
        """
        Raise BudgetExceeded if ``model`` can't cover its planned output.
        With ``uncut=True``, "truncate" mode checks the full response instead of
        a budget-shortened one.
        """
        if uncut and mode == "truncate":
            tokens = caps.uncut_output_tokens(model, prompt_tokens, eot, limit, thinking)
            priced = caps.priced_prompt_tokens(mode, prompt_tokens, self._prompt_margin)
        else:
            plan = self._plan_output(model, prompt_tokens, eot, mode, limit, thinking)
            tokens, priced = plan.completion_tokens, plan.prompt_tokens
        self._tracker.check_affordability(model, priced, tokens)

    def _warn_no_stream_usage(self, model: str) -> None:
        if model in self._no_stream_usage_warned:
            return
        self._no_stream_usage_warned.add(model)
        warnings.warn(
            f"Stream for model '{model}' reported no usage; recording visible text tokens only. "
            "Hidden reasoning tokens (if any) are not counted, so spend may be under-recorded.",
            UserWarning,
            stacklevel=3,
        )

    def _cover_fallback(
        self, candidate: str, tc, eot: int, mode: str, limit: Optional[int], thinking: Optional[int], reserved: float
    ) -> tuple:
        """
        Plan a fallback candidate and top up the reservation if it costs more
        than what is already held. Returns (plan, new_reserved).
        Raises BudgetExceeded if the extra amount can't be reserved.
        """
        # Our own reservation is already counted as spent, so it is available to this call.
        plan = self._plan_output(
            candidate, tc(candidate), eot, mode, limit, thinking,
            remaining=self._tracker.remaining + reserved,
        )
        needed = self._tracker.estimate_cost(candidate, plan.prompt_tokens, plan.completion_tokens)
        if needed > reserved:
            self._tracker.check_and_reserve(needed - reserved)
            reserved = needed
        return plan, reserved

    def _execution_model_candidates(self, tier: ModelTier, chosen_model: str) -> List[str]:
        extras = self._big_fallback_models if tier == ModelTier.BIG else self._small_fallback_models
        return [chosen_model] + [m for m in extras if m and m != chosen_model]

    def _append_telemetry(self, step: StepResult) -> None:
        if not self._telemetry_jsonl_path:
            return
        parent = os.path.dirname(self._telemetry_jsonl_path)
        if parent:
            try:
                os.makedirs(parent, exist_ok=True)
            except OSError:
                return
        payload = step.to_dict()
        payload["timestamp_unix_ms"] = int(time.time() * 1000)
        payload["router_budget_usd"] = self.budget
        lock = _telemetry_lock(self._telemetry_jsonl_path)
        try:
            with lock:
                with open(self._telemetry_jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(payload, ensure_ascii=True) + "\n")
        except OSError:
            pass
