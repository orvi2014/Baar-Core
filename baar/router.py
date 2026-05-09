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
from baar.core.budget import BudgetTracker, BudgetExceeded
from baar.core.exceptions import TaskRejected, BudgetExhausted
from baar.core.models import StepResult, RoutingLog
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
except (ImportError, AttributeError):
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
) -> object:
    """Synchronous LiteLLM completion with exponential backoff on transient errors."""
    delay = 1.0
    kwargs = dict(model=model, messages=messages, stream=stream)
    if timeout is not None:
        kwargs["timeout"] = timeout
    for attempt in range(max_retries + 1):
        try:
            return litellm.completion(**kwargs)
        except Exception as exc:
            if attempt < max_retries and _is_retryable(exc):
                time.sleep(delay * (1.0 + random.uniform(-0.1, 0.3)))
                delay = min(delay * 2, 30.0)
            else:
                raise


async def _async_completion_with_retry(
    model: str,
    messages: list,
    max_retries: int = 2,
    stream: bool = False,
    timeout: Optional[float] = None,
) -> object:
    """Async LiteLLM completion with exponential backoff on transient errors."""
    delay = 1.0
    kwargs = dict(model=model, messages=messages, stream=stream)
    if timeout is not None:
        kwargs["timeout"] = timeout
    for attempt in range(max_retries + 1):
        try:
            return await litellm.acompletion(**kwargs)
        except Exception as exc:
            if attempt < max_retries and _is_retryable(exc):
                await asyncio.sleep(delay * (1.0 + random.uniform(-0.1, 0.3)))
                delay = min(delay * 2, 30.0)
            else:
                raise


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
        ))

    def _configure(self, config: BAARConfig) -> None:
        """Single-source initialization from a BAARConfig. Called by __init__ and from_config."""
        if hasattr(self, '_config'):
            raise RuntimeError("BAARRouter is already configured. Create a new instance to change configuration.")
        if config.budget <= 0:
            raise ValueError("Budget must be positive")
        if config.min_cost_threshold < 0:
            raise ValueError("min_cost_threshold must be non-negative")

        self.small_model = config.small_model
        self.big_model = config.big_model
        self.system_prompt = config.system_prompt
        self._min_cost_threshold = float(config.min_cost_threshold)
        self._expected_output_tokens = max(1, int(config.expected_output_tokens))
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

        self._tracker = BudgetTracker(total_budget=config.budget, store=config.store)
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
    def config(self) -> BAARConfig:
        return self._config

    # ── Sync execution ─────────────────────────────────────────────────────────

    def chat(self, task: str, *, messages: Optional[List[dict]] = None) -> str:
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
        msgs = self._build_messages(task, messages)

        # Cache token counts — tiktoken is O(n); same text must not be re-tokenized
        # once per check. Lazy dict so fallback models are only counted if used.
        _tc_cache: dict[str, int] = {}

        def tc(model: str) -> int:
            if model not in _tc_cache:
                _tc_cache[model] = token_counter(messages=msgs, model=model)
            return _tc_cache[model]

        eot = self._expected_output_tokens

        # ── 0. Hard Kill-Switch Pre-check ─────────────────────────────────────
        prompt_tokens_est = tc(self.small_model)
        estimated_small_floor = self._tracker.estimate_cost(
            self.small_model, prompt_tokens_est, eot
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
            self._tracker.check_affordability(self.small_model, prompt_tokens_est, eot)
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

        # ── 3. BIG affordability check — downgrade if necessary ───────────────
        if decision.tier == ModelTier.BIG:
            try:
                self._tracker.check_affordability(self.big_model, tc(self.big_model), eot)
            except BudgetExceeded as e:
                decision = self._router.force_small(decision, str(e))

        # Final pre-flight check for the chosen model
        model_to_use = decision.model
        self._tracker.check_affordability(model_to_use, tc(model_to_use), eot)

        # ── 3b. Value gate ────────────────────────────────────────────────────
        decision, model_to_use = self._apply_value_gate(task, decision, model_to_use, tc, eot)

        # ── 4. Execution with concurrency-safe budget reservation ─────────────
        # Initialise locals before reservation so a KeyboardInterrupt in the
        # gap between check_and_reserve and the try: cannot leak the reservation
        # into a persistent store (mirrors the structure of achat()).
        start_time = time.perf_counter()
        response = None
        attempted_models: List[str] = []
        failover_errors: List[str] = []
        reserved = self._tracker.estimate_cost(model_to_use, tc(model_to_use), eot)
        with self._sync_budget_lock:
            self._tracker.check_and_reserve(reserved)
        with self._step_counter_lock:
            self._step_counter += 1
            step_num = self._step_counter

        try:
            for i, candidate in enumerate(self._execution_model_candidates(decision.tier, model_to_use)):
                attempted_models.append(candidate)
                if i > 0:
                    try:
                        self._tracker.check_affordability(candidate, tc(candidate), eot)
                    except BudgetExceeded as e:
                        failover_errors.append(f"{candidate}: affordability failed ({e})")
                        continue
                try:
                    response = _sync_completion_with_retry(
                        candidate, msgs, timeout=self._call_timeout
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
        )
        self._log.add(step_result)
        self._append_telemetry(step_result)
        return step_result.response_text

    def stream_chat(
        self,
        task: str,
        *,
        messages: Optional[List[dict]] = None,
    ) -> Iterator[str]:
        """
        Streaming version of chat() — yields text chunks as they arrive.

        Budget guardrails are identical to chat(): preflight checks run before
        the first token, and the reservation is always cleaned up via try/finally.
        Token counts are accumulated during streaming and recorded afterward.
        """
        # Build messages first so token counting includes full history
        msgs = self._build_messages(task, messages)

        _tc_cache: dict[str, int] = {}

        def tc(model: str) -> int:
            if model not in _tc_cache:
                _tc_cache[model] = token_counter(messages=msgs, model=model)
            return _tc_cache[model]

        eot = self._expected_output_tokens

        # ── 0. Kill-switch pre-check ───────────────────────────────────────────
        prompt_tokens_est = tc(self.small_model)
        estimated_small_floor = self._tracker.estimate_cost(
            self.small_model, prompt_tokens_est, eot
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
            self._tracker.check_affordability(self.small_model, prompt_tokens_est, eot)
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

        if decision.tier == ModelTier.BIG:
            try:
                self._tracker.check_affordability(self.big_model, tc(self.big_model), eot)
            except BudgetExceeded as e:
                decision = self._router.force_small(decision, str(e))

        model_to_use = decision.model
        self._tracker.check_affordability(model_to_use, tc(model_to_use), eot)
        decision, model_to_use = self._apply_value_gate(task, decision, model_to_use, tc, eot)

        # ── 2. Reserve budget atomically, then find a working stream ──────────
        # Locals are initialised before reservation so a KeyboardInterrupt in
        # the gap cannot leak the reservation into a persistent store.
        attempted_models: List[str] = []
        failover_errors: List[str] = []
        response_stream = None
        model_to_use_final = model_to_use
        reserved = self._tracker.estimate_cost(model_to_use, tc(model_to_use), eot)
        with self._sync_budget_lock:
            self._tracker.check_and_reserve(reserved)
        with self._step_counter_lock:
            self._step_counter += 1
            step_num = self._step_counter

        try:
            for i, candidate in enumerate(self._execution_model_candidates(decision.tier, model_to_use)):
                attempted_models.append(candidate)
                if i > 0:
                    try:
                        self._tracker.check_affordability(candidate, tc(candidate), eot)
                    except BudgetExceeded as e:
                        failover_errors.append(f"{candidate}: affordability failed ({e})")
                        continue
                try:
                    response_stream = _sync_completion_with_retry(
                        candidate, msgs, stream=True, timeout=self._call_timeout
                    )
                    model_to_use_final = candidate
                    break
                except Exception as e:
                    failover_errors.append(
                        f"{candidate}: completion failed ({type(e).__name__}: {e})"
                    )
        except BaseException:
            if response_stream is None:
                self._tracker.cancel_reservation(reserved)
            raise

        if response_stream is None:
            self._tracker.cancel_reservation(reserved)
            failures = "; ".join(failover_errors) or "no candidate model succeeded"
            raise RuntimeError(f"All provider/model attempts failed: {failures}")

        # ── 3. Stream, then record ─────────────────────────────────────────────
        parts: list[str] = []
        start_time = time.perf_counter()
        prompt_tokens_for_record = 0  # set inside try; default protects finally if interrupted early

        try:
            if model_to_use_final != decision.model:
                decision = replace(
                    decision,
                    model=model_to_use_final,
                    reason=f"{decision.reason} [FAILOVER→{model_to_use_final}]",
                )
            prompt_tokens_for_record = tc(model_to_use_final)
            for chunk in response_stream:
                delta = chunk.choices[0].delta.content or ""
                parts.append(delta)
                yield delta
        finally:
            full_text = "".join(parts)
            latency_ms = (time.perf_counter() - start_time) * 1000
            completion_tokens = token_counter(full_text, model=model_to_use_final)
            try:
                # Record actual cost BEFORE cancelling the reservation so that
                # concurrent threads never see a transient under-count window
                # (spent would drop then rise, allowing over-allocation).
                record = self._tracker.record_manual(
                    model_to_use_final, prompt_tokens_for_record, completion_tokens
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
                )
                self._log.add(step_result)
                self._append_telemetry(step_result)
            except Exception as _e:
                self._log.errors.append(f"post-stream recording failed: {type(_e).__name__}: {_e}")
            finally:
                self._tracker.cancel_reservation(reserved)

    def run(self, tasks: List[str]) -> RoutingLog:
        """
        Execute a series of tasks sequentially.
        Stops early if the hard budget cap is hit or max_consecutive_errors reached.
        Unexpected errors are recorded in log.errors rather than silently dropped.
        """
        consecutive_errors = 0
        for task in tasks:
            try:
                self.chat(task)
                consecutive_errors = 0
            except (BudgetExceeded, BudgetExhausted):
                break
            except TaskRejected:
                consecutive_errors = 0
                continue
            except Exception as e:
                self._log.errors.append(f"{type(e).__name__}: {e}")
                consecutive_errors += 1
                if consecutive_errors >= self._max_consecutive_errors:
                    break
        return self._log

    # ── Async execution ────────────────────────────────────────────────────────

    async def achat(self, task: str, *, messages: Optional[List[dict]] = None) -> str:
        """
        Async version of chat() — identical hard financial guardrails,
        non-blocking for asyncio / FastAPI / aiohttp applications.

        Concurrency safety: a per-instance asyncio.Lock atomically reserves the
        estimated cost before the await, preventing concurrent callers from both
        passing the affordability check and together overshooting the budget.
        """
        # Build messages first so token counting includes full history
        msgs = self._build_messages(task, messages)

        _tc_cache: dict[str, int] = {}

        def tc(model: str) -> int:
            if model not in _tc_cache:
                _tc_cache[model] = token_counter(messages=msgs, model=model)
            return _tc_cache[model]

        eot = self._expected_output_tokens

        # ── 0. Hard Kill-Switch Pre-check ─────────────────────────────────────
        prompt_tokens_est = tc(self.small_model)
        estimated_small_floor = self._tracker.estimate_cost(
            self.small_model, prompt_tokens_est, eot
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
            self._tracker.check_affordability(self.small_model, prompt_tokens_est, eot)
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

        # ── 3. BIG affordability check — downgrade if necessary ───────────────
        if decision.tier == ModelTier.BIG:
            try:
                self._tracker.check_affordability(self.big_model, tc(self.big_model), eot)
            except BudgetExceeded as e:
                decision = self._router.force_small(decision, str(e))

        model_to_use = decision.model
        self._tracker.check_affordability(model_to_use, tc(model_to_use), eot)

        # ── 3b. Value gate (same logic as chat(), fully ported) ───────────────
        decision, model_to_use = await self._aapply_value_gate(task, decision, model_to_use, tc, eot)

        # ── 4. Execution with concurrency-safe budget reservation ─────────────
        start_time = time.perf_counter()
        response = None
        attempted_models: List[str] = []
        failover_errors: List[str] = []

        # Atomically check + reserve to prevent concurrent overspend.
        async with self._budget_async_lock:
            reserved = self._tracker.estimate_cost(model_to_use, tc(model_to_use), eot)
            self._tracker.check_and_reserve(reserved)
        with self._step_counter_lock:
            self._step_counter += 1
            step_num = self._step_counter

        # Use try/finally (not except) so BaseException (KeyboardInterrupt, CancelledError etc.)
        # also releases the reservation — prevents permanent budget under-reporting in persistent stores.
        try:
            for candidate in self._execution_model_candidates(decision.tier, model_to_use):
                attempted_models.append(candidate)
                # Do NOT re-check affordability here: we already have a reservation
                # that covers this call. A redundant check against the already-reduced
                # remaining would reject the primary candidate at low budget (BUG FIX).
                try:
                    response = await _async_completion_with_retry(
                        candidate, msgs, timeout=self._call_timeout
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
        )
        self._log.add(step_result)
        self._append_telemetry(step_result)
        return step_result.response_text

    async def astream_chat(
        self,
        task: str,
        *,
        messages: Optional[List[dict]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Async streaming version of achat() — yields text chunks as they arrive.

        Budget guardrails are identical to achat(). Reservation is held during
        streaming and cleaned up via try/finally.
        """
        # Build messages first so token counting includes full history
        msgs = self._build_messages(task, messages)

        _tc_cache: dict[str, int] = {}

        def tc(model: str) -> int:
            if model not in _tc_cache:
                _tc_cache[model] = token_counter(messages=msgs, model=model)
            return _tc_cache[model]

        eot = self._expected_output_tokens
        prompt_tokens_est = tc(self.small_model)
        estimated_small_floor = self._tracker.estimate_cost(
            self.small_model, prompt_tokens_est, eot
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
            self._tracker.check_affordability(self.small_model, prompt_tokens_est, eot)
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

        if decision.tier == ModelTier.BIG:
            try:
                self._tracker.check_affordability(self.big_model, tc(self.big_model), eot)
            except BudgetExceeded as e:
                decision = self._router.force_small(decision, str(e))

        model_to_use = decision.model
        self._tracker.check_affordability(model_to_use, tc(model_to_use), eot)
        decision, model_to_use = await self._aapply_value_gate(task, decision, model_to_use, tc, eot)

        # ── Reserve budget atomically, then find a working stream ─────────────
        # Reservation precedes the API call so concurrent coroutines cannot both
        # pass the affordability check and both open streams (wasted API calls).
        async with self._budget_async_lock:
            reserved = self._tracker.estimate_cost(model_to_use, tc(model_to_use), eot)
            self._tracker.check_and_reserve(reserved)
        with self._step_counter_lock:
            self._step_counter += 1
            step_num = self._step_counter

        attempted_models: List[str] = []
        failover_errors: List[str] = []
        response_stream = None
        model_to_use_final = model_to_use

        try:
            for i, candidate in enumerate(self._execution_model_candidates(decision.tier, model_to_use)):
                attempted_models.append(candidate)
                if i > 0:
                    try:
                        self._tracker.check_affordability(candidate, tc(candidate), eot)
                    except BudgetExceeded as e:
                        failover_errors.append(f"{candidate}: affordability failed ({e})")
                        continue
                try:
                    response_stream = await _async_completion_with_retry(
                        candidate, msgs, stream=True, timeout=self._call_timeout
                    )
                    model_to_use_final = candidate
                    break
                except Exception as e:
                    failover_errors.append(
                        f"{candidate}: completion failed ({type(e).__name__}: {e})"
                    )
        except BaseException:
            if response_stream is None:
                self._tracker.cancel_reservation(reserved)
            raise

        if response_stream is None:
            self._tracker.cancel_reservation(reserved)
            failures = "; ".join(failover_errors) or "no candidate model succeeded"
            raise RuntimeError(f"All provider/model attempts failed: {failures}")

        parts: list[str] = []
        start_time = time.perf_counter()
        prompt_tokens_for_record = 0  # set inside try; default protects finally if interrupted early

        try:
            if model_to_use_final != decision.model:
                decision = replace(
                    decision,
                    model=model_to_use_final,
                    reason=f"{decision.reason} [FAILOVER→{model_to_use_final}]",
                )
            prompt_tokens_for_record = tc(model_to_use_final)
            async for chunk in response_stream:
                delta = chunk.choices[0].delta.content or ""
                parts.append(delta)
                yield delta
        finally:
            full_text = "".join(parts)
            latency_ms = (time.perf_counter() - start_time) * 1000
            completion_tokens = token_counter(full_text, model=model_to_use_final)
            try:
                record = self._tracker.record_manual(
                    model_to_use_final, prompt_tokens_for_record, completion_tokens
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
                )
                self._log.add(step_result)
                self._append_telemetry(step_result)
            except Exception as _e:
                self._log.errors.append(f"post-stream recording failed: {type(_e).__name__}: {_e}")
            finally:
                self._tracker.cancel_reservation(reserved)

    async def arun(self, tasks: List[str]) -> RoutingLog:
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
                    await self.achat(task)
                    consecutive_errors = 0
                except (BudgetExceeded, BudgetExhausted):
                    break
                except TaskRejected:
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
                    if budget_stop.is_set():
                        return
                    if err_count[0] >= self._max_consecutive_errors:
                        return
                    try:
                        await self.achat(task)
                        err_count[0] = 0
                    except (BudgetExceeded, BudgetExhausted):
                        budget_stop.set()
                    except TaskRejected:
                        err_count[0] = 0
                    except Exception as e:
                        self._log.errors.append(f"{type(e).__name__}: {e}")
                        err_count[0] += 1

            await asyncio.gather(*[_one(t) for t in tasks])
        return self._log

    # ── Internal helpers ───────────────────────────────────────────────────────

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
        eot: int,
        estimated_value: float,
    ) -> tuple:
        """
        Shared value gate logic called by both sync and async paths.
        Returns (decision, model_to_use). Raises TaskRejected if the task is rejected.
        """
        prompt_tokens = tc(model_to_use)
        estimated_cost = self._tracker.estimate_cost(model_to_use, prompt_tokens, eot)

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
                self._tracker.check_affordability(model_to_use, tc(model_to_use), eot)
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
                self._tracker.check_affordability(model_to_use, tc(model_to_use), eot)

        return decision, model_to_use

    def _apply_value_gate(
        self,
        task: str,
        decision,
        model_to_use: str,
        tc,
        eot: int,
    ) -> tuple:
        """
        Apply the value gate if value_fn is configured (sync path).
        Returns (decision, model_to_use). Raises TaskRejected if the task is rejected.
        """
        if self._value_fn is None:
            return decision, model_to_use
        estimated_value = float(self._value_fn(task))
        return self._run_value_gate(task, decision, model_to_use, tc, eot, estimated_value)

    async def _aapply_value_gate(
        self,
        task: str,
        decision,
        model_to_use: str,
        tc,
        eot: int,
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
        return self._run_value_gate(task, decision, model_to_use, tc, eot, estimated_value)

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
        with lock:
            with open(self._telemetry_jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=True) + "\n")
