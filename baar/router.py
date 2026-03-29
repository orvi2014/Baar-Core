"""
baar/router.py — Main BAARRouter orchestration class.
Handles the end-to-end flow: routing, budget constraints, and execution.
"""

import json
import os
import time
import litellm
from dataclasses import replace
from typing import Callable, List, Optional

from baar.core.router import Router, ModelTier
from baar.core.budget import BudgetTracker, BudgetExceeded
from baar.core.exceptions import TaskRejected
from baar.core.models import StepResult, RoutingLog


def token_counter(text: str, model: str = "gpt-4o") -> int:
    """
    Estimate token count for a task. 
    Used for pre-flight budget checks (BCD).
    """
    try:
        # LiteLLM's token counter is quite accurate for most models
        return litellm.token_counter(model=model, text=text)
    except Exception:
        # Fallback to rough estimate if litellm or tiktoken fails
        return len(text) // 4


import importlib.metadata

def _check_litellm_version():
    try:
        version = importlib.metadata.version("litellm")
    except importlib.metadata.PackageNotFoundError:
        return

    bad_versions = ["1.82.7", "1.82.8"]
    if version in bad_versions:
        raise RuntimeError(
            f"Insecure LiteLLM version detected: {version}. "
            "Please downgrade to a safe version."
        )


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
        value_reject_alpha: float = 0.3,
        max_reject_rate: float = 0.5,
        min_samples_for_reject_clamp: int = 20,
        small_fallback_models: Optional[List[str]] = None,
        big_fallback_models: Optional[List[str]] = None,
        telemetry_jsonl_path: Optional[str] = None,
        routing_cache_size: int = 256,
        routing_cache_enabled: bool = True,
        small_exploration_rate: float = 0.0,
    ):
        _check_litellm_version()
        # Ensure clean output for users by disabling litellm's internal logging
        litellm.set_verbose = False
        litellm.suppress_debug_info = True

        if budget <= 0:
            raise ValueError("Budget must be positive")

        self.small_model = small_model
        self.big_model = big_model
        self.system_prompt = system_prompt

        self._tracker = BudgetTracker(total_budget=budget)
        self._router = Router(
            small_model=small_model,
            big_model=big_model,
            complexity_threshold=complexity_threshold,
            use_llm_router=use_llm_router,
            routing_cache_size=routing_cache_size,
            routing_cache_enabled=routing_cache_enabled,
            small_exploration_rate=small_exploration_rate,
        )
        self._log = RoutingLog(
            budget=budget,
            small_model=small_model,
            big_model=big_model,
        )
        self._value_fn = value_fn
        self._value_reject_alpha = max(0.0, float(value_reject_alpha))
        self._max_reject_rate = max(0.0, min(1.0, float(max_reject_rate)))
        self._min_samples_for_reject_clamp = max(1, int(min_samples_for_reject_clamp))
        self._value_gate_samples = 0
        self._value_gate_rejects = 0
        self._small_fallback_models = list(small_fallback_models or [])
        self._big_fallback_models = list(big_fallback_models or [])
        self._telemetry_jsonl_path = telemetry_jsonl_path

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

    def chat(self, task: str) -> str:
        """
        Execute a single routed chat call with hard financial guardrails.
        """
        # 0. Hard Kill-Switch Pre-check
        # If budget is already near zero, reject instantly before any LLM calls
        min_cost_threshold = 0.0001
        if self.remaining < min_cost_threshold:
            raise RuntimeError(
                "Kill-switch activated: "
                f"budget too low (${self.remaining:.6f}). "
                "Request rejected locally with zero network calls."
            )

        # 1. Fast Heuristic Pre-check (Zero-Call safety)
        # Check if we can afford even the cheapest model for this task
        try:
            prompt_tokens_est = token_counter(task, model=self.small_model)
            self._tracker.check_affordability(self.small_model, prompt_tokens_est)
        except BudgetExceeded as e:
            raise RuntimeError(
                "Kill-switch activated: "
                f"budget too low (${self.remaining:.6f}) for the cheapest safe call "
                f"on {self.small_model}. Request rejected locally with zero network calls."
            ) from e

        # 2. Routing Decision
        # This might involve an LLM call if use_llm_router=True
        decision = self._router.decide(
            task=task,
            remaining_budget=self.remaining,
            budget_utilization=self._tracker.utilization,
        )

        # 3. Budget Pressure / BCD (Pre-flight cost check)
        # If we chose BIG, check if we can actually afford it.
        if decision.tier == ModelTier.BIG:
            prompt_tokens = token_counter(task, model=self.big_model)
            try:
                self._tracker.check_affordability(self.big_model, prompt_tokens)
            except BudgetExceeded as e:
                # Downgrade to SMALL if BIG is too expensive
                decision = self._router.force_small(decision, str(e))

        # Final pre-flight check (ensure we can afford the chosen model)
        model_to_use = decision.model
        prompt_tokens = token_counter(task, model=model_to_use)
        self._tracker.check_affordability(model_to_use, prompt_tokens)

        # 3b. Optional value gate (same USD basis as estimate_cost)
        # Policy:
        #   value < cost * alpha  -> reject
        #   value < cost          -> downshift to SMALL
        #   value >= cost         -> keep routed choice
        if self._value_fn is not None:
            estimated_value = float(self._value_fn(task))
            estimated_cost = self._tracker.estimate_cost(model_to_use, prompt_tokens)
            decision = replace(
                decision,
                estimated_value=round(estimated_value, 8),
                estimated_cost_usd=round(estimated_cost, 8),
            )
            self._value_gate_samples += 1
            reject_bar = estimated_cost * self._value_reject_alpha

            if estimated_value < reject_bar:
                projected_ratio = (self._value_gate_rejects + 1) / max(1, self._value_gate_samples)
                clamp_active = (
                    self._value_gate_samples >= self._min_samples_for_reject_clamp
                    and projected_ratio > self._max_reject_rate
                )

                if clamp_active:
                    # Prevent collapse: convert reject into SMALL execution.
                    decision = self._router.force_small(
                        decision,
                        (
                            "reject clamp active "
                            f"(projected reject_rate={projected_ratio:.2f} > {self._max_reject_rate:.2f})"
                        ),
                    )
                    decision = replace(
                        decision,
                        reason=(
                            f"[REJECT CLAMP→SMALL] estimated_value {estimated_value:.8f} "
                            f"< reject_bar {reject_bar:.8f}"
                        ),
                        estimated_value=round(estimated_value, 8),
                        estimated_cost_usd=round(estimated_cost, 8),
                    )
                    model_to_use = self.small_model
                    prompt_tokens = token_counter(task, model=model_to_use)
                    self._tracker.check_affordability(model_to_use, prompt_tokens)
                else:
                    self._value_gate_rejects += 1
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
                        step_num=len(self._log.steps) + 1,
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
                # Medium zone: execute on SMALL instead of BIG.
                if decision.tier == ModelTier.BIG:
                    decision = self._router.force_small(
                        decision,
                        (
                            f"value {estimated_value:.8f} < cost {estimated_cost:.8f}; "
                            "downshift to small"
                        ),
                    )
                    decision = replace(
                        decision,
                        estimated_value=round(estimated_value, 8),
                        estimated_cost_usd=round(estimated_cost, 8),
                    )
                    model_to_use = self.small_model
                    prompt_tokens = token_counter(task, model=model_to_use)
                    self._tracker.check_affordability(model_to_use, prompt_tokens)

        # 4. Execution (with provider/model failover)
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": task})

        start_time = time.perf_counter()
        response = None
        attempted_models: List[str] = []
        failover_errors: List[str] = []
        execution_models = self._execution_model_candidates(decision.tier, model_to_use)
        if not execution_models:
            execution_models = [model_to_use]

        for candidate_model in execution_models:
            attempted_models.append(candidate_model)
            candidate_prompt_tokens = token_counter(task, model=candidate_model)
            try:
                self._tracker.check_affordability(candidate_model, candidate_prompt_tokens)
            except BudgetExceeded as e:
                failover_errors.append(f"{candidate_model}: affordability failed ({e})")
                continue

            try:
                # LiteLLM handles the underlying API calls (OpenAI, Anthropic, etc.)
                response = litellm.completion(
                    model=candidate_model,
                    messages=messages,
                )
                model_to_use = candidate_model
                break
            except Exception as e:
                failover_errors.append(f"{candidate_model}: completion failed ({type(e).__name__}: {e})")
                continue

        if response is None:
            failures = "; ".join(failover_errors) if failover_errors else "no candidate model succeeded"
            raise RuntimeError(f"All provider/model attempts failed: {failures}")

        if model_to_use != decision.model:
            decision = replace(
                decision,
                model=model_to_use,
                reason=f"{decision.reason} [FAILOVER→{model_to_use}]",
            )
        
        latency_ms = (time.perf_counter() - start_time) * 1000

        # 5. Recording & Logging
        record = self._tracker.record(response, model=model_to_use)
        
        step_result = StepResult(
            step_num=len(self._log.steps) + 1,
            task=task,
            decision=decision,
            response_text=response.choices[0].message.content,
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

    def run(self, tasks: List[str]) -> RoutingLog:
        """
        Execute a series of tasks sequentially.
        Stops early if the hard budget cap is hit.
        """
        for task in tasks:
            try:
                self.chat(task)
            except BudgetExceeded:
                # Graceful stop: we hit the hard budget cap
                break
            except TaskRejected:
                # Logged step already; continue with remaining tasks
                continue
            except Exception:
                # For integration/benchmark purposes, treat other errors as stop
                break
        return self._log

    def _execution_model_candidates(self, tier: ModelTier, chosen_model: str) -> List[str]:
        if tier == ModelTier.BIG:
            extras = self._big_fallback_models
        else:
            extras = self._small_fallback_models
        ordered = [chosen_model] + [m for m in extras if m and m != chosen_model]
        return ordered

    def _append_telemetry(self, step: StepResult) -> None:
        """Append structured step telemetry as JSONL when enabled."""
        if not self._telemetry_jsonl_path:
            return

        parent = os.path.dirname(self._telemetry_jsonl_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        payload = step.to_dict()
        payload["timestamp_unix_ms"] = int(time.time() * 1000)
        payload["router_budget_usd"] = self.budget

        with open(self._telemetry_jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")
