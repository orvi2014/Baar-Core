"""
baar/router.py — Main BAARRouter orchestration class.
Handles the end-to-end flow: routing, budget constraints, and execution.
"""

import time
import litellm
from typing import List, Optional

from baar.core.router import Router, ModelTier
from baar.core.budget import BudgetTracker, BudgetExceeded
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
        complexity_threshold: float = 0.65,
        use_llm_router: bool = True,
        system_prompt: Optional[str] = None,
    ):
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
        )
        self._log = RoutingLog(
            budget=budget,
            small_model=small_model,
            big_model=big_model,
        )

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
        Execute a single routed chat call.
        
        0. Pre-flight check: Ensure there is some budget left before routing.
        1. Decide Model: Route to SMALL or BIG based on semantic complexity.
        2. BCD (Budget-Constrained Decoding): If BIG is chosen but 
           unaffordable, downgrade to SMALL to preserve budget.
        3. Execution: Call the chosen model via LiteLLM.
        4. Audit: Record the real cost and add to the routing log.
        """
        # 0. Fast Budget Kill-Switch (Zero-Call safety)
        # Check if we can afford the cheapest model for the given task
        prompt_tokens_small = token_counter(task, model=self.small_model)
        self._tracker.check_affordability(self.small_model, prompt_tokens_small)

        # 1. Routing Decision
        decision = self._router.decide(
            task=task,
            remaining_budget=self.remaining,
            budget_utilization=self._tracker.utilization,
        )

        # 2. Budget Pressure / BCD
        # If we chose BIG, check if we can actually afford it.
        if decision.tier == ModelTier.BIG:
            prompt_tokens = token_counter(task, model=self.big_model)
            try:
                self._tracker.check_affordability(self.big_model, prompt_tokens)
            except BudgetExceeded as e:
                # Downgrade to SMALL if BIG is too expensive
                decision = self._router.force_small(decision, str(e))

        # Final pre-flight check (ensure we can afford at least the chosen model)
        model_to_use = decision.model
        prompt_tokens = token_counter(task, model=model_to_use)
        self._tracker.check_affordability(model_to_use, prompt_tokens)

        # 3. Execution
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": task})

        start_time = time.perf_counter()
        
        # LiteLLM handles the underlying API calls (OpenAI, Anthropic, etc.)
        response = litellm.completion(
            model=model_to_use,
            messages=messages,
        )
        
        latency_ms = (time.perf_counter() - start_time) * 1000

        # 4. Recording & Logging
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
        )
        self._log.add(step_result)

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
            except Exception:
                # For integration/benchmark purposes, treat other errors as stop
                break
        return self._log
