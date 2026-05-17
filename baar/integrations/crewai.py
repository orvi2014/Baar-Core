"""
baar/integrations/crewai.py — CrewAI integration.

Two integration styles:

  BaarCrewLLM
      A drop-in CrewAI LLM backed by BAARRouter.  Pass it as ``llm=`` to any
      CrewAI Agent to get semantic complexity routing and a hard budget cap on
      every LLM call.  No proxy, no subprocess — the router is called directly
      on each turn.

  BaarStepCallback
      A step-callback for existing CrewAI crews that enforces a budget kill-
      switch between task steps without touching the crew's LLM configuration.
      Pass an instance as ``step_callback=`` when constructing a Crew.

Install:
    pip install baar-core[crewai]

Quick start:
    from baar import BAARRouter
    from baar.integrations.crewai import BaarCrewLLM
    from crewai import Agent, Task, Crew

    router = BAARRouter(budget=1.00)
    llm    = BaarCrewLLM(router=router)

    researcher = Agent(
        role="Research Analyst",
        goal="Find accurate information",
        backstory="Expert researcher with a focus on AI trends",
        llm=llm,
    )
    task = Task(
        description="Compare the top 3 open-source agent frameworks",
        agent=researcher,
        expected_output="A concise markdown table",
    )
    crew = Crew(agents=[researcher], tasks=[task])
    result = crew.kickoff()

    print(result.raw)
    print(f"Spent ${router.spent:.4f} of $1.00")
    print(f"Remaining ${router.remaining:.4f}")
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

try:
    from crewai import LLM as _CrewAILLM
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "crewai is required for this integration. "
        "Install with:  pip install baar-core[crewai]"
    ) from exc

from baar.router import BAARRouter
from baar.core.exceptions import BudgetExhausted
from baar.core.models import RoutingLog


# ── Helpers ───────────────────────────────────────────────────────────────────

def _last_user_content(messages: List[Dict[str, Any]]) -> str:
    """Return the last user-role message content — used as the routing signal."""
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content", "")
            return content if isinstance(content, str) else str(content)
    # Fall back to the last message regardless of role
    if messages:
        content = messages[-1].get("content", "")
        return content if isinstance(content, str) else str(content)
    return ""


# ── BaarCrewLLM ───────────────────────────────────────────────────────────────

class BaarCrewLLM(_CrewAILLM):
    """
    A CrewAI LLM backed by BAARRouter.

    Every LLM call goes through Baar's full pipeline:
      1. Hard pre-flight kill-switch — raises BudgetExhausted before any
         network call when the budget is too low.
      2. Semantic complexity routing — routes to small or big model based
         on task complexity.
      3. Budget-aware downgrade — falls back to small model automatically
         when running low.

    Usage::

        from baar import BAARRouter
        from baar.integrations.crewai import BaarCrewLLM
        from crewai import Agent, Task, Crew

        router = BAARRouter(budget=1.00)
        llm    = BaarCrewLLM(router=router)

        researcher = Agent(
            role="Research Analyst",
            goal="Find accurate information",
            backstory="Expert researcher with a focus on AI trends",
            llm=llm,
        )
        task = Task(
            description="Compare the top 3 open-source agent frameworks",
            agent=researcher,
            expected_output="A concise markdown table",
        )
        crew   = Crew(agents=[researcher], tasks=[task])
        result = crew.kickoff()

        print(result.raw)
        print(f"Spent ${router.spent:.4f} of $1.00")

    Multi-agent crew with shared budget::

        router     = BAARRouter(budget=2.00)
        researcher = Agent(role="Researcher", llm=BaarCrewLLM(router=router), ...)
        writer     = Agent(role="Writer",     llm=BaarCrewLLM(router=router), ...)
        # Both agents draw from the same $2.00 pool.
    """

    def __init__(self, router: BAARRouter, **kwargs: Any) -> None:
        # CrewAI's LLM.__init__ requires a model name for internal
        # logging/metadata.  Use the router's small model as the identifier;
        # actual per-call model selection is done by BAARRouter.
        kwargs.setdefault("model", router.small_model)
        super().__init__(**kwargs)
        self._baar_router = router

    # ── Budget properties ─────────────────────────────────────────────────────

    @property
    def spent(self) -> float:
        """Total USD spent so far through this LLM."""
        return self._baar_router.spent

    @property
    def remaining(self) -> float:
        """USD remaining before the kill-switch fires."""
        return self._baar_router.remaining

    @property
    def budget(self) -> float:
        """Hard budget cap in USD."""
        return self._baar_router.budget

    @property
    def log(self) -> RoutingLog:
        """Full routing audit trail for this session."""
        return self._baar_router.log

    # ── LLM interface ─────────────────────────────────────────────────────────

    def call(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        callbacks: Optional[List[Any]] = None,
        available_functions: Optional[Dict[str, Callable]] = None,
    ) -> str:
        """Route the LLM call through BAARRouter (sync)."""
        task = _last_user_content(messages)
        return self._baar_router.chat(task, messages=messages)

    async def acall(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        callbacks: Optional[List[Any]] = None,
        available_functions: Optional[Dict[str, Callable]] = None,
    ) -> str:
        """Route the LLM call through BAARRouter (async)."""
        task = _last_user_content(messages)
        return await self._baar_router.achat(task, messages=messages)


# ── BaarStepCallback ──────────────────────────────────────────────────────────

class BaarStepCallback:
    """
    CrewAI step callback that enforces a hard budget kill-switch between steps.

    Pass an instance as ``step_callback=`` to a Crew to add budget protection
    to any existing crew without changing the crew's LLM configuration.

    After each agent step CrewAI calls this object.  If the router's remaining
    budget has reached zero, BudgetExhausted is raised — CrewAI propagates it
    as a task error and stops the crew gracefully.

    Usage::

        from baar import BAARRouter
        from baar.integrations.crewai import BaarStepCallback
        from crewai import Crew

        router = BAARRouter(budget=0.50)
        guard  = BaarStepCallback(router)

        crew = Crew(
            agents=[researcher, writer],
            tasks=[research_task, write_task],
            step_callback=guard,
        )
        crew.kickoff()
        print(f"Spent ${router.spent:.4f} of $0.50")

    Note:
        This callback fires *after* each step, so the last step may slightly
        overshoot the budget.  For zero-overshoot pre-flight enforcement use
        BaarCrewLLM as the agent's LLM instead.
    """

    def __init__(self, router: BAARRouter) -> None:
        self._router = router

    def __call__(self, step_output: Any) -> None:
        """Called by CrewAI after each agent step."""
        if self._router.remaining <= 0:
            raise BudgetExhausted(
                f"Baar kill-switch: budget exhausted after this step "
                f"(${self._router.spent:.4f} spent of ${self._router.budget:.4f}). "
                "Crew execution stopped.",
                remaining=self._router.remaining,
            )

    @property
    def spent(self) -> float:
        return self._router.spent

    @property
    def remaining(self) -> float:
        return self._router.remaining
