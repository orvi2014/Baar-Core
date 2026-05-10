"""
baar/integrations/langchain.py — LangChain & LangGraph adapters.

Two integration points:

  BaarCallbackHandler
      Drop into any existing LangChain chain or agent via callbacks=[...].
      Enforces the hard kill-switch before every LLM call and records actual
      spend afterwards.  Does NOT change which model is used — the chain's
      own LLM is still called.  Use this when you already have a chain and
      want budget protection with zero refactoring.

  BaarChatModel
      A BaseChatModel that routes every request through BAARRouter — semantic
      complexity scoring, budget-aware model selection, and hard pre-flight
      rejection.  Use this when you want full Baar routing inside a LangChain
      chain or LangGraph node.

Install:
    pip install baar-core[langchain]
    # or: pip install langchain-core
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, Iterator, List, Optional, Sequence

try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import (
        AIMessage,
        AIMessageChunk,
        BaseMessage,
        HumanMessage,
        SystemMessage,
    )
    from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult, LLMResult
    from pydantic import PrivateAttr
except ImportError as exc:
    raise ImportError(
        "LangChain is required for this integration. "
        "Install it with:  pip install baar-core[langchain]"
    ) from exc

from baar.router import BAARRouter, token_counter
from baar.core.exceptions import BudgetExhausted
from baar.core.budget import BudgetExceeded


# ── Helpers ───────────────────────────────────────────────────────────────────

def _messages_to_dicts(messages: Sequence[BaseMessage]) -> List[Dict[str, str]]:
    """Convert LangChain message objects to the dict format BAARRouter expects."""
    role_map = {
        "human": "user",
        "ai": "assistant",
        "system": "system",
        "tool": "tool",
        "function": "function",
    }
    result = []
    for m in messages:
        role = role_map.get(m.type, "user")
        content = m.content if isinstance(m.content, str) else str(m.content)
        result.append({"role": role, "content": content})
    return result


def _last_human_content(messages: Sequence[BaseMessage]) -> str:
    """Return the content of the last HumanMessage — used as the routing signal."""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m.content if isinstance(m.content, str) else str(m.content)
    # Fall back to the last message regardless of type
    if messages:
        m = messages[-1]
        return m.content if isinstance(m.content, str) else str(m.content)
    return ""


def _extract_model(serialized: Dict[str, Any], kwargs: Dict[str, Any]) -> str:
    """Best-effort model name extraction from LangChain callback context."""
    # LangChain puts the model name in different places depending on the LLM class
    for key in ("model_name", "model"):
        val = serialized.get("kwargs", {}).get(key)
        if val:
            return str(val)
    invocation = kwargs.get("invocation_params", {})
    for key in ("model", "model_name", "engine"):
        val = invocation.get(key)
        if val:
            return str(val)
    return "gpt-4o-mini"  # safe default for cost estimation


# ── BaarCallbackHandler ───────────────────────────────────────────────────────

class BaarCallbackHandler(BaseCallbackHandler):
    """
    LangChain callback that enforces Baar's hard budget kill-switch on any
    chain or agent — without changing a single line of your existing code.

    Usage::

        from baar import BAARRouter
        from baar.integrations.langchain import BaarCallbackHandler
        from langchain_openai import ChatOpenAI

        router  = BAARRouter(budget=0.05)
        handler = BaarCallbackHandler(router)

        llm   = ChatOpenAI(model="gpt-4o", callbacks=[handler])
        chain = prompt | llm | parser
        # BudgetExhausted is raised before the API call when budget is too low

    The handler performs a pre-flight cost estimate in on_llm_start /
    on_chat_model_start and raises BudgetExhausted if the call would overshoot
    the remaining budget.  Actual spend is recorded in on_llm_end so the
    router's remaining balance stays accurate.
    """

    def __init__(self, router: BAARRouter) -> None:
        super().__init__()
        self._router = router
        self._pending_model: str = ""

    # ── pre-flight ─────────────────────────────────────────────────────────────

    def _preflight(self, model: str, text: str) -> None:
        prompt_tokens = token_counter(text=text, model=model)
        try:
            self._router._tracker.check_affordability(model, prompt_tokens)
        except BudgetExceeded as exc:
            raise BudgetExhausted(
                f"Baar kill-switch: budget too low for '{model}' "
                f"(need ~${exc.requested:.6f}, have ${exc.remaining:.6f}). "
                "No API call was made.",
                remaining=exc.remaining,
            ) from exc

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any,
    ) -> None:
        model = _extract_model(serialized, kwargs)
        self._pending_model = model
        self._preflight(model, " ".join(prompts))

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        **kwargs: Any,
    ) -> None:
        model = _extract_model(serialized, kwargs)
        self._pending_model = model
        flat_text = " ".join(
            m.content for batch in messages for m in batch
            if isinstance(m.content, str)
        )
        self._preflight(model, flat_text)

    # ── post-call spend recording ──────────────────────────────────────────────

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        model = self._pending_model or "gpt-4o-mini"
        llm_output = response.llm_output or {}
        usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        if prompt_tokens or completion_tokens:
            self._router._tracker.record_manual(
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )


# ── BaarChatModel ─────────────────────────────────────────────────────────────

class BaarChatModel(BaseChatModel):
    """
    A LangChain ChatModel backed by BAARRouter.

    Every call goes through Baar's full pipeline:
      1. Hard pre-flight kill-switch (zero network cost if budget is exhausted)
      2. Semantic complexity routing (cheap vs capable model)
      3. Budget-aware downgrade (falls back to cheap model when running low)

    Use this as a drop-in replacement for ChatOpenAI / ChatAnthropic anywhere
    you want routing + hard budget enforcement.

    Usage::

        from baar import BAARRouter
        from baar.integrations.langchain import BaarChatModel

        router = BAARRouter(budget=0.10, small_model="gpt-4o-mini", big_model="gpt-4o")
        llm    = BaarChatModel(router=router)

        # Use anywhere a BaseChatModel is expected
        chain = prompt | llm | StrOutputParser()
        reply = chain.invoke({"question": "Explain the GIL"})

    LangGraph::

        from langgraph.graph import StateGraph
        from langgraph.prebuilt import create_react_agent

        agent = create_react_agent(model=llm, tools=[...])

    Budget state is shared with the BAARRouter instance, so calls made via
    BaarChatModel and direct router.chat() calls both draw from the same pool.
    """

    # Keep the router as a private Pydantic attribute so it is excluded from
    # serialisation (BAARRouter is not a Pydantic model).
    _router: BAARRouter = PrivateAttr()

    # Expose a minimal set of serialisable fields so LangChain's tracing /
    # LangSmith integration can log meaningful metadata.
    small_model: str = "gpt-4o-mini"
    big_model: str = "gpt-4o"
    budget: float = 0.10

    def __init__(self, router: BAARRouter, **kwargs: Any) -> None:
        kwargs.setdefault("small_model", router.small_model)
        kwargs.setdefault("big_model", router.big_model)
        kwargs.setdefault("budget", router._tracker.total_budget)
        super().__init__(**kwargs)
        self._router = router

    @property
    def _llm_type(self) -> str:
        return "baar"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "small_model": self.small_model,
            "big_model": self.big_model,
            "budget_usd": self.budget,
            "spent_usd": round(self._router.spent, 6),
            "remaining_usd": round(self._router.remaining, 6),
        }

    # ── sync ──────────────────────────────────────────────────────────────────

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        task = _last_human_content(messages)
        msg_dicts = _messages_to_dicts(messages)
        reply = self._router.chat(task, messages=msg_dicts)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=reply))])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        task = _last_human_content(messages)
        msg_dicts = _messages_to_dicts(messages)
        for chunk_text in self._router.stream_chat(task, messages=msg_dicts):
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=chunk_text))
            if run_manager:
                run_manager.on_llm_new_token(chunk_text, chunk=chunk)
            yield chunk

    # ── async ─────────────────────────────────────────────────────────────────

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        task = _last_human_content(messages)
        msg_dicts = _messages_to_dicts(messages)
        reply = await self._router.achat(task, messages=msg_dicts)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=reply))])
