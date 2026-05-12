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

import uuid
import warnings
from typing import Any, Dict, Iterator, List, Optional, Sequence

try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import (
        AIMessage,
        AIMessageChunk,
        BaseMessage,
        HumanMessage,
    )
    from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult, LLMResult
    from pydantic import PrivateAttr
except ImportError as exc:  # pragma: no cover
    raise ImportError(  # pragma: no cover
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
    warnings.warn(
        "BaarCallbackHandler: could not extract model name from LangChain callback context. "
        "Falling back to 'gpt-4o-mini' for cost estimation — actual costs may differ significantly "
        "if you are using a different provider. Ensure your LLM class populates "
        "serialized['kwargs']['model_name'] or invocation_params['model'].",
        UserWarning,
        stacklevel=3,
    )
    return "gpt-4o-mini"


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

    # Bug fix: raise_error=False (the BaseCallbackHandler default) causes LangChain's
    # handle_event to catch and swallow any exception raised in on_llm_start, silently
    # logging it as a warning. The kill-switch never fires. Setting raise_error=True
    # forces handle_event to re-raise, so BudgetExhausted propagates to the caller.
    raise_error: bool = True

    def __init__(self, router: BAARRouter) -> None:
        super().__init__()
        self._router = router
        # Bug fix: a single _pending_model string is a race condition under concurrent
        # calls — thread A's on_llm_end reads thread B's model name. Keying by run_id
        # (the unique UUID LangChain assigns per call) gives each call its own slot.
        self._pending_models: Dict[uuid.UUID, str] = {}
        # Reservation amounts per run_id — see _preflight for why this is needed.
        self._pending_reservations: Dict[uuid.UUID, float] = {}

    # ── pre-flight ─────────────────────────────────────────────────────────────

    def _preflight(self, model: str, text: str) -> float:
        prompt_tokens = token_counter(text=text, model=model)
        estimated = self._router._tracker.estimate_cost(model, prompt_tokens)
        # Use check_and_reserve (atomic check + deduct) rather than check_affordability
        # (read-only). Under concurrent async LangChain calls, two preflights can both
        # pass a read-only check and together overshoot the budget. Reserving atomically
        # prevents that. The reservation is cancelled in on_llm_end / on_llm_error once
        # actual spend is recorded.
        try:
            self._router._tracker.check_and_reserve(estimated)
        except BudgetExceeded as exc:
            raise BudgetExhausted(
                f"Baar kill-switch: budget too low for '{model}' "
                f"(need ~${exc.requested:.6f}, have ${exc.remaining:.6f}). "
                "No API call was made.",
                remaining=exc.remaining,
            ) from exc
        return estimated

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        model = _extract_model(serialized, kwargs)
        self._pending_models[run_id] = model
        self._pending_reservations[run_id] = self._preflight(model, " ".join(prompts))

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        model = _extract_model(serialized, kwargs)
        self._pending_models[run_id] = model
        # Convert all content to str — skipping non-string content (tool results,
        # multi-modal payloads) silently underestimates tokens for agent chains.
        flat_text = " ".join(
            m.content if isinstance(m.content, str) else str(m.content)
            for batch in messages for m in batch
        )
        self._pending_reservations[run_id] = self._preflight(model, flat_text)

    # ── post-call spend recording ──────────────────────────────────────────────

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        model = self._pending_models.pop(run_id, "gpt-4o-mini")
        reserved = self._pending_reservations.pop(run_id, 0.0)
        llm_output = response.llm_output or {}
        usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        if prompt_tokens or completion_tokens:
            # Cancel the preflight reservation before recording actual spend so the
            # store reflects real cost rather than estimate + actual (double-counting).
            if reserved:
                self._router._tracker.cancel_reservation(reserved)
            self._router._tracker.record_manual(
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        else:
            # Provider did not report token usage. Retaining the preflight reservation
            # as spend prevents silent budget inflation — better to over-count by
            # estimate than to charge $0 for a call that actually cost something.
            warnings.warn(
                f"BaarCallbackHandler: '{model}' completed with no token usage in "
                "llm_output. Preflight cost estimate retained as spend. "
                "For accurate tracking, use a LangChain LLM that populates "
                "llm_output['token_usage'] or llm_output['usage'].",
                UserWarning,
                stacklevel=2,
            )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        # Clean up both slots and release the budget reservation so no cost is
        # charged when the LLM call itself failed.
        self._pending_models.pop(run_id, None)
        reserved = self._pending_reservations.pop(run_id, 0.0)
        if reserved:
            self._router._tracker.cancel_reservation(reserved)


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

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ):
        # Use astream_chat (native async generator) instead of stream_chat (sync).
        # Calling a sync blocking iterator inside an async generator stalls the
        # event loop on every chunk — defeating async entirely.
        task = _last_human_content(messages)
        msg_dicts = _messages_to_dicts(messages)
        async for chunk_text in self._router.astream_chat(task, messages=msg_dicts):
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=chunk_text))
            if run_manager:
                await run_manager.on_llm_new_token(chunk_text, chunk=chunk)
            yield chunk
