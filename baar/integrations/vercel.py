"""
baar/integrations/vercel.py — OpenAI-compatible HTTP server adapter.

Wraps BAARRouter as a /v1/chat/completions endpoint so any OpenAI-compatible
client — Vercel AI SDK, LangChain, LlamaIndex, curl — can use Baar's routing
and hard budget enforcement without code changes on the client side.

Install:
    pip install baar-core[vercel]
    # or: pip install fastapi uvicorn

Usage (Python server):
    from baar import BAARRouter
    from baar.integrations.vercel import create_app

    router = BAARRouter(budget=0.10)
    app    = create_app(router)

    # uvicorn mymodule:app --host 0.0.0.0 --port 8000

Usage (Vercel AI SDK — JavaScript/TypeScript):
    import { createOpenAI } from '@ai-sdk/openai';
    import { streamText } from 'ai';

    const baar = createOpenAI({
        baseURL: 'https://your-server.vercel.app/v1',
        apiKey: 'not-needed',
    });

    const { textStream } = streamText({
        model: baar('baar'),   // model name ignored — Baar routes automatically
        messages: [{ role: 'user', content: 'Hello!' }],
    });

HTTP error codes:
    402  Budget exhausted — remaining balance too low for this call
    422  Task rejected   — value gate determined task is not worth the cost
    500  Internal error  — provider failure or unexpected exception

Security note:
    Pass api_key to create_app() to require Authorization: Bearer <key> on all
    requests.  Without it, the endpoint is open to any caller.

Concurrency note:
    BAARRouter is thread-safe and asyncio-safe.  The check-and-reserve step is
    atomic, so concurrent requests cannot together overshoot the budget.
    The 'model' field in responses reflects the actual tier used (small/big),
    read from router.log after each call.  Under very high concurrency the log
    read races with new steps — use telemetry_jsonl_path for authoritative audit.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Union

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover
    raise ImportError(  # pragma: no cover
        "FastAPI is required for the Vercel integration. "
        "Install it with:  pip install baar-core[vercel]"
    ) from exc

from baar.router import BAARRouter
from baar.core.exceptions import BudgetExhausted, TaskRejected
from baar.core.budget import BudgetExceeded


# ── Request / response models ─────────────────────────────────────────────────

class ChatMessage(BaseModel):
    # Unknown OpenAI message fields are kept and forwarded to the provider.
    model_config = {"extra": "allow"}

    role: str
    # None for assistant messages that only carry tool_calls; a list for
    # multi-part (text/image) content.
    content: Optional[Union[str, List[Any]]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

    def to_provider(self) -> Dict[str, Any]:
        d = self.model_dump(exclude_none=True)
        if "content" not in d and not self.tool_calls:
            d["content"] = ""
        return d


class ChatRequest(BaseModel):
    model: str = "baar"
    messages: List[ChatMessage]
    stream: bool = False
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    # Extra OpenAI fields accepted but ignored — makes SDK compatibility seamless
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[Any] = None
    user: Optional[str] = None
    # Tool calling. Legacy `functions` / `function_call` are mapped onto these.
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    functions: Optional[List[Dict[str, Any]]] = None
    function_call: Optional[Any] = None

    def provider_tools(self):
        """(tools, tool_choice) to forward, or (None, None) when the request has none."""
        tools = self.tools
        tool_choice = self.tool_choice
        if not tools and self.functions:
            tools = [{"type": "function", "function": f} for f in self.functions]
            fc = self.function_call
            if isinstance(fc, dict) and fc.get("name"):
                tool_choice = {"type": "function", "function": {"name": fc["name"]}}
            elif fc in ("auto", "none"):
                tool_choice = fc
        return (tools or None), (tool_choice if tools else None)


# ── SSE helpers ───────────────────────────────────────────────────────────────

def _make_chunk(completion_id: str, created: int, model: str, content: str, finish_reason: Optional[str] = None) -> str:
    delta = {"content": content} if content else {}
    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(payload)}\n\n"


def _make_completion(completion_id: str, created: int, model: str, content: str,
                     prompt_tokens: int = 0, completion_tokens: int = 0) -> dict:
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _budget_error(exc: BudgetExhausted) -> HTTPException:
    return HTTPException(
        status_code=402,
        detail={
            "error": {
                "message": str(exc),
                "type": "budget_exhausted",
                "code": "quota_exceeded",
                "remaining_usd": exc.remaining,
            }
        },
    )


def _reject_error(exc: TaskRejected) -> HTTPException:
    return HTTPException(
        status_code=422,
        detail={
            "error": {
                "message": str(exc),
                "type": "task_rejected",
                "code": "value_gate_reject",
                "estimated_value": exc.estimated_value,
                "estimated_cost_usd": exc.estimated_cost_usd,
            }
        },
    )


# ── App factory ───────────────────────────────────────────────────────────────

def create_app(
    router: BAARRouter,
    *,
    api_key: Optional[str] = None,
    title: str = "Baar OpenAI-compatible API",
) -> FastAPI:
    """
    Create a FastAPI application that exposes BAARRouter as an
    OpenAI-compatible /v1/chat/completions endpoint.

    Args:
        router:  A configured BAARRouter instance.
        api_key: When set, all requests must include
                 ``Authorization: Bearer <api_key>``.  Leave None to disable
                 authentication (suitable for localhost / internal networks).
        title:   OpenAPI docs title.

    Returns:
        A FastAPI app ready to serve with uvicorn or any ASGI server.
    """
    app = FastAPI(title=title)

    # ── Auth middleware ────────────────────────────────────────────────────────

    if api_key:
        @app.middleware("http")
        async def _auth(request: Request, call_next):
            if request.url.path.startswith("/v1"):
                auth = request.headers.get("Authorization", "")
                if auth != f"Bearer {api_key}":
                    return JSONResponse(
                        status_code=401,
                        content={"error": {"message": "Invalid API key", "type": "auth_error"}},
                    )
            return await call_next(request)

    # ── /v1/models ────────────────────────────────────────────────────────────

    @app.get("/v1/models")
    async def list_models():
        """Returns a minimal model list so SDK auto-discovery works."""
        now = int(time.time())
        return {
            "object": "list",
            "data": [
                {
                    "id": "baar",
                    "object": "model",
                    "created": now,
                    "owned_by": "baar-core",
                    "description": (
                        f"BAARRouter — small: {router.small_model}, "
                        f"big: {router.big_model}, "
                        f"budget: ${router.budget:.4f}"
                    ),
                }
            ],
        }

    # ── /v1/chat/completions ──────────────────────────────────────────────────

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatRequest):
        task = _last_user_content(req.messages)
        messages = [m.to_provider() for m in req.messages]
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        tools, tool_choice = req.provider_tools()

        if tools:
            # Tool-calling requests go through acomplete(), which forwards the
            # schemas, counts them in the pre-flight estimate, and returns the
            # model's tool_calls.
            if req.stream:
                return StreamingResponse(
                    _stream_tool_response(router, task, messages, tools, tool_choice,
                                          completion_id, created, req.max_tokens),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                )
            try:
                response = await router.acomplete(
                    task, messages=messages, tools=tools, tool_choice=tool_choice,
                    **_output_kwargs(req.max_tokens),
                )
            except BudgetExhausted as exc:
                raise _budget_error(exc)
            except (BudgetExceeded,) as exc:
                raise _budget_error(BudgetExhausted(str(exc), remaining=router.remaining))
            except TaskRejected as exc:
                raise _reject_error(exc)
            _, _, model_used = _last_step_info(router)
            return _completion_from_response(completion_id, created, model_used, response)

        if req.stream:
            return StreamingResponse(
                _stream_response(router, task, messages, completion_id, created, req.max_tokens),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",  # disables nginx buffering
                },
            )

        # ── Non-streaming ──────────────────────────────────────────────────────
        try:
            reply = await router.achat(task, messages=messages, **_output_kwargs(req.max_tokens))
        except BudgetExhausted as exc:
            raise _budget_error(exc)
        except (BudgetExceeded,) as exc:
            raise _budget_error(BudgetExhausted(str(exc), remaining=router.remaining))
        except TaskRejected as exc:
            raise _reject_error(exc)

        # Read token counts from the last log step (best-effort under concurrency)
        prompt_tokens, completion_tokens, model_used = _last_step_info(router)
        return _make_completion(
            completion_id, created, model_used, reply, prompt_tokens, completion_tokens
        )

    return app


def _output_kwargs(max_tokens: Optional[int]) -> Dict[str, Any]:
    """Forward the request's own max_tokens as the router's output limit, only when set."""
    return {"max_output_tokens": max_tokens} if max_tokens is not None else {}


# ── Streaming generator ───────────────────────────────────────────────────────

async def _stream_response(
    router: BAARRouter,
    task: str,
    messages: List[Dict[str, str]],
    completion_id: str,
    created: int,
    max_tokens: Optional[int] = None,
) -> AsyncIterator[str]:
    model_used = "baar"
    try:
        # Opening delta — role announcement
        yield _make_chunk(completion_id, created, model_used, "", finish_reason=None)

        async for chunk_text in router.astream_chat(task, messages=messages, **_output_kwargs(max_tokens)):
            yield _make_chunk(completion_id, created, model_used, chunk_text)

        # After streaming completes, update model name from log
        _, _, model_used = _last_step_info(router)

        # Final chunk — finish_reason=stop
        yield _make_chunk(completion_id, created, model_used, "", finish_reason="stop")
        yield "data: [DONE]\n\n"

    except BudgetExhausted as exc:
        error_chunk = json.dumps({
            "error": {
                "message": str(exc),
                "type": "budget_exhausted",
                "code": "quota_exceeded",
                "remaining_usd": exc.remaining,
            }
        })
        yield f"data: {error_chunk}\n\n"
        yield "data: [DONE]\n\n"

    except TaskRejected as exc:
        error_chunk = json.dumps({
            "error": {
                "message": str(exc),
                "type": "task_rejected",
                "code": "value_gate_reject",
            }
        })
        yield f"data: {error_chunk}\n\n"
        yield "data: [DONE]\n\n"


async def _stream_tool_response(
    router: BAARRouter,
    task: str,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    tool_choice: Any,
    completion_id: str,
    created: int,
    max_tokens: Optional[int] = None,
) -> AsyncIterator[str]:
    """
    Streaming for tool-calling requests. The call runs to completion under the
    same guardrails, then is replayed as OpenAI chunks: content, then each tool
    call as a single delta, then the real finish_reason. Clients that accumulate
    deltas (the OpenAI SDKs, Vercel AI SDK, agent frameworks) see a normal stream.
    """
    try:
        response = await router.acomplete(
            task, messages=messages, tools=tools, tool_choice=tool_choice,
            **_output_kwargs(max_tokens),
        )
        _, _, model_used = _last_step_info(router)
        choice = response.choices[0]
        msg = choice.message
        yield _delta_chunk(completion_id, created, model_used, {"role": "assistant"})
        if getattr(msg, "content", None):
            yield _delta_chunk(completion_id, created, model_used, {"content": msg.content})
        calls = [_tool_call_dict(tc) for tc in (getattr(msg, "tool_calls", None) or [])]
        if calls:
            yield _delta_chunk(completion_id, created, model_used,
                               {"tool_calls": [dict(c, index=i) for i, c in enumerate(calls)]})
        yield _delta_chunk(completion_id, created, model_used, {},
                           finish_reason=choice.finish_reason or ("tool_calls" if calls else "stop"))
        yield "data: [DONE]\n\n"
    except (BudgetExhausted, BudgetExceeded) as exc:
        remaining = getattr(exc, "remaining", router.remaining)
        yield "data: " + json.dumps({"error": {
            "message": str(exc), "type": "budget_exhausted",
            "code": "quota_exceeded", "remaining_usd": remaining}}) + "\n\n"
        yield "data: [DONE]\n\n"
    except TaskRejected as exc:
        yield "data: " + json.dumps({"error": {
            "message": str(exc), "type": "task_rejected", "code": "value_gate_reject"}}) + "\n\n"
        yield "data: [DONE]\n\n"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _delta_chunk(completion_id: str, created: int, model: str, delta: Dict[str, Any],
                 finish_reason: Optional[str] = None) -> str:
    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(payload)}\n\n"


def _tool_call_dict(tc: Any) -> Dict[str, Any]:
    """OpenAI wire format for a tool call (LiteLLM object or plain dict)."""
    get = (lambda o, k: o.get(k)) if isinstance(tc, dict) else getattr
    fn = get(tc, "function")
    return {
        "id": get(tc, "id"),
        "type": get(tc, "type") or "function",
        "function": {
            "name": get(fn, "name") if fn is not None else None,
            "arguments": (get(fn, "arguments") if fn is not None else None) or "",
        },
    }


def _completion_from_response(completion_id: str, created: int, model: str, response: Any) -> dict:
    choice = response.choices[0]
    msg = choice.message
    calls = [_tool_call_dict(tc) for tc in (getattr(msg, "tool_calls", None) or [])]
    message: Dict[str, Any] = {"role": "assistant", "content": getattr(msg, "content", None)}
    if calls:
        message["tool_calls"] = calls
    usage = getattr(response, "usage", None)
    pt = int(getattr(usage, "prompt_tokens", 0) or 0)
    ct = int(getattr(usage, "completion_tokens", 0) or 0)
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": choice.finish_reason or ("tool_calls" if calls else "stop"),
        }],
        "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct},
    }


def _content_text(content: Any) -> str:
    """Plain text of a message's content (string, multi-part list, or None)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(str(p.get("text", "")))
            elif isinstance(p, str):
                parts.append(p)
        return " ".join(parts)
    return str(content)


def _last_user_content(messages: List[ChatMessage]) -> str:
    for m in reversed(messages):
        if m.role == "user":
            return _content_text(m.content)
    return _content_text(messages[-1].content) if messages else ""


def _last_step_info(router: BAARRouter):
    """Best-effort: read prompt_tokens, completion_tokens, model from last log step."""
    if router.log.steps:
        step = list(router.log.steps)[-1]
        return step.prompt_tokens, step.completion_tokens, step.model_used
    return 0, 0, "baar"
