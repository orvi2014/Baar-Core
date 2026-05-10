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
from typing import Any, AsyncIterator, Dict, List, Optional

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel, Field
except ImportError as exc:
    raise ImportError(
        "FastAPI is required for the Vercel integration. "
        "Install it with:  pip install baar-core[vercel]"
    ) from exc

from baar.router import BAARRouter
from baar.core.exceptions import BudgetExhausted, TaskRejected
from baar.core.budget import BudgetExceeded


# ── Request / response models ─────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str


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
        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())

        if req.stream:
            return StreamingResponse(
                _stream_response(router, task, messages, completion_id, created),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",  # disables nginx buffering
                },
            )

        # ── Non-streaming ──────────────────────────────────────────────────────
        try:
            reply = await router.achat(task, messages=messages)
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


# ── Streaming generator ───────────────────────────────────────────────────────

async def _stream_response(
    router: BAARRouter,
    task: str,
    messages: List[Dict[str, str]],
    completion_id: str,
    created: int,
) -> AsyncIterator[str]:
    model_used = "baar"
    try:
        # Opening delta — role announcement
        yield _make_chunk(completion_id, created, model_used, "", finish_reason=None)

        async for chunk_text in router.astream_chat(task, messages=messages):
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _last_user_content(messages: List[ChatMessage]) -> str:
    for m in reversed(messages):
        if m.role == "user":
            return m.content
    return messages[-1].content if messages else ""


def _last_step_info(router: BAARRouter):
    """Best-effort: read prompt_tokens, completion_tokens, model from last log step."""
    if router.log.steps:
        step = list(router.log.steps)[-1]
        return step.prompt_tokens, step.completion_tokens, step.model_used
    return 0, 0, "baar"
