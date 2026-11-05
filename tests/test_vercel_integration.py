"""
tests/test_vercel_integration.py — OpenAI-compatible server adapter tests.

Runs entirely without API keys. All LiteLLM and budget calls are mocked.
"""

import json
import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")

from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient

from baar import BAARRouter, BudgetExhausted
from baar.core.exceptions import TaskRejected
from baar.integrations.vercel import create_app, _last_user_content, ChatMessage


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_router(budget: float = 1.0) -> BAARRouter:
    return BAARRouter(budget=budget, use_llm_router=False)


def make_client(router: BAARRouter, api_key: str = None) -> TestClient:
    app = create_app(router, api_key=api_key)
    return TestClient(app, raise_server_exceptions=False)


CHAT_PAYLOAD = {
    "model": "baar",
    "messages": [{"role": "user", "content": "Say hello"}],
    "stream": False,
}


# ── /v1/models ────────────────────────────────────────────────────────────────

class TestModelsEndpoint:
    def test_returns_model_list(self):
        client = make_client(make_router())
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "baar"

    def test_model_includes_router_info(self):
        router = make_router(budget=0.05)
        client = make_client(router)
        resp = client.get("/v1/models")
        description = resp.json()["data"][0]["description"]
        assert "0.0500" in description
        assert router.small_model in description
        assert router.big_model in description


# ── /v1/chat/completions — non-streaming ─────────────────────────────────────

class TestNonStreaming:
    def test_returns_openai_compatible_response(self):
        router = make_router()
        client = make_client(router)
        with patch.object(router, "achat", new=AsyncMock(return_value="Hello!")):
            resp = client.post("/v1/chat/completions", json=CHAT_PAYLOAD)
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["message"]["role"] == "assistant"
        assert body["choices"][0]["message"]["content"] == "Hello!"
        assert body["choices"][0]["finish_reason"] == "stop"
        assert "usage" in body
        assert body["usage"]["total_tokens"] >= 0

    def test_response_has_required_openai_fields(self):
        router = make_router()
        client = make_client(router)
        with patch.object(router, "achat", new=AsyncMock(return_value="Hi")):
            resp = client.post("/v1/chat/completions", json=CHAT_PAYLOAD)
        body = resp.json()
        assert "id" in body
        assert body["id"].startswith("chatcmpl-")
        assert "created" in body
        assert "model" in body

    def test_extra_openai_fields_accepted(self):
        router = make_router()
        client = make_client(router)
        payload = {**CHAT_PAYLOAD, "temperature": 0.7, "top_p": 0.9, "user": "user-123"}
        with patch.object(router, "achat", new=AsyncMock(return_value="Ok")):
            resp = client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 200


# ── /v1/chat/completions — budget errors ─────────────────────────────────────

class TestBudgetErrors:
    def test_budget_exhausted_returns_402(self):
        router = make_router(budget=0.000001)
        client = make_client(router)
        with patch.object(router, "achat", side_effect=BudgetExhausted("broke", remaining=0.0)):
            resp = client.post("/v1/chat/completions", json=CHAT_PAYLOAD)
        assert resp.status_code == 402
        err = resp.json()["detail"]["error"]
        assert err["type"] == "budget_exhausted"
        assert err["code"] == "quota_exceeded"
        assert "remaining_usd" in err

    def test_task_rejected_returns_422(self):
        router = make_router()
        client = make_client(router)
        with patch.object(
            router, "achat",
            side_effect=TaskRejected(
                "low value", estimated_value=0.001, estimated_cost_usd=0.01
            ),
        ):
            resp = client.post("/v1/chat/completions", json=CHAT_PAYLOAD)
        assert resp.status_code == 422
        err = resp.json()["detail"]["error"]
        assert err["type"] == "task_rejected"
        assert err["code"] == "value_gate_reject"
        assert "estimated_value" in err


# ── /v1/chat/completions — streaming ─────────────────────────────────────────

class TestStreaming:
    @patch("baar.core.budget.cost_per_token", return_value=(0.000001, 0.000002))
    @patch("litellm.completion")
    def test_streaming_returns_sse(self, mock_comp, mock_cpt):
        chunks = ["Hello", ", ", "world", "!"]
        mock_comp.return_value = iter([
            MagicMock(choices=[MagicMock(delta=MagicMock(content=c))]) for c in chunks
        ] + [MagicMock(choices=[MagicMock(delta=MagicMock(content=None))])])

        router = make_router()

        async def fake_astream(task, *, messages=None):
            for c in chunks:
                yield c

        with patch.object(router, "astream_chat", side_effect=fake_astream):
            client = make_client(router)
            payload = {**CHAT_PAYLOAD, "stream": True}
            resp = client.post("/v1/chat/completions", json=payload)

        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

        lines = [l for l in resp.text.splitlines() if l.startswith("data:")]
        assert any("[DONE]" in l for l in lines)

        content_chunks = []
        for line in lines:
            raw = line[len("data:"):].strip()
            if raw == "[DONE]":
                continue
            obj = json.loads(raw)
            delta = obj["choices"][0]["delta"]
            if delta.get("content"):
                content_chunks.append(delta["content"])

        assert "".join(content_chunks) == "".join(chunks)

    @patch("baar.core.budget.cost_per_token", return_value=(0.000001, 0.000002))
    def test_streaming_budget_exhausted_sends_error_chunk(self, mock_cpt):
        router = make_router()

        async def fake_astream(task, *, messages=None):
            raise BudgetExhausted("no budget", remaining=0.0)
            yield  # make it an async generator

        with patch.object(router, "astream_chat", side_effect=fake_astream):
            client = make_client(router)
            payload = {**CHAT_PAYLOAD, "stream": True}
            resp = client.post("/v1/chat/completions", json=payload)

        assert resp.status_code == 200  # SSE always 200; error inside stream
        lines = [l for l in resp.text.splitlines() if l.startswith("data:")]
        error_lines = [l for l in lines if "budget_exhausted" in l]
        assert len(error_lines) >= 1


# ── Authentication ────────────────────────────────────────────────────────────

class TestAuthentication:
    def test_no_auth_required_when_api_key_not_set(self):
        client = make_client(make_router())
        resp = client.get("/v1/models")
        assert resp.status_code == 200

    def test_valid_api_key_passes(self):
        client = make_client(make_router(), api_key="secret")
        resp = client.get("/v1/models", headers={"Authorization": "Bearer secret"})
        assert resp.status_code == 200

    def test_missing_api_key_returns_401(self):
        client = make_client(make_router(), api_key="secret")
        resp = client.get("/v1/models")
        assert resp.status_code == 401

    def test_wrong_api_key_returns_401(self):
        client = make_client(make_router(), api_key="secret")
        resp = client.get("/v1/models", headers={"Authorization": "Bearer wrong"})
        assert resp.status_code == 401


# ── Helpers ───────────────────────────────────────────────────────────────────

class TestHelpers:
    def test_last_user_content_picks_last_user_message(self):
        messages = [
            ChatMessage(role="system", content="You are helpful"),
            ChatMessage(role="user", content="First"),
            ChatMessage(role="assistant", content="Reply"),
            ChatMessage(role="user", content="Second"),
        ]
        assert _last_user_content(messages) == "Second"

    def test_last_user_content_falls_back_to_last_message(self):
        messages = [ChatMessage(role="assistant", content="Hi")]
        assert _last_user_content(messages) == "Hi"


# ── Tool calling through the proxy ────────────────────────────────────────────

import litellm as _litellm

_TOOLS = [{
    "type": "function",
    "function": {
        "name": "lookup",
        "description": "Return the paragraph for an article title.",
        "parameters": {"type": "object", "properties": {"title": {"type": "string"}},
                       "required": ["title"]},
    },
}]


def _tool_call_response():
    return _litellm.ModelResponse(
        model="gpt-4o-mini",
        choices=[{
            "index": 0,
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1", "type": "function",
                    "function": {"name": "lookup", "arguments": "{\"title\": \"Paris\"}"},
                }],
            },
        }],
        usage={"prompt_tokens": 80, "completion_tokens": 12, "total_tokens": 92},
    )


def _tool_payload(**extra):
    return {
        "model": "baar",
        "messages": [
            {"role": "user", "content": "Where is the Louvre?"},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_0", "type": "function",
                "function": {"name": "lookup", "arguments": "{\"title\": \"Louvre\"}"}}]},
            {"role": "tool", "tool_call_id": "call_0", "content": "The Louvre is in Paris."},
        ],
        "tools": _TOOLS,
        "tool_choice": "auto",
        **extra,
    }


@patch("baar.core.budget.cost_per_token", return_value=(0.000001, 0.000002))
class TestToolCalling:
    def test_tools_forwarded_and_tool_calls_returned(self, _cpt):
        router = make_router()
        mock = AsyncMock(return_value=_tool_call_response())
        with patch("baar.router.litellm.acompletion", new=mock):
            resp = make_client(router).post("/v1/chat/completions", json=_tool_payload())
        assert resp.status_code == 200, resp.text
        sent = mock.call_args.kwargs
        assert sent["tools"] == _TOOLS
        assert sent["tool_choice"] == "auto"
        # null-content assistant message and tool result reach the provider intact
        assert sent["messages"][-2]["tool_calls"][0]["id"] == "call_0"
        assert sent["messages"][-1] == {"role": "tool", "tool_call_id": "call_0",
                                        "content": "The Louvre is in Paris."}
        body = resp.json()
        choice = body["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        assert choice["message"]["tool_calls"][0]["function"] == {
            "name": "lookup", "arguments": "{\"title\": \"Paris\"}"}
        assert body["usage"]["prompt_tokens"] == 80

    def test_tool_schema_counted_in_preflight(self, _cpt):
        router = make_router()
        seen = []
        import baar.router as br
        real = br.token_counter

        def spy(*a, **kw):
            seen.append(kw.get("tools"))
            return real(*a, **kw)

        with patch("baar.router.litellm.acompletion", new=AsyncMock(return_value=_tool_call_response())), \
             patch("baar.router.token_counter", side_effect=spy):
            make_client(router).post("/v1/chat/completions", json=_tool_payload())
        assert _TOOLS in seen

    def test_streaming_tool_calls(self, _cpt):
        router = make_router()
        with patch("baar.router.litellm.acompletion", new=AsyncMock(return_value=_tool_call_response())):
            resp = make_client(router).post("/v1/chat/completions", json=_tool_payload(stream=True))
        assert resp.status_code == 200
        events = [l[6:] for l in resp.text.splitlines() if l.startswith("data: ")]
        assert events[-1] == "[DONE]"
        chunks = [json.loads(e) for e in events[:-1]]
        deltas = [c["choices"][0]["delta"] for c in chunks]
        calls = [d["tool_calls"] for d in deltas if "tool_calls" in d]
        assert calls and calls[0][0]["index"] == 0
        assert calls[0][0]["function"]["name"] == "lookup"
        assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"

    def test_legacy_functions_mapped_to_tools(self, _cpt):
        router = make_router()
        mock = AsyncMock(return_value=_tool_call_response())
        payload = {"model": "baar",
                   "messages": [{"role": "user", "content": "Where is the Louvre?"}],
                   "functions": [_TOOLS[0]["function"]],
                   "function_call": {"name": "lookup"}}
        with patch("baar.router.litellm.acompletion", new=mock):
            resp = make_client(router).post("/v1/chat/completions", json=payload)
        assert resp.status_code == 200
        assert mock.call_args.kwargs["tools"] == _TOOLS
        assert mock.call_args.kwargs["tool_choice"] == {"type": "function",
                                                        "function": {"name": "lookup"}}

    def test_budget_exhausted_with_tools_returns_402(self, _cpt):
        router = make_router()
        with patch.object(router, "acomplete", side_effect=BudgetExhausted("broke", remaining=0.0)):
            resp = make_client(router).post("/v1/chat/completions", json=_tool_payload())
        assert resp.status_code == 402

    def test_null_content_message_accepted_without_tools(self, _cpt):
        router = make_router()
        with patch.object(router, "achat", new=AsyncMock(return_value="ok")):
            payload = _tool_payload()
            del payload["tools"], payload["tool_choice"]
            resp = make_client(router).post("/v1/chat/completions", json=payload)
        assert resp.status_code == 200

    def test_multipart_content_accepted(self, _cpt):
        router = make_router()
        with patch.object(router, "achat", new=AsyncMock(return_value="ok")) as m:
            resp = make_client(router).post("/v1/chat/completions", json={
                "model": "baar",
                "messages": [{"role": "user", "content": [{"type": "text", "text": "hi there"}]}]})
        assert resp.status_code == 200
        assert m.call_args.args[0] == "hi there"


def test_router_complete_returns_tool_calls_sync():
    router = make_router()
    with patch("baar.core.budget.cost_per_token", return_value=(0.000001, 0.000002)), \
         patch("baar.router.litellm.completion", return_value=_tool_call_response()) as mock:
        resp = router.complete("Where is the Louvre?", tools=_TOOLS)
        text = router.chat("Where is the Louvre?", tools=_TOOLS)
    assert resp.choices[0].message.tool_calls[0].function.name == "lookup"
    assert text == ""
    assert mock.call_args.kwargs["tools"] == _TOOLS
