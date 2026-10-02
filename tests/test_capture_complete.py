"""Capture, completed: streamed answers captured whole, the customer id, and outcomes."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import anthropic
import httpx
import openai
import pytest

import agentcompile
from agentcompile import _capture, _outcome
from agentcompile._assemble import assemble

MESSAGES = [{"role": "user", "content": "cancel order #W1"}]


def _sse(events: list[tuple[str | None, Any]]) -> bytes:
    out = []
    for name, data in events:
        line = f"event: {name}\n" if name else ""
        out.append(f"{line}data: {data if isinstance(data, str) else json.dumps(data)}\n\n")
    return "".join(out).encode()


def _chunk(delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
    return {
        "id": "s1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "m",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


OPENAI_STREAM = _sse(
    [
        (None, _chunk({"role": "assistant", "content": "Let me "})),
        (None, _chunk({"content": "check."})),
        (
            None,
            _chunk(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "t1",
                            "type": "function",
                            "function": {"name": "get_order", "arguments": '{"id":'},
                        }
                    ]
                }
            ),
        ),
        (None, _chunk({"tool_calls": [{"index": 0, "function": {"arguments": '"#W1"}'}}]})),
        (None, _chunk({}, "tool_calls")),
        (None, "[DONE]"),
    ]
)

ANTHROPIC_STREAM = _sse(
    [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg1",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 5, "output_tokens": 0},
                },
            },
        ),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "On it."},
            },
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {
                    "type": "tool_use",
                    "id": "tu1",
                    "name": "get_order",
                    "input": {},
                },
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": '{"id": "#W1"}'},
            },
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 1}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                "usage": {"output_tokens": 9},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
)


class Service:
    def __init__(self) -> None:
        self.captured: list[dict[str, Any]] = []
        self.outcomes: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        if request.url.path == "/v1/capture":
            self.captured += body["exchanges"]
        elif request.url.path == "/v1/outcome":
            self.outcomes += body["outcomes"]
        else:
            return httpx.Response(200, json={"action": "forward", "reason": "x"})
        return httpx.Response(200, json={})


def _anthropic_provider(body: bytes) -> Any:
    """anthropic 1.x ships its own httpx fork (httpx2), as in test_wrap.py."""
    try:
        import httpx2 as hx
    except ImportError:  # older anthropic: plain httpx
        hx = httpx
    headers = {"content-type": "text/event-stream"}
    return hx.Client(
        transport=hx.MockTransport(lambda r: hx.Response(200, content=body, headers=headers))
    )


def _provider(body: bytes) -> httpx.MockTransport:
    return httpx.MockTransport(
        lambda r: httpx.Response(
            200, content=body, headers={"content-type": "text/event-stream"}
        )
    )


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("AGENTCOMPILE_SCRUB_KEY", "k")
    yield
    _capture._ALL.clear()


def _openai(service: Service, body: bytes = OPENAI_STREAM) -> Any:
    real = openai.OpenAI(
        api_key="sk",
        base_url="http://p.test/v1",
        http_client=httpx.Client(transport=_provider(body)),
    )
    return agentcompile.wrap(
        real,
        base_url="http://ac.test",
        trail=False,
        capture=True,
        http_client=httpx.Client(transport=httpx.MockTransport(service)),
    )


def test_a_streamed_openai_answer_is_captured_whole() -> None:
    service = Service()
    stream = _openai(service).chat.completions.create(
        model="m", messages=MESSAGES, stream=True, conversation_id="c1"
    )
    seen = [chunk.choices[0].delta for chunk in stream]  # the agent reads it as always
    assert len(seen) == 5
    agentcompile.flush(5)
    [record] = service.captured
    message = record["response"]["choices"][0]["message"]
    assert message["content"] == "Let me check."
    assert message["tool_calls"][0]["function"] == {
        "name": "get_order",
        "arguments": '{"id":"#W1"}',
    }
    assert record["response"]["choices"][0]["finish_reason"] == "tool_calls"
    assert record["stream"] is True and record["stream_complete"] is True


def test_a_stream_the_agent_stops_reading_is_marked_incomplete() -> None:
    service = Service()
    stream = _openai(service).chat.completions.create(
        model="m", messages=MESSAGES, stream=True, conversation_id="c1"
    )
    for _ in stream:
        break
    stream.close()
    agentcompile.flush(5)
    [record] = service.captured
    assert record["stream_complete"] is False


def test_a_streamed_anthropic_answer_is_captured_whole() -> None:
    service = Service()
    real = anthropic.Anthropic(
        api_key="sk",
        base_url="http://p.test",
        http_client=_anthropic_provider(ANTHROPIC_STREAM),
    )
    client = agentcompile.wrap(
        real,
        base_url="http://ac.test",
        trail=False,
        capture=True,
        http_client=httpx.Client(transport=httpx.MockTransport(service)),
    )
    with client.messages.create(
        model="claude", max_tokens=50, messages=MESSAGES, stream=True, conversation_id="c1"
    ) as stream:
        kinds = [event.type for event in stream]
    assert kinds[0] == "message_start" and kinds[-1] == "message_stop"
    agentcompile.flush(5)
    [record] = service.captured
    response = record["response"]
    assert response["content"][0] == {"type": "text", "text": "On it."}
    assert response["content"][1]["input"] == {"id": "#W1"}
    assert response["stop_reason"] == "tool_use" and response["usage"]["output_tokens"] == 9


def test_async_streams_are_captured_too() -> None:
    service = Service()

    async def run() -> list[Any]:
        real = openai.AsyncOpenAI(
            api_key="sk",
            base_url="http://p.test/v1",
            http_client=httpx.AsyncClient(transport=_provider(OPENAI_STREAM)),
        )
        client = agentcompile.wrap(real, base_url="http://ac.test", trail=False, capture=True)
        list(_capture._ALL.values())[-1]._http = httpx.Client(
            transport=httpx.MockTransport(service)
        )
        stream = await client.chat.completions.create(
            model="m", messages=MESSAGES, stream=True, conversation_id="c1"
        )
        return [chunk async for chunk in stream]

    chunks = asyncio.run(run())
    assert len(chunks) == 5
    agentcompile.flush(5)
    [record] = service.captured
    assert record["response"]["choices"][0]["message"]["content"] == "Let me check."


def test_a_compiled_stream_is_captured_whole() -> None:
    class Compiling(Service):
        def __call__(self, request: httpx.Request) -> httpx.Response:
            if request.url.path == "/v1/decide":
                return httpx.Response(200, json={"action": "say", "text": "Done."})
            return super().__call__(request)

    service = Compiling()
    stream = _openai(service).chat.completions.create(
        model="m", messages=MESSAGES, stream=True, conversation_id="c1"
    )
    list(stream)
    agentcompile.flush(5)
    [record] = service.captured
    assert record["response"]["choices"][0]["message"]["content"] == "Done."


def test_the_customer_id_is_captured_scrubbed() -> None:
    service = Service()
    client = _openai(service, OPENAI_STREAM)
    with agentcompile.conversation("c1", customer="mia@example.com"):
        list(client.chat.completions.create(model="m", messages=MESSAGES, stream=True))
    list(
        client.chat.completions.create(
            model="m",
            messages=MESSAGES,
            stream=True,
            conversation_id="c2",
            customer_id="cust_42",
        )
    )
    agentcompile.flush(5)
    first, second = service.captured
    assert first["end_user"].startswith("<email:") and "mia" not in json.dumps(first)
    assert second["end_user"] == "cust_42"


def test_an_outcome_is_reported_in_the_background() -> None:
    service = Service()
    _openai(service)
    _outcome._settings[-1] = agentcompile.Settings("http://ac.test", "ack_acme.k", None, 2.0)

    real_client = httpx.Client

    class Routed(real_client):  # type: ignore[misc,valid-type]
        def __init__(self, **kw: Any) -> None:
            super().__init__(transport=httpx.MockTransport(service), **kw)

    import agentcompile._outcome as module

    module.httpx.Client = Routed  # type: ignore[misc]
    try:
        agentcompile.outcome("c1", "escalated", note="customer asked for a human")
        agentcompile.flush(5)
    finally:
        module.httpx.Client = real_client  # type: ignore[misc]
    [record] = service.outcomes
    assert record["conversation_id"] == "c1" and record["outcome"] == "escalated"
    assert record["note"] == "customer asked for a human"


def test_an_unknown_outcome_is_refused() -> None:
    with pytest.raises(ValueError):
        agentcompile.outcome("c1", "fine")


def test_assembly_never_breaks_the_agent() -> None:
    assert assemble("openai", [{"choices": "nonsense"}]) is None
    assert assemble("anthropic", [{"type": "content_block_delta"}]) is None
