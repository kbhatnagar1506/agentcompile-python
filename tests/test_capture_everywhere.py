"""Capture beyond chat.completions and messages: the Responses API through wrap(), and any
framework through the capturing httpx transport (no client to wrap)."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import anthropic
import httpx
import openai
import pytest

import agentcompile
from agentcompile import _capture
from agentcompile._decide import Settings
from agentcompile._transport import decode, events, provider_of

RESPONSE = {
    "id": "resp_1",
    "object": "response",
    "created_at": 0,
    "model": "gpt-x",
    "status": "completed",
    "output": [
        {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "get_order",
            "arguments": '{"order_id": "#W1"}',
            "status": "completed",
        }
    ],
    "parallel_tool_calls": True,
    "tool_choice": "auto",
    "tools": [],
}
COMPLETION = {
    "id": "c1",
    "object": "chat.completion",
    "created": 0,
    "model": "gpt-x",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "from the model"},
            "finish_reason": "stop",
        }
    ],
}
MESSAGE = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-x",
    "content": [{"type": "text", "text": "from the model"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 1, "output_tokens": 1},
}
TOOLS = [{"type": "function", "name": "get_order", "parameters": {"type": "object"}}]


def sse(events_: list[dict[str, Any]], done: bool = False) -> bytes:
    lines = [f"event: {e.get('type', 'message')}\ndata: {json.dumps(e)}\n\n" for e in events_]
    if done:
        lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


RESPONSE_EVENTS = [
    {
        "type": "response.created",
        "response": {**RESPONSE, "status": "in_progress", "output": []},
    },
    {"type": "response.output_item.added", "output_index": 0, "item": RESPONSE["output"][0]},
    {"type": "response.output_item.done", "output_index": 0, "item": RESPONSE["output"][0]},
    {"type": "response.completed", "response": RESPONSE},
]


class Service:
    """AgentCompile: records capture batches; any decide call is a test failure."""

    def __init__(self) -> None:
        self.batches: list[dict[str, Any]] = []
        self.decides = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/capture":
            self.batches.append(json.loads(request.content))
            return httpx.Response(200, json={"kept": 1})
        self.decides += 1
        return httpx.Response(200, json={"action": "forward", "reason": "x"})

    @property
    def records(self) -> list[dict[str, Any]]:
        return [r for b in self.batches for r in b["exchanges"]]


class Provider:
    """The model: answers each path with its own shape, streamed when asked."""

    def __init__(self, status: int = 200, encoding: str = "", preread: bool = False) -> None:
        self.calls: list[httpx.Request] = []
        self.status, self.encoding, self.preread = status, encoding, preread

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        body = json.loads(request.content) if request.content else {}
        path = request.url.path
        if self.status != 200:
            return httpx.Response(self.status, json={"error": {"message": "nope"}})
        if path.endswith("/responses"):
            if body.get("stream"):
                return self._sse(sse(RESPONSE_EVENTS))
            return self._json(RESPONSE)
        if path.endswith("/messages"):
            if body.get("stream"):
                return self._sse(sse(_anthropic_events()))
            return self._json(MESSAGE)
        if path.endswith("/chat/completions"):
            if body.get("stream"):
                return self._sse(sse(_chat_chunks(), done=True))
            return self._json(COMPLETION)
        return httpx.Response(200, json={"data": []})

    def _json(self, data: dict[str, Any]) -> httpx.Response:
        raw = json.dumps(data).encode()
        headers = {"content-type": "application/json"}
        if self.encoding == "gzip":
            raw, headers["content-encoding"] = gzip.compress(raw), "gzip"
        return httpx.Response(200, content=raw, headers=headers)

    def _sse(self, raw: bytes) -> httpx.Response:
        headers = {"content-type": "text/event-stream"}
        if self.encoding == "gzip":
            raw, headers["content-encoding"] = gzip.compress(raw), "gzip"
        if self.preread:
            return httpx.Response(200, content=raw, headers=headers)
        return httpx.Response(200, stream=Wire(raw), headers=headers)


class Wire(httpx.SyncByteStream, httpx.AsyncByteStream):
    """A body still on the wire, in small pieces (bytes given to httpx.Response arrive read)."""

    def __init__(self, raw: bytes) -> None:
        self.pieces = [raw[i : i + 7] for i in range(0, len(raw), 7)]

    def __iter__(self) -> Any:
        yield from self.pieces

    async def __aiter__(self) -> Any:
        for piece in self.pieces:
            yield piece


def _chat_chunks() -> list[dict[str, Any]]:
    base = {"id": "c1", "object": "chat.completion.chunk", "created": 0, "model": "gpt-x"}
    return [
        {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "from "}}]},
        {**base, "choices": [{"index": 0, "delta": {"content": "the model"}}]},
        {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ]


def _anthropic_events() -> list[dict[str, Any]]:
    return [
        {"type": "message_start", "message": {**MESSAGE, "content": [], "stop_reason": None}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "from the model"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 3},
        },
        {"type": "message_stop"},
    ]


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("AGENTCOMPILE_SCRUB_KEY", "test-scrub-key")
    yield
    _capture._ALL.clear()


def _wrapped(service: Service, provider: Provider, tmp_path: Path, **kw: Any) -> Any:
    real = openai.OpenAI(
        api_key="sk-test",
        base_url="http://provider.test/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(provider)),
    )
    return agentcompile.wrap(
        real,
        key="ack_acme.k",
        base_url="http://ac.test",
        trail=tmp_path / "trail.jsonl",
        http_client=httpx.Client(transport=httpx.MockTransport(service)),
        capture=True,
        **kw,
    )


def _trail(tmp_path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in (tmp_path / "trail.jsonl").read_text().splitlines()]


def _capturing(service: Service) -> _capture.Capturer:
    """A sender flush_all() reaches, unscrubbed so the tests can read what was sent."""
    return _capture.capturer_for(
        Settings("http://ac.test", "ack_acme.k", None, 1.0),
        httpx.Client(transport=httpx.MockTransport(service)),
        None,
    )


# -- the Responses API through wrap() -------------------------------------------------------


def test_a_responses_call_is_captured_in_its_own_shape_and_never_asked_about(
    tmp_path: Path,
) -> None:
    service, provider = Service(), Provider()
    client = _wrapped(service, provider, tmp_path)
    resp = client.responses.create(
        model="gpt-x",
        input=[{"role": "user", "content": "where is order #W1? I'm ann@example.com"}],
        instructions="You are a support agent.",
        tools=TOOLS,
        previous_response_id="resp_0",
        conversation_id="c1",
        customer_id="ann@example.com",
    )
    assert resp.output[0].name == "get_order"
    sent = json.loads(provider.calls[0].content)
    assert "conversation_id" not in sent and "customer_id" not in sent
    agentcompile.flush(2)
    (record,) = service.records
    assert service.decides == 0
    assert record["provider"] == "openai" and record["conversation_id"] == "c1"
    request = record["request"]
    assert set(request) == {"model", "input", "instructions", "tools", "previous_response_id"}
    assert "ann@example.com" not in json.dumps(record)  # scrubbed here, before sending
    assert record["end_user"].startswith("<email:")
    assert record["response"]["output"][0]["call_id"] == "call_1"
    event = _trail(tmp_path)[0]
    assert event["route"] == "unsupported" and event["reason"] == "responses api"


def test_a_streamed_responses_call_is_captured_whole(tmp_path: Path) -> None:
    service, provider = Service(), Provider()
    client = _wrapped(service, provider, tmp_path)
    stream = client.responses.create(
        model="gpt-x", input="where is my order?", stream=True, conversation_id="c1"
    )
    kinds = [event.type for event in stream]
    assert kinds[-1] == "response.completed"
    agentcompile.flush(2)
    (record,) = service.records
    assert record["stream"] is True and record["stream_complete"] is True
    assert record["response"]["id"] == "resp_1"
    assert record["response"]["output"][0]["name"] == "get_order"
    assert record["request"]["input"] == "where is my order?"


def test_a_responses_stream_stopped_early_keeps_the_items_it_finished() -> None:
    from agentcompile._assemble import assemble

    built = assemble("openai", RESPONSE_EVENTS[:3])
    assert built is not None
    assert built["id"] == "resp_1" and built["output"][0]["call_id"] == "call_1"


def test_without_a_conversation_a_responses_call_is_still_captured(tmp_path: Path) -> None:
    service, provider = Service(), Provider()
    _wrapped(service, provider, tmp_path).responses.create(model="gpt-x", input="hi")
    agentcompile.flush(2)
    assert service.records[0]["conversation_id"] is None
    assert _trail(tmp_path)[0]["route"] == "no-conversation"


async def test_the_async_client_captures_responses_too(tmp_path: Path) -> None:
    service, provider = Service(), Provider()

    async def answer(request: httpx.Request) -> httpx.Response:
        return provider(request)

    real = openai.AsyncOpenAI(
        api_key="sk-test",
        base_url="http://provider.test/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(answer)),
    )
    client = agentcompile.wrap(
        real,
        key="ack_acme.k",
        base_url="http://ac.test",
        trail=False,
        http_client=httpx.Client(transport=httpx.MockTransport(service)),
        capture=True,
    )
    await client.responses.create(model="gpt-x", input="hi", conversation_id="c9")
    stream = await client.responses.create(
        model="gpt-x", input="hi", stream=True, conversation_id="c9"
    )
    async for _ in stream:
        pass
    agentcompile.flush(2)
    assert [r["response"]["id"] for r in service.records] == ["resp_1", "resp_1"]
    assert service.records[1]["stream_complete"] is True


def test_chat_completions_still_decide_on_a_client_that_has_responses(tmp_path: Path) -> None:
    service, provider = Service(), Provider()
    client = _wrapped(service, provider, tmp_path)
    client.chat.completions.create(
        model="gpt-x", messages=[{"role": "user", "content": "hi"}], conversation_id="c1"
    )
    assert service.decides == 1


# -- any framework, through the transport -----------------------------------------------------


def _framework_client(service: Service, provider: Provider) -> openai.OpenAI:
    """A framework's own OpenAI client, built with the capturing transport: no wrap()."""
    transport = agentcompile.CaptureTransport(
        _capturing(service), httpx.MockTransport(provider)
    )
    return openai.OpenAI(
        api_key="sk-test",
        base_url="http://provider.test/v1",
        http_client=httpx.Client(transport=transport),
    )


def test_the_transport_captures_a_framework_call_in_its_conversation() -> None:
    service, provider = Service(), Provider()
    client = _framework_client(service, provider)
    with agentcompile.conversation("t1", customer="cust-7"):
        resp = client.chat.completions.create(
            model="gpt-x", messages=[{"role": "user", "content": "hi"}]
        )
    assert resp.choices[0].message.content == "from the model"
    _capture.flush_all(2)
    (record,) = service.records
    assert record["conversation_id"] == "t1" and record["end_user"] == "cust-7"
    assert record["request"]["messages"][0]["content"] == "hi"
    assert record["response"]["choices"][0]["message"]["content"] == "from the model"
    assert service.decides == 0  # capture only


def test_the_transport_assembles_streams_in_every_shape() -> None:
    service, provider = Service(), Provider()
    client = _framework_client(service, provider)
    with agentcompile.conversation("t2"):
        chunks = list(
            client.chat.completions.create(
                model="gpt-x", messages=[{"role": "user", "content": "hi"}], stream=True
            )
        )
        list(client.responses.create(model="gpt-x", input="hi", stream=True))
    assert len(chunks) == 3
    _capture.flush_all(2)
    chat, responses = service.records
    assert chat["response"]["choices"][0]["message"]["content"] == "from the model"
    assert chat["stream_complete"] is True
    assert responses["response"]["id"] == "resp_1"


def test_the_transport_reads_compressed_answers() -> None:
    service, provider = Service(), Provider(encoding="gzip")
    client = _framework_client(service, provider)
    with agentcompile.conversation("t3"):
        resp = client.chat.completions.create(
            model="gpt-x", messages=[{"role": "user", "content": "hi"}]
        )
        streamed = list(
            client.chat.completions.create(
                model="gpt-x", messages=[{"role": "user", "content": "hi"}], stream=True
            )
        )
    assert resp.choices[0].message.content == "from the model" and len(streamed) == 3
    _capture.flush_all(2)
    assert [r["response"]["choices"][0]["message"]["content"] for r in service.records] == [
        "from the model",
        "from the model",
    ]


def test_the_transport_captures_anthropic_calls_over_httpx2() -> None:
    httpx2 = pytest.importorskip("httpx2")  # only the newer provider SDKs bring it

    service, provider = Service(), Provider()

    def answer(request: Any) -> Any:
        mine = provider(
            httpx.Request(request.method, str(request.url), content=request.content)
        )
        raw = mine.read()
        on_the_wire = iter([raw[i : i + 7] for i in range(0, len(raw), 7)])
        return httpx2.Response(
            mine.status_code, headers=dict(mine.headers), content=on_the_wire
        )

    sync, _ = agentcompile._transport.classes("httpx2")
    transport = sync(_capturing(service), httpx2.MockTransport(answer))
    client = anthropic.Anthropic(
        api_key="sk-test",
        base_url="http://provider.test",
        http_client=httpx2.Client(transport=transport),
    )
    with agentcompile.conversation("t4"):
        client.messages.create(
            model="claude-x", max_tokens=10, messages=[{"role": "user", "content": "hi"}]
        )
        with client.messages.stream(
            model="claude-x", max_tokens=10, messages=[{"role": "user", "content": "hi"}]
        ) as stream:
            assert stream.get_final_text() == "from the model"
    _capture.flush_all(2)
    whole, streamed = service.records
    assert whole["provider"] == streamed["provider"] == "anthropic"
    assert streamed["response"]["content"][0]["text"] == "from the model"
    assert streamed["stream_complete"] is True


def test_the_library_is_taken_from_the_wrapped_transport() -> None:
    httpx2 = pytest.importorskip("httpx2")  # only the newer provider SDKs bring it

    from agentcompile._transport import lib_of

    assert lib_of(httpx2.HTTPTransport(), None) == "httpx2"
    assert lib_of(httpx.HTTPTransport(), None) == "httpx"
    assert lib_of(None, None) == "httpx" and lib_of(None, "httpx2") == "httpx2"
    made = agentcompile.transport(httpx2.HTTPTransport(), key="ack_acme.k")
    assert isinstance(made, httpx2.BaseTransport)


def test_the_transport_leaves_other_calls_and_failures_alone() -> None:
    service = Service()
    client = _framework_client(service, Provider())
    client.models.list()  # not a model call
    failing = _framework_client(service, Provider(status=400))
    with pytest.raises(openai.BadRequestError):
        failing.chat.completions.create(
            model="gpt-x", messages=[{"role": "user", "content": "hi"}]
        )
    _capture.flush_all(2)
    assert service.records == []


async def test_the_async_transport_captures_whole_and_streamed() -> None:
    service, provider = Service(), Provider()

    async def answer(request: httpx.Request) -> httpx.Response:
        return provider(request)

    transport = agentcompile.AsyncCaptureTransport(
        _capturing(service), httpx.MockTransport(answer)
    )
    client = openai.AsyncOpenAI(
        api_key="sk-test",
        base_url="http://provider.test/v1",
        http_client=httpx.AsyncClient(transport=transport),
    )
    with agentcompile.conversation("t5"):
        await client.responses.create(model="gpt-x", input="hi")
        stream = await client.chat.completions.create(
            model="gpt-x", messages=[{"role": "user", "content": "hi"}], stream=True
        )
        async for _ in stream:
            pass
    _capture.flush_all(2)
    whole, streamed = service.records
    assert whole["conversation_id"] == streamed["conversation_id"] == "t5"
    assert streamed["response"]["choices"][0]["message"]["content"] == "from the model"


def test_a_stream_a_transport_already_read_is_captured_from_its_body() -> None:
    service = Service()
    client = _framework_client(service, Provider(preread=True))
    with agentcompile.conversation("t7"):
        list(
            client.chat.completions.create(
                model="gpt-x", messages=[{"role": "user", "content": "hi"}], stream=True
            )
        )
    _capture.flush_all(2)
    (record,) = service.records
    assert record["response"]["choices"][0]["message"]["content"] == "from the model"


def test_a_stream_belongs_to_the_conversation_it_was_started_in() -> None:
    service, provider = Service(), Provider()
    client = _framework_client(service, provider)
    with agentcompile.conversation("started-here"):
        stream = client.chat.completions.create(
            model="gpt-x", messages=[{"role": "user", "content": "hi"}], stream=True
        )
    with agentcompile.conversation("read-here"):
        list(stream)
    _capture.flush_all(2)
    assert service.records[0]["conversation_id"] == "started-here"


def test_a_stream_closed_early_is_captured_once_and_marked() -> None:
    service, provider = Service(), Provider()
    client = _framework_client(service, provider)
    with agentcompile.conversation("t6"):
        stream = client.chat.completions.create(
            model="gpt-x", messages=[{"role": "user", "content": "hi"}], stream=True
        )
        next(iter(stream))
        stream.close()
    _capture.flush_all(2)
    (record,) = service.records
    assert record["stream_complete"] is False


def test_http_client_helpers_build_capturing_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTCOMPILE_KEY", "ack_acme.k")
    sync, async_ = agentcompile.http_client(), agentcompile.async_http_client()
    assert isinstance(sync._transport, agentcompile.CaptureTransport)
    assert isinstance(async_._transport, agentcompile.AsyncCaptureTransport)
    assert sync.timeout.read == 600.0
    httpx2 = pytest.importorskip("httpx2")  # only the newer provider SDKs bring it
    assert isinstance(agentcompile.http_client(lib="httpx2"), httpx2.Client)
    assert isinstance(agentcompile.async_http_client(lib="httpx2"), httpx2.AsyncClient)


# -- the pieces ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "path", "expected"),
    [
        ("POST", "/v1/chat/completions", "openai"),
        ("POST", "/v1beta/openai/chat/completions", "openai"),
        ("POST", "/openai/deployments/d/chat/completions", "openai"),
        ("POST", "/v1/responses", "openai"),
        ("POST", "/v1/messages", "anthropic"),
        ("POST", "/v1/messages/count_tokens", None),
        ("POST", "/v1/threads/t1/messages", None),
        ("GET", "/v1/responses", None),
        ("POST", "/v1/embeddings", None),
    ],
)
def test_which_requests_are_model_calls(method: str, path: str, expected: str | None) -> None:
    assert provider_of(httpx.Request(method, f"http://x{path}")) == expected


def test_decode_undoes_gzip_deflate_and_refuses_the_unknown() -> None:
    import zlib

    assert decode(gzip.compress(b"hi"), "gzip") == b"hi"
    assert decode(zlib.compress(b"hi"), "deflate") == b"hi"
    raw = zlib.compressobj(wbits=-15)
    assert decode(raw.compress(b"hi") + raw.flush(), "deflate") == b"hi"
    assert decode(b"hi", "") == b"hi" and decode(b"hi", "identity") == b"hi"
    assert decode(b"hi", "zstd") is None


def test_events_reads_server_sent_events() -> None:
    text = 'event: a\ndata: {"x": 1}\n\n: comment\n\ndata: [DONE]\n\ndata: not json\n\n'
    assert events(text + 'data: {"y":\ndata: 2}\n\n') == [{"x": 1}, {"y": 2}]
