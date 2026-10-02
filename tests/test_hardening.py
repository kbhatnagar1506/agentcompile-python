"""What keeps the wrapper safe in production: a decision is bounded in time and tells the
server how long it waits, a sick service is skipped, a request a compiled answer couldn't honour
goes to the model, streams read like the SDKs' own, and nothing grows without bound."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import httpx
import openai
import pytest

import agentcompile
from agentcompile import _capture, _outcome
from agentcompile._assemble import CapturingStream
from agentcompile._decide import AsyncDecider, Breaker, Settings
from agentcompile._streams import AsyncCompiledStream, CompiledStream
from agentcompile._trail import Trail

CALL = {
    "action": "tool_call",
    "tool": "get_order_details",
    "args": {"order_id": "#W1"},
    "call_id": "call_1",
}
MESSAGES = [{"role": "user", "content": "cancel order #W1"}]
TOOLS = [
    {
        "type": "function",
        "function": {"name": "get_order_details", "parameters": {"type": "object"}},
    }
]
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


class Service:
    def __init__(self, decision: Any = CALL, status: int = 200, sleep: float = 0.0) -> None:
        self.decision, self.status, self.sleep = decision, status, sleep
        self.decides: list[httpx.Request] = []
        self.captures: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/capture":
            self.captures.append(json.loads(request.content))
            return httpx.Response(200, json={"kept": 1})
        self.decides.append(request)
        if self.sleep:
            time.sleep(self.sleep)
        return httpx.Response(self.status, json=self.decision)


def _client(service: Service, tmp_path: Path, **kw: Any) -> tuple[Any, list[Any]]:
    provider_calls: list[Any] = []

    def provider(request: httpx.Request) -> httpx.Response:
        provider_calls.append(json.loads(request.content))
        return httpx.Response(200, json=COMPLETION)

    real = openai.OpenAI(
        api_key="sk-test",
        base_url="http://provider.test/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(provider)),
    )
    client = agentcompile.wrap(
        real,
        key="ack_acme.k",
        base_url="http://ac.test",
        trail=tmp_path / "trail.jsonl",
        http_client=httpx.Client(transport=httpx.MockTransport(service)),
        **kw,
    )
    return client, provider_calls


def _trail(tmp_path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (tmp_path / "trail.jsonl").read_text().splitlines()]


@pytest.fixture(autouse=True)
def _no_leftover_senders() -> Any:
    yield
    _capture._ALL.clear()


# -- what the server is told ---------------------------------------------------------------


def test_the_server_hears_how_long_we_wait_and_whether_it_is_shadow(tmp_path: Path) -> None:
    service = Service()
    live, _ = _client(service, tmp_path, timeout=1.5)
    live.chat.completions.create(model="m", messages=MESSAGES, tools=TOOLS, conversation_id="c")
    shadow, _ = _client(service, tmp_path, mode="shadow")
    shadow.chat.completions.create(
        model="m", messages=MESSAGES, tools=TOOLS, conversation_id="c"
    )
    first, second = service.decides
    assert first.headers["x-agentcompiler-deadline-ms"] == "1500"
    assert "x-agentcompiler-mode" not in first.headers
    assert second.headers["x-agentcompiler-mode"] == "shadow"


# -- requests a compiled answer can't honour --------------------------------------------------


@pytest.mark.parametrize(
    ("extra", "why"),
    [
        ({"n": 2}, "n"),
        ({"tool_choice": "required"}, "tool_choice"),
        ({"tool_choice": {"type": "function", "function": {"name": "x"}}}, "tool_choice"),
        ({"response_format": {"type": "json_object"}}, "response_format"),
    ],
)
def test_a_request_we_could_not_honour_goes_to_the_model_unasked(
    tmp_path: Path, extra: dict[str, Any], why: str
) -> None:
    service = Service()
    client, provider = _client(service, tmp_path)
    resp = client.chat.completions.create(
        model="m", messages=MESSAGES, tools=TOOLS, conversation_id="c", **extra
    )
    assert resp.choices[0].message.content == "from the model"
    assert service.decides == [] and len(provider) == 1
    event = _trail(tmp_path)[0]
    assert event["route"] == "unsupported" and event["reason"] == why


def test_auto_tool_choice_and_text_format_are_still_answered(tmp_path: Path) -> None:
    service = Service()
    client, provider = _client(service, tmp_path)
    resp = client.chat.completions.create(
        model="m",
        messages=MESSAGES,
        tools=TOOLS,
        tool_choice="auto",
        response_format={"type": "text"},
        n=1,
        conversation_id="c",
    )
    assert resp.choices[0].message.tool_calls and provider == []


def test_a_tool_the_agent_did_not_offer_is_never_answered(tmp_path: Path) -> None:
    service = Service()
    client, provider = _client(service, tmp_path)
    other = [{"type": "function", "function": {"name": "cancel_order"}}]
    resp = client.chat.completions.create(
        model="m", messages=MESSAGES, tools=other, conversation_id="c"
    )
    assert resp.choices[0].message.content == "from the model" and len(provider) == 1
    event = _trail(tmp_path)[0]
    assert event["route"] == "fail-open" and event["reason"] == "tool not offered"


# -- bounded in time ---------------------------------------------------------------------------


def test_a_slow_service_costs_at_most_the_timeout(tmp_path: Path) -> None:
    service = Service(sleep=1.0)
    client, provider = _client(service, tmp_path, timeout=0.2)
    started = time.perf_counter()
    resp = client.chat.completions.create(
        model="m", messages=MESSAGES, tools=TOOLS, conversation_id="c"
    )
    assert time.perf_counter() - started < 0.6
    assert resp.choices[0].message.content == "from the model" and len(provider) == 1
    event = _trail(tmp_path)[0]
    assert event["route"] == "fail-open" and event["reason"] == "deadline"


async def test_the_async_client_is_bounded_too() -> None:
    async def slow(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(1.0)
        return httpx.Response(200, json=CALL)

    decider = AsyncDecider(
        Settings("http://ac.test", "k", None, 0.2),
        httpx.AsyncClient(transport=httpx.MockTransport(slow)),
    )
    started = time.perf_counter()
    decision, _, error = await decider.decide("openai", "c", {"messages": MESSAGES})
    assert decision is None and error == "deadline"
    assert time.perf_counter() - started < 0.6


def test_a_sick_service_is_skipped_then_tried_again(tmp_path: Path) -> None:
    service = Service(status=503)
    client, provider = _client(service, tmp_path)

    def call() -> None:
        client.chat.completions.create(
            model="m", messages=MESSAGES, tools=TOOLS, conversation_id="c"
        )

    for _ in range(5):
        call()
    assert len(service.decides) == 5
    call()  # open: straight to the model, not asked
    assert len(service.decides) == 5 and len(provider) == 6
    assert _trail(tmp_path)[-1]["reason"] == "circuit open"


def test_the_breaker_lets_one_call_through_after_a_while() -> None:
    now = [0.0]
    breaker = Breaker(threshold=2, open_for=10.0, clock=lambda: now[0])
    breaker.record(False)
    assert breaker.allow()
    breaker.record(False)
    assert not breaker.allow()
    now[0] = 11.0
    assert breaker.allow() and not breaker.allow()  # one trial at a time
    breaker.record(True)
    assert breaker.allow() and breaker.allow()  # closed again
    breaker.record(False)
    assert breaker.allow()  # one failure isn't a pattern


# -- streams read like the SDKs' own ----------------------------------------------------------


def test_next_reads_a_compiled_stream_from_where_it_is() -> None:
    stream = CompiledStream([1, 2, 3])
    assert next(stream) == 1
    assert list(stream) == [2, 3]


async def test_anext_works_without_iterating_first() -> None:
    stream = AsyncCompiledStream([1, 2])
    assert await stream.__anext__() == 1
    assert [x async for x in stream] == [2]


def test_next_reads_a_captured_stream_and_capture_still_sees_it_whole() -> None:
    seen: list[Any] = []
    stream = CapturingStream(iter([{"a": 1}, {"a": 2}]), lambda c, done: seen.append((c, done)))
    assert next(stream) == {"a": 1}
    assert list(stream) == [{"a": 2}]
    assert seen == [([{"a": 1}, {"a": 2}], True)]


def test_a_compiled_stream_answers_next_from_the_wrapped_client(tmp_path: Path) -> None:
    client, _ = _client(Service(), tmp_path)
    stream = client.chat.completions.create(
        model="m", messages=MESSAGES, tools=TOOLS, conversation_id="c", stream=True
    )
    first = next(stream)
    assert first.choices[0].delta.tool_calls[0].function.name == "get_order_details"


# -- nothing grows without bound ---------------------------------------------------------


def test_the_trail_moves_aside_when_full(tmp_path: Path) -> None:
    trail = Trail(tmp_path / "trail.jsonl")
    trail.max_bytes = 200
    for n in range(20):
        trail.record(route="forwarded", n=n)
    moved = tmp_path / "trail.jsonl.1"
    assert moved.exists() and moved.stat().st_size <= 200 + 100  # one line past the limit
    current = tmp_path / "trail.jsonl"
    assert not current.exists() or current.stat().st_size <= 200 + 100


def test_capture_batches_stay_under_the_byte_limit_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_capture, "MAX_BATCH_BYTES", 1100)
    monkeypatch.setattr(_capture, "MAX_RECORD_BYTES", 2000)
    service = Service()
    capturer = _capture.Capturer(
        Settings("http://ac.test", "ack_acme.k", None, 1.0),
        httpx.Client(transport=httpx.MockTransport(service)),
    )
    capturer._start = lambda: None  # type: ignore[method-assign]
    for n in range(4):
        capturer.add("openai", f"c{n}", {"messages": [{"content": "x" * 300}]}, {"n": n})
    capturer.add("openai", "huge", {"messages": [{"content": "x" * 3000}]}, {})
    capturer.flush()
    sent = [r["conversation_id"] for batch in service.captures for r in batch["exchanges"]]
    assert sent == ["c0", "c1", "c2", "c3"] and len(service.captures) == 2
    assert capturer.dropped == 1


def test_wrapping_again_reuses_the_sender_and_keeps_one_destination(tmp_path: Path) -> None:
    service = Service()
    http = httpx.Client(transport=httpx.MockTransport(service))
    for _ in range(3):
        agentcompile.wrap(
            openai.OpenAI(api_key="sk-test"),
            key="ack_acme.k",
            base_url="http://ac.test",
            trail=False,
            http_client=http,
            capture=True,
            scrub=False,
        )
    assert len(_capture._ALL) == 1
    assert len(_outcome._settings) == 1
