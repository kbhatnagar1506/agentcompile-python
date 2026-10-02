"""Capture (wrap(..., capture=True)): each call's request and answer reach AgentCompile in the
background, in the importer's exchange shape, and never slow or break the call."""

from __future__ import annotations

import json
from typing import Any

import httpx
import openai
import pytest

import agentcompile
from agentcompile import _capture

MESSAGES = [{"role": "user", "content": "cancel order #W1"}]
FORWARD = {"action": "forward", "reason": "no job matched"}
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
    """AgentCompile: answers decide with forward, records capture batches."""

    def __init__(self, capture_status: int = 200) -> None:
        self.batches: list[dict[str, Any]] = []
        self.capture_status = capture_status

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/capture":
            self.batches.append(json.loads(request.content))
            assert request.headers["x-agentcompiler-key"] == "ack_acme.k"
            assert "x-agentcompiler-conversation" not in request.headers
            return httpx.Response(self.capture_status, json={"kept": 1})
        return httpx.Response(200, json=FORWARD)


def _client(service: Service, **kw: Any) -> Any:
    provider = openai.OpenAI(
        api_key="sk-test",
        base_url="http://provider.test/v1",
        http_client=httpx.Client(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json=COMPLETION))
        ),
    )
    return agentcompile.wrap(
        provider,
        key="ack_acme.k",
        base_url="http://ac.test",
        trail=False,
        http_client=httpx.Client(transport=httpx.MockTransport(service)),
        **kw,
    )


@pytest.fixture(autouse=True)
def _no_leftover_senders() -> Any:
    yield
    _capture._ALL.clear()


def test_capture_is_off_unless_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENTCOMPILE_CAPTURE", raising=False)
    service = Service()
    _client(service).chat.completions.create(
        model="gpt-x", messages=MESSAGES, conversation_id="c1"
    )
    agentcompile.flush(2)
    assert service.batches == []


def test_each_call_is_sent_in_the_importers_exchange_shape() -> None:
    service = Service()
    client = _client(service, capture=True)
    for _ in range(3):
        client.chat.completions.create(model="gpt-x", messages=MESSAGES, conversation_id="c1")
    agentcompile.flush(5)
    sent = [x for batch in service.batches for x in batch["exchanges"]]
    assert len(sent) == 3
    first = sent[0]
    assert first["provider"] == "openai" and first["conversation_id"] == "c1"
    assert first["request"]["messages"] == MESSAGES and first["request"]["model"] == "gpt-x"
    assert first["response"]["choices"][0]["message"]["content"] == "from the model"
    assert first["timestamp"]
    assert "conversation_id" not in first["request"]  # our keyword never reaches the model


def test_the_environment_turns_capture_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTCOMPILE_CAPTURE", "1")
    service = Service()
    _client(service).chat.completions.create(
        model="gpt-x", messages=MESSAGES, conversation_id="c1"
    )
    agentcompile.flush(5)
    assert len(service.batches) == 1


def test_a_failing_service_never_breaks_the_call() -> None:
    service = Service(capture_status=500)
    client = _client(service, capture=True)
    resp = client.chat.completions.create(
        model="gpt-x", messages=MESSAGES, conversation_id="c1"
    )
    assert resp.choices[0].message.content == "from the model"
    agentcompile.flush(5)
    assert (
        list(_capture._ALL.values())[-1].failed == 1
        and list(_capture._ALL.values())[-1].sent == 0
    )


def test_a_full_queue_drops_the_oldest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_capture, "MAX_QUEUE", 2)
    settings = agentcompile.Settings("http://ac.test", "ack_acme.k", None, 1.0)
    capturer = _capture.Capturer(
        settings, httpx.Client(transport=httpx.MockTransport(Service()))
    )
    capturer._start = lambda: None  # type: ignore[method-assign]
    for n in range(3):
        capturer.add("openai", f"c{n}", {"messages": []}, {"n": n})
    assert capturer.dropped == 1
    assert [r["conversation_id"] for r in capturer._queue] == ["c1", "c2"]


def test_a_streamed_answer_is_marked_not_captured() -> None:
    settings = agentcompile.Settings("http://ac.test", "ack_acme.k", None, 1.0)
    capturer = _capture.Capturer(
        settings, httpx.Client(transport=httpx.MockTransport(Service()))
    )
    capturer._start = lambda: None  # type: ignore[method-assign]
    capturer.add("openai", "c1", {"messages": MESSAGES, "stream": True}, object(), stream=True)
    [record] = list(capturer._queue)
    assert record["response"] is None and record["stream"] is True


def test_a_record_that_cannot_be_built_is_dropped_quietly() -> None:
    settings = agentcompile.Settings("http://ac.test", "ack_acme.k", None, 1.0)
    capturer = _capture.Capturer(
        settings, httpx.Client(transport=httpx.MockTransport(Service()))
    )
    capturer.add("openai", "c1", None, None)  # type: ignore[arg-type]
    assert capturer.dropped == 1
