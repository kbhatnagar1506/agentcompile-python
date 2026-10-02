"""agentcompile.wrap against a fake decision service and a fake provider (httpx MockTransport):
no network, the official openai and anthropic clients end to end."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anthropic
import httpx
import openai
import pytest

import agentcompile

CALL = {
    "action": "tool_call",
    "tool": "get_order_details",
    "args": {"order_id": "#W1"},
    "call_id": "call_1",
}
SAY = {"action": "say", "text": "Order #W1 is cancelled."}
FORWARD = {"action": "forward", "reason": "no job matched"}


class Fake:
    """The decision service and the provider, recording what each received."""

    def __init__(
        self,
        decision: Any = FORWARD,
        decide_status: int = 200,
        decide_error: Exception | None = None,
    ) -> None:
        self.decision, self.decide_status, self.decide_error = (
            decision,
            decide_status,
            decide_error,
        )
        self.decide_requests: list[httpx.Request] = []
        self.provider_requests: list[dict[str, Any]] = []

    def decide(self, request: httpx.Request) -> httpx.Response:
        self.decide_requests.append(request)
        if self.decide_error:
            raise self.decide_error
        return httpx.Response(self.decide_status, json=self.decision)

    def provider(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.provider_requests.append(body)
        if request.url.path.endswith("/messages"):
            return httpx.Response(
                200,
                json={
                    "id": "msg_real",
                    "type": "message",
                    "role": "assistant",
                    "model": body["model"],
                    "content": [{"type": "text", "text": "from the model"}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 5, "output_tokens": 3},
                },
            )
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-real",
                "object": "chat.completion",
                "created": 1,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "from the model"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )


def _openai(fake: Fake, tmp_path: Path, **kw: Any) -> Any:
    real = openai.OpenAI(
        api_key="sk-test",
        base_url="http://provider.test/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(fake.provider)),
    )
    return agentcompile.wrap(
        real,
        key="ack_acme.k",
        base_url="http://ac.test",
        trail=tmp_path / "trail.jsonl",
        http_client=httpx.Client(transport=httpx.MockTransport(fake.decide)),
        **kw,
    )


def _anthropic(fake: Fake, tmp_path: Path, **kw: Any) -> Any:
    real = anthropic.Anthropic(
        api_key="sk-test", base_url="http://provider.test", http_client=_anthropic_http(fake)
    )
    return agentcompile.wrap(
        real,
        key="ack_acme.k",
        base_url="http://ac.test",
        trail=tmp_path / "trail.jsonl",
        http_client=httpx.Client(transport=httpx.MockTransport(fake.decide)),
        **kw,
    )


def _anthropic_http(fake: Fake) -> Any:
    """anthropic 1.x ships its own httpx fork (httpx2); its mock transport hands us its own
    request type, so adapt it to the provider fake's httpx one."""
    try:
        import httpx2 as hx
    except ImportError:  # older anthropic: plain httpx
        hx = httpx

    def handler(request: Any) -> Any:
        r = fake.provider(
            httpx.Request(request.method, str(request.url), content=request.content)
        )
        return hx.Response(r.status_code, json=r.json())

    return hx.Client(transport=hx.MockTransport(handler))


def _trail(tmp_path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (tmp_path / "trail.jsonl").read_text().splitlines()]


MESSAGES = [{"role": "user", "content": "cancel order #W1"}]
TOOLS = [
    {
        "type": "function",
        "function": {"name": "get_order_details", "parameters": {"type": "object"}},
    }
]
ANTHROPIC_TOOLS = [{"name": "get_order_details", "input_schema": {"type": "object"}}]


def test_a_compiled_tool_call_is_a_real_chat_completion_and_the_model_is_not_called(
    tmp_path: Path,
) -> None:
    fake = Fake(CALL)
    client = _openai(fake, tmp_path)
    resp = client.chat.completions.create(
        model="gpt-x", messages=MESSAGES, tools=TOOLS, conversation_id="c1"
    )
    call = resp.choices[0].message.tool_calls[0]
    assert resp.choices[0].finish_reason == "tool_calls"
    assert call.function.name == "get_order_details" and json.loads(
        call.function.arguments
    ) == {"order_id": "#W1"}
    assert fake.provider_requests == []
    sent = fake.decide_requests[0]
    assert sent.headers["x-agentcompiler-key"] == "ack_acme.k"
    assert sent.headers["x-agentcompiler-conversation"] == "c1"
    assert "x-agentcompiler-company" not in sent.headers  # the key names the company
    assert json.loads(sent.content)["request"]["messages"] == MESSAGES
    assert _trail(tmp_path)[0]["route"] == "compiled"


def test_forward_calls_the_model_with_our_keyword_removed(tmp_path: Path) -> None:
    fake = Fake(FORWARD)
    resp = _openai(fake, tmp_path).chat.completions.create(
        model="gpt-x", messages=MESSAGES, tools=TOOLS, conversation_id="c1"
    )
    assert resp.choices[0].message.content == "from the model"
    assert "conversation_id" not in fake.provider_requests[0]
    event = _trail(tmp_path)[0]
    assert event["route"] == "forwarded" and event["reason"] == "no job matched"


def test_the_trail_keeps_what_agentcompile_decided_along_the_way(tmp_path: Path) -> None:
    events = [
        {"decision": "activate_recipe", "recipe": "cancel_order"},
        {"decision": "forward", "reason": "the customer asked a question"},
    ]
    fake = Fake({**FORWARD, "events": events})
    _openai(fake, tmp_path).chat.completions.create(
        model="gpt-x", messages=MESSAGES, tools=TOOLS, conversation_id="c1"
    )
    assert _trail(tmp_path)[0]["events"] == events


def test_malformed_events_are_dropped_not_fatal(tmp_path: Path) -> None:
    fake = Fake({**CALL, "events": ["not a dict", {"decision": "emit_read"}]})
    _openai(fake, tmp_path).chat.completions.create(
        model="gpt-x", messages=MESSAGES, tools=TOOLS, conversation_id="c1"
    )
    event = _trail(tmp_path)[0]
    assert event["route"] == "compiled" and event["events"] == [{"decision": "emit_read"}]


@pytest.mark.parametrize(
    "fake",
    [
        Fake(decide_status=500),
        Fake(decide_error=httpx.ConnectError("down")),
        Fake(decide_error=httpx.ReadTimeout("slow")),
        Fake({"action": "nonsense"}),
    ],
)
def test_any_decision_problem_fails_open_to_the_model(fake: Fake, tmp_path: Path) -> None:
    resp = _openai(fake, tmp_path).chat.completions.create(
        model="gpt-x", messages=MESSAGES, tools=TOOLS, conversation_id="c1"
    )
    assert resp.choices[0].message.content == "from the model"
    assert _trail(tmp_path)[0]["route"] == "fail-open"


def test_no_conversation_id_skips_the_decision(tmp_path: Path) -> None:
    fake = Fake(CALL)
    _openai(fake, tmp_path).chat.completions.create(model="gpt-x", messages=MESSAGES)
    assert fake.decide_requests == [] and len(fake.provider_requests) == 1
    assert _trail(tmp_path)[0]["route"] == "no-conversation"


def test_the_conversation_block_sets_the_id(tmp_path: Path) -> None:
    fake = Fake(CALL)
    with agentcompile.conversation("ticket-9"):
        _openai(fake, tmp_path).chat.completions.create(model="gpt-x", messages=MESSAGES)
    assert fake.decide_requests[0].headers["x-agentcompiler-conversation"] == "ticket-9"


def test_shadow_mode_always_calls_the_model_and_records_what_it_would_have_done(
    tmp_path: Path,
) -> None:
    fake = Fake(CALL)
    resp = _openai(fake, tmp_path, mode="shadow").chat.completions.create(
        model="gpt-x", messages=MESSAGES, tools=TOOLS, conversation_id="c1"
    )
    assert resp.choices[0].message.content == "from the model"
    event = _trail(tmp_path)[0]
    assert event["route"] == "shadow" and event["tool"] == "get_order_details"


def test_streaming_a_compiled_tool_call(tmp_path: Path) -> None:
    fake = Fake(CALL)
    stream = _openai(fake, tmp_path).chat.completions.create(
        model="gpt-x", messages=MESSAGES, tools=TOOLS, conversation_id="c1", stream=True
    )
    with stream:
        chunks = list(stream)
    calls = [c for ch in chunks for c in (ch.choices[0].delta.tool_calls or [])]
    assert calls[0].function.name == "get_order_details"
    assert chunks[-1].choices[0].finish_reason == "tool_calls"


def test_pydantic_messages_from_earlier_responses_are_sent_as_json(tmp_path: Path) -> None:
    fake = Fake(CALL)
    client = _openai(fake, tmp_path)
    first = client.chat.completions.create(
        model="gpt-x", messages=MESSAGES, tools=TOOLS, conversation_id="c1"
    )
    history = [
        *MESSAGES,
        first.choices[0].message,
        {"role": "tool", "tool_call_id": "call_1", "content": "{}"},
    ]
    client.chat.completions.create(
        model="gpt-x", messages=history, tools=TOOLS, conversation_id="c1"
    )
    sent = json.loads(fake.decide_requests[1].content)["request"]["messages"]
    assert sent[1]["tool_calls"][0]["function"]["name"] == "get_order_details"


def test_anthropic_compiled_reply_and_stream(tmp_path: Path) -> None:
    fake = Fake(SAY)
    client = _anthropic(fake, tmp_path)
    msg = client.messages.create(
        model="claude-x",
        max_tokens=100,
        system="be nice",
        messages=MESSAGES,
        conversation_id="c1",
    )
    assert msg.content[0].text == "Order #W1 is cancelled." and msg.stop_reason == "end_turn"
    assert json.loads(fake.decide_requests[0].content)["request"]["system"] == "be nice"
    fake.decision = CALL
    events = list(
        client.messages.create(
            model="claude-x",
            max_tokens=100,
            messages=MESSAGES,
            tools=ANTHROPIC_TOOLS,
            conversation_id="c1",
            stream=True,
        )
    )
    assert next(e.type for e in events) == "message_start" and events[-1].type == "message_stop"
    assert json.loads(events[2].delta.partial_json) == {"order_id": "#W1"}
    assert fake.provider_requests == []


def test_anthropic_forward(tmp_path: Path) -> None:
    fake = Fake(FORWARD)
    msg = _anthropic(fake, tmp_path).messages.create(
        model="claude-x", max_tokens=100, messages=MESSAGES, conversation_id="c1"
    )
    assert msg.content[0].text == "from the model"


async def test_async_openai_compiled_and_forward(tmp_path: Path) -> None:
    fake = Fake(CALL)
    real = openai.AsyncOpenAI(
        api_key="sk-test",
        base_url="http://provider.test/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(fake.provider)),
    )
    client = agentcompile.wrap(
        real,
        key="ack_acme.k",
        base_url="http://ac.test",
        trail=tmp_path / "trail.jsonl",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(fake.decide)),
    )
    resp = await client.chat.completions.create(
        model="gpt-x", messages=MESSAGES, tools=TOOLS, conversation_id="c1"
    )
    assert resp.choices[0].message.tool_calls[0].function.name == "get_order_details"
    fake.decision = FORWARD
    resp = await client.chat.completions.create(
        model="gpt-x", messages=MESSAGES, tools=TOOLS, conversation_id="c1"
    )
    assert resp.choices[0].message.content == "from the model"


def test_everything_else_on_the_client_still_works(tmp_path: Path) -> None:
    client = _openai(Fake(), tmp_path)
    assert client.api_key == "sk-test"
    assert hasattr(client.chat.completions, "with_raw_response")


def test_the_trail_command_prints_a_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from agentcompile.cli import main

    fake = Fake(CALL)
    client = _openai(fake, tmp_path)
    client.chat.completions.create(
        model="gpt-x", messages=MESSAGES, tools=TOOLS, conversation_id="c1"
    )
    fake.decision = FORWARD
    client.chat.completions.create(
        model="gpt-x", messages=MESSAGES, tools=TOOLS, conversation_id="c1"
    )
    assert main(["trail", "--path", str(tmp_path / "trail.jsonl")]) == 0
    out = capsys.readouterr().out
    assert (
        "compiled" in out
        and "2 calls: compiled 1, forwarded 1. Model calls avoided: 1 (50%)." in out
    )


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    from agentcompile.cli import main

    with pytest.raises(SystemExit) as done:
        main(["--version"])
    assert done.value.code == 0
    assert capsys.readouterr().out.startswith("agentcompile ")
