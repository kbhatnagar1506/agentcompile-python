"""The SDK side of the SDK <-> server contract: tests/contract/ is vendored from the server
repo (kbhatnagar1506/agent-compiler, tests/contract/; scripts/sync-contract.sh). Every golden
request is one this SDK sends byte for byte, and every golden response is one it parses.

With AGENTCOMPILE_CONTRACT_URL set (CI starts the server's contract server, real endpoint,
fake models: `python -m tests.contract.server`), the same requests go to it and a wrapped
openai client runs a whole job end to end: wrap -> decide -> tool call -> answer."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
import openai
import pytest

import agentcompile
from agentcompile import _decide
from agentcompile._decide import Decision, Settings, _decision_request, _read
from agentcompile._payload import payload

HERE = Path(__file__).resolve().parent / "contract"
SCHEMAS: dict[str, Any] = json.loads((HERE / "schemas.json").read_text())["schemas"]
GOLDENS = {p.stem: json.loads(p.read_text()) for p in sorted((HERE / "golden").glob("*.json"))}
LIVE = os.environ.get("AGENTCOMPILE_CONTRACT_URL")
KEY = "ack_acme.contract-test"  # the contract server's test key; not a secret

# Requests a golden holds that this SDK does not send as recorded, and why. Each is xfail
# (strict): when the SDK starts sending it, the test fails until this entry goes.
KNOWN_GAPS = {
    "decide_end_user": "the SDK never sends the end user (OpenAI `user`, Anthropic "
    "metadata.user_id, or x-agentcompiler-end-user) to /v1/decide: recipes keyed on the "
    "session's user hand off for SDK callers",
}


def problems(value: Any, schema: dict[str, Any], where: str = "$") -> list[str]:
    """Why `value` doesn't match `schema` (schemas.json's subset of JSON Schema)."""
    if "$ref" in schema:
        return problems(value, SCHEMAS[schema["$ref"]], where)
    kinds: dict[str, Any] = {
        "object": dict,
        "array": list,
        "string": str,
        "boolean": bool,
        "integer": int,
        "number": (int, float),
    }
    kind = schema.get("type")
    if kind is not None and (
        not isinstance(value, kinds[kind])
        or (kind in ("integer", "number") and isinstance(value, bool))
    ):
        return [f"{where}: expected {kind}, got {type(value).__name__}"]
    found = []
    if "const" in schema and value != schema["const"]:
        found.append(f"{where}: expected {schema['const']!r}, got {value!r}")
    if "enum" in schema and value not in schema["enum"]:
        found.append(f"{where}: {value!r} not one of {schema['enum']!r}")
    if "minLength" in schema and isinstance(value, str) and len(value) < schema["minLength"]:
        found.append(f"{where}: shorter than {schema['minLength']}")
    if "minimum" in schema and isinstance(value, (int, float)) and value < schema["minimum"]:
        found.append(f"{where}: below {schema['minimum']}")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        found += [
            f"{where}: missing {n!r}" for n in schema.get("required", []) if n not in value
        ]
        for name, item in value.items():
            if name in properties:
                found += problems(item, properties[name], f"{where}.{name}")
            elif schema.get("additionalProperties") is False:
                found.append(f"{where}: unexpected field {name!r}")
    if isinstance(value, list) and "items" in schema:
        for i, item in enumerate(value):
            found += problems(item, schema["items"], f"{where}[{i}]")
    return found


def _sdk_shaped(golden: dict[str, Any]) -> bool:
    """A decision request of the kind this SDK sends: a provider it speaks, its key, a
    conversation id and its deadline (the others are what a proxy or a bug might send)."""
    request = golden["request"]
    headers = request["headers"]
    return (
        request["path"] == "/v1/decide"
        and "body" in request
        and request["body"].get("provider") in ("openai", "anthropic")
        and {"x-agentcompiler-key", "x-agentcompiler-conversation"} <= set(headers)
        and "x-agentcompiler-deadline-ms" in headers
    )


def _gap(name: str) -> Any:
    marks = (
        [pytest.mark.xfail(strict=True, reason=KNOWN_GAPS[name])] if name in KNOWN_GAPS else []
    )
    return pytest.param(name, marks=marks)


SENT = [_gap(n) for n, g in GOLDENS.items() if _sdk_shaped(g)]
DECISIONS = [n for n, g in GOLDENS.items() if g["request"]["path"] == "/v1/decide"]


def test_the_vendored_contract_covers_every_route_this_sdk_calls() -> None:
    assert len(GOLDENS) >= 10
    paths = {g["request"]["path"] for g in GOLDENS.values()}
    assert {"/v1/decide", "/v1/capture", "/v1/outcome"} <= paths
    schemas = {g["response"]["schema"] for g in GOLDENS.values()}
    assert {"decision.tool_call", "decision.say", "decision.forward"} <= schemas


@pytest.mark.parametrize("name", SENT)
def test_the_sdk_sends_each_golden_decision_request(name: str) -> None:
    request = GOLDENS[name]["request"]
    headers, body = request["headers"], request["body"]
    settings = Settings(
        "http://ac.test",
        headers["x-agentcompiler-key"],
        None,
        int(headers["x-agentcompiler-deadline-ms"]) / 1000,
        headers.get("x-agentcompiler-mode", "live"),
    )
    url, sent_headers, sent_body = _decision_request(
        settings,
        body["provider"],
        headers["x-agentcompiler-conversation"],
        payload(body["request"]),
    )
    assert url == "http://ac.test" + request["path"]
    assert sent_headers == headers
    assert sent_body == body


@pytest.mark.parametrize("name", DECISIONS)
def test_the_sdk_parses_each_golden_decision_response(name: str) -> None:
    expected = GOLDENS[name]["response"]
    example = expected["example"]
    assert problems(example, SCHEMAS[expected["schema"]]) == []
    decision, _, error = _read(httpx.Response(expected["status"], json=example), 1.0)
    if expected["status"] != 200:
        assert decision is None and error == f"HTTP {expected['status']}"  # fails open
        return
    assert error is None and decision is not None
    assert decision.action == example["action"]
    assert decision.events == tuple(example.get("events", []))
    if decision.action == "tool_call":
        assert (decision.tool, decision.args, decision.call_id) == (
            example["tool"],
            example["args"],
            example["call_id"],
        )
    elif decision.action == "say":
        assert decision.text == example["text"]
    else:
        assert decision.reason == example["reason"]


def _provider(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-model",
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


def _model() -> Any:
    return openai.OpenAI(
        api_key="sk-test",
        base_url="http://provider.test/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(_provider)),
    )


@pytest.mark.parametrize(
    "name",
    [n for n in DECISIONS if GOLDENS[n]["request"].get("body", {}).get("provider") == "openai"],
)
def test_a_wrapped_client_answers_each_golden_decision(name: str, tmp_path: Path) -> None:
    """The decision as the agent's own loop sees it: a tool call or text from the wrapped
    client, or the model's own answer when the service forwards or refuses."""
    expected = GOLDENS[name]["response"]
    example = expected["example"]
    service = httpx.Client(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(expected["status"], json=example)
        )
    )
    client: Any = agentcompile.wrap(
        _model(), key=KEY, base_url="http://ac.test", trail=False, http_client=service
    )
    sent = GOLDENS[name]["request"]["body"]["request"]
    message = (
        client.chat.completions.create(**{**sent, "conversation_id": "c1"}).choices[0].message
    )
    if expected["status"] == 200 and example["action"] == "tool_call":
        call = message.tool_calls[0]
        assert call.function.name == example["tool"]
        assert json.loads(call.function.arguments) == example["args"]
    elif expected["status"] == 200 and example["action"] == "say":
        assert message.content == example["text"]
    else:
        assert message.content == "from the model"


def test_outcome_and_capture_requests_have_the_golden_shape(tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def service(request: httpx.Request) -> httpx.Response:
        seen[request.url.path] = (dict(request.headers), json.loads(request.content))
        return httpx.Response(200, json={"kept": 1, "dropped": 0})

    http = httpx.Client(transport=httpx.MockTransport(service))
    client: Any = agentcompile.wrap(
        _model(),
        key=KEY,
        base_url="http://ac.test",
        trail=False,
        capture=True,
        scrub=False,
        http_client=http,
    )
    client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}]
    )
    agentcompile.flush()
    headers, body = seen["/v1/capture"]
    golden = GOLDENS["capture_batch"]["request"]
    assert {k: headers.get(k) for k in golden["headers"]} == golden["headers"]
    record = body["exchanges"][0]
    assert set(golden["body"]["exchanges"][0]) <= set(record)
    assert problems(body, {"type": "object", "required": ["exchanges"]}) == []


def test_outcome_request_has_the_golden_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    from agentcompile import _outcome

    seen: list[tuple[str, dict[str, str], Any]] = []

    class Recording(httpx.Client):
        def post(self, url: Any, **kw: Any) -> httpx.Response:  # type: ignore[override]
            seen.append((str(url), dict(kw.get("headers") or {}), kw.get("json")))
            return httpx.Response(200, json={"kept": 1, "dropped": 0})

    monkeypatch.setattr(_outcome.httpx, "Client", Recording)
    _outcome.remember(Settings("http://ac.test", KEY, None, 5.0))
    agentcompile.outcome("golden-capture-1", "resolved", note="refunded")
    _outcome.flush()
    url, headers, body = seen[0]
    golden = GOLDENS["outcome"]["request"]
    assert url == "http://ac.test" + golden["path"]
    assert headers == golden["headers"]
    sent, recorded = body["outcomes"][0], golden["body"]["outcomes"][0]
    assert set(sent) == set(recorded)
    assert {k: sent[k] for k in ("conversation_id", "outcome", "note")} == {
        k: recorded[k] for k in ("conversation_id", "outcome", "note")
    }


# --- against the real server (CI: the server's contract server) ----------------------------

live = pytest.mark.skipif(not LIVE, reason="AGENTCOMPILE_CONTRACT_URL not set")


@live
@pytest.mark.parametrize("name", sorted(GOLDENS))
def test_the_server_answers_each_golden_request(name: str) -> None:
    golden = GOLDENS[name]
    request, expected = golden["request"], golden["response"]
    headers = dict(request["headers"])
    if "x-agentcompiler-conversation" in headers:  # its own conversation on a shared server
        headers["x-agentcompiler-conversation"] += f"-{name}-py-{os.getpid()}"
    with httpx.Client(base_url=LIVE, timeout=10) as http:
        if request["method"] == "GET":
            response = http.get(request["path"], headers=headers)
        elif "raw_body" in request:
            headers["content-type"] = "application/json"
            response = http.post(request["path"], content=request["raw_body"], headers=headers)
        else:
            response = http.post(request["path"], json=request["body"], headers=headers)
    assert response.status_code == expected["status"], response.text
    body = response.json()
    assert problems(body, SCHEMAS[expected["schema"]]) == []
    assert {k: body.get(k) for k in expected["expect"]} == expected["expect"]
    if request["path"] == "/v1/decide" and response.status_code == 200:
        assert Decision.parse(body) is not None


ORDERS = {
    "#W1": {"order_id": "#W1", "status": "pending", "items": [{"name": "Desk Lamp"}]},
    "#W2": {"order_id": "#W2", "status": "delivered", "items": [{"name": "Kettle"}]},
}


def _run_tool(name: str, args: dict[str, Any]) -> str:
    """The agent's own tools (the contract server's backend: one user, two orders)."""
    user = "jo_park_1234"
    if name.startswith("find_user_id"):
        return user
    if name == "get_user_details":
        return json.dumps({"user_id": user, "orders": list(ORDERS), "payment_methods": {}})
    if name == "get_order_details":
        order = ORDERS[args["order_id"]]
        return json.dumps({**order, "user_id": user, "payment_history": []})
    if name == "cancel_pending_order":
        return json.dumps({**ORDERS[args["order_id"]], "status": "cancelled"})
    return "Error: unknown tool"


@live
def test_end_to_end_wrap_decide_tool_call_answer(tmp_path: Path) -> None:
    """A whole job through the real endpoint: the wrapped client asks who the customer is,
    runs the lookups as tool calls the agent's loop executes, proposes, and cancels on yes;
    the model is never called."""
    calls: list[Any] = []

    def provider(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return _provider(request)

    model = openai.OpenAI(
        api_key="sk-test",
        base_url="http://provider.test/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(provider)),
    )
    trail = tmp_path / "trail.jsonl"
    client: Any = agentcompile.wrap(model, key=KEY, base_url=LIVE, trail=trail)
    tools = [
        {"type": "function", "function": {"name": n, "parameters": {"type": "object"}}}
        for n in (
            "find_user_id_by_email",
            "find_user_id_by_name_zip",
            "get_user_details",
            "get_order_details",
            "cancel_pending_order",
        )
    ]
    messages: list[Any] = [
        {"role": "system", "content": "You are acme's retail support agent."},
        {"role": "user", "content": "Please cancel my order with the desk lamp."},
    ]
    conversation = f"e2e-py-{os.getpid()}"
    said, ran = [], []
    for reply in ("sure, it's jo@example.com", "I don't need it anymore", "yes", None):
        for _ in range(12):
            message = (
                client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    tools=tools,
                    conversation_id=conversation,
                )
                .choices[0]
                .message
            )
            messages.append(message.model_dump(exclude_none=True))
            if not message.tool_calls:
                break
            for call in message.tool_calls:
                ran.append(call.function.name)
                output = _run_tool(call.function.name, json.loads(call.function.arguments))
                messages.append({"role": "tool", "tool_call_id": call.id, "content": output})
        said.append(message.content)
        if reply is not None:
            messages.append({"role": "user", "content": reply})
    routes = [json.loads(line)["route"] for line in trail.read_text().splitlines()]
    assert set(routes) == {"compiled"}, routes
    assert calls == []  # the model was never called
    assert said[0] == "To find your account, could you tell me your email?"
    assert ran[0] == "find_user_id_by_email" and ran[-1] == "cancel_pending_order"
    assert said[-1].startswith("Done: cancel pending order completed")


@live
def test_capture_and_outcome_reach_the_server() -> None:
    from agentcompile import _capture, _outcome

    seen: list[int] = []

    class Counting(httpx.Client):
        def post(self, url: Any, **kw: Any) -> httpx.Response:  # type: ignore[override]
            response = super().post(url, **kw)
            seen.append(response.status_code)
            return response

    client: Any = agentcompile.wrap(
        _model(),
        key=KEY,
        base_url=LIVE,
        trail=False,
        capture=True,
        scrub=False,
        http_client=Counting(),
    )
    client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}]
    )
    agentcompile.flush()
    capturers = list(_capture._ALL.values())
    assert sum(c.sent for c in capturers) >= 1 and sum(c.failed for c in capturers) == 0
    before = _outcome.sent
    agentcompile.outcome("golden-capture-1", "resolved")
    _outcome.flush()
    assert _outcome.sent == before + 1


def test_decide_pool_constant_is_what_the_load_test_relies_on() -> None:
    """The server's load test drives 32 conversations through one process."""
    assert _decide.MAX_DECIDE_THREADS >= 32
