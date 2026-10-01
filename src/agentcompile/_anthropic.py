"""Anthropic-style clients (Anthropic, AsyncAnthropic, AnthropicBedrock, AnthropicVertex):
compiled answers shaped exactly like messages.create() results."""

from __future__ import annotations

import json
import uuid
from typing import Any

from ._decide import Decision
from ._streams import AsyncCompiledStream, CompiledStream


def _block(decision: Decision) -> dict[str, Any]:
    if decision.action == "tool_call":
        return {
            "type": "tool_use",
            "id": decision.call_id or f"toolu_ac_{uuid.uuid4().hex[:12]}",
            "name": decision.tool,
            "input": decision.args,
        }
    return {"type": "text", "text": decision.text or ""}


def build(decision: Decision, kwargs: dict[str, Any], stream: bool, is_async: bool) -> Any:
    from anthropic.types import (
        Message,
        RawContentBlockDeltaEvent,
        RawContentBlockStartEvent,
        RawContentBlockStopEvent,
        RawMessageDeltaEvent,
        RawMessageStartEvent,
        RawMessageStopEvent,
    )

    model = str(kwargs.get("model") or "agentcompile")
    block = _block(decision)
    stop = "tool_use" if decision.action == "tool_call" else "end_turn"
    message = {
        "id": f"msg_ac_{uuid.uuid4().hex[:16]}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [block],
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }
    if not stream:
        return Message.model_validate(message)
    if block["type"] == "tool_use":
        start_block = {**block, "input": {}}
        delta = {"type": "input_json_delta", "partial_json": json.dumps(block["input"])}
    else:
        start_block = {"type": "text", "text": ""}
        delta = {"type": "text_delta", "text": block["text"]}
    events: list[Any] = [
        RawMessageStartEvent.model_validate(
            {
                "type": "message_start",
                "message": {**message, "content": [], "stop_reason": None},
            }
        ),
        RawContentBlockStartEvent.model_validate(
            {"type": "content_block_start", "index": 0, "content_block": start_block}
        ),
        RawContentBlockDeltaEvent.model_validate(
            {"type": "content_block_delta", "index": 0, "delta": delta}
        ),
        RawContentBlockStopEvent.model_validate({"type": "content_block_stop", "index": 0}),
        RawMessageDeltaEvent.model_validate(
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop, "stop_sequence": None},
                "usage": {"output_tokens": 0},
            }
        ),
        RawMessageStopEvent.model_validate({"type": "message_stop"}),
    ]
    return AsyncCompiledStream(events) if is_async else CompiledStream(events)
