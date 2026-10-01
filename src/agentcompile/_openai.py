"""OpenAI-style clients (OpenAI, AsyncOpenAI, AzureOpenAI and any OpenAI-compatible base_url):
compiled answers shaped exactly like chat.completions.create() results."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from ._decide import Decision
from ._streams import AsyncCompiledStream, CompiledStream


def _message(decision: Decision) -> dict[str, Any]:
    if decision.action == "tool_call":
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": decision.call_id or f"call_ac_{uuid.uuid4().hex[:12]}",
                    "type": "function",
                    "function": {"name": decision.tool, "arguments": json.dumps(decision.args)},
                }
            ],
        }
    return {"role": "assistant", "content": decision.text}


def build(decision: Decision, kwargs: dict[str, Any], stream: bool, is_async: bool) -> Any:
    from openai.types.chat import ChatCompletion, ChatCompletionChunk

    model = str(kwargs.get("model") or "agentcompile")
    ident, created = f"chatcmpl-ac-{uuid.uuid4().hex[:16]}", int(time.time())
    message = _message(decision)
    finish = "tool_calls" if decision.action == "tool_call" else "stop"
    if not stream:
        return ChatCompletion.model_validate(
            {
                "id": ident,
                "object": "chat.completion",
                "created": created,
                "model": model,
                "choices": [
                    {"index": 0, "message": message, "finish_reason": finish, "logprobs": None}
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
        )
    delta: dict[str, Any] = {"role": "assistant", "content": message.get("content")}
    if "tool_calls" in message:
        delta["tool_calls"] = [
            {**call, "index": i} for i, call in enumerate(message["tool_calls"])
        ]
    base = {"id": ident, "object": "chat.completion.chunk", "created": created, "model": model}
    chunks: list[Any] = [
        ChatCompletionChunk.model_validate(
            {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
        ),
        ChatCompletionChunk.model_validate(
            {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]}
        ),
    ]
    return AsyncCompiledStream(chunks) if is_async else CompiledStream(chunks)
