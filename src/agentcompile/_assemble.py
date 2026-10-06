"""Streamed answers, captured whole: each chunk passes to your agent untouched as it arrives, a
copy is kept, and when the stream ends the pieces are assembled into the same shape a
non-streamed answer has (a chat completion, a Responses API response, or an Anthropic message),
so AgentCompile reads streamed and non-streamed calls the same way."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from typing import Any, Callable

from ._payload import jsonable

OnDone = Callable[[list[Any], bool], None]  # (chunks as JSON, the stream ran to its end)


def assemble(provider: str, chunks: list[Any]) -> dict[str, Any] | None:
    try:
        if provider != "openai":
            return _anthropic(chunks)
        if any(str(c.get("type", "")).startswith("response.") for c in chunks):
            return _responses(chunks)
        return _openai(chunks)
    except Exception:  # never break the agent over a shape we didn't expect
        return None


def _openai(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    first = chunks[0] if chunks else {}
    content: list[str] = []
    calls: dict[int, dict[str, Any]] = {}
    finish = None
    usage = None
    for chunk in chunks:
        usage = chunk.get("usage") or usage
        for choice in chunk.get("choices") or []:
            if choice.get("index", 0) != 0:
                continue
            delta = choice.get("delta") or {}
            if delta.get("content"):
                content.append(delta["content"])
            for call in delta.get("tool_calls") or []:
                slot = calls.setdefault(
                    call.get("index", 0),
                    {"id": None, "type": "function", "function": {"name": "", "arguments": ""}},
                )
                slot["id"] = call.get("id") or slot["id"]
                fn = call.get("function") or {}
                slot["function"]["name"] += fn.get("name") or ""
                slot["function"]["arguments"] += fn.get("arguments") or ""
            finish = choice.get("finish_reason") or finish
    message: dict[str, Any] = {"role": "assistant", "content": "".join(content) or None}
    if calls:
        message["tool_calls"] = [calls[i] for i in sorted(calls)]
    out: dict[str, Any] = {
        "id": first.get("id"),
        "object": "chat.completion",
        "created": first.get("created"),
        "model": first.get("model"),
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
    }
    if usage:
        out["usage"] = usage
    return out


#: The events that end a Responses API stream, each carrying the whole response.
_RESPONSE_DONE = ("response.completed", "response.incomplete", "response.failed")


def _responses(events: list[dict[str, Any]]) -> dict[str, Any]:
    """A Responses API stream: its last event carries the whole response. A stream stopped
    early has none, so the response is rebuilt from the items it finished."""
    for event in reversed(events):
        if event.get("type") in _RESPONSE_DONE and isinstance(event.get("response"), dict):
            return dict(event["response"])
    response: dict[str, Any] = {}
    items: dict[int, dict[str, Any]] = {}
    for event in events:
        if event.get("type") == "response.created" and isinstance(event.get("response"), dict):
            response = dict(event["response"])
        elif event.get("type") == "response.output_item.done":
            items[int(event.get("output_index", len(items)))] = dict(event.get("item") or {})
    response["output"] = [items[i] for i in sorted(items)]
    return response


def _anthropic(events: list[dict[str, Any]]) -> dict[str, Any]:
    message: dict[str, Any] = {}
    blocks: dict[int, dict[str, Any]] = {}
    partial: dict[int, list[str]] = {}
    for event in events:
        kind = event.get("type")
        if kind == "message_start":
            message = dict(event.get("message") or {})
        elif kind == "content_block_start":
            blocks[event["index"]] = dict(event.get("content_block") or {})
        elif kind == "content_block_delta":
            block = blocks.setdefault(event["index"], {"type": "text", "text": ""})
            delta = event.get("delta") or {}
            if delta.get("type") == "text_delta":
                block["text"] = block.get("text", "") + delta.get("text", "")
            elif delta.get("type") == "input_json_delta":
                partial.setdefault(event["index"], []).append(delta.get("partial_json", ""))
            elif delta.get("type") == "thinking_delta":
                block["thinking"] = block.get("thinking", "") + delta.get("thinking", "")
        elif kind == "message_delta":
            delta = event.get("delta") or {}
            message["stop_reason"] = delta.get("stop_reason", message.get("stop_reason"))
            message["stop_sequence"] = delta.get("stop_sequence")
            usage = dict(message.get("usage") or {})
            usage.update(event.get("usage") or {})
            message["usage"] = usage
    for index, pieces in partial.items():
        text = "".join(pieces)
        blocks[index]["input"] = json.loads(text) if text else {}
    message["content"] = [blocks[i] for i in sorted(blocks)]
    return message


class CapturingStream:
    """Your stream, unchanged, with a copy of each chunk kept for capture."""

    def __init__(self, stream: Any, on_done: OnDone) -> None:
        self._stream = stream
        self._on_done = on_done
        self._chunks: list[Any] = []
        self._finished = False
        self._it = self._iterate()  # one pass, shared by `for` and next()

    def _iterate(self) -> Iterator[Any]:
        complete = False
        try:
            for item in self._stream:
                self._chunks.append(jsonable(item))
                yield item
            complete = True
        finally:
            self._finish(complete=complete)

    def __iter__(self) -> Iterator[Any]:
        return self._it

    def __next__(self) -> Any:
        return next(self._it)

    def __enter__(self) -> CapturingStream:
        enter = getattr(self._stream, "__enter__", None)
        if enter is not None:
            enter()
        return self

    def __exit__(self, *exc: Any) -> Any:
        self._finish(complete=False)
        leave = getattr(self._stream, "__exit__", None)
        return leave(*exc) if leave is not None else None

    def close(self) -> None:
        self._finish(complete=False)
        close = getattr(self._stream, "close", None)
        if close is not None:
            close()

    def _finish(self, *, complete: bool) -> None:
        if not self._finished:
            self._finished = True
            self._on_done(self._chunks, complete)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


class AsyncCapturingStream:
    """The async version."""

    def __init__(self, stream: Any, on_done: OnDone) -> None:
        self._stream = stream
        self._on_done = on_done
        self._chunks: list[Any] = []
        self._finished = False
        self._it = self._iterate()

    async def _iterate(self) -> AsyncIterator[Any]:
        complete = False
        try:
            async for item in self._stream:
                self._chunks.append(jsonable(item))
                yield item
            complete = True
        finally:
            self._finish(complete=complete)

    def __aiter__(self) -> AsyncIterator[Any]:
        return self._it

    async def __anext__(self) -> Any:
        return await self._it.__anext__()

    async def __aenter__(self) -> AsyncCapturingStream:
        enter = getattr(self._stream, "__aenter__", None)
        if enter is not None:
            await enter()
        return self

    async def __aexit__(self, *exc: Any) -> Any:
        self._finish(complete=False)
        leave = getattr(self._stream, "__aexit__", None)
        return (await leave(*exc)) if leave is not None else None

    async def close(self) -> None:
        self._finish(complete=False)
        close = getattr(self._stream, "close", None)
        if close is not None:
            result = close()
            if hasattr(result, "__await__"):
                await result

    def _finish(self, *, complete: bool) -> None:
        if not self._finished:
            self._finished = True
            self._on_done(self._chunks, complete)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)
