"""Capture from any framework: an HTTP transport that sees the model calls a framework makes
(LangChain, LiteLLM, CrewAI, Pydantic AI, the OpenAI Agents SDK, your own loop) and captures
them, with no client to wrap. Capture only: every call goes to the model unchanged and is
never answered compiled (that takes `wrap`).

    http = agentcompile.http_client(key="ack_...")   # async: agentcompile.async_http_client()
    llm = ChatOpenAI(model="gpt-5", http_client=http)
    with agentcompile.conversation(ticket.id):
        agent.invoke(...)

Seen: POSTs to `/chat/completions`, `/responses` (OpenAI and anything OpenAI-compatible) and
`/messages` (Anthropic). Everything else passes through untouched. Failed calls (not 2xx) are
not captured. Nothing here ever raises into the framework.

Built for httpx and for httpx2 (the newer provider SDKs' fork, which refuses httpx objects):
the same classes, made once per library.
"""

from __future__ import annotations

import importlib
import json
import zlib
from collections.abc import AsyncIterator, Callable, Iterator
from functools import cache
from typing import Any

import httpx

from ._conversation import current, current_customer


def provider_of(request: Any) -> str | None:
    """Which provider's shape a request is, or None for a request we don't capture."""
    if request.method != "POST":
        return None
    path = request.url.path.rstrip("/")
    if path.endswith(("/chat/completions", "/responses")):
        return "openai"
    # Anthropic's create; not OpenAI's Assistants threads, which also end in /messages.
    if path.endswith("/messages") and "/threads/" not in path:
        return "anthropic"
    return None


def body_of(request: Any) -> dict[str, Any] | None:
    try:
        body = json.loads(request.content)
    except Exception:  # a body not read yet, or not JSON: not a model call we read
        return None
    if isinstance(body, dict) and ("messages" in body or "input" in body):
        return body
    return None


def decode(raw: bytes, encoding: str) -> bytes | None:
    """The body as sent, before any content encoding; None for one we can't undo. A body cut
    short (a stream closed early) decodes as far as it goes."""
    for coding in reversed([c.strip().lower() for c in encoding.split(",") if c.strip()]):
        if coding == "identity":
            continue
        if coding in ("gzip", "x-gzip", "deflate"):
            try:
                raw = zlib.decompressobj(47).decompress(raw)  # gzip or zlib, by header
            except zlib.error:
                raw = zlib.decompressobj(-15).decompress(raw)  # raw deflate
        elif coding == "br":
            try:
                import brotli  # type: ignore[import-not-found]
            except ImportError:
                return None
            raw = brotli.decompress(raw)
        else:
            return None
    return raw


def events(text: str) -> list[dict[str, Any]]:
    """The JSON objects in a server-sent event stream, in order ([DONE] and comments
    skipped)."""
    found: list[dict[str, Any]] = []
    for block in text.replace("\r\n", "\n").split("\n\n"):
        data = "\n".join(
            line[5:].lstrip() for line in block.split("\n") if line.startswith("data:")
        )
        if not data or data == "[DONE]":
            continue
        try:
            value = json.loads(data)
        except ValueError:
            continue
        if isinstance(value, dict):
            found.append(value)
    return found


_ENDS = ("response.completed", "response.incomplete", "response.failed", "message_stop")


def ended(text: str, chunks: list[dict[str, Any]]) -> bool:
    """Whether a stream reached its own end marker. SDKs stop reading at the marker, before
    the last bytes, so a stream can be whole without its transport stream running out."""
    if "data: [DONE]" in text or any(c.get("type") in _ENDS for c in chunks):
        return True
    last = chunks[-1] if chunks else {}
    return any(choice.get("finish_reason") for choice in last.get("choices") or [])


class _Seen:
    """One call in flight: what to capture once its answer is whole."""

    def __init__(self, provider: str, body: dict[str, Any], response: Any) -> None:
        self.provider = provider
        self.body = body
        self.encoding = response.headers.get("content-encoding", "")
        self.raw: list[bytes] = []
        self.finished = False


class _Capture:
    """What every transport shares: which calls to see, and handing them to the capturer."""

    def __init__(self, capturer: Any) -> None:
        self.capturer = capturer

    def seen(self, request: Any, response: Any) -> _Seen | None:
        try:
            provider = provider_of(request)
            if provider is None or not 200 <= response.status_code < 300:
                return None
            body = body_of(request)
            return None if body is None else _Seen(provider, body, response)
        except Exception:
            return None

    def whole(self, seen: _Seen, response: Any) -> None:
        """A call answered in one body (already read)."""
        try:
            self.capturer.add(
                seen.provider,
                current(),
                seen.body,
                response.json(),
                customer=current_customer(),
            )
        except Exception:
            self.capturer.dropped += 1

    def streamed(self) -> Callable[[_Seen, bool], None]:
        """The callback for a stream: its conversation is the one the call was made in, not
        whatever is current when the framework finishes reading."""
        conversation, customer = current(), current_customer()

        def done(seen: _Seen, ran_out: bool) -> None:
            if seen.finished:
                return
            seen.finished = True
            try:
                raw = decode(b"".join(seen.raw), seen.encoding)
                if raw is None:
                    self.capturer.dropped += 1
                    return
                text = raw.decode("utf-8", "replace")
                chunks = events(text)
                self.capturer.add_stream(
                    seen.provider,
                    conversation,
                    seen.body,
                    chunks,
                    ran_out or ended(text, chunks),
                    customer,
                )
            except Exception:
                self.capturer.dropped += 1

        return done


@cache
def classes(lib_name: str) -> tuple[type, type]:
    """(CaptureTransport, AsyncCaptureTransport) for one HTTP library: "httpx" or "httpx2"."""
    lib = httpx if lib_name == "httpx" else importlib.import_module(lib_name)

    def already_read(seen: _Seen, response: Any, done: Callable[[_Seen, bool], None]) -> bool:
        """A stream the transport below already read whole (a mock, a cache, a replay):
        captured from its body now, since nothing will flow through a tee."""
        try:
            content = response.content
        except lib.ResponseNotRead:
            return False
        seen.raw, seen.encoding = [content], ""  # the library has already decoded it
        done(seen, True)
        return True

    class Tee(lib.SyncByteStream):  # type: ignore[misc, name-defined]
        def __init__(self, stream: Any, seen: _Seen, done: Callable[[_Seen, bool], None]):
            self._stream, self._seen, self._done = stream, seen, done

        def __iter__(self) -> Iterator[bytes]:
            ran_out = False
            try:
                for chunk in self._stream:
                    self._seen.raw.append(chunk)
                    yield chunk
                ran_out = True
            finally:
                self._done(self._seen, ran_out)

        def close(self) -> None:
            self._done(self._seen, False)
            close = getattr(self._stream, "close", None)
            if close is not None:
                close()

    class AsyncTee(lib.AsyncByteStream):  # type: ignore[misc, name-defined]
        def __init__(self, stream: Any, seen: _Seen, done: Callable[[_Seen, bool], None]):
            self._stream, self._seen, self._done = stream, seen, done

        async def __aiter__(self) -> AsyncIterator[bytes]:
            ran_out = False
            try:
                async for chunk in self._stream:
                    self._seen.raw.append(chunk)
                    yield chunk
                ran_out = True
            finally:
                self._done(self._seen, ran_out)

        async def aclose(self) -> None:
            self._done(self._seen, False)
            aclose = getattr(self._stream, "aclose", None)
            if aclose is not None:
                await aclose()

    class CaptureTransport(lib.BaseTransport):  # type: ignore[misc, name-defined]
        """A transport that captures the model calls passing through it."""

        def __init__(self, capturer: Any, wrapped: Any = None) -> None:
            self._capture = _Capture(capturer)
            self._wrapped = wrapped or lib.HTTPTransport()

        def handle_request(self, request: Any) -> Any:
            response = self._wrapped.handle_request(request)
            seen = self._capture.seen(request, response)
            if seen is None:
                return response
            if seen.body.get("stream"):
                done = self._capture.streamed()
                if not already_read(seen, response, done):
                    response.stream = Tee(response.stream, seen, done)
                return response
            response.read()
            self._capture.whole(seen, response)
            return response

        def close(self) -> None:
            self._wrapped.close()

    class AsyncCaptureTransport(lib.AsyncBaseTransport):  # type: ignore[misc, name-defined]
        """The async version."""

        def __init__(self, capturer: Any, wrapped: Any = None) -> None:
            self._capture = _Capture(capturer)
            self._wrapped = wrapped or lib.AsyncHTTPTransport()

        async def handle_async_request(self, request: Any) -> Any:
            response = await self._wrapped.handle_async_request(request)
            seen = self._capture.seen(request, response)
            if seen is None:
                return response
            if seen.body.get("stream"):
                done = self._capture.streamed()
                if not already_read(seen, response, done):
                    response.stream = AsyncTee(response.stream, seen, done)
                return response
            await response.aread()
            self._capture.whole(seen, response)
            return response

        async def aclose(self) -> None:
            await self._wrapped.aclose()

    return CaptureTransport, AsyncCaptureTransport


def lib_of(wrapped: Any, lib: str | None) -> str:
    """The library a transport is for: as named, else the wrapped transport's own."""
    if lib:
        return lib
    module = type(wrapped).__module__.partition(".")[0] if wrapped is not None else ""
    return module if module in ("httpx", "httpx2") else "httpx"


CaptureTransport, AsyncCaptureTransport = classes("httpx")
