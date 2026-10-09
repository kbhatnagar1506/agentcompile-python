"""The decision call to AgentCompile. Never raises: any problem is a `None` decision, which the
wrapper treats as forward (fail open).

Bounded: the whole call (name lookup, connect, send, read) takes at most `timeout` from when
it is sent, and the server is told how long that is so it answers in time. Sync calls run on
a pool that grows with the calls in flight, so one waits for a thread only past
MAX_DECIDE_THREADS at once (and gives up if none frees within `timeout`). After a few
failures in a row the breaker opens: calls go straight to the model, without asking, until it
tries again.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx

COMPANY_HEADER = "x-agentcompiler-company"
CONVERSATION_HEADER = "x-agentcompiler-conversation"
KEY_HEADER = "x-agentcompiler-key"
MODE_HEADER = "x-agentcompiler-mode"
DEADLINE_HEADER = "x-agentcompiler-deadline-ms"

FAILURES_TO_OPEN = 5
OPEN_FOR_S = 30.0


@dataclass(frozen=True)
class Decision:
    action: str  # "tool_call" | "say" | "forward"
    tool: str | None = None
    args: dict[str, Any] = field(default_factory=dict)
    call_id: str | None = None
    text: str | None = None
    reason: str | None = None
    # What AgentCompile decided along the way (which job, what it asked, why it handed
    # off), as the server reported it: kept so the trail can show it.
    events: tuple[dict[str, Any], ...] = ()

    @classmethod
    def parse(cls, body: Any) -> Decision | None:
        if not isinstance(body, dict):
            return None
        action = body.get("action")
        raw = body.get("events")
        events = tuple(e for e in (raw if isinstance(raw, list) else []) if isinstance(e, dict))
        if (
            action == "tool_call"
            and isinstance(body.get("tool"), str)
            and isinstance(body.get("args"), dict)
        ):
            return cls(
                action,
                tool=body["tool"],
                args=body["args"],
                call_id=str(body.get("call_id") or ""),
                events=events,
            )
        if action == "say" and isinstance(body.get("text"), str):
            return cls(action, text=body["text"], events=events)
        if action == "forward":
            return cls(action, reason=str(body.get("reason") or ""), events=events)
        return None


@dataclass(frozen=True)
class Settings:
    base_url: str
    key: str | None
    company: str | None
    timeout: float
    mode: str = "live"


def _request(
    settings: Settings, provider: str, conversation_id: str, payload: dict[str, Any]
) -> tuple[str, dict[str, str], dict[str, Any]]:
    headers = {CONVERSATION_HEADER: conversation_id}
    if settings.company:  # optional: the key names the company
        headers[COMPANY_HEADER] = settings.company
    if settings.key:
        headers[KEY_HEADER] = settings.key
    return (
        f"{settings.base_url.rstrip('/')}/v1/decide",
        headers,
        {"provider": provider, "request": payload},
    )


def _decision_request(
    settings: Settings, provider: str, conversation_id: str, payload: dict[str, Any]
) -> tuple[str, dict[str, str], dict[str, Any]]:
    url, headers, body = _request(settings, provider, conversation_id, payload)
    headers[DEADLINE_HEADER] = str(int(settings.timeout * 1000))
    if settings.mode == "shadow":  # the server decides but keeps no state
        headers[MODE_HEADER] = "shadow"
    return url, headers, body


class Breaker:
    """After `threshold` failures in a row (no answer, or a 5xx or 429), stop asking for
    `open_for` seconds; then let one call through to see whether AgentCompile is back."""

    def __init__(
        self,
        threshold: int = FAILURES_TO_OPEN,
        open_for: float = OPEN_FOR_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.threshold, self.open_for, self._clock = threshold, open_for, clock
        self._failures = 0
        self._open_until = 0.0
        self._trying = False
        self._lock = threading.Lock()

    def allow(self) -> bool:
        with self._lock:
            if self._failures < self.threshold:
                return True
            if self._clock() < self._open_until or self._trying:
                return False
            self._trying = True  # the one call that finds out
            return True

    def record(self, ok: bool) -> None:
        with self._lock:
            self._trying = False
            if ok:
                self._failures = 0
                return
            self._failures += 1
            if self._failures >= self.threshold:
                self._open_until = self._clock() + self.open_for


def _answered(status: int) -> bool:
    """The service is up: anything but a 5xx or being told to back off."""
    return status < 500 and status != 429


def _read(response: Any, ms: float) -> tuple[Decision | None, float, str | None]:
    if response.status_code != 200:
        return None, ms, f"HTTP {response.status_code}"
    decision = Decision.parse(response.json())
    return decision, ms, None if decision else "malformed decision"


# Threads sync decision calls may use at once, across every wrapped client in the process.
# The pool grows to it as calls overlap (a thread is started only when none is idle), so an
# agent serving many conversations from one process never queues one conversation's decision
# behind others' (with a fixed 8, 32 conversations in one process failed open whenever the
# service slowed down: the queued calls' time ran out before they were sent).
MAX_DECIDE_THREADS = 256

_POOL: concurrent.futures.ThreadPoolExecutor | None = None
_POOL_LOCK = threading.Lock()


def _pool() -> concurrent.futures.ThreadPoolExecutor:
    """Where sync decision calls run, so the caller waits a bounded time (httpx's timeouts
    are per phase, and a name lookup has none). A call past its time finishes there, bounded
    by httpx's own timeouts, while the caller has moved on."""
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = concurrent.futures.ThreadPoolExecutor(
                MAX_DECIDE_THREADS, "agentcompile-decide"
            )
        return _POOL


class _Sent:
    """When a pooled call started (was sent): its deadline runs from there."""

    def __init__(self) -> None:
        self.event = threading.Event()
        self.at = 0.0

    def mark(self) -> None:
        self.at = time.monotonic()
        self.event.set()


def _timed(sent: _Sent, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    sent.mark()
    return fn(*args, **kwargs)


class Decider:
    def __init__(self, settings: Settings, http: httpx.Client | None = None) -> None:
        self.settings = settings
        self._http = http or httpx.Client(timeout=settings.timeout)
        self.breaker = Breaker()

    def decide(
        self, provider: str, conversation_id: str, payload: dict[str, Any]
    ) -> tuple[Decision | None, float, str | None]:
        """(decision or None, milliseconds, error text). The call has `timeout` from when it
        is sent; waiting for a free thread (only past MAX_DECIDE_THREADS calls at once) is
        bounded by `timeout` too."""
        if not self.breaker.allow():
            return None, 0.0, "circuit open"
        url, headers, body = _decision_request(
            self.settings, provider, conversation_id, payload
        )
        timeout = self.settings.timeout
        start = time.perf_counter()
        future = None
        sent = _Sent()
        try:
            future = _pool().submit(
                _timed, sent, self._http.post, url, headers=headers, json=body, timeout=timeout
            )
            if not sent.event.wait(timeout):
                raise concurrent.futures.TimeoutError("no thread free to send it")
            left = sent.at + timeout - time.monotonic()
            response = future.result(timeout=max(0.0, left))
        except concurrent.futures.TimeoutError:
            if future is not None:
                future.cancel()
            self.breaker.record(False)
            return None, (time.perf_counter() - start) * 1000, "deadline"
        except Exception as exc:  # fail open
            self.breaker.record(False)
            return None, (time.perf_counter() - start) * 1000, type(exc).__name__
        self.breaker.record(_answered(response.status_code))
        try:
            return _read(response, (time.perf_counter() - start) * 1000)
        except Exception as exc:
            return None, (time.perf_counter() - start) * 1000, type(exc).__name__


class AsyncDecider:
    def __init__(self, settings: Settings, http: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self._http = http or httpx.AsyncClient(timeout=settings.timeout)
        self.breaker = Breaker()

    async def decide(
        self, provider: str, conversation_id: str, payload: dict[str, Any]
    ) -> tuple[Decision | None, float, str | None]:
        if not self.breaker.allow():
            return None, 0.0, "circuit open"
        url, headers, body = _decision_request(
            self.settings, provider, conversation_id, payload
        )
        start = time.perf_counter()
        try:
            response = await asyncio.wait_for(
                self._http.post(url, headers=headers, json=body, timeout=self.settings.timeout),
                timeout=self.settings.timeout,
            )
        except asyncio.TimeoutError:
            self.breaker.record(False)
            return None, (time.perf_counter() - start) * 1000, "deadline"
        except Exception as exc:  # fail open
            self.breaker.record(False)
            return None, (time.perf_counter() - start) * 1000, type(exc).__name__
        self.breaker.record(_answered(response.status_code))
        try:
            return _read(response, (time.perf_counter() - start) * 1000)
        except Exception as exc:
            return None, (time.perf_counter() - start) * 1000, type(exc).__name__
