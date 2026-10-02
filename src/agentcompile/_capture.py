"""Capture: each model call's request and answer, sent to AgentCompile in the background so it
can find the jobs your agent repeats (opt-in: `wrap(..., capture=True)`).

Never in the way: calls are queued and sent by one background thread in batches; a full queue
drops the oldest, a failed send is dropped and counted, and nothing here ever raises into your
agent. Each record is one line in the exchange-log shape AgentCompile's importer reads:
{"provider", "conversation_id", "timestamp", "request", "response"}.
"""

from __future__ import annotations

import atexit
import collections
import threading
import time
from datetime import datetime, timezone
from typing import Any

import httpx

from ._assemble import assemble
from ._decide import CONVERSATION_HEADER, Settings, _request
from ._payload import jsonable, payload
from ._scrub import scrub_call, scrub_value

MAX_QUEUE = 2000
BATCH = 50
INTERVAL = 1.0


_ALL: list[Capturer] = []


def flush_all(timeout: float = 5.0) -> None:
    for capturer in list(_ALL):
        capturer.flush(timeout)


class Capturer:
    def __init__(
        self,
        settings: Settings,
        http: httpx.Client | None = None,
        scrub_key: bytes | None = None,
    ) -> None:
        self.settings = settings
        # None: send as is (scrub=False). Otherwise personal data is tokenized before queueing.
        self.scrub_key = scrub_key
        self._http = http
        self._queue: collections.deque[dict[str, Any]] = collections.deque(maxlen=MAX_QUEUE)
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self.sent = 0
        self.dropped = 0
        self.failed = 0
        atexit.register(self.flush, 2.0)
        _ALL.append(self)

    def add(
        self,
        provider: str,
        conversation_id: str | None,
        kwargs: dict[str, Any],
        response: Any,
        stream: bool = False,
        streamed: dict[str, Any] | None = None,
        customer: str | None = None,
    ) -> None:
        """Queue one call. Never raises. `stream` without `streamed`: the answer wasn't kept
        (the record says so); `streamed`: the answer was assembled from its stream."""
        try:
            request, answer = payload(kwargs), None if stream else jsonable(response)
            if self.scrub_key is not None:
                request = scrub_call(request, self.scrub_key)
                answer = scrub_call(answer, self.scrub_key)
            record = {
                "provider": provider,
                "conversation_id": conversation_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "request": request,
                "response": answer,
            }
            if customer:
                # The customer's id, scrubbed like everything else (an email becomes a token).
                record["end_user"] = (
                    scrub_value(customer, self.scrub_key, "customer")
                    if self.scrub_key is not None
                    else customer
                )
            if self.scrub_key is not None:
                record["scrubbed"] = True
            if stream:
                record["stream"] = True
            if streamed is not None:
                record["stream"] = True
                record["stream_complete"] = streamed["complete"]
            with self._lock:
                if len(self._queue) == self._queue.maxlen:
                    self.dropped += 1  # the deque drops the oldest
                self._queue.append(record)
            self._start()
            if len(self._queue) >= BATCH:
                self._wake.set()
        except Exception:
            self.dropped += 1

    def add_stream(
        self,
        provider: str,
        conversation_id: str | None,
        kwargs: dict[str, Any],
        chunks: list[Any],
        complete: bool,
        customer: str | None = None,
    ) -> None:
        """Queue a streamed call once its stream is done: the chunks assembled into the answer
        a non-streamed call would have had. A stream stopped early is kept and marked."""
        self.add(
            provider,
            conversation_id,
            kwargs,
            assemble(provider, chunks),
            streamed={"complete": complete, "chunks": len(chunks)},
            customer=customer,
        )

    def flush(self, timeout: float = 5.0) -> None:
        """Send everything queued now (tests, and at exit)."""
        deadline = time.monotonic() + timeout
        while self._queue and time.monotonic() < deadline:
            self._send_batch()

    def _start(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, name="agentcompile-capture")
            self._thread.daemon = True
            self._thread.start()

    def _run(self) -> None:
        while True:
            self._wake.wait(INTERVAL)
            self._wake.clear()
            while self._queue:
                self._send_batch()

    def _send_batch(self) -> None:
        with self._lock:
            batch = [self._queue.popleft() for _ in range(min(BATCH, len(self._queue)))]
        if not batch:
            return
        url, headers, _ = _request(self.settings, batch[0]["provider"], "capture", {})
        headers.pop(CONVERSATION_HEADER, None)
        url = url.rsplit("/v1/decide", 1)[0] + "/v1/capture"
        try:
            if self._http is None:
                self._http = httpx.Client(timeout=self.settings.timeout)
            response = self._http.post(url, headers=headers, json={"exchanges": batch})
            if response.status_code == 200:
                self.sent += len(batch)
            else:
                self.failed += len(batch)
        except Exception:
            self.failed += len(batch)
