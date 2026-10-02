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
import json
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
# The server takes up to 8 MiB a request: batches stay under 4 MiB, and a call bigger than
# MAX_RECORD_BYTES on its own (a huge history) is dropped and counted.
MAX_BATCH_BYTES = 4 * 1024 * 1024
MAX_RECORD_BYTES = 7 * 1024 * 1024
INTERVAL = 1.0


_ALL: dict[tuple[Any, ...], Capturer] = {}
_ALL_LOCK = threading.Lock()


def flush_all(timeout: float = 5.0) -> None:
    for capturer in list(_ALL.values()):
        capturer.flush(timeout)


def capturer_for(
    settings: Settings, http: httpx.Client | None, scrub_key: bytes | None
) -> Capturer:
    """One sender (one queue, one thread) per destination: wrapping a client per request, as
    some apps do, reuses it instead of starting another each time."""
    key = (settings.base_url, settings.key, settings.company, scrub_key, id(http))
    with _ALL_LOCK:
        found = _ALL.get(key)
        if found is None:
            found = _ALL[key] = Capturer(settings, http, scrub_key)
        return found


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

    def _take(self) -> list[str]:
        """Up to BATCH queued calls, encoded, under MAX_BATCH_BYTES together (the first one
        alone may be up to MAX_RECORD_BYTES); what doesn't fit goes back to the front."""
        with self._lock:
            records = [self._queue.popleft() for _ in range(min(BATCH, len(self._queue)))]
        batch: list[str] = []
        size = 0
        for i, record in enumerate(records):
            try:
                line = json.dumps(record, default=str)
            except Exception:
                self.dropped += 1
                continue
            if len(line) > MAX_RECORD_BYTES:
                self.dropped += 1
                continue
            if batch and size + len(line) > MAX_BATCH_BYTES:
                with self._lock:  # the rest waits for the next batch, in order
                    self._queue.extendleft(reversed(records[i:]))
                break
            batch.append(line)
            size += len(line)
        return batch

    def _send_batch(self) -> None:
        batch = self._take()
        if not batch:
            return
        url, headers, _ = _request(self.settings, "openai", "capture", {})
        headers.pop(CONVERSATION_HEADER, None)
        headers["content-type"] = "application/json"
        url = url.rsplit("/v1/decide", 1)[0] + "/v1/capture"
        try:
            if self._http is None:
                self._http = httpx.Client(timeout=self.settings.timeout)
            body = '{"exchanges": [' + ", ".join(batch) + "]}"
            response = self._http.post(url, headers=headers, content=body.encode("utf-8"))
            if response.status_code == 200:
                self.sent += len(batch)
            else:
                self.failed += len(batch)
        except Exception:
            self.failed += len(batch)
