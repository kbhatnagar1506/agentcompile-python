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

from ._decide import CONVERSATION_HEADER, Settings, _request
from ._payload import jsonable, payload

MAX_QUEUE = 2000
BATCH = 50
INTERVAL = 1.0


_ALL: list[Capturer] = []


def flush_all(timeout: float = 5.0) -> None:
    for capturer in list(_ALL):
        capturer.flush(timeout)


class Capturer:
    def __init__(self, settings: Settings, http: httpx.Client | None = None) -> None:
        self.settings = settings
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
    ) -> None:
        """Queue one call. Never raises. A streamed answer is not captured yet: the record
        keeps the request and says so."""
        try:
            record = {
                "provider": provider,
                "conversation_id": conversation_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "request": payload(kwargs),
                "response": None if stream else jsonable(response),
            }
            if stream:
                record["stream"] = True
            with self._lock:
                if len(self._queue) == self._queue.maxlen:
                    self.dropped += 1  # the deque drops the oldest
                self._queue.append(record)
            self._start()
            if len(self._queue) >= BATCH:
                self._wake.set()
        except Exception:
            self.dropped += 1

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
