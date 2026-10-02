"""The trail: one JSON line per model call, so you can see what AgentCompile did with each one.

Routes: `compiled` (answered without your model), `forwarded` (sent to your model),
`fail-open` (AgentCompile failed or was slow, or answered with a tool you didn't offer; sent to
your model), `shadow` (decided but always forwarded), `no-conversation` (no conversation id;
sent to your model), `unsupported` (a request a compiled answer couldn't honour: `n` > 1, a
forced `tool_choice`, a `response_format`; sent to your model without asking).

The file is kept under 10 MB: past that it moves to `trail.jsonl.1` (the one before that is
dropped) and a new one starts."""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

MAX_BYTES = 10 * 1024 * 1024
DEFAULT_PATH = Path(
    os.environ.get("AGENTCOMPILE_TRAIL", Path.home() / ".agentcompile" / "trail.jsonl")
)

OnEvent = Callable[[dict[str, Any]], None]


class Trail:
    def __init__(
        self, path: str | Path | bool | None = True, on_event: OnEvent | None = None
    ) -> None:
        self.path: Path | None = (
            DEFAULT_PATH if path is True else (Path(path) if path else None)
        )
        self.on_event = on_event
        self.max_bytes = MAX_BYTES
        self._lock = threading.Lock()

    def record(self, **event: Any) -> dict[str, Any]:
        event = {
            "ts": round(time.time(), 3),
            **{k: v for k, v in event.items() if v is not None},
        }
        if self.path is not None:
            try:
                with self._lock:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    with self.path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(event, default=str) + "\n")
                        full = fh.tell() > self.max_bytes
                    if full:
                        os.replace(self.path, self.path.with_name(self.path.name + ".1"))
            except OSError:
                pass  # the trail never breaks the agent
        if self.on_event is not None:
            with contextlib.suppress(Exception):
                self.on_event(event)
        return event
