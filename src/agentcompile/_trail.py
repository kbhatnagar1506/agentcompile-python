"""The trail: one JSON line per model call, so you can see what AgentCompile did with each one.

Routes: `compiled` (answered without your model), `forwarded` (sent to your model),
`fail-open` (AgentCompile failed or was slow; sent to your model), `shadow` (decided but always
forwarded), `no-conversation` (no conversation id; sent to your model)."""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

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
            except OSError:
                pass  # the trail never breaks the agent
        if self.on_event is not None:
            with contextlib.suppress(Exception):
                self.on_event(event)
        return event
