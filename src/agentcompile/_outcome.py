"""How a conversation ended, reported by you: `agentcompile.outcome(ticket.id, "resolved")`.

AgentCompile learns only from conversations that went well; an outcome you report is the
strongest evidence it has. Sent in the background (never blocks, never raises), using the
key and address of the client you wrapped last.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any

import httpx

from ._decide import CONVERSATION_HEADER, Settings, _request

OUTCOMES = ("resolved", "escalated", "unresolved", "abandoned", "reopened", "complaint")

_settings: list[Settings] = []  # the most recently wrapped client's, last
_threads: list[threading.Thread] = []
sent = 0
failed = 0


def remember(settings: Settings) -> None:
    _settings.append(settings)


def outcome(conversation_id: str, label: str, note: str | None = None) -> None:
    """Report how a conversation ended: one of resolved, escalated, unresolved, abandoned,
    reopened, complaint. Raises ValueError for anything else (a typo would be silently lost)."""
    if label not in OUTCOMES:
        raise ValueError(f"outcome must be one of {', '.join(OUTCOMES)}")
    if not _settings:
        return  # nothing wrapped yet: nowhere to send it
    record: dict[str, Any] = {
        "conversation_id": conversation_id,
        "outcome": label,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if note:
        record["note"] = note
    thread = threading.Thread(target=_send, args=(_settings[-1], record), daemon=True)
    _threads.append(thread)
    thread.start()


def _send(settings: Settings, record: dict[str, Any]) -> None:
    global sent, failed
    url, headers, _ = _request(settings, "openai", "outcome", {})
    headers.pop(CONVERSATION_HEADER, None)
    url = url.rsplit("/v1/decide", 1)[0] + "/v1/outcome"
    try:
        with httpx.Client(timeout=settings.timeout) as http:
            response = http.post(url, headers=headers, json={"outcomes": [record]})
        if response.status_code == 200:
            sent += 1
        else:
            failed += 1
    except Exception:
        failed += 1


def flush(timeout: float = 5.0) -> None:
    for thread in list(_threads):
        thread.join(timeout)
    _threads.clear()
