"""The decision call to AgentCompile. Never raises: any problem is a `None` decision, which the
wrapper treats as forward (fail open)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import httpx

COMPANY_HEADER = "x-agentcompiler-company"
CONVERSATION_HEADER = "x-agentcompiler-conversation"
KEY_HEADER = "x-agentcompiler-key"


@dataclass(frozen=True)
class Decision:
    action: str  # "tool_call" | "say" | "forward"
    tool: str | None = None
    args: dict[str, Any] = field(default_factory=dict)
    call_id: str | None = None
    text: str | None = None
    reason: str | None = None

    @classmethod
    def parse(cls, body: Any) -> Decision | None:
        if not isinstance(body, dict):
            return None
        action = body.get("action")
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
            )
        if action == "say" and isinstance(body.get("text"), str):
            return cls(action, text=body["text"])
        if action == "forward":
            return cls(action, reason=str(body.get("reason") or ""))
        return None


@dataclass(frozen=True)
class Settings:
    base_url: str
    key: str | None
    company: str | None
    timeout: float


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


class Decider:
    def __init__(self, settings: Settings, http: httpx.Client | None = None) -> None:
        self.settings = settings
        self._http = http or httpx.Client(timeout=settings.timeout)

    def decide(
        self, provider: str, conversation_id: str, payload: dict[str, Any]
    ) -> tuple[Decision | None, float, str | None]:
        """(decision or None, milliseconds, error text)."""
        url, headers, body = _request(self.settings, provider, conversation_id, payload)
        start = time.perf_counter()
        try:
            response = self._http.post(
                url, headers=headers, json=body, timeout=self.settings.timeout
            )
            ms = (time.perf_counter() - start) * 1000
            if response.status_code != 200:
                return None, ms, f"HTTP {response.status_code}"
            decision = Decision.parse(response.json())
            return decision, ms, None if decision else "malformed decision"
        except Exception as exc:  # fail open
            return None, (time.perf_counter() - start) * 1000, type(exc).__name__


class AsyncDecider:
    def __init__(self, settings: Settings, http: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self._http = http or httpx.AsyncClient(timeout=settings.timeout)

    async def decide(
        self, provider: str, conversation_id: str, payload: dict[str, Any]
    ) -> tuple[Decision | None, float, str | None]:
        url, headers, body = _request(self.settings, provider, conversation_id, payload)
        start = time.perf_counter()
        try:
            response = await self._http.post(
                url, headers=headers, json=body, timeout=self.settings.timeout
            )
            ms = (time.perf_counter() - start) * 1000
            if response.status_code != 200:
                return None, ms, f"HTTP {response.status_code}"
            decision = Decision.parse(response.json())
            return decision, ms, None if decision else "malformed decision"
        except Exception as exc:  # fail open
            return None, (time.perf_counter() - start) * 1000, type(exc).__name__
