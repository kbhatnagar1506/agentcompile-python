"""AgentCompile: wrap your agent's model client.

    import agentcompile
    client = agentcompile.wrap(OpenAI(), key="ack_...")
    client.chat.completions.create(model=..., messages=..., conversation_id=ticket.id)

Known jobs run compiled (no model call); everything else goes to your model unchanged, with your
own provider key. If AgentCompile is down or slow, calls go straight to your model. Every call
is written to the trail (`agentcompile trail`).
"""

from __future__ import annotations

import inspect
import os
from typing import Any

from ._capture import Capturer, flush_all
from ._conversation import conversation
from ._core import LIVE, SHADOW, Proxy, Router, async_create, sync_create
from ._decide import AsyncDecider, Decider, Settings
from ._trail import OnEvent, Trail

__all__ = ["LIVE", "SHADOW", "Settings", "conversation", "flush", "wrap"]
try:
    from ._version import __version__
except ImportError:  # running from a source checkout that was never built
    __version__ = "0.0.0"

DEFAULT_URL = "https://api.tryagentcompile.com"


def wrap(
    client: Any,
    *,
    key: str | None = None,
    base_url: str | None = None,
    mode: str = LIVE,
    timeout: float = 2.0,
    trail: str | bool | None = True,
    on_event: OnEvent | None = None,
    company: str | None = None,
    http_client: Any = None,
    capture: bool | None = None,
) -> Any:
    """Wrap an OpenAI- or Anthropic-style client; returns an object you use exactly like it.

    key: your AgentCompile key (or AGENTCOMPILE_KEY). base_url: AGENTCOMPILE_URL or the default.
    mode: "live" answers known jobs; "shadow" decides but always calls your model, so you can
    see what it would have done. timeout: seconds to wait for a decision before failing open.
    trail: a path, True for ~/.agentcompile/trail.jsonl, or False. on_event: called per call.
    capture: send each call's request and answer to AgentCompile in the background, so it can
    find the jobs your agent repeats (opt-in; or AGENTCOMPILE_CAPTURE=1). Never slows a call.
    """
    settings = Settings(
        base_url=base_url or os.environ.get("AGENTCOMPILE_URL", DEFAULT_URL),
        key=key or os.environ.get("AGENTCOMPILE_KEY"),
        company=company or os.environ.get("AGENTCOMPILE_COMPANY"),
        timeout=timeout,
    )
    the_trail = Trail(trail, on_event)
    if capture is None:
        capture = os.environ.get("AGENTCOMPILE_CAPTURE", "") in ("1", "true", "yes")
    # One sender per wrapped client; it always sends from its own thread (sync client).
    capturer = (
        Capturer(settings, http_client if _is_sync_http(http_client) else None)
        if capture
        else None
    )

    chat = getattr(client, "chat", None)
    completions = getattr(chat, "completions", None)
    if completions is not None and hasattr(completions, "create"):
        from . import _openai

        router = _router("openai", completions.create, settings, the_trail, mode, http_client)
        router.capturer = capturer
        create = _create(router, completions.create, _openai.build)
        return Proxy(
            client,
            {"chat": Proxy(chat, {"completions": Proxy(completions, {"create": create})})},
        )

    messages = getattr(client, "messages", None)
    if messages is not None and hasattr(messages, "create"):
        from . import _anthropic

        router = _router("anthropic", messages.create, settings, the_trail, mode, http_client)
        router.capturer = capturer
        create = _create(router, messages.create, _anthropic.build)
        return Proxy(client, {"messages": Proxy(messages, {"create": create})})

    raise TypeError("agentcompile.wrap expects an OpenAI- or Anthropic-style client")


def flush(timeout: float = 5.0) -> None:
    """Send every captured call still queued (short scripts and tests; also runs at exit)."""
    flush_all(timeout)


def _router(
    provider: str, create: Any, settings: Settings, trail: Trail, mode: str, http_client: Any
) -> Router:
    decider: Any = (
        AsyncDecider(settings, http_client)
        if _is_async(create)
        else Decider(settings, http_client)
    )
    return Router(provider, decider, trail, mode)


def _create(router: Router, original: Any, build: Any) -> Any:
    return (
        async_create(router, original, build)
        if _is_async(original)
        else sync_create(router, original, build)
    )


def _is_sync_http(http_client: Any) -> bool:
    """Any client with a plain (not async) post(): httpx, httpx2, a test client."""
    post = getattr(http_client, "post", None)
    return post is not None and not inspect.iscoroutinefunction(inspect.unwrap(post))


def _is_async(fn: Any) -> bool:
    """The SDKs wrap create() in decorators; look through them."""
    return inspect.iscoroutinefunction(inspect.unwrap(fn))
