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

from ._capture import capturer_for, flush_all
from ._conversation import conversation
from ._core import (
    LIVE,
    SHADOW,
    Proxy,
    Router,
    async_create,
    async_forward,
    sync_create,
    sync_forward,
)
from ._decide import AsyncDecider, Decider, Settings
from ._outcome import OUTCOMES, outcome
from ._outcome import flush as _flush_outcomes
from ._outcome import remember as _remember
from ._scrub import load_key
from ._trail import OnEvent, Trail
from ._transport import AsyncCaptureTransport, CaptureTransport, classes, lib_of

__all__ = [
    "LIVE",
    "OUTCOMES",
    "SHADOW",
    "AsyncCaptureTransport",
    "CaptureTransport",
    "Settings",
    "async_http_client",
    "async_transport",
    "conversation",
    "flush",
    "http_client",
    "outcome",
    "transport",
    "wrap",
]
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
    scrub: bool = True,
) -> Any:
    """Wrap an OpenAI- or Anthropic-style client; returns an object you use exactly like it.

    key: your AgentCompile key (or AGENTCOMPILE_KEY). base_url: AGENTCOMPILE_URL or the default.
    mode: "live" answers known jobs; "shadow" decides but always calls your model, so you can
    see what it would have done. timeout: seconds to wait for a decision before failing open.
    trail: a path, True for ~/.agentcompile/trail.jsonl, or False. on_event: called per call.
    capture: send each call's request and answer to AgentCompile in the background, so it can
    find the jobs your agent repeats (opt-in; or AGENTCOMPILE_CAPTURE=1). Never slows a call.
    scrub: with capture, turn emails, cards, phones and account numbers into keyed tokens on
    this machine before anything is sent (default True; the key is AGENTCOMPILE_SCRUB_KEY or
    one created once in ~/.agentcompile/).
    """
    settings = _settings(key, base_url, company, timeout, mode)
    the_trail = Trail(trail, on_event)
    _remember(settings)  # outcome() sends with the client wrapped last
    if capture is None:
        capture = os.environ.get("AGENTCOMPILE_CAPTURE", "") in ("1", "true", "yes")
    # One sender per wrapped client; it always sends from its own thread (sync client).
    capturer = (
        capturer_for(
            settings,
            http_client if _is_sync_http(http_client) else None,
            load_key() if scrub else None,
        )
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
        overrides: dict[str, Any] = {
            "chat": Proxy(chat, {"completions": Proxy(completions, {"create": create})})
        }
        responses = getattr(client, "responses", None)
        if responses is not None and hasattr(responses, "create"):
            # The Responses API (the OpenAI Agents SDK's default): captured, always the model's.
            forward = _router(
                "openai", responses.create, settings, the_trail, mode, http_client
            )
            forward.capturer = capturer
            make = async_forward if _is_async(responses.create) else sync_forward
            overrides["responses"] = Proxy(
                responses, {"create": make(forward, responses.create)}
            )
        return Proxy(client, overrides)

    messages = getattr(client, "messages", None)
    if messages is not None and hasattr(messages, "create"):
        from . import _anthropic

        router = _router("anthropic", messages.create, settings, the_trail, mode, http_client)
        router.capturer = capturer
        create = _create(router, messages.create, _anthropic.build)
        return Proxy(client, {"messages": Proxy(messages, {"create": create})})

    raise TypeError("agentcompile.wrap expects an OpenAI- or Anthropic-style client")


def transport(
    wrapped: Any = None,
    *,
    lib: str | None = None,
    key: str | None = None,
    base_url: str | None = None,
    company: str | None = None,
    scrub: bool = True,
    timeout: float = 2.0,
) -> Any:
    """A transport that captures every model call through it, for frameworks whose client you
    don't hold (LangChain, LiteLLM, CrewAI, Pydantic AI, ...). Capture only: calls are never
    answered compiled. `wrapped`: the transport to send through (default the library's own).
    `lib`: "httpx" (default) or "httpx2", the fork newer provider SDKs require; taken from
    `wrapped` when it is given. The other arguments are wrap()'s; `timeout` bounds each
    capture send, not your calls."""
    sync, _ = classes(lib_of(wrapped, lib))
    return sync(_capturer(key, base_url, company, scrub, timeout), wrapped)


def async_transport(
    wrapped: Any = None,
    *,
    lib: str | None = None,
    key: str | None = None,
    base_url: str | None = None,
    company: str | None = None,
    scrub: bool = True,
    timeout: float = 2.0,
) -> Any:
    """The async version of transport()."""
    _, async_ = classes(lib_of(wrapped, lib))
    return async_(_capturer(key, base_url, company, scrub, timeout), wrapped)


def http_client(*, lib: str = "httpx", **options: Any) -> Any:
    """A client whose model calls are captured: pass it as the framework's `http_client`.
    `lib="httpx2"` for SDKs built on httpx2. Takes transport()'s keyword arguments."""
    import importlib

    module = importlib.import_module(lib)
    return module.Client(transport=transport(lib=lib, **options), timeout=_CALL_TIMEOUT)


def async_http_client(*, lib: str = "httpx", **options: Any) -> Any:
    """An async client whose model calls are captured."""
    import importlib

    module = importlib.import_module(lib)
    return module.AsyncClient(
        transport=async_transport(lib=lib, **options), timeout=_CALL_TIMEOUT
    )


# The provider SDKs' own default: a model call can take minutes.
_CALL_TIMEOUT = 600.0


def _settings(
    key: str | None, base_url: str | None, company: str | None, timeout: float, mode: str
) -> Settings:
    return Settings(
        base_url=base_url or os.environ.get("AGENTCOMPILE_URL", DEFAULT_URL),
        key=key or os.environ.get("AGENTCOMPILE_KEY"),
        company=company or os.environ.get("AGENTCOMPILE_COMPANY"),
        timeout=timeout,
        mode=mode,
    )


def _capturer(
    key: str | None, base_url: str | None, company: str | None, scrub: bool, timeout: float
) -> Any:
    settings = _settings(key, base_url, company, timeout, LIVE)
    _remember(settings)
    return capturer_for(settings, None, load_key() if scrub else None)


def flush(timeout: float = 5.0) -> None:
    """Send every captured call and outcome still queued (short scripts and tests; also runs
    at exit)."""
    flush_all(timeout)
    _flush_outcomes(timeout)


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
