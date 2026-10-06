"""What every wrapped create() does: decide, then answer compiled or call the real method."""

from __future__ import annotations

import contextvars
import time
from typing import Any, Callable

from ._assemble import AsyncCapturingStream, CapturingStream
from ._conversation import current, current_customer
from ._decide import Decision
from ._payload import payload
from ._trail import Trail

LIVE, SHADOW = "live", "shadow"


def _given(value: Any) -> Any:
    """None for an argument left out, however the SDK spells that (NOT_GIVEN, omit)."""
    return None if type(value).__name__ in ("NotGiven", "Omit") else value


def unsupported(kwargs: dict[str, Any]) -> str | None:
    """Why a compiled answer couldn't honour this request (then it goes to the model, and no
    decision is asked for): several answers, a forced or forbidden tool, a fixed output
    shape."""
    n = _given(kwargs.get("n"))
    if n is not None and n != 1:
        return "n"
    if _given(kwargs.get("functions")) is not None:
        return "functions"
    shape = _given(kwargs.get("response_format"))
    if shape is not None and not (isinstance(shape, dict) and shape.get("type") == "text"):
        return "response_format"
    choice = _given(kwargs.get("tool_choice"))
    auto = choice in (None, "auto") or (
        isinstance(choice, dict) and choice.get("type") == "auto"
    )
    if not auto:
        return "tool_choice"
    return None


def offered(kwargs: dict[str, Any], tool: str | None) -> bool:
    """Whether the request offers `tool` (OpenAI's {"function": {"name"}}, or a top-level
    name)."""
    for spec in _given(kwargs.get("tools")) or []:
        spec = spec.model_dump() if hasattr(spec, "model_dump") else spec
        if not isinstance(spec, dict):
            continue
        function = spec.get("function")
        name = function.get("name") if isinstance(function, dict) else spec.get("name")
        if name == tool:
            return True
    return False


class Router:
    """Holds the decision client, the trail and the mode for one wrapped client."""

    def __init__(
        self, provider: str, decider: Any, trail: Trail, mode: str, capturer: Any = None
    ) -> None:
        if mode not in (LIVE, SHADOW):
            raise ValueError("mode must be 'live' or 'shadow'")
        self.capturer = capturer
        self._customer: contextvars.ContextVar[str | None] = contextvars.ContextVar(
            "agentcompile_call_customer", default=None
        )
        self.provider = provider
        self.decider = decider
        self.trail = trail
        self.mode = mode

    def prepare(self, kwargs: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
        """(conversation id, kwargs for the real call): our keywords never reach the model.
        The customer id rides along for capture only."""
        kwargs = dict(kwargs)
        conversation_id = kwargs.pop("conversation_id", None) or current()
        self._customer.set(kwargs.pop("customer_id", None) or current_customer())
        return conversation_id, kwargs

    def plan(
        self,
        conversation_id: str | None,
        decision: Decision | None,
        error: str | None,
        kwargs: dict[str, Any] | None = None,
    ) -> tuple[str, Decision | None, str | None]:
        """(route, the decision to answer with or None to forward, error text)."""
        if conversation_id is None:
            return "no-conversation", None, error
        if kwargs is not None and (why := unsupported(kwargs)) is not None:
            return "unsupported", None, why
        if decision is None:
            return "fail-open", None, error
        if decision.action == "forward":
            return "forwarded", None, error
        if self.mode == SHADOW:
            return "shadow", None, error
        if decision.action == "tool_call" and not offered(kwargs or {}, decision.tool):
            # A tool the agent didn't offer this turn: its loop couldn't run it.
            return "fail-open", None, "tool not offered"
        return "compiled", decision, error

    def capture(
        self,
        conversation_id: str | None,
        kwargs: dict[str, Any],
        result: Any,
        stream: bool,
        is_async: bool = False,
    ) -> Any:
        """Queue the call for capture; returns the result to hand back (a stream comes back
        wrapped, so its answer is captured once the agent has read it)."""
        if self.capturer is None:
            return result
        customer = self._customer.get()
        if not stream:
            self.capturer.add(self.provider, conversation_id, kwargs, result, customer=customer)
            return result
        capturer, provider = self.capturer, self.provider

        def done(chunks: list[Any], complete: bool) -> None:
            capturer.add_stream(provider, conversation_id, kwargs, chunks, complete, customer)

        return (AsyncCapturingStream if is_async else CapturingStream)(result, done)

    def record(
        self,
        *,
        route: str,
        conversation_id: str | None,
        kwargs: dict[str, Any],
        decision: Decision | None,
        decide_ms: float | None,
        error: str | None,
        started: float,
        stream: bool,
    ) -> None:
        self.trail.record(
            conversation=conversation_id,
            provider=self.provider,
            model=kwargs.get("model"),
            route=route,
            action=decision.action if decision else None,
            tool=decision.tool if decision else None,
            reason=(decision.reason if decision and decision.action == "forward" else None)
            or error,
            events=list(decision.events) if decision and decision.events else None,
            decide_ms=round(decide_ms, 1) if decide_ms is not None else None,
            total_ms=round((time.perf_counter() - started) * 1000, 1),
            stream=stream or None,
        )


def sync_create(
    router: Router, original: Callable[..., Any], build: Callable[..., Any]
) -> Callable[..., Any]:
    def create(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        conversation_id, real_kwargs = router.prepare(kwargs)
        decision = decide_ms = error = None
        if conversation_id is not None and unsupported(real_kwargs) is None:
            decision, decide_ms, error = router.decider.decide(
                router.provider, conversation_id, payload(real_kwargs)
            )
        route, answer, error = router.plan(conversation_id, decision, error, real_kwargs)
        stream = bool(real_kwargs.get("stream"))
        if answer is not None:
            try:
                result = build(answer, real_kwargs, stream, False)
            except (
                Exception
            ) as exc:  # fail open: a compiled answer we can't shape goes to the model
                route, error = "fail-open", f"build: {type(exc).__name__}"
            else:
                router.record(
                    route=route,
                    conversation_id=conversation_id,
                    kwargs=real_kwargs,
                    decision=decision,
                    decide_ms=decide_ms,
                    error=error,
                    started=started,
                    stream=stream,
                )
                return router.capture(conversation_id, real_kwargs, result, stream)
        result = original(*args, **real_kwargs)
        result = router.capture(conversation_id, real_kwargs, result, stream)
        router.record(
            route=route,
            conversation_id=conversation_id,
            kwargs=real_kwargs,
            decision=decision,
            decide_ms=decide_ms,
            error=error,
            started=started,
            stream=stream,
        )
        return result

    return create


def async_create(
    router: Router, original: Callable[..., Any], build: Callable[..., Any]
) -> Callable[..., Any]:
    async def create(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        conversation_id, real_kwargs = router.prepare(kwargs)
        decision = decide_ms = error = None
        if conversation_id is not None and unsupported(real_kwargs) is None:
            decision, decide_ms, error = await router.decider.decide(
                router.provider, conversation_id, payload(real_kwargs)
            )
        route, answer, error = router.plan(conversation_id, decision, error, real_kwargs)
        stream = bool(real_kwargs.get("stream"))
        if answer is not None:
            try:
                result = build(answer, real_kwargs, stream, True)
            except Exception as exc:
                route, error = "fail-open", f"build: {type(exc).__name__}"
            else:
                router.record(
                    route=route,
                    conversation_id=conversation_id,
                    kwargs=real_kwargs,
                    decision=decision,
                    decide_ms=decide_ms,
                    error=error,
                    started=started,
                    stream=stream,
                )
                return router.capture(conversation_id, real_kwargs, result, stream, True)
        result = await original(*args, **real_kwargs)
        result = router.capture(conversation_id, real_kwargs, result, stream, True)
        router.record(
            route=route,
            conversation_id=conversation_id,
            kwargs=real_kwargs,
            decision=decision,
            decide_ms=decide_ms,
            error=error,
            started=started,
            stream=stream,
        )
        return result

    return create


#: Why a Responses API call is never asked about: compiled answers come in Chat Completions' and
#: Anthropic's shapes, so these calls go to the model, captured like any other.
RESPONSES_API = "responses api"


def _forward_only(router: Router, conversation_id: str | None) -> tuple[str, str | None]:
    return (
        ("no-conversation", None) if conversation_id is None else ("unsupported", RESPONSES_API)
    )


def sync_forward(router: Router, original: Callable[..., Any]) -> Callable[..., Any]:
    """create() for an API we capture but don't answer: straight to the model."""

    def create(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        conversation_id, real_kwargs = router.prepare(kwargs)
        stream = bool(real_kwargs.get("stream"))
        result = original(*args, **real_kwargs)
        result = router.capture(conversation_id, real_kwargs, result, stream)
        route, error = _forward_only(router, conversation_id)
        router.record(
            route=route,
            conversation_id=conversation_id,
            kwargs=real_kwargs,
            decision=None,
            decide_ms=None,
            error=error,
            started=started,
            stream=stream,
        )
        return result

    return create


def async_forward(router: Router, original: Callable[..., Any]) -> Callable[..., Any]:
    async def create(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        conversation_id, real_kwargs = router.prepare(kwargs)
        stream = bool(real_kwargs.get("stream"))
        result = await original(*args, **real_kwargs)
        result = router.capture(conversation_id, real_kwargs, result, stream, True)
        route, error = _forward_only(router, conversation_id)
        router.record(
            route=route,
            conversation_id=conversation_id,
            kwargs=real_kwargs,
            decision=None,
            decide_ms=None,
            error=error,
            started=started,
            stream=stream,
        )
        return result

    return create


class Proxy:
    """Everything delegates to the real object except the attributes we override."""

    def __init__(self, target: Any, overrides: dict[str, Any]) -> None:
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_overrides", overrides)

    def __getattr__(self, name: str) -> Any:
        overrides = object.__getattribute__(self, "_overrides")
        if name in overrides:
            return overrides[name]
        return getattr(object.__getattribute__(self, "_target"), name)

    def __repr__(self) -> str:
        return f"agentcompile({object.__getattribute__(self, '_target')!r})"
