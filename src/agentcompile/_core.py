"""What every wrapped create() does: decide, then answer compiled or call the real method."""

from __future__ import annotations

import time
from typing import Any, Callable

from ._conversation import current
from ._decide import Decision
from ._payload import payload
from ._trail import Trail

LIVE, SHADOW = "live", "shadow"


class Router:
    """Holds the decision client, the trail and the mode for one wrapped client."""

    def __init__(
        self, provider: str, decider: Any, trail: Trail, mode: str, capturer: Any = None
    ) -> None:
        if mode not in (LIVE, SHADOW):
            raise ValueError("mode must be 'live' or 'shadow'")
        self.capturer = capturer
        self.provider = provider
        self.decider = decider
        self.trail = trail
        self.mode = mode

    def prepare(self, kwargs: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
        """(conversation id, kwargs for the real call): our keyword never reaches the model."""
        kwargs = dict(kwargs)
        conversation_id = kwargs.pop("conversation_id", None) or current()
        return conversation_id, kwargs

    def plan(
        self, conversation_id: str | None, decision: Decision | None, error: str | None
    ) -> tuple[str, Decision | None]:
        """(route, the decision to answer with or None to forward)."""
        if conversation_id is None:
            return "no-conversation", None
        if decision is None:
            return "fail-open", None
        if decision.action == "forward":
            return "forwarded", None
        if self.mode == SHADOW:
            return "shadow", None
        return "compiled", decision

    def capture(
        self, conversation_id: str | None, kwargs: dict[str, Any], result: Any, stream: bool
    ) -> None:
        if self.capturer is not None:
            self.capturer.add(self.provider, conversation_id, kwargs, result, stream)

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
        if conversation_id is not None:
            decision, decide_ms, error = router.decider.decide(
                router.provider, conversation_id, payload(real_kwargs)
            )
        route, answer = router.plan(conversation_id, decision, error)
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
                router.capture(conversation_id, real_kwargs, result, stream)
                return result
        result = original(*args, **real_kwargs)
        router.capture(conversation_id, real_kwargs, result, stream)
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
        if conversation_id is not None:
            decision, decide_ms, error = await router.decider.decide(
                router.provider, conversation_id, payload(real_kwargs)
            )
        route, answer = router.plan(conversation_id, decision, error)
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
                router.capture(conversation_id, real_kwargs, result, stream)
                return result
        result = await original(*args, **real_kwargs)
        router.capture(conversation_id, real_kwargs, result, stream)
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
