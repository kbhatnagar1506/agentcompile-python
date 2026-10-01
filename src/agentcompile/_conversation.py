"""The conversation id: a `conversation_id=` keyword on create(), or a context block."""

from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Iterator

_current: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "agentcompile_conversation", default=None
)


@contextlib.contextmanager
def conversation(conversation_id: str) -> Iterator[None]:
    """Every model call inside the block belongs to this conversation:

    with agentcompile.conversation(ticket.id):
        run_agent(...)
    """
    token = _current.set(conversation_id)
    try:
        yield
    finally:
        _current.reset(token)


def current() -> str | None:
    return _current.get()
