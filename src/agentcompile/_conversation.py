"""The conversation id: a `conversation_id=` keyword on create(), or a context block."""

from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Iterator

_current: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "agentcompile_conversation", default=None
)
_customer: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "agentcompile_customer", default=None
)


@contextlib.contextmanager
def conversation(conversation_id: str, customer: str | None = None) -> Iterator[None]:
    """Every model call inside the block belongs to this conversation (and, optionally, this
    customer: a stable id for the person, so repeat jobs can be counted per customer):

    with agentcompile.conversation(ticket.id, customer=ticket.customer_id):
        run_agent(...)
    """
    token = _current.set(conversation_id)
    customer_token = _customer.set(customer)
    try:
        yield
    finally:
        _customer.reset(customer_token)
        _current.reset(token)


def current() -> str | None:
    return _current.get()


def current_customer() -> str | None:
    return _customer.get()
