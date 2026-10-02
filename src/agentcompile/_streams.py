"""Compiled answers as streams: iterators shaped like the SDKs' Stream objects (one pass:
`for`, `next()` and `__anext__` all read from the same position, as theirs do)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any


class CompiledStream:
    def __init__(self, items: list[Any]) -> None:
        self._it = iter(items)

    def __iter__(self) -> Iterator[Any]:
        return self._it

    def __next__(self) -> Any:
        return next(self._it)

    def __enter__(self) -> CompiledStream:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def close(self) -> None:
        return None


class AsyncCompiledStream:
    def __init__(self, items: list[Any]) -> None:
        self._it = iter(items)

    def __aiter__(self) -> AsyncCompiledStream:
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration from None

    async def __aenter__(self) -> AsyncCompiledStream:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def close(self) -> None:
        return None
