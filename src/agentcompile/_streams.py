"""Compiled answers as streams: iterators shaped like the SDKs' Stream objects."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any


class CompiledStream:
    def __init__(self, items: list[Any]) -> None:
        self._items = items

    def __iter__(self) -> Iterator[Any]:
        return iter(self._items)

    def __enter__(self) -> CompiledStream:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def close(self) -> None:
        return None


class AsyncCompiledStream:
    def __init__(self, items: list[Any]) -> None:
        self._items = items

    def __aiter__(self) -> AsyncCompiledStream:
        self._it = iter(self._items)
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
