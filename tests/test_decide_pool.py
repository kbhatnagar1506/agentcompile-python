"""Many conversations in one process: sync decision calls never wait behind each other, and a
call's time runs from when it is sent (agentcompile/_decide.py)."""

from __future__ import annotations

import concurrent.futures
import threading
import time
from collections.abc import Iterator

import httpx
import pytest

from agentcompile import _decide
from agentcompile._decide import Decider, Settings

SAY = {"action": "say", "text": "done"}
MESSAGES = [{"role": "user", "content": "cancel order #W1"}]


def _slow(seconds: float) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        time.sleep(seconds)
        return httpx.Response(200, json=SAY)

    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.fixture
def pool_of(monkeypatch: pytest.MonkeyPatch) -> Iterator[object]:
    """Swap the process's decision pool for one of a given size."""
    made: list[concurrent.futures.ThreadPoolExecutor] = []

    def use(size: int) -> concurrent.futures.ThreadPoolExecutor:
        pool = concurrent.futures.ThreadPoolExecutor(size, "test-decide")
        made.append(pool)
        monkeypatch.setattr(_decide, "_POOL", pool)
        return pool

    yield use
    for pool in made:
        pool.shutdown(wait=False, cancel_futures=True)


def test_thirty_two_conversations_at_once_all_get_their_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The service takes 0.3 s; the budget is 1 s. With 8 threads the last calls waited
    1.2 s for a thread and failed open; the pool now grows with the calls in flight."""
    monkeypatch.setattr(_decide, "_POOL", None)
    decider = Decider(Settings("http://ac.test", "k", None, 1.0), _slow(0.3))
    start = threading.Barrier(32)

    def one(i: int) -> str | None:
        start.wait()
        _, _, error = decider.decide("openai", f"c{i}", {"messages": MESSAGES})
        return error

    with concurrent.futures.ThreadPoolExecutor(32) as callers:
        errors = list(callers.map(one, range(32)))
    assert errors == [None] * 32
    assert _decide.MAX_DECIDE_THREADS >= 32


def test_the_time_runs_from_when_the_call_is_sent(pool_of: object) -> None:
    pool_of(1)  # type: ignore[operator]
    busy = Decider(Settings("http://ac.test", "k", None, 1.0), _slow(0.3))
    waiting = Decider(Settings("http://ac.test", "k", None, 0.5), _slow(0.3))
    first = threading.Thread(target=busy.decide, args=("openai", "a", {"messages": MESSAGES}))
    first.start()
    time.sleep(0.05)
    # Waits ~0.25 s for the one thread, then the call itself takes 0.3 s: 0.55 s in all,
    # past 0.5 s counted from the queue, within it counted from the send.
    decision, _, error = waiting.decide("openai", "b", {"messages": MESSAGES})
    first.join()
    assert error is None and decision is not None and decision.text == "done"


def test_a_call_no_thread_frees_up_for_fails_open_within_its_time(pool_of: object) -> None:
    pool = pool_of(1)  # type: ignore[operator]
    release = threading.Event()
    pool.submit(release.wait, 5)  # the only thread, held
    decider = Decider(Settings("http://ac.test", "k", None, 0.2), _slow(0.0))
    started = time.perf_counter()
    decision, _, error = decider.decide("openai", "c", {"messages": MESSAGES})
    release.set()
    assert decision is None and error == "deadline"
    assert time.perf_counter() - started < 0.5


def test_a_slow_call_once_sent_still_gives_up_at_its_time(pool_of: object) -> None:
    pool_of(4)  # type: ignore[operator]
    decider = Decider(Settings("http://ac.test", "k", None, 0.2), _slow(1.0))
    started = time.perf_counter()
    decision, _, error = decider.decide("openai", "c", {"messages": MESSAGES})
    assert decision is None and error == "deadline"
    assert time.perf_counter() - started < 0.5
