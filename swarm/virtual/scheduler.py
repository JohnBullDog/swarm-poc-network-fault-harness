"""A minimal deterministic discrete-event scheduler.

Generator-based processes, so the virtual engine's gateway and worker logic reads
like the asyncio code it mirrors: a process yields an event and resumes when it
fires. Written rather than taken from a library so the whole time model is
inspectable in one short file -- the engine's credibility rests on it.

Determinism. Every event carries a monotonically increasing sequence number, and
ties at the same timestamp break by that number, so a run is reproducible to the
last event regardless of dict ordering or platform.
"""

from __future__ import annotations

import heapq
from typing import Any, Callable, Generator, Iterable


class Event:
    """Something a process can wait for. Fires at most once."""

    __slots__ = ("env", "callbacks", "fired", "value", "name")

    def __init__(self, env: "Env", name: str = ""):
        self.env = env
        self.callbacks: list[Callable[[Event], None]] = []
        self.fired = False
        self.value: Any = None
        self.name = name

    def succeed(self, value: Any = None) -> "Event":
        if self.fired:
            return self
        self.fired = True
        self.value = value
        for cb in self.callbacks:
            cb(self)
        self.callbacks.clear()
        return self

    def add_callback(self, cb: Callable[["Event"], None]) -> None:
        if self.fired:
            cb(self)
        else:
            self.callbacks.append(cb)

    def __bool__(self) -> bool:
        return self.fired


Process = Generator[Event, Any, Any]


class Env:
    def __init__(self, until: float = float("inf")):
        self.now = 0.0
        self.until = until
        self._queue: list[tuple[float, int, Event]] = []
        self._seq = 0
        self.n_events = 0

    # -- primitives -------------------------------------------------------
    def schedule(self, delay: float, event: Event) -> Event:
        self._seq += 1
        heapq.heappush(self._queue, (self.now + max(0.0, delay), self._seq, event))
        return event

    def timeout(self, delay: float, value: Any = None) -> Event:
        ev = Event(self, "timeout")
        ev.value = value
        return self.schedule(delay, ev)

    def event(self, name: str = "") -> Event:
        return Event(self, name)

    def any_of(self, events: Iterable[Event]) -> Event:
        """Fires when the first of `events` fires; its value is the winner.

        Used wherever the engine races a result against a timeout, which is how
        a hop timeout is modelled.
        """
        out = self.event("any_of")
        for src in events:
            src.add_callback(lambda e, out=out: out.succeed(e))
        return out

    def process(self, gen: Process) -> Event:
        """Run a generator as a process. The returned event fires when it ends."""
        done = self.event("process_done")

        def step(_: Event | None = None) -> None:
            try:
                ev = next(gen) if _ is None else gen.send(_.value)
            except StopIteration as stop:
                done.succeed(getattr(stop, "value", None))
                return
            ev.add_callback(step)

        # Start the process at the current instant rather than immediately, so
        # that process creation order cannot affect the outcome.
        starter = self.timeout(0.0)
        starter.add_callback(lambda _: step(None))
        return done

    # -- the loop ---------------------------------------------------------
    def step(self) -> bool:
        if not self._queue:
            return False
        when, _, event = self._queue[0]
        if when > self.until:
            return False
        heapq.heappop(self._queue)
        self.now = when
        self.n_events += 1
        event.succeed(event.value)
        return True

    def run(self, until: float | None = None) -> float:
        if until is not None:
            self.until = until
        while self.step():
            pass
        self.now = min(self.until, max(self.now, 0.0))
        return self.now
