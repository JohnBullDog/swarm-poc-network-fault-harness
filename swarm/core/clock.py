"""Two clocks, one interface.

Simulated time and real time are tracked separately everywhere in this harness,
and every event record carries both. The distinction matters because the two
engines relate them differently:

  WallClock     sim time advances with real time. Used by container runs, where
                a stage really sleeps and tc netem really delays packets.
                `time_scale` can dilate sim time, but container runs must leave
                it at 1.0 -- netem delays are real milliseconds and cannot be
                dilated along with it.

  VirtualClock  sim time is advanced by the event scheduler and has no relation
                to real time at all. A thousand-node hour runs in seconds.

Cross-container comparability: all containers share the host kernel's
CLOCK_REALTIME, and every component is given the same RUN_EPOCH, so sim_ts is
computed from one origin and no clock synchronisation is needed.
"""

from __future__ import annotations

import asyncio
import time
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    def sim(self) -> float:
        """Seconds of simulated time since the run epoch."""

    def wall(self) -> float:
        """Absolute host wall-clock time (unix seconds)."""

    def real_elapsed(self) -> float:
        """Seconds of real time since the run epoch."""

    async def sleep(self, sim_seconds: float) -> None:
        """Advance `sim_seconds` of simulated time."""


class WallClock:
    """Real time. `sim()` is real elapsed time scaled by `time_scale`."""

    kind = "wall"

    def __init__(self, run_epoch: float | None = None, time_scale: float = 1.0):
        self.run_epoch = run_epoch if run_epoch is not None else time.time()
        self.time_scale = time_scale

    def sim(self) -> float:
        return (time.time() - self.run_epoch) * self.time_scale

    def wall(self) -> float:
        return time.time()

    def real_elapsed(self) -> float:
        return time.time() - self.run_epoch

    async def sleep(self, sim_seconds: float) -> None:
        if sim_seconds > 0:
            await asyncio.sleep(sim_seconds / self.time_scale)

    def busy_sleep(self, sim_seconds: float) -> None:
        """Block the thread for a sim interval.

        Synthetic stage compute uses this rather than `await sleep` when a
        scenario asks a worker to occupy its CPU share, so that CPU contention
        between co-located containers is real rather than elided.
        """
        if sim_seconds <= 0:
            return
        end = time.perf_counter() + sim_seconds / self.time_scale
        while time.perf_counter() < end:
            pass


class VirtualClock:
    """Simulated time, advanced only by the discrete-event scheduler."""

    kind = "virtual"
    time_scale = float("inf")

    def __init__(self, scheduler, run_epoch: float | None = None):
        self._sched = scheduler
        self.run_epoch = run_epoch if run_epoch is not None else time.time()

    def sim(self) -> float:
        return self._sched.now

    def wall(self) -> float:
        return time.time()

    def real_elapsed(self) -> float:
        return time.time() - self.run_epoch

    async def sleep(self, sim_seconds: float) -> None:  # pragma: no cover
        raise NotImplementedError(
            "the virtual engine schedules timeouts directly; it does not await")
