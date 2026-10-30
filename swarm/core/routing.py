"""Replica selection and the retry policy.

The gateway picks one replica per stage per request and retries a failed hop on
a surviving replica. Both engines share this so a failover decision is identical
in a container run and a virtual run.

Selection is deterministic given the scenario seed: round-robin walks a fixed
order, and `least_loaded` breaks ties by that same order. Repeatable runs are a
stated requirement, so nothing here consults an unseeded source of randomness.
"""

from __future__ import annotations

from dataclasses import dataclass, field

POLICIES = ("round_robin", "least_loaded", "random")


@dataclass
class Router:
    depth: int
    policy: str = "round_robin"
    seed: int = 42
    _cursor: dict[int, int] = field(default_factory=dict)
    _rng: object = None

    def __post_init__(self) -> None:
        if self.policy not in POLICIES:
            raise ValueError(f"unknown routing policy {self.policy!r}")
        if self.policy == "random":
            import random
            self._rng = random.Random(self.seed)

    def pick(self, stage: int, live: list[str],
             load: dict[str, int] | None = None,
             exclude: set[str] | None = None) -> str | None:
        """Choose a replica for `stage`, or None if the stage has no live replica."""
        cands = sorted(n for n in live if not (exclude and n in exclude))
        if not cands:
            return None
        if self.policy == "round_robin":
            i = self._cursor.get(stage, 0) % len(cands)
            self._cursor[stage] = i + 1
            return cands[i]
        if self.policy == "least_loaded":
            load = load or {}
            return min(cands, key=lambda n: (load.get(n, 0), n))
        return self._rng.choice(cands)  # type: ignore[union-attr]

    def route(self, live_by_stage: dict[int, list[str]],
              load: dict[str, int] | None = None) -> list[str] | None:
        """A full path, one replica per stage, or None if any stage is uncovered."""
        path: list[str] = []
        for k in range(self.depth):
            pick = self.pick(k, live_by_stage.get(k, []), load)
            if pick is None:
                return None
            path.append(pick)
        return path


@dataclass
class RetryPolicy:
    """Per-hop retry. A hop that exhausts its attempts fails the request."""

    max_attempts: int = 3
    backoff_s: float = 0.05
    timeout_s: float = 5.0

    def backoff_for(self, attempt: int) -> float:
        """Linear backoff: attempt 1 waits 0, each later attempt one step more.

        Linear rather than exponential so that detection and recovery times stay
        legible against the heartbeat interval instead of being dominated by
        client-side waiting.
        """
        return max(0, attempt - 1) * self.backoff_s
