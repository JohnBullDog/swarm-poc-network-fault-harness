"""Heartbeat bookkeeping and death detection.

Pure logic with time passed in, so the container coordinator and the virtual
engine detect deaths by exactly the same rule. Detection latency is one of the
headline measurements (experiment 4), so this must not differ between engines.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NodeState:
    node_id: str
    tier: int
    stage: int
    alive: bool = True
    last_seen: float = 0.0
    hb_count: int = 0
    joins: int = 0
    deaths: int = 0


@dataclass
class Membership:
    """Tracks liveness from heartbeats; marks a node dead after a configured gap."""

    death_after_s: float
    nodes: dict[str, NodeState] = field(default_factory=dict)

    def add(self, node_id: str, tier: int, stage: int, now: float) -> None:
        self.nodes[node_id] = NodeState(node_id, tier, stage, True, now, joins=1)

    def heartbeat(self, node_id: str, now: float) -> bool:
        """Record a heartbeat. Returns True if this revived a node marked dead."""
        st = self.nodes.get(node_id)
        if st is None:
            return False
        st.last_seen = now
        st.hb_count += 1
        if not st.alive:
            st.alive = True
            st.joins += 1
            return True
        return False

    def tick(self, now: float) -> list[tuple[str, float]]:
        """Mark nodes dead whose heartbeat gap exceeds the threshold.

        Returns (node_id, observed_gap_s) for each newly-dead node.
        """
        dead: list[tuple[str, float]] = []
        for st in self.nodes.values():
            if not st.alive:
                continue
            gap = now - st.last_seen
            if gap > self.death_after_s:
                st.alive = False
                st.deaths += 1
                dead.append((st.node_id, gap))
        return dead

    def live(self) -> set[str]:
        return {n for n, st in self.nodes.items() if st.alive}

    def live_by_stage(self) -> dict[int, list[str]]:
        out: dict[int, list[str]] = {}
        for st in self.nodes.values():
            if st.alive:
                out.setdefault(st.stage, []).append(st.node_id)
        for v in out.values():
            v.sort()
        return out

    def gap(self, node_id: str, now: float) -> float:
        return now - self.nodes[node_id].last_seen

    def covered(self, depth: int) -> bool:
        """True when every stage has at least one live replica."""
        live = self.live_by_stage()
        return all(live.get(k) for k in range(depth))
