"""Deterministic churn: turn a scenario's churn config into a concrete schedule.

The schedule is generated once, on the host, before either engine starts. Both
engines then replay the identical list of faults at the identical simulated
times. That is what makes a container run and a virtual run of the same
scenario comparable: the faults are not re-drawn per engine.

`protect_last_replica` is resolved here too, at generation time rather than at
runtime, so neither engine has to make a live decision that the other might
make differently.
"""

from __future__ import annotations

import random

from .placement import Plan
from .scenario import FaultEvent, Scenario

DOWN_ACTIONS = ("disconnect", "pause", "kill")
UP_ACTIONS = ("rejoin", "unpause")


def resolve_target(target: str, plan: Plan) -> list[str]:
    """Expand a fault target into node ids.

    Accepts an exact node id, `stage:K` (all replicas of that stage),
    `tier:T` (all nodes of that tier), or `all`.
    """
    if target == "all":
        return sorted(plan.node_tiers)
    if target.startswith("stage:"):
        k = int(target.split(":", 1)[1])
        return sorted(plan.stages[k].nodes)
    if target.startswith("tier:"):
        t = int(target.split(":", 1)[1])
        return sorted(n for n, tt in plan.node_tiers.items() if tt == t)
    if target in plan.node_tiers:
        return [target]
    raise ValueError(f"fault target {target!r} matches no node")


def _expand(scn: Scenario, plan: Plan) -> list[FaultEvent]:
    out: list[FaultEvent] = []
    for ev in scn.churn.schedule:
        for node in resolve_target(ev.target, plan):
            out.append(FaultEvent(at_s=ev.at_s, action=ev.action, target=node))
    return out


def _random_model(scn: Scenario, plan: Plan) -> list[FaultEvent]:
    """Alternating up/down sessions per node, drawn from per-tier exponentials.

    Each node is seeded from (scenario seed, node id) so a node's trace does not
    depend on dict iteration order or on how many other nodes there are.
    """
    down = scn.churn.method if scn.churn.method in DOWN_ACTIONS else "disconnect"
    up = "unpause" if down == "pause" else "rejoin"
    out: list[FaultEvent] = []

    for node in sorted(plan.node_tiers):
        tier = plan.node_tiers[node]
        spec = scn.tier_spec(tier)
        sess = scn.churn.session_mean_s.get(tier, spec.session_mean_s)
        dead = scn.churn.downtime_mean_s.get(tier, spec.downtime_mean_s)
        rng = random.Random(f"{scn.seed}:{node}")

        t = rng.expovariate(1.0 / sess)           # first departure
        while t < scn.duration_s:
            out.append(FaultEvent(at_s=t, action=down, target=node))
            if down == "kill":
                break                             # a killed node never returns
            t += rng.expovariate(1.0 / dead)
            if t >= scn.duration_s:
                break
            out.append(FaultEvent(at_s=t, action=up, target=node))
            t += rng.expovariate(1.0 / sess)
    return out


def _filter_consistent(scn: Scenario, plan: Plan,
                       events: list[FaultEvent]) -> tuple[list[FaultEvent], list[dict]]:
    """Drop impossible or stage-killing events, in time order.

    Returns the kept events and a record of every drop, so the run manifest can
    state exactly which faults were suppressed and why rather than silently
    changing the experiment.
    """
    stage_of = {n: plan.stage_of(n) for n in plan.node_tiers}
    live = {n: True for n in plan.node_tiers}
    kept: list[FaultEvent] = []
    dropped: list[dict] = []

    # Stable order: by time, then down-actions before up-actions at the same
    # instant, then node id. Deterministic regardless of generation order.
    def key(ev: FaultEvent) -> tuple:
        return (ev.at_s, 0 if ev.action in DOWN_ACTIONS else 1, ev.target)

    for ev in sorted(events, key=key):
        node = ev.target
        going_down = ev.action in DOWN_ACTIONS
        if going_down and not live[node]:
            dropped.append({"at_s": ev.at_s, "action": ev.action,
                            "target": node, "why": "already down"})
            continue
        if not going_down and live[node]:
            dropped.append({"at_s": ev.at_s, "action": ev.action,
                            "target": node, "why": "already up"})
            continue
        if going_down and scn.churn.protect_last_replica:
            k = stage_of[node]
            peers = [n for n in plan.stages[k].nodes if n != node and live[n]]
            if not peers:
                dropped.append({"at_s": ev.at_s, "action": ev.action, "target": node,
                                "why": f"would leave stage {k} with no live replica"})
                continue
        live[node] = not going_down
        kept.append(ev)

    return kept, dropped


def generate(scn: Scenario, plan: Plan) -> tuple[list[FaultEvent], list[dict]]:
    """The full resolved fault schedule, plus the list of suppressed faults."""
    if scn.churn.kind == "none":
        return [], []
    if scn.churn.kind == "schedule":
        raw = _expand(scn, plan)
    elif scn.churn.kind == "random":
        raw = _random_model(scn, plan)
    else:
        raise ValueError(f"unknown churn kind {scn.churn.kind!r}")
    return _filter_consistent(scn, plan, raw)


def summarise(events: list[FaultEvent], duration_s: float) -> dict:
    """Headline churn figures for the run manifest."""
    downs = [e for e in events if e.action in DOWN_ACTIONS]
    ups = [e for e in events if e.action in UP_ACTIONS]
    return {
        "n_fault_events": len(events),
        "n_departures": len(downs),
        "n_rejoins": len(ups),
        "departures_per_min": len(downs) / (duration_s / 60.0) if duration_s else 0.0,
        "first_fault_s": min((e.at_s for e in events), default=None),
        "last_fault_s": max((e.at_s for e in events), default=None),
    }
