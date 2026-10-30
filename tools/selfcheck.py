"""Invariant checks for the shared core, runnable without Docker.

These are the properties the experiments assume. If one of them breaks, every
number the harness produces is suspect, so this runs in a second and is worth
running before any sweep.

The headline check reproduces the research report's appendix worked example from
the cost model alone. That example was derived independently of this code, so
agreement is a real check rather than a tautology.

    python -m tools.selfcheck
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from swarm.core import churn as churn_mod                 # noqa: E402
from swarm.core import costmodel as cm                    # noqa: E402
from swarm.core import placement, scenario as scn_mod     # noqa: E402
from swarm.core.events import (DERIVED_TABLES, EVENT_SCHEMA,   # noqa: E402
                               columns_for)
from swarm.core.membership import Membership              # noqa: E402
from swarm.core.routing import Router                     # noqa: E402
from swarm.core.scenario import (Churn, Composition, FaultEvent, LinkProfile,
                                 Scenario, Workload)      # noqa: E402

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    results.append((PASS if condition else FAIL, name, detail))


def close(a: float, b: float, tol: float = 0.02) -> bool:
    return abs(a - b) <= tol * max(abs(a), abs(b), 1e-12)


# ---------------------------------------------------------------------------
def check_report_example_a() -> None:
    """Appendix A: 1000 Pi-class nodes, 1T MoE @ INT4, 60B active, D=50.

    The report states: 30 GB read per token, ~3.05 s per token single-stream,
    X_max ~333 tokens/s, and G ~= N = 1000 to fill the pool.
    """
    p = cm.REPORT_MOE
    check("appendix A: P_a*b = 30 GB",
          close(p.weight_read_bytes, 30e9),
          f"{p.weight_read_bytes / 1e9:.2f} GB")

    b_pi = 10 * cm.GB
    t = cm.token_latency(p, b_pi, depth=50, link_delay_s=0.001,
                         link_bw=1e9, context_len=0)
    check("appendix A: T_token ~= 3.05 s", close(t, 3.05, 0.01), f"{t:.4f} s")

    peak = cm.x_max(p, [b_pi] * 1000, context_len=0)
    check("appendix A: X_max ~= 333 tok/s", close(peak, 333.3, 0.01),
          f"{peak:.1f} tok/s")
    check("appendix A: G to fill ~= 1000",
          cm.concurrency_to_fill(1000) == 1000)


def check_report_example_b() -> None:
    """Appendix B: mixed pool, 2750 units, aggregate 50,000 GB/s, X_max ~1667."""
    units = {0: 2000, 1: 500, 2: 200, 3: 50}
    bws = [cm.TIERS[t].mem_bw for t, n in units.items() for _ in range(n)]
    total = sum(bws)
    check("appendix B: aggregate bandwidth = 50,000 GB/s",
          close(total, 50_000 * cm.GB), f"{total / cm.GB:,.0f} GB/s")
    peak = cm.x_max(cm.REPORT_MOE, bws, 0)
    check("appendix B: X_max ~= 1667 tok/s", close(peak, 1666.7, 0.01),
          f"{peak:.1f} tok/s")
    t3_share = cm.TIERS[3].mem_bw * 50 / total
    check("appendix B: 50 tier-3 nodes supply ~60% of bandwidth",
          close(t3_share, 0.60, 0.02), f"{t3_share * 100:.1f}%")


def check_stage_availability() -> None:
    """1 - (1-p)^W, and the monotonicity that sizing W from it depends on."""
    check("availability: W=1 gives p", close(cm.stage_availability(0.4, 1), 0.4))
    check("availability: p=0.4, W=3 -> 0.784",
          close(cm.stage_availability(0.4, 3), 0.784))
    vals = [cm.stage_availability(0.4, w) for w in range(1, 8)]
    check("availability rises monotonically with W",
          all(b > a for a, b in zip(vals, vals[1:])))


def _scn(**kw) -> Scenario:
    base = dict(
        name="selfcheck", duration_s=60.0, model_profile="poc_small",
        composition=Composition(depth=3, width=2, tiers={2: 6}),
        network={2: LinkProfile(delay_ms=5.0, rate_mbit=500.0)},
        workload=Workload(concurrency=4, prompt_tokens=32, gen_tokens=8),
    )
    base.update(kw)
    return Scenario(**base)


def check_placement() -> None:
    scn = _scn()
    plan = placement.build(scn)
    check("placement: shares sum to 1",
          close(sum(s.share for s in plan.stages), 1.0, 1e-9))
    check("placement: layers partition the model exactly",
          sum(s.n_layers for s in plan.stages) == scn.profile.layers
          and all(s.n_layers >= 1 for s in plan.stages),
          f"{[s.n_layers for s in plan.stages]}")
    check("placement: layer blocks are contiguous and ordered",
          all(a.layer_end == b.layer_start
              for a, b in zip(plan.stages, plan.stages[1:])))
    check("placement: node count = D*W", len(plan.nodes) == 6)

    # Equal sharding on a mixed pool must let the weakest node gate its stage;
    # capability-proportional must even the stage times out. This is the
    # report's dominant failure mode, so the two must differ here.
    mixed = _scn(composition=Composition(depth=3, width=1, tiers={3: 1, 2: 1, 0: 1}),
                 network={0: LinkProfile(delay_ms=50.0, rate_mbit=10.0),
                          2: LinkProfile(delay_ms=5.0, rate_mbit=500.0),
                          3: LinkProfile(delay_ms=1.0, rate_mbit=1000.0)})
    eq = placement.build(mixed.with_(placement="equal")).stage_times_mean(0)
    cap = placement.build(mixed.with_(placement="capability_proportional")) \
        .stage_times_mean(0)
    spread_eq = max(eq) / min(eq)
    spread_cap = max(cap) / min(cap)
    check("placement: capability-proportional evens stage times vs equal",
          spread_cap < spread_eq,
          f"spread equal={spread_eq:.1f}x, proportional={spread_cap:.1f}x")

    # A deep pipeline of weak nodes cannot have fewer layers than stages.
    try:
        placement.build(_scn(composition=Composition(depth=40, width=1,
                                                     tiers={2: 40})))
        ok = False
    except ValueError:
        ok = True
    check("placement: rejects D > model layers", ok)


def check_scenario_validation() -> None:
    cases = [
        ("tier count must equal D*W",
         dict(composition=Composition(depth=3, width=2, tiers={2: 5}))),
        ("container mode forbids time dilation", dict(time_scale=4.0)),
        ("real mode needs containers", dict(mode="real", engine="virtual")),
        ("death_after must exceed heartbeat",
         dict(coordinator=scn_mod.CoordinatorCfg(heartbeat_ms=1000,
                                                 death_after_ms=500))),
        ("poisson needs a rate",
         dict(workload=Workload(concurrency=1, arrival="poisson"))),
        ("unknown placement", dict(placement="nonsense")),
    ]
    for label, kw in cases:
        try:
            scn_mod.validate(_scn(**kw))
            ok = False
        except ValueError:
            ok = True
        check(f"validation rejects: {label}", ok)
    try:
        scn_mod.validate(_scn())
        ok = True
    except ValueError as exc:
        ok, _ = False, exc
    check("validation accepts a sound scenario", ok)


def check_scenario_roundtrip() -> None:
    scn = _scn(churn=Churn(kind="schedule",
                           schedule=(FaultEvent(12.0, "kill", "n-s1-r0"),)))
    again = scn_mod.from_dict(scn.to_dict())
    check("scenario survives a dict round-trip",
          again.to_json() == scn.to_json())
    check("round-trip keeps int-keyed tier maps",
          again.composition.tiers == {2: 6} and 2 in again.network)


def check_churn_determinism() -> None:
    scn = _scn(duration_s=1800.0, seed=7,
               composition=Composition(depth=3, width=3, tiers={1: 9}),
               churn=Churn(kind="random"))
    plan = placement.build(scn)
    a, _ = churn_mod.generate(scn, plan)
    b, _ = churn_mod.generate(scn, plan)
    key = lambda evs: [(round(e.at_s, 9), e.action, e.target) for e in evs]
    check("churn: identical on regeneration", key(a) == key(b),
          f"{len(a)} events")

    other = churn_mod.generate(scn.with_(seed=8), plan)[0]
    check("churn: a different seed gives a different trace",
          key(a) != key(other))

    check("churn: events are time-ordered",
          all(x.at_s <= y.at_s for x, y in zip(a, a[1:])))

    # The guard must never let a stage reach zero live replicas, and every
    # suppression must be recorded.
    scn2 = scn.with_(churn=Churn(kind="random", protect_last_replica=True))
    kept, dropped = churn_mod.generate(scn2, plan)
    live = {n: True for n in plan.node_tiers}
    worst = {}
    for ev in kept:
        live[ev.target] = ev.action not in churn_mod.DOWN_ACTIONS
        for st in plan.stages:
            n_live = sum(1 for n in st.nodes if live[n])
            worst[st.stage] = min(worst.get(st.stage, 99), n_live)
    check("churn: protect_last_replica never empties a stage",
          all(v >= 1 for v in worst.values()), f"min live per stage={worst}")
    check("churn: suppressed faults are all recorded with a reason",
          all("why" in d for d in dropped), f"{len(dropped)} suppressed")


def check_membership() -> None:
    m = Membership(death_after_s=2.0)
    m.add("a", 2, 0, 0.0)
    m.add("b", 2, 0, 0.0)
    check("membership: nothing dies before the threshold", m.tick(1.9) == [])
    dead = m.tick(2.5)
    check("membership: both die past the threshold", len(dead) == 2)
    check("membership: a death is reported once", m.tick(3.0) == [])
    check("membership: a heartbeat revives", m.heartbeat("a", 3.1) is True)
    check("membership: live set reflects it", m.live() == {"a"})
    check("membership: coverage needs every stage",
          m.covered(1) is True and m.covered(2) is False)


def check_routing() -> None:
    r = Router(depth=3, policy="round_robin", seed=1)
    picks = [r.pick(0, ["n-b", "n-a"]) for _ in range(4)]
    check("routing: round-robin cycles a sorted list",
          picks == ["n-a", "n-b", "n-a", "n-b"], f"{picks}")
    check("routing: no live replica gives None", r.pick(0, []) is None)
    check("routing: exclusion is honoured",
          r.pick(0, ["n-a", "n-b"], exclude={"n-a"}) == "n-b")
    full = r.route({0: ["a"], 1: ["b"], 2: ["c"]})
    check("routing: a full path is one node per stage", full == ["a", "b", "c"])
    check("routing: an uncovered stage gives no path",
          r.route({0: ["a"], 1: [], 2: ["c"]}) is None)
    ll = Router(depth=1, policy="least_loaded")
    check("routing: least_loaded prefers the idle replica",
          ll.pick(0, ["a", "b"], load={"a": 5, "b": 0}) == "b")


def check_event_schema() -> None:
    check("events: every kind declares columns",
          all(len(columns_for(k)) > 7 for k in EVENT_SCHEMA))
    check("events: every kind carries both clocks",
          all({"sim_ts", "wall_ts"} <= set(columns_for(k)) for k in EVENT_SCHEMA))
    for kind in ("hop", "token_done", "membership", "fault_applied"):
        check(f"events: '{kind}' is declared", kind in EVENT_SCHEMA)
    # `recovery` and `runs` are derived by the collector, not emitted by any
    # component, and do not carry the common event columns. Declaring recovery
    # as an event kind made its file permanently mismatch its own schema.
    check("events: derived tables are not declared as event kinds",
          all(t not in EVENT_SCHEMA for t in DERIVED_TABLES),
          f"derived: {list(DERIVED_TABLES)}")

    # Every declared kind must actually be emitted somewhere, or the schema is
    # describing a table that will never exist.
    import ast
    import pathlib
    emitted = set()
    for path in pathlib.Path(REPO / "swarm").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in ("emit", "emit_at")
                    and node.args
                    and isinstance(node.args[0], ast.Constant)):
                emitted.add(node.args[0].value)
    orphans = sorted(set(EVENT_SCHEMA) - emitted)
    check("events: every declared kind is emitted somewhere",
          not orphans, f"orphans: {orphans}")


def check_virtual_scheduler() -> None:
    from swarm.virtual.scheduler import Env

    order: list[tuple[float, str]] = []

    def proc(env, name, delay, times):
        for _ in range(times):
            yield env.timeout(delay)
            order.append((env.now, name))

    def run_once() -> list[tuple[float, str]]:
        order.clear()
        env = Env()
        env.process(proc(env, "a", 1.0, 3))
        env.process(proc(env, "b", 1.0, 3))     # ties with a at every instant
        env.process(proc(env, "c", 0.5, 4))
        env.run(until=10.0)
        return list(order)

    first, second = run_once(), run_once()
    check("scheduler: a run is reproducible", first == second,
          f"{len(first)} wakeups")
    check("scheduler: events fire in time order",
          all(a[0] <= b[0] for a, b in zip(first, first[1:])))

    env = Env()
    won: list[str] = []

    def racer(env):
        slow = env.timeout(5.0, "slow")
        fast = env.timeout(1.0, "fast")
        winner = yield env.any_of([slow, fast])
        won.append(winner.value)

    env.process(racer(env))
    env.run(until=10.0)
    check("scheduler: any_of returns the first to fire", won == ["fast"], f"{won}")


def check_prediction_consistency() -> None:
    scn = _scn()
    plan = placement.build(scn)
    p = plan.predictions()
    check("predictions: full path = report form + entry hop",
          close(p["t_token_s_mean"], p["t_token_s_report"] + p["entry_hop_s"], 1e-9))
    check("predictions: closed-loop is the min of its two bounds",
          close(p["x_closed_loop_tok_s"],
                min(p["x_delay_bound_tok_s"], p["x_capacity_compute_tok_s"]), 1e-9))
    check("predictions: compute capacity >= report capacity",
          p["x_capacity_compute_tok_s"] >= p["x_capacity_tok_s"],
          f"{p['x_capacity_compute_tok_s']:.1f} vs {p['x_capacity_tok_s']:.1f}")
    check("predictions: best <= mean <= worst latency",
          p["t_token_s_best"] <= p["t_token_s_mean"] <= p["t_token_s_worst"])

    # The ramp must over-predict the closed-loop rate wherever the network term
    # is non-negligible; that gap is a finding, so assert it is present.
    check("predictions: the ramp exceeds the closed-loop bound here",
          p["x_of_g_tok_s"] > p["x_closed_loop_tok_s"],
          f"ramp={p['x_of_g_tok_s']:.1f} vs closed={p['x_closed_loop_tok_s']:.1f}")

    # A homogeneous pool is the case where the report's single-B form is exact.
    homo = _scn(composition=Composition(depth=3, width=1, tiers={2: 3}),
                workload=Workload(concurrency=1, prompt_tokens=0, gen_tokens=1))
    hp = placement.build(homo)
    direct = cm.token_latency(homo.profile, homo.mem_bw_of(2), 3,
                              homo.link_for(2).delay_s,
                              homo.link_for(2).rate_bytes, 0)
    check("predictions: report form matches single-B expression when homogeneous",
          close(hp.predictions(0)["t_token_s_report"], direct, 1e-6),
          f"{hp.predictions(0)['t_token_s_report']:.6f} vs {direct:.6f}")


def check_model_profiles() -> None:
    q = cm.QWEN_0_5B
    check("qwen profile: stage-held bytes = 1.432 GB",
          close(q.weight_read_bytes, 1.4318e9, 0.01),
          f"{q.weight_read_bytes / 1e9:.3f} GB")
    check("qwen profile: activation a = 2*d*b",
          close(q.activation_bytes, 2 * 896 * 4))
    check("poc_small: ~23 ms per stage at D=3 on tier 2",
          close(cm.POC_SMALL.weight_read_bytes / 3 / cm.TIERS[2].mem_bw,
                0.0200, 0.05),
          f"{cm.POC_SMALL.weight_read_bytes / 3 / cm.TIERS[2].mem_bw * 1000:.1f} ms")
    for name, prof in cm.MODEL_PROFILES.items():
        check(f"profile '{name}': active <= total params",
              prof.active_params <= prof.total_params)


# ---------------------------------------------------------------------------
def main() -> int:
    for fn in (check_report_example_a, check_report_example_b,
               check_stage_availability, check_placement,
               check_scenario_validation, check_scenario_roundtrip,
               check_churn_determinism, check_membership, check_routing,
               check_event_schema, check_virtual_scheduler,
               check_prediction_consistency, check_model_profiles):
        try:
            fn()
        except Exception as exc:        # a crash is a failure, not a stack trace
            check(f"{fn.__name__} raised", False, f"{type(exc).__name__}: {exc}")

    width = max(len(name) for _, name, _ in results)
    failed = 0
    for status, name, detail in results:
        if status == FAIL:
            failed += 1
        mark = " ok " if status == PASS else "FAIL"
        print(f"  [{mark}] {name:<{width}}  {detail}")
    print(f"\n{len(results) - failed}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
