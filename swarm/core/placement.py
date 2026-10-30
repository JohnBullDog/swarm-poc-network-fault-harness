"""Static placement: assign model layers to stages, and stages to nodes.

Placement is static per scenario, as the design doc specifies -- failover only
reroutes among surviving replicas of a stage, it never re-places blocks.

Two strategies, so the experiments can compare them directly:

  capability_proportional  W_k proportional to the stage holders' bandwidth, so
                           every stage time t_k is roughly equal and no device
                           gates the pipeline (report Section 8).
  equal                    every stage holds 1/D of the model, which lets the
                           weakest node set the clock for everyone.

This module is pure: both engines import it, so a container run and a virtual
run of the same scenario place work identically.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import costmodel as cm
from .scenario import Scenario


@dataclass
class StagePlan:
    stage: int
    nodes: list[str]                 # the W replica node ids
    tiers: list[int]                 # tier of each replica
    share: float                     # f_k, fraction of active weight bytes
    weight_bytes: float              # f_k * P_a * b
    layer_start: int                 # real mode: first layer held (inclusive)
    layer_end: int                   # real mode: last layer held (exclusive)

    @property
    def n_layers(self) -> int:
        return self.layer_end - self.layer_start


@dataclass
class Plan:
    scenario: Scenario
    stages: list[StagePlan]
    node_tiers: dict[str, int] = field(default_factory=dict)
    _stage_index: dict[str, int] | None = field(default=None, repr=False)

    # -- lookups ----------------------------------------------------------
    @property
    def depth(self) -> int:
        return len(self.stages)

    @property
    def nodes(self) -> list[str]:
        return [n for st in self.stages for n in st.nodes]

    def stage_of(self, node_id: str) -> int:
        """Which stage holds this node.

        Memoised on first use. The scan it replaces was O(D*W), and callers
        invoke it once per node during setup, so it was O(N^2) overall -- about
        a million list-membership tests at the report's N=1000, repeated by the
        engine, the membership table and the node table.
        """
        index = self._stage_index
        if index is None:
            index = {n: st.stage for st in self.stages for n in st.nodes}
            self._stage_index = index
        try:
            return index[node_id]
        except KeyError:
            raise KeyError(node_id) from None

    def plan_for(self, node_id: str) -> StagePlan:
        return self.stages[self.stage_of(node_id)]

    # -- cost-model predictions -------------------------------------------
    def stage_time(self, stage: int, node_id: str, context_len: int,
                   with_overhead: bool = True) -> float:
        """t_k for one specific replica, including its hop cost.

        `with_overhead` adds the fitted per-stage and per-hop constants from
        `scenario.synthetic`. The design doc's synthetic stage time is "active
        bytes read divided by the emulated memory bandwidth, plus measured
        overhead", and those constants are what tools/calibrate.py measures.
        Passing False gives the report's bare expression, which is what
        `t_token_s_report` is compared against.
        """
        st = self.stages[stage]
        scn = self.scenario
        tier = self.node_tiers[node_id]
        link = scn.link_for(tier)
        kv = st.share * context_len * scn.profile.kv_bytes_per_token
        serial = scn.profile.activation_bytes / link.rate_bytes
        hop = link.delay_s + serial
        if with_overhead:
            hop += scn.synthetic.hop_overhead_ms / 1000.0
        t = cm.stage_time(st.weight_bytes, kv, scn.mem_bw_of(tier), hop)
        if with_overhead:
            t += scn.synthetic.stage_overhead_ms / 1000.0
        return t

    def stage_times_mean(self, context_len: int,
                         with_overhead: bool = True) -> list[float]:
        """Mean t_k across each stage's replicas."""
        return [
            sum(self.stage_time(st.stage, n, context_len, with_overhead)
                for n in st.nodes) / len(st.nodes)
            for st in self.stages
        ]

    def compute_time(self, stage: int, node_id: str, context_len: int) -> float:
        """A node's occupancy serving one token: compute, no link term."""
        st = self.stages[stage]
        scn = self.scenario
        tier = self.node_tiers[node_id]
        kv = st.share * context_len * scn.profile.kv_bytes_per_token
        return ((st.weight_bytes + kv) / scn.mem_bw_of(tier)
                + scn.synthetic.stage_overhead_ms / 1000.0)

    def compute_times_mean(self, context_len: int) -> list[float]:
        return [
            sum(self.compute_time(st.stage, n, context_len) for n in st.nodes)
            / len(st.nodes)
            for st in self.stages
        ]

    def entry_hop_s(self) -> float:
        """The gateway -> stage 0 hop.

        The activation flows forward only: gateway -> s0 -> ... -> s(D-1) ->
        gateway. That is D stage egresses (each stage forwards once, the last
        back to the gateway) plus this one gateway egress. The report's
        D*(l + a/beta) term covers the D stage egresses exactly; this entry hop
        is the one addition, and it is real -- a token has to get into the
        pipeline. It is reported separately so the measurement can be compared
        against the report's form as well as against the full path.
        """
        link = self.scenario.gateway_link
        return (link.delay_s
                + self.scenario.profile.activation_bytes / link.rate_bytes
                + self.scenario.synthetic.hop_overhead_ms / 1000.0)

    def predictions(self, context_len: int | None = None) -> dict:
        """Every quantity the experiments compare measurements against.

        T_token is the sum of per-stage times rather than the report's single-B
        form, because a mixed-tier path has no single B. The two agree exactly
        when the pool is homogeneous.
        """
        scn = self.scenario
        s = context_len if context_len is not None else scn.workload.prompt_tokens
        per_stage = self.stage_times_mean(s)
        bws = [scn.mem_bw_of(t) for t in self.node_tiers.values()]
        peak = cm.x_max(scn.profile, bws, s)
        n = len(self.node_tiers)
        g = scn.workload.concurrency
        entry = self.entry_hop_s()

        best = sum(min(self.stage_time(st.stage, n_, s) for n_ in st.nodes)
                   for st in self.stages)
        worst = sum(max(self.stage_time(st.stage, n_, s) for n_ in st.nodes)
                    for st in self.stages)

        return {
            # The report's expression: compute + D*(l + a/beta), with neither
            # the entry hop nor the fitted overheads.
            "t_token_s_report": sum(self.stage_times_mean(s, with_overhead=False)),
            # The full measurable path, including the gateway's entry hop.
            "t_token_s_mean": sum(per_stage) + entry,
            "t_token_s_best": best + entry,
            "t_token_s_worst": worst + entry,
            "entry_hop_s": entry,
            "stage_times_s": per_stage,
            "slowest_stage": int(max(range(len(per_stage)), key=lambda i: per_stage[i])),
            "x_max_tok_s": peak,
            "x_of_g_tok_s": cm.x_of_g(peak, g, n),
            "x_pipe_tok_s": cm.x_pipe(per_stage),

            # The report's ramp X(G) = X_max * min(1, G/N) is an upper bound. It
            # treats G streams as keeping G nodes continuously busy, but a
            # stream is idle while its activation is on the wire, so it can only
            # be reached when the network term is negligible beside compute.
            #
            # For a closed-loop workload the achievable rate is the smaller of
            # two classical bounds:
            #   delay bound     G / T_token    -- G streams, each serial
            #   capacity bound  W / max_k t_k  -- the slowest stage's replicas
            # Reporting these alongside the ramp is what lets the concurrency
            # sweep show which bound actually binds at each G.
            "x_delay_bound_tok_s": g / (sum(per_stage) + entry)
            if (sum(per_stage) + entry) > 0 else 0.0,
            # Capacity in the report's terms, where t_k includes the hop.
            "x_capacity_tok_s": scn.composition.width * cm.x_pipe(per_stage),
            # Capacity by node occupancy, which is what actually limits a
            # stage: a node is busy for its compute only, while the activation's
            # transit overlaps with it serving the next token. This is the
            # bound the measured saturation plateau should be compared against.
            "x_capacity_compute_tok_s": scn.composition.width
            * cm.x_pipe(self.compute_times_mean(s)),
            "x_closed_loop_tok_s": min(
                g / (sum(per_stage) + entry) if (sum(per_stage) + entry) > 0 else 0.0,
                scn.composition.width * cm.x_pipe(self.compute_times_mean(s))),
            "n_nodes": n,
            "concurrency_to_fill": cm.concurrency_to_fill(n),
            "bytes_per_token": scn.profile.bytes_per_token(s),
            "activation_bytes": scn.profile.activation_bytes,
            "kv_working_set_bytes": scn.profile.kv_working_set(s, g),
            "stage_availability": [
                cm.stage_availability(
                    min(cm.TIERS[t].availability for t in st.tiers), len(st.nodes))
                for st in self.stages
            ],
        }


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
def _largest_remainder(shares: list[float], total: int) -> list[int]:
    """Split `total` integer units across `shares`, preserving proportions.

    Every stage gets at least one unit, since a stage holding zero layers is
    not a stage.
    """
    n = len(shares)
    if total < n:
        raise ValueError(f"cannot split {total} layers across {n} stages")
    raw = [s * total for s in shares]
    base = [max(1, int(r)) for r in raw]
    while sum(base) > total:                      # shrink the largest holders
        i = max(range(n), key=lambda j: (base[j], raw[j]))
        if base[i] > 1:
            base[i] -= 1
        else:
            break
    rem = sorted(range(n), key=lambda j: raw[j] - int(raw[j]), reverse=True)
    i = 0
    while sum(base) < total:
        base[rem[i % n]] += 1
        i += 1
    return base


def build(scn: Scenario) -> Plan:
    node_tiers = scn.node_tiers()
    by_stage = scn.nodes_by_stage()
    depth = scn.composition.depth

    # Each stage's capability: the mean bandwidth of its replicas. Replicas hold
    # the same block, so one number per stage is needed; the mean is the right
    # summary when a scenario mixes tiers within a stage, and is exact in the
    # common case where a stage's replicas share a tier.
    stage_bw = [
        sum(scn.mem_bw_of(node_tiers[n]) for n in by_stage[k]) / len(by_stage[k])
        for k in range(depth)
    ]

    if scn.placement == "equal":
        shares = [1.0 / depth] * depth
    elif scn.placement == "capability_proportional":
        total = sum(stage_bw)
        shares = [b / total for b in stage_bw]
    else:
        raise ValueError(f"unknown placement {scn.placement!r}")

    layer_counts = _largest_remainder(shares, scn.profile.layers)
    stages: list[StagePlan] = []
    cursor = 0
    for k in range(depth):
        nodes = by_stage[k]
        stages.append(StagePlan(
            stage=k,
            nodes=nodes,
            tiers=[node_tiers[n] for n in nodes],
            share=shares[k],
            weight_bytes=shares[k] * scn.profile.weight_read_bytes,
            layer_start=cursor,
            layer_end=cursor + layer_counts[k],
        ))
        cursor += layer_counts[k]

    return Plan(scenario=scn, stages=stages, node_tiers=node_tiers)
