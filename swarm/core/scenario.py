"""Scenario schema: one file fully determines one run.

A scenario is plain data. It is authored as YAML on the host, validated here,
and handed to either engine (containers or virtual clock) unchanged. The same
`Scenario` object drives both, which is what makes the calibration in
experiment 7 a comparison of engines rather than of configurations.

Scenario files use friendly units (GB/s, ms, Mbit/s); everything is converted to
SI on load so the cost model only ever sees bytes and seconds.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from .costmodel import MBIT, MODEL_PROFILES, TIERS, ModelProfile, TierSpec

PLACEMENTS = ("capability_proportional", "equal")
MODES = ("synthetic", "real")
ENGINES = ("container", "virtual")
ARRIVALS = ("closed_loop", "poisson")
CHURN_KINDS = ("none", "schedule", "random")
FAULT_ACTIONS = ("disconnect", "rejoin", "pause", "unpause", "kill")


# ---------------------------------------------------------------------------
# Pieces
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LinkProfile:
    """Emulated link conditions. Applied with tc netem + tbf in container mode,
    and computed analytically in the virtual engine."""

    delay_ms: float = 0.0
    jitter_ms: float = 0.0
    loss_pct: float = 0.0
    rate_mbit: float = 1000.0

    @property
    def delay_s(self) -> float:
        return self.delay_ms / 1000.0

    @property
    def jitter_s(self) -> float:
        return self.jitter_ms / 1000.0

    @property
    def rate_bytes(self) -> float:
        return self.rate_mbit * MBIT

    @property
    def loss_frac(self) -> float:
        return self.loss_pct / 100.0


@dataclass(frozen=True)
class Composition:
    """The D x W layout and which tiers supply the N = D*W nodes."""

    depth: int                                  # D, pipeline stages
    width: int                                  # W, replicas per stage
    tiers: dict[int, int] = field(default_factory=dict)  # tier -> node count
    tier_assignment: str = "by_stage"           # by_stage | round_robin

    @property
    def n_nodes(self) -> int:
        return self.depth * self.width


@dataclass(frozen=True)
class Workload:
    concurrency: int = 1                        # G, concurrent agent streams
    prompt_tokens: int = 32                     # s at request start
    gen_tokens: int = 16                        # tokens generated per request
    arrival: str = "closed_loop"
    request_rate: float = 0.0                   # req/s, poisson arrival only
    max_requests: int = 0                       # 0 = unbounded, stop on duration


@dataclass(frozen=True)
class FaultEvent:
    """One scheduled fault. `target` is a node id, or "tier:N" / "stage:K"."""

    at_s: float
    action: str
    target: str

    def __post_init__(self) -> None:
        if self.action not in FAULT_ACTIONS:
            raise ValueError(f"unknown fault action {self.action!r}")


@dataclass(frozen=True)
class Churn:
    kind: str = "none"
    schedule: tuple[FaultEvent, ...] = ()
    # random model: per-tier session/downtime means override TIERS defaults
    session_mean_s: dict[int, float] = field(default_factory=dict)
    downtime_mean_s: dict[int, float] = field(default_factory=dict)
    # how disconnection is realised in container mode
    method: str = "disconnect"                  # disconnect | pause | kill
    protect_last_replica: bool = True           # never take a stage to zero


@dataclass(frozen=True)
class CoordinatorCfg:
    heartbeat_ms: float = 500.0
    death_after_ms: float = 2000.0
    sample_interval_ms: float = 500.0


@dataclass(frozen=True)
class RetryCfg:
    max_attempts: int = 3                       # per hop
    hop_timeout_ms: float = 5000.0
    backoff_ms: float = 50.0


@dataclass(frozen=True)
class SyntheticCfg:
    """Knobs specific to synthetic compute.

    `stage_overhead_ms` and `hop_overhead_ms` are the "plus measured overhead"
    of the design doc: they are not guesses but fitted from container runs by
    tools/calibrate.py, and the fitted values are written back into scenarios.
    """

    mem_bw_gbs: dict[int, float] = field(default_factory=dict)  # tier -> GB/s override

    # The harness's own runtime cost, fitted by tools/calibrate.py: everything
    # the cost model has no term for. `stage_overhead_ms` covers the event
    # loop's scheduling granularity and the per-request Python work;
    # `hop_overhead_ms` covers TCP handshake, HTTP framing, veth traversal and
    # the Docker bridge.
    #
    # These are PREDICTION-SIDE corrections. A container stage worker does NOT
    # sleep them: it sleeps the bare cost-model time, and this overhead is what
    # the container runtime then adds on top by itself. Adding them to the sleep
    # as well would make the fit circular -- the residual that the fit measures
    # would include the sleep the previous fit had just introduced, and
    # successive fits would diverge (1.2 -> 2.2 -> 3.3 ms).
    #
    # The virtual engine DOES add them to its service time, because it has no
    # container runtime of its own and must reproduce one to match.
    stage_overhead_ms: float = 0.0
    hop_overhead_ms: float = 0.0
    # worker request parallelism; models the per-node CPU limit as c servers
    servers_per_node: int = 1
    # Occupy the CPU for the stage time instead of awaiting it. Off by default:
    # the semaphore already serialises a memory-bound stage, and burning CPU in
    # every container would cap how many nodes one host can hold. Used for the
    # mode check, where real CPU contention between co-located nodes matters.
    busy_compute: bool = False


@dataclass(frozen=True)
class RealCfg:
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    model_dir: str = "/models/qwen2.5-0.5b"
    dtype: str = "float32"
    torch_threads: int = 1


# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Scenario:
    name: str
    seed: int = 42
    mode: str = "synthetic"
    engine: str = "container"
    duration_s: float = 60.0
    # Tokens completed before this instant are excluded from steady-state
    # statistics. The pipeline needs filling and containers need settling, and
    # mixing that transient into a throughput number would understate it.
    warmup_s: float = 5.0
    # Simulated seconds per real second. Container mode must stay at 1.0:
    # tc netem delays are real milliseconds and cannot be dilated with it.
    time_scale: float = 1.0
    model_profile: str = "poc_small"
    composition: Composition = field(default_factory=lambda: Composition(3, 1, {2: 3}))
    placement: str = "capability_proportional"
    network: dict[int, LinkProfile] = field(default_factory=dict)  # tier -> link
    # The gateway's own link. The report puts orchestration off-swarm on
    # reliable hardware, so this defaults to a fast link. It is shaped like any
    # other, because the entry hop gateway->stage0 is real and would otherwise
    # be the one unmeasured hop in the path.
    gateway_link: LinkProfile = field(
        default_factory=lambda: LinkProfile(delay_ms=1.0, rate_mbit=1000.0))
    churn: Churn = field(default_factory=Churn)
    workload: Workload = field(default_factory=Workload)
    coordinator: CoordinatorCfg = field(default_factory=CoordinatorCfg)
    retry: RetryCfg = field(default_factory=RetryCfg)
    synthetic: SyntheticCfg = field(default_factory=SyntheticCfg)
    real: RealCfg = field(default_factory=RealCfg)
    # Per-hop logging produces D+1 rows per token. At report scale (N=1000,
    # D=50) that is millions of rows per run, which costs more to write and read
    # than it informs. Token-level rows are always kept; only the hop
    # decomposition is dropped, and the manifest records that it was.
    log_hops: bool = True
    experiment: str = ""        # which planned experiment this belongs to
    sweep_point: dict[str, Any] = field(default_factory=dict)  # the varied axis
    notes: str = ""

    # -- derived ----------------------------------------------------------
    @property
    def profile(self) -> ModelProfile:
        return MODEL_PROFILES[self.model_profile]

    def tier_of(self, node_id: str) -> int:
        return self.node_tiers()[node_id]

    def link_for(self, tier: int) -> LinkProfile:
        if tier in self.network:
            return self.network[tier]
        spec = TIERS[tier]
        return LinkProfile(delay_ms=spec.delay_ms, rate_mbit=spec.rate_mbit)

    def tier_spec(self, tier: int) -> TierSpec:
        return TIERS[tier]

    def mem_bw_of(self, tier: int) -> float:
        """Emulated memory bandwidth B_i for a tier, in bytes/s."""
        override = self.synthetic.mem_bw_gbs.get(tier)
        if override is not None:
            return override * 1_000_000_000.0
        return TIERS[tier].mem_bw

    def slots(self) -> list[tuple[int, int]]:
        """Every (stage, replica) slot, stage-major."""
        return [(k, w) for k in range(self.composition.depth)
                for w in range(self.composition.width)]

    @staticmethod
    def node_id(stage: int, replica: int) -> str:
        return f"n-s{stage}-r{replica}"

    def node_tiers(self) -> dict[str, int]:
        """Map node id -> tier, deterministically from the composition."""
        comp = self.composition
        want = sum(comp.tiers.values())
        if want != comp.n_nodes:
            raise ValueError(
                f"composition tiers total {want} but D*W = {comp.n_nodes}")
        # Strongest tier first, so "by_stage" puts strong nodes on early stages.
        pool: list[int] = []
        for tier in sorted(comp.tiers, reverse=True):
            pool.extend([tier] * comp.tiers[tier])
        slots = self.slots()
        if comp.tier_assignment == "round_robin":
            # Spread tiers across stages so each stage gets a mix of replicas.
            order = sorted(range(len(slots)),
                           key=lambda i: (slots[i][1], slots[i][0]))
            assign = {}
            for pos, idx in enumerate(order):
                assign[slots[idx]] = pool[pos]
            pairs = [(s, assign[s]) for s in slots]
        elif comp.tier_assignment == "by_stage":
            pairs = list(zip(slots, pool))
        else:
            raise ValueError(f"unknown tier_assignment {comp.tier_assignment!r}")
        return {self.node_id(k, w): t for (k, w), t in pairs}

    def nodes_by_stage(self) -> dict[int, list[str]]:
        out: dict[int, list[str]] = {k: [] for k in range(self.composition.depth)}
        for k, w in self.slots():
            out[k].append(self.node_id(k, w))
        return out

    # -- (de)serialisation ------------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        d["network"] = {str(k): asdict(v) for k, v in self.network.items()}
        d["gateway_link"] = asdict(self.gateway_link)
        d["synthetic"]["mem_bw_gbs"] = {
            str(k): v for k, v in self.synthetic.mem_bw_gbs.items()}
        d["composition"]["tiers"] = {str(k): v for k, v in self.composition.tiers.items()}
        d["churn"]["schedule"] = [asdict(f) for f in self.churn.schedule]
        d["churn"]["session_mean_s"] = {
            str(k): v for k, v in self.churn.session_mean_s.items()}
        d["churn"]["downtime_mean_s"] = {
            str(k): v for k, v in self.churn.downtime_mean_s.items()}
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    def with_(self, **kw) -> "Scenario":
        """A copy with fields replaced -- used to build sweep points."""
        return replace(self, **kw)


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------
def _int_keys(d: dict | None) -> dict[int, Any]:
    return {int(k): v for k, v in (d or {}).items()}


def from_dict(raw: dict) -> Scenario:
    raw = dict(raw)

    comp_raw = dict(raw.pop("composition", {}) or {})
    comp_raw["tiers"] = _int_keys(comp_raw.get("tiers"))
    composition = Composition(**comp_raw)

    network = {t: LinkProfile(**(v or {}))
               for t, v in _int_keys(raw.pop("network", {})).items()}

    gw_raw = raw.pop("gateway_link", None)
    gateway_link = (LinkProfile(**gw_raw) if gw_raw
                    else LinkProfile(delay_ms=1.0, rate_mbit=1000.0))

    churn_raw = dict(raw.pop("churn", {}) or {})
    churn_raw["schedule"] = tuple(
        FaultEvent(**f) for f in (churn_raw.get("schedule") or []))
    churn_raw["session_mean_s"] = _int_keys(churn_raw.get("session_mean_s"))
    churn_raw["downtime_mean_s"] = _int_keys(churn_raw.get("downtime_mean_s"))
    churn = Churn(**churn_raw)

    synth_raw = dict(raw.pop("synthetic", {}) or {})
    synth_raw["mem_bw_gbs"] = _int_keys(synth_raw.get("mem_bw_gbs"))

    scn = Scenario(
        composition=composition,
        network=network,
        gateway_link=gateway_link,
        churn=churn,
        workload=Workload(**(raw.pop("workload", {}) or {})),
        coordinator=CoordinatorCfg(**(raw.pop("coordinator", {}) or {})),
        retry=RetryCfg(**(raw.pop("retry", {}) or {})),
        synthetic=SyntheticCfg(**synth_raw),
        real=RealCfg(**(raw.pop("real", {}) or {})),
        **raw,
    )
    validate(scn)
    return scn


def load(path: str | Path) -> Scenario:
    import yaml  # host-only dependency
    with open(path, "r", encoding="utf-8") as fh:
        return from_dict(yaml.safe_load(fh) or {})


def validate(scn: Scenario) -> None:
    errs: list[str] = []
    if scn.mode not in MODES:
        errs.append(f"mode must be one of {MODES}")
    if scn.engine not in ENGINES:
        errs.append(f"engine must be one of {ENGINES}")
    if scn.placement not in PLACEMENTS:
        errs.append(f"placement must be one of {PLACEMENTS}")
    if scn.model_profile not in MODEL_PROFILES:
        errs.append(f"unknown model_profile {scn.model_profile!r}")
    if scn.workload.arrival not in ARRIVALS:
        errs.append(f"arrival must be one of {ARRIVALS}")
    if scn.churn.kind not in CHURN_KINDS:
        errs.append(f"churn.kind must be one of {CHURN_KINDS}")
    if scn.composition.depth < 1 or scn.composition.width < 1:
        errs.append("composition depth and width must be >= 1")
    try:
        scn.node_tiers()
    except ValueError as exc:
        errs.append(str(exc))
    for tier in list(scn.composition.tiers) + list(scn.network):
        if tier not in TIERS:
            errs.append(f"unknown tier {tier}")
    if scn.engine == "container" and scn.time_scale != 1.0:
        errs.append("container mode requires time_scale = 1.0: tc netem delays "
                    "are real milliseconds and cannot be dilated")
    if scn.mode == "real" and scn.engine == "virtual":
        errs.append("real mode needs containers; the virtual engine has no model")
    if scn.coordinator.death_after_ms <= scn.coordinator.heartbeat_ms:
        errs.append("death_after_ms must exceed heartbeat_ms")
    if scn.workload.arrival == "poisson" and scn.workload.request_rate <= 0:
        errs.append("poisson arrival needs request_rate > 0")
    if errs:
        raise ValueError(f"invalid scenario {scn.name!r}:\n  - " + "\n  - ".join(errs))
