"""Cost model from the research report (Section 5) and its device-tier reference table.

Four expressions govern the whole system. Everything the harness predicts, and
everything synthetic mode sleeps for, derives from these:

    T_token  ~=  (P_a*b + s*q) / B   +   D * (l + a/beta)         # per-token latency
    X_max    ~=  sum_i(B_i) / (P_a*b + s*q)                       # peak throughput
    X(G)     ~=  X_max * min(1, G/N)                              # utilisation ramp
    X_pipe    =  1 / max_k(t_k),  t_k = (W_k*b + KV_k)/B_k + l    # straggler gate

Units are SI throughout: bytes, bytes/s, seconds. Scenario files use friendly
units (GB/s, ms, Mbit/s) and are converted on load.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

GB = 1_000_000_000.0
MBIT = 1_000_000.0 / 8.0  # Mbit/s -> bytes/s


# ---------------------------------------------------------------------------
# Model profiles
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelProfile:
    """A model as the cost model sees it: bytes read per token, and payload size."""

    name: str
    total_params: float       # P
    active_params: float      # P_a  (== P for a dense model)
    bytes_per_param: float    # b    (2 = FP16, 0.5 = INT4, 4 = FP32)
    layers: int               # L
    hidden_dim: int           # d
    kv_bytes_per_elem: float  # b_kv
    note: str = ""

    @property
    def weight_read_bytes(self) -> float:
        """P_a * b -- weight bytes one token reads through the whole model."""
        return self.active_params * self.bytes_per_param

    @property
    def kv_bytes_per_token(self) -> float:
        """q = 2*L*d*b_kv -- KV bytes cached per context token."""
        return 2.0 * self.layers * self.hidden_dim * self.kv_bytes_per_elem

    @property
    def activation_bytes(self) -> float:
        """a ~= 2*d*b -- the hidden state crossing each hop."""
        return 2.0 * self.hidden_dim * self.bytes_per_param

    def bytes_per_token(self, context_len: int) -> float:
        """P_a*b + s*q -- the full per-token read, weights plus KV."""
        return self.weight_read_bytes + context_len * self.kv_bytes_per_token

    def kv_working_set(self, context_len: int, concurrency: int) -> float:
        """KV ~= 2*L*d*b_kv * s * G (report Section 7.1)."""
        return self.kv_bytes_per_token * context_len * concurrency

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(
            weight_read_bytes=self.weight_read_bytes,
            kv_bytes_per_token=self.kv_bytes_per_token,
            activation_bytes=self.activation_bytes,
        )
        return d


# The report's appendix model: 1T-param sparse MoE at INT4, 60B active.
# P_a*b = 30 GB. Used for report-scale runs in the virtual engine.
REPORT_MOE = ModelProfile(
    name="report_moe",
    total_params=1e12,
    active_params=60e9,
    bytes_per_param=0.5,
    layers=80,
    hidden_dim=8192,
    kv_bytes_per_elem=1.0,
    note="Appendix worked example: 1T MoE @ INT4, 60B active, P_a*b = 30 GB.",
)

# Scaled down so synthetic stage times land in the tens of milliseconds, the
# same order as the emulated per-hop delays (1-50 ms). This keeps the
# compute:network ratio realistic on a single host, where a 3 s/token stage
# would make every run network-trivial and painfully slow to gather data from.
POC_SMALL = ModelProfile(
    name="poc_small",
    total_params=50e9,
    active_params=5.4e9,
    bytes_per_param=0.5,
    layers=24,
    hidden_dim=2048,
    kv_bytes_per_elem=1.0,
    note="Container-scale MoE: P_a*b = 2.7 GB -> ~20 ms/stage on a tier-2 node at D=3.",
)

# The real model used in real mode, as the cost model sees it. Qwen2.5-0.5B is
# dense (P_a == P), 24 layers, hidden 896, run on CPU in float32 (b = 4).
QWEN_0_5B = ModelProfile(
    name="qwen_0_5b",
    total_params=494e6,
    # The 24 decoder layers only: 494M total minus the 136M tied
    # embedding/output matrix, which lives on the gateway with tokenisation and
    # the head, not on a stage worker. P_a must count what the *stages* read per
    # token, or predicted stage time is inflated by weights no stage holds.
    active_params=358e6,
    bytes_per_param=4.0,
    layers=24,
    hidden_dim=896,
    kv_bytes_per_elem=4.0,
    note="Qwen2.5-0.5B-Instruct, dense, FP32 on CPU. Stage-held P_a*b = 1.43 GB; "
         "the 544 MB tied embedding/head sits on the gateway.",
)

MODEL_PROFILES = {p.name: p for p in (REPORT_MOE, POC_SMALL, QWEN_0_5B)}


# ---------------------------------------------------------------------------
# Device tiers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TierSpec:
    """One device tier.

    `mem_bw` comes from the research report's reference table (Section 3) and is
    what the cost model uses. The `cpu_cores`, `mem_limit_mb`, `rate_mbit` and
    `delay_ms` fields are the single-host container emulation from the PoC
    design doc, which preserves tier *ratios*, not absolute device speeds.
    """

    tier: int
    stands_for: str
    mem_bw: float            # B_i, bytes/s -- report reference value
    availability: float      # p_i
    cpu_cores: float         # container CPU share
    mem_limit_mb: int        # container memory limit
    rate_mbit: float         # emulated link bandwidth
    delay_ms: float          # emulated base link delay
    session_mean_s: float    # mean uptime, random churn model
    downtime_mean_s: float   # mean downtime, random churn model


# Availability and session lengths: the report gives these only qualitatively
# ("low, bursty" .. "high"), so the numbers below are the harness's own
# defaults, chosen to span the range it describes. They are scenario-overridable
# and recorded in every run manifest, so results stay attributable.
TIERS = {
    3: TierSpec(3, "Gaming PC (GPU)",         600 * GB, 0.97, 2.0, 4096, 1000.0,  1.0, 3600.0,  60.0),
    2: TierSpec(2, "Laptop / idle desktop",    45 * GB, 0.90, 1.0, 2048,  500.0,  5.0, 1800.0, 120.0),
    1: TierSpec(1, "Single-board (Pi-class)",  10 * GB, 0.75, 0.5, 1024,  100.0, 10.0,  600.0, 180.0),
    0: TierSpec(0, "Microcontroller / IoT",     3 * GB, 0.40, 0.1,  256,   10.0, 50.0,  120.0, 180.0),
}


# ---------------------------------------------------------------------------
# The four governing expressions
# ---------------------------------------------------------------------------
def token_latency(profile: ModelProfile, mem_bw: float, depth: int,
                  link_delay_s: float, link_bw: float,
                  context_len: int = 0) -> float:
    """T_token ~= (P_a*b + s*q)/B + D*(l + a/beta).

    `mem_bw` is ONE device's bandwidth, not the pool's sum: this is the report's
    decisive point that adding devices never shrinks the compute term.
    """
    compute = profile.bytes_per_token(context_len) / mem_bw
    network = depth * (link_delay_s + profile.activation_bytes / link_bw)
    return compute + network


def x_max(profile: ModelProfile, mem_bws: list[float], context_len: int = 0) -> float:
    """X_max ~= sum(B_i) / (P_a*b + s*q) -- peak throughput, tokens/s."""
    return sum(mem_bws) / profile.bytes_per_token(context_len)


def x_of_g(peak: float, concurrency: int, n_devices: int) -> float:
    """X(G) ~= X_max * min(1, G/N) -- the utilisation ramp."""
    if n_devices <= 0:
        return 0.0
    return peak * min(1.0, concurrency / n_devices)


def stage_time(stage_weight_bytes: float, stage_kv_bytes: float,
               mem_bw: float, link_delay_s: float) -> float:
    """t_k = (W_k*b + KV_k)/B_k + l -- one stage's service time for one token."""
    return (stage_weight_bytes + stage_kv_bytes) / mem_bw + link_delay_s


def x_pipe(stage_times: list[float]) -> float:
    """X_pipe = 1/max_k(t_k) -- a pipeline runs no faster than its slowest stage."""
    worst = max(stage_times) if stage_times else 0.0
    return 1.0 / worst if worst > 0 else 0.0


def stage_availability(p: float, width: int) -> float:
    """1 - (1-p)^W -- a stage is usable if any one of its W replicas is up."""
    return 1.0 - (1.0 - p) ** width


def concurrency_to_fill(n_devices: int) -> int:
    """G ~= N streams are needed to fill the pool (report Section 7.3)."""
    return n_devices
