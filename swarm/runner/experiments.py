"""The planned experiments, as scenario generators.

Each experiment is a base scenario plus a list of sweep points. Every point is a
complete, independently repeatable scenario with its own run directory; the
driver then concatenates their one-row `runs` tables into a single tidy dataset
per experiment.

Only the stated axis varies within an experiment. Where a value had to be chosen
rather than swept, the reason is given, because a defaulted constant that
happens to bind the result is the easiest way to draw a wrong conclusion from a
sweep.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

from ..core.scenario import (Churn, Composition, CoordinatorCfg, FaultEvent,
                             LinkProfile, RetryCfg, Scenario, SyntheticCfg,
                             Workload)

# A hop timeout must be set relative to a token's own latency. The default of
# 5000 ms used during bring-up turned out to be ~65 token-times, which made a
# single node loss stall a stream for a third of the run and dominate every
# throughput figure. 500 ms is ~6 token-times for the poc_small profile: long
# enough not to abort healthy slow hops, short enough that recovery is governed
# by detection rather than by client-side waiting.
HOP_TIMEOUT_MS = 500.0

# Predictions use the BARE cost model: both overhead corrections are zero.
#
# An earlier version fitted them from measured hops and folded them into the
# predictions, which made the agreement look better (a latency ratio of 1.017
# rather than ~1.07). Two things ruled that out.
#
# First it was circular: the stage worker slept `bare + configured overhead`
# while the fit measured the residual against `bare`, so each fit re-absorbed
# the sleep the previous one had added, and successive fits diverged
# (1.2 -> 2.2 -> 3.3 ms). The worker now sleeps the bare time only.
#
# Second, once the circularity was removed, the residual turned out not to be a
# constant at all: 1.199 ms per stage at D=3/G=4 against 0.219 ms at D=6/G=1,
# because asyncio's sleep wakeups are delayed by however busy the event loop
# is. A single fitted figure would be a fudge factor that flattered the model
# while hiding its real accuracy.
#
# So these stay at zero, `pred_t_token_s` is the cost model itself, and the
# harness's own cost is reported as the measured/predicted ratio rather than
# subtracted out of it. The fit is still run (tools/calibrate.py) and still
# recorded, as a measurement of the harness rather than a parameter of it.
STAGE_OVERHEAD_MS = 0.0
HOP_OVERHEAD_MS = 0.0


def _synth(**kw) -> SyntheticCfg:
    """SyntheticCfg carrying the fitted overheads unless overridden."""
    kw.setdefault("stage_overhead_ms", STAGE_OVERHEAD_MS)
    kw.setdefault("hop_overhead_ms", HOP_OVERHEAD_MS)
    kw.setdefault("servers_per_node", 1)
    return SyntheticCfg(**kw)


@dataclass
class Experiment:
    key: str
    title: str
    question: str
    build: Callable[[], Iterator[Scenario]]


def _base(name: str, experiment: str, **kw) -> Scenario:
    """A tier-2 homogeneous base. Tier 2 is the mid tier, so neither the memory
    bandwidth nor the link is at an extreme of the emulated range."""
    defaults = dict(
        name=name,
        experiment=experiment,
        seed=42,
        mode="synthetic",
        engine="container",
        duration_s=60.0,
        warmup_s=5.0,
        model_profile="poc_small",
        placement="capability_proportional",
        composition=Composition(depth=3, width=2, tiers={2: 6}),
        network={2: LinkProfile(delay_ms=5.0, rate_mbit=500.0)},
        workload=Workload(concurrency=4, prompt_tokens=32, gen_tokens=8),
        coordinator=CoordinatorCfg(heartbeat_ms=500.0, death_after_ms=2000.0,
                                   sample_interval_ms=500.0),
        retry=RetryCfg(max_attempts=3, hop_timeout_ms=HOP_TIMEOUT_MS,
                       backoff_ms=50.0),
        synthetic=_synth(),
        churn=Churn(kind="none"),
    )
    defaults.update(kw)
    return Scenario(**defaults)


# ---------------------------------------------------------------------------
# 1. Concurrency sweep
# ---------------------------------------------------------------------------
CONCURRENCIES = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)


def exp1_concurrency() -> Iterator[Scenario]:
    """Vary G on a fixed composition; compare against the utilisation curve.

    N is held at 6 (D=3, W=2) so the sweep crosses G = N at a point well inside
    the range, which is where the report's ramp predicts utilisation to
    saturate. No churn: this experiment measures the clean ramp, and a stalled
    stream would show up as lost throughput that has nothing to do with G.
    """
    # A saturated closed-loop system's latency grows as G/X, so the top of this
    # sweep reaches a few hundred milliseconds per token by construction. With
    # the 500 ms default, the first run of this sweep had tokens at G=32 time out
    # and retry, and the retries added load that collapsed throughput to a fifth
    # of the plateau -- a real failure mode, but one that measures the timeout
    # rather than the utilisation curve this experiment is about. 2000 ms leaves
    # roughly 6x headroom over the worst latency the sweep should reach.
    # The hop timeout is swept deliberately in exp4 instead.
    timeout_ms = 2000.0
    for g in CONCURRENCIES:
        yield _base(
            f"exp1_concurrency_G{g:02d}", "1_concurrency",
            workload=Workload(concurrency=g, prompt_tokens=32, gen_tokens=8),
            retry=RetryCfg(max_attempts=3, hop_timeout_ms=timeout_ms,
                           backoff_ms=50.0),
            sweep_point={"concurrency_G": g, "hop_timeout_ms": timeout_ms},
        )


# ---------------------------------------------------------------------------
# 2. Delay sweep
# ---------------------------------------------------------------------------
DEPTHS = (2, 3, 4, 6)
DELAYS_MS = (1.0, 5.0, 25.0, 50.0)


def exp2_delay() -> Iterator[Scenario]:
    """Vary per-hop delay and depth to measure how hop latency accumulates.

    W=1 and G=1. Replicas would let queueing confound the delay term, and any
    concurrency above 1 adds queue wait to the measured per-token latency, which
    is precisely the quantity being decomposed here. The model profile's total
    compute is fixed and split across D stages, so compute stays constant at
    ~60 ms while the network term grows with D -- which isolates D*(l + a/beta).
    """
    for depth in DEPTHS:
        for delay in DELAYS_MS:
            yield _base(
                f"exp2_delay_D{depth}_l{int(delay):02d}ms", "2_delay",
                duration_s=45.0,
                composition=Composition(depth=depth, width=1,
                                        tiers={2: depth}),
                network={2: LinkProfile(delay_ms=delay, rate_mbit=500.0)},
                workload=Workload(concurrency=1, prompt_tokens=32, gen_tokens=8),
                sweep_point={"depth_D": depth, "delay_ms": delay},
            )


# ---------------------------------------------------------------------------
# 4. Disconnection and failover
# ---------------------------------------------------------------------------
WIDTHS = (1, 2, 3)
HEARTBEATS_MS = (250.0, 500.0, 1000.0)
TIMEOUTS_MS = (250.0, 500.0, 1000.0, 2000.0, 5000.0)
KILL_AT_S = 20.0


def _failover(name: str, width: int, heartbeat_ms: float, timeout_ms: float,
              sweep: dict) -> Scenario:
    # The victim is always a middle-stage replica: stage 0 is reachable directly
    # from the gateway and the last stage reports to it, so a middle stage is
    # the case where an upstream worker has to discover the failure.
    victim = "n-s1-r0"
    return _base(
        name, "4_failover",
        duration_s=50.0,
        composition=Composition(depth=3, width=width, tiers={2: 3 * width}),
        workload=Workload(concurrency=6, prompt_tokens=32, gen_tokens=8),
        coordinator=CoordinatorCfg(heartbeat_ms=heartbeat_ms,
                                   death_after_ms=4 * heartbeat_ms,
                                   sample_interval_ms=500.0),
        retry=RetryCfg(max_attempts=3, hop_timeout_ms=timeout_ms,
                       backoff_ms=50.0),
        churn=Churn(
            kind="schedule",
            # W=1 means killing the victim takes its stage to zero replicas, so
            # the guard must be off for that arm or the fault would be silently
            # suppressed and the run would measure nothing. The W=1 arm is the
            # control: it shows what a loss costs with no replica to fail over
            # to.
            protect_last_replica=(width > 1),
            schedule=(FaultEvent(at_s=KILL_AT_S, action="kill", target=victim),),
        ),
        sweep_point=sweep,
    )


def exp4_failover() -> Iterator[Scenario]:
    """Detection, recovery and failed requests vs replica count and heartbeat.

    Two sub-sweeps. The first is the one the design doc asks for: replica count
    crossed with heartbeat interval. The second sweeps the hop timeout at fixed
    W and heartbeat, added because bring-up showed recovery time tracking the
    timeout rather than the detection threshold -- so the timeout is a
    first-order variable here, not a constant to be defaulted.
    """
    for width in WIDTHS:
        for hb in HEARTBEATS_MS:
            yield _failover(
                f"exp4_failover_W{width}_hb{int(hb):04d}",
                width, hb, HOP_TIMEOUT_MS,
                {"width_W": width, "heartbeat_ms": hb,
                 "hop_timeout_ms": HOP_TIMEOUT_MS, "arm": "W_x_heartbeat"},
            )
    for timeout in TIMEOUTS_MS:
        if timeout == HOP_TIMEOUT_MS:
            continue        # already covered by the arm above at W=2, hb=500
        yield _failover(
            f"exp4_timeout_t{int(timeout):04d}", 2, 500.0, timeout,
            {"width_W": 2, "heartbeat_ms": 500.0,
             "hop_timeout_ms": timeout, "arm": "hop_timeout"},
        )


# ---------------------------------------------------------------------------
# 7. Mode check
# ---------------------------------------------------------------------------
MODE_CHECK_G = (1, 2, 4)


def exp7_mode_check() -> Iterator[Scenario]:
    """Run the same small scenario in real and synthetic mode and compare.

    This is the experiment that licenses every synthetic result: if synthetic
    timings do not track real ones here, nothing measured in synthetic mode at
    larger scale can be trusted.

    Both arms use the qwen_0_5b profile so the cost model describes the same
    model the real arm actually runs. The synthetic arm's emulated memory
    bandwidth is left at the tier-2 reference value here; tools/calibrate.py
    fits it, and the fitted overheads, from the real arm's measurements and
    writes a second synthetic arm using them.

    busy_compute is on for the synthetic arm: the real arm genuinely occupies
    its CPU share, so a comparison against a synthetic arm that merely sleeps
    would differ by CPU contention rather than by timing fidelity.
    """
    # Real mode needs a far larger hop timeout than the synthetic experiments.
    # The first token of a request is a prefill: it pushes all 32 prompt
    # positions through each stage's 8 layers, roughly 32x a decode step's work.
    # At the 500 ms default every request died on its first token, with hops
    # flowing and no errors logged -- the pipeline was working and the client
    # was giving up on it. Both arms use the same value so the comparison is
    # not confounded by the timeout.
    timeout_ms = 30000.0
    for g in MODE_CHECK_G:
        common = dict(
            duration_s=90.0,       # real tokens are slow; 90 s to get a sample
            warmup_s=10.0,         # first-call torch overhead plus a warm cache
            model_profile="qwen_0_5b",
            composition=Composition(depth=3, width=1, tiers={2: 3}),
            network={2: LinkProfile(delay_ms=5.0, rate_mbit=500.0)},
            workload=Workload(concurrency=g, prompt_tokens=32, gen_tokens=8),
            retry=RetryCfg(max_attempts=3, hop_timeout_ms=timeout_ms,
                           backoff_ms=50.0),
        )
        yield _base(f"exp7_real_G{g}", "7_mode_check", mode="real",
                    sweep_point={"concurrency_G": g, "arm": "real"}, **common)
        yield _base(f"exp7_synth_G{g}", "7_mode_check", mode="synthetic",
                    synthetic=_synth(busy_compute=True),
                    sweep_point={"concurrency_G": g, "arm": "synthetic"},
                    **common)


# ---------------------------------------------------------------------------
# 1s. Concurrency sweep at report scale (virtual engine only)
# ---------------------------------------------------------------------------
SCALE_CONCURRENCIES = (1, 10, 50, 100, 250, 500, 1000, 1500, 2000)


def exp1_scale_concurrency() -> Iterator[Scenario]:
    """The concurrency ramp at the research report's own appendix scale.

    This is deliberately a reconstruction of worked example A: 1000 Pi-class
    (tier 1) nodes serving a 1T-parameter MoE at INT4 with 60B active per token,
    cut into D = 50 stages. The report derives, for that configuration:

        per-token latency   30 GB / 10 GB/s + 50 x 1 ms  ~=  3.05 s
        peak throughput     10,000 GB/s / 30 GB          ~=  333 tokens/s
        concurrency to fill G ~= N = 1000 streams

    so the engine's output can be checked against numbers derived independently
    of it. N = 1000 is roughly 16x what this host can run as containers, which
    is the whole reason the virtual engine exists.

    Virtual-only, and only as trustworthy as tools/compare_engines.py says the
    engine is on the axes involved. Per-hop logging is off: at D = 50 it would
    be 51 rows per token and millions of rows per run.
    """
    for g in SCALE_CONCURRENCIES:
        yield _base(
            f"exp1s_scale_G{g:04d}", "1s_concurrency_scale",
            engine="virtual",
            duration_s=180.0,
            warmup_s=20.0,          # a 3 s/token pipeline fills slowly
            model_profile="report_moe",
            composition=Composition(depth=50, width=20, tiers={1: 1000}),
            network={1: LinkProfile(delay_ms=1.0, rate_mbit=1000.0)},
            workload=Workload(concurrency=g, prompt_tokens=0, gen_tokens=4),
            coordinator=CoordinatorCfg(heartbeat_ms=1000.0,
                                       death_after_ms=4000.0,
                                       sample_interval_ms=10000.0),
            retry=RetryCfg(max_attempts=3, hop_timeout_ms=20000.0,
                           backoff_ms=50.0),
            log_hops=False,
            sweep_point={"concurrency_G": g, "n_nodes": 1000},
        )


EXPERIMENTS = {
    "exp1": Experiment("exp1", "Concurrency sweep",
                       "How does throughput track the utilisation curve as G rises?",
                       exp1_concurrency),
    "exp2": Experiment("exp2", "Delay sweep",
                       "How does per-hop latency accumulate with depth and delay?",
                       exp2_delay),
    "exp4": Experiment("exp4", "Disconnection and failover",
                       "How fast is a loss detected and routed around, and at "
                       "what cost in failed requests?",
                       exp4_failover),
    "exp7": Experiment("exp7", "Mode check",
                       "Do synthetic timings track real ones?",
                       exp7_mode_check),
    "exp1s": Experiment("exp1s", "Concurrency sweep at report scale",
                        "Does the ramp hold at N=1000, and where does it break?",
                        exp1_scale_concurrency),
}
