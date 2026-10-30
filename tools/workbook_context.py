"""Reference content for the results workbook: experiment design, the cost
model, the tier table, and provenance.

Data only, no rendering, so it can be read on its own and reused.
"""

from __future__ import annotations

# (sheet, experiment-in-data, title, question, varied, held constant, runs,
#  engine, mode, design notes)
DESIGN = [
    ("exp1 Concurrency", "1_concurrency", "Concurrency sweep",
     "How does throughput track the utilisation curve as concurrency rises, "
     "and where does the system stop scaling?",
     "G: 1, 2, 3, 4, 6, 8, 12, 16, 24, 32",
     "D=3, W=2, N=6, all tier 2; per-hop delay 5 ms; 500 Mbit/s; prompt 32, "
     "gen 8; no churn; hop timeout 2000 ms",
     10, "container", "synthetic",
     "N=6 so the sweep crosses G=N well inside the range. No churn: a stalled "
     "stream would show up as lost throughput that has nothing to do with G. "
     "The hop timeout is deliberately generous because a saturated "
     "closed-loop system's latency grows as G/X; a shorter one caused tokens "
     "to time out, retry, add load and collapse throughput to a fifth of the "
     "plateau."),
    ("exp2 Delay", "2_delay", "Delay sweep",
     "How does per-hop latency accumulate with pipeline depth and link delay?",
     "D: 2, 3, 4, 6  crossed with  per-hop delay: 1, 5, 25, 50 ms",
     "W=1, G=1, all tier 2; 500 Mbit/s; prompt 32, gen 8; no churn",
     16, "container", "synthetic",
     "W=1 and G=1 deliberately: replicas would let queueing confound the hop "
     "term, and any concurrency above 1 adds queue wait to the very quantity "
     "being decomposed. Total compute is fixed and split across D stages, so "
     "compute stays about 60 ms while the network term grows with D. Only the "
     "tier-2 link is swept, so the expected slope is D, not D+1: the "
     "gateway's entry hop is held fixed and lands in the intercept."),
    ("exp4 Failover", "4_failover", "Disconnection and failover",
     "How fast is a node loss detected and routed around, and at what cost in "
     "failed requests?",
     "Arm W_x_heartbeat: W in {1,2,3} crossed with heartbeat in "
     "{250,500,1000} ms. Arm hop_timeout: timeout in {250,1000,2000,5000} ms "
     "at W=2, heartbeat 500 ms",
     "D=3, G=6, all tier 2; delay 5 ms; death_after = 4x heartbeat; one kill "
     "of a middle-stage replica at t=20 s; duration 50 s",
     13, "container", "synthetic",
     "The victim is always a middle stage: stage 0 is reached directly from "
     "the gateway and the last stage reports back to it, so a middle stage is "
     "the case where an upstream worker has to discover the failure. The W=1 "
     "arm is the control and has the last-replica guard disabled, or the "
     "fault would be silently suppressed and the run would measure nothing. "
     "The second arm exists because bring-up showed recovery time tracking "
     "the client timeout rather than the detection threshold."),
    ("exp7 Mode check", "7_mode_check", "Mode check (real vs synthetic)",
     "Do synthetic timings track real ones? This is the experiment that "
     "licenses every other synthetic result.",
     "G: 1, 2, 4  crossed with  arm: real, synthetic on nominal bandwidth, "
     "synthetic on fitted B_eff",
     "D=3, W=1, all tier 2; Qwen2.5-0.5B split 8/8/8 layers; CPU float32; "
     "greedy decoding; prompt 32, gen 8; hop timeout 30000 ms; duration 90 s",
     9, "container", "real and synthetic",
     "Three phases in order: real runs first, then the effective memory "
     "bandwidth fitted from them, then synthetic arms on both the nominal and "
     "the fitted bandwidth. Comparing a real arm against a synthetic arm fed "
     "the tier table's nominal figure would measure that mismatch rather than "
     "synthetic mode's fidelity. The large timeout is required because the "
     "first token of each request is a prefill, roughly 32x a decode step."),
    ("exp1s Report scale", "1s_concurrency_scale",
     "Concurrency at research-report scale",
     "Does the utilisation ramp hold at the research report's own scale, and "
     "where does it break?",
     "G: 1, 10, 50, 100, 250, 500, 1000, 1500, 2000",
     "D=50, W=20, N=1000, all tier 1; delay 1 ms; report_moe (1T MoE, 60B "
     "active, INT4); gen 4; no churn; per-hop logging off; duration 180 s",
     9, "virtual", "synthetic",
     "A reconstruction of the research report's appendix example A, whose "
     "figures were derived independently of this code, so agreement is a real "
     "check rather than a tautology. N=1000 is about 16x what this host can "
     "run as containers, so this is virtual-engine only and is trustworthy "
     "only on the axes the engine comparison validates. Per-hop logging is "
     "off: at D=50 it would be 51 rows per token."),
]

# (quantity, expression, meaning and what was measured about it)
COST_MODEL = [
    ("T_token", "(P_a*b + s*q)/B + D*(l + a/beta)",
     "Per-token latency. The first term uses ONE device's bandwidth B, not "
     "the sum over devices: adding devices never shrinks it. Measured here: "
     "the D*(l + a/beta) term is confirmed to within 0.2% (experiment 2)."),
    ("X_max", "sum_i(B_i) / (P_a*b + s*q)",
     "Peak throughput: aggregate memory bandwidth divided by bytes read per "
     "token. Grows linearly with device count. Measured saturation reaches "
     "0.86-0.98 of it depending on scale."),
    ("X(G)", "X_max * min(1, G/N)",
     "The utilisation ramp. Measured here to be an UPPER BOUND, not a "
     "predictor: it assumes a stream computes continuously, so its slack is "
     "the network's share of per-token time. 32% optimistic at container "
     "scale (21% network), 1-2% at report scale (1.6% network)."),
    ("closed-loop rate", "min( G/T_token , W/max_k(compute_k) )",
     "Not in the research report; added here. The smaller of a delay bound "
     "and a capacity bound. Tracks measurement at 0.88 +/- 0.03 across a 32x "
     "range of G, where the ramp is 32% out."),
    ("X_pipe", "1 / max_k(t_k),  t_k = (W_k*b + KV_k)/B_k + l",
     "A pipeline runs no faster than its slowest stage. This is where "
     "heterogeneity bites: equal sharding across tiers differing 100x in "
     "bandwidth lets the weakest node set the clock for everyone."),
    ("stage availability", "1 - (1-p)^W",
     "A stage is usable while any one of its W replicas is reachable. "
     "Measured as a cliff rather than a slope: at W=1 a single kill failed "
     "over a thousand requests and never recovered; at W=2, zero to three."),
    ("KV working set", "2*L*d*b_kv * s * G",
     "Scales with context length and concurrency, and competes with weights "
     "for the same scarce per-node memory. Not stressed by these experiments "
     "(short prompts), so no finding is offered on it."),
]

SYMBOLS = [
    ("N", "count", "Devices in the pool"),
    ("D, W", "count", "Pipeline depth (stages) and replica width; N = D x W"),
    ("B_i", "bytes/s", "Device i's memory bandwidth - the governing resource, "
                       "because decode is memory-bound, not compute-bound"),
    ("P, P_a", "count", "Total and per-token active parameters "
                        "(P_a much smaller than P for a sparse MoE)"),
    ("b", "bytes", "Bytes per parameter: 2 = FP16, 0.5 = INT4, 4 = FP32"),
    ("L, d", "count", "Model layers and hidden dimension"),
    ("s, q", "tokens, bytes", "Context length and KV bytes per cached token"),
    ("a", "bytes", "Activation payload crossing each hop, about 2*d*b"),
    ("l, beta", "seconds, bytes/s", "Per-hop link latency and link bandwidth"),
    ("G", "count", "Concurrent independent token streams in flight"),
    ("p_i", "fraction", "Device availability: fraction of time reachable"),
    ("t_k", "seconds", "Service time of stage k for one token"),
]

# (profile, P, P_a, b, L, d, P_a*b in GB, notes)
PROFILES = [
    ("report_moe", "1e12", "60e9", 0.5, 80, 8192, 30.0,
     "The research report's appendix model: a 1T-parameter sparse MoE at "
     "INT4. Used for the report-scale virtual runs."),
    ("poc_small", "50e9", "5.4e9", 0.5, 24, 2048, 2.7,
     "Scaled so synthetic stage times land in the tens of milliseconds, the "
     "same order as the emulated hop delays (1-50 ms). The report's own model "
     "would give about 3 s per stage, making every container run "
     "network-trivial and absurdly slow to gather data from."),
    ("qwen_0_5b", "494e6", "358e6", 4.0, 24, 896, 1.432,
     "Qwen2.5-0.5B-Instruct, used in real mode. active_params counts the 24 "
     "decoder layers only: the 136M tied embedding/output matrix sits on the "
     "gateway, not on a stage worker. Verified against the downloaded "
     "weights (1.432 GB in layers, 544 MB outside)."),
]

# (tier, stands for, B GB/s, availability, cpu, mem MB, link Mbit/s, delay ms)
TIERS_TABLE = [
    (3, "Gaming PC (GPU)", 600, 0.97, 2.0, 4096, 1000, 1),
    (2, "Laptop / idle desktop (CPU)", 45, 0.90, 1.0, 2048, 500, 5),
    (1, "Single-board computer (Pi-class)", 10, 0.75, 0.5, 1024, 100, 10),
    (0, "Microcontroller / IoT node", 3, 0.40, 0.1, 256, 10, 50),
]

TIER_NOTE = (
    "Memory bandwidth and availability drive the cost model; the CPU, memory, "
    "link and delay columns are the single-host container emulation. These "
    "preserve the tiers' RATIOS, not absolute device speeds. Experiment 7 "
    "measures a tier-2 container's real effective bandwidth as 6.9 GB/s "
    "against the 45 GB/s nominal - 6.5x slower, and closer to the tier-1 "
    "reference than to the tier-2 one it was configured as. Availability and "
    "churn session lengths are this harness's own defaults: the research "
    "report gives them only qualitatively (\"low, bursty\" to \"high\")."
)

METHOD = [
    ("Method", "run ordering",
     "Sequential. Two concurrent runs would each measure the other's CPU and "
     "memory-bandwidth interference."),
    ("Method", "clock",
     "All containers share the host's CLOCK_REALTIME and are given one "
     "RUN_EPOCH, so one-way hop times are measured directly with no clock "
     "synchronisation."),
    ("Method", "network shaping",
     "tc netem (delay, jitter, loss) nested under tbf (rate cap), applied by "
     "each container to its own egress at startup. netem shapes EGRESS, so a "
     "hop A->B costs A's configured delay."),
    ("Method", "activation flow",
     "Forward-only: gateway -> s0 -> ... -> s(D-1) -> gateway, giving D+1 "
     "hops. An earlier nested-HTTP version charged 2D hop delays, which would "
     "have doubled the delay sweep's slope."),
    ("Method", "real model",
     "Qwen2.5-0.5B-Instruct, fetched once and bind-mounted read-only, greedy "
     "decoding so runs are reproducible and output is checkable."),
    ("Method", "steady state",
     "Statistics exclude a warm-up interval and, where noted, fault windows. "
     "Nothing is deleted: rows carry in_steady_state and fault_window labels."),
]

LIMITS = [
    "ONE HOST. Every container shares its CPU and memory bandwidth. The cgroup "
    "limits reduce but do not eliminate the interference.",
    "TIERS PRESERVE RATIOS, NOT ABSOLUTE SPEEDS. Nothing here says what a "
    "Raspberry Pi would actually do. Experiment 7 measures the gap for tier 2 "
    "as 6.5x.",
    "SYNTHETIC MODE MODELS NEITHER CPU CONTENTION NOR PREFILL COST. It "
    "over-predicts throughput by 1.6-2.2x even after bandwidth calibration, "
    "and its latency error grows with concurrency: 3.9% at G=1, 31% at G=4. "
    "Synthetic results speak to network and churn effects, not to a real "
    "deployment's absolute throughput.",
    "VIRTUAL-ENGINE RESULTS are valid for per-token latency, stage compute, "
    "network time and AGGREGATE queueing (agreeing with containers to a few "
    "per cent) and over-predict throughput by 6-8%. They are NOT valid for "
    "queue-wait distribution or tail latency: deterministic service "
    "concentrates all waiting in a third of hops where real jitter smears it.",
    "PREDICTIONS CARRY NO FITTED CORRECTION. The measured/predicted ratio is "
    "the harness's own cost, reported rather than subtracted out. An earlier "
    "fitted correction was removed after it proved circular and then "
    "load-dependent.",
    "tc IS NOT A HOME NETWORK. It models delay and loss well, not the full "
    "behaviour of real links. The virtual engine's loss model charges one "
    "round trip per lost segment, which is TCP's best case, so its loss "
    "results are a lower bound on the real penalty.",
    "NO INTEGRITY, SECURITY OR TRUST MODELLING. All containers are trusted. "
    "The research report's verification questions are explicitly out of scope "
    "for this harness.",
]
