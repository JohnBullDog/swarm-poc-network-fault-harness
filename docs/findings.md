# Findings

What the harness measured, what it says about the cost model, and what it does
not support. Every number here comes from a tidy table in `datasets/`, and
every claim names the table it came from.

Results are grouped as: how the harness was calibrated, what the experiments
found, and what the harness's own bring-up revealed — that last group matters
because four of the five problems found during bring-up would each have
produced a plausible-looking but wrong result.

---

## 0. Calibration

### Predictions are the bare cost model, and why

The design document calls for synthetic stage time to be bytes read over
emulated bandwidth "plus measured overhead", and that is what was built first:
`tools/calibrate.py` fitted a per-stage and per-hop residual, and both the
worker's sleep and the prediction added them. It appeared to work very well —
predicted per-token latency moved from 76.2 ms to 82.9 ms against a measured
p50 of 84.2 ms, a ratio of **1.017**.

That number was an artifact. The worker slept `bare + configured overhead`
while the fit measured the residual against `bare`, so every fit re-absorbed
the sleep that the previous fit had introduced. Refitting on the next round
returned 2.226 ms where 1.204 had been configured; a third round would have
returned about 3.3. The loop diverges by construction.

What exposed it was the delay sweep: measured/predicted rose with depth (1.050
at D=2 to 1.112 at D=6) while the slope against per-hop delay stayed correct to
0.5%. A correct network term with a growing additive error points at a
per-stage constant, not at the hop model.

With the worker changed to sleep the bare time, the residual turned out not to
be a host constant at all:

| configuration | bare stage compute | measured residual |
|---|---|---|
| D=3, G=4 | 20.0 ms | 1.199 ms |
| D=6, G=1 | 10.0 ms | 0.219 ms |

`asyncio` sleep wakeups are delayed by however busy the event loop is, so the
overhead tracks load rather than hardware. A single fitted figure would have
been a fudge factor: it would improve every reported ratio while telling the
reader nothing.

**So predictions here are the cost model itself, with no fitted correction, and
the harness's own cost appears as the measured/predicted ratio.** The fit is
still computed and recorded in `datasets/analysis/calibration.json` as a measurement of
the harness, not a parameter of it.

The general lesson, which applies well beyond this harness: a correction fitted
from the same quantity it then modifies will always improve the fit, and will
always be meaningless.

### Host and build

| | |
|---|---|
| Host | Windows 11, 16 cores, 16 GB RAM; Docker Desktop on WSL2 (kernel 6.18) |
| Container memory ceiling | 11 GB (raised from the 8 GB WSL2 default via `.wslconfig`) |
| Practical container node count | ~60; the research report's examples are at N=1000 and N=2750 |
| `tc` support | `sch_netem`, `sch_tbf`, `sch_htb` all present and verified working inside containers |

The netem path was verified before anything was built on it: a configured
100 ms delay with 10 ms jitter measured 93–105 ms (mean 97.6 ms) against a
0.34 ms unshaped baseline, and `tbf` rate-capping nests under it correctly.

---

## 1. Concurrency sweep

D=3, W=2, N=6 tier-2 nodes, `poc_small`, no churn. `X_max` = 99.9 tok/s.
Source: `datasets/summary/exp1_runs.csv`.

| G | measured X | report's ramp | closed-loop bound | X/ramp | X/closed | p50 | busy | failed |
|---|---|---|---|---|---|---|---|---|
| 1 | 11.23 | 16.6 | 13.1 | 0.674 | 0.855 | 84.5 ms | 0.116 | 0 |
| 2 | 22.56 | 33.3 | 26.3 | 0.678 | 0.860 | 84.2 ms | 0.234 | 0 |
| 4 | 44.94 | 66.6 | 52.5 | 0.675 | 0.856 | 84.1 ms | 0.469 | 0 |
| 6 | 67.82 | 99.9 | 78.8 | 0.679 | 0.861 | 83.9 ms | 0.700 | 0 |
| 8 | 88.87 | 99.9 | 99.9 | 0.890 | 0.890 | 85.0 ms | 0.911 | 0 |
| 12 | 89.55 | 99.9 | 99.9 | 0.897 | 0.897 | 127.9 ms | 0.916 | 0 |
| 16 | 92.21 | 99.9 | 99.9 | 0.923 | 0.923 | 169.6 ms | 0.933 | 0 |
| 24 | 89.88 | 99.9 | 99.9 | 0.900 | 0.900 | 255.3 ms | 0.926 | 0 |
| 32 | 90.16 | 99.9 | 99.9 | 0.903 | 0.903 | 339.1 ms | 0.933 | 0 |

**The report's utilisation ramp is an upper bound, not a predictor.**
`X_max · min(1, G/N)` overshoots measurement by 32% throughout the linear
region (X/ramp = 0.674–0.679, remarkably steady) and by about 10% at the
plateau. It assumes a stream keeps a node busy continuously, but a stream is
idle while its activation is on the wire.

**The closed-loop bound tracks measurement at 0.88 ± 0.03** across a 32× range
of G. The achievable rate is `min(G/T_token, W/max_k compute_k)`: the delay
bound below the knee, the capacity bound above it.

**Two regimes, with the knee where predicted.** Below G=8 throughput is linear
in G and latency is flat at ~84 ms — adding streams adds work with no
contention. From G=8 throughput is pinned at ~90 tok/s and latency grows
linearly (128 → 339 ms) while `busy_frac` saturates at 0.91–0.93. Nothing
degrades; streams simply queue.

**Saturated efficiency is 0.89–0.92 of `X_max`**, inside the report's predicted
"best case ~80–95%" band, under exactly the conditions it names for that band
(capability-proportional sharding, G ≥ N, stable nodes, local links).

Zero failed requests and zero retries at every point.

The flat **1.10–1.12** latency ratio below the knee is the harness's own cost
over the bare cost model (section 0), not a model error. Above the knee the
ratio is not meaningful: the prediction is unloaded latency, while the
measurement includes queueing that Little's law accounts for exactly.

---

## 2. Delay sweep

D ∈ {2,3,4,6}, W=1, G=1, per-hop delay ℓ ∈ {1,5,25,50} ms. 16 points.
Source: `datasets/summary/exp2_runs.csv`.

Latency against per-hop delay is linear with slope **exactly D**:

| D | measured slope | predicted | error | R² |
|---|---|---|---|---|
| 2 | 2.0013 | 2 | +0.07% | 1.000000 |
| 3 | 3.0039 | 3 | +0.13% | 1.000000 |
| 4 | 3.9930 | 4 | −0.18% | 0.999991 |
| 6 | 6.0113 | 6 | +0.19% | 0.999998 |

**This is the report's `D·(ℓ + a/β)` term confirmed to within 0.2%**, with
essentially perfect linearity over a 50× range of ℓ.

The slope is D rather than D+1 because only the tier-2 link is swept: tier-2
nodes send D of the path's D+1 hops (each stage forwards once, the last to the
gateway), while the gateway's entry hop is held fixed and contributes to the
intercept instead.

The practical reading is the report's: **deep pipelines over slow links are the
combination to avoid.** At ℓ=1 ms, going from D=2 to D=6 costs 17.5 ms
(71→89 ms); at ℓ=50 ms it costs 213 ms (170→383 ms). Depth is nearly free on a
LAN and dominant on WAN-like links.

Absolute latency runs 1.03–1.16× the bare model — the harness's cost, which
grows mildly with D because each stage adds event-loop work.

---

## 4. Disconnection and failover

D=3, G=6, one `kill` of a middle-stage replica at t=20 s. `death_after` =
4 × heartbeat. Sources: `datasets/summary/exp4_runs.csv`, `datasets/detail/exp4_recovery.csv`.

### Replica count × heartbeat interval

| W | heartbeat | detection | recovery | failed | ok |
|---|---|---|---|---|---|
| 1 | 250 ms | 0.725 s | **never** | **1167** | 108 |
| 1 | 500 ms | 1.566 s | **never** | **1136** | 108 |
| 1 | 1000 ms | 3.513 s | **never** | **1069** | 108 |
| 2 | 250 ms | 0.649 s | 0.657 s | 0 | 330 |
| 2 | 500 ms | 1.518 s | 1.534 s | 3 | 325 |
| 2 | 1000 ms | 3.012 s | 3.095 s | 1 | 318 |
| 3 | 250 ms | 0.664 s | 0.715 s | 0 | 416 |
| 3 | 500 ms | 1.250 s | 1.292 s | 0 | 413 |
| 3 | 1000 ms | 2.835 s | 2.922 s | 1 | 400 |

**Detection time is set by the heartbeat interval, and by nothing else.** It
scales with it almost exactly (0.65–0.73 s at 250 ms, 1.25–1.57 s at 500 ms,
2.83–3.51 s at 1000 ms) and is independent of W.

**W=1 is categorically different, not merely worse.** With no surviving replica
the stage is gone for the rest of the run: recovery never occurs and over a
thousand requests fail. Faster heartbeats do not help — they detect the loss
sooner and there is still nothing to route to. This is the report's
`1 − (1−p)^W` availability argument showing up as a cliff rather than a slope,
and it is why replication is for survival before it is for throughput.

**W=2 is enough to make a single loss almost free** (0–3 failed requests of
~328); W=3 adds little against one fault, as expected.

### Hop timeout (W=2, heartbeat 500 ms)

| timeout | detection | recovery | failed | retries |
|---|---|---|---|---|
| 250 ms | 1.488 s | 1.509 s | 2 | 22 |
| 1000 ms | 1.450 s | 1.934 s | 0 | 15 |
| 2000 ms | 1.589 s | 1.845 s | 0 | 8 |
| 5000 ms | 1.425 s | **3.934 s** | 0 | 9 |

**Recovery time is governed by the client's hop timeout, not by detection.**
Detection is flat at 1.43–1.59 s across the sweep, as it must be at fixed
heartbeat, while recovery grows from 1.51 s to 3.93 s as the timeout goes from
250 ms to 5 s. A stream blocked on a dead node cannot retry until its timeout
expires, however quickly the coordinator noticed.

**Three quantities, three independent knobs:**

| Quantity | Governed by |
|---|---|
| detection time | heartbeat interval / death threshold |
| recovery time | client hop timeout |
| failed requests | replica count W |

Tuning the wrong one does nothing. The timeout also has an optimum rather than
a monotone: at 250 ms it is short enough to abort healthy slow hops (2 failures,
22 retries — the most of any arm), and at 5 s it triples recovery.

One measurement caveat: `fault_sim_ts` is recorded when the `docker kill`
command returns, and the container stops being heard from slightly before that,
so `detection_s` is marginally understated. Each fault's `apply_ms` is recorded
in the manifest so the size of that effect is auditable.

---

## 7. Mode check

**The experiment that licenses every synthetic result, and the answer is a
qualified no.**

D=3, W=1, Qwen2.5-0.5B split 8/8/8 layers across three tier-2 containers, CPU
float32, greedy decoding. Sources: `datasets/analysis/exp7_mode_check.csv`,
`datasets/analysis/exp7_effective_bandwidth.json`.

Real mode works end to end: 0 failed requests, and the output is correct —
the fixed prompt ending *"In practice this means"* continues as
`" that the model is not reliable. The"`.

### Tier reference bandwidths are not achievable absolute values

Fitting `B_eff = stage weight bytes / measured compute`:

| | |
|---|---|
| fitted `B_eff` | **6.90 GB/s** (mean 6.74, sd 1.41, range 4.99–8.32, n=9) |
| tier-2 nominal | 45 GB/s |
| ratio | **6.5× slower than nominal** |

A single-core container running fp32 transformer decode achieves about 6.9 GB/s
— closer to the tier-1 reference (10 GB/s) than to the tier-2 one it was
configured as. This is the design doc's "tier emulation preserves ratios, not
absolute device speeds" stated as a number.

### Synthetic tracks real only at low concurrency

| arm | G | real p50 | synthetic p50 | error |
|---|---|---|---|---|
| nominal (45 GB/s) | 1 | 241.8 ms | 56.4 ms | **−76.7%** |
| nominal | 2 | 293.0 ms | 56.5 ms | −80.7% |
| nominal | 4 | 414.2 ms | 59.2 ms | −85.7% |
| calibrated (6.9 GB/s) | 1 | 241.8 ms | 232.4 ms | **−3.9%** |
| calibrated | 2 | 293.0 ms | 232.8 ms | −20.5% |
| calibrated | 4 | 414.2 ms | 285.0 ms | −31.2% |

Three conclusions, in order of importance:

**1. Uncalibrated synthetic mode is useless for absolute timing** — 77–86%
error. The emulated bandwidth must be fitted against real measurement, not
taken from the tier table.

**2. Calibrated, it matches decode latency at G=1 to 3.9%.** Seven of the eight
tokens in each request are decodes, so p50 is essentially the decode
comparison. The memory-bound model of a decode step is sound.

**3. It degrades with concurrency, because it cannot reproduce CPU
contention.** Real per-stage compute rises with load while synthetic stays flat:

| G | real stage compute (median) | synthetic (calibrated) |
|---|---|---|
| 1 | 57.7 ms | ~70 ms |
| 2 | 68.8 ms (+19%) | ~70 ms |
| 4 | 94.7 ms (+64%) | ~70 ms |

Co-located containers contend for the host's real memory bandwidth; a synthetic
worker sleeping a fixed interval cannot.

**Throughput is over-predicted more than latency** — by 1.6–2.2× even
calibrated. Real mode's *mean* per-stage compute (94.8–133.3 ms) far exceeds its
median (57.7–94.7 ms) because the first token of each request is a prefill,
pushing all 32 prompt positions through every layer. Synthetic mode models every
token identically and has no prefill/decode asymmetry at all. The research
report notes prefill as a distinct regime; the harness does not model it.

### What this licenses

Synthetic container results may be read as measurements of **network and churn
effects**, which is what the design document scopes them to. They may **not** be
read as predicting a real deployment's absolute throughput: they omit CPU
contention and prefill cost, both of which inflate them.

---

## 1s. Report scale, virtual engine

A reconstruction of the research report's appendix example A: 1000 Pi-class
(tier-1) nodes, 1T-parameter MoE at INT4 with 60B active, D=50, W=20, ℓ=1 ms.
N=1000 is ~16× what this host can run as containers.
Source: `datasets/summary/exp1s_runs.csv`.

| G | measured X | report's ramp | X/ramp | p50 | efficiency |
|---|---|---|---|---|---|
| 1 | 0.3 | 0.3 | 0.993 | 3.055 s | 0.001 |
| 100 | 33.0 | 33.3 | 0.991 | 3.055 s | 0.099 |
| 500 | 163.9 | 166.7 | 0.984 | 3.055 s | 0.492 |
| 1000 | 327.5 | 333.3 | 0.983 | 3.054 s | **0.983** |
| 1500 | 333.4 | 333.3 | 1.000 | 4.500 s | 1.000 |
| 2000 | 333.4 | 333.3 | 1.000 | 6.000 s | 1.000 |

The engine reproduces the report's independently-derived figures:

| | this harness | report appendix A |
|---|---|---|
| `X_max` | 333.3 tok/s | ~333 tok/s |
| `T_token` | 3.054 s | ~3.05 s |
| G to fill the pool | ~1000 | ~N = 1000 |

**This resolves the apparent conflict with experiment 1.** There the report's
ramp over-predicted by 32%; here it is accurate to 1–2%. The ramp assumes a
stream is computing continuously, so its error tracks the share of per-token
time spent on the network:

| | network share of `T_token` | ramp error |
|---|---|---|
| container scale (D=3, ℓ=5 ms) | 16 / 76 ms = 21% | 32% |
| report scale (D=50, ℓ=1 ms) | 50 / 3054 ms = 1.6% | 1–2% |

So `X(G) = X_max · min(1, G/N)` is not wrong — it is a bound whose slack is the
network fraction. At the report's own scale, where a 3-second memory-bound
compute dwarfs 50 ms of hops, it is an excellent predictor. On a fast LAN with
a shallow pipeline and short compute, it is not.

---

## Engine agreement

`tools/compare_engines.py` runs each scenario on both engines, equalising the
harness's own measured overhead per configuration so that what remains is
attributable to the engines' models. Sources:
`datasets/analysis/engine_comparison_exp1.csv`, `..._exp2.csv`.

| axis | exp1 median err | exp2 median err | tolerance | verdict |
|---|---|---|---|---|
| per-token p50 | 1.6% | 1.1% | 10% | **agree** |
| throughput | 7.0% | 6.2% | 10% | **agree** |
| stage compute | 0.0% | 0.0% | 10% | **agree** |
| stage network | 4.1% | 2.9% | 25% | **agree** |
| stage queue wait | 60.8% | 100% | 30% | see below |

**Latency, compute and network agree closely.** The virtual engine reproduces
container timing to within a few percent on every axis that composes into
latency.

**Throughput is over-predicted by 6–8%,** consistently. The virtual engine has
no jitter, so it never loses a scheduling slot.

**Queue wait needs reading by regime, not as one number:**

| regime | container mean | virtual mean | error |
|---|---|---|---|
| below the knee (G<12) | 0.35 ms | 0.12 ms | −100% (both ≈ 0) |
| at saturation (G≥12) | 15.5–89.1 ms | 14.7–84.4 ms | **5.1% median** |

Below saturation there is no queueing to compare and the relative error is an
artifact of dividing by nearly zero. Where queueing actually exists the two
engines agree to about 5%.

They distribute it differently, though. At G=16 the container mean is 29.4 ms
and the virtual mean 28.6 ms, but the *medians* are 22.3 ms and 0.004 ms: the
virtual engine's service and arrivals are both deterministic, so exactly a
third of hops wait ~86 ms and the rest wait nothing, a D/D/1 pattern, while
container jitter smears the same total across every hop. **The virtual engine
may be trusted for aggregate queueing and not for its distribution** — tail
latency and percentile questions at saturation need containers.

### Verdict

Virtual-engine results are trustworthy for per-token latency, stage compute,
network time and aggregate queueing, and over-predict throughput by a knowable
6–8%. They should not be used for queue-wait distributions or tail latency.
That is the basis on which the report-scale results above are offered.

---

## Findings about the harness itself

These are not results about distributed inference; they are the reasons the
results above can be believed. Each was found by the data disagreeing with the
model in a way that had a specific, locatable cause.

### Buffered events are measurements
The event writer flushed every 64 records. Containers are stopped by signal at
the end of a run, so each node silently lost whatever was still buffered — 31
hops per node in the first milestone run, 14% of the total. Detected by
checking an invariant that should hold exactly: hops per token must equal the
path length. It was 2.57 instead of 3.00.

Fixed by flushing on a timer as well as by count, and closing the log on
SIGTERM. The invariant now holds exactly.

### A transport artifact can imitate a real effect
The first implementation nested the HTTP calls, so a token's response unwound
back through every stage. That made a round trip pay `2D` one-way link delays
instead of `D`. Measured latency was 1.31× predicted.

This is the most dangerous class of error here, because the delay sweep exists
to measure how hop latency accumulates with depth — and this artifact would
have doubled its slope while still producing a clean, linear, entirely
plausible result. Fixed by making the pipeline forward-only, with the last
stage reporting to the gateway directly: the latency ratio fell to 1.10, and
the remaining residual was the container runtime's own cost (section 0).

### An unlogged hop hides in the residual
The hop table covered `D` hops of a `D+1` hop path: workers log the hops they
receive, and nobody logged the final stage's return to the gateway. Summed hop
times fell short of measured token latency by 6.5 ms — exactly the last stage's
link delay. Found by checking that the decomposition adds up to the whole.

### Awaiting an acknowledgement puts the network inside service time
Workers awaited the downstream acknowledgement *inside* the server loop, so a
stage could not begin the next token until the next stage had acknowledged the
last. That put a full link round trip inside service time, capping a stage at
`1/(compute + 2·hop)` instead of `1/compute`.

| | pre-fix | post-fix | predicted bound |
|---|---|---|---|
| Throughput at G=8 | 56.8 tok/s | 84.4 tok/s | 94.2 |
| p50 latency | 134.1 ms | 88.7 ms | 82.9 |
| Mean `busy_frac` | 0.621 | 0.912 | — |
| Queue wait by stage | [42.4, 7.4, 0.2] ms | [2.2, 0.8, 1.4] ms | — |

The pre-fix signature was distinctive and, read carelessly, looked like a real
finding: a hard throughput plateau with all queueing piled on stage 0 and later
stages idle. The arithmetic gave it away — `1/(0.0228 + 0.012) × 2 replicas =
57.5 tok/s` against a measured plateau of 56.8–58.1.

**The virtual engine is what located it.** The two engines run the same
decision code and disagreed on throughput; the virtual engine occupies a node
for its compute only, which is what the cost model says, so the disagreement
pointed directly at the container transport. This is the clearest argument for
having built both: an analytic model that merely agreed with the containers
would have confirmed the bug instead of exposing it.

### A hop timeout below saturated latency causes a retry storm
In a saturated closed-loop system latency grows as `G/X`, so a fixed timeout is
eventually exceeded by ordinary queueing. At G=32 with a 500 ms timeout, tokens
timed out, retried, added load, and collapsed throughput to a fifth of the
plateau (20.2 tok/s against 57) with 216 failed requests and 1682 retries — and
**zero hop errors**, since nothing had actually failed.

This is a real failure mode, and it is the "churn storm / unbounded retries"
worst case the research report describes. But it measures the timeout rather
than the utilisation curve, so the concurrency sweep now runs with headroom over
its worst expected latency, and the timeout is swept deliberately in experiment
4 instead.

### Dispatching on the wrong attribute starts a thousand containers
`drive.py` called the container runner for every scenario, without checking
`scn.engine`. Pointed at the report-scale sweep that meant generating a
compose file with 1002 services and asking Docker to start all of them on a
host sized for about sixty. It did not fail loudly; it simply made no progress
for ten minutes, and left 1002 created containers behind.

The lesson for the harness is that a scenario carries the engine it belongs to
and the driver must honour it. The lesson for reading the output is better: the
virtual engine runs that same configuration end to end, including collection,
in **2.7 seconds**, so "this is taking a while" was never a plausible reading
of the symptom.

### Separating collection from running pays for itself
The collector reads raw event logs and derives everything else, so every
analysis change — relabelling fault windows, tightening the recovery
definition, adding the closed-loop bounds — was re-applied to existing runs
without re-running a single container. Several of the fixes above were validated
this way in seconds.
