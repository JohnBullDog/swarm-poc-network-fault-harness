# Design notes: decisions, deviations, and known limits

Choices that a reader checking the numbers needs to know about, with the reason
for each. Where the implementation departs from the design document, it says so
and why.

## Deliberate deviations from the design document

### The fault controller runs host-side, not in a container
Detaching a container from a network and pausing or killing it requires the
Docker API. Reaching it from inside a container means mounting the Docker
socket, which is unrestricted control of the host daemon, and buys nothing on a
single host. The capability is identical either way. It still reads the
scenario's churn schedule, applies delay/loss/bandwidth/disconnect/pause/kill
exactly as specified, and reports each fault's simulated instant to the
coordinator.

### Baseline link shaping happens inside each container at startup
`docker/entrypoint.sh` applies the container's own `tbf` + `netem` profile
before the component starts. Applying it from the host afterwards leaves a
window in which the first requests cross an unshaped link, which would
contaminate the early samples of every run.

### A virtual-clock engine was added
Not in the design document. This host has 16 GB of RAM, not the 32 GB the
document assumes, and WSL2 caps container memory further; at ~30-40 MB per idle
container that is roughly 60 nodes, where the research report's worked examples
are at N = 1000 and N = 2750. The virtual engine reuses every decision module
and replaces only the clock and the transport. Container mode remains the
measurement of record; virtual results are used only on axes where
`tools/compare_engines.py` shows the engines agree.

## Transport

**HTTP/1.1 over aiohttp, everywhere.** The document allows HTTP or gRPC, "one
choice used everywhere". HTTP was chosen because gRPC multiplexes streams onto
one TCP connection, which would mask the per-hop loss behaviour the loss/jitter
experiment exists to observe, and because the `grpcio` wheel would roughly
double an image whose size caps the node count.

**Metadata in a header, activation in the body.** Metadata travels in
`X-Swarm-Meta`; the activation is the raw request body. Base64 inside a JSON
envelope would inflate every payload by a third and corrupt the `a/β`
serialisation term the delay sweep measures.

**Forward-only flow.** `gateway → s0 → … → s(D-1) → gateway`. The first
implementation nested the HTTP calls, so the response unwound back through every
stage and a round trip paid `2D` one-way link delays instead of `D`. Measured
latency was 1.31× predicted; after the change it was 1.10×, and the remaining
residual was accounted for by measured overhead. A transport artifact that
doubles the delay sweep's slope would have been indistinguishable from a real
finding about hop accumulation.

## Measurement decisions

### Hops are timed one-way, across containers, on the shared host clock
The sender stamps `sent_wall`; the receiver subtracts it. All containers share
the host kernel's `CLOCK_REALTIME` and every component is given the same
`RUN_EPOCH`, so no clock synchronisation is needed and `net_s` is true one-way
transit including the netem delay. This is what the design document's
shared-clock note exists to enable.

### netem shapes egress, so a hop costs the *sender's* delay
A hop `A → B` costs `A`'s configured delay. Per-tier delay is therefore set to
that tier's one-way link latency `ℓ`. A request/response round trip over one hop
costs `ℓ_A + ℓ_B`; the analysis reports one-way hop time, so it reports `ℓ`.

### The path has D+1 hops, and all of them are logged
`D` stage egresses (each stage forwards once, the last to the gateway) plus the
gateway's entry hop. The gateway logs the return hop as `stage == D`. Without
that row, summed hop times fall short of measured token latency by exactly the
last stage's link delay — which is how the omission was found.

The gateway's own link is shaped for the same reason: otherwise the entry hop is
the one unmeasured hop in the path, and measured latency comes out one hop
cheaper than the model predicts.

### Compare the model against p50, not the mean
A hop timeout during a fault window puts multi-second outliers into the latency
sample. In one bring-up run the mean was 195 ms against a p50 of 83.7 ms and a
prediction of 76 ms: the mean was reporting the stalls, not the latency. Both
statistics are kept in `runs.csv`; only p50 is a like-for-like comparison.

### Nothing is dropped; transients are labelled
Warm-up and fault-affected rows carry `in_steady_state` and `fault_window`
flags rather than being deleted, so a reader can see what a summary excluded and
choose differently.

### Fault windows close at recovery, not at detection
Streams blocked on a hop timeout keep stalling after the coordinator has
noticed the loss. Closing the window at detection left most of the disruption
inside the "clean" period.

### Three recovery quantities, not one
`reroute_s` (first token using the stage on another replica) is near-instant
under round-robin and says little on its own — another replica was already
taking its turn. `recovery_s` measures the first such token *after detection*:
one that succeeded because the system had routed around the failure.
`detection_s` is a measured difference between the fault controller's report and
the coordinator's decision, not an inference from the heartbeat interval.

## Cost-model refinements the measurements forced

### The utilisation ramp is an upper bound
`X(G) = X_max · min(1, G/N)` treats G streams as keeping G nodes continuously
busy, but a stream is idle while its activation is on the wire. Measured
throughput ran at ~0.65 of the ramp across the whole linear region. The
achievable closed-loop rate is the smaller of two classical bounds:

```
delay bound     G / T_token
capacity bound  W / max_k (compute_k)
```

Both are reported alongside the ramp, so a sweep shows which bound binds where.

### Capacity is set by compute, not by t_k
The report's `t_k = (W_k·b + KV_k)/B_k + ℓ` includes the hop, but a node is
occupied only for its compute — the activation's transit overlaps with it
serving the next token. `x_capacity_compute_tok_s` uses compute only and is the
bound the measured saturation plateau should be compared against.
`x_capacity_tok_s` keeps the report's form for reference.

This distinction was not academic: the first implementation awaited the
downstream acknowledgement inside the server loop, which really did put a link
round trip inside service time and capped a stage at `1/(compute + 2·hop)`. It
showed up as a plateau at 57 tok/s where the nodes could do 94, with all the
queueing on stage 0 and later stages idle. The virtual engine, which occupies a
node for compute only, disagreed with the containers — and that disagreement is
what located the bug.

### The overhead correction was removed: it was circular, then it wasn't constant
The design document defines synthetic stage time as bytes read over emulated
bandwidth "plus measured overhead", and the first implementation did exactly
that: `tools/calibrate.py` fitted a per-stage and per-hop residual and both the
worker's sleep and the prediction added them. Predicted latency moved from
76.2 ms to 82.9 ms against a measured p50 of 84.2 ms — a ratio of 1.017, which
looked like a well-calibrated harness.

It was not. The worker slept `bare + configured overhead` while the fit measured
the residual against `bare`, so each fit re-absorbed the sleep the previous fit
had introduced. Refitting on the next round of runs returned 2.226 ms where
1.204 had been configured, and a third round would have returned ~3.3: the loop
diverges by construction. The giveaway was in the delay sweep, where
measured/predicted climbed with depth (1.050 at D=2 to 1.112 at D=6) while the
*slope* against delay stayed correct to 0.5% — an additive per-stage error, not
a network-term error.

The worker now sleeps the bare cost-model time. With the circularity removed,
the residual is not a host constant:

| configuration | bare stage compute | residual |
|---|---|---|
| D=3, G=4 | 20.0 ms | 1.199 ms |
| D=6, G=1 | 10.0 ms | 0.219 ms |

`asyncio` sleep wakeups are delayed by whatever else the event loop is doing, so
this tracks load rather than hardware. Predictions are therefore the bare cost
model, and the harness's own cost is reported as the measured/predicted ratio.
The fit is still run and recorded, as a measurement of the harness rather than
a parameter of it.

The general lesson: a correction fitted from the same quantity it then modifies
will always improve the fit, and will always be meaningless.

### Hop timeout must be scaled to token latency
The 5000 ms default used during bring-up was ~65 token-times: a single node loss
stalled a stream for a third of the run and dominated every throughput figure.
Worse, in a saturated closed-loop system latency grows as `G/X`, so a timeout
below the saturated latency causes tokens to time out, retry, add load, and
collapse throughput — measured at a fifth of the plateau at G=32. The
concurrency sweep therefore uses a timeout with headroom over its worst expected
latency, and the timeout is swept deliberately in exp4, where bring-up showed
recovery time tracking it rather than the detection threshold.

## Model profiles

| Profile | `P_a·b` | Used for |
|---|---|---|
| `report_moe` | 30 GB | report-scale virtual runs; reproduces the appendix example |
| `poc_small` | 2.7 GB | container runs; ~23 ms/stage at D=3 on tier 2 |
| `qwen_0_5b` | 1.43 GB | real mode |

`poc_small` is scaled so synthetic stage times land in the tens of
milliseconds, the same order as the emulated per-hop delays (1–50 ms). The
report's own model would give ~3 s per token per stage, which would make every
run network-trivial and gather data absurdly slowly on one host.

`qwen_0_5b` declares `active_params` as the 24 decoder layers only (358M), not
the full 494M: the 136M tied embedding/output matrix lives on the gateway, not
on a stage. `tools/fetch_model.py` confirms the split — 1.432 GB in layers,
544 MB outside — against the 1.432 GB the profile declares.

## Determinism

- Churn schedules are generated once on the host from `(seed, node_id)` and
  replayed identically by both engines; faults are not re-drawn per engine.
- `protect_last_replica` is resolved at generation time, not at runtime, so
  neither engine makes a live decision the other might make differently. Every
  suppressed fault is recorded in the manifest with its reason, so a guard
  cannot silently change an experiment.
- Replica selection is round-robin over a sorted list; nothing consults an
  unseeded random source.
- Real-mode decoding is greedy, so the same scenario produces the same token
  sequence and output can be checked after a failover.
- The virtual scheduler breaks timestamp ties by a monotonic sequence number, so
  a run is reproducible to the last event.

## Known limits

1. **One host.** All containers compete for the same CPU and memory bandwidth.
   cgroup limits reduce but do not eliminate this. Runs are therefore sequential.
2. **`tc` is not a home network.** It models delay and loss well, not full link
   behaviour. The virtual engine's loss model is more optimistic still: one
   round trip per lost segment, which is TCP's fast-retransmit best case. A tail
   loss costs an RTO instead, so virtual loss results are a lower bound.
3. **Tiers preserve ratios, not absolute speeds.** Nothing here says what a
   Raspberry Pi would do.
4. **Per-tier availability and churn session lengths are this harness's
   defaults.** The research report gives them only qualitatively ("low, bursty"
   … "high"). Every value used is recorded in each run's manifest.
5. **Container node count caps at roughly 60 on this host.** Anything above that
   is virtual-engine only, and only on axes where the engines agree.
6. **`a ≈ 2·d·b`** is the report's definition of activation payload and is what
   predictions use. Real mode actually sends `d·b` bytes per decode token (and
   `T·d·b` on a prefill), a factor-of-two difference in the sub-millisecond
   `a/β` term.
