# Swarm PoC: network fault harness

A measurement harness for pipelined, distributed LLM inference across
heterogeneous, unreliable nodes. It exists to **gather data, not to demonstrate
features**, and answers one question:

> How do simulated network delays and disconnections affect a pipelined,
> distributed LLM inference network, across different network compositions?

Docker containers stand in for volunteer devices on a single host. Measurements
test a cost model's predictions for latency, throughput and utilisation, and
produce the traces that reliable-capacity estimation needs.

## The cost model under test

Four expressions govern the system. Everything the harness predicts, and
everything synthetic mode sleeps for, derives from them:

```
T_token  ≈  (P_a·b + s·q) / B   +   D·(ℓ + a/β)      per-token latency
X_max    ≈  Σ B_i / (P_a·b + s·q)                    peak throughput
X(G)     ≈  X_max · min(1, G/N)                      utilisation ramp
X_pipe    =  1 / max_k t_k,  t_k = (W_k·b + KV_k)/B_k + ℓ    straggler gate
```

The decisive feature of the first is that it uses **one** device's bandwidth
`B`, not the sum: adding devices never shrinks it. The second grows linearly
with device count, but only if enough independent requests are in flight.

## Two engines, and why there are two

| | Container engine | Virtual engine |
|---|---|---|
| Clock | real; a stage really sleeps | simulated; advanced by an event scheduler |
| Network | real TCP through `tc netem` + `tbf` on a Docker bridge | analytic delay, jitter, loss and serialisation |
| Scale on one host | ~60 nodes (16 GB RAM) | 1000+ nodes |
| Speed | wall-clock | 100–300× faster than real time |
| Status | **the measurement of record** | trusted only where it agrees with containers |

Both engines import the *same* decision code — `costmodel`, `placement`,
`routing`, `membership`, `churn`, `events`. They differ only in how time passes
and how a message crosses a link. That is what makes the comparison in
`tools/compare_engines.py` a comparison of engines rather than of
configurations, and it is the gate on using virtual results at all.

Simulated and real time are tracked separately everywhere; every event row
carries both `sim_ts` and `wall_ts`.

## Two compute modes

| | Real mode | Synthetic mode |
|---|---|---|
| Stage worker | a contiguous block of Qwen2.5-0.5B layers on CPU | sleeps the time the cost model predicts |
| Payload | real hidden-state tensors | placeholder bytes of identical size |
| Measures | real compute contention, serialisation, output correctness after failover | network and churn effects at scale |

Synthetic stage time is the bare `(W_k·b + KV_k)/B_k`. The container runtime's
own overhead is measured rather than modelled — see Calibration.

## Components

| Component | Role |
|---|---|
| Gateway + load generator | generates G concurrent agent streams; tokenisation, embedding and the output head in real mode; picks one replica per stage; retries a failed hop |
| Coordinator | membership by heartbeat, death detection, maintains the replica set the gateway routes over. Placement is static; failover only reroutes among survivors. |
| Stage workers | hold one stage, accept an activation, compute, forward onward. Heartbeat and log every hop. |
| Fault controller | applies the churn schedule: delay, jitter, loss, bandwidth caps, disconnect, pause, kill, rejoin |
| Collector | merges per-component event logs into tidy tables |

Activations flow **forward only**: `gateway → s0 → … → s(D-1) → gateway`. An
earlier version nested the HTTP calls so the response unwound back through every
stage, which made a round trip pay `2D` one-way link delays instead of `D` — an
artifact of the transport that would have doubled the delay sweep's slope.

The fault controller runs host-side rather than in a container: detaching a
container from a network and pausing or killing it needs the Docker API, and
reaching it from inside a container means mounting the Docker socket — full
control of the host daemon — for no gain on a single host.

## Device tiers

Tiers are emulated **relative to each other**; one host cannot reproduce a real
GPU or an IoT chip. Memory bandwidth and availability drive the cost model; the
cgroup and `tc` limits preserve the tiers' ratios.

| Tier | Stands for | B (mem bw) | CPU | Memory | Link | Delay |
|---|---|---|---|---|---|---|
| 3 | Gaming PC (GPU) | 600 GB/s | 2.0 | 4 GB | 1 Gbit/s | 1 ms |
| 2 | Laptop / idle desktop | 45 GB/s | 1.0 | 2 GB | 500 Mbit/s | 5 ms |
| 1 | Single-board (Pi-class) | 10 GB/s | 0.5 | 1 GB | 100 Mbit/s | 10 ms |
| 0 | Microcontroller / IoT | 3 GB/s | 0.1 | 256 MB | 10 Mbit/s | 50 ms |

Bandwidths are reference values; per-tier availability and churn session lengths
are this harness's own defaults, since the source gives them only
qualitatively. Every value used is recorded in each run's manifest.

## Calibration

**Predictions are the bare cost model.** No fitted correction is folded into
them, and the harness's own cost is reported as the measured/predicted ratio
rather than subtracted out of it. Two findings forced that choice.

An earlier version fitted a per-stage and per-hop overhead and added both to the
prediction, which made the agreement look better — a latency ratio of 1.017
instead of ~1.07. But the fit was **circular**: the stage worker slept
`bare + configured overhead` while the fit measured the residual against
`bare`, so each fit re-absorbed the sleep the previous one had added. Successive
fits diverged: 1.2 → 2.2 → 3.3 ms per stage. The worker now sleeps the bare
cost-model time, and the container runtime's overhead is whatever it adds on
top.

With the circularity gone, the residual turned out **not to be a constant**:

| configuration | bare stage compute | measured residual |
|---|---|---|
| D=3, G=4 | 20.0 ms | 1.199 ms |
| D=6, G=1 | 10.0 ms | 0.219 ms |

`asyncio`'s sleep wakeups are delayed by however busy the event loop is, so the
overhead scales with load, not with the host. A single fitted figure would have
been a fudge factor that flattered the model while hiding its real accuracy.

`tools/calibrate.py` still measures and records this (`datasets/analysis/calibration.json`)
— as a measurement *of* the harness, not a parameter *for* it.

**Engine agreement** (`tools/compare_engines.py`) runs the same scenarios on
both engines and reports relative error per axis, with a verdict per axis. A
disagreement is reported, not reconciled.

## Running it

Requires Docker, and Python 3.12 on the host.

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
docker build -f docker/Dockerfile.synthetic -t swarm-poc:synthetic .

# one scenario
python -m swarm.runner.run scenarios/m1_minimal.yaml

# a planned experiment (sequential; --resume to continue an interrupted sweep)
python -m swarm.runner.drive exp1

# the same scenario on the simulated clock
python -m swarm.virtual.run scenarios/m1_minimal.yaml

# calibration and engine agreement
python -m tools.calibrate
python -m tools.compare_engines exp1
```

Real mode additionally needs the model fetched once and the heavier image:

```bash
python -m tools.fetch_model
docker build -f docker/Dockerfile.real -t swarm-poc:real .
python -m tools.mode_check          # experiment 7, three phases
```

`exp7` is run through `tools/mode_check.py` rather than the generic driver,
because it needs three phases in order: real runs first, then the effective
memory bandwidth fitted from them, then synthetic arms on both the nominal and
the fitted bandwidth. The driver would run the arms without that fit and the
comparison would measure the wrong thing.

Check the output is tidy at any point:

```bash
python -m tools.selfcheck            # invariants, no Docker needed
python -m tools.verify_tidy          # schema, types, joins, no silent loss
```

Runs are **sequential by design**. Every container shares one host, so two
concurrent runs would each measure the other's interference.

## Experiments

| | Question |
|---|---|
| `exp1` | Concurrency sweep: how does throughput track the utilisation curve as G rises? |
| `exp2` | Delay sweep: how does per-hop latency accumulate with depth and delay? |
| `exp4` | Disconnection and failover: how fast is a loss detected and routed around, at what cost in failed requests? |
| `exp7` | Mode check: do synthetic timings track real ones? |

`exp7` is the experiment that licenses every synthetic result. If synthetic
timings do not track real ones there, nothing measured in synthetic mode at
larger scale can be trusted.

### Results

- **[`datasets/swarm_poc_results.xlsx`](datasets/swarm_poc_results.xlsx)** —
  master workbook: a Summary sheet whose figures are live formulas over the
  data, one sheet per experiment, every run combined on one sheet for
  pivoting, and a data dictionary. Rebuild with `python -m tools.build_workbook`.
- **[`docs/findings.md`](docs/findings.md)** — the write-up, every number
  sourced to a table.
- **[`docs/data-dictionary.md`](docs/data-dictionary.md)** — full output schema.
- [`datasets/`](datasets/) — the data, laid out as `summary/` (one row per
  run), `detail/` (one row per observation), and `analysis/` (calibration,
  engine and mode comparisons). Join everything on `run_id`;
  [`datasets/README.md`](datasets/README.md) is the guide.

## Interpreting the results

Three limits apply, and they bound what any number here can mean:

1. **One host.** All containers compete for the same CPU and memory bandwidth.
   The cgroup limits reduce but do not eliminate this.
2. **`tc` is not a home network.** It models delay and loss well, not the full
   behaviour of real links. The virtual engine's loss model is more optimistic
   still: it charges one round trip per lost segment, which is TCP's best case.
3. **Tiers preserve ratios, not absolute speeds.** Results speak to relative
   effects, not to what a Raspberry Pi would do.

## Scope

In scope: pipeline-parallel inference across containerised stage workers with
replicas; tier emulation; injected delay, jitter, loss, bandwidth caps and
disconnection; churn; replica failover; a concurrent load generator; complete
timestamped logging with repeatable runs.

Out of scope: the model ladder and capacity governor; specialists and the demand
queue; integrity checking and security (all containers are trusted);
multi-host deployment and real volunteer devices.

## Repository layout

```
swarm/core/          cost model, scenario schema, placement, routing,
                     membership, churn, clocks, event schema  (shared by both engines)
swarm/components/    gateway, coordinator, stage worker, fault controller,
                     real-mode stage and gateway
swarm/runner/        compose generation, the run orchestrator, experiment definitions
swarm/virtual/       discrete-event scheduler and the virtual engine
swarm/analysis/      the collector that produces tidy tables, and figures
tools/               model fetch, calibration, engine comparison, mode check,
                     tidiness checks, workbook build
scenarios/           milestone scenarios
datasets/            results: summary/, detail/, analysis/, and the workbook
docs/                findings, data dictionary, design notes, figures
```

## References

The cost model and the architecture this harness tests are developed in a
separate research write-up, which builds on:

- Borzunov et al. — *Petals: Collaborative Inference and Fine-tuning of Large
  Models*. The closest system: transformer blocks distributed BitTorrent-style
  across volunteer machines, with fault tolerance and rerouting.
- Shoeybi et al. — *Megatron-LM: Training Multi-Billion Parameter Language
  Models Using Model Parallelism*. Tensor parallelism, excluded here by design:
  its per-layer all-reduce dominates over commodity links.
- Huang et al. — *GPipe: Efficient Training of Giant Neural Networks using
  Pipeline Parallelism*.
- Narayanan et al. — *PipeDream: Generalized Pipeline Parallelism for DNN
  Training*. GPipe and PipeDream introduced the micro-batching and scheduling
  that pipeline throughput relies on.
- Rajbhandari et al. — *ZeRO-Infinity* / ZeRO-Inference, and Sheng et al. —
  *FlexGen*. The single-node end of the same bandwidth-bound tradeoff.
- Cihangiroglu and Nocera — activation-drift detection using secret canary
  inputs, which must separate tampering from exactly the heterogeneity-induced
  variation measured here.
- *VeriLLM* — Merkle commitments on intermediate states plus sampled rerun
  verification.
- Wan et al. — *DAWN* (AAAI 2026). Synthesises the communication graph among
  LLM agents; the kind of structured, correlated agent load that would run on
  the substrate modelled here.

Model used in real mode: **Qwen2.5-0.5B-Instruct** (Qwen team, Alibaba Cloud),
fetched from Hugging Face and run on CPU.
