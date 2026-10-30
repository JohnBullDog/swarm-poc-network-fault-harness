# Data dictionary

Every run writes a self-contained directory. Raw event logs are append-only JSON
lines, one file per component; the collector turns them into one tidy table per
event kind plus two derived tables.

```
runs/<run_id>/
  scenario.yaml      the scenario as given, verbatim
  manifest.json      resolved scenario, placement, predictions, host facts,
                     the full fault schedule, and any suppressed faults
  compose.yaml       the generated stack (container runs only)
  docker-logs.txt    container stdout/stderr (container runs only)
  raw/*.jsonl        one event log per component, exactly as written
  tidy/*.csv         one table per event kind, plus derived tables
  tidy/*.parquet     the same tables, where non-empty
  tidy/_steady_window.json
```

## Conventions

| Convention | Meaning |
|---|---|
| `*_ts` | an instant, in seconds |
| `*_s` | a duration, in seconds |
| `sim_ts` | **simulated** time since the run epoch |
| `wall_ts` | absolute host unix time |
| one row | one observation; nothing is nested |
| `path` | the routed node list, stored as a JSON string so the table stays flat |

`sim_ts` and `wall_ts` are both present on every row in both engines. In
container runs they advance together (`time_scale` is forced to 1.0, because tc
netem delays are real milliseconds and cannot be dilated). In virtual runs
`sim_ts` is the scheduler's clock and has no relation to `wall_ts`.

Scenario parameters are **not** copied onto event rows. They live once per run,
in `tidy/runs.csv`, keyed by `run_id`. Join on `run_id`.

Nothing is deleted. The warm-up transient and fault-affected periods are
*labelled*, so a reader can see what was excluded from a summary and choose
differently:

| Column | On tables | Meaning |
|---|---|---|
| `in_steady_state` | `token_done`, `hop`, `hop_error`, `node_sample`, `throughput_sample`, `request_end` | row falls inside `[run_start + warmup_s, last token]` |
| `fault_window` | `token_done`, `hop`, `request_end`, `throughput_sample` | row falls between a fault and its recovery |

## Common columns

Present on every row of every event table.

| Column | Type | Meaning |
|---|---|---|
| `run_id` | str | the run this row belongs to |
| `engine` | str | `container` or `virtual` |
| `mode` | str | `synthetic` or `real` |
| `src` | str | component that wrote it (`gateway`, `coordinator`, a node id, `virtual_engine`) |
| `kind` | str | the event kind, equal to the table name |
| `wall_ts` | float | absolute host unix time |
| `sim_ts` | float | simulated seconds since the run epoch |

## Event tables

### `hop` — one row per activation arrival
The per-hop decomposition. A path of depth *D* produces *D+1* rows per token:
*D* stage arrivals plus the return into the gateway.

| Column | Meaning |
|---|---|
| `request_id`, `token_idx` | which token |
| `stage` | the receiving stage; `stage == D` marks the return hop into the gateway |
| `node_id`, `tier` | who received it (`gateway` for the return hop) |
| `attempt` | 1 on the first try; higher after a retry |
| `net_s` | **one-way** network time, from the sender's `sent_wall` to arrival. Includes the emulated link delay. Measured across containers using the shared host clock. |
| `queue_wait_s` | time between arrival and the start of compute |
| `compute_s` | stage service time |
| `total_s` | `net_s + queue_wait_s + compute_s` |
| `payload_bytes` | exact bytes on the wire for this hop |

`net_s` is the **sender's** egress delay, because netem shapes egress. A hop
`A -> B` costs `A`'s configured delay.

### `token_done` — one row per completed token
| Column | Meaning |
|---|---|
| `request_id`, `stream_id`, `token_idx` | which token, from which simulated agent |
| `latency_s` | end-to-end, gateway send to gateway receive |
| `context_len` | context length after this token |
| `retries`, `failovers` | incurred by *this* token |
| `path` | JSON list of the nodes it traversed |

### `request_end` — one row per finished request
`status` is `ok`, `failed`, or `aborted` (still running when the run ended).
`retries` and `failovers` are totals across the request's tokens. In real mode
`error` carries the decoded continuation for a successful request, so output can
be inspected after a failover.

### `membership` — one row per membership change
| Column | Meaning |
|---|---|
| `event` | `join` or `death_detected` |
| `gap_s` | observed heartbeat gap that triggered a death |
| `fault_sim_ts` | when the fault controller reported applying the fault |
| `detection_s` | `sim_ts - fault_sim_ts`; a measured difference, not an inference from the heartbeat interval |

### `fault_applied` — one row per injected fault
`action` is `disconnect`, `rejoin`, `pause`, `unpause` or `kill`. `detail` is a
JSON string carrying how long the docker command itself took (`apply_ms`) and
the scheduled time, so that cost is visible rather than buried.

### `node_sample`, `heartbeat`, `throughput_sample`
Periodic samples. `node_sample` carries `inflight`, `queue_depth`, `busy_frac`
(fraction of the sample window the node's servers were busy) and `hb_gap_s`.
Coordinator-written `node_sample` rows carry only `hb_gap_s`; worker-written
rows carry the rest.

## Derived tables

### `recovery` — one row per disconnection
Derived by joining faults, membership and tokens. Three different quantities,
kept separate because they answer differently:

| Column | Meaning |
|---|---|
| `detection_s` | fault to the coordinator marking the node dead |
| `reroute_s` | fault to the first token using that stage on another replica. Under round-robin this is near-instant and says little on its own. |
| `recovery_s` | fault to the first such token **after detection** — a token that succeeded because the system had routed around the failure |
| `stall_s` | fault to the last error naming the node, bounded to the last completed token |
| `tokens_failed_between` | hop errors naming the node before recovery |

### `runs` — one row per run
Scenario parameters, cost-model predictions and measurements side by side.

Predictions:

| Column | Expression |
|---|---|
| `pred_t_token_s_report` | the report's `(P_a·b + s·q)/B + D(ℓ + a/β)`, without the gateway entry hop |
| `pred_t_token_s` | the full measurable path: the above plus the gateway entry hop |
| `pred_x_max_tok_s` | `ΣB_i / (P_a·b + s·q)` |
| `pred_x_of_g_tok_s` | `X_max · min(1, G/N)` — the report's ramp |
| `pred_x_delay_bound_tok_s` | `G / T_token` |
| `pred_x_capacity_tok_s` | `W / max_k t_k` |
| `pred_x_closed_loop_tok_s` | `min(delay bound, capacity bound)` |
| `pred_x_pipe_tok_s` | `1 / max_k t_k` |

Measurements: `meas_throughput_tok_s`, `meas_t_token_s_{mean,p50,p95,p99}`, the
same two with a `_nofault` suffix computed outside fault windows,
`requests_{started,ok,failed}`, `retries_total`, `failovers_total`,
`hop_errors`, `mean_busy_frac`, `stalled_tokens`, and `stage<k>_*_mean` per
stage.

Comparisons: `throughput_efficiency` (measured / `X_max`),
`latency_ratio_meas_over_pred`, and `latency_ratio_p50_over_pred`.

**Compare the model against p50, not the mean.** A hop timeout during a fault
window puts multi-second outliers in the latency sample; the mean then reports
the stalls rather than the steady-state latency the model predicts. Both are
kept, but only one is a like-for-like comparison.

## Consolidated datasets

`datasets/<exp>_runs.csv` concatenates the one-row `runs` tables across a
sweep, and is self-contained — every parameter and prediction is on the row.
`datasets/<exp>_{hop,token_done,recovery,membership,throughput_sample}.csv`
concatenate the per-observation tables, so a summary can be checked rather than
taken on trust.

`datasets/calibration.json` records the harness's measured per-stage and
per-hop overhead and its spread. These are *not* folded into any prediction:
they are load-dependent, not host constants, so predictions are the bare cost
model and the harness's cost appears as `latency_ratio_p50_over_pred`. `datasets/engine_comparison_<exp>.csv` and its `_verdict.json` record
how far the virtual engine agrees with the containers, per axis.
