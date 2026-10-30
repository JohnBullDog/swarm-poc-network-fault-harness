# datasets

Output of the experimental campaign. Everything here is generated — rebuild it
with `swarm.runner.drive`, `tools.mode_check`, `tools.compare_engines` and
`tools.build_workbook`.

## Start here

**[`swarm_poc_results.xlsx`](swarm_poc_results.xlsx)** — the master workbook.
It opens on a Summary whose figures are live formulas over the data, and it
carries the three sheets a reader needs before any number: *Experiment design*
(what varied, what was held constant, and why), *Cost model & tiers* (the
expressions under test), and *Provenance & limits* (what bounds these results).

For prose, see [`../docs/findings.md`](../docs/findings.md).

## Layout

```
summary/    one row per run          <- join key for everything else
detail/     one row per observation  <- check a summary rather than trust it
analysis/   cross-run artifacts      <- calibration, engine and mode comparisons
_progress/  sweep resume state       <- bookkeeping, not data (git-ignored)
```

### `summary/`

`<exp>_runs.csv` (and `.parquet`): one row per run, carrying every scenario
parameter, every cost-model prediction, and every measurement side by side.
Self-contained — readable without the run directories it came from.

| file | rows | what |
|---|---|---|
| `exp1_runs.csv` | 10 | concurrency sweep, G = 1 … 32 |
| `exp2_runs.csv` | 16 | delay sweep, D ∈ {2,3,4,6} × ℓ ∈ {1,5,25,50} ms |
| `exp4_runs.csv` | 13 | disconnection and failover |
| `exp1s_runs.csv` | 9 | concurrency at report scale, N = 1000 (virtual engine) |

Every column is defined, with units, on the workbook's *Data dictionary* sheet.

### `detail/`

Per-observation tables concatenated across a sweep, so any summary figure can
be recomputed rather than taken on trust. Join to `summary/` on `run_id`.

| file | one row per |
|---|---|
| `<exp>_hop.csv.gz` | activation arrival — `net_s`, `queue_wait_s`, `compute_s` per hop |
| `<exp>_token_done.csv.gz` | completed token — end-to-end latency and the path it took |
| `exp4_recovery.csv` | injected fault — detection, recovery, stall |
| `exp4_membership.csv` | membership change — join, death detected |
| `<exp>_throughput_sample.csv` | one-second throughput sample |

The two wide tables are gzipped; `pandas.read_csv` opens them unchanged.

Rows carry `in_steady_state` and `fault_window` labels. **Nothing is deleted** —
transients are labelled so a reader can see what a summary excluded and choose
differently.

### `analysis/`

| file | what |
|---|---|
| `calibration.json` | the harness's own per-stage and per-hop overhead, measured. **Not** folded into any prediction: it is load-dependent, not a host constant. |
| `exp7_mode_check.csv` | real mode against synthetic, both arms |
| `exp7_effective_bandwidth.json` | fitted `B_eff` = 6.9 GB/s against a 45 GB/s nominal |
| `engine_comparison_<exp>.csv` | virtual engine against containers, per axis |
| `engine_comparison_<exp>_verdict.json` | per-axis median/max error and tolerance |

## Conventions

| | |
|---|---|
| `run_id` | the join key, on every table |
| `*_s` | a duration in seconds |
| `*_ts` | an instant; `sim_ts` is simulated time, `wall_ts` is unix time |
| `pred_*` | cost-model prediction, with **no fitted correction** |
| `meas_*` | measurement |
| one row | one observation; nothing nested except `path`, a JSON string |

Compare the model against `meas_t_token_s_p50`, not the mean: the mean carries
multi-second stalls from fault windows and reports those rather than the
steady-state latency the model predicts.

## Before drawing conclusions

These results do not validate peer-to-peer LLM inference in general. In
particular: every point is **n = 1 on a single seed**, so run-to-run variance
is unquantified and nothing here has error bars; synthetic mode over-predicts
throughput by 1.6–2.2× because it models neither CPU contention nor prefill
cost; tier bandwidths preserve ratios, not absolute device speeds (measured
6.5× optimistic for tier 2); and the virtual engine is invalid for tail
latency. The workbook's *Provenance & limits* sheet states all of these.
