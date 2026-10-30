"""Definitions for every column in the run-level datasets.

Kept as data, and checked for completeness by `coverage()`, so the dictionary
cannot silently fall behind the tables it describes. The workbook build fails
if any column in the data has no entry here.

Each entry is (group, units, definition). `units` is explicit on every numeric
column: a reader should never have to infer whether a latency is in seconds or
milliseconds from the magnitude.
"""

from __future__ import annotations

import re

SECONDS = "seconds"
MS = "milliseconds"
TOKS = "tokens/second"
COUNT = "count"
FRAC = "fraction 0-1"
RATIO = "ratio"
TEXT = "text"

# --- exact column names ----------------------------------------------------
COLUMNS: dict[str, tuple[str, str, str]] = {
    # identity and provenance
    "run_id": ("identity", TEXT,
               "Unique identifier for one run. The join key: every per-event "
               "table in datasets/ carries this column."),
    "image_id": ("identity", TEXT,
                 "Docker image the containers ran. Lets a dataset spanning a "
                 "rebuild be detected from the data rather than from memory."),
    "experiment": ("identity", TEXT,
                   "Which planned experiment this run belongs to: "
                   "1_concurrency, 2_delay, 4_failover, 1s_concurrency_scale, "
                   "7_mode_check, or milestone."),
    "scenario": ("identity", TEXT,
                 "Scenario name, unique within an experiment. Encodes the "
                 "sweep point, e.g. exp2_delay_D6_l25ms."),
    "seed": ("identity", COUNT,
             "Random seed. Fixes churn draws, replica selection order and "
             "(in the virtual engine) link jitter, so a run is repeatable."),

    # how the run was configured
    "engine": ("setup", TEXT,
               "container = real clock, real TCP shaped by tc netem/tbf, one "
               "Docker container per node. virtual = simulated clock, "
               "analytic link, no containers."),
    "mode": ("setup", TEXT,
             "synthetic = the worker sleeps the cost model's predicted time. "
             "real = the worker runs a contiguous block of Qwen2.5-0.5B "
             "layers on CPU."),
    "depth_D": ("setup", COUNT,
                "D, pipeline stages. The model's layers are cut into D "
                "contiguous blocks; one token traverses all of them in order."),
    "width_W": ("setup", COUNT,
                "W, replicas per stage. Each replica holds the same block. "
                "Sets fault tolerance: a stage survives while one replica is "
                "reachable."),
    "n_nodes_N": ("setup", COUNT, "N = D x W, total worker nodes."),
    "concurrency_G": ("setup", COUNT,
                      "G, independent agent streams held in flight. Closed "
                      "loop: a stream issues its next request the instant the "
                      "last one finishes."),
    "placement": ("setup", TEXT,
                  "capability_proportional = each stage's weight-bytes are "
                  "proportional to its holders' bandwidth, equalising stage "
                  "times. equal = every stage holds 1/D, letting the weakest "
                  "node gate the pipeline."),
    "model_profile": ("setup", TEXT,
                      "Which model the cost model describes: report_moe "
                      "(1T MoE, 60B active, P_a*b = 30 GB), poc_small "
                      "(P_a*b = 2.7 GB), qwen_0_5b (P_a*b = 1.43 GB, real)."),
    "prompt_tokens": ("setup", COUNT,
                      "s, context length at request start. Drives the KV term "
                      "s*q and, in real mode, the prefill cost."),
    "gen_tokens": ("setup", COUNT, "Tokens generated per request."),
    "duration_s": ("setup", SECONDS,
                   "Configured load duration, measured from the instant the "
                   "load generator starts, not from container boot."),
    "warmup_s": ("setup", SECONDS,
                 "Leading interval excluded from steady-state statistics. "
                 "Rows are labelled, never deleted."),
    "heartbeat_ms": ("setup", MS,
                     "Interval at which each node heartbeats the coordinator."),
    "death_after_ms": ("setup", MS,
                       "Heartbeat gap after which the coordinator marks a node "
                       "dead and removes it from the routable replica set."),
    "churn_kind": ("setup", TEXT,
                   "none, schedule (an explicit fault list) or random "
                   "(per-tier exponential up/down sessions)."),
    "n_faults": ("setup", COUNT, "Fault events in the resolved schedule."),
    "n_departures": ("setup", COUNT,
                     "Of those, events that took a node down "
                     "(disconnect/pause/kill)."),
    "n_tier1": ("setup", COUNT, "Nodes drawn from tier 1 (Pi-class, 10 GB/s)."),
    "n_tier2": ("setup", COUNT,
                "Nodes drawn from tier 2 (laptop/desktop, 45 GB/s)."),

    # cost-model predictions
    "pred_t_token_s": ("prediction", SECONDS,
                       "Predicted per-token latency over the full measurable "
                       "path: the report's expression plus the gateway's entry "
                       "hop. Carries NO fitted correction."),
    "pred_t_token_s_report": ("prediction", SECONDS,
                              "The research report's expression as written: "
                              "(P_a*b + s*q)/B + D*(l + a/beta), without the "
                              "gateway entry hop."),
    "pred_entry_hop_s": ("prediction", SECONDS,
                         "The gateway -> stage 0 hop. The difference between "
                         "the two predictions above."),
    "pred_x_max_tok_s": ("prediction", TOKS,
                         "Peak throughput, sum(B_i)/(P_a*b + s*q): aggregate "
                         "memory bandwidth divided by bytes read per token."),
    "pred_x_of_g_tok_s": ("prediction", TOKS,
                          "The report's utilisation ramp, "
                          "X_max*min(1, G/N). An UPPER BOUND: it assumes a "
                          "stream is computing continuously, so its slack is "
                          "the network's share of per-token time."),
    "pred_x_delay_bound_tok_s": ("prediction", TOKS,
                                 "G/T_token. The rate G serial streams can "
                                 "sustain. Binds below the knee."),
    "pred_x_capacity_tok_s": ("prediction", TOKS,
                              "W/max_k(t_k) with t_k as the report defines it, "
                              "including the hop. Kept for reference."),
    "pred_x_capacity_compute_tok_s": ("prediction", TOKS,
                                      "W/max_k(compute_k), compute only. A "
                                      "node is occupied for its compute; the "
                                      "activation's transit overlaps with it "
                                      "serving the next token. This is the "
                                      "bound the saturation plateau meets."),
    "pred_x_closed_loop_tok_s": ("prediction", TOKS,
                                 "min(delay bound, compute capacity bound). "
                                 "The quantity measured throughput actually "
                                 "tracks."),
    "pred_x_pipe_tok_s": ("prediction", TOKS,
                          "1/max_k(t_k), the single-pipeline straggler gate."),
    "pred_slowest_stage": ("prediction", COUNT,
                           "Index of the stage with the largest predicted t_k."),

    # the steady-state window
    "steady_lo_s": ("window", SECONDS,
                    "Simulated instant steady state begins: run_start + "
                    "warmup_s."),
    "steady_hi_s": ("window", SECONDS, "Simulated instant of the last token."),
    "steady_window_s": ("window", SECONDS,
                        "steady_hi_s - steady_lo_s. The denominator of "
                        "measured throughput."),
    "fault_window_s": ("window", SECONDS,
                       "Total simulated time inside a fault window (fault to "
                       "recovery), excluded from the *_nofault figures."),

    # measurements
    "tokens_total": ("measurement", COUNT, "Tokens completed over the whole run."),
    "tokens_steady": ("measurement", COUNT,
                      "Tokens completed inside the steady-state window. The "
                      "numerator of measured throughput."),
    "tokens_clean": ("measurement", COUNT,
                     "Steady-state tokens outside any fault window."),
    "meas_throughput_tok_s": ("measurement", TOKS,
                              "tokens_steady / steady_window_s."),
    "meas_t_token_s_p50": ("measurement", SECONDS,
                           "Median per-token latency, gateway send to gateway "
                           "receive. COMPARE THE MODEL AGAINST THIS, not the "
                           "mean."),
    "meas_t_token_s_mean": ("measurement", SECONDS,
                            "Mean per-token latency. Carries multi-second "
                            "stalls from fault windows, so it reports the "
                            "stalls rather than the steady-state latency."),
    "meas_t_token_s_p95": ("measurement", SECONDS, "95th percentile latency."),
    "meas_t_token_s_p99": ("measurement", SECONDS,
                           "99th percentile latency. Where hop timeouts show up."),
    "meas_throughput_tok_s_nofault": ("measurement", TOKS,
                                      "Throughput over the steady window with "
                                      "fault windows removed."),
    "meas_t_token_s_p50_nofault": ("measurement", SECONDS,
                                   "Median latency outside fault windows."),
    "meas_t_token_s_mean_nofault": ("measurement", SECONDS,
                                    "Mean latency outside fault windows."),
    "requests_started": ("measurement", COUNT, "Requests begun."),
    "requests_ok": ("measurement", COUNT, "Requests that generated every token."),
    "requests_failed": ("measurement", COUNT,
                        "Requests that exhausted their retries. The headline "
                        "cost of a fault."),
    "retries_total": ("measurement", COUNT,
                      "Token-level retry attempts across the run."),
    "failovers_total": ("measurement", COUNT,
                        "Retries that excluded a named failed node and "
                        "re-routed around it."),
    "hop_errors": ("measurement", COUNT,
                   "Hop-level failures logged by workers or the gateway."),
    "stalled_tokens": ("measurement", COUNT,
                       "Steady-state tokens whose latency exceeded 5x the "
                       "median: the stall tail."),
    "mean_busy_frac": ("measurement", FRAC,
                       "Mean fraction of the sample window a node's servers "
                       "were busy. ~0.9 indicates saturation."),
    "detection_s_mean": ("failover", SECONDS,
                         "Mean interval from a fault being applied to the "
                         "coordinator marking the node dead. Governed by the "
                         "heartbeat interval and death threshold."),
    "recovery_s_mean": ("failover", SECONDS,
                        "Mean interval from a fault to the first successful "
                        "token routed around it AFTER detection. Governed by "
                        "the client hop timeout, not by detection. Blank when "
                        "the stage never recovered (W=1)."),

    # model against measurement
    "throughput_efficiency": ("comparison", RATIO,
                              "meas_throughput_tok_s / pred_x_max_tok_s. The "
                              "report's 'throughput efficiency': the fraction "
                              "of aggregate bandwidth converted into tokens."),
    "latency_ratio_p50_over_pred": ("comparison", RATIO,
                                    "Measured p50 / predicted T_token. Above "
                                    "the saturation knee this is not a model "
                                    "error: the prediction is unloaded latency "
                                    "and the measurement includes queueing."),
    "latency_ratio_meas_over_pred": ("comparison", RATIO,
                                     "Measured MEAN / predicted. Contaminated "
                                     "by stalls; prefer the p50 ratio."),
}

# --- generated column families --------------------------------------------
STAGE_RE = re.compile(r"^stage(\d+)_(compute_s_mean|queue_wait_s_mean|"
                      r"net_s_mean|hops)$")
STAGE_DEFS = {
    "compute_s_mean": (SECONDS,
                       "Mean service time at stage {k}: time between starting "
                       "and finishing the token, excluding queueing and network."),
    "queue_wait_s_mean": (SECONDS,
                          "Mean time a token waited at stage {k} for a free "
                          "server before compute began."),
    "net_s_mean": (SECONDS,
                   "Mean one-way network time into stage {k}, from the "
                   "sender's send instant to arrival, measured on the shared "
                   "host clock. Includes the emulated link delay. Note this is "
                   "the SENDER's egress delay, since netem shapes egress."),
    "hops": (COUNT, "Hop records observed at stage {k}."),
}
SWEEP_RE = re.compile(r"^sweep_(.+)$")
SWEEP_DEFS = {
    "concurrency_G": (COUNT, "Swept axis: concurrent streams."),
    "delay_ms": (MS, "Swept axis: emulated one-way per-hop link delay."),
    "depth_D": (COUNT, "Swept axis: pipeline depth."),
    "width_W": (COUNT, "Swept axis: replicas per stage."),
    "heartbeat_ms": (MS, "Swept axis: heartbeat interval."),
    "hop_timeout_ms": (MS,
                       "Swept axis: how long the gateway waits for a token "
                       "before retrying."),
    "n_nodes": (COUNT, "Swept axis: pool size."),
    "arm": (TEXT,
            "Which sub-sweep a row belongs to. W_x_heartbeat varies replica "
            "count against heartbeat interval; hop_timeout varies the client "
            "timeout at fixed W=2 and heartbeat 500 ms."),
}


def describe(column: str) -> tuple[str, str, str] | None:
    """(group, units, definition) for a column, or None if undocumented."""
    if column in COLUMNS:
        return COLUMNS[column]
    m = STAGE_RE.match(column)
    if m:
        k, kind = m.group(1), m.group(2)
        units, text = STAGE_DEFS[kind]
        return ("per-stage", units, text.format(k=k))
    m = SWEEP_RE.match(column)
    if m and m.group(1) in SWEEP_DEFS:
        units, text = SWEEP_DEFS[m.group(1)]
        return ("sweep axis", units,
                text + " Blank for runs where this axis was held fixed.")
    return None


def coverage(columns) -> list[str]:
    """Columns with no definition. The workbook build refuses to ship a
    non-empty result, so the dictionary cannot fall behind the data."""
    return [c for c in columns if describe(c) is None]
