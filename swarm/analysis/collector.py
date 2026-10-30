"""Merge one run's raw event logs into tidy tables.

Tidy rules this module enforces:

  * One table per event kind. One row per observation. No mixed-type columns.
  * Columns come from the declared schema, not from whatever the run emitted,
    so a table has the same shape whether or not this run produced those rows.
    That is what lets a sweep's runs be concatenated without columns appearing
    and disappearing.
  * Scenario parameters live once, in runs.csv, keyed by run_id -- not copied
    onto every event row.
  * Nothing is dropped. The warm-up transient is *labelled*
    (`in_steady_state`), not deleted, so a reader can see what was excluded and
    change their mind.

Two tables are derived rather than logged, because computing them needs events
from several components joined together:

  recovery   per disconnection: detection latency, and time to the first
             successful token rerouted around the lost node.
  runs       one row per run: the scenario's parameters, the cost model's
             predictions, and the matching measurements side by side.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..core.events import EVENT_SCHEMA, columns_for, read_jsonl

DOWN_ACTIONS = ("disconnect", "pause", "kill")


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load_raw(run_dir: Path) -> pd.DataFrame:
    """Every event from every component, in one frame, sorted by simulated time."""
    rows: list[dict] = []
    raw = run_dir / "raw"
    for path in sorted(raw.glob("*.jsonl")):
        rows.extend(read_jsonl(path))
    if not rows:
        return pd.DataFrame(columns=["run_id", "kind", "sim_ts", "wall_ts"])
    df = pd.DataFrame(rows)
    return df.sort_values(["sim_ts", "kind"], kind="stable").reset_index(drop=True)


def split_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """One frame per event kind, with the declared columns in declared order."""
    out: dict[str, pd.DataFrame] = {}
    for kind in EVENT_SCHEMA:
        cols = columns_for(kind)
        if df.empty or "kind" not in df.columns:
            out[kind] = pd.DataFrame(columns=cols)
            continue
        sub = df[df["kind"] == kind]
        for col in cols:
            if col not in sub.columns:
                sub = sub.assign(**{col: pd.NA})
        out[kind] = sub[cols].reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# steady state
# ---------------------------------------------------------------------------
def steady_window(tables: dict[str, pd.DataFrame], manifest: dict) -> tuple[float, float]:
    """[start, end] of the interval that counts as steady state.

    Starts at the gateway's `run_start` plus the scenario's warm-up, so that
    container boot and pipeline fill are excluded. Ends at the last token, since
    the final fraction of a second may be a partially-elapsed window.
    """
    scn = manifest.get("scenario", {})
    warmup = float(scn.get("warmup_s") or 0.0)
    rs = tables.get("run_start")
    start = float(rs["sim_ts"].iloc[0]) if rs is not None and len(rs) else 0.0
    start += warmup
    tok = tables.get("token_done")
    end = float(tok["sim_ts"].max()) if tok is not None and len(tok) else start
    return start, max(start, end)


def fault_windows(recovery: pd.DataFrame) -> list[tuple[float, float]]:
    """Intervals in which a fault was still disrupting the run.

    A window opens when the fault is applied and closes at recovery -- the first
    successful token routed around the loss after detection -- not at detection
    itself. Streams blocked on a hop timeout keep stalling after the coordinator
    has noticed, so closing the window at detection would leave most of the
    disruption inside the "clean" period and understate clean throughput.
    """
    if recovery is None or len(recovery) == 0:
        return []
    out: list[tuple[float, float]] = []
    for _, r in recovery.iterrows():
        t0 = float(r["fault_sim_ts"])
        end = r["first_ok_sim_ts"]
        if pd.isna(end):
            end = r["detected_sim_ts"]
        out.append((t0, float(end) if not pd.isna(end) else t0))
    return out


def label_fault_window(tables: dict[str, pd.DataFrame],
                       windows: list[tuple[float, float]]) -> None:
    """Flag rows that fall inside a fault window. Nothing is dropped."""
    for name in ("token_done", "hop", "request_end", "throughput_sample"):
        t = tables.get(name)
        if t is None:
            continue
        if len(t) == 0:
            tables[name] = t.assign(fault_window=pd.Series(dtype="boolean"))
            continue
        ts = pd.to_numeric(t["sim_ts"], errors="coerce")
        flag = pd.Series(False, index=t.index)
        for lo, hi in windows:
            flag |= (ts >= lo) & (ts <= hi)
        tables[name] = t.assign(fault_window=flag.astype("boolean"))


def label_steady(tables: dict[str, pd.DataFrame], lo: float, hi: float) -> None:
    for name in ("token_done", "hop", "hop_error", "node_sample",
                 "throughput_sample", "request_end"):
        t = tables.get(name)
        if t is None:
            continue
        if len(t) == 0:
            tables[name] = t.assign(in_steady_state=pd.Series(dtype="boolean"))
        else:
            ts = pd.to_numeric(t["sim_ts"], errors="coerce")
            tables[name] = t.assign(
                in_steady_state=((ts >= lo) & (ts <= hi)).astype("boolean"))


# ---------------------------------------------------------------------------
# derived: recovery
# ---------------------------------------------------------------------------
def derive_recovery(tables: dict[str, pd.DataFrame], manifest: dict) -> pd.DataFrame:
    """Per disconnection: detection latency and time to reroute around the node.

    `recovery_s` is the interval from the fault to the first *successful* token
    that traversed the lost node's stage on a different replica. A token that
    happened to avoid the node before the loss was even detected does not count,
    because it demonstrates nothing about recovery.
    """
    cols = ["run_id", "node_id", "tier", "stage", "action", "fault_sim_ts",
            "detected_sim_ts", "first_ok_sim_ts", "detection_s", "recovery_s",
            "reroute_s", "stall_s", "tokens_failed_between", "scheduled_at_s",
            "apply_ms"]
    faults = tables.get("fault_applied", pd.DataFrame())
    if faults is None or len(faults) == 0:
        return pd.DataFrame(columns=cols)

    member = tables.get("membership", pd.DataFrame())
    tok = tables.get("token_done", pd.DataFrame())
    errs = tables.get("hop_error", pd.DataFrame())
    run_id = manifest.get("run_id")
    t_last = (float(pd.to_numeric(tok["sim_ts"], errors="coerce").max())
              if tok is not None and len(tok) else float("inf"))
    node_stage = {n: s for s, nodes in
                  ((st["stage"], st["nodes"]) for st in manifest["plan"]["stages"])
                  for n in nodes}
    applied = {(a["node_id"], round(float(a["fault_sim_ts"]), 6)): a
               for a in manifest.get("faults_applied", [])}

    # Pre-parse token paths once; the path is stored as a JSON string to keep
    # the table flat, so it has to be decoded to ask which nodes a token used.
    tok_paths = []
    if len(tok):
        for _, r in tok.iterrows():
            try:
                tok_paths.append(set(json.loads(r["path"]) or []))
            except (TypeError, json.JSONDecodeError):
                tok_paths.append(set())

    rows = []
    for _, f in faults.iterrows():
        if f["action"] not in DOWN_ACTIONS:
            continue
        node = f["node_id"]
        t_fault = float(f["sim_ts"])
        stage = node_stage.get(node, f.get("stage"))

        det = pd.NA
        if len(member):
            m = member[(member["node_id"] == node)
                       & (member["event"] == "death_detected")
                       & (pd.to_numeric(member["sim_ts"], errors="coerce") >= t_fault)]
            if len(m):
                det = float(pd.to_numeric(m["sim_ts"]).min())

        # Two different questions, kept apart because they answer differently:
        #
        #   reroute_s   the first token after the fault that used the lost node's
        #               stage on another replica. Under round-robin routing this
        #               happens almost immediately and says little -- another
        #               replica was already taking its turn.
        #   recovery_s  the first such token AFTER the coordinator detected the
        #               loss. This is recovery in the sense the design doc
        #               means: a token that succeeded because the system had
        #               routed around the failure, not in spite of it.
        reroute = pd.NA
        first_ok = pd.NA
        after = det if det is not pd.NA else t_fault
        if len(tok):
            ts = pd.to_numeric(tok["sim_ts"], errors="coerce").to_numpy()
            for i in range(len(tok)):
                if ts[i] <= t_fault:
                    continue
                path = tok_paths[i]
                if node in path:
                    continue
                if not any(node_stage.get(p) == stage for p in path):
                    continue
                if reroute is pd.NA:
                    reroute = float(ts[i])
                if ts[i] > after:
                    first_ok = float(ts[i])
                    break

        # How long the system kept trying the lost node before giving up on it.
        # Bounded to the last completed token: errors logged during teardown,
        # after the load generator stopped, say nothing about the run.
        stall = pd.NA
        if len(errs):
            e_ts = pd.to_numeric(errs["sim_ts"], errors="coerce")
            e = errs[(errs["node_id"] == node) & (e_ts >= t_fault) & (e_ts <= t_last)]
            if len(e):
                stall = float(pd.to_numeric(e["sim_ts"]).max()) - t_fault

        failed_between = 0
        if len(errs):
            e_ts = pd.to_numeric(errs["sim_ts"], errors="coerce")
            upto = first_ok if first_ok is not pd.NA else float("inf")
            failed_between = int(((e_ts > t_fault) & (e_ts <= upto)
                                  & (errs["node_id"] == node)).sum())

        meta = applied.get((node, round(t_fault, 6)), {})
        rows.append({
            "run_id": run_id, "node_id": node, "tier": f.get("tier"),
            "stage": stage, "action": f["action"], "fault_sim_ts": t_fault,
            "detected_sim_ts": det, "first_ok_sim_ts": first_ok,
            "detection_s": (det - t_fault) if det is not pd.NA else pd.NA,
            "recovery_s": (first_ok - t_fault) if first_ok is not pd.NA else pd.NA,
            "reroute_s": (reroute - t_fault) if reroute is not pd.NA else pd.NA,
            "stall_s": stall,
            "tokens_failed_between": failed_between,
            "scheduled_at_s": meta.get("scheduled_at_s"),
            "apply_ms": meta.get("apply_ms"),
        })
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# derived: one row per run
# ---------------------------------------------------------------------------
def derive_run_row(tables: dict[str, pd.DataFrame], manifest: dict,
                   recovery: pd.DataFrame, lo: float, hi: float) -> pd.DataFrame:
    scn = manifest["scenario"]
    preds = manifest["predictions"]
    comp = scn["composition"]
    wl = scn["workload"]
    coord = scn["coordinator"]

    tok = tables["token_done"]
    tok_s = tok[tok["in_steady_state"].fillna(False)] if len(tok) else tok
    hop = tables["hop"]
    hop_s = hop[hop["in_steady_state"].fillna(False)] if len(hop) else hop
    reqs = tables["request_end"]
    ns = tables["node_sample"]

    window = max(1e-9, hi - lo)
    lat = pd.to_numeric(tok_s["latency_s"], errors="coerce").dropna() if len(tok_s) \
        else pd.Series(dtype=float)

    tiers = manifest["plan"]["node_tiers"]
    tier_counts = {f"n_tier{t}": sum(1 for v in tiers.values() if v == t)
                   for t in sorted(set(tiers.values()))}

    busy = pd.to_numeric(ns["busy_frac"], errors="coerce").dropna() if len(ns) else \
        pd.Series(dtype=float)

    row = {
        "run_id": manifest["run_id"],
        # The image that ran, so a dataset spanning a rebuild is detectable
        # from the dataset itself rather than from memory.
        "image_id": (manifest.get("host", {}).get("image_synthetic")
                     or "").split("|")[0][:19],
        "experiment": manifest.get("experiment") or "",
        "scenario": scn["name"],
        "engine": manifest["engine"],
        "mode": manifest["mode"],
        "seed": scn["seed"],
        # -- what was asked for ------------------------------------------
        "depth_D": comp["depth"],
        "width_W": comp["width"],
        "n_nodes_N": preds["n_nodes"],
        "concurrency_G": wl["concurrency"],
        "placement": scn["placement"],
        "model_profile": scn["model_profile"],
        "prompt_tokens": wl["prompt_tokens"],
        "gen_tokens": wl["gen_tokens"],
        "duration_s": scn["duration_s"],
        "warmup_s": scn["warmup_s"],
        "heartbeat_ms": coord["heartbeat_ms"],
        "death_after_ms": coord["death_after_ms"],
        "churn_kind": scn["churn"]["kind"],
        "n_faults": manifest["churn"]["summary"]["n_fault_events"],
        "n_departures": manifest["churn"]["summary"]["n_departures"],
        # -- what the cost model predicted --------------------------------
        "pred_t_token_s": preds["t_token_s_mean"],
        "pred_x_max_tok_s": preds["x_max_tok_s"],
        "pred_x_of_g_tok_s": preds["x_of_g_tok_s"],
        "pred_x_pipe_tok_s": preds["x_pipe_tok_s"],
        "pred_x_delay_bound_tok_s": preds.get("x_delay_bound_tok_s"),
        "pred_x_capacity_tok_s": preds.get("x_capacity_tok_s"),
        "pred_x_capacity_compute_tok_s": preds.get("x_capacity_compute_tok_s"),
        "pred_x_closed_loop_tok_s": preds.get("x_closed_loop_tok_s"),
        "pred_t_token_s_report": preds.get("t_token_s_report"),
        "pred_entry_hop_s": preds.get("entry_hop_s"),
        "pred_slowest_stage": preds["slowest_stage"],
        # -- what was measured --------------------------------------------
        "steady_lo_s": lo,
        "steady_hi_s": hi,
        "steady_window_s": window,
        "tokens_total": int(len(tok)),
        "tokens_steady": int(len(tok_s)),
        "meas_throughput_tok_s": len(tok_s) / window if len(tok_s) else 0.0,
        "meas_t_token_s_mean": float(lat.mean()) if len(lat) else None,
        "meas_t_token_s_p50": float(lat.quantile(0.50)) if len(lat) else None,
        "meas_t_token_s_p95": float(lat.quantile(0.95)) if len(lat) else None,
        "meas_t_token_s_p99": float(lat.quantile(0.99)) if len(lat) else None,
        "requests_started": int(len(reqs)),
        "requests_ok": int((reqs["status"] == "ok").sum()) if len(reqs) else 0,
        "requests_failed": int((reqs["status"] == "failed").sum()) if len(reqs) else 0,
        "retries_total": int(pd.to_numeric(reqs["retries"], errors="coerce")
                             .fillna(0).sum()) if len(reqs) else 0,
        "failovers_total": int(pd.to_numeric(reqs["failovers"], errors="coerce")
                               .fillna(0).sum()) if len(reqs) else 0,
        "hop_errors": int(len(tables["hop_error"])),
        "mean_busy_frac": float(busy.mean()) if len(busy) else None,
        "detection_s_mean": float(pd.to_numeric(recovery["detection_s"],
                                                errors="coerce").mean())
        if len(recovery) else None,
        "recovery_s_mean": float(pd.to_numeric(recovery["recovery_s"],
                                               errors="coerce").mean())
        if len(recovery) else None,
        # -- model vs measurement ----------------------------------------
        "throughput_efficiency": ((len(tok_s) / window) / preds["x_max_tok_s"])
        if preds["x_max_tok_s"] else None,
        "latency_ratio_meas_over_pred": (float(lat.mean()) / preds["t_token_s_mean"])
        if len(lat) and preds["t_token_s_mean"] else None,
        # p50 is the statistic to compare against the cost model. A hop timeout
        # during a fault window puts multi-second outliers in the sample, and the
        # mean then reports the stalls rather than the steady-state latency the
        # model predicts. Both are kept; only one is a like-for-like comparison.
        "latency_ratio_p50_over_pred": (float(lat.quantile(0.50))
                                        / preds["t_token_s_mean"])
        if len(lat) and preds["t_token_s_mean"] else None,
    }

    # Throughput over the steady window with fault windows removed, so a churn
    # run still yields a clean-period rate alongside its degraded one.
    if len(tok_s) and "fault_window" in tok_s.columns:
        clean = tok_s[~tok_s["fault_window"].fillna(False)]
        fault_time = sum(max(0.0, min(hi, b) - max(lo, a))
                         for a, b in (manifest.get("_fault_windows") or []))
        clean_window = max(1e-9, window - fault_time)
        row["tokens_clean"] = int(len(clean))
        row["fault_window_s"] = fault_time
        row["meas_throughput_tok_s_nofault"] = len(clean) / clean_window
        lat_c = pd.to_numeric(clean["latency_s"], errors="coerce").dropna()
        row["meas_t_token_s_p50_nofault"] = (float(lat_c.quantile(0.50))
                                             if len(lat_c) else None)
        row["meas_t_token_s_mean_nofault"] = (float(lat_c.mean())
                                              if len(lat_c) else None)
        row["stalled_tokens"] = int((pd.to_numeric(tok_s["latency_s"],
                                                   errors="coerce")
                                     > 5 * float(lat.quantile(0.50))).sum()) \
            if len(lat) else 0
    row.update(tier_counts)

    # Per-stage means, one column group per stage: the straggler evidence.
    if len(hop_s):
        for stage, grp in hop_s.groupby("stage"):
            k = int(stage)
            row[f"stage{k}_compute_s_mean"] = float(
                pd.to_numeric(grp["compute_s"], errors="coerce").mean())
            row[f"stage{k}_queue_wait_s_mean"] = float(
                pd.to_numeric(grp["queue_wait_s"], errors="coerce").mean())
            row[f"stage{k}_net_s_mean"] = float(
                pd.to_numeric(grp["net_s"], errors="coerce").mean())
            row[f"stage{k}_hops"] = int(len(grp))

    for key, val in (manifest.get("sweep_point") or {}).items():
        row[f"sweep_{key}"] = val
    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def collect(run_dir: Path, write_parquet: bool = True) -> dict[str, int]:
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    tidy = run_dir / "tidy"
    tidy.mkdir(parents=True, exist_ok=True)

    raw = load_raw(run_dir)
    tables = split_tables(raw)

    # Order matters: recovery is derived from raw events, then fault windows are
    # defined by recovery, then rows are labelled, then the run row is summarised
    # from the labelled tables.
    tables["recovery"] = derive_recovery(tables, manifest)
    windows = fault_windows(tables["recovery"])
    manifest["_fault_windows"] = windows
    label_fault_window(tables, windows)
    lo, hi = steady_window(tables, manifest)
    label_steady(tables, lo, hi)

    tables["runs"] = derive_run_row(tables, manifest, tables["recovery"], lo, hi)

    counts: dict[str, int] = {}
    for name, frame in tables.items():
        if name in ("note",):
            continue
        frame.to_csv(tidy / f"{name}.csv", index=False)
        if write_parquet and len(frame):
            try:
                frame.to_parquet(tidy / f"{name}.parquet", index=False)
            except Exception:
                pass            # CSV is the source of truth; parquet is a convenience
        counts[name] = len(frame)

    (tidy / "_steady_window.json").write_text(
        json.dumps({"lo_sim_ts": lo, "hi_sim_ts": hi, "window_s": hi - lo},
                   indent=2), encoding="utf-8")
    return {k: v for k, v in counts.items() if v}
