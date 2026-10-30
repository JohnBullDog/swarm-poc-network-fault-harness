"""Build the master results workbook from the consolidated datasets.

One file a reader can open and navigate without knowing the CSV layout:

    Summary          headline results, computed by formula from the data sheets
    exp1 Concurrency  throughput and latency vs G, with the bound ratios
    exp2 Delay        latency vs per-hop delay, with SLOPE() per depth
    exp4 Failover     detection, recovery and failures vs W / heartbeat / timeout
    exp7 Mode check   real vs synthetic, both arms
    exp1s Report scale  N=1000 virtual sweep
    Recovery          one row per injected fault
    Engine comparison virtual vs container, per axis
    All runs          every run, every column, for pivoting
    Data dictionary   what each column means

Derived figures are written as Excel formulas rather than values, so the
workbook recalculates if a reader edits or extends the data, and so the
arithmetic behind every headline number is visible in the cell.

    python -m tools.build_workbook
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DATASETS = REPO / "datasets"
OUT = DATASETS / "swarm_poc_results.xlsx"

FONT = "Arial"
HEAD_FILL = PatternFill("solid", fgColor="1F3864")
HEAD_FONT = Font(name=FONT, size=10, bold=True, color="FFFFFF")
BODY_FONT = Font(name=FONT, size=10)
TITLE_FONT = Font(name=FONT, size=14, bold=True, color="1F3864")
NOTE_FONT = Font(name=FONT, size=9, italic=True, color="555555")
FORMULA_FONT = Font(name=FONT, size=10)          # black: computed
INPUT_FONT = Font(name=FONT, size=10, color="0000FF")   # blue: from data
THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(bottom=THIN)

# Columns worth putting in front of a reader, per experiment. The full set
# lives on "All runs".
CORE = ["run_id", "scenario", "engine", "mode", "depth_D", "width_W",
        "n_nodes_N", "concurrency_G", "placement", "model_profile",
        "duration_s", "tokens_steady"]
PRED = ["pred_t_token_s_report", "pred_t_token_s", "pred_x_max_tok_s",
        "pred_x_of_g_tok_s", "pred_x_delay_bound_tok_s",
        "pred_x_capacity_compute_tok_s", "pred_x_closed_loop_tok_s"]
MEAS = ["meas_throughput_tok_s", "meas_t_token_s_p50", "meas_t_token_s_p95",
        "meas_t_token_s_mean", "mean_busy_frac", "requests_ok",
        "requests_failed", "retries_total", "failovers_total",
        "throughput_efficiency", "latency_ratio_p50_over_pred"]


def _style_header(ws, row: int, ncols: int) -> None:
    for c in range(1, ncols + 1):
        cell = ws.cell(row=row, column=c)
        cell.fill = HEAD_FILL
        cell.font = HEAD_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center",
                                   wrap_text=True)
        cell.border = BORDER
    ws.freeze_panes = ws.cell(row=row + 1, column=1)


def _autosize(ws, max_width: int = 30) -> None:
    for col in ws.columns:
        letter = get_column_letter(col[0].column)
        width = max((len(str(c.value)) for c in col if c.value is not None),
                    default=8)
        ws.column_dimensions[letter].width = min(max(10, width + 2), max_width)


def write_frame(ws, df: pd.DataFrame, start_row: int = 1,
                number_formats: dict[str, str] | None = None) -> int:
    """Write a frame with a styled header. Returns the first data row."""
    number_formats = number_formats or {}
    for j, name in enumerate(df.columns, start=1):
        ws.cell(row=start_row, column=j, value=name)
    _style_header(ws, start_row, len(df.columns))

    for i, (_, rec) in enumerate(df.iterrows(), start=start_row + 1):
        for j, name in enumerate(df.columns, start=1):
            val = rec[name]
            if pd.isna(val):
                val = None
            elif hasattr(val, "item"):
                val = val.item()
            cell = ws.cell(row=i, column=j, value=val)
            cell.font = INPUT_FONT if isinstance(val, (int, float)) else BODY_FONT
            fmt = number_formats.get(name)
            if fmt and isinstance(val, (int, float)):
                cell.number_format = fmt
    return start_row + 1


def add_title(ws, title: str, note: str = "") -> int:
    ws["A1"] = title
    ws["A1"].font = TITLE_FONT
    row = 2
    if note:
        ws["A2"] = note
        ws["A2"].font = NOTE_FONT
        row = 3
    return row + 1


def load(name: str) -> pd.DataFrame | None:
    path = DATASETS / name
    return pd.read_csv(path) if path.exists() else None


# ---------------------------------------------------------------------------
def sheet_exp1(wb: Workbook) -> dict:
    d = load("exp1_runs.csv")
    if d is None:
        return {}
    d = d.sort_values("concurrency_G")
    cols = ["concurrency_G", "n_nodes_N", "meas_throughput_tok_s",
            "pred_x_of_g_tok_s", "pred_x_delay_bound_tok_s",
            "pred_x_capacity_compute_tok_s", "pred_x_closed_loop_tok_s",
            "meas_t_token_s_p50", "meas_t_token_s_p95", "pred_t_token_s",
            "mean_busy_frac", "requests_ok", "requests_failed", "run_id"]
    view = d[cols]

    ws = wb.create_sheet("exp1 Concurrency")
    r0 = add_title(
        ws, "Experiment 1 - Concurrency sweep",
        "D=3, W=2, N=6 tier-2 nodes, poc_small, no churn. Blue = measured or "
        "predicted from the dataset; black = computed in-sheet.")
    first = write_frame(ws, view, r0, {
        "meas_throughput_tok_s": "0.00", "pred_x_of_g_tok_s": "0.0",
        "pred_x_delay_bound_tok_s": "0.0",
        "pred_x_capacity_compute_tok_s": "0.0",
        "pred_x_closed_loop_tok_s": "0.0",
        "meas_t_token_s_p50": "0.000", "meas_t_token_s_p95": "0.000",
        "pred_t_token_s": "0.000", "mean_busy_frac": "0.000"})
    last = first + len(view) - 1

    # Derived ratios as formulas, so the arithmetic is visible and live.
    n = len(view.columns)
    for j, head in enumerate(["X / ramp", "X / closed-loop", "p50 (ms)"],
                             start=n + 1):
        ws.cell(row=first - 1, column=j, value=head)
    _style_header(ws, first - 1, n + 3)
    for i in range(first, last + 1):
        ws.cell(row=i, column=n + 1, value=f"=C{i}/D{i}").number_format = "0.000"
        ws.cell(row=i, column=n + 2, value=f"=C{i}/G{i}").number_format = "0.000"
        ws.cell(row=i, column=n + 3, value=f"=H{i}*1000").number_format = "0.0"
        for j in (n + 1, n + 2, n + 3):
            ws.cell(row=i, column=j).font = FORMULA_FONT
    _autosize(ws)
    return {"first": first, "last": last, "ratio_col": get_column_letter(n + 2)}


def sheet_exp2(wb: Workbook) -> dict:
    d = load("exp2_runs.csv")
    if d is None:
        return {}
    d = d.sort_values(["depth_D", "sweep_delay_ms"])
    cols = ["depth_D", "sweep_delay_ms", "meas_t_token_s_p50", "pred_t_token_s",
            "pred_t_token_s_report", "meas_throughput_tok_s",
            "latency_ratio_p50_over_pred", "tokens_steady", "run_id"]
    view = d[cols]

    ws = wb.create_sheet("exp2 Delay")
    r0 = add_title(
        ws, "Experiment 2 - Delay sweep",
        "W=1, G=1 so no queueing confounds the hop term. The slope of latency "
        "against per-hop delay should equal D: only the tier-2 link is swept, "
        "and tier-2 nodes send D of the path's D+1 hops.")
    first = write_frame(ws, view, r0, {
        "meas_t_token_s_p50": "0.0000", "pred_t_token_s": "0.0000",
        "pred_t_token_s_report": "0.0000", "meas_throughput_tok_s": "0.00",
        "latency_ratio_p50_over_pred": "0.000"})
    last = first + len(view) - 1

    # p50 in ms, so SLOPE reads directly as "ms of latency per ms of hop delay".
    n = len(view.columns)
    ws.cell(row=first - 1, column=n + 1, value="p50 (ms)")
    _style_header(ws, first - 1, n + 1)
    for i in range(first, last + 1):
        c = ws.cell(row=i, column=n + 1, value=f"=C{i}*1000")
        c.number_format = "0.00"
        c.font = FORMULA_FONT
    ms = get_column_letter(n + 1)

    blocks = []
    row = last + 3
    ws.cell(row=row, column=1, value="Measured slope per depth (SLOPE of p50 in ms against delay in ms)")
    ws.cell(row=row, column=1).font = Font(name=FONT, size=11, bold=True)
    row += 1
    for j, head in enumerate(["depth D", "rows", "measured slope",
                              "predicted slope (= D)", "error %", "|error| %",
                              "R2"], start=1):
        ws.cell(row=row, column=j, value=head)
    _style_header(ws, row, 7)
    hdr = row
    row += 1
    for depth, grp in view.groupby("depth_D"):
        idx = [first + view.index.get_loc(i) for i in grp.index]
        lo, hi = min(idx), max(idx)
        ws.cell(row=row, column=1, value=int(depth)).font = INPUT_FONT
        ws.cell(row=row, column=2, value=f"{lo}:{hi}").font = BODY_FONT
        ws.cell(row=row, column=3,
                value=f"=SLOPE({ms}{lo}:{ms}{hi},B{lo}:B{hi})").number_format = "0.0000"
        ws.cell(row=row, column=4, value=f"=A{row}").number_format = "0"
        ws.cell(row=row, column=5,
                value=f"=(C{row}-D{row})/D{row}").number_format = "0.00%"
        ws.cell(row=row, column=6, value=f"=ABS(E{row})").number_format = "0.00%"
        ws.cell(row=row, column=7,
                value=f"=RSQ({ms}{lo}:{ms}{hi},B{lo}:B{hi})").number_format = "0.000000"
        for j in (3, 4, 5, 6, 7):
            ws.cell(row=row, column=j).font = FORMULA_FONT
        blocks.append(row)
        row += 1
    _autosize(ws)
    return {"slope_rows": blocks, "hdr": hdr}


def sheet_exp4(wb: Workbook) -> None:
    d = load("exp4_runs.csv")
    rec = load("exp4_recovery.csv")
    if d is None:
        return
    cols = ["sweep_arm", "width_W", "heartbeat_ms", "death_after_ms",
            "sweep_hop_timeout_ms", "concurrency_G", "requests_ok",
            "requests_failed", "retries_total", "detection_s_mean",
            "recovery_s_mean", "meas_throughput_tok_s", "run_id"]
    view = d[[c for c in cols if c in d.columns]].sort_values(
        ["sweep_arm", "width_W", "heartbeat_ms"])
    ws = wb.create_sheet("exp4 Failover")
    r0 = add_title(
        ws, "Experiment 4 - Disconnection and failover",
        "One kill of a middle-stage replica at t=20 s. death_after = 4 x "
        "heartbeat. Arm W_x_heartbeat varies replica count and heartbeat; arm "
        "hop_timeout varies the client timeout at W=2, heartbeat 500 ms. "
        "A blank recovery means the stage never recovered (W=1: no replica).")
    write_frame(ws, view, r0, {
        "detection_s_mean": "0.000", "recovery_s_mean": "0.000",
        "meas_throughput_tok_s": "0.00"})
    _autosize(ws)

    if rec is not None and len(rec):
        ws2 = wb.create_sheet("Recovery per fault")
        r0 = add_title(
            ws2, "Recovery - one row per injected fault",
            "detection_s: fault to the coordinator marking the node dead. "
            "recovery_s: fault to the first successful token routed around it "
            "AFTER detection. reroute_s: first token avoiding it at all, which "
            "under round-robin is near-instant and says little. stall_s: fault "
            "to the last error naming the node.")
        write_frame(ws2, rec, r0, {
            "detection_s": "0.000", "recovery_s": "0.000",
            "reroute_s": "0.000", "stall_s": "0.000",
            "fault_sim_ts": "0.000", "detected_sim_ts": "0.000",
            "first_ok_sim_ts": "0.000"})
        _autosize(ws2)


def sheet_exp7(wb: Workbook) -> dict:
    d = load("exp7_mode_check.csv")
    if d is None:
        return {}
    d = d.sort_values(["arm", "concurrency_G"]).reset_index(drop=True)
    ws = wb.create_sheet("exp7 Mode check")
    r0 = add_title(
        ws, "Experiment 7 - Mode check: does synthetic timing track real?",
        "Qwen2.5-0.5B split 8/8/8 layers over three tier-2 containers, CPU "
        "float32, greedy decoding. 'nominal' uses the tier-2 reference "
        "bandwidth of 45 GB/s; 'calibrated' uses B_eff fitted from the real "
        "runs. This experiment is what licenses every synthetic result.")
    first = write_frame(ws, d, r0, {
        "real_p50_ms": "0.0", "synthetic_p50_ms": "0.0",
        "p50_rel_err": "0.0%", "throughput_rel_err": "0.0%",
        "real_throughput_tok_s": "0.00",
        "synthetic_throughput_tok_s": "0.00", "b_eff_gbs": "0.00"})
    last = first + len(d) - 1

    n = len(d.columns)
    ws.cell(row=first - 1, column=n + 1, value="|p50 error|")
    _style_header(ws, first - 1, n + 1)
    err_col = get_column_letter(d.columns.get_loc("p50_rel_err") + 1)
    for i in range(first, last + 1):
        c = ws.cell(row=i, column=n + 1, value=f"=ABS({err_col}{i})")
        c.number_format = "0.0%"
        c.font = FORMULA_FONT
    _autosize(ws)

    # Each arm is now a contiguous block, so the summary can take a plain
    # MEDIAN over a range instead of an array-entered MEDIAN(IF(...)), which
    # openpyxl cannot mark as an array formula and Excel would mis-evaluate.
    blocks = {}
    for arm, grp in d.groupby("arm"):
        idx = [first + i for i in grp.index]
        blocks[arm] = (min(idx), max(idx))
    return {"first": first, "last": last,
            "abs_col": get_column_letter(n + 1), "blocks": blocks,
            "beff_col": get_column_letter(d.columns.get_loc("b_eff_gbs") + 1)}


def sheet_exp1s(wb: Workbook) -> None:
    d = load("exp1s_runs.csv")
    if d is None:
        return
    d = d.sort_values("concurrency_G")
    cols = ["concurrency_G", "n_nodes_N", "depth_D", "width_W",
            "meas_throughput_tok_s", "pred_x_of_g_tok_s",
            "pred_x_closed_loop_tok_s", "pred_x_max_tok_s",
            "meas_t_token_s_p50", "pred_t_token_s", "throughput_efficiency",
            "tokens_steady", "run_id"]
    ws = wb.create_sheet("exp1s Report scale")
    r0 = add_title(
        ws, "Experiment 1s - Concurrency at report scale (virtual engine)",
        "Reconstruction of the research report's appendix example A: 1000 "
        "Pi-class nodes, 1T-param MoE at INT4 with 60B active, D=50, W=20. "
        "Virtual engine only - 1000 nodes is ~16x what this host runs as "
        "containers. Trustworthy only on the axes the engine comparison "
        "validates.")
    first = write_frame(ws, d[cols], r0, {
        "meas_throughput_tok_s": "0.0", "pred_x_of_g_tok_s": "0.0",
        "pred_x_closed_loop_tok_s": "0.0", "pred_x_max_tok_s": "0.0",
        "meas_t_token_s_p50": "0.000", "pred_t_token_s": "0.000",
        "throughput_efficiency": "0.000"})
    last = first + len(d) - 1
    n = len(cols)
    ws.cell(row=first - 1, column=n + 1, value="X / ramp")
    _style_header(ws, first - 1, n + 1)
    for i in range(first, last + 1):
        c = ws.cell(row=i, column=n + 1, value=f"=E{i}/F{i}")
        c.number_format = "0.000"
        c.font = FORMULA_FONT
    _autosize(ws)


def sheet_engines(wb: Workbook) -> None:
    frames = []
    for key in ("exp1", "exp2"):
        f = load(f"engine_comparison_{key}.csv")
        if f is not None:
            f = f.copy()
            f.insert(0, "experiment", key)
            frames.append(f)
    if not frames:
        return
    d = pd.concat(frames, ignore_index=True, sort=False)
    ws = wb.create_sheet("Engine comparison")
    r0 = add_title(
        ws, "Engine agreement - virtual against containers",
        "The harness's own measured overhead is fitted per configuration and "
        "given to the virtual engine, so what remains is attributable to the "
        "engines' models rather than to containerisation. Queue wait is "
        "compared as a MEAN: the distributions have the same mass but "
        "different shapes, so their medians legitimately differ.")
    write_frame(ws, d, r0, {
        "t_token_p50_ms_container": "0.00", "t_token_p50_ms_virtual": "0.00",
        "t_token_p50_ms_rel_err": "0.0%",
        "throughput_tok_s_container": "0.00",
        "throughput_tok_s_virtual": "0.00",
        "throughput_tok_s_rel_err": "0.0%",
        "stage_compute_ms_rel_err": "0.0%", "stage_net_ms_rel_err": "0.0%",
        "stage_queue_ms_rel_err": "0.0%"})
    _autosize(ws)


def sheet_all_runs(wb: Workbook) -> None:
    frames = []
    for key in ("exp1", "exp2", "exp4", "exp1s"):
        f = load(f"{key}_runs.csv")
        if f is not None:
            frames.append(f)
    if not frames:
        return
    d = pd.concat(frames, ignore_index=True, sort=False)
    lead = [c for c in CORE + PRED + MEAS if c in d.columns]
    rest = [c for c in d.columns if c not in lead]
    d = d[lead + rest]
    ws = wb.create_sheet("All runs")
    r0 = add_title(
        ws, "All runs - every experiment, every column",
        "One row per run, keyed by run_id. Every event table in datasets/ "
        "joins to this on run_id. Blank cells are columns that do not apply to "
        "that experiment (per-stage groups depend on depth; sweep_* on the "
        "axis varied).")
    write_frame(ws, d, r0)
    _autosize(ws, max_width=24)


DICTIONARY = [
    ("run_id", "key", "Unique per run. Join every event table on this."),
    ("engine", "setup", "container (real clock, real TCP via tc netem) or virtual (simulated clock)."),
    ("mode", "setup", "synthetic (worker sleeps the cost model's time) or real (Qwen2.5-0.5B layers on CPU)."),
    ("depth_D / width_W / n_nodes_N", "setup", "Pipeline stages, replicas per stage, and N = D x W."),
    ("concurrency_G", "setup", "Concurrent agent streams held in flight (closed loop)."),
    ("pred_t_token_s_report", "prediction", "The report's expression: (P_a*b + s*q)/B + D*(l + a/beta). No entry hop."),
    ("pred_t_token_s", "prediction", "The full measurable path: the above plus the gateway's entry hop. No fitted fudge factor."),
    ("pred_x_max_tok_s", "prediction", "Peak throughput, sum(B_i) / (P_a*b + s*q)."),
    ("pred_x_of_g_tok_s", "prediction", "The report's utilisation ramp, X_max * min(1, G/N). An upper bound."),
    ("pred_x_delay_bound_tok_s", "prediction", "G / T_token. Binds below the knee."),
    ("pred_x_capacity_compute_tok_s", "prediction", "W / max_k(compute_k). Binds above the knee. Compute only: transit overlaps with serving the next token."),
    ("pred_x_closed_loop_tok_s", "prediction", "min(delay bound, capacity bound). The quantity measurement actually tracks."),
    ("meas_throughput_tok_s", "measurement", "Steady-state tokens per second."),
    ("meas_t_token_s_p50 / _p95 / _mean", "measurement", "Per-token latency. Compare the model against p50: the mean carries fault-window stalls."),
    ("mean_busy_frac", "measurement", "Fraction of the sample window a node's servers were busy."),
    ("throughput_efficiency", "measurement", "Measured throughput over X_max."),
    ("latency_ratio_p50_over_pred", "measurement", "Measured p50 over predicted. This is the harness's own cost, not a model error."),
    ("detection_s", "failover", "Fault to the coordinator marking the node dead. Set by the heartbeat interval."),
    ("recovery_s", "failover", "Fault to the first successful token routed around the loss AFTER detection. Set by the client hop timeout."),
    ("reroute_s", "failover", "Fault to the first token using that stage on another replica. Near-instant under round-robin; says little alone."),
    ("stall_s", "failover", "Fault to the last error naming the node."),
    ("in_steady_state", "label", "On event tables: row falls after run_start + warmup. Nothing is dropped, only labelled."),
    ("fault_window", "label", "On event tables: row falls between a fault and its recovery."),
    ("*_s / *_ts", "convention", "_s is a duration in seconds; _ts is an instant. sim_ts is simulated time, wall_ts is unix time."),
]


def sheet_dictionary(wb: Workbook) -> None:
    ws = wb.create_sheet("Data dictionary")
    r0 = add_title(
        ws, "Data dictionary",
        "Full schema, including the per-event tables not in this workbook, is "
        "in docs/data-dictionary.md.")
    df = pd.DataFrame(DICTIONARY, columns=["column", "kind", "meaning"])
    write_frame(ws, df, r0)
    ws.column_dimensions["A"].width = 34
    ws.column_dimensions["B"].width = 14
    ws.column_dimensions["C"].width = 110
    for row in ws.iter_rows(min_row=r0 + 1):
        row[2].alignment = Alignment(wrap_text=True, vertical="top")


def sheet_summary(wb: Workbook, e1: dict, e2: dict, e7: dict) -> None:
    ws = wb.create_sheet("Summary", 0)
    ws["A1"] = "Swarm PoC - network fault harness: results summary"
    ws["A1"].font = Font(name=FONT, size=16, bold=True, color="1F3864")
    ws["A2"] = ("Every figure below is a formula over the data sheets, not a "
                "typed-in number. Full write-up: docs/findings.md")
    ws["A2"].font = NOTE_FONT

    row = 4
    for j, head in enumerate(["Result", "Value", "Where from"], start=1):
        ws.cell(row=row, column=j, value=head)
    _style_header(ws, row, 3)
    row += 1

    def put(label: str, formula: str, fmt: str, source: str) -> None:
        nonlocal row
        ws.cell(row=row, column=1, value=label).font = BODY_FONT
        c = ws.cell(row=row, column=2, value=formula)
        c.number_format = fmt
        c.font = FORMULA_FONT
        ws.cell(row=row, column=3, value=source).font = NOTE_FONT
        row += 1

    if e1:
        rc, f, l = e1["ratio_col"], e1["first"], e1["last"]
        put("Concurrency: mean measured / closed-loop bound",
            f"=AVERAGE('exp1 Concurrency'!{rc}{f}:{rc}{l})", "0.000",
            "exp1. The bound tracks measurement across a 32x range of G.")
        put("Concurrency: worst measured / closed-loop bound",
            f"=MIN('exp1 Concurrency'!{rc}{f}:{rc}{l})", "0.000", "exp1")
        put("Concurrency: total failed requests",
            f"=SUM('exp1 Concurrency'!M{f}:M{l})", "0",
            "exp1. Zero across the whole sweep.")
        put("Concurrency: peak measured throughput (tok/s)",
            f"=MAX('exp1 Concurrency'!C{f}:C{l})", "0.00",
            "exp1. Against a capacity bound in column F.")
    if e2 and e2.get("slope_rows"):
        rows = e2["slope_rows"]
        lo, hi = min(rows), max(rows)
        put("Delay: worst |slope error| vs predicted D",
            f"=MAX('exp2 Delay'!F{lo}:F{hi})", "0.00%",
            "exp2, over the per-depth |error| column.")
        put("Delay: lowest R2 of the linear fit",
            f"=MIN('exp2 Delay'!G{lo}:G{hi})", "0.000000",
            "exp2. Latency is linear in per-hop delay with slope D.")
    if e7:
        ac, blocks = e7["abs_col"], e7["blocks"]
        for arm, label, note in [
            ("synthetic_nominal",
             "Mode check: median |error|, synthetic on nominal bandwidth",
             "exp7. Tier reference bandwidths are ratios, not achievable values."),
            ("synthetic_calibrated",
             "Mode check: median |error|, synthetic on fitted B_eff",
             "exp7. Good at G=1, degrades with concurrency: no CPU contention."),
        ]:
            if arm in blocks:
                lo2, hi2 = blocks[arm]
                put(label, f"=MEDIAN('exp7 Mode check'!{ac}{lo2}:{ac}{hi2})",
                    "0.0%", note)
        put("Mode check: fitted effective bandwidth (GB/s)",
            f"=MAX('exp7 Mode check'!{e7['beff_col']}{e7['first']}:"
            f"{e7['beff_col']}{e7['last']})", "0.00",
            "exp7. Against a tier-2 nominal of 45 GB/s.")

    row += 1
    ws.cell(row=row, column=1,
            value="Read these with the limits in docs/findings.md").font = \
        Font(name=FONT, size=11, bold=True)
    row += 1
    for line in [
        "One host: all containers share its CPU and memory bandwidth; runs are "
        "therefore sequential.",
        "Tier emulation preserves ratios, not absolute device speeds - exp7 "
        "measures the gap as 6.5x for tier 2.",
        "Synthetic mode models neither CPU contention nor prefill cost, so it "
        "over-predicts throughput by 1.6-2.2x even calibrated.",
        "Virtual-engine results are valid for latency, compute, network and "
        "aggregate queueing; not for queue-wait distribution or tail latency.",
        "Predictions carry no fitted correction: the measured/predicted ratio "
        "is the harness's own cost, reported rather than subtracted out.",
    ]:
        ws.cell(row=row, column=1, value="- " + line).font = NOTE_FONT
        row += 1

    ws.column_dimensions["A"].width = 56
    ws.column_dimensions["B"].width = 16
    ws.column_dimensions["C"].width = 88
    for r in range(5, row):
        ws.cell(row=r, column=3).alignment = Alignment(wrap_text=True,
                                                       vertical="top")


def main() -> int:
    wb = Workbook()
    wb.remove(wb.active)
    e1 = sheet_exp1(wb)
    e2 = sheet_exp2(wb)
    sheet_exp4(wb)
    e7 = sheet_exp7(wb)
    sheet_exp1s(wb)
    sheet_engines(wb)
    sheet_all_runs(wb)
    sheet_dictionary(wb)
    sheet_summary(wb, e1, e2, e7)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUT)
    print(f"wrote {OUT}")
    print("sheets: " + ", ".join(wb.sheetnames))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
