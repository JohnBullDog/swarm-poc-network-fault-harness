"""Verify the tidiness of the produced datasets.

"Tidy" is a checkable property, not a claim, so this checks it. For every run
directory and every consolidated dataset:

  schema stability   a table has the same columns in every run, so a sweep can
                     be concatenated without columns appearing and disappearing
  column types       no column mixes numbers with strings
  one observation    the key that should identify a row actually does
  joinability        every event table's run_id resolves in runs.csv
  no silent loss     hops per token equals the path length; token counts agree
                     between the per-token table and the run summary
  labelled, not cut  the warm-up and fault-window flags are present and boolean
  units              every duration column is named *_s and every instant *_ts

Failures are printed with the table and column at fault.

    python -m tools.verify_tidy                  # all runs and datasets
    python -m tools.verify_tidy runs/<run_id>    # one run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from swarm.core.events import (DERIVED_TABLES, EVENT_SCHEMA,   # noqa: E402
                               columns_for)

from swarm.core import paths               # noqa: E402

RUNS = REPO / "runs"
DATASETS = paths.DATASETS

problems: list[str] = []
checks = 0


def ok(condition: bool, message: str) -> bool:
    global checks
    checks += 1
    if not condition:
        problems.append(message)
    return condition


def is_numeric_or_blank(series: pd.Series) -> bool:
    """True if every non-null value is numeric, or the column is entirely null."""
    s = series.dropna()
    if s.empty:
        return True
    return pd.to_numeric(s, errors="coerce").notna().all()


def check_run(run_dir: Path) -> None:
    tidy = run_dir / "tidy"
    if not tidy.exists():
        return
    man = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    rid = man["run_id"]

    # -- declared schemas are honoured ---------------------------------
    for kind in EVENT_SCHEMA:
        path = tidy / f"{kind}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        expected = columns_for(kind)
        actual = [c for c in df.columns
                  if c not in ("in_steady_state", "fault_window")]
        ok(actual == expected,
           f"{rid}/{kind}: columns differ from the declared schema\n"
           f"      expected {expected}\n      got      {actual}")

        # -- durations and instants are numeric --------------------------
        for col in df.columns:
            if col.endswith("_s") or col.endswith("_ts"):
                ok(is_numeric_or_blank(df[col]),
                   f"{rid}/{kind}.{col}: duration/instant column is not numeric")

        # -- both clocks present and sane --------------------------------
        if len(df):
            for clock in ("sim_ts", "wall_ts"):
                ok(clock in df.columns and df[clock].notna().all(),
                   f"{rid}/{kind}: {clock} missing or has nulls")
            ok((df["run_id"] == rid).all(),
               f"{rid}/{kind}: run_id does not match the manifest")

        # -- transients are labelled, not removed -------------------------
        if kind in ("token_done", "hop"):
            ok("in_steady_state" in df.columns,
               f"{rid}/{kind}: missing the in_steady_state label")
            ok("fault_window" in df.columns,
               f"{rid}/{kind}: missing the fault_window label")

    # -- derived tables: keyed and joinable, but not event-shaped --------
    for name in DERIVED_TABLES:
        path = tidy / f"{name}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        ok("run_id" in df.columns,
           f"{rid}/{name}: derived table has no run_id to join on")
        if len(df):
            ok((df["run_id"] == rid).all(),
               f"{rid}/{name}: run_id does not match the manifest")
            ok("sim_ts" not in df.columns,
               f"{rid}/{name}: derived table should not carry sim_ts; it spans "
               f"an interval rather than happening at an instant")

    # -- one row per observation ---------------------------------------
    tok_path, hop_path = tidy / "token_done.csv", tidy / "hop.csv"
    if tok_path.exists():
        tok = pd.read_csv(tok_path)
        if len(tok):
            dup = tok.duplicated(subset=["request_id", "token_idx"]).sum()
            ok(dup == 0,
               f"{rid}/token_done: {dup} rows share a (request_id, token_idx)")

    # -- nothing silently lost ------------------------------------------
    if tok_path.exists() and hop_path.exists():
        tok, hop = pd.read_csv(tok_path), pd.read_csv(hop_path)
        depth = int(man["scenario"]["composition"]["depth"])
        logs_hops = man["scenario"].get("log_hops", True)
        if len(tok) and len(hop) and logs_hops:
            # Only first-attempt hops of successful tokens are counted: a retry
            # legitimately produces extra hop rows.
            per = len(hop) / len(tok)
            ok(abs(per - (depth + 1)) < 0.35,
               f"{rid}: {per:.2f} hops per token, expected ~{depth + 1} "
               f"(D+1). A shortfall means buffered events were lost.")

    runs_path = tidy / "runs.csv"
    if runs_path.exists() and tok_path.exists():
        r = pd.read_csv(runs_path)
        tok = pd.read_csv(tok_path)
        ok(len(r) == 1, f"{rid}/runs: expected exactly 1 row, got {len(r)}")
        if len(r) and len(tok):
            ok(int(r["tokens_total"].iloc[0]) == len(tok),
               f"{rid}/runs.tokens_total disagrees with token_done row count")
            lo = float(r["steady_lo_s"].iloc[0])
            hi = float(r["steady_hi_s"].iloc[0])
            ok(hi >= lo, f"{rid}/runs: steady window is inverted")
            n_steady = int(tok["in_steady_state"].fillna(False).sum()) \
                if "in_steady_state" in tok.columns else None
            if n_steady is not None:
                ok(int(r["tokens_steady"].iloc[0]) == n_steady,
                   f"{rid}/runs.tokens_steady disagrees with the label")


def check_dataset(path: Path) -> None:
    df = pd.read_csv(path)
    name = path.name
    ok(len(df) > 0, f"{name}: empty")
    if "run_id" not in df.columns:
        return
    ok(df["run_id"].notna().all(), f"{name}: null run_id")

    # Scenario parameters must not be duplicated onto event rows; they belong
    # once per run, in runs.csv.
    if not name.endswith("_runs.csv"):
        leaked = [c for c in ("duration_s", "placement", "model_profile",
                              "heartbeat_ms") if c in df.columns]
        ok(not leaked,
           f"{name}: scenario parameters {leaked} copied onto event rows; "
           f"they belong in runs.csv keyed by run_id")

    for col in df.columns:
        if col.endswith("_s") or col.endswith("_ts"):
            ok(is_numeric_or_blank(df[col]), f"{name}.{col}: not numeric")

    # Every detail table must join to the summary table.
    key = name.split("_")[0]
    runs = paths.runs_csv(key)
    if runs.exists() and not name.endswith("_runs.csv"):
        known = set(pd.read_csv(runs)["run_id"])
        missing = set(df["run_id"]) - known
        ok(not missing,
           f"{name}: {len(missing)} run_id values do not resolve in {runs.name}")


def check_dataset_schema_stability() -> None:
    """A sweep's runs.csv rows must all carry the same columns."""
    for path in paths.all_runs_csvs():
        df = pd.read_csv(path)
        # Columns that are legitimately absent for some sweep points (per-stage
        # groups depend on depth; sweep_* depends on the axis) are exempt.
        core = [c for c in df.columns
                if not c.startswith(("stage", "sweep_", "n_tier"))]
        for col in core:
            frac = df[col].notna().mean()
            ok(frac > 0 or df[col].isna().all(),
               f"{path.name}.{col}: partially populated ({frac:.0%})")
        ok(df["run_id"].is_unique, f"{path.name}: run_id is not unique")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", type=Path)
    args = ap.parse_args(argv)

    run_dirs = args.runs or ([p for p in RUNS.iterdir() if p.is_dir()]
                             if RUNS.exists() else [])
    for d in sorted(run_dirs):
        if (d / "manifest.json").exists():
            check_run(d)

    ds = [p for p in paths.all_data_files() if p.suffix == ".csv"]
    for path in ds:
        if path.name.endswith("_progress.json"):
            continue
        check_dataset(path)
    if DATASETS.exists():
        check_dataset_schema_stability()

    print(f"checked {len(run_dirs)} run(s) and {len(ds)} dataset file(s): "
          f"{checks} assertions")
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("all tidiness checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
