"""Calibration layer: does the virtual engine agree with the containers?

This is the gate on every virtual-engine result. The virtual engine exists to
reach N = 1000, which this host cannot host as containers; but a simulated
number is worth only as much as its agreement with a measured one at a scale
where both can run. So the same scenarios are run on both engines and compared
point by point.

What is compared, and why those:

  t_token p50        the quantity the cost model predicts. p50, not mean, so a
                     single stall does not decide the comparison.
  throughput         the quantity the concurrency sweep is about.
  stage compute      whether the service-time model is right.
  stage net          whether the link model is right. This is where the two
                     engines are most likely to diverge, because the virtual
                     link is an analytic delay while the container link is real
                     TCP over a bridge.
  queue wait         whether contention is modelled, which is what decides
                     behaviour near saturation.

A disagreement is reported, not reconciled. If the engines differ by more than
the tolerance on some axis, the honest conclusion is that virtual results are
untrustworthy on that axis, and the write-up says so.

Usage:
    python -m tools.compare_engines exp1            # run both engines, compare
    python -m tools.compare_engines exp1 --reuse    # reuse existing container runs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from swarm.core.scenario import SyntheticCfg           # noqa: E402
from swarm.runner.experiments import EXPERIMENTS       # noqa: E402
from swarm.virtual.run import run_virtual              # noqa: E402

from tools.calibrate import fit_one                    # noqa: E402

DATASETS = REPO / "datasets"
RUNS = REPO / "runs"

# Tolerances. Latency and throughput are the headline quantities and are held to
# 10%; per-stage network time is allowed 25% because the virtual link is an
# analytic delay and the container link is real TCP over a Docker bridge, and
# queue wait is allowed 50% because it is a difference of two small numbers near
# an empty queue and its relative error explodes there.
TOLERANCE = {
    "t_token_p50_ms": 0.10,
    "throughput_tok_s": 0.10,
    "stage_compute_ms": 0.10,
    "stage_net_ms": 0.25,
    # Queue wait is a mean of a saturation-dependent quantity, and is near zero
    # below the knee where its relative error is meaningless; 30% on the mean.
    "stage_queue_ms": 0.30,
}


def summarise_run(run_dir: Path) -> dict | None:
    tidy = run_dir / "tidy"
    if not (tidy / "runs.csv").exists():
        return None
    row = pd.read_csv(tidy / "runs.csv").iloc[0]
    hop = pd.read_csv(tidy / "hop.csv") if (tidy / "hop.csv").exists() \
        else pd.DataFrame()
    if len(hop):
        if "in_steady_state" in hop.columns:
            hop = hop[hop["in_steady_state"] == True]      # noqa: E712
        if "fault_window" in hop.columns:
            hop = hop[hop["fault_window"] != True]         # noqa: E712
        depth = int(row["depth_D"])
        stages = hop[pd.to_numeric(hop["stage"], errors="coerce") < depth]
        # Compute and network use the median: both are tight and unimodal, and
        # the median ignores real mode's slow prefill tokens.
        #
        # Queue wait uses the MEAN. Comparing medians here was wrong and made
        # the engines look irreconcilable (0.004 ms against 22.3 ms, a 100%
        # error). The two distributions have the same mass but different shapes:
        # the virtual engine's service and arrivals are both deterministic, so
        # at saturation exactly a third of hops wait ~86 ms and the rest wait
        # nothing -- a D/D/1 pattern -- while container jitter smears the same
        # total across every hop. Queue wait composes additively into latency,
        # so the mean is the quantity that has to agree, and it does.
        compute = pd.to_numeric(stages["compute_s"], errors="coerce").median() * 1000
        # Network uses the mean, not the median, for the same reason as queue
        # wait but a different mechanism: the hops pooled here have *different*
        # link delays (the gateway's entry hop against the stages' own), so the
        # sample is bimodal and its median sits on whichever mode happens to
        # have one more row. At D=2 that put the two engines 50% apart while
        # their per-stage times agreed to 1.6%. The mean is stable for a mixture.
        net = pd.to_numeric(stages["net_s"], errors="coerce").mean() * 1000
        queue = pd.to_numeric(stages["queue_wait_s"], errors="coerce").mean() * 1000
        queue_med = pd.to_numeric(stages["queue_wait_s"],
                                  errors="coerce").median() * 1000
    else:
        compute = net = queue = queue_med = float("nan")

    return {
        "run_id": row["run_id"],
        "scenario": row["scenario"],
        "engine": row["engine"],
        "t_token_p50_ms": float(row["meas_t_token_s_p50"]) * 1000
        if pd.notna(row["meas_t_token_s_p50"]) else float("nan"),
        "throughput_tok_s": float(row["meas_throughput_tok_s"]),
        "stage_compute_ms": compute,
        "stage_net_ms": net,
        "stage_queue_ms": queue,
        "stage_queue_median_ms": queue_med,
        "tokens": int(row["tokens_steady"]),
        "pred_t_token_ms": float(row["pred_t_token_s"]) * 1000,
        "pred_closed_loop_tok_s": float(row["pred_x_closed_loop_tok_s"])
        if pd.notna(row.get("pred_x_closed_loop_tok_s")) else float("nan"),
    }


def find_container_runs(scenario_names: set[str]) -> dict[str, Path]:
    """Most recent container run directory per scenario name."""
    out: dict[str, Path] = {}
    for d in sorted(RUNS.iterdir()):
        man = d / "manifest.json"
        if not (d.is_dir() and man.exists()):
            continue
        try:
            m = json.loads(man.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if m.get("engine") != "container":
            continue
        name = m.get("scenario", {}).get("name")
        if name in scenario_names:
            out[name] = d          # sorted order means the last wins
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", choices=sorted(EXPERIMENTS))
    ap.add_argument("--reuse", action="store_true",
                    help="reuse existing container runs instead of requiring fresh")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    exp = EXPERIMENTS[args.experiment]
    scenarios = [s for s in exp.build() if s.mode != "real"]
    names = {s.name for s in scenarios}

    container = find_container_runs(names)
    missing = names - set(container)
    if missing and not args.reuse:
        print(f"no container runs for {len(missing)} scenarios; "
              f"run `python -m swarm.runner.drive {args.experiment}` first")
        return 1
    if missing:
        print(f"note: {len(missing)} scenarios have no container run and are skipped")

    rows = []
    for scn in scenarios:
        if scn.name not in container:
            continue
        run_dir = container[scn.name]
        c = summarise_run(run_dir)
        if c is None:
            continue

        # Give the virtual engine this configuration's measured container
        # overhead, fitted from this very run.
        #
        # The question here is whether the *engine* reproduces the containers,
        # not whether the cost model does. The containers carry a real runtime
        # cost -- event-loop scheduling, TCP, the Docker bridge -- that the
        # virtual engine has no equivalent of, and that cost is load-dependent
        # rather than a host constant, so it cannot be set once globally. Fitting
        # it per configuration is what makes the remaining difference
        # attributable to the engines' models rather than to the harness.
        fit = fit_one(run_dir) or {}
        stage_ms = fit.get("stage_overhead_ms") or 0.0
        hop_ms = fit.get("hop_overhead_ms") or 0.0
        vscn = scn.with_(
            engine="virtual",
            synthetic=SyntheticCfg(
                mem_bw_gbs=dict(scn.synthetic.mem_bw_gbs),
                servers_per_node=scn.synthetic.servers_per_node,
                busy_compute=False,
                stage_overhead_ms=stage_ms,
                hop_overhead_ms=hop_ms))
        vd = run_virtual(vscn, tag="cmp", quiet=True)
        v = summarise_run(vd)
        if v is None:
            continue

        row = {"scenario": scn.name,
               "concurrency_G": scn.workload.concurrency,
               "depth_D": scn.composition.depth,
               "width_W": scn.composition.width,
               "container_run": c["run_id"], "virtual_run": v["run_id"],
               "tokens_container": c["tokens"], "tokens_virtual": v["tokens"],
               "fitted_stage_overhead_ms": stage_ms,
               "fitted_hop_overhead_ms": hop_ms,
               "stage_queue_median_ms_container": c["stage_queue_median_ms"],
               "stage_queue_median_ms_virtual": v["stage_queue_median_ms"]}
        for key in TOLERANCE:
            cv, vv = c[key], v[key]
            row[f"{key}_container"] = cv
            row[f"{key}_virtual"] = vv
            rel = (vv - cv) / cv if (cv and cv == cv and vv == vv) else float("nan")
            row[f"{key}_rel_err"] = rel
            row[f"{key}_within_tol"] = (abs(rel) <= TOLERANCE[key]) \
                if rel == rel else None
        rows.append(row)
        print(f"  {scn.name:34s} "
              f"p50 {row['t_token_p50_ms_container']:7.2f} / "
              f"{row['t_token_p50_ms_virtual']:7.2f} ms "
              f"({row['t_token_p50_ms_rel_err']*100:+6.1f}%)   "
              f"X {row['throughput_tok_s_container']:6.2f} / "
              f"{row['throughput_tok_s_virtual']:6.2f} tok/s "
              f"({row['throughput_tok_s_rel_err']*100:+6.1f}%)")

    if not rows:
        print("nothing to compare")
        return 1

    df = pd.DataFrame(rows)
    out = args.out or DATASETS / f"engine_comparison_{args.experiment}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"\n{'axis':20s} {'median |rel err|':>17s} {'max |rel err|':>14s} "
          f"{'tol':>6s} {'within':>8s}")
    verdicts = {}
    for key, tol in TOLERANCE.items():
        err = df[f"{key}_rel_err"].abs().dropna()
        if not len(err):
            continue
        within = df[f"{key}_within_tol"].dropna()
        frac = within.mean() if len(within) else float("nan")
        verdicts[key] = {"median_abs_rel_err": float(err.median()),
                         "max_abs_rel_err": float(err.max()),
                         "tolerance": tol,
                         "fraction_within": float(frac)}
        print(f"{key:20s} {err.median()*100:16.1f}% {err.max()*100:13.1f}% "
              f"{tol*100:5.0f}% {frac*100:7.0f}%")

    verdict_path = DATASETS / f"engine_comparison_{args.experiment}_verdict.json"
    verdict_path.write_text(json.dumps(verdicts, indent=2), encoding="utf-8")
    print(f"\nwritten: {out}\n         {verdict_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
