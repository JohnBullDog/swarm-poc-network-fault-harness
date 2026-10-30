"""Experiment 7: does synthetic mode's timing track real mode's?

This is the experiment that licenses every synthetic result. If synthetic
timings do not track real ones here, nothing measured in synthetic mode at
larger scale can be trusted.

It runs in three phases, because a fair comparison cannot be made in one.

  Phase 1  real arms. Stage workers run real blocks of Qwen2.5-0.5B layers on
           CPU and the gateway runs embedding and the output head.

  Phase 2  fit the effective memory bandwidth from phase 1:

               B_eff = stage weight bytes / measured compute time

           This is the step that makes the comparison meaningful. A tier's
           nominal bandwidth is a reference value from the research report
           (tier 2 = 45 GB/s); a single-core container doing fp32 transformer
           decode does not achieve it. Comparing a real arm against a synthetic
           arm fed the nominal figure would measure that mismatch rather than
           synthetic mode's fidelity. Tier emulation is documented as preserving
           ratios, not absolute device speeds, and this is where that bites.

  Phase 3  two synthetic arms: one on the nominal bandwidth, to show the size of
           the mismatch, and one on the fitted B_eff, which is the actual
           fidelity test.

The verdict is reported per concurrency point and pooled, and a disagreement is
reported rather than reconciled.

    python -m tools.mode_check
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from swarm.core import placement                            # noqa: E402
from swarm.core.scenario import SyntheticCfg                # noqa: E402
from swarm.runner.experiments import (HOP_OVERHEAD_MS, MODE_CHECK_G,  # noqa: E402
                                      STAGE_OVERHEAD_MS, exp7_mode_check)
from swarm.runner.run import REPO as RUN_REPO, run_scenario  # noqa: E402

from swarm.core import paths               # noqa: E402

DATASETS = paths.DATASETS
TOLERANCE = 0.15        # 15% on per-token latency is the bar for "tracks"


def stage_compute_ms(run_dir: Path, depth: int) -> dict[int, float]:
    """Median measured compute time per stage, in ms."""
    path = run_dir / "tidy" / "hop.csv"
    if not path.exists():
        return {}
    hop = pd.read_csv(path)
    if "in_steady_state" in hop.columns:
        hop = hop[hop["in_steady_state"] == True]           # noqa: E712
    out: dict[int, float] = {}
    for stage, grp in hop.groupby("stage"):
        k = int(stage)
        if k >= depth:
            continue
        out[k] = float(pd.to_numeric(grp["compute_s"],
                                     errors="coerce").median()) * 1000
    return out


def summarise(run_dir: Path) -> dict:
    row = pd.read_csv(run_dir / "tidy" / "runs.csv").iloc[0]
    man = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    depth = int(row["depth_D"])
    return {
        "run_id": row["run_id"],
        "scenario": row["scenario"],
        "mode": row["mode"],
        "concurrency_G": int(row["concurrency_G"]),
        "t_token_p50_ms": float(row["meas_t_token_s_p50"]) * 1000
        if pd.notna(row["meas_t_token_s_p50"]) else float("nan"),
        "throughput_tok_s": float(row["meas_throughput_tok_s"]),
        "tokens": int(row["tokens_steady"]),
        "stage_compute_ms": stage_compute_ms(run_dir, depth),
        "depth": depth,
        "weight_bytes": [s["weight_bytes"] for s in man["plan"]["stages"]],
        "requests_ok": int(row["requests_ok"]),
        "requests_failed": int(row["requests_failed"]),
    }


def fit_effective_bandwidth(reals: list[dict]) -> dict:
    """B_eff per stage, pooled, from measured real compute times."""
    per_stage: list[float] = []
    detail = []
    for r in reals:
        for stage, ms in sorted(r["stage_compute_ms"].items()):
            wb = r["weight_bytes"][stage]
            # Subtract the fitted stage overhead so B_eff describes the memory
            # read itself and not the Python around it.
            compute_s = max(1e-9, (ms - STAGE_OVERHEAD_MS) / 1000.0)
            bw = wb / compute_s
            per_stage.append(bw)
            detail.append({"run": r["run_id"], "stage": stage,
                           "weight_bytes": wb, "compute_ms": ms,
                           "b_eff_gbs": bw / 1e9})
    if not per_stage:
        return {}
    return {
        "b_eff_bytes_s_median": statistics.median(per_stage),
        "b_eff_gbs_median": statistics.median(per_stage) / 1e9,
        "b_eff_gbs_mean": statistics.fmean(per_stage) / 1e9,
        "b_eff_gbs_stdev": (statistics.stdev(per_stage) / 1e9
                            if len(per_stage) > 1 else 0.0),
        "b_eff_gbs_min": min(per_stage) / 1e9,
        "b_eff_gbs_max": max(per_stage) / 1e9,
        "n_samples": len(per_stage),
        "per_stage": detail,
    }


async def run_all(model_dir: Path, limit_g: tuple[int, ...] | None = None) -> int:
    scenarios = list(exp7_mode_check())
    reals = [s for s in scenarios if s.mode == "real"]
    if limit_g:
        reals = [s for s in reals if s.workload.concurrency in limit_g]

    print("=" * 72)
    print("Experiment 7: mode check")
    print("=" * 72)

    # -- phase 1 ----------------------------------------------------------
    print(f"\nphase 1: {len(reals)} real-mode runs")
    real_sums = []
    for i, scn in enumerate(reals, 1):
        print(f"\n[{i}/{len(reals)}] {scn.name}")
        rd = await run_scenario(scn, quiet=True, model_dir=model_dir)
        s = summarise(rd)
        real_sums.append(s)
        print(f"    p50={s['t_token_p50_ms']:.1f} ms  X={s['throughput_tok_s']:.2f} tok/s"
              f"  stage compute={[round(v, 1) for _, v in sorted(s['stage_compute_ms'].items())]} ms"
              f"  ok={s['requests_ok']} failed={s['requests_failed']}")

    if not real_sums:
        print("no real runs completed")
        return 1

    # -- phase 2 ----------------------------------------------------------
    fit = fit_effective_bandwidth(real_sums)
    if not fit:
        print("could not fit effective bandwidth")
        return 1
    b_eff = fit["b_eff_gbs_median"]
    print(f"\nphase 2: fitted effective memory bandwidth")
    print(f"    B_eff = {b_eff:.2f} GB/s  "
          f"(mean {fit['b_eff_gbs_mean']:.2f}, sd {fit['b_eff_gbs_stdev']:.2f}, "
          f"range [{fit['b_eff_gbs_min']:.2f}, {fit['b_eff_gbs_max']:.2f}], "
          f"n={fit['n_samples']})")
    print(f"    tier 2 nominal is 45.00 GB/s -> real is {45.0 / b_eff:.1f}x slower")

    # -- phase 3 ----------------------------------------------------------
    print("\nphase 3: synthetic arms, nominal and calibrated")
    synth_sums: list[dict] = []
    for scn in reals:
        g = scn.workload.concurrency
        for arm, cfg in (
            ("synthetic_nominal",
             SyntheticCfg(servers_per_node=1, busy_compute=True,
                          stage_overhead_ms=STAGE_OVERHEAD_MS,
                          hop_overhead_ms=HOP_OVERHEAD_MS)),
            ("synthetic_calibrated",
             SyntheticCfg(servers_per_node=1, busy_compute=True,
                          stage_overhead_ms=STAGE_OVERHEAD_MS,
                          hop_overhead_ms=HOP_OVERHEAD_MS,
                          mem_bw_gbs={2: b_eff})),
        ):
            s_scn = scn.with_(
                name=f"exp7_{arm}_G{g}", mode="synthetic", synthetic=cfg,
                sweep_point={"concurrency_G": g, "arm": arm,
                             "b_eff_gbs": b_eff if "calib" in arm else 45.0})
            print(f"\n  {s_scn.name}")
            rd = await run_scenario(s_scn, quiet=True, model_dir=None)
            s = summarise(rd)
            s["arm"] = arm
            synth_sums.append(s)
            print(f"    p50={s['t_token_p50_ms']:.1f} ms  "
                  f"X={s['throughput_tok_s']:.2f} tok/s  "
                  f"stage compute="
                  f"{[round(v, 1) for _, v in sorted(s['stage_compute_ms'].items())]} ms")

    # -- verdict ----------------------------------------------------------
    rows = []
    by_g = {s["concurrency_G"]: s for s in real_sums}
    for s in synth_sums:
        r = by_g.get(s["concurrency_G"])
        if r is None:
            continue
        rel_lat = (s["t_token_p50_ms"] - r["t_token_p50_ms"]) / r["t_token_p50_ms"]
        rel_x = (s["throughput_tok_s"] - r["throughput_tok_s"]) \
            / max(1e-9, r["throughput_tok_s"])
        rows.append({
            "concurrency_G": s["concurrency_G"],
            "arm": s["arm"],
            "real_run": r["run_id"], "synthetic_run": s["run_id"],
            "real_p50_ms": r["t_token_p50_ms"],
            "synthetic_p50_ms": s["t_token_p50_ms"],
            "p50_rel_err": rel_lat,
            "real_throughput_tok_s": r["throughput_tok_s"],
            "synthetic_throughput_tok_s": s["throughput_tok_s"],
            "throughput_rel_err": rel_x,
            "within_tolerance": abs(rel_lat) <= TOLERANCE,
            "b_eff_gbs": b_eff,
        })

    df = pd.DataFrame(rows)
    paths.ensure()
    df.to_csv(paths.analysis_file("exp7_mode_check.csv"), index=False)
    paths.analysis_file("exp7_effective_bandwidth.json").write_text(
        json.dumps(fit, indent=2), encoding="utf-8")

    print(f"\n{'arm':22s} {'G':>2s} {'real p50':>9s} {'synth p50':>10s} "
          f"{'err':>8s} {'within':>7s}")
    for _, r in df.iterrows():
        print(f"{r['arm']:22s} {r['concurrency_G']:2d} "
              f"{r['real_p50_ms']:8.1f}ms {r['synthetic_p50_ms']:9.1f}ms "
              f"{r['p50_rel_err']*100:+7.1f}% "
              f"{'yes' if r['within_tolerance'] else 'NO':>7s}")

    for arm in df["arm"].unique():
        sub = df[df["arm"] == arm]
        med = sub["p50_rel_err"].abs().median() * 100
        frac = sub["within_tolerance"].mean() * 100
        print(f"\n{arm}: median |error| {med:.1f}%, "
              f"{frac:.0f}% of points within {TOLERANCE*100:.0f}%")

    print(f"\nwritten: {paths.analysis_file('exp7_mode_check.csv')}")
    print(f"         {paths.analysis_file('exp7_effective_bandwidth.json')}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path,
                    default=RUN_REPO / "models")  # mounted at /models:ro
    ap.add_argument("--only-g", type=int, nargs="*", default=None,
                    help="restrict to these concurrency points")
    args = ap.parse_args(argv)
    limit = tuple(args.only_g) if args.only_g else None
    return asyncio.run(run_all(args.model_dir, limit))


if __name__ == "__main__":
    raise SystemExit(main())
