"""Measure the harness's own cost, as residuals against the bare cost model.

This measures the container runtime's overhead; it does not produce a parameter
to feed back into the model. Two findings are why.

The fit is only valid because the stage worker sleeps the *bare* cost-model
time. An earlier version had the worker sleep `bare + configured overhead`
while this tool measured the residual against `bare`, so each fit re-absorbed
the sleep the previous fit had introduced and successive fits diverged
(1.2 -> 2.2 -> 3.3 ms per stage).

And with that circularity removed, the stage residual is not a constant: it
ranges from about 0.2 ms per stage on an idle event loop to about 1.2 ms under
load, because asyncio's sleep wakeups are delayed by whatever else the loop is
doing. Treating it as one host constant would be a fudge factor. The experiments
therefore predict with the bare cost model and report the measured/predicted
ratio; this tool quantifies that ratio's origin.

The two residuals measured:

  stage_overhead_ms   measured compute_s minus (W_k*b + KV_k)/B_k
                      -- the asyncio sleep's own granularity and the per-request
                         Python work around it.

  hop_overhead_ms     measured net_s minus (link delay + a/beta)
                      -- TCP handshake, HTTP framing, container veth traversal
                         and the Docker bridge, none of which the cost model has
                         a term for.

Both are fitted per run and then pooled. They are reported with their spread:
a constant with a wide spread across runs is not a constant, and the report
would say so rather than quietly using the mean.

Usage:
    python -m tools.calibrate runs/m1_minimal-* runs/exp1_*
    python -m tools.calibrate --write    # also update swarm/runner/experiments.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from swarm.core import placement                        # noqa: E402
from swarm.core.scenario import from_dict               # noqa: E402


def fit_one(run_dir: Path) -> dict | None:
    """Fit both overheads from one run's hop table."""
    man_path = run_dir / "manifest.json"
    hop_path = run_dir / "tidy" / "hop.csv"
    if not (man_path.exists() and hop_path.exists()):
        return None
    manifest = json.loads(man_path.read_text(encoding="utf-8"))
    if manifest.get("mode") != "synthetic":
        return None

    scn = from_dict(manifest["scenario"])
    plan = placement.build(scn)
    hop = pd.read_csv(hop_path)
    if "in_steady_state" in hop.columns:
        hop = hop[hop["in_steady_state"] == True]       # noqa: E712
    if "fault_window" in hop.columns:
        hop = hop[hop["fault_window"] != True]          # noqa: E712
    if not len(hop):
        return None

    # Exclude the gateway's return hop (stage == depth): it has no compute, and
    # its net_s is the last stage's egress, already represented by that stage.
    stages = hop[pd.to_numeric(hop["stage"], errors="coerce") < plan.depth]
    ctx = scn.workload.prompt_tokens

    stage_res: list[float] = []
    hop_res: list[float] = []
    for stage_idx, grp in stages.groupby("stage"):
        k = int(stage_idx)
        st = plan.stages[k]
        for node, g2 in grp.groupby("node_id"):
            if node not in plan.node_tiers:
                continue
            tier = plan.node_tiers[node]
            link = scn.link_for(tier)
            kv = st.share * ctx * scn.profile.kv_bytes_per_token
            pred_compute = (st.weight_bytes + kv) / scn.mem_bw_of(tier)
            meas_compute = pd.to_numeric(g2["compute_s"], errors="coerce").median()
            if pd.notna(meas_compute):
                stage_res.append(float(meas_compute - pred_compute) * 1000.0)

            # net_s on stage k's row is the *sender's* egress: the gateway for
            # stage 0, stage k-1 otherwise. Only stages above 0 are used, so the
            # sender is always a worker whose link profile is known.
            if k == 0:
                continue
            prev_tier = plan.node_tiers[plan.stages[k - 1].nodes[0]]
            prev_link = scn.link_for(prev_tier)
            pred_net = (prev_link.delay_s
                        + scn.profile.activation_bytes / prev_link.rate_bytes)
            meas_net = pd.to_numeric(g2["net_s"], errors="coerce").median()
            if pd.notna(meas_net):
                hop_res.append(float(meas_net - pred_net) * 1000.0)

    if not stage_res and not hop_res:
        return None
    # Plain floats, not numpy scalars: these values are written into scenarios
    # and serialised to YAML, and PyYAML cannot represent np.float64.
    return {
        "run_id": manifest["run_id"],
        "scenario": scn.name,
        "depth": plan.depth,
        "n_nodes": len(plan.nodes),
        "concurrency": scn.workload.concurrency,
        "stage_overhead_ms": float(statistics.median(stage_res)) if stage_res else None,
        "hop_overhead_ms": float(statistics.median(hop_res)) if hop_res else None,
        "n_stage_samples": len(stage_res),
        "n_hop_samples": len(hop_res),
    }


def pool(fits: list[dict]) -> dict:
    stage = [f["stage_overhead_ms"] for f in fits
             if f["stage_overhead_ms"] is not None]
    hops = [f["hop_overhead_ms"] for f in fits if f["hop_overhead_ms"] is not None]

    def summarise(xs: list[float]) -> dict:
        if not xs:
            return {"median": None, "min": None, "max": None, "n": 0}
        return {
            "median": round(statistics.median(xs), 4),
            "mean": round(statistics.fmean(xs), 4),
            "stdev": round(statistics.stdev(xs), 4) if len(xs) > 1 else 0.0,
            "min": round(min(xs), 4),
            "max": round(max(xs), 4),
            "n": len(xs),
        }

    return {"stage_overhead_ms": summarise(stage),
            "hop_overhead_ms": summarise(hops),
            "per_run": fits}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", type=Path,
                    help="run directories (default: every synthetic run under runs/)")
    from swarm.core import paths
    ap.add_argument("--out", type=Path,
                    default=paths.analysis_file("calibration.json"))
    args = ap.parse_args(argv)

    dirs = args.runs or sorted(p for p in (REPO / "runs").iterdir() if p.is_dir())
    fits = [f for f in (fit_one(d) for d in dirs) if f]
    if not fits:
        print("no synthetic runs with hop data found")
        return 1

    result = pool(fits)
    print(f"{'run':44s} {'D':>2s} {'G':>3s} {'stage_ms':>9s} {'hop_ms':>8s}")
    for f in result["per_run"]:
        s = f["stage_overhead_ms"]
        h = f["hop_overhead_ms"]
        print(f"{f['scenario'][:44]:44s} {f['depth']:2d} {f['concurrency']:3d} "
              f"{s if s is None else round(s, 3):>9} "
              f"{h if h is None else round(h, 3):>8}")

    print("\npooled across runs:")
    for key in ("stage_overhead_ms", "hop_overhead_ms"):
        st = result[key]
        if st["n"]:
            print(f"  {key:20s} median={st['median']:7.3f}  mean={st['mean']:7.3f}"
                  f"  sd={st['stdev']:6.3f}  range=[{st['min']:.3f}, {st['max']:.3f}]"
                  f"  n={st['n']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
