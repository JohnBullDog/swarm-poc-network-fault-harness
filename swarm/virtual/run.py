"""Run a scenario on the virtual clock.

Writes the same run directory layout as a container run -- manifest, raw event
logs, tidy tables -- so the collector, the datasets and every analysis work on
both engines without a special case. The manifest records both the simulated
duration and the real seconds spent computing it, since the whole point of this
engine is that those differ by orders of magnitude.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from ..core import churn as churn_mod
from ..core import placement, scenario as scn_mod
from .engine import VirtualEngine

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "runs"


def run_virtual(scn: scn_mod.Scenario, *, tag: str = "",
                source_path: Path | None = None, quiet: bool = False) -> Path:
    if scn.engine != "virtual":
        scn = scn.with_(engine="virtual")
    if scn.mode == "real":
        raise ValueError("the virtual engine has no model; real mode needs containers")

    from ..runner.run import make_run_id

    plan = placement.build(scn)
    schedule, suppressed = churn_mod.generate(scn, plan)
    run_id = make_run_id(scn.name, tag or "virtual")
    run_dir = RUNS / run_id
    (run_dir / "raw").mkdir(parents=True, exist_ok=True)

    preds = plan.predictions()
    run_epoch = time.time()
    manifest = {
        "run_id": run_id,
        "run_epoch": run_epoch,
        "run_epoch_iso": datetime.fromtimestamp(run_epoch, timezone.utc).isoformat(),
        "engine": "virtual",
        "mode": scn.mode,
        "experiment": scn.experiment,
        "sweep_point": scn.sweep_point,
        "scenario": scn.to_dict(),
        "model_profile": scn.profile.to_dict(),
        "plan": {
            "node_tiers": plan.node_tiers,
            "stages": [
                {"stage": s.stage, "nodes": s.nodes, "tiers": s.tiers,
                 "share": s.share, "weight_bytes": s.weight_bytes,
                 "layer_start": s.layer_start, "layer_end": s.layer_end}
                for s in plan.stages
            ],
        },
        "predictions": preds,
        "churn": {
            "summary": churn_mod.summarise(schedule, scn.duration_s),
            "schedule": [{"at_s": e.at_s, "action": e.action, "target": e.target}
                         for e in schedule],
            "suppressed": suppressed,
        },
        "host": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "note": "virtual engine: no containers, simulated clock",
        },
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=float), encoding="utf-8")
    if source_path and source_path.exists():
        import shutil
        shutil.copy2(source_path, run_dir / "scenario.yaml")
    else:
        import yaml
        (run_dir / "scenario.yaml").write_text(
            yaml.safe_dump(scn.to_dict(), sort_keys=False), encoding="utf-8")

    if not quiet:
        print(f"\n=== {run_id} (virtual) ===")
        print(f"  D={plan.depth} W={scn.composition.width} N={len(plan.nodes)} "
              f"G={scn.workload.concurrency}  sim_duration={scn.duration_s:g}s")
        print(f"  predicted: T_token={preds['t_token_s_mean']*1000:.1f} ms  "
              f"X_max={preds['x_max_tok_s']:.1f}  "
              f"X_closed_loop={preds['x_closed_loop_tok_s']:.1f} tok/s")

    t0 = time.perf_counter()
    engine = VirtualEngine(scn, run_dir, run_id)
    result = engine.run()
    real_s = time.perf_counter() - t0

    manifest["virtual"] = {
        **result,
        "real_seconds": real_s,
        "speedup_vs_wall_clock": (scn.duration_s / real_s) if real_s > 0 else None,
        "events_per_real_second": (result["sim_events"] / real_s) if real_s > 0 else None,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=float), encoding="utf-8")

    if not quiet:
        print(f"  simulated {scn.duration_s:g}s in {real_s:.2f}s real "
              f"({manifest['virtual']['speedup_vs_wall_clock']:.0f}x), "
              f"{result['sim_events']:,} events")

    from ..analysis.collector import collect
    tables = collect(run_dir)
    if not quiet:
        print("  tidy tables: " + ", ".join(
            f"{k}({v})" for k, v in sorted(tables.items())))
    return run_dir


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run one scenario on the virtual clock.")
    ap.add_argument("scenario", type=Path)
    ap.add_argument("--tag", default="")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    scn = scn_mod.load(args.scenario)
    run_virtual(scn, tag=args.tag, source_path=args.scenario, quiet=args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
