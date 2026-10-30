"""Run one scenario end to end: generate the stack, start it, inject faults,
collect the data.

Every run produces a self-contained directory under runs/:

    runs/<run_id>/
        scenario.yaml        the scenario as given, copied verbatim
        manifest.json        resolved scenario, plan, predictions, host facts,
                             the full fault schedule and any suppressed faults
        compose.yaml         the generated stack
        raw/*.jsonl          one event log per component, as written
        tidy/*.csv|parquet   one table per event kind, plus derived tables

The manifest is written before the stack starts, so a run that crashes still
leaves behind exactly what was asked for and what was predicted.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from ..components.fault_controller import FaultController
from ..core import churn as churn_mod
from ..core import placement, scenario as scn_mod
from ..core.clock import WallClock

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "runs"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def sanitise(text: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_") else "-" for c in text).lower()


def make_run_id(name: str, tag: str = "") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parts = [sanitise(name), stamp] + ([sanitise(tag)] if tag else [])
    return "-".join(parts)


def host_facts() -> dict:
    def cmd(args: list[str]) -> str:
        try:
            return subprocess.run(args, capture_output=True, text=True,
                                  timeout=30).stdout.strip()
        except Exception:
            return ""

    facts = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "processor": platform.processor(),
        "docker_version": cmd(["docker", "--version"]),
        "compose_version": cmd(["docker", "compose", "version", "--short"]),
        "docker_info": cmd(["docker", "info", "--format",
                            "{{.ServerVersion}}|{{.NCPU}}|{{.MemTotal}}|{{.OSType}}"]),
        "git_commit": cmd(["git", "-C", str(REPO), "rev-parse", "HEAD"]),
        "git_dirty": bool(cmd(["git", "-C", str(REPO), "status", "--porcelain"])),
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }
    info = facts["docker_info"].split("|")
    if len(info) == 4:
        facts["docker_ncpu"] = int(info[1]) if info[1].isdigit() else None
        facts["docker_mem_total"] = int(info[2]) if info[2].isdigit() else None

    # Which image actually ran. Without this, a sweep run before a rebuild and
    # one run after are indistinguishable in the output, and a code change
    # mid-campaign would silently produce a dataset spanning two builds.
    for tag in ("swarm-poc:synthetic", "swarm-poc:real"):
        key = "image_" + tag.split(":")[1]
        facts[key] = cmd(["docker", "image", "inspect", "-f",
                          "{{.Id}}|{{.Created}}", tag])
    return facts


def http_json(url: str, timeout: float = 3.0) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return None


def compose(args: list[str], compose_file: Path, stack: str,
            check: bool = True, quiet: bool = False) -> subprocess.CompletedProcess:
    full = ["docker", "compose", "-f", str(compose_file), "-p", stack] + args
    proc = subprocess.run(full, capture_output=True, text=True)
    if proc.returncode != 0 and check:
        raise RuntimeError(
            f"`{' '.join(full)}` failed rc={proc.returncode}\n"
            f"{proc.stdout}\n{proc.stderr}")
    if not quiet and proc.stderr.strip():
        tail = proc.stderr.strip().splitlines()[-6:]
        for line in tail:
            print(f"    | {line}")
    return proc


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------
async def run_scenario(scn: scn_mod.Scenario, *, tag: str = "",
                       keep_up: bool = False, quiet: bool = False,
                       model_dir: Path | None = None,
                       source_path: Path | None = None) -> Path:
    plan = placement.build(scn)
    schedule, suppressed = churn_mod.generate(scn, plan)

    run_id = make_run_id(scn.name, tag)
    stack = sanitise(f"swarm-{run_id}")[:60]
    run_dir = RUNS / run_id
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    run_epoch = time.time()
    clock = WallClock(run_epoch=run_epoch, time_scale=scn.time_scale)

    # -- manifest, written before anything starts --------------------------
    preds = plan.predictions()
    manifest = {
        "run_id": run_id,
        "stack": stack,
        "run_epoch": run_epoch,
        "run_epoch_iso": datetime.fromtimestamp(run_epoch, timezone.utc).isoformat(),
        "engine": scn.engine,
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
        "tier_emulation": {
            str(t): {
                "mem_bw_bytes": scn.mem_bw_of(t),
                "cpu_cores": scn.tier_spec(t).cpu_cores,
                "mem_limit_mb": scn.tier_spec(t).mem_limit_mb,
                "link": {
                    "delay_ms": scn.link_for(t).delay_ms,
                    "jitter_ms": scn.link_for(t).jitter_ms,
                    "loss_pct": scn.link_for(t).loss_pct,
                    "rate_mbit": scn.link_for(t).rate_mbit,
                },
            }
            for t in sorted(set(plan.node_tiers.values()))
        },
        "churn": {
            "summary": churn_mod.summarise(schedule, scn.duration_s),
            "schedule": [{"at_s": e.at_s, "action": e.action, "target": e.target}
                         for e in schedule],
            "suppressed": suppressed,
        },
        "host": host_facts(),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=float), encoding="utf-8")
    if source_path and source_path.exists():
        shutil.copy2(source_path, run_dir / "scenario.yaml")
    else:
        import yaml
        (run_dir / "scenario.yaml").write_text(
            yaml.safe_dump(scn.to_dict(), sort_keys=False), encoding="utf-8")

    # -- stack -------------------------------------------------------------
    from . import compose as compose_gen
    doc = compose_gen.build(scn, plan, run_id, run_epoch, raw_dir, stack,
                            model_host_dir=model_dir)
    compose_file = compose_gen.write(doc, run_dir / "compose.yaml")

    print(f"\n=== {run_id} ===")
    print(f"  mode={scn.mode} engine={scn.engine} D={plan.depth} "
          f"W={scn.composition.width} N={len(plan.nodes)} G={scn.workload.concurrency}")
    print(f"  predicted: T_token={preds['t_token_s_mean']*1000:.1f} ms  "
          f"X_max={preds['x_max_tok_s']:.1f} tok/s  "
          f"X(G)={preds['x_of_g_tok_s']:.1f} tok/s  "
          f"X_pipe={preds['x_pipe_tok_s']:.1f} tok/s")

    controller = FaultController(
        scenario=scn, plan=plan, schedule=schedule, stack=stack,
        network=f"{stack}-net", clock=clock, verbose=not quiet)

    t_up = time.perf_counter()
    try:
        compose(["up", "-d", "--remove-orphans"], compose_file, stack, quiet=quiet)
        print(f"  stack up in {time.perf_counter() - t_up:.1f}s; waiting for readiness")

        # -- wait for the gateway to start generating load -----------------
        deadline = time.time() + 180
        status = None
        while time.time() < deadline:
            status = http_json("http://127.0.0.1:8200/status")
            if status and status.get("ready"):
                break
            await asyncio.sleep(0.5)
        if not (status and status.get("ready")):
            raise RuntimeError("gateway never became ready; see raw/gateway.jsonl")

        offset = float(status["load_start_sim_ts"])
        controller.offset_s = offset
        print(f"  load started at sim_ts={offset:.2f}s "
              f"(boot took {offset:.1f}s of simulated time)")

        fault_task = asyncio.create_task(controller.run())

        # -- progress ------------------------------------------------------
        end_sim = offset + scn.duration_s
        while True:
            status = http_json("http://127.0.0.1:8200/status")
            if status is None:
                await asyncio.sleep(1.0)
                continue
            if status.get("done"):
                break
            if not quiet:
                frac = max(0.0, min(1.0, (status["sim_ts"] - offset) / scn.duration_s))
                print(f"\r  [{'#' * int(frac*30):<30}] "
                      f"{frac*100:5.1f}%  tokens={status['tokens']:<7} "
                      f"ok={status['ok']:<5} failed={status['failed']:<4} "
                      f"inflight={status['inflight']:<3}", end="", flush=True)
            if status["sim_ts"] > end_sim + 60:
                print("\n  ! gateway overran its duration; stopping")
                break
            await asyncio.sleep(1.0)
        if not quiet:
            print()

        await asyncio.wait_for(fault_task, timeout=30)
        await asyncio.sleep(2.0)        # let final events flush to the volume

        manifest["faults_applied"] = controller.applied
        manifest["final_status"] = http_json("http://127.0.0.1:8200/status")
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, default=float), encoding="utf-8")

    finally:
        if keep_up:
            print(f"  stack left running: docker compose -f {compose_file} "
                  f"-p {stack} down -v")
        else:
            compose(["logs", "--no-color", "--tail", "200"], compose_file, stack,
                    check=False, quiet=True)
            logs = compose(["logs", "--no-color"], compose_file, stack,
                           check=False, quiet=True)
            (run_dir / "docker-logs.txt").write_text(
                logs.stdout + logs.stderr, encoding="utf-8")
            compose(["down", "-v", "--remove-orphans", "-t", "3"], compose_file,
                    stack, check=False, quiet=True)
            print("  stack down")

    # -- tidy --------------------------------------------------------------
    from ..analysis.collector import collect
    tables = collect(run_dir)
    print(f"  tidy tables: " + ", ".join(
        f"{name}({rows})" for name, rows in sorted(tables.items())))
    return run_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run one swarm PoC scenario.")
    ap.add_argument("scenario", type=Path, help="path to a scenario YAML file")
    ap.add_argument("--tag", default="", help="suffix for the run id")
    ap.add_argument("--keep-up", action="store_true",
                    help="leave the stack running for inspection")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--model-dir", type=Path, default=REPO / "models",
                    help="host directory holding the split model (real mode)")
    args = ap.parse_args(argv)

    scn = scn_mod.load(args.scenario)
    model_dir = args.model_dir if scn.mode == "real" else None
    asyncio.run(run_scenario(scn, tag=args.tag, keep_up=args.keep_up,
                             quiet=args.quiet, model_dir=model_dir,
                             source_path=args.scenario))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
