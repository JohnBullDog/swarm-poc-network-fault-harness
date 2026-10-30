"""Run a whole experiment: every sweep point, then one tidy dataset.

Runs are sequential by design. The host is shared by every container, so two
concurrent runs would contend for the same CPU and memory bandwidth and each
would measure the other's interference. A sweep therefore takes as long as it
takes, which is the right trade for data that means something.

Progress is written to datasets/<key>_progress.json after every point, so an
interrupted sweep can be resumed with --resume rather than restarted.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import pandas as pd

from .experiments import EXPERIMENTS
from .run import REPO, run_scenario
from ..virtual.run import run_virtual

DATASETS = REPO / "datasets"


def dataset_for(key: str) -> Path:
    return DATASETS / f"{key}_runs.csv"


def consolidate(key: str, run_dirs: list[Path]) -> Path:
    """Concatenate the one-row `runs` tables into the experiment's dataset.

    Rows are keyed by run_id, and every scenario parameter and prediction is
    already on the row, so this file is self-contained: it can be read without
    the run directories it came from.
    """
    DATASETS.mkdir(parents=True, exist_ok=True)
    frames = []
    for rd in run_dirs:
        path = rd / "tidy" / "runs.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise RuntimeError(f"no runs tables found for {key}")
    df = pd.concat(frames, ignore_index=True, sort=False)
    out = dataset_for(key)
    df.to_csv(out, index=False)
    try:
        df.to_parquet(DATASETS / f"{key}_runs.parquet", index=False)
    except Exception:
        pass
    return out


def consolidate_detail(key: str, run_dirs: list[Path],
                       tables: tuple[str, ...] = ("hop", "token_done",
                                                  "recovery", "membership",
                                                  "throughput_sample")) -> list[Path]:
    """Also concatenate the per-observation tables across the sweep.

    These are the tables a reader needs to check the summary rather than take
    them on trust: per-hop timings, per-token latencies, and every detection.

    The wide tables are gzipped. A sweep's `hop` table runs to tens of megabytes
    of highly repetitive text, which compresses roughly eightfold; pandas reads
    a `.csv.gz` with no extra argument, so nothing downstream has to care.
    """
    big = {"hop", "token_done"}
    out: list[Path] = []
    for name in tables:
        frames = []
        for rd in run_dirs:
            path = rd / "tidy" / f"{name}.csv"
            if path.exists():
                frame = pd.read_csv(path)
                if len(frame):
                    frames.append(frame)
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True, sort=False)
        suffix = ".csv.gz" if name in big else ".csv"
        path = DATASETS / f"{key}_{name}{suffix}"
        df.to_csv(path, index=False,
                  compression="gzip" if name in big else None)
        # Drop a previous uncompressed copy, so a re-run does not leave two
        # versions of the same table with different sizes.
        stale = DATASETS / f"{key}_{name}.csv"
        if name in big and stale.exists():
            stale.unlink()
        out.append(path)
    return out


async def drive(key: str, *, resume: bool = False, limit: int = 0,
                quiet: bool = True, model_dir: Path | None = None) -> Path:
    exp = EXPERIMENTS[key]
    scenarios = list(exp.build())
    if limit:
        scenarios = scenarios[:limit]

    DATASETS.mkdir(parents=True, exist_ok=True)
    prog_path = DATASETS / f"{key}_progress.json"
    progress = {}
    if resume and prog_path.exists():
        progress = json.loads(prog_path.read_text(encoding="utf-8"))
    done: dict[str, str] = progress.get("done", {})

    print(f"\n{'=' * 72}")
    print(f"{exp.key}: {exp.title}")
    print(f"  {exp.question}")
    print(f"  {len(scenarios)} sweep points, sequential")
    print(f"{'=' * 72}")

    run_dirs: list[Path] = []
    t_start = time.time()
    for i, scn in enumerate(scenarios, 1):
        if scn.name in done:
            rd = Path(done[scn.name])
            if rd.exists():
                print(f"[{i}/{len(scenarios)}] {scn.name}: already done, skipping")
                run_dirs.append(rd)
                continue
        print(f"\n[{i}/{len(scenarios)}] {scn.name}")
        # Dispatch on the engine. Without this, a virtual scenario went through
        # the container path and tried to start D*W+2 containers -- 1002 of them
        # at report scale, on a host that fits about sixty.
        if scn.engine == "virtual":
            rd = run_virtual(scn, quiet=quiet)
        else:
            rd = await run_scenario(scn, quiet=quiet, model_dir=model_dir)
        run_dirs.append(rd)
        done[scn.name] = str(rd)
        prog_path.write_text(json.dumps(
            {"key": key, "done": done,
             "elapsed_s": time.time() - t_start}, indent=2), encoding="utf-8")

    out = consolidate(key, run_dirs)
    details = consolidate_detail(key, run_dirs)
    mins = (time.time() - t_start) / 60.0
    print(f"\n{exp.key} complete in {mins:.1f} min")
    print(f"  dataset: {out}")
    for d in details:
        print(f"           {d}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run a planned experiment.")
    ap.add_argument("experiment", choices=sorted(EXPERIMENTS),
                    help="which experiment to run")
    ap.add_argument("--resume", action="store_true",
                    help="skip sweep points already completed")
    ap.add_argument("--limit", type=int, default=0,
                    help="run only the first N points (for a smoke test)")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--model-dir", type=Path, default=REPO / "models")
    args = ap.parse_args(argv)

    asyncio.run(drive(args.experiment, resume=args.resume, limit=args.limit,
                      quiet=not args.verbose, model_dir=args.model_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
