"""Where published data lives.

One definition of the layout, imported by everything that reads or writes it,
so a reorganisation is a change here rather than in eight modules — and so a
new sweep cannot quietly scatter files back into the top level.

    datasets/
        README.md               what is here and where to start
        swarm_poc_results.xlsx  the master workbook
        summary/                one row per run, per experiment
        detail/                 one row per observation (hops, tokens, ...)
        analysis/               cross-run artifacts: calibration, comparisons
        _progress/              sweep resume state; bookkeeping, not data
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

DATASETS = REPO / "datasets"
SUMMARY = DATASETS / "summary"
DETAIL = DATASETS / "detail"
ANALYSIS = DATASETS / "analysis"
PROGRESS = DATASETS / "_progress"
WORKBOOK = DATASETS / "swarm_poc_results.xlsx"

# Per-observation tables consolidated across a sweep. The wide two are gzipped:
# a sweep's hop table runs to tens of megabytes of repetitive text.
DETAIL_TABLES = ("hop", "token_done", "recovery", "membership",
                 "throughput_sample")
GZIP_TABLES = {"hop", "token_done"}


def ensure() -> None:
    for d in (DATASETS, SUMMARY, DETAIL, ANALYSIS, PROGRESS):
        d.mkdir(parents=True, exist_ok=True)


def runs_csv(key: str) -> Path:
    """One row per run, for one experiment."""
    return SUMMARY / f"{key}_runs.csv"


def runs_parquet(key: str) -> Path:
    return SUMMARY / f"{key}_runs.parquet"


def detail_csv(key: str, table: str) -> Path:
    suffix = ".csv.gz" if table in GZIP_TABLES else ".csv"
    return DETAIL / f"{key}_{table}{suffix}"


def analysis_file(name: str) -> Path:
    return ANALYSIS / name


def progress_file(key: str) -> Path:
    return PROGRESS / f"{key}_progress.json"


def all_runs_csvs() -> list[Path]:
    """Every experiment's run-level table, in a stable order."""
    return sorted(SUMMARY.glob("*_runs.csv"))


def experiment_keys() -> list[str]:
    return [p.name[: -len("_runs.csv")] for p in all_runs_csvs()]


def all_data_files() -> list[Path]:
    """Every published data file, for checkers that sweep the lot."""
    out: list[Path] = []
    for d in (SUMMARY, DETAIL, ANALYSIS):
        if d.exists():
            out.extend(sorted(p for p in d.iterdir() if p.is_file()))
    return out
