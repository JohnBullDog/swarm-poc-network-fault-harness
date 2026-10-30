"""Event schema and the JSON-lines writer.

Every component appends to its own file on the shared volume, so there is no
locking and no interleaving loss. The collector merges them into one tidy table
per event kind.

The schema is declared here rather than inferred from whatever a run happened to
emit. That matters for tidy output: a table has the same columns whether or not
a particular run produced any rows for it, so runs can be concatenated across a
sweep without columns appearing and disappearing.

Column conventions:
  *_ts   an instant, seconds. `sim_ts` is simulated, `wall_ts` is unix time.
  *_s    a duration, seconds.
  One row = one observation. Nothing is nested except `path`, which is a
  routed-node list and is stored as a JSON string so the table stays flat.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

# Present on every row, written by the EventLog itself.
COMMON_FIELDS = ("run_id", "engine", "mode", "src", "kind", "wall_ts", "sim_ts")

# kind -> the fields that kind adds. Order is the column order in tidy output.
EVENT_SCHEMA: dict[str, tuple[str, ...]] = {
    "run_start": ("scenario", "depth", "width", "n_nodes", "concurrency", "seed"),
    "run_end": ("reason", "requests_started", "requests_ok", "requests_failed"),

    "request_start": ("request_id", "stream_id", "prompt_tokens", "gen_tokens"),
    "request_end": ("request_id", "stream_id", "status", "tokens_done",
                    "retries", "failovers", "latency_s", "error"),

    "token_done": ("request_id", "stream_id", "token_idx", "latency_s",
                   "context_len", "retries", "failovers", "path"),

    "hop": ("request_id", "token_idx", "stage", "node_id", "tier", "attempt",
            "queue_wait_s", "compute_s", "net_s", "total_s", "payload_bytes"),
    "hop_error": ("request_id", "token_idx", "stage", "node_id", "tier",
                  "attempt", "error", "elapsed_s"),

    "heartbeat": ("node_id", "tier", "stage", "seq"),
    "node_sample": ("node_id", "tier", "stage", "inflight", "queue_depth",
                    "busy_frac", "tokens_done", "hb_gap_s"),

    "membership": ("node_id", "tier", "stage", "event", "gap_s",
                   "fault_sim_ts", "detection_s", "reason"),
    "fault_applied": ("node_id", "tier", "stage", "action", "detail"),

    "throughput_sample": ("window_s", "tokens", "tokens_per_s",
                          "requests_inflight", "live_nodes"),
    "note": ("message", "detail"),
}

# Which tidy table each kind lands in. Kinds sharing a table must share a schema.
TABLE_OF_KIND = {k: k for k in EVENT_SCHEMA}

# Tables the collector derives rather than any component emitting them. They do
# not carry the common event columns, because a derived row is not an event that
# happened at an instant -- `recovery` spans a fault and its repair, and `runs`
# summarises a whole run. They are listed here so a schema check can tell a
# derived table apart from an event table it has the wrong columns for.
DERIVED_TABLES = ("recovery", "runs")


def columns_for(kind: str) -> list[str]:
    return list(COMMON_FIELDS) + list(EVENT_SCHEMA[kind])


class EventLog:
    """Append-only JSON-lines writer, one per component."""

    def __init__(self, path: str | Path, run_id: str, engine: str, mode: str,
                 src: str, clock, flush_every: int = 16,
                 flush_interval_s: float = 1.0, strict: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.engine = engine
        self.mode = mode
        self.src = src
        self.clock = clock
        self.flush_every = flush_every
        # Containers are stopped by signal at the end of a run, so buffered
        # events are lost unless they are flushed on a timer as well as by
        # count. A low-rate component could otherwise sit on a partial buffer
        # for the whole run and lose it. Buffered events are measurements.
        self.flush_interval_s = flush_interval_s
        self.strict = strict
        self.counts: dict[str, int] = {}
        self._fh = open(self.path, "a", encoding="utf-8", buffering=1 << 16)
        self._since_flush = 0
        self._last_flush = time.monotonic()

    def emit(self, kind: str, **fields: Any) -> None:
        if self._fh.closed:
            # A periodic loop can outlive the log by a scheduling tick at
            # shutdown. Dropping that last sample is correct; raising from it
            # would surface as a spurious task exception in the run's logs.
            return
        allowed = EVENT_SCHEMA.get(kind)
        if allowed is None:
            raise KeyError(f"unknown event kind {kind!r}")
        if self.strict:
            extra = set(fields) - set(allowed)
            if extra:
                raise KeyError(f"event {kind!r} got undeclared fields {sorted(extra)}")
        rec = {
            "run_id": self.run_id,
            "engine": self.engine,
            "mode": self.mode,
            "src": self.src,
            "kind": kind,
            "wall_ts": self.clock.wall(),
            "sim_ts": self.clock.sim(),
        }
        for name in allowed:
            rec[name] = fields.get(name)
        self._fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
        self.counts[kind] = self.counts.get(kind, 0) + 1
        self._maybe_flush()

    def emit_at(self, kind: str, sim_ts: float, **fields: Any) -> None:
        """Emit with an explicit simulated timestamp.

        The virtual engine uses this when logging an event that happened at a
        scheduled instant rather than at the scheduler's current position.
        """
        allowed = EVENT_SCHEMA[kind]
        rec = {
            "run_id": self.run_id, "engine": self.engine, "mode": self.mode,
            "src": self.src, "kind": kind,
            "wall_ts": self.clock.wall(), "sim_ts": sim_ts,
        }
        for name in allowed:
            rec[name] = fields.get(name)
        self._fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
        self.counts[kind] = self.counts.get(kind, 0) + 1
        self._maybe_flush()

    def _maybe_flush(self) -> None:
        self._since_flush += 1
        if (self._since_flush >= self.flush_every
                or time.monotonic() - self._last_flush >= self.flush_interval_s):
            self.flush()

    def flush(self) -> None:
        self._fh.flush()
        try:
            os.fsync(self._fh.fileno())
        except OSError:
            pass            # a bind-mounted volume may refuse fsync; buffering is enough
        self._since_flush = 0
        self._last_flush = time.monotonic()

    def close(self) -> None:
        try:
            self.flush()
        finally:
            self._fh.close()

    def __enter__(self) -> "EventLog":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file, skipping any trailing partial line from a killed writer."""
    out: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue    # a container killed mid-write leaves one short line
    return out
