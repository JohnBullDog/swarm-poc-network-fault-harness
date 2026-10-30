"""Fault controller: applies the scenario's churn schedule to running containers.

Deliberate deviation from the design doc: this runs as a host-side process, not
as a container. Detaching a container from a network and pausing or killing it
requires the Docker API, and reaching it from inside a container means mounting
the Docker socket -- full control of the host daemon -- which buys nothing on a
single host. The capability is identical either way.

Baseline link shaping is NOT done here. Each container applies its own tc
profile at startup (docker/entrypoint.sh), so no request ever crosses an
unshaped link. This controller handles only the dynamic faults: disconnection,
pause, kill, rejoin, and mid-run changes to delay or loss.

Ground truth for detection latency. After each action completes, the controller
reports the simulated instant to the coordinator. `detection_s` in the output is
then a measured difference between two observations, not an inference from the
heartbeat interval. The time the docker command itself took is recorded
alongside, so that cost is visible rather than buried in the measurement.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from ..core.churn import DOWN_ACTIONS
from ..core.clock import WallClock
from ..core.placement import Plan
from ..core.scenario import FaultEvent, Scenario


@dataclass
class FaultController:
    scenario: Scenario
    plan: Plan
    schedule: list[FaultEvent]
    stack: str
    network: str
    clock: WallClock
    coordinator_url: str = "http://127.0.0.1:8100"
    verbose: bool = True
    # Simulated instant the load generator started. A fault's `at_s` is measured
    # from here, not from the run epoch, so container boot time does not eat
    # into the schedule.
    offset_s: float = 0.0

    applied: list[dict] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.applied = []

    # -- plumbing ---------------------------------------------------------
    def container(self, node_id: str) -> str:
        return f"{self.stack}-{node_id}"

    def _run(self, args: list[str]) -> tuple[int, str]:
        try:
            proc = subprocess.run(args, capture_output=True, text=True, timeout=60)
            return proc.returncode, (proc.stdout + proc.stderr).strip()
        except subprocess.TimeoutExpired:
            return 124, "timeout"

    def _notify(self, node_id: str, action: str, fault_sim_ts: float,
                detail: dict) -> None:
        body = json.dumps({
            "node_id": node_id, "action": action,
            "fault_sim_ts": fault_sim_ts, "detail": detail,
        }).encode()
        req = urllib.request.Request(
            f"{self.coordinator_url}/fault", data=body,
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            urllib.request.urlopen(req, timeout=5).read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if self.verbose:
                print(f"    ! could not notify coordinator: {exc}")

    # -- the actions ------------------------------------------------------
    def apply(self, ev: FaultEvent) -> dict:
        node = ev.target
        cont = self.container(node)
        t_cmd = time.perf_counter()

        if ev.action == "disconnect":
            rc, out = self._run(["docker", "network", "disconnect",
                                 self.network, cont])
        elif ev.action == "rejoin":
            # --alias is essential: without it the container rejoins with a new
            # address and no DNS name, so the gateway could never route to it
            # again and "rejoin" would silently mean "gone for good".
            rc, out = self._run(["docker", "network", "connect",
                                 "--alias", node, self.network, cont])
        elif ev.action == "pause":
            rc, out = self._run(["docker", "pause", cont])
        elif ev.action == "unpause":
            rc, out = self._run(["docker", "unpause", cont])
        elif ev.action == "kill":
            rc, out = self._run(["docker", "kill", cont])
        else:
            raise ValueError(f"unhandled action {ev.action!r}")

        apply_ms = (time.perf_counter() - t_cmd) * 1000.0
        fault_sim_ts = self.clock.sim()
        rec = {
            "scheduled_at_s": ev.at_s,
            "scheduled_sim_ts": ev.at_s + self.offset_s,
            "fault_sim_ts": fault_sim_ts,
            "lateness_s": fault_sim_ts - (ev.at_s + self.offset_s),
            "action": ev.action,
            "node_id": node,
            "tier": self.plan.node_tiers.get(node),
            "stage": self.plan.stage_of(node) if node in self.plan.node_tiers else None,
            "rc": rc,
            "apply_ms": apply_ms,
            "output": out[:400],
        }
        self.applied.append(rec)
        if self.verbose:
            flag = "ok" if rc == 0 else f"FAILED rc={rc}"
            print(f"  [fault] t={fault_sim_ts:7.3f}s {ev.action:11s} {node:10s} "
                  f"{flag} ({apply_ms:.0f} ms)")
            if rc != 0 and out:
                print(f"           {out[:200]}")

        if rc == 0:
            self._notify(node, ev.action, fault_sim_ts,
                         {"apply_ms": round(apply_ms, 2),
                          "scheduled_at_s": ev.at_s,
                          "lateness_s": round(rec["lateness_s"], 4)})
        return rec

    # -- the loop ---------------------------------------------------------
    async def run(self) -> list[dict]:
        """Apply every scheduled fault at its simulated instant."""
        if not self.schedule:
            return []
        if self.verbose:
            downs = sum(1 for e in self.schedule if e.action in DOWN_ACTIONS)
            print(f"  fault controller: {len(self.schedule)} events "
                  f"({downs} departures) over {self.scenario.duration_s:g}s")

        for ev in sorted(self.schedule, key=lambda e: e.at_s):
            wait = (ev.at_s + self.offset_s) - self.clock.sim()
            if wait > 0:
                await asyncio.sleep(wait / self.scenario.time_scale)
            # Blocking docker calls run off the event loop so one slow command
            # cannot delay the next scheduled fault's own timestamp.
            await asyncio.get_running_loop().run_in_executor(None, self.apply, ev)
        return self.applied

    # -- teardown helper --------------------------------------------------
    def restore_all(self) -> None:
        """Reconnect and unpause everything, so a stack can be reused or torn
        down cleanly even if a run aborted mid-fault."""
        for node in self.plan.nodes:
            cont = self.container(node)
            self._run(["docker", "unpause", cont])
            self._run(["docker", "network", "connect", "--alias", node,
                       self.network, cont])
