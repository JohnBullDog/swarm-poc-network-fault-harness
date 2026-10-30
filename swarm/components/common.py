"""Container-side bootstrap shared by the gateway, coordinator and stage workers.

Every container receives the same scenario and plan as JSON in its environment,
plus the one RUN_EPOCH that all components measure simulated time from. Nothing
is re-derived per container, so a worker's view of the placement is identical to
the gateway's.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from ..core import placement
from ..core.clock import WallClock
from ..core.events import EventLog
from ..core.scenario import Scenario, from_dict


@dataclass
class Ctx:
    run_id: str
    scenario: Scenario
    plan: placement.Plan
    clock: WallClock
    log: EventLog
    node_id: str = ""
    tier: int = -1
    stage: int = -1

    @property
    def scn(self) -> Scenario:
        return self.scenario


def _env(name: str, default: str | None = None) -> str:
    val = os.environ.get(name, default)
    if val is None:
        raise RuntimeError(f"missing required environment variable {name}")
    return val


def bootstrap(src: str) -> Ctx:
    """Build the context for this container from the environment."""
    run_id = _env("RUN_ID")
    run_epoch = float(_env("RUN_EPOCH"))
    log_dir = _env("LOG_DIR", "/logs")
    scn = from_dict(json.loads(_env("SCENARIO_JSON")))
    plan = placement.build(scn)
    clock = WallClock(run_epoch=run_epoch, time_scale=scn.time_scale)

    node_id = os.environ.get("NODE_ID", "")
    log = EventLog(
        path=f"{log_dir}/{src if not node_id else node_id}.jsonl",
        run_id=run_id, engine=scn.engine, mode=scn.mode, src=src, clock=clock,
    )
    ctx = Ctx(run_id=run_id, scenario=scn, plan=plan, clock=clock, log=log,
              node_id=node_id)
    if node_id:
        ctx.tier = plan.node_tiers[node_id]
        ctx.stage = plan.stage_of(node_id)
    return ctx


def worker_url(node_id: str, port: int = 8000) -> str:
    """Workers are reachable by their node id: it is the compose service name."""
    return f"http://{node_id}:{port}"
