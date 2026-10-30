"""Coordinator: membership by heartbeat, death detection, and the live replica set.

Trusted for liveness and coordination only, as the architecture notes specify.
Placement is static per scenario: when a node disappears the coordinator removes
it from the replica set the gateway routes over, and nothing is re-placed.

Detection time is a headline measurement, so the fault controller tells the
coordinator the simulated instant each fault was applied (`/fault`). The
coordinator then reports detection latency as
`detected_sim_ts - fault_sim_ts`, measured rather than inferred from the
heartbeat interval.
"""

from __future__ import annotations

import asyncio
import signal
import json

from aiohttp import web

from ..core.membership import Membership
from .common import Ctx, bootstrap

PORT = 8100


class Coordinator:
    def __init__(self, ctx: Ctx):
        self.ctx = ctx
        self.scn = ctx.scenario
        self.plan = ctx.plan
        self.mem = Membership(
            death_after_s=self.scn.coordinator.death_after_ms / 1000.0)
        now = ctx.clock.sim()
        for node, tier in self.plan.node_tiers.items():
            self.mem.add(node, tier, self.plan.stage_of(node), now)
        # node -> simulated instant of the fault the controller last applied
        self.pending_fault: dict[str, float] = {}
        self.epoch = 0          # bumped on every membership change

    # -- endpoints --------------------------------------------------------
    async def handle_heartbeat(self, request: web.Request) -> web.Response:
        body = await request.json()
        node = body["node_id"]
        now = self.ctx.clock.sim()
        revived = self.mem.heartbeat(node, now)
        if revived:
            self.epoch += 1
            st = self.mem.nodes[node]
            self.ctx.log.emit("membership", node_id=node, tier=st.tier,
                              stage=st.stage, event="join", gap_s=None,
                              fault_sim_ts=None, detection_s=None,
                              reason="heartbeat resumed")
            self.pending_fault.pop(node, None)
        return web.json_response({"ok": True, "epoch": self.epoch})

    async def handle_members(self, _: web.Request) -> web.Response:
        """The replica set the gateway routes over."""
        return web.json_response({
            "epoch": self.epoch,
            "live_by_stage": {str(k): v for k, v in self.mem.live_by_stage().items()},
            "covered": self.mem.covered(self.plan.depth),
        })

    async def handle_fault(self, request: web.Request) -> web.Response:
        """The fault controller declares when and to whom it applied a fault.

        This is the ground truth that detection latency is measured against. The
        coordinator never uses it to decide liveness -- only heartbeats do that.
        """
        body = await request.json()
        node = body["node_id"]
        at = float(body["fault_sim_ts"])
        action = body.get("action", "")
        st = self.mem.nodes.get(node)
        if action in ("disconnect", "pause", "kill"):
            self.pending_fault[node] = at
        self.ctx.log.emit("fault_applied", node_id=node,
                          tier=st.tier if st else None,
                          stage=st.stage if st else None,
                          action=action, detail=json.dumps(body.get("detail") or {}))
        return web.json_response({"ok": True})

    async def handle_health(self, _: web.Request) -> web.Response:
        return web.json_response({"ok": True, "epoch": self.epoch})

    # -- detection loop ---------------------------------------------------
    async def detect_loop(self) -> None:
        """Mark nodes dead once their heartbeat gap exceeds the threshold.

        Polled at a fraction of the heartbeat interval so the measured detection
        time is dominated by the configured threshold rather than by this loop's
        own granularity.
        """
        interval = max(0.02, self.scn.coordinator.heartbeat_ms / 1000.0 / 5.0)
        while True:
            now = self.ctx.clock.sim()
            for node, gap in self.mem.tick(now):
                self.epoch += 1
                st = self.mem.nodes[node]
                fault_at = self.pending_fault.get(node)
                detection = (now - fault_at) if fault_at is not None else None
                self.ctx.log.emit(
                    "membership", node_id=node, tier=st.tier, stage=st.stage,
                    event="death_detected", gap_s=gap, fault_sim_ts=fault_at,
                    detection_s=detection,
                    reason=f"no heartbeat for {gap:.3f}s")
            await self.ctx.clock.sleep(interval)

    async def sample_loop(self) -> None:
        interval = self.scn.coordinator.sample_interval_ms / 1000.0
        while True:
            await self.ctx.clock.sleep(interval)
            now = self.ctx.clock.sim()
            for node, st in self.mem.nodes.items():
                self.ctx.log.emit(
                    "node_sample", node_id=node, tier=st.tier, stage=st.stage,
                    inflight=None, queue_depth=None, busy_frac=None,
                    tokens_done=None, hb_gap_s=round(now - st.last_seen, 6))


async def main() -> None:
    ctx = bootstrap("coordinator")
    coord = Coordinator(ctx)
    app = web.Application()
    app.add_routes([
        web.post("/heartbeat", coord.handle_heartbeat),
        web.get("/members", coord.handle_members),
        web.post("/fault", coord.handle_fault),
        web.get("/health", coord.handle_health),
    ])
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    await web.TCPSite(runner, "0.0.0.0", PORT).start()
    asyncio.create_task(coord.detect_loop())
    asyncio.create_task(coord.sample_loop())
    ctx.log.emit("note", message="coordinator up", detail=json.dumps({
        "nodes": len(coord.mem.nodes),
        "death_after_s": coord.mem.death_after_s}))
    ctx.log.flush()

    stopping = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, stopping.set)
        except NotImplementedError:
            pass
    try:
        await stopping.wait()
    finally:
        ctx.log.close()     # flush before the container dies


if __name__ == "__main__":
    asyncio.run(main())
