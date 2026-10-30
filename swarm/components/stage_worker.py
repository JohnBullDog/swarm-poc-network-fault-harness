"""Stage worker: holds one pipeline stage, computes, forwards to the next stage.

Flow is forward-only: gateway -> s0 -> s1 -> ... -> s(D-1) -> gateway. A worker
acknowledges the incoming activation immediately, queues it, computes it, and
pushes it onward; the last stage posts the result to the gateway directly.

This matters for measurement. An earlier version nested the HTTP calls so the
response unwound back through every stage, which made a round trip pay 2D
one-way link delays instead of D. That is an artifact of the transport, not a
property of the system, and it would have doubled the slope of the delay sweep.
Forward-only flow costs exactly D stage egresses plus the gateway's entry hop.

Wire format. Metadata travels in the `X-Swarm-Meta` header and the activation
travels as the raw body, so the bytes on the wire are exactly the activation
size `a`. Base64 in a JSON envelope would inflate the payload by a third and
corrupt the a/beta serialisation term the delay sweep measures.

Timing. Hops are timed across containers using the shared host clock: the sender
stamps `sent_wall` and the receiver subtracts it, giving true one-way network
time including the netem delay. This is what the design doc's shared-clock note
exists to enable.

What synthetic compute sleeps for: only the memory-bound term,
(W_k*b + KV_k)/B_k plus fitted overhead. The network term is NOT slept -- tc
netem applies it for real on the wire, and sleeping it too would double-count.

Concurrency. A stage is memory-bandwidth-bound, so concurrent requests cannot
each have full bandwidth. Activations wait in a queue and are served by
`servers_per_node` workers, so queueing is produced rather than assumed; that
queueing is what makes the X(G) = X_max * min(1, G/N) ramp emerge.
"""

from __future__ import annotations

import asyncio
import json
import signal

from aiohttp import ClientSession, ClientTimeout, web

from .common import Ctx, bootstrap, worker_url

PORT = 8000


class StageWorker:
    def __init__(self, ctx: Ctx):
        self.ctx = ctx
        self.scn = ctx.scenario
        self.plan = ctx.plan
        self.sp = ctx.plan.stages[ctx.stage]
        self.mem_bw = self.scn.mem_bw_of(ctx.tier)
        self.n_servers = max(1, self.scn.synthetic.servers_per_node)

        self.queue: asyncio.Queue = asyncio.Queue()
        self.inflight = 0
        self.tokens_done = 0
        self.busy_s = 0.0
        self.hb_seq = 0
        self.session: ClientSession | None = None
        self._sends: set = set()          # in-flight handoffs, kept from GC
        self.real = None
        self.ready = False
        self._stop = asyncio.Event()

    # -- compute ----------------------------------------------------------
    def compute_seconds(self, context_len: int) -> float:
        """The memory-bound service time for one token on this stage.

        This is the bare cost-model time: (W_k*b + KV_k)/B_k, and nothing else.
        The fitted `stage_overhead_ms` is deliberately NOT added here. The
        container runtime adds its own overhead on top of this sleep all by
        itself, and that overhead is what the fit measures; sleeping the fitted
        value as well would make the calibration circular and divergent.
        """
        kv = self.sp.share * context_len * self.scn.profile.kv_bytes_per_token
        return (self.sp.weight_bytes + kv) / self.mem_bw

    async def do_compute(self, payload: bytes, meta: dict) -> bytes:
        if self.scn.mode == "real":
            return await asyncio.get_running_loop().run_in_executor(
                None, self.real.forward, payload, meta)
        secs = self.compute_seconds(int(meta.get("context_len") or 0))
        if self.scn.synthetic.busy_compute:
            await asyncio.get_running_loop().run_in_executor(
                None, self.ctx.clock.busy_sleep, secs)
        else:
            await self.ctx.clock.sleep(secs)
        return b"\x00" * int(self.scn.profile.activation_bytes)

    # -- intake -----------------------------------------------------------
    async def handle_forward(self, request: web.Request) -> web.Response:
        """Accept an activation and queue it. Returns as soon as it is queued.

        The sender is not made to wait for this stage's compute: in a forward-only
        pipeline the sender has no further interest in the token, and holding its
        connection open would make one stage's queue look like its upstream's.
        """
        try:
            meta = json.loads(request.headers["X-Swarm-Meta"])
        except (KeyError, json.JSONDecodeError):
            raise web.HTTPBadRequest(text="missing or bad X-Swarm-Meta")
        payload = await request.read()
        meta["_arrive_sim"] = self.ctx.clock.sim()
        meta["_arrive_wall"] = self.ctx.clock.wall()
        meta["_payload_bytes"] = len(payload)
        self.queue.put_nowait((meta, payload))
        return web.Response(status=202)

    # -- service ----------------------------------------------------------
    async def server_loop(self) -> None:
        """One of `servers_per_node` servers draining the stage's queue."""
        while not self._stop.is_set():
            meta, payload = await self.queue.get()
            try:
                await self.serve_one(meta, payload)
            except Exception as exc:                       # never kill a server
                self.ctx.log.emit(
                    "hop_error", request_id=meta.get("request_id"),
                    token_idx=meta.get("token_idx"), stage=self.ctx.stage,
                    node_id=self.ctx.node_id, tier=self.ctx.tier,
                    attempt=meta.get("attempt"),
                    error=f"server loop: {type(exc).__name__}: {exc}",
                    elapsed_s=None)
            finally:
                self.queue.task_done()

    async def serve_one(self, meta: dict, payload: bytes) -> None:
        arrive = meta.pop("_arrive_sim")
        arrive_wall = meta.pop("_arrive_wall")
        payload_bytes = meta.pop("_payload_bytes")
        sent_wall = float(meta.get("sent_wall") or arrive_wall)
        net_s = max(0.0, arrive_wall - sent_wall)

        start = self.ctx.clock.sim()
        queue_wait_s = max(0.0, start - arrive)
        self.inflight += 1
        try:
            out = await self.do_compute(payload, meta)
        finally:
            self.inflight -= 1
        compute_s = self.ctx.clock.sim() - start
        self.busy_s += compute_s
        self.tokens_done += 1

        if self.scn.log_hops:
            self.ctx.log.emit(
                "hop", request_id=meta.get("request_id"),
                token_idx=meta.get("token_idx"), stage=self.ctx.stage,
                node_id=self.ctx.node_id, tier=self.ctx.tier,
                attempt=meta.get("attempt"), queue_wait_s=queue_wait_s,
                compute_s=compute_s, net_s=net_s,
                total_s=net_s + queue_wait_s + compute_s,
                payload_bytes=payload_bytes)

        path = meta.get("path") or []
        hop = int(meta.get("hop") or 0)
        reply_to = meta.get("reply_to") or "http://gateway:8200"

        fwd = {k: v for k, v in meta.items() if not k.startswith("_")}
        if hop + 1 < len(path):
            target = f"{worker_url(path[hop + 1])}/forward"
            failed_node, failed_stage = path[hop + 1], self.ctx.stage + 1
        else:
            target = f"{reply_to}/result"
            fwd["final_stage"] = self.ctx.stage
            failed_node, failed_stage = None, None
        fwd["hop"] = hop + 1

        # Hand the activation off in a task rather than awaiting the downstream
        # acknowledgement here.
        #
        # Awaiting it inside the server loop put a full link round trip inside
        # this stage's service time: the loop could not start the next token
        # until the next stage had acknowledged this one. Measured, that capped
        # a stage at 1/(compute + 2*hop) instead of 1/compute -- a plateau of
        # ~57 tok/s where the node could do ~88 -- and piled all the queueing on
        # stage 0 while later stages sat idle. The cost model occupies a stage
        # for its compute only; transit overlaps with serving the next token.
        task = asyncio.create_task(
            self.deliver(target, out, fwd, meta, failed_node, failed_stage,
                         reply_to, arrive))
        self._sends.add(task)
        task.add_done_callback(self._sends.discard)

    async def deliver(self, target: str, out: bytes, fwd: dict, meta: dict,
                      failed_node: str | None, failed_stage: int | None,
                      reply_to: str, arrive: float) -> None:
        """Push the activation onward. Runs outside the server loop."""
        fwd["sent_wall"] = self.ctx.clock.wall()
        try:
            async with self.session.post(
                target, data=out,
                headers={"X-Swarm-Meta": json.dumps(fwd)},
            ) as resp:
                await resp.read()
                if resp.status >= 400:
                    raise RuntimeError(f"HTTP {resp.status}")
        except Exception as exc:
            self.ctx.log.emit(
                "hop_error", request_id=meta.get("request_id"),
                token_idx=meta.get("token_idx"),
                stage=failed_stage if failed_stage is not None else self.ctx.stage,
                node_id=failed_node or "gateway", tier=self.plan.node_tiers.get(
                    failed_node) if failed_node else None,
                attempt=meta.get("attempt"),
                error=f"{type(exc).__name__}: {exc}",
                elapsed_s=self.ctx.clock.sim() - arrive)
            # Tell the gateway directly: in a forward-only pipeline there is no
            # caller left holding a connection to report the failure to.
            await self.report_failure(reply_to, meta, failed_node, failed_stage,
                                      type(exc).__name__)

    async def report_failure(self, reply_to: str, meta: dict,
                             failed_node: str | None, failed_stage: int | None,
                             error: str) -> None:
        body = {
            "request_id": meta.get("request_id"),
            "token_idx": meta.get("token_idx"),
            "attempt": meta.get("attempt"),
            "failed_node": failed_node,
            "failed_stage": failed_stage,
            "reporter": self.ctx.node_id,
            "error": error,
        }
        try:
            async with self.session.post(f"{reply_to}/pipeline_error",
                                         json=body) as resp:
                await resp.read()
        except Exception:
            pass        # the gateway's own hop timeout is the backstop

    # -- liveness and reporting -------------------------------------------
    async def handle_health(self, _: web.Request) -> web.Response:
        """503 until the stage can actually serve.

        In real mode `load()` takes a minute or more to read and convert this
        stage's weights. The server and the heartbeat start first, so the
        coordinator sees a loading node as alive rather than dead, and this
        endpoint is what keeps the gateway from routing to it meanwhile. The
        architecture notes model exactly this as a Loading state: alive, but
        admitting no work.
        """
        if not self.ready:
            return web.json_response(
                {"node_id": self.ctx.node_id, "ok": False, "state": "loading"},
                status=503)
        return web.json_response({"node_id": self.ctx.node_id, "ok": True})

    async def handle_stats(self, _: web.Request) -> web.Response:
        return web.json_response({
            "node_id": self.ctx.node_id, "tier": self.ctx.tier,
            "stage": self.ctx.stage, "inflight": self.inflight,
            "queue_depth": self.queue.qsize(), "tokens_done": self.tokens_done,
            "busy_s": self.busy_s})

    async def heartbeat_loop(self) -> None:
        url = "http://coordinator:8100/heartbeat"
        interval = self.scn.coordinator.heartbeat_ms / 1000.0
        while not self._stop.is_set():
            self.hb_seq += 1
            body = {"node_id": self.ctx.node_id, "tier": self.ctx.tier,
                    "stage": self.ctx.stage, "seq": self.hb_seq,
                    "sent_wall": self.ctx.clock.wall()}
            try:
                async with self.session.post(url, json=body) as resp:
                    await resp.read()
                self.ctx.log.emit("heartbeat", node_id=self.ctx.node_id,
                                  tier=self.ctx.tier, stage=self.ctx.stage,
                                  seq=self.hb_seq)
            except Exception:
                pass            # a partitioned node simply fails to be heard
            await self.ctx.clock.sleep(interval)

    async def sample_loop(self) -> None:
        interval = self.scn.coordinator.sample_interval_ms / 1000.0
        last_busy, last_t = 0.0, self.ctx.clock.sim()
        while not self._stop.is_set():
            await self.ctx.clock.sleep(interval)
            now = self.ctx.clock.sim()
            window = max(1e-9, now - last_t)
            busy_frac = (self.busy_s - last_busy) / (window * self.n_servers)
            last_busy, last_t = self.busy_s, now
            self.ctx.log.emit(
                "node_sample", node_id=self.ctx.node_id, tier=self.ctx.tier,
                stage=self.ctx.stage, inflight=self.inflight,
                queue_depth=self.queue.qsize(), busy_frac=round(busy_frac, 6),
                tokens_done=self.tokens_done, hb_gap_s=None)

    # -- lifecycle --------------------------------------------------------
    async def start(self) -> None:
        self.session = ClientSession(
            timeout=ClientTimeout(total=self.scn.retry.hop_timeout_ms / 1000.0))

        app = web.Application(client_max_size=1 << 26)
        app.add_routes([
            web.post("/forward", self.handle_forward),
            web.get("/health", self.handle_health),
            web.get("/stats", self.handle_stats),
        ])
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        await web.TCPSite(runner, "0.0.0.0", PORT).start()

        for _ in range(self.n_servers):
            asyncio.create_task(self.server_loop())
        asyncio.create_task(self.heartbeat_loop())
        asyncio.create_task(self.sample_loop())

        # Load the model only after the server and the heartbeat are up.
        # Reading and converting this stage's weights takes a minute or more,
        # and loading first meant the coordinator declared every node dead
        # before any of them had served a token. /health reports 503 until this
        # finishes, so the gateway still will not route here early.
        if self.scn.mode == "real":
            from .realstage import RealStage      # torch only in the real image
            self.real = RealStage(self.ctx)
            await asyncio.get_running_loop().run_in_executor(None, self.real.load)
        self.ready = True

        self.ctx.log.emit("note", message="worker up", detail=json.dumps({
            "node_id": self.ctx.node_id, "tier": self.ctx.tier,
            "stage": self.ctx.stage,
            "layers": [self.sp.layer_start, self.sp.layer_end],
            "share": round(self.sp.share, 6),
            "servers": self.n_servers,
            "compute_ms_at_s0": round(self.compute_seconds(0) * 1000, 4)}))
        self.ctx.log.flush()


async def main() -> None:
    ctx = bootstrap("stage_worker")
    worker = StageWorker(ctx)
    await worker.start()

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
        # Flush before the container dies: buffered events are measurements.
        worker._stop.set()
        ctx.log.close()


if __name__ == "__main__":
    asyncio.run(main())
