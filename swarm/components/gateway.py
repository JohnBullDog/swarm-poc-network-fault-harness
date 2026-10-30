"""Gateway and load generator.

Stands in for the swarm's front door and for the concurrent agents hitting it.
Per the design doc it runs tokenisation, embedding and the output head in real
mode, picks one replica per stage for each request, and retries a failed hop.

Failover model. The gateway chooses the whole path up front and embeds it in the
request; workers forward along it. When a hop fails, the worker reports which
node failed and the gateway re-routes that token with the failed node excluded.
A mid-pipeline failure therefore discards the work already done on earlier
stages, which is the real cost of pipeline failover and is what the report means
by failures propagating as stalls rather than only as lost capacity.

Concurrency. `closed_loop` keeps exactly G streams in flight, which is the
condition the X(G) = X_max * min(1, G/N) ramp is defined under. `poisson` is
available for open-loop arrival where queue growth matters.
"""

from __future__ import annotations

import asyncio
import json
import random
import time

from aiohttp import ClientSession, ClientTimeout, web

from ..core.routing import RetryPolicy, Router
from .common import Ctx, bootstrap, worker_url

PORT = 8200
SELF_URL = "http://gateway:8200"


class HopFailed(Exception):
    """A worker could not reach the next stage, and said which node it was."""

    def __init__(self, failed_node: str | None, failed_stage: int | None,
                 error: str, reporter: str | None = None):
        super().__init__(f"{error} at {failed_node} (stage {failed_stage})")
        self.failed_node = failed_node
        self.failed_stage = failed_stage
        self.error = error
        self.reporter = reporter


class Gateway:
    def __init__(self, ctx: Ctx):
        self.ctx = ctx
        self.scn = ctx.scenario
        self.plan = ctx.plan
        self.wl = self.scn.workload
        self.router = Router(depth=self.plan.depth, policy="round_robin",
                             seed=self.scn.seed)
        self.retry = RetryPolicy(
            max_attempts=self.scn.retry.max_attempts,
            backoff_s=self.scn.retry.backoff_ms / 1000.0,
            timeout_s=self.scn.retry.hop_timeout_ms / 1000.0)
        self.live_by_stage: dict[int, list[str]] = {
            k: list(st.nodes) for k, st in enumerate(self.plan.stages)}
        self.epoch = -1
        self.session: ClientSession | None = None
        self.real = None
        self.rng = random.Random(self.scn.seed)

        # Simulated instant the load generator began. Container boot takes
        # seconds and varies with node count, so the fault schedule is offset by
        # this rather than by the run epoch: a fault "at t=30s" means 30s into
        # the load, not 30s after docker was told to start.
        self.load_start: float | None = None

        self.req_seq = 0
        self.started = 0
        self.ok = 0
        self.failed = 0
        self.tokens_total = 0
        self.inflight = 0
        self._stop = asyncio.Event()

        # In forward-only flow the result arrives as a fresh POST from the last
        # stage, not as the response to the gateway's own request. Each in-flight
        # token therefore waits on a future keyed by (request, token, attempt).
        self.waiting: dict[tuple, asyncio.Future] = {}

    def _key(self, request_id: str, token_idx: int, attempt: int) -> tuple:
        return (request_id, token_idx, attempt)

    # -- pipeline callbacks ----------------------------------------------
    async def handle_result(self, request: web.Request) -> web.Response:
        """The last stage delivering a finished token."""
        try:
            meta = json.loads(request.headers["X-Swarm-Meta"])
        except (KeyError, json.JSONDecodeError):
            raise web.HTTPBadRequest(text="missing or bad X-Swarm-Meta")
        body = await request.read()

        # Log the return hop. The path has D+1 hops -- one gateway egress and D
        # stage egresses, the last of which lands here -- but workers only log
        # the hops they receive. Without this row the hop table accounts for
        # D hops of a D+1 hop path, and summed hop times would fall short of
        # the measured token latency by exactly the last stage's link delay.
        # `stage == depth` marks the return hop into the gateway.
        arrive_wall = self.ctx.clock.wall()
        sent_wall = float(meta.get("sent_wall") or arrive_wall)
        if self.scn.log_hops:
            self.ctx.log.emit(
                "hop", request_id=meta.get("request_id"),
                token_idx=meta.get("token_idx"), stage=self.plan.depth,
                node_id="gateway", tier=None, attempt=meta.get("attempt"),
                queue_wait_s=0.0, compute_s=0.0,
                net_s=max(0.0, arrive_wall - sent_wall),
                total_s=max(0.0, arrive_wall - sent_wall),
                payload_bytes=len(body))

        key = self._key(meta.get("request_id"), meta.get("token_idx"),
                        int(meta.get("attempt") or 1))
        fut = self.waiting.get(key)
        if fut is not None and not fut.done():
            fut.set_result((body, meta))
        return web.Response(status=202)

    async def handle_pipeline_error(self, request: web.Request) -> web.Response:
        """A worker reporting that it could not reach the next stage."""
        body = await request.json()
        key = self._key(body.get("request_id"), body.get("token_idx"),
                        int(body.get("attempt") or 1))
        fut = self.waiting.get(key)
        if fut is not None and not fut.done():
            fut.set_exception(HopFailed(
                failed_node=body.get("failed_node"),
                failed_stage=body.get("failed_stage"),
                error=body.get("error") or "unknown",
                reporter=body.get("reporter")))
        return web.Response(status=202)

    # -- membership -------------------------------------------------------
    async def members_loop(self) -> None:
        """Track the coordinator's replica set.

        Polled rather than pushed: the gateway must not learn of a death any
        faster than the coordinator detected it, or recovery time would be
        measured against information the real system would not have.
        """
        interval = max(0.02, self.scn.coordinator.heartbeat_ms / 1000.0 / 5.0)
        while not self._stop.is_set():
            try:
                async with self.session.get(
                    "http://coordinator:8100/members") as resp:
                    body = await resp.json()
                if body["epoch"] != self.epoch:
                    self.epoch = body["epoch"]
                    self.live_by_stage = {
                        int(k): v for k, v in body["live_by_stage"].items()}
            except Exception:
                pass
            await self.ctx.clock.sleep(interval)

    # -- one token --------------------------------------------------------
    async def send_token(self, request_id: str, token_idx: int,
                         context_len: int, payload: bytes,
                         shape: list[int] | None,
                         state=None) -> tuple[bytes, dict]:
        """Push one token through the pipeline, re-routing around failures.

        Returns (result_bytes, info) where info carries retries and failovers.
        Raises RuntimeError when the attempts are exhausted.
        """
        excluded: set[str] = set()
        retries = 0
        failovers = 0
        last_err = "unknown"

        for attempt in range(1, self.retry.max_attempts + 1):
            if attempt > 1:
                retries += 1
                back = self.retry.backoff_for(attempt)
                if back:
                    await self.ctx.clock.sleep(back)

            path = self.router.route(
                {k: [n for n in v if n not in excluded]
                 for k, v in self.live_by_stage.items()})
            if path is None:
                last_err = "no covered path"
                continue

            # In real mode the payload depends on the path: if this attempt
            # routes to a replica that holds no KV cache for the request, the
            # whole context has to be resent so the stages can re-prefill.
            reprefill = False
            if self.real is not None and state is not None:
                payload, shape, reprefill = self.real.payload_for_path(
                    state, token_idx, path)

            meta = {
                "request_id": request_id, "token_idx": token_idx,
                "context_len": context_len, "path": path, "hop": 0,
                "attempt": attempt, "reply_to": SELF_URL,
                "reprefill": reprefill,
                "sent_wall": self.ctx.clock.wall(),
            }
            if shape is not None:
                meta["shape"] = shape

            key = self._key(request_id, token_idx, attempt)
            fut: asyncio.Future = asyncio.get_running_loop().create_future()
            self.waiting[key] = fut
            try:
                async with self.session.post(
                    f"{worker_url(path[0])}/forward", data=payload,
                    headers={"X-Swarm-Meta": json.dumps(meta)},
                ) as resp:
                    await resp.read()
                    if resp.status >= 400:
                        raise RuntimeError(f"entry HTTP {resp.status}")

                body, rmeta = await asyncio.wait_for(
                    fut, timeout=self.retry.timeout_s)
                return body, {"retries": retries, "failovers": failovers,
                              "path": path, "meta": rmeta}

            except HopFailed as exc:
                if exc.failed_node:
                    excluded.add(exc.failed_node)
                    failovers += 1
                last_err = str(exc)
            except asyncio.TimeoutError:
                # A node died holding the token, so no worker was left to report
                # it. Nothing is excluded here: the coordinator's death
                # detection is what prunes the live set, and pretending the
                # gateway knew sooner would flatter the recovery measurement.
                last_err = "hop timeout"
            except Exception as exc:
                excluded.add(path[0])       # the entry node itself is unreachable
                failovers += 1
                last_err = f"{type(exc).__name__}: {exc}"
            finally:
                self.waiting.pop(key, None)

        raise RuntimeError(last_err)

    # -- one request ------------------------------------------------------
    async def run_request(self, stream_id: int) -> None:
        self.req_seq += 1
        rid = f"r{self.req_seq:06d}"
        self.started += 1
        self.inflight += 1
        t0 = self.ctx.clock.sim()
        ctx_len = self.wl.prompt_tokens
        retries = failovers = tokens_done = 0
        status, err = "ok", None

        self.ctx.log.emit("request_start", request_id=rid, stream_id=stream_id,
                          prompt_tokens=self.wl.prompt_tokens,
                          gen_tokens=self.wl.gen_tokens)

        state = self.real.begin(rid, self.wl.prompt_tokens) if self.real else None
        try:
            for tok in range(self.wl.gen_tokens):
                if self._stop.is_set():
                    status = "aborted"
                    break
                tstart = self.ctx.clock.sim()
                # In real mode the payload depends on the chosen path (a path
                # change forces a re-prefill), so send_token builds it per
                # attempt. Building one here as well would embed the context
                # twice for every token and charge the gateway double.
                payload, shape = (
                    (b"", None) if self.real
                    else (b"\x00" * int(self.scn.profile.activation_bytes), None))
                try:
                    out, info = await self.send_token(rid, tok, ctx_len, payload,
                                                      shape, state)
                except RuntimeError as exc:
                    status, err = "failed", str(exc)
                    break
                retries += info["retries"]
                failovers += info["failovers"]
                if self.real:
                    self.real.accept(state, out, info["meta"])
                tokens_done += 1
                self.tokens_total += 1
                ctx_len += 1
                self.ctx.log.emit(
                    "token_done", request_id=rid, stream_id=stream_id,
                    token_idx=tok, latency_s=self.ctx.clock.sim() - tstart,
                    context_len=ctx_len, retries=info["retries"],
                    failovers=info["failovers"], path=json.dumps(info["path"]))
        finally:
            self.inflight -= 1
            if status == "ok":
                self.ok += 1
            elif status == "failed":
                self.failed += 1
            text = self.real.finish(state) if self.real else None
            self.ctx.log.emit(
                "request_end", request_id=rid, stream_id=stream_id, status=status,
                tokens_done=tokens_done, retries=retries, failovers=failovers,
                latency_s=self.ctx.clock.sim() - t0,
                error=err if err else (text if text else None))

    # -- load generation --------------------------------------------------
    async def closed_loop_stream(self, stream_id: int) -> None:
        """One simulated agent: a new request the instant the last one ends."""
        while not self._stop.is_set():
            await self.run_request(stream_id)
            if self.wl.max_requests and self.started >= self.wl.max_requests:
                return

    async def poisson_source(self) -> None:
        tasks: set[asyncio.Task] = set()
        sid = 0
        while not self._stop.is_set():
            await self.ctx.clock.sleep(self.rng.expovariate(self.wl.request_rate))
            if self.wl.max_requests and self.started >= self.wl.max_requests:
                break
            sid += 1
            t = asyncio.create_task(self.run_request(sid))
            tasks.add(t)
            t.add_done_callback(tasks.discard)

    async def throughput_loop(self) -> None:
        window = 1.0
        last_tokens, last_t = 0, self.ctx.clock.sim()
        while not self._stop.is_set():
            await self.ctx.clock.sleep(window)
            now = self.ctx.clock.sim()
            dt = max(1e-9, now - last_t)
            delta = self.tokens_total - last_tokens
            last_tokens, last_t = self.tokens_total, now
            self.ctx.log.emit(
                "throughput_sample", window_s=dt, tokens=delta,
                tokens_per_s=delta / dt, requests_inflight=self.inflight,
                live_nodes=sum(len(v) for v in self.live_by_stage.values()))

    # -- readiness --------------------------------------------------------
    async def wait_ready(self, timeout_s: float = 300.0) -> bool:
        """Block until every node answers /health and the coordinator sees them all.

        Starting the load before the stack is up would put container startup
        skew into the first requests' latencies. The timeout is generous because
        in real mode each worker spends a minute or more reading and converting
        its block of weights, and reports 503 here until it can serve.
        """
        deadline = time.time() + timeout_s
        want = set(self.plan.nodes)
        while time.time() < deadline:
            healthy = set()
            for node in want:
                try:
                    async with self.session.get(
                        f"{worker_url(node)}/health",
                        timeout=ClientTimeout(total=2.0)) as resp:
                        if resp.status == 200:
                            healthy.add(node)
                except Exception:
                    pass
            if healthy == want:
                try:
                    async with self.session.get(
                        "http://coordinator:8100/members") as resp:
                        body = await resp.json()
                    seen = sum(len(v) for v in body["live_by_stage"].values())
                    if seen == len(want) and body["covered"]:
                        return True
                except Exception:
                    pass
            await asyncio.sleep(0.5)
        return False

    # -- lifecycle --------------------------------------------------------
    async def run(self) -> None:
        self.session = ClientSession(
            timeout=ClientTimeout(total=self.retry.timeout_s))
        if self.scn.mode == "real":
            from .realgateway import RealGateway
            self.real = RealGateway(self.ctx)
            self.real.load()

        ready = await self.wait_ready()
        self.ctx.log.emit("note", message="stack ready" if ready
                          else "stack NOT fully ready; starting anyway",
                          detail=json.dumps({"epoch": self.epoch}))

        asyncio.create_task(self.members_loop())
        asyncio.create_task(self.throughput_loop())

        self.load_start = self.ctx.clock.sim()
        preds = self.plan.predictions()
        self.ctx.log.emit(
            "run_start", scenario=self.scn.name, depth=self.plan.depth,
            width=self.scn.composition.width, n_nodes=len(self.plan.nodes),
            concurrency=self.wl.concurrency, seed=self.scn.seed)
        self.ctx.log.emit("note", message="predictions",
                          detail=json.dumps(preds, default=float))

        if self.wl.arrival == "closed_loop":
            streams = [asyncio.create_task(self.closed_loop_stream(i))
                       for i in range(self.wl.concurrency)]
        else:
            streams = [asyncio.create_task(self.poisson_source())]

        await asyncio.sleep(self.scn.duration_s / self.scn.time_scale)
        self._stop.set()
        await asyncio.gather(*streams, return_exceptions=True)

        self.ctx.log.emit("run_end", reason="duration elapsed",
                          requests_started=self.started, requests_ok=self.ok,
                          requests_failed=self.failed)
        await self.session.close()


async def main() -> None:
    ctx = bootstrap("gateway")
    gw = Gateway(ctx)

    # /result and /pipeline_error are the pipeline's return path; /status is for
    # the host runner, so it can watch progress without reading the log volume
    # mid-run.
    app = web.Application(client_max_size=1 << 26)
    app.add_routes([
        web.post("/result", gw.handle_result),
        web.post("/pipeline_error", gw.handle_pipeline_error),
    ])
    app.add_routes([web.get("/status", lambda _: web.json_response({
        "started": gw.started, "ok": gw.ok, "failed": gw.failed,
        "tokens": gw.tokens_total, "inflight": gw.inflight,
        "ready": gw.load_start is not None,
        "load_start_sim_ts": gw.load_start,
        "done": gw._stop.is_set(), "sim_ts": ctx.clock.sim()}))])
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    await web.TCPSite(runner, "0.0.0.0", PORT).start()

    try:
        await gw.run()
    finally:
        ctx.log.close()
        # Hold the container open briefly so the runner can read the final
        # /status and know the run ended cleanly rather than by a crash.
        await asyncio.sleep(10)


if __name__ == "__main__":
    asyncio.run(main())
