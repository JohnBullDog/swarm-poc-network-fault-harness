"""The virtual-time engine: the same swarm, with a simulated clock.

Why it exists. Container mode is the measurement of record, but this host caps
it at roughly sixty nodes, while the research report's worked examples are at
N = 1000 and N = 2750. The virtual engine runs those sizes in seconds of real
time by advancing a simulated clock instead of sleeping.

What it shares with container mode. Everything that makes a decision:
`costmodel` for stage times, `placement` for the layer split, `routing` for
replica selection and retry, `membership` for death detection, `churn` for the
fault schedule, and `events` for the output schema. The two engines differ only
in how time passes and how a message crosses a link. That is what makes the
mode check a comparison of engines rather than of configurations.

What it models rather than reproduces:

  link       base delay, per-hop jitter, serialisation (payload/rate), the
             fitted hop overhead, and packet loss as retransmission delay.
  node       a queue with `servers_per_node` servers; service time is the cost
             model's t_k, so a stage is memory-bound exactly as in synthetic
             container mode.
  failure    a send to a dead node fails fast, as a refused connection does; a
             node that dies holding a token loses it, and the gateway learns
             only by timing out.

What it does not model, and what therefore limits its use: real TCP dynamics
(congestion control, window growth, head-of-line blocking), the host's own CPU
contention between co-located containers, and the Docker bridge. Those are
exactly the effects container mode exists to measure, and the calibration in
tools/compare_engines.py is what says how far the two may be trusted to agree.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

from ..core import churn as churn_mod
from ..core import placement
from ..core.clock import VirtualClock
from ..core.events import EventLog
from ..core.membership import Membership
from ..core.routing import RetryPolicy, Router
from ..core.scenario import Scenario
from .scheduler import Env, Event

MTU_PAYLOAD = 1448          # bytes of payload in one TCP segment over Ethernet
CONNECT_FAIL_S = 0.002      # a refused connection returns about this fast


# ---------------------------------------------------------------------------
# link
# ---------------------------------------------------------------------------
@dataclass
class Link:
    """One directed link: a sender's egress, matching how tc netem shapes."""

    delay_s: float
    jitter_s: float
    loss_frac: float
    rate_bytes: float
    overhead_s: float
    rng: random.Random

    def transit(self, payload_bytes: int) -> float:
        d = self.delay_s + self.overhead_s + payload_bytes / self.rate_bytes
        if self.jitter_s > 0:
            # netem's `distribution normal` draws a normal deviate around the
            # base delay. Clamped at zero: a link cannot deliver before it sent.
            d = max(0.0, d + self.rng.gauss(0.0, self.jitter_s))
        if self.loss_frac > 0:
            packets = max(1, math.ceil(payload_bytes / MTU_PAYLOAD))
            lost = sum(1 for _ in range(packets)
                       if self.rng.random() < self.loss_frac)
            if lost:
                # Fast retransmit: one round trip per lost segment. This is the
                # optimistic end of TCP's behaviour -- a tail loss or a window
                # collapse costs an RTO, which is far worse -- so loss results
                # from this engine are a lower bound on the real penalty.
                d += lost * 2.0 * (self.delay_s + self.overhead_s)
        return d


# ---------------------------------------------------------------------------
# node
# ---------------------------------------------------------------------------
@dataclass
class VNode:
    node_id: str
    tier: int
    stage: int
    servers: int
    alive: bool = True
    busy: int = 0
    queue: list = field(default_factory=list)
    tokens_done: int = 0
    busy_s: float = 0.0

    @property
    def queue_depth(self) -> int:
        return len(self.queue)


# ---------------------------------------------------------------------------
# engine
# ---------------------------------------------------------------------------
class VirtualEngine:
    def __init__(self, scn: Scenario, run_dir: Path, run_id: str):
        self.scn = scn
        self.plan = placement.build(scn)
        self.schedule, self.suppressed = churn_mod.generate(scn, self.plan)
        self.env = Env()
        self.clock = VirtualClock(self.env)
        self.rng = random.Random(scn.seed)

        raw = run_dir / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        self.log = EventLog(raw / "virtual.jsonl", run_id, "virtual", scn.mode,
                            "virtual_engine", self.clock, flush_every=4096,
                            flush_interval_s=1e9)

        self.nodes: dict[str, VNode] = {
            n: VNode(n, self.plan.node_tiers[n], self.plan.stage_of(n),
                     max(1, scn.synthetic.servers_per_node))
            for n in self.plan.nodes
        }
        self.links: dict[str, Link] = {}
        for node, tier in self.plan.node_tiers.items():
            link = scn.link_for(tier)
            self.links[node] = Link(
                link.delay_s, link.jitter_s, link.loss_frac, link.rate_bytes,
                scn.synthetic.hop_overhead_ms / 1000.0,
                random.Random(f"{scn.seed}:link:{node}"))
        gl = scn.gateway_link
        self.links["gateway"] = Link(
            gl.delay_s, gl.jitter_s, gl.loss_frac, gl.rate_bytes,
            scn.synthetic.hop_overhead_ms / 1000.0,
            random.Random(f"{scn.seed}:link:gateway"))

        self.mem = Membership(death_after_s=scn.coordinator.death_after_ms / 1000.0)
        for node, tier in self.plan.node_tiers.items():
            self.mem.add(node, tier, self.plan.stage_of(node), 0.0)
        self.live_by_stage: dict[int, list[str]] = {
            k: list(st.nodes) for k, st in enumerate(self.plan.stages)}
        self.epoch = 0
        self.pending_fault: dict[str, float] = {}

        self.router = Router(depth=self.plan.depth, policy="round_robin",
                             seed=scn.seed)
        self.retry = RetryPolicy(
            max_attempts=scn.retry.max_attempts,
            backoff_s=scn.retry.backoff_ms / 1000.0,
            timeout_s=scn.retry.hop_timeout_ms / 1000.0)

        self.req_seq = 0
        self.started = self.ok = self.failed = 0
        self.tokens_total = 0
        self.inflight = 0
        self.load_start = 0.0
        self.stopped = False

    # -- node service ----------------------------------------------------
    def stage_service_s(self, node: str, stage: int, context_len: int) -> float:
        """t_k without its link term: the link is charged separately per hop."""
        st = self.plan.stages[stage]
        scn = self.scn
        tier = self.nodes[node].tier
        kv = st.share * context_len * scn.profile.kv_bytes_per_token
        return ((st.weight_bytes + kv) / scn.mem_bw_of(tier)
                + scn.synthetic.stage_overhead_ms / 1000.0)

    def serve(self, node: str, stage: int, context_len: int, meta: dict):
        """Queue at a node, be served, and report the hop. A process."""
        vn = self.nodes[node]
        arrive = self.env.now
        if vn.busy >= vn.servers:
            waiter = self.env.event("queue")
            vn.queue.append(waiter)
            yield waiter
        vn.busy += 1
        start = self.env.now
        service = self.stage_service_s(node, stage, context_len)
        yield self.env.timeout(service)
        vn.busy -= 1
        vn.busy_s += service
        vn.tokens_done += 1
        if vn.queue:
            vn.queue.pop(0).succeed()

        net_s = meta.pop("_net_s", 0.0)
        if self.scn.log_hops:
            self.log.emit(
                "hop", request_id=meta["request_id"], token_idx=meta["token_idx"],
                stage=stage, node_id=node, tier=vn.tier,
                attempt=meta["attempt"], queue_wait_s=start - arrive,
                compute_s=service, net_s=net_s,
                total_s=net_s + (start - arrive) + service,
                payload_bytes=meta["payload_bytes"])
        return True

    # -- one token through the pipeline ----------------------------------
    def push_token(self, meta: dict, path: list[str], context_len: int,
                   result: Event):
        """Walk the activation forward. Fails the token by not firing `result`."""
        payload = meta["payload_bytes"]
        sender = "gateway"

        for hop, node in enumerate(path):
            # A send to a node already down fails fast, as a refused connection
            # does, and the gateway is told after one link traversal back.
            if not self.nodes[node].alive:
                yield self.env.timeout(CONNECT_FAIL_S
                                       + self.links[sender].transit(0))
                self.log.emit(
                    "hop_error", request_id=meta["request_id"],
                    token_idx=meta["token_idx"], stage=self.nodes[node].stage,
                    node_id=node, tier=self.nodes[node].tier,
                    attempt=meta["attempt"], error="ConnectionRefused",
                    elapsed_s=self.env.now - meta["t_start"])
                result.succeed({"ok": False, "failed_node": node})
                return

            transit = self.links[sender].transit(payload)
            yield self.env.timeout(transit)
            meta["_net_s"] = transit

            if not self.nodes[node].alive:
                # Died while the activation was in flight: nobody is left to
                # report it, so the gateway can only time out.
                return
            yield from self.serve(node, hop, context_len, meta)
            if not self.nodes[node].alive:
                return
            sender = node

        # Last stage back to the gateway.
        transit = self.links[sender].transit(payload)
        yield self.env.timeout(transit)
        if self.scn.log_hops:
            self.log.emit(
                "hop", request_id=meta["request_id"],
                token_idx=meta["token_idx"], stage=self.plan.depth,
                node_id="gateway", tier=None, attempt=meta["attempt"],
                queue_wait_s=0.0, compute_s=0.0, net_s=transit,
                total_s=transit, payload_bytes=payload)
        result.succeed({"ok": True})

    def send_token(self, request_id: str, token_idx: int, context_len: int):
        """One token, with retry and failover. Returns (ok, retries, failovers)."""
        excluded: set[str] = set()
        retries = failovers = 0
        payload = int(self.scn.profile.activation_bytes)

        for attempt in range(1, self.retry.max_attempts + 1):
            if attempt > 1:
                retries += 1
                back = self.retry.backoff_for(attempt)
                if back:
                    yield self.env.timeout(back)

            path = self.router.route(
                {k: [n for n in v if n not in excluded]
                 for k, v in self.live_by_stage.items()})
            if path is None:
                yield self.env.timeout(self.retry.backoff_s)
                continue

            meta = {"request_id": request_id, "token_idx": token_idx,
                    "attempt": attempt, "payload_bytes": payload,
                    "t_start": self.env.now}
            result = self.env.event("token_result")
            self.env.process(self.push_token(meta, path, context_len, result))
            timeout = self.env.timeout(self.retry.timeout_s, {"timeout": True})
            winner = yield self.env.any_of([result, timeout])

            value = winner.value
            if isinstance(value, dict) and value.get("ok"):
                return True, retries, failovers, path
            if isinstance(value, dict) and value.get("failed_node"):
                excluded.add(value["failed_node"])
                failovers += 1
        return False, retries, failovers, None

    # -- requests and load ------------------------------------------------
    def run_request(self, stream_id: int):
        self.req_seq += 1
        rid = f"r{self.req_seq:06d}"
        self.started += 1
        self.inflight += 1
        t0 = self.env.now
        ctx = self.scn.workload.prompt_tokens
        retries = failovers = done = 0
        status = "ok"

        self.log.emit("request_start", request_id=rid, stream_id=stream_id,
                      prompt_tokens=self.scn.workload.prompt_tokens,
                      gen_tokens=self.scn.workload.gen_tokens)

        for tok in range(self.scn.workload.gen_tokens):
            if self.stopped:
                status = "aborted"
                break
            tstart = self.env.now
            ok, r, f, path = yield from self.send_token(rid, tok, ctx)
            retries += r
            failovers += f
            if not ok:
                status = "failed"
                break
            done += 1
            self.tokens_total += 1
            ctx += 1
            self.log.emit("token_done", request_id=rid, stream_id=stream_id,
                          token_idx=tok, latency_s=self.env.now - tstart,
                          context_len=ctx, retries=r, failovers=f,
                          path=json.dumps(path))

        self.inflight -= 1
        if status == "ok":
            self.ok += 1
        elif status == "failed":
            self.failed += 1
        self.log.emit("request_end", request_id=rid, stream_id=stream_id,
                      status=status, tokens_done=done, retries=retries,
                      failovers=failovers, latency_s=self.env.now - t0,
                      error=None)

    def stream(self, stream_id: int):
        while not self.stopped:
            yield from self.run_request(stream_id)
            if (self.scn.workload.max_requests
                    and self.started >= self.scn.workload.max_requests):
                return

    def poisson_source(self):
        sid = 0
        rate = self.scn.workload.request_rate
        while not self.stopped:
            yield self.env.timeout(self.rng.expovariate(rate))
            sid += 1
            self.env.process(self.run_request(sid))

    # -- control plane ----------------------------------------------------
    def heartbeat(self, node: str):
        interval = self.scn.coordinator.heartbeat_ms / 1000.0
        link = self.links[node]
        while not self.stopped:
            if self.nodes[node].alive:
                # The heartbeat crosses the same shaped link as everything else,
                # so a slow or lossy node is also slow to be heard from.
                yield self.env.timeout(link.transit(256))
                if self.nodes[node].alive:
                    if self.mem.heartbeat(node, self.env.now):
                        self.epoch += 1
                        self.refresh_live()
                        st = self.mem.nodes[node]
                        self.log.emit("membership", node_id=node, tier=st.tier,
                                      stage=st.stage, event="join", gap_s=None,
                                      fault_sim_ts=None, detection_s=None,
                                      reason="heartbeat resumed")
                        self.pending_fault.pop(node, None)
                yield self.env.timeout(max(0.0, interval - link.transit(256)))
            else:
                yield self.env.timeout(interval)

    def detect(self):
        interval = max(0.02, self.scn.coordinator.heartbeat_ms / 1000.0 / 5.0)
        while not self.stopped:
            yield self.env.timeout(interval)
            for node, gap in self.mem.tick(self.env.now):
                self.epoch += 1
                self.refresh_live()
                st = self.mem.nodes[node]
                at = self.pending_fault.get(node)
                self.log.emit(
                    "membership", node_id=node, tier=st.tier, stage=st.stage,
                    event="death_detected", gap_s=gap, fault_sim_ts=at,
                    detection_s=(self.env.now - at) if at is not None else None,
                    reason=f"no heartbeat for {gap:.3f}s")

    def refresh_live(self) -> None:
        self.live_by_stage = self.mem.live_by_stage()
        for k in range(self.plan.depth):
            self.live_by_stage.setdefault(k, [])

    def faults(self):
        for ev in sorted(self.schedule, key=lambda e: e.at_s):
            target = self.env.now
            wait = (ev.at_s + self.load_start) - target
            if wait > 0:
                yield self.env.timeout(wait)
            node = ev.target
            vn = self.nodes[node]
            if ev.action in churn_mod.DOWN_ACTIONS:
                vn.alive = False
                self.pending_fault[node] = self.env.now
            else:
                vn.alive = True
                self.mem.heartbeat(node, self.env.now)
                self.epoch += 1
                self.refresh_live()
            self.log.emit("fault_applied", node_id=node, tier=vn.tier,
                          stage=vn.stage, action=ev.action,
                          detail=json.dumps({"scheduled_at_s": ev.at_s}))

    def sample(self):
        interval = self.scn.coordinator.sample_interval_ms / 1000.0
        last = {n: 0.0 for n in self.nodes}
        last_t = self.env.now
        while not self.stopped:
            yield self.env.timeout(interval)
            window = max(1e-9, self.env.now - last_t)
            for node, vn in self.nodes.items():
                self.log.emit(
                    "node_sample", node_id=node, tier=vn.tier, stage=vn.stage,
                    inflight=vn.busy, queue_depth=vn.queue_depth,
                    busy_frac=round((vn.busy_s - last[node])
                                    / (window * vn.servers), 6),
                    tokens_done=vn.tokens_done,
                    hb_gap_s=round(self.env.now
                                   - self.mem.nodes[node].last_seen, 6))
                last[node] = vn.busy_s
            last_t = self.env.now

    def throughput(self):
        last, last_t = 0, self.env.now
        while not self.stopped:
            yield self.env.timeout(1.0)
            dt = max(1e-9, self.env.now - last_t)
            delta = self.tokens_total - last
            last, last_t = self.tokens_total, self.env.now
            self.log.emit("throughput_sample", window_s=dt, tokens=delta,
                          tokens_per_s=delta / dt,
                          requests_inflight=self.inflight,
                          live_nodes=sum(len(v) for v in self.live_by_stage.values()))

    # -- run ---------------------------------------------------------------
    def run(self) -> dict:
        preds = self.plan.predictions()
        self.log.emit("run_start", scenario=self.scn.name, depth=self.plan.depth,
                      width=self.scn.composition.width,
                      n_nodes=len(self.plan.nodes),
                      concurrency=self.scn.workload.concurrency,
                      seed=self.scn.seed)
        self.log.emit("note", message="predictions",
                      detail=json.dumps(preds, default=float))

        for node in self.nodes:
            self.env.process(self.heartbeat(node))
        self.env.process(self.detect())
        self.env.process(self.sample())
        self.env.process(self.throughput())
        self.env.process(self.faults())

        if self.scn.workload.arrival == "closed_loop":
            for i in range(self.scn.workload.concurrency):
                self.env.process(self.stream(i))
        else:
            self.env.process(self.poisson_source())

        self.env.run(until=self.load_start + self.scn.duration_s)
        self.stopped = True
        self.log.emit("run_end", reason="duration elapsed",
                      requests_started=self.started, requests_ok=self.ok,
                      requests_failed=self.failed)
        self.log.close()
        return {"sim_events": self.env.n_events, "sim_end_s": self.env.now,
                "requests_started": self.started, "requests_ok": self.ok,
                "requests_failed": self.failed, "tokens": self.tokens_total}
