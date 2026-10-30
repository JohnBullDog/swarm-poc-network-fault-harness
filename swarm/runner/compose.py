"""Generate a docker-compose file from a scenario.

One service per node, named by its node id, so Docker's DNS resolves a node id
directly and the gateway's routing table needs no address translation.

Tier emulation is applied here: the cgroup CPU and memory limits from the design
doc's table, plus the tc parameters the container shapes its own link with.
These preserve the tiers' ratios on one host; they do not reproduce absolute
device speeds, which is a stated limit on interpreting the results.
"""

from __future__ import annotations

from pathlib import Path

from ..core.placement import Plan
from ..core.scenario import Scenario

NETWORK = "swarmnet"
LOG_DIR = "/logs"


def _burst_kbit(rate_mbit: float) -> int:
    """Token-bucket burst for a rate cap.

    Too small a burst throttles well below the nominal rate because a single
    packet cannot be admitted in one tick; too large lets the cap be exceeded in
    bursts. rate/4 in kbit, floored at 32, keeps both errors small across the
    10-1000 Mbit range the tiers span.
    """
    return max(32, int(rate_mbit / 4))


def _env_common(scn: Scenario, run_id: str, run_epoch: float) -> dict[str, str]:
    return {
        "RUN_ID": run_id,
        "RUN_EPOCH": f"{run_epoch:.6f}",
        "LOG_DIR": LOG_DIR,
        "SCENARIO_JSON": scn.to_json(),
        "PYTHONPATH": "/app",
    }


def build(scn: Scenario, plan: Plan, run_id: str, run_epoch: float,
          log_host_dir: Path, stack: str,
          image_synthetic: str = "swarm-poc:synthetic",
          image_real: str = "swarm-poc:real",
          model_host_dir: Path | None = None) -> dict:
    """The compose document as a plain dict, ready to be dumped as YAML."""
    worker_image = image_real if scn.mode == "real" else image_synthetic
    gateway_image = worker_image          # the gateway runs embedding and the head
    common = _env_common(scn, run_id, run_epoch)

    # Absolute paths only. Docker reads a relative source as the *name* of a
    # named volume, so "models:/models:ro" silently becomes a reference to a
    # volume called "models" and the stack fails to start with "refers to
    # undefined volume".
    log_host_dir = Path(log_host_dir).resolve()
    if model_host_dir is not None:
        model_host_dir = Path(model_host_dir).resolve()

    # Bind-mount the run's raw log directory straight onto the host, so events
    # land in their final home and no export step can lose them. Event volume is
    # a few hundred KB/s at most, well inside what a bind mount sustains.
    log_mount = f"{log_host_dir.as_posix()}:{LOG_DIR}"

    services: dict[str, dict] = {}

    services["coordinator"] = {
        "image": image_synthetic,          # never needs the model
        "container_name": f"{stack}-coordinator",
        "hostname": "coordinator",
        "environment": {**common, "COMPONENT_MODULE": "swarm.components.coordinator"},
        "volumes": [log_mount],
        "networks": [NETWORK],
        "ports": ["8100:8100"],
        "init": True,
        "stop_grace_period": "2s",
    }

    # The gateway's egress is shaped too. Its hop into stage 0 is a real hop and
    # would otherwise be the only unshaped link in the path, quietly making the
    # measured per-token latency one hop cheaper than the model predicts.
    gl = scn.gateway_link
    gw_env = {
        **common,
        "COMPONENT_MODULE": "swarm.components.gateway",
        "TC_ENABLE": "1",
        "TC_DEV": "eth0",
        "TC_RATE_MBIT": f"{gl.rate_mbit:g}",
        "TC_DELAY_MS": f"{gl.delay_ms:g}",
        "TC_JITTER_MS": f"{gl.jitter_ms:g}",
        "TC_LOSS_PCT": f"{gl.loss_pct:g}",
        "TC_BURST_KBIT": str(_burst_kbit(gl.rate_mbit)),
    }
    services["gateway"] = {
        "image": gateway_image,
        "container_name": f"{stack}-gateway",
        "hostname": "gateway",
        "environment": gw_env,
        "volumes": [log_mount],
        "networks": [NETWORK],
        "ports": ["8200:8200"],
        "cap_add": ["NET_ADMIN"],
        "depends_on": ["coordinator"],
        "init": True,
        "stop_grace_period": "5s",
    }
    if scn.mode == "real" and model_host_dir is not None:
        services["gateway"]["volumes"].append(
            f"{model_host_dir.as_posix()}:/models:ro")

    for node in plan.nodes:
        tier = plan.node_tiers[node]
        spec = scn.tier_spec(tier)
        link = scn.link_for(tier)
        env = {
            **common,
            "COMPONENT_MODULE": "swarm.components.stage_worker",
            "NODE_ID": node,
            "TC_ENABLE": "1",
            "TC_DEV": "eth0",
            "TC_RATE_MBIT": f"{link.rate_mbit:g}",
            "TC_DELAY_MS": f"{link.delay_ms:g}",
            "TC_JITTER_MS": f"{link.jitter_ms:g}",
            "TC_LOSS_PCT": f"{link.loss_pct:g}",
            "TC_BURST_KBIT": str(_burst_kbit(link.rate_mbit)),
        }
        if scn.mode == "real":
            env["TORCH_NUM_THREADS"] = str(scn.real.torch_threads)
        svc = {
            "image": worker_image,
            "container_name": f"{stack}-{node}",
            "hostname": node,
            "environment": env,
            "volumes": [log_mount],
            "networks": [NETWORK],
            "cap_add": ["NET_ADMIN"],        # required to shape its own link
            "cpus": float(spec.cpu_cores),
            "mem_limit": f"{spec.mem_limit_mb}m",
            "depends_on": ["coordinator"],
            "init": True,
            "stop_grace_period": "2s",
        }
        if scn.mode == "real" and model_host_dir is not None:
            svc["volumes"].append(f"{model_host_dir.as_posix()}:/models:ro")
        services[node] = svc

    return {
        "name": stack,
        "services": services,
        "networks": {NETWORK: {"name": f"{stack}-net", "driver": "bridge"}},
    }


def write(doc: dict, path: Path) -> Path:
    import yaml
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        yaml.safe_dump(doc, fh, sort_keys=False, width=10_000)
    return path
