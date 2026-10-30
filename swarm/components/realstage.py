"""Real-mode stage: a contiguous block of transformer layers on CPU.

Only imported in the real image, so synthetic containers never pull torch.

How the block is built. Instantiating the full model on every worker would load
the 544 MB tied embedding matrix six times over for nothing. Instead each worker
builds a Qwen2Model whose config declares only the layers it owns and a
one-token vocabulary, then loads just its own layers' tensors out of the
safetensors file with `safe_open`, which reads the requested tensors rather than
the whole file. Declaring `num_hidden_layers` as the block size also makes the
layers' internal indices 0..n-1, so the KV cache indexes correctly without any
remapping.

The final norm and the output head stay on the gateway, as the design doc
specifies, so every stage uses an identity norm.

KV cache and failover. Cache is held per request. A request whose path changes
mid-flight arrives with `reprefill` set, because KV cache does not transfer
between nodes: the gateway resends the whole context and the stage recomputes
it. That recompute is the real cost of pipeline failover with a warm cache, and
it is measured rather than hidden.
"""

from __future__ import annotations

import copy
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from safetensors import safe_open
from transformers import AutoConfig, DynamicCache
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model

MAX_CACHED_REQUESTS = 64


class RealStage:
    def __init__(self, ctx):
        self.ctx = ctx
        self.scn = ctx.scenario
        self.sp = ctx.plan.stages[ctx.stage]
        self.model_dir = Path(self.scn.real.model_dir)
        self.dtype = getattr(torch, self.scn.real.dtype)
        self.model: Qwen2Model | None = None
        self.caches: OrderedDict[str, DynamicCache] = OrderedDict()
        self.hidden = 0

    # -- loading ----------------------------------------------------------
    def load(self) -> None:
        torch.set_num_threads(max(1, self.scn.real.torch_threads))
        torch.set_grad_enabled(False)

        cfg = AutoConfig.from_pretrained(str(self.model_dir))
        self.hidden = cfg.hidden_size
        start, end = self.sp.layer_start, self.sp.layer_end
        n_local = end - start

        local_cfg = copy.deepcopy(cfg)
        local_cfg.num_hidden_layers = n_local
        local_cfg.vocab_size = 1          # the real embedding lives on the gateway
        local_cfg._attn_implementation = "eager"

        model = Qwen2Model(local_cfg)
        model.norm = nn.Identity()        # the gateway applies norm and the head
        model.eval().to(self.dtype)

        sd, n_tensors, n_bytes = {}, 0, 0
        for path in self._weight_files():
            with safe_open(str(path), framework="pt") as fh:
                for key in fh.keys():
                    if not key.startswith("model.layers."):
                        continue
                    gi = int(key.split(".")[2])
                    if not (start <= gi < end):
                        continue
                    local = key.replace(f"model.layers.{gi}.",
                                        f"layers.{gi - start}.", 1)
                    tensor = fh.get_tensor(key).to(self.dtype)
                    sd[local] = tensor
                    n_tensors += 1
                    n_bytes += tensor.numel() * tensor.element_size()

        missing, unexpected = model.load_state_dict(sd, strict=False)
        # Only the stand-in embedding and the removed norm may be missing; a
        # missing projection weight would mean this stage is computing with
        # random values, which must fail loudly rather than produce plausible
        # nonsense.
        real_missing = [k for k in missing
                        if not k.startswith("embed_tokens")
                        and not k.startswith("norm")]
        if real_missing or unexpected:
            raise RuntimeError(
                f"stage {self.ctx.stage} weight load mismatch: "
                f"missing={real_missing[:5]} unexpected={list(unexpected)[:5]}")

        self.model = model
        self.ctx.log.emit("note", message="real stage loaded", detail=json.dumps({
            "layers": [start, end], "n_local": n_local, "tensors": n_tensors,
            "weight_bytes": n_bytes, "dtype": self.scn.real.dtype,
            "threads": self.scn.real.torch_threads,
            "predicted_weight_bytes": self.sp.weight_bytes}))

    def _weight_files(self) -> list[Path]:
        index = self.model_dir / "model.safetensors.index.json"
        if index.exists():
            meta = json.loads(index.read_text(encoding="utf-8"))
            names = sorted(set(meta["weight_map"].values()))
            return [self.model_dir / n for n in names]
        single = self.model_dir / "model.safetensors"
        if single.exists():
            return [single]
        raise FileNotFoundError(f"no safetensors found in {self.model_dir}")

    # -- cache ------------------------------------------------------------
    def _cache_for(self, request_id: str, reprefill: bool) -> DynamicCache:
        if reprefill:
            self.caches.pop(request_id, None)
        cache = self.caches.get(request_id)
        if cache is None:
            cache = DynamicCache()
            self.caches[request_id] = cache
            while len(self.caches) > MAX_CACHED_REQUESTS:
                self.caches.popitem(last=False)   # oldest request's cache
        else:
            self.caches.move_to_end(request_id)
        return cache

    def release(self, request_id: str) -> None:
        self.caches.pop(request_id, None)

    # -- the hop ----------------------------------------------------------
    def forward(self, payload: bytes, meta: dict) -> bytes:
        """Run this block over the incoming hidden state. Runs in an executor.

        `no_grad` is applied here, not once at load time: torch's grad mode is
        thread-local, and this runs on an executor thread rather than the one
        that called `load()`. Without it every output tensor requires grad, the
        autograd graph grows for the life of the request, and `.numpy()` refuses
        outright.
        """
        shape = meta.get("shape") or [1, self.hidden]
        seq = int(shape[0])
        h = torch.from_numpy(
            np.frombuffer(payload, dtype=np.float32).reshape(1, seq, shape[1]).copy()
        ).to(self.dtype)

        cache = self._cache_for(str(meta.get("request_id")),
                                bool(meta.get("reprefill")))
        with torch.no_grad():
            out = self.model(inputs_embeds=h, past_key_values=cache,
                             use_cache=True)
            hs = out.last_hidden_state.to(torch.float32).numpy()
        meta["shape"] = [hs.shape[1], hs.shape[2]]
        return hs.tobytes()
