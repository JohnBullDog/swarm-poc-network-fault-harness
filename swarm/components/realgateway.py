"""Real-mode gateway: tokenisation, embedding, the final norm and the output head.

Only imported in the real image.

Per the design doc the gateway owns everything outside the stage blocks, which
here is the tied embedding/output matrix and the final norm. That also means the
544 MB embedding is loaded once, on the gateway, instead of on every worker.

Decoding is greedy. Sampling would make two runs of the same scenario produce
different token sequences, and the mode check needs the real arm to be
reproducible and its output checkable after a failover.

Re-prefill on failover. KV cache does not transfer between nodes, so when a
token's path differs from the path that last served this request, the whole
context is resent with `reprefill` set and every stage recomputes it. The
gateway decides this per attempt, after the path is chosen, because that is when
the change is known.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer

# A single fixed prompt, so every request in every run decodes the same
# sequence. Repeatability is a stated requirement, and a fixed prompt also makes
# "is the output still correct after a failover?" a question with one answer.
PROMPT_TEXT = (
    "The following is a short technical note about distributed computing. "
    "Pooling many small unreliable machines can store a model that no single "
    "machine could hold, but the result is a throughput engine and never a "
    "low latency one, because each generated token must pass through every "
    "layer in sequence and that chain cannot be made shorter by adding more "
    "machines to the pool. In practice this means that"
)


@dataclass
class ReqState:
    request_id: str
    token_ids: list[int]
    prompt_len: int
    last_path: list[str] | None = None
    generated: list[int] = field(default_factory=list)


class RealGateway:
    def __init__(self, ctx):
        self.ctx = ctx
        self.scn = ctx.scenario
        self.model_dir = Path(self.scn.real.model_dir)
        self.dtype = getattr(torch, self.scn.real.dtype)
        self.tokenizer = None
        self.embed: nn.Embedding | None = None
        self.norm = None
        self.hidden = 0
        self.prompt_ids: list[int] = []

    # -- loading ----------------------------------------------------------
    def load(self) -> None:
        torch.set_num_threads(max(1, self.scn.real.torch_threads))
        torch.set_grad_enabled(False)

        cfg = AutoConfig.from_pretrained(str(self.model_dir))
        self.hidden = cfg.hidden_size
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))

        want = {"model.embed_tokens.weight", "model.norm.weight"}
        if not getattr(cfg, "tie_word_embeddings", True):
            want.add("lm_head.weight")
        found: dict[str, torch.Tensor] = {}
        for path in self._weight_files():
            with safe_open(str(path), framework="pt") as fh:
                for key in fh.keys():
                    if key in want:
                        found[key] = fh.get_tensor(key).to(self.dtype)
        missing = want - set(found)
        if missing:
            raise RuntimeError(f"gateway could not load {sorted(missing)}")

        emb = found["model.embed_tokens.weight"]
        self.embed = nn.Embedding.from_pretrained(emb, freeze=True)
        # Tied weights: the output head is the embedding matrix transposed.
        self.head_weight = found.get("lm_head.weight", emb)

        eps = getattr(cfg, "rms_norm_eps", 1e-6)
        norm_w = found["model.norm.weight"]

        def rms_norm(x: torch.Tensor) -> torch.Tensor:
            var = x.pow(2).mean(-1, keepdim=True)
            return norm_w * (x * torch.rsqrt(var + eps))

        self.norm = rms_norm

        ids = self.tokenizer(PROMPT_TEXT, add_special_tokens=False)["input_ids"]
        need = self.scn.workload.prompt_tokens
        if len(ids) < need:
            ids = (ids * (need // len(ids) + 1))
        self.prompt_ids = ids[:need]

        self.ctx.log.emit("note", message="real gateway loaded", detail=json.dumps({
            "hidden": self.hidden, "vocab": int(emb.shape[0]),
            "tied": bool(getattr(cfg, "tie_word_embeddings", True)),
            "prompt_tokens": len(self.prompt_ids),
            "embed_bytes": int(emb.numel() * emb.element_size())}))

    def _weight_files(self) -> list[Path]:
        index = self.model_dir / "model.safetensors.index.json"
        if index.exists():
            meta = json.loads(index.read_text(encoding="utf-8"))
            return [self.model_dir / n for n in sorted(set(meta["weight_map"].values()))]
        single = self.model_dir / "model.safetensors"
        if single.exists():
            return [single]
        raise FileNotFoundError(f"no safetensors found in {self.model_dir}")

    # -- per request ------------------------------------------------------
    def begin(self, request_id: str, prompt_tokens: int) -> ReqState:
        return ReqState(request_id=request_id, token_ids=list(self.prompt_ids),
                        prompt_len=len(self.prompt_ids))

    def payload_for_path(self, state: ReqState, token_idx: int,
                         path: list[str]) -> tuple[bytes, list[int], bool]:
        """Payload for this attempt, plus whether the stages must re-prefill.

        The first token is a prefill by definition. Afterwards a full resend is
        needed only when the path changed, because the new node holds no KV cache
        for this request.
        """
        reprefill = (token_idx == 0) or (state.last_path is not None
                                         and path != state.last_path)
        payload, shape = self._embed_for(state, full=reprefill)
        return payload, shape, reprefill

    def _embed_for(self, state: ReqState, full: bool) -> tuple[bytes, list[int]]:
        ids = state.token_ids if full else state.token_ids[-1:]
        with torch.no_grad():
            h = self.embed(torch.tensor([ids], dtype=torch.long))
        arr = h.to(torch.float32).numpy()
        return arr.tobytes(), [arr.shape[1], arr.shape[2]]

    def accept(self, state: ReqState, out: bytes, meta: dict) -> None:
        """Apply the final norm and head to the last position, pick the token."""
        shape = meta.get("shape") or [1, self.hidden]
        seq, hid = int(shape[0]), int(shape[1])
        h = torch.from_numpy(
            np.frombuffer(out, dtype=np.float32).reshape(seq, hid).copy()
        ).to(self.dtype)
        with torch.no_grad():
            last = self.norm(h[-1:])
            logits = last @ self.head_weight.T
            nxt = int(torch.argmax(logits, dim=-1)[0])
        state.token_ids.append(nxt)
        state.generated.append(nxt)
        state.last_path = list(meta.get("path") or [])

    def finish(self, state: ReqState) -> str | None:
        """The generated continuation, so output correctness is inspectable."""
        if not state.generated:
            return None
        return self.tokenizer.decode(state.generated, skip_special_tokens=True)
