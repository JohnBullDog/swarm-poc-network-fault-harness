"""Download the real-mode model once, to a directory the containers mount read-only.

Downloading inside each container would fetch the same gigabyte six or more
times per run and would make runs depend on network availability. The model is
fetched once here and bind-mounted read-only, so a run is reproducible offline
and every worker reads the same bytes.

Only the files the harness actually opens are fetched: the safetensors weights,
the config, and the tokenizer. No PyTorch .bin duplicates, no ONNX exports.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_REPO_ID = "Qwen/Qwen2.5-0.5B-Instruct"
PATTERNS = [
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
]


def fetch(repo_id: str, dest: Path) -> Path:
    from huggingface_hub import snapshot_download

    dest.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=repo_id,
        local_dir=str(dest),
        allow_patterns=PATTERNS,
    )
    return Path(path)


def describe(model_dir: Path) -> dict:
    """Report what was fetched, including per-layer sizes.

    The per-layer byte count is what the cost model's P_a*b should match for
    this model, so it is printed rather than assumed.
    """
    from safetensors import safe_open

    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    files = sorted(model_dir.glob("*.safetensors"))
    layer_bytes: dict[int, int] = {}
    other_bytes = 0
    total_params = 0

    for path in files:
        with safe_open(str(path), framework="np") as fh:  # numpy: the host has no torch
            for key in fh.keys():
                slice_ = fh.get_slice(key)
                shape = slice_.get_shape()
                n = 1
                for d in shape:
                    n *= d
                total_params += n
                nbytes = n * 4          # as loaded: float32
                if key.startswith("model.layers."):
                    gi = int(key.split(".")[2])
                    layer_bytes[gi] = layer_bytes.get(gi, 0) + nbytes
                else:
                    other_bytes += nbytes

    n_layers = cfg["num_hidden_layers"]
    per_layer = sum(layer_bytes.values()) / max(1, len(layer_bytes))
    return {
        "repo_files": [p.name for p in files],
        "hidden_size": cfg["hidden_size"],
        "num_hidden_layers": n_layers,
        "vocab_size": cfg["vocab_size"],
        "tie_word_embeddings": cfg.get("tie_word_embeddings", True),
        "total_params": total_params,
        "layer_bytes_fp32_total": sum(layer_bytes.values()),
        "layer_bytes_fp32_per_layer": per_layer,
        "non_layer_bytes_fp32": other_bytes,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    ap.add_argument("--dest", type=Path, default=REPO / "models" / "qwen2.5-0.5b")
    args = ap.parse_args(argv)

    print(f"fetching {args.repo_id} -> {args.dest}")
    fetch(args.repo_id, args.dest)
    info = describe(args.dest)

    print("\nfetched:")
    for key, val in info.items():
        if isinstance(val, float):
            print(f"  {key:32s} {val:,.0f}")
        elif isinstance(val, int):
            print(f"  {key:32s} {val:,}")
        else:
            print(f"  {key:32s} {val}")

    gb = info["layer_bytes_fp32_total"] / 1e9
    print(f"\n  stage-held weight bytes (fp32): {gb:.3f} GB")
    print("  -> this is what the qwen_0_5b profile's P_a*b must equal; "
          "costmodel.QWEN_0_5B declares "
          f"{358e6 * 4 / 1e9:.3f} GB")

    (args.dest.parent / "model_info.json").write_text(
        json.dumps(info, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
