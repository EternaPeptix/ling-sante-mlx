"""Single-layer / head loader + captured visit loader for rapid unit testing (<1s load!)."""
import json
import os
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(_root / "vendor"), str(_root)]

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
from mlx_lm.models.bailing_moe_v3 import Model, ModelArgs  # noqa: E402
from mlx_lm.utils import infer_quant_config, load_config  # noqa: E402

DEFAULT_MODEL = os.path.expanduser("~/science/models/lingsante-mlx-mxfp4-bf16")
DEFAULT_VISITS = os.path.expanduser("~/science/runs/visits")


def load_layer(layer_idx: int, model_dir: str = DEFAULT_MODEL):
    """Instantiate a minimal model shell and load ONLY weights for layer `layer_idx` (~4.7 GB, <1s)."""
    model_path = Path(model_dir)
    cfg = load_config(model_path)
    # Build only 1 dummy layer? No, ModelArgs needs num_hidden_layers to determine is_linear for layer_idx.
    # Creating Model(args) without eval/load is just tiny un-evaled arrays (<1ms), or we slice cleanly.
    args = ModelArgs.from_dict(cfg)
    model = Model(args)
    w = mx.load(str(model_path / f"model-layer-{layer_idx:03d}.safetensors"))
    w = model.sanitize(w)

    def pred(p, m):
        if not hasattr(m, "to_quantized") or f"{p}.scales" not in w:
            return False
        return infer_quant_config(p, m, w)

    nn.quantize(model, group_size=32, bits=4, mode="mxfp4", class_predicate=pred)
    model.load_weights(list(w.items()), strict=False)
    model.eval()
    layer = model.model.layers[layer_idx]
    mx.eval(layer.parameters())
    return layer, args


def load_head(model_dir: str = DEFAULT_MODEL):
    """Load ONLY model.norm and lm_head from model-misc.safetensors (~1.3 GB)."""
    model_path = Path(model_dir)
    cfg = load_config(model_path)
    args = ModelArgs.from_dict(cfg)
    w = mx.load(str(model_path / "model-misc.safetensors"))
    norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
    lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
    norm.load_weights([("weight", w["model.norm.weight"])])
    lm_head.load_weights([("weight", w["lm_head.weight"])])
    norm.eval()
    lm_head.eval()
    mx.eval(norm.parameters(), lm_head.parameters())
    return norm, lm_head, args


def load_visits(layer_idx, visits_dir: str = DEFAULT_VISITS, limit: int = 12):
    """Load captured single-stream visit snapshots for `layer_idx` (int or 'head')."""
    vdir = Path(visits_dir)
    idx = json.loads((vdir / "index.json").read_text())
    entries = [e for e in idx["entries"] if e["layer"] == layer_idx][:limit]
    return [mx.load(str(vdir / e["file"])) for e in entries]


def bit_diff_count(a: mx.array, b: mx.array) -> int:
    """Return number of mismatching elements by raw bit pattern."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return -1
    if a.dtype in (mx.float32, mx.int32, mx.uint32):
        va, vb = a.view(mx.uint32), b.view(mx.uint32)
    elif a.dtype in (mx.bfloat16, mx.float16, mx.int16, mx.uint16):
        va, vb = a.view(mx.uint16), b.view(mx.uint16)
    elif a.dtype == mx.bool_:
        va, vb = a.astype(mx.uint8), b.astype(mx.uint8)
    else:
        va, vb = a, b
    return int(mx.sum(va != vb).item())
