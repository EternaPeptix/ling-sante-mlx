"""Native F8_E4M3 block-scaled linear modules and loader attachment for BailingMoeV3."""
import os
from pathlib import Path

import mlx.core as mx
from mlx import nn

from mlx_lm.layers.attention import MultiLinear

from .rows.gemv_fp8 import dequant_fp8_multilinear_to_bf16, dequant_fp8_to_bf16


class FP8BlockLinear(nn.Module):
    """Exact block-scaled 128x128 F8_E4M3 linear storing uint8 weights + float32 scale_inv."""

    def __init__(self, out_dims: int, in_dims: int):
        super().__init__()
        self.weight_fp8 = mx.zeros((out_dims, in_dims), dtype=mx.uint8)
        self.weight_scale_inv = mx.zeros(
            ((out_dims + 127) // 128, (in_dims + 127) // 128), dtype=mx.float32
        )

    @property
    def weight(self) -> mx.array:
        return dequant_fp8_to_bf16(self.weight_fp8, self.weight_scale_inv)

    def __call__(self, x: mx.array) -> mx.array:
        w = dequant_fp8_to_bf16(self.weight_fp8, self.weight_scale_inv)
        if x.dtype != w.dtype:
            w = w.astype(x.dtype)
        return x @ w.T


class FP8MultiLinear(nn.Module):
    """Exact block-scaled F8_E4M3 MultiLinear for MLA embed_q / unembed_out."""

    def __init__(self, input_dims: int, output_dims: int, num_heads: int, scale_by_n: bool):
        super().__init__()
        self.weight_fp8 = mx.zeros((num_heads, output_dims, input_dims), dtype=mx.uint8)
        self.weight_scale_inv = mx.zeros((num_heads, 4), dtype=mx.float32)
        self.scale_by_n = bool(scale_by_n)

    @property
    def weight(self) -> mx.array:
        return dequant_fp8_multilinear_to_bf16(
            self.weight_fp8, self.weight_scale_inv, self.scale_by_n
        )

    def __call__(self, x: mx.array, transpose: bool = True) -> mx.array:
        w = dequant_fp8_multilinear_to_bf16(
            self.weight_fp8, self.weight_scale_inv, self.scale_by_n
        )
        if x.dtype != w.dtype:
            w = w.astype(x.dtype)
        if transpose:
            return x @ w.swapaxes(-1, -2)
        return x @ w


def _set_module_at(root, path: str, new_mod: nn.Module):
    parts = path.split(".")
    cur = root
    for p in parts[:-1]:
        cur = cur[int(p)] if isinstance(cur, list) else getattr(cur, p)
    setattr(cur, parts[-1], new_mod)


def _get_module_at(root, path: str):
    cur = root
    for p in path.split("."):
        cur = cur[int(p)] if isinstance(cur, list) else getattr(cur, p)
    return cur


def attach_lingsante_fp8(
    model,
    weights: dict,
    model_path: str | Path,
    *,
    layer_indices: set[int] | None = None,
) -> int:
    """Replace eligible linears with FP8BlockLinear/FP8MultiLinear from `<model_path>/fp8`."""
    if os.environ.get("LINGSANTE_FP8", "1") in ("0", ""):
        return 0
    if "bailing_moe_v3" not in type(model).__module__:
        return 0
    fp8_dir = Path(model_path) / "fp8"
    if not fp8_dir.is_dir():
        return 0

    if layer_indices is not None:
        fp8_files = [
            fp8_dir / f"model-fp8-layer-{li:03d}.safetensors"
            for li in sorted(layer_indices)
            if (fp8_dir / f"model-fp8-layer-{li:03d}.safetensors").exists()
        ]
    else:
        fp8_files = sorted(fp8_dir.glob("model-fp8-*.safetensors"))
    if not fp8_files:
        return 0

    fp8_weights = {}
    for wf in fp8_files:
        fp8_weights.update(mx.load(str(wf)))

    replaced = 0
    for k in sorted(list(fp8_weights.keys())):
        if not k.endswith(".weight_fp8"):
            continue
        mod_path = k[: -len(".weight_fp8")]
        sc_key = f"{mod_path}.weight_scale_inv"
        if sc_key not in fp8_weights:
            continue
        m = _get_module_at(model, mod_path)
        w_fp8 = fp8_weights[k]
        w_sc = fp8_weights[sc_key]
        if isinstance(m, nn.Linear):
            out_dims, in_dims = w_fp8.shape
            _set_module_at(model, mod_path, FP8BlockLinear(out_dims, in_dims))
        elif isinstance(m, MultiLinear):
            num_heads, out_dims, in_dims = w_fp8.shape
            _set_module_at(
                model,
                mod_path,
                FP8MultiLinear(in_dims, out_dims, num_heads, scale_by_n=mod_path.endswith(".embed_q")),
            )
        else:
            continue
        weights.pop(f"{mod_path}.weight", None)
        weights[k] = w_fp8
        weights[sc_key] = w_sc
        replaced += 1

    model._lingsante_fp8 = replaced > 0
    return replaced
