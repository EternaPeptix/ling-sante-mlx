"""Convert LingSante built-in MTP layer (model.layers.72.*) on CPU and verify against raw shards.

Produces a clean standalone MLX safetensors file (`model-mtp.safetensors`) with relative keys
for `BailingMTPLayer`:
  - `enorm.weight`, `hnorm.weight`, `eh_proj.weight`, `final_layernorm.weight`
  - `input_layernorm.weight`, `post_attention_layernorm.weight`
  - `attention.{q_a_proj,q_a_layernorm,q_b_proj,kv_a_proj_with_mqa,kv_a_layernorm,embed_q,unembed_out,g_proj,dense}.weight`
  - `mlp.{gate.weight,gate.expert_bias,shared_experts.*,switch_mlp.{gate,up,down}_proj.{weight,scales}}`

Usage:
  python tools/convert_mtp.py --src SRC_DIR --dst OUT_FILE [--cpu] [--verify]
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE.parent / "vendor")]

from convert import Source, dequant_fp8, pack_experts  # noqa: E402


def convert_mtp_layer(src: Source, layer_idx: int, cfg: dict) -> dict[str, mx.array]:
    pre = f"model.layers.{layer_idx}"
    out: dict[str, mx.array] = {}
    names = [k for k in src.index if k.startswith(pre + ".") and ".experts." not in k]
    if not names:
        raise RuntimeError(f"No tensors found for {pre}.* in {src.src}")

    num_heads = cfg["num_attention_heads"]
    qk_nope = cfg["qk_nope_head_dim"]
    v_dim = cfg["v_head_dim"]

    for k in sorted(names):
        if k.endswith(".weight_scale_inv"):
            continue
        if not src.available(k):
            raise RuntimeError(f"Tensor {k} incomplete in source shards")
        dt = src.index[k][1]
        t = src.get(k)
        if dt == "F8_E4M3":
            s = src.get(k + "_scale_inv")
            t = dequant_fp8(t, s).astype(mx.bfloat16)

        rel = k[len(pre) + 1 :]
        if rel == "attention.kv_b_proj.weight":
            v = t.reshape(num_heads, qk_nope + v_dim, -1)
            out["attention.embed_q.weight"] = mx.contiguous(v[:, :qk_nope, :].swapaxes(-1, -2))
            out["attention.unembed_out.weight"] = mx.contiguous(v[:, qk_nope:, :])
            continue
        out[rel] = t

    for proj in ("gate_proj", "up_proj", "down_proj"):
        for e in range(cfg["num_experts"]):
            for suf in ("weight", "weight_scale_inv"):
                if not src.available(f"{pre}.mlp.experts.{e}.{proj}.{suf}"):
                    raise RuntimeError(f"Missing expert {e} {proj}.{suf} in {pre}")
        w, s = pack_experts(src, f"{pre}.mlp", proj, cfg["num_experts"])
        out[f"mlp.switch_mlp.{proj}.weight"] = w
        out[f"mlp.switch_mlp.{proj}.scales"] = s

    return out


def verify_mtp(src_dir: Path, out_file: Path, layer_idx: int, n_samples: int = 64) -> int:
    import torch
    from ref_layer import Src

    src = Src(str(src_dir))
    w = mx.load(str(out_file))
    pre = f"model.layers.{layer_idx}"
    random.seed(0)
    bad = 0
    cnt = {"expert": 0, "fp8": 0, "copy": 0}

    for _ in range(n_samples):
        e = random.randrange(src.cfg["num_experts"])
        proj = random.choice(["gate_proj", "up_proj", "down_proj"])
        W = w[f"mlp.switch_mlp.{proj}.weight"][e]
        S = w[f"mlp.switch_mlp.{proj}.scales"][e]
        got = np.array(mx.dequantize(W, S, group_size=32, bits=4, mode="mxfp4", dtype=mx.float32))
        ref = src.fp4(f"{pre}.mlp.experts.{e}.{proj}.weight").numpy()
        same = np.array_equal(got.view(np.uint32), ref.view(np.uint32)) or np.array_equal(got, ref)
        cnt["expert"] += 1
        if not same:
            bad += 1
            print(f"EXPERT MISMATCH e={e} {proj} max={np.abs(got - ref).max()}")

    # Reconstruct kv_b_proj from split embed_q & unembed_out
    nh = src.cfg["num_attention_heads"]
    qk_nope = src.cfg["qk_nope_head_dim"]
    v_dim = src.cfg["v_head_dim"]
    wk = w["attention.embed_q.weight"].swapaxes(-1, -2)  # (H, qk_nope, kv_lora)
    wv = w["attention.unembed_out.weight"]              # (H, v_dim, kv_lora)
    kv_rec = mx.concatenate([wk, wv], axis=1).reshape(nh * (qk_nope + v_dim), -1)
    ref_kv = src.t(f"{pre}.attention.kv_b_proj.weight").to(torch.bfloat16)
    got_kv = torch.from_numpy(np.array(kv_rec.astype(mx.float32))).to(torch.bfloat16)
    cnt["fp8"] += 1
    if not torch.equal(ref_kv, got_kv):
        bad += 1
        print("MISMATCH kv_b_proj split/reconstruct")

    for rel, v in w.items():
        if rel.startswith(("mlp.switch_mlp.", "attention.embed_q.", "attention.unembed_out.")):
            continue
        k = f"{pre}.{rel}"
        dt = src.idx[k][1]
        ref = src.t(k)
        if dt == "F8_E4M3":
            r = ref.to(torch.bfloat16)
            g = torch.from_numpy(np.array(v.astype(mx.float32))).to(torch.bfloat16)
            same = torch.equal(r, g)
            cnt["fp8"] += 1
        else:
            same = torch.equal(ref, torch.from_numpy(np.array(v.astype(mx.float32))).reshape(ref.shape))
            cnt["copy"] += 1
        if not same:
            bad += 1
            print("MISMATCH", rel)

    print("verify checked", cnt, "mismatches", bad, flush=True)
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--layer", type=int, default=72)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--samples", type=int, default=64)
    a = ap.parse_args()

    if a.cpu:
        mx.set_default_device(mx.cpu)

    src_dir = Path(os.path.expanduser(a.src))
    dst_file = Path(os.path.expanduser(a.dst))
    dst_file.parent.mkdir(parents=True, exist_ok=True)

    if not a.verify_only:
        cfg = json.loads((src_dir / "config.json").read_text())
        src = Source(src_dir)
        t0 = time.perf_counter()
        out = convert_mtp_layer(src, a.layer, cfg)
        mx.eval(out)
        tmp = dst_file.with_name(f".tmp-{dst_file.name}")
        mx.save_safetensors(str(tmp), out, metadata={"format": "mlx"})
        tmp.rename(dst_file)
        nb = sum(v.nbytes for v in out.values())
        print(f"converted layer {a.layer} -> {dst_file}: {len(out)} tensors, {nb/1e9:.3f} GB in {time.perf_counter()-t0:.1f}s", flush=True)
        src.close()
        del out
        mx.clear_cache()

    if a.verify or a.verify_only:
        bad = verify_mtp(src_dir, dst_file, a.layer, a.samples)
        if bad:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
