"""Check the converted directory against the source with the independent decoder in ref_layer.py.

 * experts: mx.dequantize(mode=mxfp4) of the stored uint32/uint8 vs a direct numpy dequantisation of the source I8/E8M0
   -> must be bit-identical (fp32).
 * FP8 linears: stored bf16 vs bf16(source fp8 * scale) -> bit-identical.
 * pass-through tensors (bf16/f32): bit-identical.

usage: verify_convert.py SRC DST [n_samples] [--cpu]
"""
import argparse
import glob
import os
import random
import sys

import mlx.core as mx
import numpy as np
import torch
from mlx.utils import tree_flatten  # noqa: F401

_here = __file__.rsplit("/", 1)[0]
sys.path[:0] = [_here, _here + "/../vendor"]
from ref_layer import Src  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("dst")
ap.add_argument("n_samples", nargs="?", type=int, default=24)
ap.add_argument("--cpu", action="store_true")
args = ap.parse_args()
if args.cpu:
    mx.set_default_device(mx.cpu)

src = Src(args.src)
dst = args.dst
n = args.n_samples
random.seed(0)

files = sorted(glob.glob(f"{dst}/model-layer-*.safetensors"))
misc_file = f"{dst}/model-misc.safetensors"
if os.path.exists(misc_file):
    files = files + [misc_file]
print("converted files:", len(files))
bad = 0
cnt = {"expert": 0, "fp8": 0, "copy": 0}
for f in files:
    is_misc = f.endswith("model-misc.safetensors")
    w = mx.load(f)
    if not is_misc:
        li = int(f.rsplit("-", 1)[1].split(".")[0])
        pre = f"model.layers.{li}"
        if f"{pre}.mlp.switch_mlp.gate_proj.weight" in w:
            for _ in range(n):
                e = random.randrange(src.cfg["num_experts"])
                proj = random.choice(["gate_proj", "up_proj", "down_proj"])
                W = w[f"{pre}.mlp.switch_mlp.{proj}.weight"][e]
                S = w[f"{pre}.mlp.switch_mlp.{proj}.scales"][e]
                got = np.array(mx.dequantize(W, S, group_size=32, bits=4, mode="mxfp4", dtype=mx.float32))
                ref = src.fp4(f"{pre}.mlp.experts.{e}.{proj}.weight").numpy()
                same = np.array_equal(got.view(np.uint32), ref.view(np.uint32)) or np.array_equal(got, ref)
                cnt["expert"] += 1
                if not same:
                    bad += 1
                    print("EXPERT MISMATCH", li, e, proj, float(np.abs(got - ref).max()))
    for k, v in w.items():
        if ".switch_mlp." in k or k.endswith((".scales", ".biases")):
            continue
        if k not in src.idx:
            if k.endswith("_conv1d.weight"):
                continue
            print("no source for", k)
            continue
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
            print("MISMATCH", k)
print("checked", cnt, "mismatches", bad)
sys.exit(1 if bad else 0)
