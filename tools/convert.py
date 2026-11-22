"""Streaming converter: LingSante-3.1-flash-MXFP4 (HF) -> MLX-loadable directory.

One *layer* at a time (bounded memory, ~5 GB peak): every tensor is read straight out of the
source shards through mmap, so no shard is ever loaded whole.

  routed experts  I8 [N, K/2] + F8_E8M0 [N, K/32]  -> mx uint32 [E, N, K/8] + uint8 [E, N, K/32]
                  (MLX mode="mxfp4", group_size 32): a pure byte reinterpretation + stack, lossless.
  FP8 linears     F8_E4M3 [N,K] * F32 scale_inv [ceil(N/128), ceil(K/128)] -> bf16   (--linears bf16)
                  or the same followed by MLX affine 8-bit group 64                  (--linears q8)
  everything else bf16 / f32 copied.
  MTP layer (index >= num_hidden_layers) is skipped (not used by the MLX model).

usage: convert.py --src DIR --dst DIR [--linears bf16|q8] [--layers a:b] [--no-misc]
"""
import argparse
import json
import re
import shutil
import struct
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

FP8_BLOCK = 128


class Source:
    """Tensor index over all source shards (name -> file, dtype, shape, byte range)."""

    def __init__(self, src: Path):
        self.src = src
        self.index = {}
        self.sizes = {}
        self._mm = {}
        for f in sorted(src.glob("model-*.safetensors")):
            size = f.stat().st_size
            self.sizes[f.name] = size
            with open(f, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                hdr = json.loads(fh.read(n))
            base = 8 + n
            for k, v in hdr.items():
                if k == "__metadata__":
                    continue
                a, b = v["data_offsets"]
                self.index[k] = (f.name, v["dtype"], tuple(v["shape"]), base + a, base + b)

    def available(self, name):
        if name not in self.index:
            return False
        f, _, _, _, end = self.index[name]
        return end <= self.sizes[f]

    def raw(self, name):
        f, dt, shape, a, b = self.index[name]
        if b > self.sizes[f]:
            raise RuntimeError(f"{name}: shard {f} incomplete")
        if f not in self._mm:
            self._mm[f] = np.memmap(self.src / f, dtype=np.uint8, mode="r")
        return dt, shape, self._mm[f][a:b]

    def close(self):
        self._mm.clear()

    def get(self, name):
        """-> mx.array in its native dtype (F8_E4M3/F8_E8M0/I8 come back as uint8/uint8/int8 bytes)."""
        dt, shape, buf = self.raw(name)
        a = np.array(buf)  # copy out of the mmap
        if dt == "BF16":
            return mx.array(a.view(np.uint16).reshape(shape)).view(mx.bfloat16)
        if dt == "F32":
            return mx.array(a.view(np.float32).reshape(shape))
        if dt in ("F8_E4M3", "F8_E8M0", "U8"):
            return mx.array(a.reshape(shape))
        if dt == "I8":
            return mx.array(a.view(np.int8).reshape(shape))
        raise ValueError(f"unsupported dtype {dt} for {name}")


def e4m3fn_table() -> mx.array:
    lut = np.zeros(256, np.float32)
    for i in range(256):
        s = -1.0 if i & 0x80 else 1.0
        e = (i >> 3) & 0xF
        m = i & 7
        if e == 0xF and m == 7:
            v = np.nan
        elif e == 0:
            v = m / 8.0 * 2.0**-6
        else:
            v = (1 + m / 8.0) * 2.0 ** (e - 7)
        lut[i] = s * v
    return mx.array(lut)


LUT = None


def dequant_fp8(w_u8: mx.array, scale: mx.array) -> mx.array:
    """FP8 e4m3 [N,K] with F32 128x128 block scale_inv -> float32 [N,K] (w * scale)."""
    global LUT
    if LUT is None:
        LUT = e4m3fn_table()
    n, k = w_u8.shape
    full = LUT[w_u8]
    s = mx.repeat(mx.repeat(scale, FP8_BLOCK, axis=0), FP8_BLOCK, axis=1)[:n, :k]
    return full * s


def pack_experts(src: Source, prefix: str, proj: str, num_experts: int):
    ws, ss = [], []
    for e in range(num_experts):
        w = src.get(f"{prefix}.experts.{e}.{proj}.weight")  # int8 [N, K/2]
        s = src.get(f"{prefix}.experts.{e}.{proj}.weight_scale_inv")  # uint8 (E8M0) [N, K/32]
        n, kb = w.shape
        ws.append(w.view(mx.uint32).reshape(n, kb // 4))
        ss.append(s)
    return mx.stack(ws), mx.stack(ss)


def convert_layer(src: Source, li: int, cfg: dict, linears: str):
    pre = f"model.layers.{li}"
    out = {}
    names = [k for k in src.index if k.startswith(pre + ".") and ".experts." not in k]
    for k in sorted(names):
        if k.endswith(".weight_scale_inv"):
            continue
        dt = src.index[k][1]
        t = src.get(k)
        if dt == "F8_E4M3":
            s = src.get(k + "_scale_inv")
            t = dequant_fp8(t, s)
            if linears == "q8" and k.endswith(".weight") and "kv_b_proj" not in k:
                wq, sc, bi = mx.quantize(t.astype(mx.bfloat16), group_size=64, bits=8, mode="affine")
                base = k[: -len(".weight")]
                out[base + ".weight"], out[base + ".scales"], out[base + ".biases"] = wq, sc, bi
                continue
            t = t.astype(mx.bfloat16)
        out[k] = t
    if li >= cfg["first_k_dense_replace"]:
        for proj in ("gate_proj", "up_proj", "down_proj"):
            w, s = pack_experts(src, f"{pre}.mlp", proj, cfg["num_experts"])
            out[f"{pre}.mlp.switch_mlp.{proj}.weight"] = w
            out[f"{pre}.mlp.switch_mlp.{proj}.scales"] = s
    return out


def layer_ready(src: Source, li: int, cfg: dict) -> bool:
    pre = f"model.layers.{li}."
    names = [k for k in src.index if k.startswith(pre)]
    if not names:
        return False
    if not all(src.available(k) for k in names):
        return False
    if li >= cfg["first_k_dense_replace"]:
        for proj in ("gate_proj", "up_proj", "down_proj"):
            for e in range(cfg["num_experts"]):
                if f"{pre}mlp.experts.{e}.{proj}.weight" not in src.index:
                    return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--linears", default="bf16", choices=["bf16", "q8"])
    ap.add_argument("--layers", default=None, help="a:b half-open range of layers (default all)")
    ap.add_argument("--no-misc", action="store_true")
    a = ap.parse_args()
    src_dir, dst = Path(a.src), Path(a.dst)
    cfg = json.loads((src_dir / "config.json").read_text())
    dst.mkdir(parents=True, exist_ok=True)
    src = Source(src_dir)
    print("indexed", len(src.index), "tensors in", len(src.sizes), "shards", flush=True)
    nl = cfg["num_hidden_layers"]
    lo, hi = (0, nl) if a.layers is None else map(int, a.layers.split(":"))

    if not a.no_misc:
        misc = {}
        for k in src.index:
            if not k.startswith("model.layers."):
                if not src.available(k):
                    print("MISC not ready:", k)
                    continue
                misc[k] = src.get(k)
        if misc:
            mx.save_safetensors(str(dst / "model-misc.safetensors"), misc, metadata={"format": "mlx"})
            print("misc:", {k: (v.shape, str(v.dtype)) for k, v in misc.items()})
        for f in src_dir.iterdir():
            if f.name in ("config.json",):
                continue
            if f.suffix in (".json", ".jinja", ".txt") and not f.name.startswith(("worker-", "conversion_", "model.safetensors.index")):
                shutil.copy(f, dst / f.name)
        newcfg = {k: v for k, v in cfg.items() if k not in ("quantization_config", "auto_map")}
        newcfg["quantization"] = {"group_size": 32, "bits": 4, "mode": "mxfp4"}
        (dst / "config.json").write_text(json.dumps(newcfg, indent=2))

    total = 0
    for li in range(lo, hi):
        f = dst / f"model-layer-{li:03d}.safetensors"
        if f.exists():
            print("skip", li, flush=True)
            continue
        if not layer_ready(src, li, cfg):
            print("layer", li, "NOT READY (shards incomplete)", flush=True)
            continue
        t0 = time.time()
        out = convert_layer(src, li, cfg, a.linears)
        mx.eval(out)
        tmp = dst / f".tmp-layer-{li:03d}.safetensors"
        mx.save_safetensors(str(tmp), out, metadata={"format": "mlx"})
        tmp.rename(f)
        nb = sum(v.nbytes for v in out.values())
        total += nb
        print(f"layer {li}: {len(out)} tensors {nb/1e9:.2f} GB  {time.time()-t0:.1f}s", flush=True)
        del out
        src.close()
        mx.clear_cache()
    print("done, bytes written", total)


if __name__ == "__main__":
    sys.exit(main())
