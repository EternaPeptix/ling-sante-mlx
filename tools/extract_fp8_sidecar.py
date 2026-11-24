"""Extract 3.634 MB FP8 scale_inv from 512S2 (read-only) and recover exact 648-tensor FP8 sidecar on 512S1.

Because bf16(E4M3_LUT[u8] * scale_inv) is a strict bijection on all 254 finite F8_E4M3 codes
(E4M3 has 3-bit significand >= 6.67% spacing vs BF16 7-bit <= 0.39% rounding error, plus signed zero
distinguishes 0x80/-0.0 vs 0x00/+0.0), transferring only the 3.634 MB `weight_scale_inv` tensors and
running exact Metal inverse bijection on 512S1's existing `model-layer-*.safetensors` reproduces the
source `F8_E4M3` bytes cleanly in seconds while verifying 0 bit diffs across all 648 FP8 tensors
and exact SHA-256 identity against sampled source layers.

SAFETY: Producer on 512S2 uses pure Python stdlib read-only file reads and never touches GPU/disk.
"""
import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(_root / "vendor"), str(_root)]

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
from mlx_lm.models.bailing_moe_v3.rows.gemv_fp8 import (  # noqa: E402
    E4M3_LUT_METAL,
    dequant_fp8_multilinear_to_bf16,
    dequant_fp8_to_bf16,
)

_PRODUCER_SCRIPT = r"""
import hashlib, json, struct, sys
from pathlib import Path

src_dir = Path(sys.argv[1]).expanduser()
sample_layers = set(int(x) for x in sys.argv[2].split(",") if x and x != "none")
num_layers = int(sys.argv[3])

records = []
for f in sorted(src_dir.glob("model-*.safetensors")):
    with open(f, "rb") as fh:
        hl = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(hl))
    base = 8 + hl
    for k, v in hdr.items():
        if k == "__metadata__" or v.get("dtype") != "F8_E4M3":
            continue
        if not k.startswith("model.layers."):
            continue
        li_str = k.split(".", 3)[2]
        if not li_str.isdigit():
            continue
        li = int(li_str)
        if li >= num_layers:
            continue
        sk = k + "_scale_inv"
        sv = hdr[sk]
        records.append((li, k, f.name, v["shape"], base + v["data_offsets"][0], base + v["data_offsets"][1],
                        sv["shape"], base + sv["data_offsets"][0], base + sv["data_offsets"][1]))

records.sort(key=lambda x: (x[0], x[1]))
open_files = {}
out = sys.stdout.buffer
for li, k, fname, w_sh, wa, wb, sc_sh, sa, sb in records:
    fh = open_files.get(fname)
    if fh is None:
        fh = open_files[fname] = open(src_dir / fname, "rb")
    fh.seek(sa)
    sc_bytes = fh.read(sb - sa)
    sha = None
    if li in sample_layers:
        h = hashlib.sha256()
        fh.seek(wa)
        rem = wb - wa
        while rem > 0:
            chunk = fh.read(min(1 << 22, rem))
            if not chunk:
                raise RuntimeError(f"unexpected EOF reading {k}")
            h.update(chunk)
            rem -= len(chunk)
        sha = h.hexdigest()
    meta = json.dumps({"key": k, "li": li, "w_shape": w_sh, "sc_shape": sc_sh, "sha256": sha}).encode()
    out.write(struct.pack("<II", len(meta), len(sc_bytes)))
    out.write(meta)
    out.write(sc_bytes)
out.write(struct.pack("<II", 0, 0))
out.flush()
for fh in open_files.values():
    fh.close()
"""

_RECOVER_HEADER = (
    "#include <metal_stdlib>\nusing namespace metal;\n"
    + E4M3_LUT_METAL
    + r"""
struct Bf4 { ushort v[4]; };
"""
)

_RECOVER_SRC = r"""
    const uint rb = threadgroup_position_in_grid.y;
    const uint cb = threadgroup_position_in_grid.x;
    const uint lid = thread_position_in_threadgroup.x;
    const uint sg_id = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;

    threadgroup ushort bf_tab[127];
    threadgroup uint sg_bad[4];

    const float sc = scale_inv[size_t(rb) * SK + cb];
    if (lid < 127) {
      bf_tab[lid] = as_type<ushort>(static_cast<bfloat16_t>(E4M3_LUT[lid] * sc));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint local_err = 0;
    const int c = int(cb) * 128 + int(lid);
    const int r0 = int(rb) * 128;
    const int r1 = min(r0 + 128, NN);
    if (c < KK) {
      for (int r = r0; r < r1; ++r) {
        ushort raw = as_type<ushort>(w_bf16[size_t(r) * KK + c]);
        uchar sign = (raw & 0x8000u) ? 0x80u : 0x00u;
        ushort mag = raw & 0x7fffu;
        int lo = 0, hi = 126;
        while (lo < hi) {
          int mid = (lo + hi) >> 1;
          if (bf_tab[mid] < mag) lo = mid + 1;
          else hi = mid;
        }
        if (bf_tab[lo] != mag) {
          local_err += 1u;
        }
        w_fp8[size_t(r) * KK + c] = sign | uchar(lo);
      }
    }
    uint s_bad = simd_sum(local_err);
    if (lane == 0) sg_bad[sg_id] = s_bad;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) {
      err_map[size_t(rb) * SK + cb] = sg_bad[0] + sg_bad[1] + sg_bad[2] + sg_bad[3];
    }
"""

_RECOVER_KERN = None


def _get_recover_kernel():
    global _RECOVER_KERN
    if _RECOVER_KERN is None:
        _RECOVER_KERN = mx.fast.metal_kernel(
            name="bv3_recover_fp8_from_bf16",
            input_names=["w_bf16", "scale_inv"],
            output_names=["w_fp8", "err_map"],
            header=_RECOVER_HEADER,
            source=_RECOVER_SRC,
            ensure_row_contiguous=True,
        )
    return _RECOVER_KERN


def recover_fp8_from_bf16(w_bf16: mx.array, scale_inv: mx.array) -> tuple[mx.array, int]:
    """Invert 2D bfloat16 tensor + (S_N, S_K) scale_inv into exact uint8 E4M3 + mismatch count."""
    N, K = w_bf16.shape
    SN, SK = scale_inv.shape
    assert w_bf16.dtype == mx.bfloat16 and scale_inv.dtype == mx.float32
    assert SN == (N + 127) // 128 and SK == (K + 127) // 128
    w_fp8, err_map = _get_recover_kernel()(
        inputs=[w_bf16, scale_inv],
        template=[("NN", N), ("KK", K), ("SK", SK)],
        grid=(SK * 128, SN, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(N, K), (SN, SK)],
        output_dtypes=[mx.uint8, mx.uint32],
    )
    mx.eval(w_fp8, err_map)
    return w_fp8, int(err_map.sum().item())


def _read_exact(stream, n: int) -> bytes:
    chunks = []
    rem = n
    while rem > 0:
        b = stream.read(rem)
        if not b:
            raise RuntimeError(f"unexpected EOF waiting for {n} bytes")
        chunks.append(b)
        rem -= len(b)
    return b"".join(chunks)


def fetch_scales_from_remote(ssh_host: str, src_dir: str, sample_layers: set[int], num_layers: int):
    t0 = time.perf_counter()
    sample_arg = ",".join(str(x) for x in sorted(sample_layers)) or "none"
    cmd = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        ssh_host,
        "python3",
        "-u",
        "-",
        src_dir,
        sample_arg,
        str(num_layers),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    proc.stdin.write(_PRODUCER_SCRIPT.encode())
    proc.stdin.close()

    by_layer = {li: [] for li in range(num_layers)}
    tot_sc_bytes = 0
    count = 0
    while True:
        hdr_len, sc_len = struct.unpack("<II", _read_exact(proc.stdout, 8))
        if hdr_len == 0 and sc_len == 0:
            break
        meta = json.loads(_read_exact(proc.stdout, hdr_len).decode())
        sc_raw = _read_exact(proc.stdout, sc_len)
        sc_arr = np.frombuffer(sc_raw, dtype=np.float32).reshape(meta["sc_shape"]).copy()
        by_layer[meta["li"]].append((meta["key"], tuple(meta["w_shape"]), sc_arr, meta["sha256"]))
        tot_sc_bytes += sc_len
        count += 1
    rc = proc.wait()
    err = proc.stderr.read().decode().strip()
    if rc != 0:
        raise RuntimeError(f"remote scale stream failed rc={rc}: {err}")
    dt = time.perf_counter() - t0
    print(f"Fetched {count} scale_inv tensors ({tot_sc_bytes/1e6:.3f} MB) in {dt:.2f}s", flush=True)
    return by_layer, count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", default="~/science/models/LingSante-3.1-flash-MXFP4")
    ap.add_argument("--dst-dir", default="~/science/models/lingsante-mlx-mxfp4-bf16")
    ap.add_argument("--ssh-host", default="100.123.125.48")
    ap.add_argument("--sample-layers", default="0,4,7,71")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    dst_dir = Path(os.path.expanduser(a.dst_dir))
    fp8_dir = dst_dir / "fp8"
    fp8_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((dst_dir / "config.json").read_text())
    num_layers = int(cfg["num_hidden_layers"])
    num_heads = int(cfg["num_attention_heads"])
    qk_nope = int(cfg["qk_nope_head_dim"])
    v_dim = int(cfg["v_head_dim"])
    sample_layers = {int(x) for x in a.sample_layers.split(",") if x}

    by_layer, sc_count = fetch_scales_from_remote(a.ssh_host, a.src_dir, sample_layers, num_layers)
    assert sc_count == 639, f"Expected 639 source F8_E4M3 weights in 0..{num_layers-1}, got {sc_count}"

    t0 = time.perf_counter()
    tot_tensors = 0
    tot_bytes = 0
    sha_checked = 0

    for li in range(num_layers):
        out_file = fp8_dir / f"model-fp8-layer-{li:03d}.safetensors"
        entries = by_layer[li]
        ref = mx.load(str(dst_dir / f"model-layer-{li:03d}.safetensors"))
        if out_file.exists() and not a.force:
            # Still verify existing file vs ref in fast pass
            saved = mx.load(str(out_file))
        else:
            saved = {}
            for key, w_shape, sc_np, exp_sha in entries:
                assert key in ref, f"Missing {key} in layer {li}"
                w_bf16 = ref[key]
                assert tuple(w_bf16.shape) == w_shape, f"Shape mismatch {key}: {w_bf16.shape} vs {w_shape}"
                sc_mx = mx.array(sc_np, dtype=mx.float32)
                w_fp8, err_cnt = recover_fp8_from_bf16(w_bf16, sc_mx)
                assert err_cnt == 0, f"Bijection recovery mismatch in {key}: err_cnt={err_cnt}"
                if exp_sha is not None:
                    got_sha = hashlib.sha256(memoryview(np.array(w_fp8))).hexdigest()
                    assert got_sha == exp_sha, f"SHA-256 mismatch for {key}: {got_sha} != {exp_sha}"
                    sha_checked += 1

                if key.endswith(".kv_b_proj.weight"):
                    base = key[: -len(".kv_b_proj.weight")]
                    head_dim = qk_nope + v_dim
                    w_3d = w_fp8.reshape(num_heads, head_dim, -1)
                    sc_2d = sc_mx.reshape(128, 4)
                    w_eq = mx.contiguous(w_3d[:, :qk_nope, :].swapaxes(-1, -2))
                    sc_eq = mx.contiguous(sc_2d[0::2, :])
                    w_uo = mx.contiguous(w_3d[:, qk_nope:, :])
                    sc_uo = mx.contiguous(sc_2d[1::2, :])
                    saved[f"{base}.embed_q.weight_fp8"] = w_eq
                    saved[f"{base}.embed_q.weight_scale_inv"] = sc_eq
                    saved[f"{base}.unembed_out.weight_fp8"] = w_uo
                    saved[f"{base}.unembed_out.weight_scale_inv"] = sc_uo
                else:
                    saved[key + "_fp8"] = w_fp8
                    saved[key + "_scale_inv"] = sc_mx

            mx.eval(saved)
            tmp = fp8_dir / f".tmp-fp8-layer-{li:03d}.safetensors"
            mx.save_safetensors(str(tmp), saved, metadata={"format": "mlx"})
            tmp.rename(out_file)

        # Forward-verify EVERY tensor against ref bf16 (0 bit diffs!)
        n_fp8 = 0
        for k in sorted(saved):
            if not k.endswith(".weight_fp8"):
                continue
            base = k[: -len(".weight_fp8")]
            w_fp8 = saved[k]
            sc = saved[f"{base}.weight_scale_inv"]
            if k.endswith(".embed_q.weight_fp8"):
                attn_pre = base[: -len(".embed_q")]
                kv_ref = ref[f"{attn_pre}.kv_b_proj.weight"].reshape(num_heads, qk_nope + v_dim, -1)
                w_ref = mx.contiguous(kv_ref[:, :qk_nope, :].swapaxes(-1, -2))
                w_got = dequant_fp8_multilinear_to_bf16(w_fp8, sc, scale_by_n=True)
            elif k.endswith(".unembed_out.weight_fp8"):
                attn_pre = base[: -len(".unembed_out")]
                kv_ref = ref[f"{attn_pre}.kv_b_proj.weight"].reshape(num_heads, qk_nope + v_dim, -1)
                w_ref = mx.contiguous(kv_ref[:, qk_nope:, :])
                w_got = dequant_fp8_multilinear_to_bf16(w_fp8, sc, scale_by_n=False)
            else:
                w_ref = ref[f"{base}.weight"]
                w_got = dequant_fp8_to_bf16(w_fp8, sc)
            mx.eval(w_got)
            diffs = int(mx.sum(w_got.view(mx.uint16) != w_ref.view(mx.uint16)).item())
            assert diffs == 0, f"Forward verify failed {k}: {diffs} bit diffs!"
            n_fp8 += 1
        nb = sum(v.nbytes for v in saved.values())
        tot_tensors += n_fp8
        tot_bytes += nb
        if li in (0, 4, 7, 71):
            print(f"  layer {li:2d}: {n_fp8} fp8 tensors  {nb/1e6:.1f} MB  verified 0 diffs", flush=True)
        del ref, saved
        mx.clear_cache()

    dt = time.perf_counter() - t0
    print(
        f"SUMMARY: {num_layers}/{num_layers} layers verified, {tot_tensors} MLX FP8 tensors, "
        f"sha_checked={sha_checked}, 0 bit mismatches, {tot_bytes/1e9:.3f} GB in {dt:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
