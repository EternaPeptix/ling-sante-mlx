"""Exact, weights-read-once MoE expert matvecs for B tokens x E experts (mxfp4 / nvfp4 / mxfp8).

Every (token, expert) output is bitwise identical to `mx.gather_qmm(...)` called for that token
alone at M=1 (`fp_gather_qmv` / `fp_gather_qmv_fast`). Inlines MLX's `fp_quantized.h` helpers.
"""
import os
import re
from functools import lru_cache

import mlx.core as mx

__all__ = ["gather_rows", "mlx_header"]

_INC_ROOT = os.path.join(os.path.dirname(mx.__file__), "include")
_INC_RE = re.compile(r'^\s*#\s*include\s+"(mlx/[^"]+)"\s*$')

_HELPERS = r"""
template <int bits, int values_per_thread>
inline void dequant_vals(const thread uint32_t* w, thread float* d) {
  if constexpr (bits == 4) {
    const thread uint16_t* ws = (const thread uint16_t*)w;
#pragma unroll
    for (int i = 0; i < (values_per_thread / 4); i++) {
      d[4 * i] = Dequantize<4>{}(ws[i]);
      d[4 * i + 1] = Dequantize<4>{}(ws[i] >> 4);
      d[4 * i + 2] = Dequantize<4>{}(ws[i] >> 8);
      d[4 * i + 3] = Dequantize<4>{}(ws[i] >> 12);
    }
  } else {
    const thread uint8_t* wb = (const thread uint8_t*)w;
#pragma unroll
    for (int i = 0; i < values_per_thread; i++) {
      d[i] = Dequantize<8>{}(wb[i]);
    }
  }
}

template <int bits, int values_per_thread>
inline float dot_vals(const thread float* x_thread, const thread float* d, float scale) {
  float accum = 0;
  if constexpr (bits == 4) {
#pragma unroll
    for (int i = 0; i < (values_per_thread / 4); i++) {
      accum +=
          (x_thread[4 * i] * d[4 * i] +
           x_thread[4 * i + 1] * d[4 * i + 1] +
           x_thread[4 * i + 2] * d[4 * i + 2] +
           x_thread[4 * i + 3] * d[4 * i + 3]);
    }
  } else {
#pragma unroll
    for (int i = 0; i < values_per_thread; i++) {
      accum += x_thread[i] * d[i];
    }
  }
  return scale * accum;
}
"""


@lru_cache(maxsize=None)
def mlx_header(rel="mlx/backend/metal/kernels/fp_quantized.h"):
    seen = set()

    def expand(path):
        if path in seen:
            return ""
        seen.add(path)
        out = []
        with open(os.path.join(_INC_ROOT, path)) as f:
            for line in f:
                m = _INC_RE.match(line)
                if m:
                    out.append(expand(m.group(1)))
                elif line.strip() == "#pragma once":
                    continue
                else:
                    out.append(line)
        return "".join(out)

    text = expand(rel)
    cut = text.index("template <typename U, int values_per_thread, int bits>\ninline void qouter")
    return text[:cut] + _HELPERS


_SRC = r"""
    const int E_ = idx_shape[1];
    const int NPAIR = idx_shape[0] * E_;
    constexpr int PF = 32 / BITS;
    constexpr int PPT = FAST ? 2 : 1;
    constexpr int VPT = PF * PPT;
    constexpr int BLOCK = VPT * 32;
    constexpr int SSTEP = GS / VPT;
    constexpr int ROWS = 4;
    typedef float U;
    typedef bfloat16_t T;

    uint3 tid = threadgroup_position_in_grid;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;
    const int q = tid.z;
    const uint e = idx[q];

    threadgroup uint pl[256];
    threadgroup int cnt_s;
    threadgroup int dup_s;
    if (simd_gid == 0) {
        int running = 0;
        bool dup = false;
        for (int base = 0; base < NPAIR; base += 32) {
            int p = base + (int)simd_lid;
            bool match = p < NPAIR && idx[p] == e;
            if (match && p < q) dup = true;
            int take = (match && p >= q) ? 1 : 0;
            int pre = simd_prefix_exclusive_sum(take);
            if (take) pl[running + pre] = (uint)p;
            running += simd_sum(take);
        }
        dup = simd_any(dup);
        if (simd_lid == 0) { cnt_s = running; dup_s = dup ? 1 : 0; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (dup_s) return;
    const int cnt = cnt_s;

    const int in_w = KK * (BITS) / 8;
    const int in_g = KK / GS;
    const int out_row = tid.y * (2 * ROWS) + simd_gid * ROWS;

    const device uint8_t* ws0 = (const device uint8_t*)w + (long)e * NN * in_w
        + (long)out_row * in_w + simd_lid * PPT * 4;
    const device uint8_t* sc0 = scales + (long)e * NN * in_g + (long)out_row * in_g + simd_lid / SSTEP;

    float inv_scale_enc = 1.0f;
    if constexpr (HGS) {
        inv_scale_enc = gsc[e] / (F8E4M3_MAX * F4E2M1_MAX);
    }

    for (int tb = 0; tb < cnt; tb += TMAX) {
        int xo[TMAX];
        int yo[TMAX];
        for (int t = 0; t < TMAX; t++) {
            int pq = (tb + t < cnt) ? (int)pl[tb + t] : q;
            xo[t] = (XPP ? pq : pq / E_) * KK + (int)simd_lid * VPT;
            yo[t] = pq * NN + out_row;
        }
        thread U result[TMAX][ROWS];
        for (int t = 0; t < TMAX; t++)
            for (int r = 0; r < ROWS; r++) result[t][r] = 0;

        const device uint8_t* ws = ws0;
        const device uint8_t* sc = sc0;
        int k = 0;
        constexpr int KEND_SUB = FAST ? 0 : BLOCK;
        for (; k < KK - KEND_SUB; k += BLOCK) {
            thread uint32_t rw[ROWS][PPT];
            U sv[ROWS];
#pragma unroll
            for (int r = 0; r < ROWS; r++) {
                const device uint32_t* wp = (const device uint32_t*)(ws + r * in_w);
#pragma unroll
                for (int p = 0; p < PPT; p++) rw[r][p] = wp[p];
                sv[r] = dequantize_scale<U, GS>((sc + r * in_g)[0]);
            }
#pragma unroll
            for (int t = 0; t < TMAX; t++) {
                if (tb + t < cnt) {
                    thread U xt[VPT];
                    load_vector<T, U, VPT>(x + xo[t] + k, xt);
#pragma unroll
                    for (int r = 0; r < ROWS; r++) {
                        thread U d[VPT];
                        dequant_vals<BITS, VPT>(rw[r], d);
                        result[t][r] += dot_vals<BITS, VPT>(xt, d, sv[r]);
                    }
                }
            }
            ws += BLOCK * 4 / PF;
            sc += BLOCK / GS;
        }
        if constexpr (!FAST) {
            const int rem = clamp((int)(KK - k - (int)simd_lid * VPT), 0, VPT);
            if (rem > 0) {
                for (int t = 0; t < TMAX; t++) {
                    if (tb + t < cnt) {
                        thread U xt[VPT];
                        load_vector_safe<T, U, VPT>(x + xo[t] + k, xt, rem);
                        for (int r = 0; r < ROWS; r++) {
                            const device uint8_t* wl = ws + r * in_w;
                            U s = dequantize_scale<U, GS>((sc + r * in_g)[0]);
                            result[t][r] += qdot_safe<U, VPT, BITS>(wl, xt, s, rem);
                        }
                    }
                }
            }
        }
#pragma unroll
        for (int t = 0; t < TMAX; t++) {
#pragma unroll
            for (int r = 0; r < ROWS; r++) {
                U v = simd_sum(result[t][r]);
                if (simd_lid == 0 && tb + t < cnt) {
                    if constexpr (HGS) {
                        y[yo[t] + r] = static_cast<T>(v * inv_scale_enc);
                    } else {
                        y[yo[t] + r] = static_cast<T>(v);
                    }
                }
            }
        }
    }
"""

_kernel = None


def _get_kernel():
    global _kernel
    if _kernel is None:
        _kernel = mx.fast.metal_kernel(
            name="bv3_gather_fpq",
            input_names=["w", "scales", "gsc", "x", "idx"],
            output_names=["y"],
            source=_SRC,
            header=mlx_header(),
            ensure_row_contiguous=True,
        )
    return _kernel


_TMAX_CAP = int(os.environ.get("LINGSANTE_GATHER_TMAX_CAP", "4"))


def _tmax(B):
    for t in (1, 2, 4, 8):
        if B <= t or t >= _TMAX_CAP:
            return t
    return 8


def gather_rows(x, w, scales, biases, idx, *, mode="mxfp4", group_size=32, bits=4, global_scale=None):
    if mode not in ("nvfp4", "mxfp8", "mxfp4"):
        raise ValueError(f"unsupported mode {mode!r}")
    if biases is not None:
        raise ValueError("biases not supported for fp-quantized mode")
    if x.dtype != mx.bfloat16:
        raise ValueError("x must be bfloat16")
    B, E = idx.shape
    xpp = x.ndim == 3
    K = x.shape[-1]
    N = w.shape[1]
    if N % 8 != 0 or K % 32 != 0 or K != scales.shape[-1] * group_size or w.shape[2] * 32 // bits != K:
        raise ValueError("unsupported shape")
    fast = (N % 8 == 0) and (K % ((32 // bits) * 2 * 32) == 0)
    hgs = global_scale is not None
    gsc = global_scale.astype(mx.float32) if hgs else mx.zeros((1,), mx.float32)
    tm = _tmax(B)
    (y,) = _get_kernel()(
        inputs=[w, scales, gsc, x, idx.astype(mx.uint32)],
        template=[
            ("GS", group_size), ("BITS", bits), ("HGS", hgs), ("FAST", fast), ("TMAX", tm),
            ("XPP", xpp), ("KK", K), ("NN", N),
        ],
        grid=(32, 2 * (N // 8), B * E),
        threadgroup=(32, 2, 1),
        output_shapes=[(B, E, N)],
        output_dtypes=[mx.bfloat16],
    )
    return y
