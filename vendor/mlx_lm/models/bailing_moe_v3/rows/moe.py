"""Fused bit-exact MXFP4 MoE kernels for BailingSparseMoE (B = 1..24).

1. `moe_gate_up_swiglu_rows(x, wg, sg, wu, su, inds, *, dedup=None, ...)`:
   Reads x via 128-bit `uint4`, deduplicates experts across B<=24 rows, computes both
   gate_proj & up_proj (MXFP4 K=4096, N=1280) with zero-spill regime-specialized tiling
   (`NSG=2` fine-grained 160-320 TGs/exp, `uint2` weight loads, MLX native `Dequantize<4>`,
   zero TG/barriers) and fuses `swiglu(gate, up)` in registers -> (B, 8, 1280) bf16.
2. `moe_down_combine_rows(h, wd, sd, inds, wts, *, shared=None, residual=None, dedup=None,
   se_mid=None, se_w_fp8=None, se_sc=None, ...)`:
   Computes down_proj (MXFP4 K=1280, N=4096) with `ROWS=4, NSG=8` (1024 TGs, 8 SGs/TG
   splitting unique experts finely to minimize dependent-layer tail waves; `ROWS=8` also
   supported via `rows=8`), `uint4` `x` loads, zero-atomic static stride, left-fold `combine8`,
   optional zero-atomic parallel FP8 `se.down_proj` fusion (`T7`), and residual add -> (B, 4096) bf16.
3. Optional `T3` runtime-on-load contiguous K-major weight+scale repacking (0 extra steady-state RAM):
   `pack_moe_gu_weights` / `unpack_moe_gu_weights` (`[E, N//4, 17408]` u8, 0B pad),
   `pack_moe_dn_weights` / `unpack_moe_dn_weights` (`[E, N//4, 2720]` u8, 0B pad),
   `moe_gate_up_swiglu_packed_rows`, and `moe_down_combine_packed_rows`.
"""
import os

import mlx.core as mx

from .gather import mlx_header
from .glue import GLUE_HEADER
from .router import LPT_DEDUP_HELPER

_HELPERS = r"""
inline float4 fp4_decode4_mlx(uint16_t w16) {
  return float4(
      Dequantize<4>{}(uint8_t(w16)),
      Dequantize<4>{}(uint8_t(w16 >> 4)),
      Dequantize<4>{}(uint8_t(w16 >> 8)),
      Dequantize<4>{}(uint8_t(w16 >> 12)));
}

inline float fp4_dot4_mlx(uint16_t w16, bfloat4 xv) {
  const float4 d = fp4_decode4_mlx(w16);
  return ((float(xv.x) * d.x + float(xv.y) * d.y) + float(xv.z) * d.z) + float(xv.w) * d.w;
}

inline float fp4_dot8_u4_mlx(uint32_t w32, uint4 xu) {
  float acc = 0.0f;
  acc += fp4_dot4_mlx(uint16_t(w32), as_type<bfloat4>(xu.xy));
  acc += fp4_dot4_mlx(uint16_t(w32 >> 16), as_type<bfloat4>(xu.zw));
  return acc;
}

inline float fp4_dot16_u4_mlx(uint2 w64, uint4 xu0, uint4 xu1) {
  float acc = 0.0f;
  acc += fp4_dot4_mlx(uint16_t(w64.x), as_type<bfloat4>(xu0.xy));
  acc += fp4_dot4_mlx(uint16_t(w64.x >> 16), as_type<bfloat4>(xu0.zw));
  acc += fp4_dot4_mlx(uint16_t(w64.y), as_type<bfloat4>(xu1.xy));
  acc += fp4_dot4_mlx(uint16_t(w64.y >> 16), as_type<bfloat4>(xu1.zw));
  return acc;
}

// Zero-Mul-256 exact F8_E4M3FN uchar4 * (sc * 256.0f) -> bfloat4 (0 bit diffs vs gemv_fp8_rows).
inline bfloat4 fp8_decode4_bf_moe(uchar4 pk, float sc256) {
  const ushort4 b4 = ushort4(pk);
  const ushort4 v4 = b4 & 127u;
  const ushort4 u4 = (v4 << 7) | (((v4 + 1u) >> 7) << 14) | ((b4 & 128u) << 8);
  return bfloat4(float4(as_type<half4>(u4)) * sc256);
}

template <int bits, int values_per_thread>
inline void dequant_vals_u32(const thread uint32_t* w, thread float* d) {
  if constexpr (bits == 4) {
#pragma unroll
    for (int p = 0; p < (values_per_thread / 8); p++) {
      const uint32_t pk = w[p];
      d[8 * p + 0] = Dequantize<4>{}(uint8_t(pk));
      d[8 * p + 1] = Dequantize<4>{}(uint8_t(pk >> 4));
      d[8 * p + 2] = Dequantize<4>{}(uint8_t(pk >> 8));
      d[8 * p + 3] = Dequantize<4>{}(uint8_t(pk >> 12));
      d[8 * p + 4] = Dequantize<4>{}(uint8_t(pk >> 16));
      d[8 * p + 5] = Dequantize<4>{}(uint8_t(pk >> 20));
      d[8 * p + 6] = Dequantize<4>{}(uint8_t(pk >> 24));
      d[8 * p + 7] = Dequantize<4>{}(uint8_t(pk >> 28));
    }
  } else {
#pragma unroll
    for (int p = 0; p < (values_per_thread / 4); p++) {
      const uint32_t pk = w[p];
      d[4 * p + 0] = Dequantize<8>{}(uint8_t(pk));
      d[4 * p + 1] = Dequantize<8>{}(uint8_t(pk >> 8));
      d[4 * p + 2] = Dequantize<8>{}(uint8_t(pk >> 16));
      d[4 * p + 3] = Dequantize<8>{}(uint8_t(pk >> 24));
    }
  }
}

// Zero-spill vectorized gate/up pass (uint4 x + uint2 w + Dequantize<4>, no xt[16], no pqs[T]).
template <int GS, int R_TILE, int T_TILE, bool WRITE_SWIGLU>
inline void moe_gu_fast_pass(
    const device uint8_t* w0,
    const device uint8_t* s0,
    const device bfloat16_t* x,
    thread const int* xo,
    const int n_tok,
    const int r0,
    const int out_row,
    const int NN,
    const int KK,
    const int in_w,
    const int in_g,
    const uint lane,
    thread bfloat16_t g_bf[T_TILE][R_TILE],
    const device uint* u_pqs,
    const int pq_base,
    const int pq_single,
    device bfloat16_t* out) {
  float res[T_TILE][R_TILE];
#pragma unroll
  for (int t = 0; t < T_TILE; t++)
#pragma unroll
    for (int r = 0; r < R_TILE; r++) res[t][r] = 0.0f;

  const device uint8_t* wp = w0 + (long)r0 * in_w;
  const device uint8_t* sp = s0 + (long)r0 * in_g;
  for (int k = 0; k < KK; k += 512) {
    if constexpr (T_TILE == 1) {
      const device uint4* xp = (const device uint4*)(x + xo[0] + k);
      const uint4 xu0 = xp[0], xu1 = xp[1];
#pragma unroll
      for (int r = 0; r < R_TILE; r++) {
        const int rr = (out_row + r0 + r < NN) ? r : 0;
        const uint2 rw = *(const device uint2*)(wp + rr * in_w);
        const float sv = dequantize_scale<float, GS>((sp + rr * in_g)[0]);
        res[0][r] += sv * fp4_dot16_u4_mlx(rw, xu0, xu1);
      }
    } else {
#pragma unroll
      for (int r = 0; r < R_TILE; r++) {
        const int rr = (out_row + r0 + r < NN) ? r : 0;
        const uint2 rw = *(const device uint2*)(wp + rr * in_w);
        const float sv = dequantize_scale<float, GS>((sp + rr * in_g)[0]);
        const float4 d0 = fp4_decode4_mlx(uint16_t(rw.x));
        const float4 d1 = fp4_decode4_mlx(uint16_t(rw.x >> 16));
        const float4 d2 = fp4_decode4_mlx(uint16_t(rw.y));
        const float4 d3 = fp4_decode4_mlx(uint16_t(rw.y >> 16));
#pragma unroll
        for (int t = 0; t < T_TILE; t++) {
          if (t < n_tok) {
            const device uint4* xp = (const device uint4*)(x + xo[t] + k);
            const uint4 xu0 = xp[0], xu1 = xp[1];
            const bfloat4 xv0 = as_type<bfloat4>(xu0.xy);
            const bfloat4 xv1 = as_type<bfloat4>(xu0.zw);
            const bfloat4 xv2 = as_type<bfloat4>(xu1.xy);
            const bfloat4 xv3 = as_type<bfloat4>(xu1.zw);
            float acc = 0.0f;
            acc += ((float(xv0.x) * d0.x + float(xv0.y) * d0.y) + float(xv0.z) * d0.z) + float(xv0.w) * d0.w;
            acc += ((float(xv1.x) * d1.x + float(xv1.y) * d1.y) + float(xv1.z) * d1.z) + float(xv1.w) * d1.w;
            acc += ((float(xv2.x) * d2.x + float(xv2.y) * d2.y) + float(xv2.z) * d2.z) + float(xv2.w) * d2.w;
            acc += ((float(xv3.x) * d3.x + float(xv3.y) * d3.y) + float(xv3.z) * d3.z) + float(xv3.w) * d3.w;
            res[t][r] += sv * acc;
          }
        }
      }
    }
    wp += 256;
    sp += 16;
  }
#pragma unroll
  for (int t = 0; t < T_TILE; t++) {
    const int pq = (WRITE_SWIGLU && lane == 0 && t < n_tok) ? ((pq_base >= 0) ? int(u_pqs[pq_base + t]) : pq_single) : 0;
#pragma unroll
    for (int r = 0; r < R_TILE; r++) {
      const float v = simd_sum(res[t][r]);
      if (lane == 0 && t < n_tok && (out_row + r0 + r) < NN) {
        if constexpr (WRITE_SWIGLU) {
          out[pq * NN + out_row + r0 + r] = glue_swiglu_bf16(g_bf[t][r], static_cast<bfloat16_t>(v));
        } else {
          g_bf[t][r] = static_cast<bfloat16_t>(v);
        }
      }
    }
  }
}

// Lean R=1, T=4 FP8 se.down_proj helper (4 float acc regs, exact SN=32 TN=4 order -> 0 bit diffs!).
template <int BB, int KK, int ROWS_TG>
inline void moe_se_down_fp8_r1(
    const device uchar* se_w,
    const device float* se_sc,
    const device bfloat16_t* se_mid,
    const int out_row,
    const int r,
    const uint lane,
    threadgroup bfloat16_t* sh_s) {
  constexpr int k_blocks = KK >> 7;
  const int row_blk = out_row >> 7;
  const device float* scp = se_sc + size_t(row_blk) * k_blocks;
  const device uchar* wp = se_w + size_t(out_row + r) * KK + int(lane) * 4;
  for (int tb = 0; tb < BB; tb += 4) {
    const int n_tok = min(4, BB - tb);
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int bn = int(lane) * 4;
    for (int i = 0; i < k_blocks; ++i) {
      const float sc256 = simd_broadcast_first(scp[i] * 256.0f);
      const bfloat4 w0 = fp8_decode4_bf_moe(*(const device uchar4*)(wp + i * 128), sc256);
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        if (t < n_tok) {
          const bfloat4 xt = *(const device bfloat4*)(se_mid + size_t(tb + t) * KK + bn);
          acc[t] += float(w0.x) * float(xt.x);
          acc[t] += float(w0.y) * float(xt.y);
          acc[t] += float(w0.z) * float(xt.z);
          acc[t] += float(w0.w) * float(xt.w);
        }
      }
      bn += 128;
    }
#pragma unroll
    for (int t = 0; t < 4; ++t) {
      if (t < n_tok) {
        float a0 = acc[t];
#pragma unroll
        for (ushort sn = 16; sn >= 1; sn >>= 1) a0 += simd_shuffle_down(a0, sn);
        if (lane == 0) sh_s[(tb + t) * ROWS_TG + r] = static_cast<bfloat16_t>(a0);
      }
    }
  }
}
"""

HEADER = mlx_header() + GLUE_HEADER + _HELPERS
_KERNELS = {}
_GU_TMAX_CAP = int(os.environ.get("LINGSANTE_MOE_GU_TMAX_CAP", os.environ.get("LINGSANTE_MOE_TMAX_CAP", "4")))
_DC_TMAX_CAP = int(os.environ.get("LINGSANTE_MOE_DC_TMAX_CAP", os.environ.get("LINGSANTE_MOE_TMAX_CAP", "2")))
_ZERO_BF16 = mx.zeros((8,), dtype=mx.bfloat16)
_ZERO_U8 = mx.zeros((8,), dtype=mx.uint8)
_ZERO_F32 = mx.zeros((8,), dtype=mx.float32)
_ZERO_U32 = mx.zeros((8,), dtype=mx.uint32)
_ZERO_I32 = mx.zeros((8,), dtype=mx.int32)


def _tmax(B, cap):
    for t in (1, 2, 4):
        if B <= t or t >= cap:
            return t
    return 4


# ---------------------------------------------------------------------------
# 1. LPT (cnt DESC, exp ASC) Packed CSR Deduplication
# ---------------------------------------------------------------------------

_DEDUP_SRC = r"""
    constexpr int NPAIR = BB * 8;
    const uint lane = thread_index_in_simdgroup;
    threadgroup uint ind_s[NPAIR];
    threadgroup uchar seen[NPAIR];
    threadgroup uint raw_key[NPAIR];
    threadgroup uint raw_meta[NPAIR];
    for (int i = int(lane); i < NPAIR; i += 32) ind_s[i] = inds[i];
    simdgroup_barrier(mem_flags::mem_threadgroup);
    lpt_dedup_sg<NPAIR>(ind_s, seen, raw_key, raw_meta, u_meta, u_pqs, n_hot, n_uniq, lane);
"""


def _dedup_kernel():
    k = _KERNELS.get("dedup")
    if k is None:
        k = _KERNELS["dedup"] = mx.fast.metal_kernel(
            name="bv3_moe_dedup_lpt",
            input_names=["inds"],
            output_names=["u_meta", "u_pqs", "n_hot", "n_uniq"],
            header=GLUE_HEADER + LPT_DEDUP_HELPER,
            source=_DEDUP_SRC,
            ensure_row_contiguous=True,
        )
    return k


def moe_dedup_rows(inds: mx.array):
    """Deduplicate `(B, 8)` expert indices into LPT `(cnt DESC, exp ASC)` packed CSR worklists."""
    B, E = inds.shape
    assert 1 <= B <= 24 and E == 8
    NPAIR = B * 8
    inds_u32 = inds if inds.dtype == mx.uint32 else inds.astype(mx.uint32)
    u_meta, u_pqs, n_hot, n_uniq = _dedup_kernel()(
        inputs=[inds_u32],
        template=[("BB", B)],
        grid=(32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(NPAIR,), (NPAIR,), (1,), (1,)],
        output_dtypes=[mx.uint32, mx.uint32, mx.int32, mx.int32],
    )
    return (u_meta, u_pqs, n_hot, n_uniq, n_uniq)


# ---------------------------------------------------------------------------
# 2. Zero-Spill Vectorized Gate + Up + SwiGLU (FAST 4-bit + Generic Fallback)
# ---------------------------------------------------------------------------

_GU_FAST_SRC = r"""
    typedef bfloat16_t T;
    const uint3 tid = threadgroup_position_in_grid;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const int out_row = int(tid.y) * (ROWS * NSG) + int(sg) * ROWS;
    if (out_row >= NN) return;

    const int in_w = KK / 2;
    const int in_g = KK / GS;
    const int nu = USE_DEDUP ? n_uniq[0] : (BB * 8);
    const int gz_step = (GZ > 0) ? GZ : (BB * 8);

    for (int q = int(tid.z); q < nu; q += gz_step) {
      uint e;
      int off0 = 0;
      int cnt = 1;
      if constexpr (USE_DEDUP) {
        const uint meta = u_meta[q];
        e = meta & 0x3FFu;
        off0 = int((meta >> 10) & 0xFFu);
        cnt = int((meta >> 18) & 0xFFu);
      } else {
        e = inds[q];
        bool dup = false;
        for (int p = 0; p < q; ++p) {
          if (inds[p] == e) dup = true;
        }
        if (dup) continue;
      }

      const device uint8_t* wg0 = (const device uint8_t*)wg + (long)e * NN * in_w + (long)out_row * in_w + lane * 8;
      const device uint8_t* sg0 = scg + (long)e * NN * in_g + (long)out_row * in_g + (lane >> 1);
      const device uint8_t* wu0 = (const device uint8_t*)wu + (long)e * NN * in_w + (long)out_row * in_w + lane * 8;
      const device uint8_t* su0 = scu + (long)e * NN * in_g + (long)out_row * in_g + (lane >> 1);

      if constexpr (!USE_DEDUP) {
        // B == 1 zero-barrier path (handles even synthetic duplicate inds cleanly!)
        int xo[1] = { int(lane) * 16 };
        T g_bf[1][ROWS];
        moe_gu_fast_pass<GS, ROWS, 1, false>(wg0, sg0, x, xo, 1, 0, out_row, NN, KK, in_w, in_g, lane, g_bf, u_pqs, -1, q, out);
        float ru[ROWS];
#pragma unroll
        for (int r = 0; r < ROWS; r++) ru[r] = 0.0f;
        const device uint8_t* wp = wu0;
        const device uint8_t* sp = su0;
        for (int k = 0; k < KK; k += 512) {
          const device uint4* xp = (const device uint4*)(x + xo[0] + k);
          const uint4 xu0 = xp[0], xu1 = xp[1];
#pragma unroll
          for (int r = 0; r < ROWS; r++) {
            const int rr = (out_row + r < NN) ? r : 0;
            ru[r] += dequantize_scale<float, GS>((sp + rr * in_g)[0]) * fp4_dot16_u4_mlx(*(const device uint2*)(wp + rr * in_w), xu0, xu1);
          }
          wp += 256; sp += 16;
        }
#pragma unroll
        for (int r = 0; r < ROWS; r++) {
          const float vu = simd_sum(ru[r]);
          if (lane == 0 && (out_row + r) < NN) {
            const T swv = glue_swiglu_bf16(g_bf[0][r], static_cast<T>(vu));
            for (int p = q; p < 8; ++p) {
              if (inds[p] == e) out[p * NN + out_row + r] = swv;
            }
          }
        }
      } else if constexpr (ADAPTIVE && ROWS == 4) {
        if (cnt == 1 || TMAX == 1) {
          const int pq0 = int(u_pqs[off0]);
          int xo[1] = { (pq0 / 8) * KK + int(lane) * 16 };
          T g_bf[1][4];
          moe_gu_fast_pass<GS, 4, 1, false>(wg0, sg0, x, xo, 1, 0, out_row, NN, KK, in_w, in_g, lane, g_bf, u_pqs, off0, pq0, out);
          moe_gu_fast_pass<GS, 4, 1, true>(wu0, su0, x, xo, 1, 0, out_row, NN, KK, in_w, in_g, lane, g_bf, u_pqs, off0, pq0, out);
        } else if (cnt == 2) {
          int xo[2];
          for (int t = 0; t < 2; t++) xo[t] = (int(u_pqs[off0 + t]) / 8) * KK + int(lane) * 16;
          T g_bf[2][4];
          moe_gu_fast_pass<GS, 4, 2, false>(wg0, sg0, x, xo, 2, 0, out_row, NN, KK, in_w, in_g, lane, g_bf, u_pqs, off0, 0, out);
          moe_gu_fast_pass<GS, 4, 2, true>(wu0, su0, x, xo, 2, 0, out_row, NN, KK, in_w, in_g, lane, g_bf, u_pqs, off0, 0, out);
        } else if (cnt <= 4 || TMAX <= 4) {
          for (int r0 = 0; r0 < 4; r0 += 2) {
            if (out_row + r0 >= NN) continue;
            for (int tb = 0; tb < cnt; tb += 4) {
              const int n_tok = min(4, cnt - tb);
              int xo[4];
              for (int t = 0; t < 4; t++) {
                const int pq = (t < n_tok) ? int(u_pqs[off0 + tb + t]) : int(u_pqs[off0]);
                xo[t] = (pq / 8) * KK + int(lane) * 16;
              }
              T g_bf[4][2];
              moe_gu_fast_pass<GS, 2, 4, false>(wg0, sg0, x, xo, n_tok, r0, out_row, NN, KK, in_w, in_g, lane, g_bf, u_pqs, off0 + tb, 0, out);
              moe_gu_fast_pass<GS, 2, 4, true>(wu0, su0, x, xo, n_tok, r0, out_row, NN, KK, in_w, in_g, lane, g_bf, u_pqs, off0 + tb, 0, out);
            }
          }
        } else {
          for (int r0 = 0; r0 < 4; r0++) {
            if (out_row + r0 >= NN) continue;
            for (int tb = 0; tb < cnt; tb += 8) {
              const int n_tok = min(8, cnt - tb);
              int xo[8];
              for (int t = 0; t < 8; t++) {
                const int pq = (t < n_tok) ? int(u_pqs[off0 + tb + t]) : int(u_pqs[off0]);
                xo[t] = (pq / 8) * KK + int(lane) * 16;
              }
              T g_bf[8][1];
              moe_gu_fast_pass<GS, 1, 8, false>(wg0, sg0, x, xo, n_tok, r0, out_row, NN, KK, in_w, in_g, lane, g_bf, u_pqs, off0 + tb, 0, out);
              moe_gu_fast_pass<GS, 1, 8, true>(wu0, su0, x, xo, n_tok, r0, out_row, NN, KK, in_w, in_g, lane, g_bf, u_pqs, off0 + tb, 0, out);
            }
          }
        }
      } else {
        for (int tb = 0; tb < cnt; tb += TMAX) {
          const int n_tok = min(TMAX, cnt - tb);
          int xo[TMAX];
          for (int t = 0; t < TMAX; t++) {
            const int pq = (t < n_tok) ? int(u_pqs[off0 + tb + t]) : int(u_pqs[off0]);
            xo[t] = (pq / 8) * KK + int(lane) * 16;
          }
          T g_bf[TMAX][ROWS];
          moe_gu_fast_pass<GS, ROWS, TMAX, false>(wg0, sg0, x, xo, n_tok, 0, out_row, NN, KK, in_w, in_g, lane, g_bf, u_pqs, off0 + tb, 0, out);
          moe_gu_fast_pass<GS, ROWS, TMAX, true>(wu0, su0, x, xo, n_tok, 0, out_row, NN, KK, in_w, in_g, lane, g_bf, u_pqs, off0 + tb, 0, out);
        }
      }
    }
"""

# Generic fallback for inline dedup (B >= 2, dedup=None) or non-fast/non-4bit shapes.
_GU_INLINE_SRC = r"""
    constexpr int NPAIR = BB * 8;
    constexpr int PF = 32 / BITS;
    constexpr int PPT = FAST ? 2 : 1;
    constexpr int VPT = PF * PPT;
    constexpr int BLOCK = VPT * 32;
    constexpr int SSTEP = GS / VPT;
    typedef float U;
    typedef bfloat16_t T;

    uint3 tid = threadgroup_position_in_grid;
    uint sg = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint ttid = sg * 32 + lane;
    const int q = tid.z;

    threadgroup uint ind_s[NPAIR];
    threadgroup uint pl[NPAIR];
    for (int i = int(ttid); i < NPAIR; i += 32 * NSG) ind_s[i] = inds[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint e = ind_s[q];
    threadgroup int dupc[2];
    if (sg == 0) {
      int cnt = 0;
      bool dup = false;
      for (int base = 0; base < NPAIR; base += 32) {
        int p = base + int(lane);
        bool match = (p < NPAIR) && (ind_s[p] == e);
        if (match && p < q) dup = true;
        int take = (match && p >= q) ? 1 : 0;
        int pre = simd_prefix_exclusive_sum(take);
        if (take) pl[cnt + pre] = (uint)p;
        cnt += simd_sum(take);
      }
      dup = simd_any(dup);
      if (lane == 0) { dupc[0] = cnt; dupc[1] = dup ? 1 : 0; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (dupc[1]) return;
    const int cnt = dupc[0];

    const int in_w = KK * BITS / 8;
    const int in_g = KK / GS;
    const int out_row = tid.y * (ROWS * NSG) + sg * ROWS;
    if (out_row >= NN) return;

    const device uint8_t* wg0 = (const device uint8_t*)wg + (long)e * NN * in_w + (long)out_row * in_w + lane * PPT * 4;
    const device uint8_t* sg0 = scg + (long)e * NN * in_g + (long)out_row * in_g + lane / SSTEP;
    const device uint8_t* wu0 = (const device uint8_t*)wu + (long)e * NN * in_w + (long)out_row * in_w + lane * PPT * 4;
    const device uint8_t* su0 = scu + (long)e * NN * in_g + (long)out_row * in_g + lane / SSTEP;

    for (int tb = 0; tb < cnt; tb += TMAX) {
      int xo[TMAX], pqs[TMAX];
      for (int t = 0; t < TMAX; t++) {
        int pq = (tb + t < cnt) ? (int)pl[tb + t] : q;
        pqs[t] = pq;
        xo[t] = (pq / 8) * KK + int(lane) * VPT;
      }
      T g_bf[TMAX][ROWS];
      constexpr int KEND_SUB = FAST ? 0 : BLOCK;
      for (int pass = 0; pass < 2; ++pass) {
        thread U res[TMAX][ROWS];
        for (int t = 0; t < TMAX; t++)
          for (int r = 0; r < ROWS; r++) res[t][r] = 0;
        const device uint8_t* wp = (pass == 0) ? wg0 : wu0;
        const device uint8_t* sp = (pass == 0) ? sg0 : su0;
        int k = 0;
        for (; k < KK - KEND_SUB; k += BLOCK) {
          for (int r = 0; r < ROWS; r++) {
            const int rr = (out_row + r < NN) ? r : 0;
            const device uint32_t* wptr = (const device uint32_t*)(wp + rr * in_w);
            thread uint32_t rw[PPT];
            for (int p = 0; p < PPT; p++) rw[p] = wptr[p];
            U sv = dequantize_scale<U, GS>((sp + rr * in_g)[0]);
            thread U d[VPT];
            dequant_vals_u32<BITS, VPT>(rw, d);
            for (int t = 0; t < TMAX; t++) {
              if (tb + t < cnt) {
                thread U xt[VPT];
                load_vector<T, U, VPT>(x + xo[t] + k, xt);
                res[t][r] += dot_vals<BITS, VPT>(xt, d, sv);
              }
            }
          }
          wp += BLOCK * 4 / PF;
          sp += BLOCK / GS;
        }
        if constexpr (!FAST) {
          const int rem = clamp((int)(KK - k - int(lane) * VPT), 0, VPT);
          if (rem > 0) {
            for (int t = 0; t < TMAX; t++) {
              if (tb + t < cnt) {
                thread U xt[VPT];
                load_vector_safe<T, U, VPT>(x + xo[t] + k, xt, rem);
                for (int r = 0; r < ROWS; r++) {
                  if (out_row + r < NN) {
                    res[t][r] += qdot_safe<U, VPT, BITS>(wp + r * in_w, xt, dequantize_scale<U, GS>((sp + r * in_g)[0]), rem);
                  }
                }
              }
            }
          }
        }
        for (int t = 0; t < TMAX; t++) {
          for (int r = 0; r < ROWS; r++) {
            U v = simd_sum(res[t][r]);
            if (pass == 0) g_bf[t][r] = static_cast<T>(v);
            else if (lane == 0 && tb + t < cnt && out_row + r < NN) {
              out[pqs[t] * NN + out_row + r] = glue_swiglu_bf16(g_bf[t][r], static_cast<T>(v));
            }
          }
        }
      }
    }
"""


# ---------------------------------------------------------------------------
# 3. High-Bandwidth ROWS=8 Down + Combine + Optional T7 se_down + Residual
# ---------------------------------------------------------------------------

_DC_FAST_SRC = r"""
    constexpr int NPAIR = BB * 8;
    typedef bfloat16_t T;

    const uint3 tid = threadgroup_position_in_grid;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint ttid = sg * 32 + lane;
    const int in_w = KK / 2;
    const int in_g = KK / GS;

    threadgroup bfloat16_t sh_s[HAS_SE_DOWN ? (BB * ROWS) : 1];
    threadgroup bfloat16_t dn_s[NPAIR * ROWS];
    const int nu = USE_DEDUP ? n_uniq[0] : (BB * 8);
    const int n_tiles = NN / ROWS;
    const int ntg_safe = (NTG_Y > 0) ? NTG_Y : 1;
    const int t_beg = (NTG_Y > 0) ? (int(tid.y) * n_tiles) / ntg_safe : int(tid.y);
    const int t_end = (NTG_Y > 0) ? ((int(tid.y) + 1) * n_tiles) / ntg_safe : (int(tid.y) + 1);

    for (int tile = t_beg; tile < t_end; ++tile) {
      const int out_row = tile * ROWS;
      if (tile > t_beg) threadgroup_barrier(mem_flags::mem_threadgroup);

      if constexpr (HAS_SE_DOWN) {
        for (int r = int(sg); r < ROWS; r += NSG) {
          moe_se_down_fp8_r1<BB, KK, ROWS>(se_w, se_sc, se_mid, out_row, r, lane, sh_s);
        }
      }

      for (int u = int(sg); u < nu; u += NSG) {
        uint e;
        int off0 = 0;
        int cnt = 1;
        if constexpr (USE_DEDUP) {
          const uint meta = u_meta[u];
          e = meta & 0x3FFu;
          off0 = int((meta >> 10) & 0xFFu);
          cnt = int((meta >> 18) & 0xFFu);
        } else {
          e = inds[u];
          bool dup = false;
          for (int p = 0; p < u; ++p) {
            if (inds[p] == e) dup = true;
          }
          if (dup) continue;
        }

        const device uint8_t* ws0 = (const device uint8_t*)w + (long)e * NN * in_w + (long)out_row * in_w + lane * 4;
        const device uint8_t* sc0 = scales + (long)e * NN * in_g + (long)out_row * in_g + (lane >> 2);

        if constexpr (!USE_DEDUP) {
          // B == 1 zero-barrier path (handles even synthetic duplicate inds cleanly!)
          const int xo0 = u * KK + int(lane) * 8;
          float res[ROWS];
#pragma unroll
          for (int r = 0; r < ROWS; r++) res[r] = 0.0f;
          const device uint8_t* ws = ws0;
          const device uint8_t* sc = sc0;
          for (int k = 0; k < KK; k += 256) {
            const uint4 xu = *(const device uint4*)(x + xo0 + k);
#pragma unroll
            for (int r = 0; r < ROWS; r++) {
              const uint32_t rw = *(const device uint32_t*)(ws + r * in_w);
              const float sv = dequantize_scale<float, GS>((sc + r * in_g)[0]);
              res[r] += sv * fp4_dot8_u4_mlx(rw, xu);
            }
            ws += 128;
            sc += 8;
          }
#pragma unroll
          for (int r = 0; r < ROWS; r++) {
            const float v = simd_sum(res[r]);
            if (lane == 0) {
              const T bv = static_cast<T>(v);
              for (int p = u; p < 8; ++p) {
                if (inds[p] == e) dn_s[p * ROWS + r] = bv;
              }
            }
          }
        } else {
          for (int tb = 0; tb < cnt; tb += TMAX) {
            const int n_tok = min(TMAX, cnt - tb);
            int xo[TMAX];
#pragma unroll
            for (int t = 0; t < TMAX; t++) {
              const int pq = (t < n_tok) ? int(u_pqs[off0 + tb + t]) : int(u_pqs[off0]);
              xo[t] = pq * KK + int(lane) * 8;
            }
            float res[TMAX][ROWS];
#pragma unroll
            for (int t = 0; t < TMAX; t++)
#pragma unroll
              for (int r = 0; r < ROWS; r++) res[t][r] = 0.0f;

            const device uint8_t* ws = ws0;
            const device uint8_t* sc = sc0;
            for (int k = 0; k < KK; k += 256) {
              uint4 xu[TMAX];
#pragma unroll
              for (int t = 0; t < TMAX; t++) {
                if (t < n_tok) xu[t] = *(const device uint4*)(x + xo[t] + k);
              }
#pragma unroll
              for (int r = 0; r < ROWS; r++) {
                const uint32_t rw = *(const device uint32_t*)(ws + r * in_w);
                const float sv = dequantize_scale<float, GS>((sc + r * in_g)[0]);
                const float4 d0 = fp4_decode4_mlx(uint16_t(rw));
                const float4 d1 = fp4_decode4_mlx(uint16_t(rw >> 16));
#pragma unroll
                for (int t = 0; t < TMAX; t++) {
                  if (t < n_tok) {
                    const bfloat4 xv0 = as_type<bfloat4>(xu[t].xy);
                    const bfloat4 xv1 = as_type<bfloat4>(xu[t].zw);
                    float acc = 0.0f;
                    acc += ((float(xv0.x) * d0.x + float(xv0.y) * d0.y) + float(xv0.z) * d0.z) + float(xv0.w) * d0.w;
                    acc += ((float(xv1.x) * d1.x + float(xv1.y) * d1.y) + float(xv1.z) * d1.z) + float(xv1.w) * d1.w;
                    res[t][r] += sv * acc;
                  }
                }
              }
              ws += 128;
              sc += 8;
            }
#pragma unroll
            for (int t = 0; t < TMAX; t++) {
              const int pq = (lane == 0 && t < n_tok) ? int(u_pqs[off0 + tb + t]) : 0;
#pragma unroll
              for (int r = 0; r < ROWS; r++) {
                const float v = simd_sum(res[t][r]);
                if (lane == 0 && t < n_tok) dn_s[pq * ROWS + r] = static_cast<T>(v);
              }
            }
          }
        }
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);

      for (int idx = int(ttid); idx < BB * ROWS; idx += 32 * NSG) {
        const int b = idx / ROWS;
        const int r = idx % ROWS;
        const int c = out_row + r;
        float ev[8], ww[8];
#pragma unroll
        for (int j = 0; j < 8; j++) {
          ev[j] = static_cast<float>(dn_s[(b * 8 + j) * ROWS + r]);
          ww[j] = wts[b * 8 + j];
        }
        bfloat16_t val = static_cast<bfloat16_t>(glue_moe_combine8(ev, ww));
        if constexpr (HAS_SE_DOWN) {
          val = val + sh_s[b * ROWS + r];
        } else if constexpr (HAS_SH) {
          val = val + shared[b * NN + c];
        }
        if constexpr (HAS_RES) val = res[b * NN + c] + val;
        out[b * NN + c] = val;
      }
    }
"""

# Generic fallback for inline dedup (B >= 2, dedup=None) or non-exact/non-4bit shapes.
_DC_INLINE_SRC = r"""
    constexpr int NPAIR = BB * 8;
    constexpr int PF = 32 / BITS;
    constexpr int PPT = FAST ? 2 : 1;
    constexpr int VPT = PF * PPT;
    constexpr int BLOCK = VPT * 32;
    constexpr int SSTEP = GS / VPT;
    constexpr int ROWS = 4;
    typedef float U;
    typedef bfloat16_t T;

    uint3 tid = threadgroup_position_in_grid;
    uint sg = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint ttid = sg * 32 + lane;
    const int out_row = tid.y * ROWS;
    const int in_w = KK * BITS / 8;
    const int in_g = KK / GS;

    threadgroup uint ind_s[NPAIR];
    threadgroup uint pl_all[NSG * NPAIR];
    threadgroup bfloat16_t dn_s[NPAIR * 4];
    for (int i = int(ttid); i < NPAIR; i += 32 * NSG) ind_s[i] = inds[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup uint* pl = pl_all + sg * NPAIR;

    for (int P0 = int(sg); P0 < NPAIR; P0 += NSG) {
      const uint e = ind_s[P0];
      int cnt = 0;
      bool dup = false;
      for (int base = 0; base < NPAIR; base += 32) {
        int p = base + int(lane);
        bool match = (p < NPAIR) && (ind_s[p] == e);
        if (match && p < P0) dup = true;
        int take = (match && p >= P0) ? 1 : 0;
        int pre = simd_prefix_exclusive_sum(take);
        if (take) pl[cnt + pre] = (uint)p;
        cnt += simd_sum(take);
      }
      dup = simd_any(dup);
      simdgroup_barrier(mem_flags::mem_threadgroup);
      if (!dup) {
        const device uint8_t* ws0 = (const device uint8_t*)w + (long)e * NN * in_w + (long)out_row * in_w + lane * PPT * 4;
        const device uint8_t* sc0 = scales + (long)e * NN * in_g + (long)out_row * in_g + lane / SSTEP;
        for (int tb = 0; tb < cnt; tb += TMAX) {
          int xo[TMAX], pqs[TMAX];
          for (int t = 0; t < TMAX; t++) {
            int pq = (tb + t < cnt) ? (int)pl[tb + t] : P0;
            pqs[t] = pq;
            xo[t] = pq * KK + int(lane) * VPT;
          }
          thread U result[TMAX][ROWS];
          for (int t = 0; t < TMAX; t++)
            for (int r = 0; r < ROWS; r++) result[t][r] = 0;
          const device uint8_t* ws = ws0;
          const device uint8_t* sc = sc0;
          int k = 0;
          constexpr int KEND_SUB = (EXACT_BLK || FAST) ? 0 : BLOCK;
          for (; k < KK - KEND_SUB; k += BLOCK) {
            for (int r = 0; r < ROWS; r++) {
              const device uint32_t* wp = (const device uint32_t*)(ws + r * in_w);
              thread uint32_t rw[PPT];
              for (int p = 0; p < PPT; p++) rw[p] = wp[p];
              U sv = dequantize_scale<U, GS>((sc + r * in_g)[0]);
              thread U d[VPT];
              dequant_vals_u32<BITS, VPT>(rw, d);
              for (int t = 0; t < TMAX; t++) {
                if (tb + t < cnt) {
                  thread U xt[VPT];
                  load_vector<T, U, VPT>(x + xo[t] + k, xt);
                  result[t][r] += dot_vals<BITS, VPT>(xt, d, sv);
                }
              }
            }
            ws += BLOCK * 4 / PF;
            sc += BLOCK / GS;
          }
          if constexpr (!(EXACT_BLK || FAST)) {
            const int rem = clamp((int)(KK - k - int(lane) * VPT), 0, VPT);
            if (rem > 0) {
              for (int t = 0; t < TMAX; t++) {
                if (tb + t < cnt) {
                  thread U xt[VPT];
                  load_vector_safe<T, U, VPT>(x + xo[t] + k, xt, rem);
                  for (int r = 0; r < ROWS; r++) {
                    result[t][r] += qdot_safe<U, VPT, BITS>(ws + r * in_w, xt, dequantize_scale<U, GS>((sc + r * in_g)[0]), rem);
                  }
                }
              }
            }
          }
          for (int t = 0; t < TMAX; t++) {
            for (int r = 0; r < ROWS; r++) {
              U v = simd_sum(result[t][r]);
              if (lane == 0 && tb + t < cnt) dn_s[pqs[t] * 4 + r] = static_cast<T>(v);
            }
          }
        }
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int idx = int(ttid); idx < BB * 4; idx += 32 * NSG) {
      const int b = idx / 4;
      const int r = idx % 4;
      const int c = out_row + r;
      float ev[8], ww[8];
      for (int j = 0; j < 8; j++) {
        ev[j] = static_cast<float>(dn_s[(b * 8 + j) * 4 + r]);
        ww[j] = wts[b * 8 + j];
      }
      bfloat16_t val = static_cast<bfloat16_t>(glue_moe_combine8(ev, ww));
      if constexpr (HAS_SH) val = val + shared[b * NN + c];
      if constexpr (HAS_RES) val = res[b * NN + c] + val;
      out[b * NN + c] = val;
    }
"""


# ---------------------------------------------------------------------------
# 4. T3 Contiguous K-Major Weight+Scale Repacking (0 Extra Steady-State RAM)
# ---------------------------------------------------------------------------
# gu_packed: [E, N//4, 17408] u8 (K=4096, GS=32, BITS=4, 0B pad, 128B aligned per tile)
# dn_packed: [E, N//4, 2720]  u8 (K=1280, GS=32, BITS=4, 0B pad)

_PACK_GU_SRC = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const int tile = int(threadgroup_position_in_grid.y) * 4 + int(sg);
    if (tile >= NN / 4) return;
    const int e = int(threadgroup_position_in_grid.z);
    const int r0 = tile * 4;
    const long row_w_base = (long)e * NN * 2048 + (long)r0 * 2048;
    const long row_s_base = (long)e * NN * 128 + (long)r0 * 128;
    device uint8_t* dst = out + ((long)e * (NN / 4) + tile) * 17408;

    const device uint8_t* wg_b = (const device uint8_t*)wg + row_w_base;
    const device uint8_t* sg_b = scg + row_s_base;
    const device uint8_t* wu_b = (const device uint8_t*)wu + row_w_base;
    const device uint8_t* su_b = scu + row_s_base;

    for (int kb = 0; kb < 8; ++kb) {
      if (lane < 4) {
        *(device uint4*)(dst + kb * 64 + lane * 16) = *(const device uint4*)(sg_b + lane * 128 + kb * 16);
        *(device uint4*)(dst + 512 + kb * 64 + lane * 16) = *(const device uint4*)(su_b + lane * 128 + kb * 16);
      }
      device uint2* dg = (device uint2*)(dst + 1024 + kb * 2048);
      device uint2* du = (device uint2*)(dst + 1024 + kb * 2048 + 1024);
      for (int r = 0; r < 4; ++r) {
        dg[r * 32 + lane] = *(const device uint2*)(wg_b + r * 2048 + kb * 256 + lane * 8);
        du[r * 32 + lane] = *(const device uint2*)(wu_b + r * 2048 + kb * 256 + lane * 8);
      }
    }
"""

_UNPACK_GU_SRC = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const int tile = int(threadgroup_position_in_grid.y) * 4 + int(sg);
    if (tile >= NN / 4) return;
    const int e = int(threadgroup_position_in_grid.z);
    const int r0 = tile * 4;
    const long row_w_base = (long)e * NN * 2048 + (long)r0 * 2048;
    const long row_s_base = (long)e * NN * 128 + (long)r0 * 128;
    const device uint8_t* src = packed + ((long)e * (NN / 4) + tile) * 17408;

    device uint8_t* wg_b = (device uint8_t*)wg + row_w_base;
    device uint8_t* sg_b = scg + row_s_base;
    device uint8_t* wu_b = (device uint8_t*)wu + row_w_base;
    device uint8_t* su_b = scu + row_s_base;

    for (int kb = 0; kb < 8; ++kb) {
      if (lane < 4) {
        *(device uint4*)(sg_b + lane * 128 + kb * 16) = *(const device uint4*)(src + kb * 64 + lane * 16);
        *(device uint4*)(su_b + lane * 128 + kb * 16) = *(const device uint4*)(src + 512 + kb * 64 + lane * 16);
      }
      const device uint2* sg_w = (const device uint2*)(src + 1024 + kb * 2048);
      const device uint2* su_w = (const device uint2*)(src + 1024 + kb * 2048 + 1024);
      for (int r = 0; r < 4; ++r) {
        *(device uint2*)(wg_b + r * 2048 + kb * 256 + lane * 8) = sg_w[r * 32 + lane];
        *(device uint2*)(wu_b + r * 2048 + kb * 256 + lane * 8) = su_w[r * 32 + lane];
      }
    }
"""

_PACK_DN_SRC = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const int tile = int(threadgroup_position_in_grid.y) * 4 + int(sg);
    if (tile >= NN / 4) return;
    const int e = int(threadgroup_position_in_grid.z);
    const int r0 = tile * 4;
    const long row_w_base = (long)e * NN * 640 + (long)r0 * 640;
    const long row_s_base = (long)e * NN * 40 + (long)r0 * 40;
    device uint8_t* dst = out + ((long)e * (NN / 4) + tile) * 2720;

    const device uint8_t* w_b = (const device uint8_t*)w + row_w_base;
    const device uint8_t* s_b = sc + row_s_base;

    for (int kb = 0; kb < 5; ++kb) {
      if (lane < 4) {
        *(device uint2*)(dst + kb * 32 + lane * 8) = *(const device uint2*)(s_b + lane * 40 + kb * 8);
      }
      device uint32_t* dw = (device uint32_t*)(dst + 160 + kb * 512);
      for (int r = 0; r < 4; ++r) {
        dw[r * 32 + lane] = *(const device uint32_t*)(w_b + r * 640 + kb * 128 + lane * 4);
      }
    }
"""

_UNPACK_DN_SRC = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const int tile = int(threadgroup_position_in_grid.y) * 4 + int(sg);
    if (tile >= NN / 4) return;
    const int e = int(threadgroup_position_in_grid.z);
    const int r0 = tile * 4;
    const long row_w_base = (long)e * NN * 640 + (long)r0 * 640;
    const long row_s_base = (long)e * NN * 40 + (long)r0 * 40;
    const device uint8_t* src = packed + ((long)e * (NN / 4) + tile) * 2720;

    device uint8_t* w_b = (device uint8_t*)w + row_w_base;
    device uint8_t* s_b = sc + row_s_base;

    for (int kb = 0; kb < 5; ++kb) {
      if (lane < 4) {
        *(device uint2*)(s_b + lane * 40 + kb * 8) = *(const device uint2*)(src + kb * 32 + lane * 8);
      }
      const device uint32_t* sw = (const device uint32_t*)(src + 160 + kb * 512);
      for (int r = 0; r < 4; ++r) {
        *(device uint32_t*)(w_b + r * 640 + kb * 128 + lane * 4) = sw[r * 32 + lane];
      }
    }
"""

_GU_PACKED_SRC = r"""
    typedef bfloat16_t T;
    const uint3 tid = threadgroup_position_in_grid;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const int n_tiles = NN / 4;
    const int tile = int(tid.y) * NSG + int(sg);
    const int out_row = tile * 4;
    if (out_row >= NN) return;

    const uint s_lane = lane >> 1;
    const int nu = USE_DEDUP ? n_uniq[0] : (BB * 8);
    const int gz_step = (GZ > 0) ? GZ : (BB * 8);

    for (int q = int(tid.z); q < nu; q += gz_step) {
      uint e;
      int off0 = 0, cnt = 1;
      if constexpr (USE_DEDUP) {
        const uint meta = u_meta[q];
        e = meta & 0x3FFu;
        off0 = int((meta >> 10) & 0xFFu);
        cnt = int((meta >> 18) & 0xFFu);
      } else {
        e = inds[q];
        bool dup = false;
        for (int p = 0; p < q; ++p) {
          if (inds[p] == e) dup = true;
        }
        if (dup) continue;
      }

      const device uint8_t* base = gu_packed + ((long)e * n_tiles + tile) * 17408;
      if (cnt == 1 || TMAX == 1) {
        const int pq0 = USE_DEDUP ? int(u_pqs[off0]) : q;
        const int xo0 = (pq0 / 8) * 4096 + int(lane) * 16;
        float rg[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float ru[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int kb = 0; kb < 8; ++kb) {
          const device uint4* xp = (const device uint4*)(x + xo0 + kb * 512);
          const uint4 xu0 = xp[0], xu1 = xp[1];
          const device uint8_t* sg_k = base + kb * 64 + s_lane;
          const device uint8_t* su_k = base + 512 + kb * 64 + s_lane;
          const device uint2* wg_k = (const device uint2*)(base + 1024 + kb * 2048) + lane;
          const device uint2* wu_k = (const device uint2*)(base + 2048 + kb * 2048) + lane;
#pragma unroll
          for (int r = 0; r < 4; ++r) {
            rg[r] += dequantize_scale<float, 32>(sg_k[r * 16]) * fp4_dot16_u4_mlx(wg_k[r * 32], xu0, xu1);
            ru[r] += dequantize_scale<float, 32>(su_k[r * 16]) * fp4_dot16_u4_mlx(wu_k[r * 32], xu0, xu1);
          }
        }
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          const float vg = simd_sum(rg[r]), vu = simd_sum(ru[r]);
          if (lane == 0) {
            const T swv = glue_swiglu_bf16(static_cast<T>(vg), static_cast<T>(vu));
            if constexpr (USE_DEDUP) {
              out[pq0 * NN + out_row + r] = swv;
            } else {
              for (int p = q; p < 8; ++p) {
                if (inds[p] == e) out[p * NN + out_row + r] = swv;
              }
            }
          }
        }
      } else if constexpr (USE_DEDUP) {
        if (cnt == 2) {
          int xo[2];
          for (int t = 0; t < 2; t++) xo[t] = (int(u_pqs[off0 + t]) / 8) * 4096 + int(lane) * 16;
          if constexpr (SINGLE_KB) {
            // Single sequential pass over 17408 B tile (reads gate_k & up_k together!)
            float rg[2][4] = {{0,0,0,0}, {0,0,0,0}};
            float ru[2][4] = {{0,0,0,0}, {0,0,0,0}};
            for (int kb = 0; kb < 8; ++kb) {
              const device uint8_t* sg_k = base + kb * 64 + s_lane;
              const device uint8_t* su_k = base + 512 + kb * 64 + s_lane;
              const device uint2* wg_k = (const device uint2*)(base + 1024 + kb * 2048) + lane;
              const device uint2* wu_k = (const device uint2*)(base + 2048 + kb * 2048) + lane;
#pragma unroll
              for (int r = 0; r < 4; ++r) {
                {
                  const uint2 rw = wg_k[r * 32];
                  const float sv = dequantize_scale<float, 32>(sg_k[r * 16]);
                  const float4 d0 = fp4_decode4_mlx(uint16_t(rw.x)), d1 = fp4_decode4_mlx(uint16_t(rw.x >> 16));
                  const float4 d2 = fp4_decode4_mlx(uint16_t(rw.y)), d3 = fp4_decode4_mlx(uint16_t(rw.y >> 16));
#pragma unroll
                  for (int t = 0; t < 2; ++t) {
                    const device uint4* xp = (const device uint4*)(x + xo[t] + kb * 512);
                    const uint4 xu0 = xp[0], xu1 = xp[1];
                    const bfloat4 xv0 = as_type<bfloat4>(xu0.xy), xv1 = as_type<bfloat4>(xu0.zw);
                    const bfloat4 xv2 = as_type<bfloat4>(xu1.xy), xv3 = as_type<bfloat4>(xu1.zw);
                    float acc = 0.0f;
                    acc += ((float(xv0.x)*d0.x + float(xv0.y)*d0.y) + float(xv0.z)*d0.z) + float(xv0.w)*d0.w;
                    acc += ((float(xv1.x)*d1.x + float(xv1.y)*d1.y) + float(xv1.z)*d1.z) + float(xv1.w)*d1.w;
                    acc += ((float(xv2.x)*d2.x + float(xv2.y)*d2.y) + float(xv2.z)*d2.z) + float(xv2.w)*d2.w;
                    acc += ((float(xv3.x)*d3.x + float(xv3.y)*d3.y) + float(xv3.z)*d3.z) + float(xv3.w)*d3.w;
                    rg[t][r] += sv * acc;
                  }
                }
                {
                  const uint2 rw = wu_k[r * 32];
                  const float sv = dequantize_scale<float, 32>(su_k[r * 16]);
                  const float4 d0 = fp4_decode4_mlx(uint16_t(rw.x)), d1 = fp4_decode4_mlx(uint16_t(rw.x >> 16));
                  const float4 d2 = fp4_decode4_mlx(uint16_t(rw.y)), d3 = fp4_decode4_mlx(uint16_t(rw.y >> 16));
#pragma unroll
                  for (int t = 0; t < 2; ++t) {
                    const device uint4* xp = (const device uint4*)(x + xo[t] + kb * 512);
                    const uint4 xu0 = xp[0], xu1 = xp[1];
                    const bfloat4 xv0 = as_type<bfloat4>(xu0.xy), xv1 = as_type<bfloat4>(xu0.zw);
                    const bfloat4 xv2 = as_type<bfloat4>(xu1.xy), xv3 = as_type<bfloat4>(xu1.zw);
                    float acc = 0.0f;
                    acc += ((float(xv0.x)*d0.x + float(xv0.y)*d0.y) + float(xv0.z)*d0.z) + float(xv0.w)*d0.w;
                    acc += ((float(xv1.x)*d1.x + float(xv1.y)*d1.y) + float(xv1.z)*d1.z) + float(xv1.w)*d1.w;
                    acc += ((float(xv2.x)*d2.x + float(xv2.y)*d2.y) + float(xv2.z)*d2.z) + float(xv2.w)*d2.w;
                    acc += ((float(xv3.x)*d3.x + float(xv3.y)*d3.y) + float(xv3.z)*d3.z) + float(xv3.w)*d3.w;
                    ru[t][r] += sv * acc;
                  }
                }
              }
            }
#pragma unroll
            for (int t = 0; t < 2; ++t) {
              const int pq = (lane == 0) ? int(u_pqs[off0 + t]) : 0;
#pragma unroll
              for (int r = 0; r < 4; ++r) {
                const float vg = simd_sum(rg[t][r]), vu = simd_sum(ru[t][r]);
                if (lane == 0) out[pq * NN + out_row + r] = glue_swiglu_bf16(static_cast<T>(vg), static_cast<T>(vu));
              }
            }
          } else {
            T g_bf[2][4];
            for (int pass = 0; pass < 2; ++pass) {
              float res[2][4] = {{0,0,0,0}, {0,0,0,0}};
              for (int kb = 0; kb < 8; ++kb) {
                const device uint8_t* sp_k = base + (pass ? 512 : 0) + kb * 64 + s_lane;
                const device uint2* wp_k = (const device uint2*)(base + (pass ? 2048 : 1024) + kb * 2048) + lane;
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                  const uint2 rw = wp_k[r * 32];
                  const float sv = dequantize_scale<float, 32>(sp_k[r * 16]);
                  const float4 d0 = fp4_decode4_mlx(uint16_t(rw.x)), d1 = fp4_decode4_mlx(uint16_t(rw.x >> 16));
                  const float4 d2 = fp4_decode4_mlx(uint16_t(rw.y)), d3 = fp4_decode4_mlx(uint16_t(rw.y >> 16));
#pragma unroll
                  for (int t = 0; t < 2; ++t) {
                    const device uint4* xp = (const device uint4*)(x + xo[t] + kb * 512);
                    const uint4 xu0 = xp[0], xu1 = xp[1];
                    const bfloat4 xv0 = as_type<bfloat4>(xu0.xy), xv1 = as_type<bfloat4>(xu0.zw);
                    const bfloat4 xv2 = as_type<bfloat4>(xu1.xy), xv3 = as_type<bfloat4>(xu1.zw);
                    float acc = 0.0f;
                    acc += ((float(xv0.x)*d0.x + float(xv0.y)*d0.y) + float(xv0.z)*d0.z) + float(xv0.w)*d0.w;
                    acc += ((float(xv1.x)*d1.x + float(xv1.y)*d1.y) + float(xv1.z)*d1.z) + float(xv1.w)*d1.w;
                    acc += ((float(xv2.x)*d2.x + float(xv2.y)*d2.y) + float(xv2.z)*d2.z) + float(xv2.w)*d2.w;
                    acc += ((float(xv3.x)*d3.x + float(xv3.y)*d3.y) + float(xv3.z)*d3.z) + float(xv3.w)*d3.w;
                    res[t][r] += sv * acc;
                  }
                }
              }
#pragma unroll
              for (int t = 0; t < 2; ++t) {
                const int pq = (pass == 1 && lane == 0) ? int(u_pqs[off0 + t]) : 0;
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                  const float v = simd_sum(res[t][r]);
                  if (lane == 0) {
                    if (pass == 0) g_bf[t][r] = static_cast<T>(v);
                    else out[pq * NN + out_row + r] = glue_swiglu_bf16(g_bf[t][r], static_cast<T>(v));
                  }
                }
              }
            }
          }
        } else if (cnt <= 4 || TMAX <= 4) {
          for (int r0 = 0; r0 < 4; r0 += 2) {
            for (int tb = 0; tb < cnt; tb += 4) {
              const int n_tok = min(4, cnt - tb);
              int xo[4];
              for (int t = 0; t < 4; t++) {
                const int pq = (t < n_tok) ? int(u_pqs[off0 + tb + t]) : int(u_pqs[off0]);
                xo[t] = (pq / 8) * 4096 + int(lane) * 16;
              }
              if constexpr (SINGLE_KB) {
                float rg[4][2] = {{0,0},{0,0},{0,0},{0,0}};
                float ru[4][2] = {{0,0},{0,0},{0,0},{0,0}};
                for (int kb = 0; kb < 8; ++kb) {
                  const device uint8_t* sg_k = base + kb * 64 + s_lane;
                  const device uint8_t* su_k = base + 512 + kb * 64 + s_lane;
                  const device uint2* wg_k = (const device uint2*)(base + 1024 + kb * 2048) + lane;
                  const device uint2* wu_k = (const device uint2*)(base + 2048 + kb * 2048) + lane;
#pragma unroll
                  for (int r = 0; r < 2; ++r) {
                    const int rr = r0 + r;
                    {
                      const uint2 rw = wg_k[rr * 32];
                      const float sv = dequantize_scale<float, 32>(sg_k[rr * 16]);
                      const float4 d0 = fp4_decode4_mlx(uint16_t(rw.x)), d1 = fp4_decode4_mlx(uint16_t(rw.x >> 16));
                      const float4 d2 = fp4_decode4_mlx(uint16_t(rw.y)), d3 = fp4_decode4_mlx(uint16_t(rw.y >> 16));
#pragma unroll
                      for (int t = 0; t < 4; ++t) {
                        if (t < n_tok) {
                          const device uint4* xp = (const device uint4*)(x + xo[t] + kb * 512);
                          const uint4 xu0 = xp[0], xu1 = xp[1];
                          const bfloat4 xv0 = as_type<bfloat4>(xu0.xy), xv1 = as_type<bfloat4>(xu0.zw);
                          const bfloat4 xv2 = as_type<bfloat4>(xu1.xy), xv3 = as_type<bfloat4>(xu1.zw);
                          float acc = 0.0f;
                          acc += ((float(xv0.x)*d0.x + float(xv0.y)*d0.y) + float(xv0.z)*d0.z) + float(xv0.w)*d0.w;
                          acc += ((float(xv1.x)*d1.x + float(xv1.y)*d1.y) + float(xv1.z)*d1.z) + float(xv1.w)*d1.w;
                          acc += ((float(xv2.x)*d2.x + float(xv2.y)*d2.y) + float(xv2.z)*d2.z) + float(xv2.w)*d2.w;
                          acc += ((float(xv3.x)*d3.x + float(xv3.y)*d3.y) + float(xv3.z)*d3.z) + float(xv3.w)*d3.w;
                          rg[t][r] += sv * acc;
                        }
                      }
                    }
                    {
                      const uint2 rw = wu_k[rr * 32];
                      const float sv = dequantize_scale<float, 32>(su_k[rr * 16]);
                      const float4 d0 = fp4_decode4_mlx(uint16_t(rw.x)), d1 = fp4_decode4_mlx(uint16_t(rw.x >> 16));
                      const float4 d2 = fp4_decode4_mlx(uint16_t(rw.y)), d3 = fp4_decode4_mlx(uint16_t(rw.y >> 16));
#pragma unroll
                      for (int t = 0; t < 4; ++t) {
                        if (t < n_tok) {
                          const device uint4* xp = (const device uint4*)(x + xo[t] + kb * 512);
                          const uint4 xu0 = xp[0], xu1 = xp[1];
                          const bfloat4 xv0 = as_type<bfloat4>(xu0.xy), xv1 = as_type<bfloat4>(xu0.zw);
                          const bfloat4 xv2 = as_type<bfloat4>(xu1.xy), xv3 = as_type<bfloat4>(xu1.zw);
                          float acc = 0.0f;
                          acc += ((float(xv0.x)*d0.x + float(xv0.y)*d0.y) + float(xv0.z)*d0.z) + float(xv0.w)*d0.w;
                          acc += ((float(xv1.x)*d1.x + float(xv1.y)*d1.y) + float(xv1.z)*d1.z) + float(xv1.w)*d1.w;
                          acc += ((float(xv2.x)*d2.x + float(xv2.y)*d2.y) + float(xv2.z)*d2.z) + float(xv2.w)*d2.w;
                          acc += ((float(xv3.x)*d3.x + float(xv3.y)*d3.y) + float(xv3.z)*d3.z) + float(xv3.w)*d3.w;
                          ru[t][r] += sv * acc;
                        }
                      }
                    }
                  }
                }
#pragma unroll
                for (int t = 0; t < 4; ++t) {
                  const int pq = (lane == 0 && t < n_tok) ? int(u_pqs[off0 + tb + t]) : 0;
#pragma unroll
                  for (int r = 0; r < 2; ++r) {
                    const float vg = simd_sum(rg[t][r]), vu = simd_sum(ru[t][r]);
                    if (lane == 0 && t < n_tok) out[pq * NN + out_row + r0 + r] = glue_swiglu_bf16(static_cast<T>(vg), static_cast<T>(vu));
                  }
                }
              } else {
                T g_bf[4][2];
                for (int pass = 0; pass < 2; ++pass) {
                  float res[4][2] = {{0,0},{0,0},{0,0},{0,0}};
                  for (int kb = 0; kb < 8; ++kb) {
                    const device uint8_t* sp_k = base + (pass ? 512 : 0) + kb * 64 + s_lane;
                    const device uint2* wp_k = (const device uint2*)(base + (pass ? 2048 : 1024) + kb * 2048) + lane;
#pragma unroll
                    for (int r = 0; r < 2; ++r) {
                      const int rr = r0 + r;
                      const uint2 rw = wp_k[rr * 32];
                      const float sv = dequantize_scale<float, 32>(sp_k[rr * 16]);
                      const float4 d0 = fp4_decode4_mlx(uint16_t(rw.x)), d1 = fp4_decode4_mlx(uint16_t(rw.x >> 16));
                      const float4 d2 = fp4_decode4_mlx(uint16_t(rw.y)), d3 = fp4_decode4_mlx(uint16_t(rw.y >> 16));
#pragma unroll
                      for (int t = 0; t < 4; ++t) {
                        if (t < n_tok) {
                          const device uint4* xp = (const device uint4*)(x + xo[t] + kb * 512);
                          const uint4 xu0 = xp[0], xu1 = xp[1];
                          const bfloat4 xv0 = as_type<bfloat4>(xu0.xy), xv1 = as_type<bfloat4>(xu0.zw);
                          const bfloat4 xv2 = as_type<bfloat4>(xu1.xy), xv3 = as_type<bfloat4>(xu1.zw);
                          float acc = 0.0f;
                          acc += ((float(xv0.x)*d0.x + float(xv0.y)*d0.y) + float(xv0.z)*d0.z) + float(xv0.w)*d0.w;
                          acc += ((float(xv1.x)*d1.x + float(xv1.y)*d1.y) + float(xv1.z)*d1.z) + float(xv1.w)*d1.w;
                          acc += ((float(xv2.x)*d2.x + float(xv2.y)*d2.y) + float(xv2.z)*d2.z) + float(xv2.w)*d2.w;
                          acc += ((float(xv3.x)*d3.x + float(xv3.y)*d3.y) + float(xv3.z)*d3.z) + float(xv3.w)*d3.w;
                          res[t][r] += sv * acc;
                        }
                      }
                    }
                  }
#pragma unroll
                  for (int t = 0; t < 4; ++t) {
                    const int pq = (pass == 1 && lane == 0 && t < n_tok) ? int(u_pqs[off0 + tb + t]) : 0;
#pragma unroll
                    for (int r = 0; r < 2; ++r) {
                      const float v = simd_sum(res[t][r]);
                      if (lane == 0 && t < n_tok) {
                        if (pass == 0) g_bf[t][r] = static_cast<T>(v);
                        else out[pq * NN + out_row + r0 + r] = glue_swiglu_bf16(g_bf[t][r], static_cast<T>(v));
                      }
                    }
                  }
                }
              }
            }
          }
        } else {
          for (int r0 = 0; r0 < 4; ++r0) {
            for (int tb = 0; tb < cnt; tb += 8) {
              const int n_tok = min(8, cnt - tb);
              int xo[8];
              for (int t = 0; t < 8; t++) {
                const int pq = (t < n_tok) ? int(u_pqs[off0 + tb + t]) : int(u_pqs[off0]);
                xo[t] = (pq / 8) * 4096 + int(lane) * 16;
              }
              if constexpr (SINGLE_KB) {
                float rg[8] = {0,0,0,0,0,0,0,0};
                float ru[8] = {0,0,0,0,0,0,0,0};
                for (int kb = 0; kb < 8; ++kb) {
                  {
                    const uint2 rw = ((const device uint2*)(base + 1024 + kb * 2048))[r0 * 32 + lane];
                    const float sv = dequantize_scale<float, 32>(base[kb * 64 + r0 * 16 + s_lane]);
                    const float4 d0 = fp4_decode4_mlx(uint16_t(rw.x)), d1 = fp4_decode4_mlx(uint16_t(rw.x >> 16));
                    const float4 d2 = fp4_decode4_mlx(uint16_t(rw.y)), d3 = fp4_decode4_mlx(uint16_t(rw.y >> 16));
#pragma unroll
                    for (int t = 0; t < 8; ++t) {
                      if (t < n_tok) {
                        const device uint4* xp = (const device uint4*)(x + xo[t] + kb * 512);
                        const uint4 xu0 = xp[0], xu1 = xp[1];
                        const bfloat4 xv0 = as_type<bfloat4>(xu0.xy), xv1 = as_type<bfloat4>(xu0.zw);
                        const bfloat4 xv2 = as_type<bfloat4>(xu1.xy), xv3 = as_type<bfloat4>(xu1.zw);
                        float acc = 0.0f;
                        acc += ((float(xv0.x)*d0.x + float(xv0.y)*d0.y) + float(xv0.z)*d0.z) + float(xv0.w)*d0.w;
                        acc += ((float(xv1.x)*d1.x + float(xv1.y)*d1.y) + float(xv1.z)*d1.z) + float(xv1.w)*d1.w;
                        acc += ((float(xv2.x)*d2.x + float(xv2.y)*d2.y) + float(xv2.z)*d2.z) + float(xv2.w)*d2.w;
                        acc += ((float(xv3.x)*d3.x + float(xv3.y)*d3.y) + float(xv3.z)*d3.z) + float(xv3.w)*d3.w;
                        rg[t] += sv * acc;
                      }
                    }
                  }
                  {
                    const uint2 rw = ((const device uint2*)(base + 2048 + kb * 2048))[r0 * 32 + lane];
                    const float sv = dequantize_scale<float, 32>(base[512 + kb * 64 + r0 * 16 + s_lane]);
                    const float4 d0 = fp4_decode4_mlx(uint16_t(rw.x)), d1 = fp4_decode4_mlx(uint16_t(rw.x >> 16));
                    const float4 d2 = fp4_decode4_mlx(uint16_t(rw.y)), d3 = fp4_decode4_mlx(uint16_t(rw.y >> 16));
#pragma unroll
                    for (int t = 0; t < 8; ++t) {
                      if (t < n_tok) {
                        const device uint4* xp = (const device uint4*)(x + xo[t] + kb * 512);
                        const uint4 xu0 = xp[0], xu1 = xp[1];
                        const bfloat4 xv0 = as_type<bfloat4>(xu0.xy), xv1 = as_type<bfloat4>(xu0.zw);
                        const bfloat4 xv2 = as_type<bfloat4>(xu1.xy), xv3 = as_type<bfloat4>(xu1.zw);
                        float acc = 0.0f;
                        acc += ((float(xv0.x)*d0.x + float(xv0.y)*d0.y) + float(xv0.z)*d0.z) + float(xv0.w)*d0.w;
                        acc += ((float(xv1.x)*d1.x + float(xv1.y)*d1.y) + float(xv1.z)*d1.z) + float(xv1.w)*d1.w;
                        acc += ((float(xv2.x)*d2.x + float(xv2.y)*d2.y) + float(xv2.z)*d2.z) + float(xv2.w)*d2.w;
                        acc += ((float(xv3.x)*d3.x + float(xv3.y)*d3.y) + float(xv3.z)*d3.z) + float(xv3.w)*d3.w;
                        ru[t] += sv * acc;
                      }
                    }
                  }
                }
#pragma unroll
                for (int t = 0; t < 8; ++t) {
                  const int pq = (lane == 0 && t < n_tok) ? int(u_pqs[off0 + tb + t]) : 0;
                  const float vg = simd_sum(rg[t]), vu = simd_sum(ru[t]);
                  if (lane == 0 && t < n_tok) out[pq * NN + out_row + r0] = glue_swiglu_bf16(static_cast<T>(vg), static_cast<T>(vu));
                }
              } else {
                T g_bf[8];
                for (int pass = 0; pass < 2; ++pass) {
                  float res[8] = {0,0,0,0,0,0,0,0};
                  for (int kb = 0; kb < 8; ++kb) {
                    const uint2 rw = ((const device uint2*)(base + (pass ? 2048 : 1024) + kb * 2048))[r0 * 32 + lane];
                    const float sv = dequantize_scale<float, 32>(base[(pass ? 512 : 0) + kb * 64 + r0 * 16 + s_lane]);
                    const float4 d0 = fp4_decode4_mlx(uint16_t(rw.x)), d1 = fp4_decode4_mlx(uint16_t(rw.x >> 16));
                    const float4 d2 = fp4_decode4_mlx(uint16_t(rw.y)), d3 = fp4_decode4_mlx(uint16_t(rw.y >> 16));
#pragma unroll
                    for (int t = 0; t < 8; ++t) {
                      if (t < n_tok) {
                        const device uint4* xp = (const device uint4*)(x + xo[t] + kb * 512);
                        const uint4 xu0 = xp[0], xu1 = xp[1];
                        const bfloat4 xv0 = as_type<bfloat4>(xu0.xy), xv1 = as_type<bfloat4>(xu0.zw);
                        const bfloat4 xv2 = as_type<bfloat4>(xu1.xy), xv3 = as_type<bfloat4>(xu1.zw);
                        float acc = 0.0f;
                        acc += ((float(xv0.x)*d0.x + float(xv0.y)*d0.y) + float(xv0.z)*d0.z) + float(xv0.w)*d0.w;
                        acc += ((float(xv1.x)*d1.x + float(xv1.y)*d1.y) + float(xv1.z)*d1.z) + float(xv1.w)*d1.w;
                        acc += ((float(xv2.x)*d2.x + float(xv2.y)*d2.y) + float(xv2.z)*d2.z) + float(xv2.w)*d2.w;
                        acc += ((float(xv3.x)*d3.x + float(xv3.y)*d3.y) + float(xv3.z)*d3.z) + float(xv3.w)*d3.w;
                        res[t] += sv * acc;
                      }
                    }
                  }
#pragma unroll
                  for (int t = 0; t < 8; ++t) {
                    const int pq = (pass == 1 && lane == 0 && t < n_tok) ? int(u_pqs[off0 + tb + t]) : 0;
                    const float v = simd_sum(res[t]);
                    if (lane == 0 && t < n_tok) {
                      if (pass == 0) g_bf[t] = static_cast<T>(v);
                      else out[pq * NN + out_row + r0] = glue_swiglu_bf16(g_bf[t], static_cast<T>(v));
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
"""

_DC_PACKED_SRC = r"""
    constexpr int NPAIR = BB * 8;
    typedef bfloat16_t T;
    const uint3 tid = threadgroup_position_in_grid;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint ttid = sg * 32 + lane;
    const int n_tiles4 = NN / 4;
    const uint s_lane = lane >> 2;
    constexpr int NT4 = ROWS / 4;

    threadgroup bfloat16_t sh_s[HAS_SE_DOWN ? (BB * ROWS) : 1];
    threadgroup bfloat16_t dn_s[NPAIR * ROWS];
    const int nu = USE_DEDUP ? n_uniq[0] : NPAIR;
    const int n_tiles = NN / ROWS;
    const int ntg_safe = (NTG_Y > 0) ? NTG_Y : 1;
    const int t_beg = (NTG_Y > 0) ? (int(tid.y) * n_tiles) / ntg_safe : int(tid.y);
    const int t_end = (NTG_Y > 0) ? ((int(tid.y) + 1) * n_tiles) / ntg_safe : (int(tid.y) + 1);

    for (int tile = t_beg; tile < t_end; ++tile) {
      const int out_row = tile * ROWS;
      const int tile0 = out_row / 4;
      if (tile > t_beg) threadgroup_barrier(mem_flags::mem_threadgroup);

      if constexpr (HAS_SE_DOWN) {
        for (int r = int(sg); r < ROWS; r += NSG) {
          moe_se_down_fp8_r1<BB, 1280, ROWS>(se_w, se_sc, se_mid, out_row, r, lane, sh_s);
        }
      }

      for (int u = int(sg); u < nu; u += NSG) {
        uint e;
        int off0 = 0, cnt = 1;
        if constexpr (USE_DEDUP) {
          const uint meta = u_meta[u];
          e = meta & 0x3FFu;
          off0 = int((meta >> 10) & 0xFFu);
          cnt = int((meta >> 18) & 0xFFu);
        } else {
          e = inds[u];
          bool dup = false;
          for (int p = 0; p < u; ++p) {
            if (inds[p] == e) dup = true;
          }
          if (dup) continue;
        }

        const device uint8_t* base0 = dn_packed + ((long)e * n_tiles4 + tile0) * 2720;
        if constexpr (!USE_DEDUP) {
          const int xo0 = u * 1280 + int(lane) * 8;
          float res[ROWS];
#pragma unroll
          for (int r = 0; r < ROWS; r++) res[r] = 0.0f;
          for (int kb = 0; kb < 5; ++kb) {
            const uint4 xu = *(const device uint4*)(x + xo0 + kb * 256);
#pragma unroll
            for (int st = 0; st < NT4; ++st) {
              const device uint8_t* base = base0 + st * 2720;
              const device uint8_t* sc_k = base + kb * 32 + s_lane;
              const device uint32_t* w_k = (const device uint32_t*)(base + 160 + kb * 512) + lane;
#pragma unroll
              for (int r = 0; r < 4; r++) {
                res[st * 4 + r] += dequantize_scale<float, 32>(sc_k[r * 8]) * fp4_dot8_u4_mlx(w_k[r * 32], xu);
              }
            }
          }
#pragma unroll
          for (int r = 0; r < ROWS; r++) {
            const float v = simd_sum(res[r]);
            if (lane == 0) {
              const T bv = static_cast<T>(v);
              for (int p = u; p < 8; ++p) {
                if (inds[p] == e) dn_s[p * ROWS + r] = bv;
              }
            }
          }
        } else {
          for (int tb = 0; tb < cnt; tb += TMAX) {
            const int n_tok = min(TMAX, cnt - tb);
            int xo[TMAX];
#pragma unroll
            for (int t = 0; t < TMAX; t++) {
              const int pq = (t < n_tok) ? int(u_pqs[off0 + tb + t]) : int(u_pqs[off0]);
              xo[t] = pq * 1280 + int(lane) * 8;
            }
            float res[TMAX][ROWS];
#pragma unroll
            for (int t = 0; t < TMAX; t++)
#pragma unroll
              for (int r = 0; r < ROWS; r++) res[t][r] = 0.0f;
            for (int kb = 0; kb < 5; ++kb) {
              uint4 xu[TMAX];
#pragma unroll
              for (int t = 0; t < TMAX; t++) {
                if (t < n_tok) xu[t] = *(const device uint4*)(x + xo[t] + kb * 256);
              }
#pragma unroll
              for (int st = 0; st < NT4; ++st) {
                const device uint8_t* base = base0 + st * 2720;
                const device uint8_t* sc_k = base + kb * 32 + s_lane;
                const device uint32_t* w_k = (const device uint32_t*)(base + 160 + kb * 512) + lane;
#pragma unroll
                for (int r = 0; r < 4; r++) {
                  const uint32_t rw = w_k[r * 32];
                  const float sv = dequantize_scale<float, 32>(sc_k[r * 8]);
                  const float4 d0 = fp4_decode4_mlx(uint16_t(rw));
                  const float4 d1 = fp4_decode4_mlx(uint16_t(rw >> 16));
#pragma unroll
                  for (int t = 0; t < TMAX; t++) {
                    if (t < n_tok) {
                      const bfloat4 xv0 = as_type<bfloat4>(xu[t].xy);
                      const bfloat4 xv1 = as_type<bfloat4>(xu[t].zw);
                      float acc = 0.0f;
                      acc += ((float(xv0.x) * d0.x + float(xv0.y) * d0.y) + float(xv0.z) * d0.z) + float(xv0.w) * d0.w;
                      acc += ((float(xv1.x) * d1.x + float(xv1.y) * d1.y) + float(xv1.z) * d1.z) + float(xv1.w) * d1.w;
                      res[t][st * 4 + r] += sv * acc;
                    }
                  }
                }
              }
            }
#pragma unroll
            for (int t = 0; t < TMAX; t++) {
              const int pq = (lane == 0 && t < n_tok) ? int(u_pqs[off0 + tb + t]) : 0;
#pragma unroll
              for (int r = 0; r < ROWS; r++) {
                const float v = simd_sum(res[t][r]);
                if (lane == 0 && t < n_tok) dn_s[pq * ROWS + r] = static_cast<T>(v);
              }
            }
          }
        }
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);

      for (int idx = int(ttid); idx < BB * ROWS; idx += 32 * NSG) {
        const int b = idx / ROWS;
        const int r = idx % ROWS;
        const int c = out_row + r;
        float ev[8], ww[8];
#pragma unroll
        for (int j = 0; j < 8; j++) {
          ev[j] = static_cast<float>(dn_s[(b * 8 + j) * ROWS + r]);
          ww[j] = wts[b * 8 + j];
        }
        bfloat16_t val = static_cast<bfloat16_t>(glue_moe_combine8(ev, ww));
        if constexpr (HAS_SE_DOWN) {
          val = val + sh_s[b * ROWS + r];
        } else if constexpr (HAS_SH) {
          val = val + shared[b * NN + c];
        }
        if constexpr (HAS_RES) val = res[b * NN + c] + val;
        out[b * NN + c] = val;
      }
    }
"""


# ---------------------------------------------------------------------------
# Kernel Builders & Public Python Wrappers
# ---------------------------------------------------------------------------

def _gu_fast_kernel():
    k = _KERNELS.get("gu_fast")
    if k is None:
        k = _KERNELS["gu_fast"] = mx.fast.metal_kernel(
            name="bv3_moe_gu_fast",
            input_names=["wg", "scg", "wu", "scu", "x", "inds", "u_meta", "u_pqs", "n_uniq"],
            output_names=["out"],
            header=HEADER,
            source=_GU_FAST_SRC,
            ensure_row_contiguous=True,
        )
    return k


def _gu_inline_kernel():
    k = _KERNELS.get("gu_inline")
    if k is None:
        k = _KERNELS["gu_inline"] = mx.fast.metal_kernel(
            name="bv3_moe_gu_inline",
            input_names=["wg", "scg", "wu", "scu", "x", "inds"],
            output_names=["out"],
            header=HEADER,
            source=_GU_INLINE_SRC,
            ensure_row_contiguous=True,
        )
    return k


def _dc_fast_kernel():
    k = _KERNELS.get("dc_fast")
    if k is None:
        k = _KERNELS["dc_fast"] = mx.fast.metal_kernel(
            name="bv3_moe_dc_fast",
            input_names=["w", "scales", "x", "inds", "u_meta", "u_pqs", "n_uniq", "wts", "shared", "res", "se_mid", "se_w", "se_sc"],
            output_names=["out"],
            header=HEADER,
            source=_DC_FAST_SRC,
            ensure_row_contiguous=True,
        )
    return k


def _dc_inline_kernel():
    k = _KERNELS.get("dc_inline")
    if k is None:
        k = _KERNELS["dc_inline"] = mx.fast.metal_kernel(
            name="bv3_moe_dc_inline",
            input_names=["w", "scales", "x", "inds", "wts", "shared", "res"],
            output_names=["out"],
            header=HEADER,
            source=_DC_INLINE_SRC,
            ensure_row_contiguous=True,
        )
    return k


_COMB8_SRC = r"""
    const uint c = thread_position_in_grid.x;
    const uint b = thread_position_in_grid.y;
    if (int(c) >= NN) return;
    float ev[8], ww[8];
#pragma unroll
    for (int j = 0; j < 8; j++) {
      ev[j] = static_cast<float>(dn[(size_t(b) * 8 + j) * NN + c]);
      ww[j] = wts[size_t(b) * 8 + j];
    }
    bfloat16_t val = static_cast<bfloat16_t>(glue_moe_combine8(ev, ww));
    if constexpr (HAS_SH) val = val + shared[size_t(b) * NN + c];
    if constexpr (HAS_RES) val = res[size_t(b) * NN + c] + val;
    out[size_t(b) * NN + c] = val;
"""


def _comb8_kernel():
    k = _KERNELS.get("comb8")
    if k is None:
        k = _KERNELS["comb8"] = mx.fast.metal_kernel(
            name="bv3_moe_combine8_sh_res",
            input_names=["dn", "wts", "shared", "res"],
            output_names=["out"],
            header=GLUE_HEADER,
            source=_COMB8_SRC,
            ensure_row_contiguous=True,
        )
    return k


def moe_combine8_sh_res_rows(dn, wts, *, shared=None, residual=None):
    B, E, N = dn.shape
    assert E == 8 and dn.dtype == mx.bfloat16
    has_sh = shared is not None
    has_res = residual is not None
    sh = shared if has_sh else _ZERO_BF16
    rs = residual if has_res else _ZERO_BF16
    (out,) = _comb8_kernel()(
        inputs=[dn, wts, sh, rs],
        template=[("NN", N), ("HAS_SH", int(has_sh)), ("HAS_RES", int(has_res))],
        grid=(N, B, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(B, N)],
        output_dtypes=[dn.dtype],
    )
    return out


def moe_gate_up_swiglu_rows(
    x,
    wg,
    sg,
    wu,
    su,
    inds,
    *,
    group_size=32,
    bits=4,
    tmax=None,
    nsg=None,
    rows=None,
    hoist_dq=None,
    dedup=None,
    adaptive=None,
    gz=None,
):
    """Fused bit-exact gate_proj + up_proj + SwiGLU -> (B, 8, N) bfloat16 (0 bit diffs)."""
    del hoist_dq
    B, E = inds.shape
    K = x.shape[-1]
    N = wg.shape[1]
    assert 1 <= B <= 24 and E == 8 and x.shape == (B, K) and x.dtype == mx.bfloat16
    fast = (N % 8 == 0) and (K % ((32 // bits) * 2 * 32) == 0)
    rw = (2 if (B == 1 and fast and bits == 4) else 4) if rows is None else int(rows)
    assert rw in (1, 2, 4) and N % rw == 0

    use_dedup = dedup is not None
    if fast and bits == 4 and (use_dedup or B == 1):
        ad = (1 if adaptive is None else int(bool(adaptive))) if (use_dedup and rw == 4) else 0
        if tmax is None:
            tm = 1 if B == 1 else (4 if B <= 4 else (8 if ad else 2))
        else:
            tm = int(tmax)
        if nsg is None:
            # Fine-grained NSG=2 (160 TGs/exp at ROWS=4, 320 at B=1 ROWS=2) minimizes dependent-layer tail waves.
            nsg = 2
        gz_val = 0 if gz is None else int(gz)
        inds_u32 = inds if inds.dtype == mx.uint32 else inds.astype(mx.uint32)
        if use_dedup:
            u_meta, u_pqs, _, _, n_uniq = dedup
        else:
            u_meta, u_pqs, n_uniq = _ZERO_U32, _ZERO_U32, _ZERO_I32
        gz_grid = gz_val if gz_val > 0 else (B * 8)
        (out,) = _gu_fast_kernel()(
            inputs=[wg, sg, wu, su, x, inds_u32, u_meta, u_pqs, n_uniq],
            template=[
                ("BB", B), ("GS", group_size), ("USE_DEDUP", int(use_dedup)), ("ADAPTIVE", ad),
                ("ROWS", rw), ("TMAX", tm), ("NSG", int(nsg)), ("GZ", gz_val), ("KK", K), ("NN", N),
            ],
            grid=(32, int(nsg) * ((N + rw * int(nsg) - 1) // (rw * int(nsg))), gz_grid),
            threadgroup=(32, int(nsg), 1),
            output_shapes=[(B, E, N)],
            output_dtypes=[mx.bfloat16],
        )
        return out

    tm = (2 if B >= 5 else _tmax(B, _GU_TMAX_CAP)) if tmax is None else int(tmax)
    nsg_v = (4 if B >= 5 else 2) if nsg is None else int(nsg)
    inds_u32 = inds if inds.dtype == mx.uint32 else inds.astype(mx.uint32)
    (out,) = _gu_inline_kernel()(
        inputs=[wg, sg, wu, su, x, inds_u32],
        template=[("BB", B), ("GS", group_size), ("BITS", bits), ("FAST", fast), ("ROWS", rw), ("TMAX", tm), ("NSG", nsg_v), ("KK", K), ("NN", N)],
        grid=(32, nsg_v * ((N + rw * nsg_v - 1) // (rw * nsg_v)), B * E),
        threadgroup=(32, nsg_v, 1),
        output_shapes=[(B, E, N)],
        output_dtypes=[mx.bfloat16],
    )
    return out


def moe_down_combine_rows(
    h,
    wd,
    sd,
    inds,
    wts,
    *,
    shared=None,
    residual=None,
    group_size=32,
    bits=4,
    nsg=None,
    tmax=None,
    dedup=None,
    hoist_dq=None,
    rows=None,
    ntg_y=None,
    se_mid=None,
    se_w_fp8=None,
    se_sc=None,
):
    """Fused bit-exact down_proj + combine8 + optional shared/se_down (T7) + residual -> (B, N) bf16."""
    del hoist_dq
    B, E, K = h.shape
    N = wd.shape[1]
    assert 1 <= B <= 24 and E == 8 and h.dtype == mx.bfloat16
    pf = 32 // bits
    fast = (N % 8 == 0) and (K % (pf * 2 * 32) == 0)
    exact_blk = (K % (pf * 32) == 0)
    has_se = se_mid is not None
    if has_se:
        assert se_w_fp8 is not None and se_sc is not None and se_mid.shape == (B, K) and K % 128 == 0
    has_sh = (shared is not None) and not has_se
    has_res = residual is not None
    sh = shared if has_sh else _ZERO_BF16
    rs = residual if has_res else _ZERO_BF16
    sm = se_mid if has_se else _ZERO_BF16
    sw8 = se_w_fp8 if has_se else _ZERO_U8
    sc8 = se_sc if has_se else _ZERO_F32

    use_dedup = dedup is not None
    if exact_blk and bits == 4 and N % 8 == 0 and (use_dedup or B == 1):
        # ROWS=4 + NSG=8 (1024 TGs, 8 SGs/TG splitting unique experts) minimizes dependent-layer tail latency (-1.0..-4.4 ms e2e).
        rw = 4 if rows is None else int(rows)
        assert rw in (4, 8) and N % rw == 0
        nsg_v = (2 if (2 <= B <= 4 and not has_se) else 8) if nsg is None else int(nsg)
        tm = (1 if B == 1 else 2) if tmax is None else int(tmax)
        ny_tpl = 0 if ntg_y is None else int(ntg_y)
        ny_grid = ny_tpl if ny_tpl > 0 else (N // rw)
        inds_u32 = inds if inds.dtype == mx.uint32 else inds.astype(mx.uint32)
        if use_dedup:
            u_meta, u_pqs, _, _, n_uniq = dedup
        else:
            u_meta, u_pqs, n_uniq = _ZERO_U32, _ZERO_U32, _ZERO_I32
        (out,) = _dc_fast_kernel()(
            inputs=[wd, sd, h, inds_u32, u_meta, u_pqs, n_uniq, wts, sh, rs, sm, sw8, sc8],
            template=[
                ("BB", B), ("GS", group_size), ("USE_DEDUP", int(use_dedup)),
                ("ROWS", rw), ("TMAX", tm), ("NSG", nsg_v), ("NTG_Y", ny_tpl),
                ("KK", K), ("NN", N), ("HAS_SH", int(has_sh)), ("HAS_RES", int(has_res)),
                ("HAS_SE_DOWN", int(has_se)),
            ],
            grid=(32, nsg_v * ny_grid, 1),
            threadgroup=(32, nsg_v, 1),
            output_shapes=[(B, N)],
            output_dtypes=[mx.bfloat16],
        )
        return out

    assert not has_se, "se_down fusion requires exact_blk 4-bit down_proj with dedup or B==1"
    nsg_v = (4 if (B <= 2 or B >= 5) else 2) if nsg is None else int(nsg)
    tm = (4 if B >= 21 else (2 if B >= 8 else _tmax(B, _DC_TMAX_CAP))) if tmax is None else int(tmax)
    inds_u32 = inds if inds.dtype == mx.uint32 else inds.astype(mx.uint32)
    (out,) = _dc_inline_kernel()(
        inputs=[wd, sd, h, inds_u32, wts, sh, rs],
        template=[
            ("BB", B), ("GS", group_size), ("BITS", bits), ("FAST", fast), ("EXACT_BLK", exact_blk),
            ("TMAX", tm), ("NSG", nsg_v), ("KK", K), ("NN", N), ("HAS_SH", int(has_sh)), ("HAS_RES", int(has_res)),
        ],
        grid=(32, nsg_v * (N // 4), 1),
        threadgroup=(32, nsg_v, 1),
        output_shapes=[(B, N)],
        output_dtypes=[mx.bfloat16],
    )
    return out


# ---------------------------------------------------------------------------
# 5. T3 Runtime-on-Load Repacking Helpers & Packed Kernels
# ---------------------------------------------------------------------------

def pack_moe_gu_weights(wg: mx.array, scg: mx.array, wu: mx.array, scu: mx.array) -> mx.array:
    """Lossless runtime repack of MXFP4 K=4096 (wg, scg, wu, scu) -> (E, N//4, 17408) u8 (0B pad)."""
    E, N, K8 = wg.shape
    assert K8 == 512 and N % 4 == 0 and scg.shape == (E, N, 128) and wu.shape == wg.shape and scu.shape == scg.shape
    k = _KERNELS.get("pack_gu")
    if k is None:
        k = _KERNELS["pack_gu"] = mx.fast.metal_kernel(
            name="bv3_moe_pack_gu", input_names=["wg", "scg", "wu", "scu"], output_names=["out"],
            header=GLUE_HEADER, source=_PACK_GU_SRC, ensure_row_contiguous=True,
        )
    nt = N // 4
    (out,) = k(
        inputs=[wg, scg, wu, scu], template=[("NN", N)],
        grid=(32, 4 * ((nt + 3) // 4), E), threadgroup=(32, 4, 1),
        output_shapes=[(E, nt, 17408)], output_dtypes=[mx.uint8],
    )
    return out


def unpack_moe_gu_weights(gu_packed: mx.array) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Lossless inverse of `pack_moe_gu_weights` for SwitchGLU prefill fallback."""
    E, NT, BPT = gu_packed.shape
    assert BPT == 17408
    N = NT * 4
    k = _KERNELS.get("unpack_gu")
    if k is None:
        k = _KERNELS["unpack_gu"] = mx.fast.metal_kernel(
            name="bv3_moe_unpack_gu", input_names=["packed"], output_names=["wg", "scg", "wu", "scu"],
            header=GLUE_HEADER, source=_UNPACK_GU_SRC, ensure_row_contiguous=True,
        )
    return tuple(
        k(
            inputs=[gu_packed], template=[("NN", N)],
            grid=(32, 4 * ((NT + 3) // 4), E), threadgroup=(32, 4, 1),
            output_shapes=[(E, N, 512), (E, N, 128), (E, N, 512), (E, N, 128)],
            output_dtypes=[mx.uint32, mx.uint8, mx.uint32, mx.uint8],
        )
    )


def pack_moe_dn_weights(wd: mx.array, scd: mx.array) -> mx.array:
    """Lossless runtime repack of MXFP4 K=1280 (wd, scd) -> (E, N//4, 2720) u8 (0B pad)."""
    E, N, K8 = wd.shape
    assert K8 == 160 and N % 4 == 0 and scd.shape == (E, N, 40)
    k = _KERNELS.get("pack_dn")
    if k is None:
        k = _KERNELS["pack_dn"] = mx.fast.metal_kernel(
            name="bv3_moe_pack_dn", input_names=["w", "sc"], output_names=["out"],
            header=GLUE_HEADER, source=_PACK_DN_SRC, ensure_row_contiguous=True,
        )
    nt = N // 4
    (out,) = k(
        inputs=[wd, scd], template=[("NN", N)],
        grid=(32, 4 * ((nt + 3) // 4), E), threadgroup=(32, 4, 1),
        output_shapes=[(E, nt, 2720)], output_dtypes=[mx.uint8],
    )
    return out


def unpack_moe_dn_weights(dn_packed: mx.array) -> tuple[mx.array, mx.array]:
    """Lossless inverse of `pack_moe_dn_weights` for SwitchGLU prefill fallback."""
    E, NT, BPT = dn_packed.shape
    assert BPT == 2720
    N = NT * 4
    k = _KERNELS.get("unpack_dn")
    if k is None:
        k = _KERNELS["unpack_dn"] = mx.fast.metal_kernel(
            name="bv3_moe_unpack_dn", input_names=["packed"], output_names=["w", "sc"],
            header=GLUE_HEADER, source=_UNPACK_DN_SRC, ensure_row_contiguous=True,
        )
    return tuple(
        k(
            inputs=[dn_packed], template=[("NN", N)],
            grid=(32, 4 * ((NT + 3) // 4), E), threadgroup=(32, 4, 1),
            output_shapes=[(E, N, 160), (E, N, 40)],
            output_dtypes=[mx.uint32, mx.uint8],
        )
    )


def moe_gate_up_swiglu_packed_rows(x, gu_packed, inds, *, dedup=None, tmax=None, nsg=None, gz=0, single_kb=False):
    """Contiguous K-major tile-packed gate+up+SwiGLU (1 buffer vs 4, 0B pad, 0 bit diffs)."""
    B, K = x.shape
    N = gu_packed.shape[1] * 4
    assert 1 <= B <= 24 and K == 4096 and x.dtype == mx.bfloat16
    if dedup is None and B >= 2:
        dedup = moe_dedup_rows(inds)
    use_dedup = dedup is not None
    k = _KERNELS.get("gu_packed")
    if k is None:
        k = _KERNELS["gu_packed"] = mx.fast.metal_kernel(
            name="bv3_moe_gu_packed",
            input_names=["gu_packed", "x", "inds", "u_meta", "u_pqs", "n_uniq"],
            output_names=["out"],
            header=HEADER, source=_GU_PACKED_SRC, ensure_row_contiguous=True,
        )
    tm = (1 if B == 1 else (4 if B <= 4 else 8)) if tmax is None else int(tmax)
    nsg_v = 2 if nsg is None else int(nsg)
    inds_u32 = inds if inds.dtype == mx.uint32 else inds.astype(mx.uint32)
    if use_dedup:
        u_meta, u_pqs, _, _, n_uniq = dedup
    else:
        u_meta, u_pqs, n_uniq = _ZERO_U32, _ZERO_U32, _ZERO_I32
    gz_grid = gz if gz > 0 else (B * 8)
    (out,) = k(
        inputs=[gu_packed, x, inds_u32, u_meta, u_pqs, n_uniq],
        template=[
            ("BB", B), ("USE_DEDUP", int(use_dedup)), ("SINGLE_KB", int(single_kb)),
            ("TMAX", tm), ("NSG", nsg_v), ("GZ", int(gz)), ("NN", N),
        ],
        grid=(32, nsg_v * ((N // 4 + nsg_v - 1) // nsg_v), gz_grid),
        threadgroup=(32, nsg_v, 1),
        output_shapes=[(B, 8, N)], output_dtypes=[mx.bfloat16],
    )
    return out


def moe_down_combine_packed_rows(
    h,
    dn_packed,
    inds,
    wts,
    *,
    shared=None,
    residual=None,
    dedup=None,
    nsg=None,
    tmax=None,
    rows=None,
    ntg_y=None,
    se_mid=None,
    se_w_fp8=None,
    se_sc=None,
):
    """Contiguous K-major tile-packed down_proj + combine8 + optional T7 se_down + residual."""
    B, E, K = h.shape
    N = dn_packed.shape[1] * 4
    assert 1 <= B <= 24 and E == 8 and K == 1280 and h.dtype == mx.bfloat16
    has_se = se_mid is not None
    has_sh = (shared is not None) and not has_se
    has_res = residual is not None
    sh = shared if has_sh else _ZERO_BF16
    rs = residual if has_res else _ZERO_BF16
    sm = se_mid if has_se else _ZERO_BF16
    sw8 = se_w_fp8 if has_se else _ZERO_U8
    sc8 = se_sc if has_se else _ZERO_F32
    if dedup is None and B >= 2:
        dedup = moe_dedup_rows(inds)
    use_dedup = dedup is not None
    rw = 4 if rows is None else int(rows)
    assert rw in (4, 8) and N % rw == 0
    nsg_v = (2 if (2 <= B <= 4 and not has_se) else 8) if nsg is None else int(nsg)
    tm = (1 if B == 1 else 2) if tmax is None else int(tmax)
    ny_tpl = 0 if ntg_y is None else int(ntg_y)
    ny_grid = ny_tpl if ny_tpl > 0 else (N // rw)
    k = _KERNELS.get("dc_packed")
    if k is None:
        k = _KERNELS["dc_packed"] = mx.fast.metal_kernel(
            name="bv3_moe_dc_packed",
            input_names=["dn_packed", "x", "inds", "u_meta", "u_pqs", "n_uniq", "wts", "shared", "res", "se_mid", "se_w", "se_sc"],
            output_names=["out"],
            header=HEADER, source=_DC_PACKED_SRC, ensure_row_contiguous=True,
        )
    inds_u32 = inds if inds.dtype == mx.uint32 else inds.astype(mx.uint32)
    if use_dedup:
        u_meta, u_pqs, _, _, n_uniq = dedup
    else:
        u_meta, u_pqs, n_uniq = _ZERO_U32, _ZERO_U32, _ZERO_I32
    (out,) = k(
        inputs=[dn_packed, h, inds_u32, u_meta, u_pqs, n_uniq, wts, sh, rs, sm, sw8, sc8],
        template=[
            ("BB", B), ("USE_DEDUP", int(use_dedup)), ("ROWS", rw), ("TMAX", tm),
            ("NSG", nsg_v), ("NTG_Y", ny_tpl), ("NN", N),
            ("HAS_SH", int(has_sh)), ("HAS_RES", int(has_res)), ("HAS_SE_DOWN", int(has_se)),
        ],
        grid=(32, nsg_v * ny_grid, 1), threadgroup=(32, nsg_v, 1),
        output_shapes=[(B, N)], output_dtypes=[mx.bfloat16],
    )
    return out
