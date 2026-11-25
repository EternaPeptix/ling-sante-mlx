"""Fused bit-exact MXFP4 MoE kernels for BailingSparseMoE (B = 1..16).

1. `moe_gate_up_swiglu_rows(x, wg, sg, wu, su, inds)`:
   Reads x once, deduplicates experts across B<=16 rows, computes both gate_proj & up_proj
   (MXFP4 K=4096, N=1280) and fuses `swiglu(gate, up)` in registers -> (B, 8, 1280) bf16.
2. `moe_down_combine_rows(h, wd, sd, inds, wts, shared=None, residual=None)`:
   Deduplicates experts across B<=16 rows, computes down_proj (MXFP4 K=1280, N=4096),
   left-fold combines `(routed * wts).sum(-2).astype(bf16)` in threadgroup memory, and
   optionally adds `shared` and `residual` -> (B, 4096) bf16 in ONE launch!
"""
import os

import mlx.core as mx

from .gather import mlx_header
from .glue import GLUE_HEADER

_U32_DEQUANT = r"""
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

template <typename T, typename U, int BITS, int GS, bool FAST, int R_TILE, int T_TILE, bool WRITE_SWIGLU>
inline void moe_gu_pass_tile(
    const device uint8_t* w0,
    const device uint8_t* s0,
    const device T* x,
    thread const int* xo,
    thread const int* pqs,
    const int n_tok,
    const int r0,
    const int out_row,
    const int NN,
    const int KK,
    const int in_w,
    const int in_g,
    const uint lane,
    thread T g_bf[T_TILE][R_TILE],
    device T* out) {
  constexpr int PF = 32 / BITS;
  constexpr int PPT = FAST ? 2 : 1;
  constexpr int VPT = PF * PPT;
  constexpr int BLOCK = VPT * 32;
  constexpr int KEND_SUB = FAST ? 0 : BLOCK;

  thread U res[T_TILE][R_TILE];
#pragma unroll
  for (int t = 0; t < T_TILE; t++)
#pragma unroll
    for (int r = 0; r < R_TILE; r++) res[t][r] = 0;

  const device uint8_t* wp = w0 + (long)r0 * in_w;
  const device uint8_t* sp = s0 + (long)r0 * in_g;
  int k = 0;
  for (; k < KK - KEND_SUB; k += BLOCK) {
    if constexpr (T_TILE == 1) {
      thread U xt[VPT];
      load_vector<T, U, VPT>(x + xo[0] + k, xt);
#pragma unroll
      for (int r = 0; r < R_TILE; r++) {
        const int rr = (out_row + r0 + r < NN) ? r : 0;
        const device uint32_t* wptr = (const device uint32_t*)(wp + rr * in_w);
        thread uint32_t rw[PPT];
#pragma unroll
        for (int p = 0; p < PPT; p++) rw[p] = wptr[p];
        U sv = dequantize_scale<U, GS>((sp + rr * in_g)[0]);
        thread U d[VPT];
        dequant_vals_u32<BITS, VPT>(rw, d);
        res[0][r] += dot_vals<BITS, VPT>(xt, d, sv);
      }
    } else {
#pragma unroll
      for (int r = 0; r < R_TILE; r++) {
        const int rr = (out_row + r0 + r < NN) ? r : 0;
        const device uint32_t* wptr = (const device uint32_t*)(wp + rr * in_w);
        thread uint32_t rw[PPT];
#pragma unroll
        for (int p = 0; p < PPT; p++) rw[p] = wptr[p];
        U sv = dequantize_scale<U, GS>((sp + rr * in_g)[0]);
        thread U d[VPT];
        dequant_vals_u32<BITS, VPT>(rw, d);
#pragma unroll
        for (int t = 0; t < T_TILE; t++) {
          if (t < n_tok) {
            thread U xt[VPT];
            load_vector<T, U, VPT>(x + xo[t] + k, xt);
            res[t][r] += dot_vals<BITS, VPT>(xt, d, sv);
          }
        }
      }
    }
    wp += BLOCK * 4 / PF;
    sp += BLOCK / GS;
  }
  if constexpr (!FAST) {
    const int rem = clamp((int)(KK - k - int(lane) * VPT), 0, VPT);
    if (rem > 0) {
      for (int t = 0; t < T_TILE; t++) {
        if (t < n_tok) {
          thread U xt[VPT];
          load_vector_safe<T, U, VPT>(x + xo[t] + k, xt, rem);
          for (int r = 0; r < R_TILE; r++) {
            if (out_row + r0 + r < NN) {
              res[t][r] += qdot_safe<U, VPT, BITS>(
                  wp + r * in_w, xt, dequantize_scale<U, GS>((sp + r * in_g)[0]), rem);
            }
          }
        }
      }
    }
  }
#pragma unroll
  for (int t = 0; t < T_TILE; t++) {
#pragma unroll
    for (int r = 0; r < R_TILE; r++) {
      U v = simd_sum(res[t][r]);
      if (lane == 0 && t < n_tok && (out_row + r0 + r) < NN) {
        if constexpr (WRITE_SWIGLU) {
          out[pqs[t] * NN + out_row + r0 + r] = glue_swiglu_bf16(g_bf[t][r], static_cast<T>(v));
        } else {
          g_bf[t][r] = static_cast<T>(v);
        }
      }
    }
  }
}
"""

HEADER = mlx_header() + GLUE_HEADER + _U32_DEQUANT
_KERNELS = {}
_GU_TMAX_CAP = int(os.environ.get("LINGSANTE_MOE_GU_TMAX_CAP", os.environ.get("LINGSANTE_MOE_TMAX_CAP", "4")))
_DC_TMAX_CAP = int(os.environ.get("LINGSANTE_MOE_DC_TMAX_CAP", os.environ.get("LINGSANTE_MOE_TMAX_CAP", "1")))
_GU_ADAPTIVE_MIN_B = int(os.environ.get("LINGSANTE_MOE_GU_ADAPTIVE_MIN_B", "5"))
_ZERO_BF16 = mx.zeros((1,), dtype=mx.bfloat16)


def _tmax(B, cap):
    for t in (1, 2, 4):
        if B <= t or t >= cap:
            return t
    return 4


_DEDUP_SRC = r"""
    constexpr int NPAIR = BB * 8;
    const uint lane = thread_index_in_simdgroup;
    threadgroup uint ind_s[NPAIR];
    threadgroup uchar seen[NPAIR];
    for (int i = int(lane); i < NPAIR; i += 32) {
      ind_s[i] = inds[i];
      seen[i] = 0;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    int uniq = 0;
    int total_out = 0;
    for (int q = 0; q < NPAIR; ++q) {
      if (seen[q]) continue;
      const uint e = ind_s[q];
      int cnt = 0;
      for (int base = 0; base < NPAIR; base += 32) {
        int p = base + int(lane);
        bool match = (p >= q) && (p < NPAIR) && (ind_s[p] == e);
        if (match) seen[p] = 1;
        int take = match ? 1 : 0;
        int pre = simd_prefix_exclusive_sum(take);
        if (take) u_pqs[total_out + cnt + pre] = (uint)p;
        cnt += simd_sum(take);
      }
      if (lane == 0) {
        u_exp[uniq] = e;
        u_off[uniq] = total_out;
        u_cnt[uniq] = cnt;
      }
      uniq += 1;
      total_out += cnt;
      simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
      for (int u = uniq; u < NPAIR; ++u) u_cnt[u] = 0;
      n_uniq[0] = uniq;
    }
"""


def _dedup_kernel():
    k = _KERNELS.get("dedup")
    if k is None:
        k = _KERNELS["dedup"] = mx.fast.metal_kernel(
            name="bv3_moe_dedup",
            input_names=["inds"],
            output_names=["u_exp", "u_off", "u_cnt", "u_pqs", "n_uniq"],
            header=GLUE_HEADER,
            source=_DEDUP_SRC,
            ensure_row_contiguous=True,
        )
    return k


def moe_dedup_rows(inds: mx.array):
    """Deduplicate `(B, 8)` expert indices into CSR worklists in 1 tiny 32-lane kernel."""
    B, E = inds.shape
    assert 1 <= B <= 24 and E == 8
    NPAIR = B * 8
    inds_u32 = inds if inds.dtype == mx.uint32 else inds.astype(mx.uint32)
    return _dedup_kernel()(
        inputs=[inds_u32],
        template=[("BB", B)],
        grid=(32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(NPAIR,), (NPAIR,), (NPAIR,), (NPAIR,), (1,)],
        output_dtypes=[mx.uint32, mx.int32, mx.int32, mx.uint32, mx.int32],
    )


_GU_PRE_INLINE = r"""
    constexpr int NPAIR = BB * 8;
    constexpr int PF = 32 / BITS;      // 8
    constexpr int PPT = FAST ? 2 : 1;  // 2 for K=4096
    constexpr int VPT = PF * PPT;      // 16
    constexpr int BLOCK = VPT * 32;    // 512
    constexpr int SSTEP = GS / VPT;    // 2
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
"""

_GU_PRE_DEDUP = r"""
    constexpr int PF = 32 / BITS;      // 8
    constexpr int PPT = FAST ? 2 : 1;  // 2 for K=4096
    constexpr int VPT = PF * PPT;      // 16
    constexpr int BLOCK = VPT * 32;    // 512
    constexpr int SSTEP = GS / VPT;    // 2
    typedef float U;
    typedef bfloat16_t T;

    uint3 tid = threadgroup_position_in_grid;
    uint sg = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    const int q = tid.z;
    if (q >= n_uniq[0]) return;

    const uint e = u_exp[q];
    const int off0 = u_off[q];
    const int cnt = u_cnt[q];
"""


def _gu_src(use_dedup: bool) -> str:
    pq_expr = "(tb + t < cnt) ? (int)u_pqs[off0 + tb + t] : (int)u_pqs[off0]" if use_dedup else "(tb + t < cnt) ? (int)pl[tb + t] : q"
    pre = _GU_PRE_DEDUP if use_dedup else _GU_PRE_INLINE
    body = r"""
    const int in_w = KK * BITS / 8;
    const int in_g = KK / GS;
    const int out_row = tid.y * (ROWS * NSG) + sg * ROWS;
    if (out_row >= NN) return;

    const device uint8_t* wg0 = (const device uint8_t*)wg + (long)e * NN * in_w + (long)out_row * in_w + lane * PPT * 4;
    const device uint8_t* sg0 = scg + (long)e * NN * in_g + (long)out_row * in_g + lane / SSTEP;
    const device uint8_t* wu0 = (const device uint8_t*)wu + (long)e * NN * in_w + (long)out_row * in_w + lane * PPT * 4;
    const device uint8_t* su0 = scu + (long)e * NN * in_g + (long)out_row * in_g + lane / SSTEP;

    if constexpr (ADAPTIVE && ROWS == 4) {
      if (cnt == 1) {
        const int tb = 0;
        const int t = 0;
        const int pq0 = __PQ_EXPR__;
        int pqs[1] = { pq0 };
        int xo[1] = { (pq0 / 8) * KK + int(lane) * VPT };
        T g_bf[1][4];
        moe_gu_pass_tile<T, U, BITS, GS, FAST, 4, 1, false>(wg0, sg0, x, xo, pqs, 1, 0, out_row, NN, KK, in_w, in_g, lane, g_bf, out);
        moe_gu_pass_tile<T, U, BITS, GS, FAST, 4, 1, true>(wu0, su0, x, xo, pqs, 1, 0, out_row, NN, KK, in_w, in_g, lane, g_bf, out);
      } else if (cnt == 2) {
        const int tb = 0;
        int xo[2], pqs[2];
        for (int t = 0; t < 2; t++) {
          int pq = __PQ_EXPR__;
          pqs[t] = pq;
          xo[t] = (pq / 8) * KK + int(lane) * VPT;
        }
        T g_bf[2][4];
        moe_gu_pass_tile<T, U, BITS, GS, FAST, 4, 2, false>(wg0, sg0, x, xo, pqs, 2, 0, out_row, NN, KK, in_w, in_g, lane, g_bf, out);
        moe_gu_pass_tile<T, U, BITS, GS, FAST, 4, 2, true>(wu0, su0, x, xo, pqs, 2, 0, out_row, NN, KK, in_w, in_g, lane, g_bf, out);
      } else if (cnt >= 3 && cnt <= 4) {
        const int tb = 0;
        int xo[4], pqs[4];
        for (int t = 0; t < 4; t++) {
          int pq = __PQ_EXPR__;
          pqs[t] = pq;
          xo[t] = (pq / 8) * KK + int(lane) * VPT;
        }
        for (int rh = 0; rh < 2; rh++) {
          const int r0 = rh * 2;
          if (out_row + r0 >= NN) continue;
          T g_bf[4][2];
          moe_gu_pass_tile<T, U, BITS, GS, FAST, 2, 4, false>(wg0, sg0, x, xo, pqs, cnt, r0, out_row, NN, KK, in_w, in_g, lane, g_bf, out);
          moe_gu_pass_tile<T, U, BITS, GS, FAST, 2, 4, true>(wu0, su0, x, xo, pqs, cnt, r0, out_row, NN, KK, in_w, in_g, lane, g_bf, out);
        }
      } else if (cnt >= 5) {
        for (int r = 0; r < 4; r++) {
          if (out_row + r >= NN) continue;
          for (int tb = 0; tb < cnt; tb += 8) {
            const int n_tok = min(8, cnt - tb);
            int xo[8], pqs[8];
            for (int t = 0; t < 8; t++) {
              int pq = __PQ_EXPR__;
              pqs[t] = pq;
              xo[t] = (pq / 8) * KK + int(lane) * VPT;
            }
            T g_bf[8][1];
            moe_gu_pass_tile<T, U, BITS, GS, FAST, 1, 8, false>(wg0, sg0, x, xo, pqs, n_tok, r, out_row, NN, KK, in_w, in_g, lane, g_bf, out);
            moe_gu_pass_tile<T, U, BITS, GS, FAST, 1, 8, true>(wu0, su0, x, xo, pqs, n_tok, r, out_row, NN, KK, in_w, in_g, lane, g_bf, out);
          }
        }
      }
      return;
    }

    for (int tb = 0; tb < cnt; tb += TMAX) {
        int xo[TMAX];
        int pqs[TMAX];
        for (int t = 0; t < TMAX; t++) {
            int pq = __PQ_EXPR__;
            pqs[t] = pq;
            xo[t] = (pq / 8) * KK + int(lane) * VPT;
        }
        T g_bf[TMAX][ROWS];
        constexpr int KEND_SUB = FAST ? 0 : BLOCK;

        // Pass 1: gate_proj
        {
            thread U res[TMAX][ROWS];
#pragma unroll
            for (int t = 0; t < TMAX; t++)
#pragma unroll
                for (int r = 0; r < ROWS; r++) res[t][r] = 0;

            const device uint8_t* wp = wg0;
            const device uint8_t* sp = sg0;
            int k = 0;
            for (; k < KK - KEND_SUB; k += BLOCK) {
                if constexpr (HOIST_DQ) {
#pragma unroll
                    for (int r = 0; r < ROWS; r++) {
                        const int rr = (out_row + r < NN) ? r : 0;
                        const device uint32_t* wptr = (const device uint32_t*)(wp + rr * in_w);
                        thread uint32_t rw[PPT];
#pragma unroll
                        for (int p = 0; p < PPT; p++) rw[p] = wptr[p];
                        U sv = dequantize_scale<U, GS>((sp + rr * in_g)[0]);
                        thread U d[VPT];
                        dequant_vals_u32<BITS, VPT>(rw, d);
#pragma unroll
                        for (int t = 0; t < TMAX; t++) {
                            if (tb + t < cnt) {
                                thread U xt[VPT];
                                load_vector<T, U, VPT>(x + xo[t] + k, xt);
                                res[t][r] += dot_vals<BITS, VPT>(xt, d, sv);
                            }
                        }
                    }
                } else {
                    thread uint32_t rw[ROWS][PPT];
                    U sv[ROWS];
#pragma unroll
                    for (int r = 0; r < ROWS; r++) {
                        const int rr = (out_row + r < NN) ? r : 0;
                        const device uint32_t* wptr = (const device uint32_t*)(wp + rr * in_w);
#pragma unroll
                        for (int p = 0; p < PPT; p++) rw[r][p] = wptr[p];
                        sv[r] = dequantize_scale<U, GS>((sp + rr * in_g)[0]);
                    }
#pragma unroll
                    for (int t = 0; t < TMAX; t++) {
                        if (tb + t < cnt) {
                            thread U xt[VPT];
                            load_vector<T, U, VPT>(x + xo[t] + k, xt);
#pragma unroll
                            for (int r = 0; r < ROWS; r++) {
                                thread U d[VPT];
                                dequant_vals_u32<BITS, VPT>(rw[r], d);
                                res[t][r] += dot_vals<BITS, VPT>(xt, d, sv[r]);
                            }
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
#pragma unroll
            for (int t = 0; t < TMAX; t++) {
#pragma unroll
                for (int r = 0; r < ROWS; r++) {
                    g_bf[t][r] = static_cast<T>(simd_sum(res[t][r]));
                }
            }
        }

        // Pass 2: up_proj + fused swiglu output
        {
            thread U res[TMAX][ROWS];
#pragma unroll
            for (int t = 0; t < TMAX; t++)
#pragma unroll
                for (int r = 0; r < ROWS; r++) res[t][r] = 0;

            const device uint8_t* wp = wu0;
            const device uint8_t* sp = su0;
            int k = 0;
            for (; k < KK - KEND_SUB; k += BLOCK) {
                if constexpr (HOIST_DQ) {
#pragma unroll
                    for (int r = 0; r < ROWS; r++) {
                        const int rr = (out_row + r < NN) ? r : 0;
                        const device uint32_t* wptr = (const device uint32_t*)(wp + rr * in_w);
                        thread uint32_t rw[PPT];
#pragma unroll
                        for (int p = 0; p < PPT; p++) rw[p] = wptr[p];
                        U sv = dequantize_scale<U, GS>((sp + rr * in_g)[0]);
                        thread U d[VPT];
                        dequant_vals_u32<BITS, VPT>(rw, d);
#pragma unroll
                        for (int t = 0; t < TMAX; t++) {
                            if (tb + t < cnt) {
                                thread U xt[VPT];
                                load_vector<T, U, VPT>(x + xo[t] + k, xt);
                                res[t][r] += dot_vals<BITS, VPT>(xt, d, sv);
                            }
                        }
                    }
                } else {
                    thread uint32_t rw[ROWS][PPT];
                    U sv[ROWS];
#pragma unroll
                    for (int r = 0; r < ROWS; r++) {
                        const int rr = (out_row + r < NN) ? r : 0;
                        const device uint32_t* wptr = (const device uint32_t*)(wp + rr * in_w);
#pragma unroll
                        for (int p = 0; p < PPT; p++) rw[r][p] = wptr[p];
                        sv[r] = dequantize_scale<U, GS>((sp + rr * in_g)[0]);
                    }
#pragma unroll
                    for (int t = 0; t < TMAX; t++) {
                        if (tb + t < cnt) {
                            thread U xt[VPT];
                            load_vector<T, U, VPT>(x + xo[t] + k, xt);
#pragma unroll
                            for (int r = 0; r < ROWS; r++) {
                                thread U d[VPT];
                                dequant_vals_u32<BITS, VPT>(rw[r], d);
                                res[t][r] += dot_vals<BITS, VPT>(xt, d, sv[r]);
                            }
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
#pragma unroll
            for (int t = 0; t < TMAX; t++) {
#pragma unroll
                for (int r = 0; r < ROWS; r++) {
                    U vu = simd_sum(res[t][r]);
                    if (lane == 0 && tb + t < cnt && out_row + r < NN) {
                        out[pqs[t] * NN + out_row + r] = glue_swiglu_bf16(g_bf[t][r], static_cast<T>(vu));
                    }
                }
            }
        }
    }
"""
    return pre + body.replace("__PQ_EXPR__", pq_expr)


_DC_HDR = r"""
    constexpr int NPAIR = BB * 8;
    constexpr int PF = 32 / BITS;      // 8
    constexpr int PPT = FAST ? 2 : 1;  // 1 for K=1280
    constexpr int VPT = PF * PPT;      // 8
    constexpr int BLOCK = VPT * 32;    // 256
    constexpr int SSTEP = GS / VPT;    // 4
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
"""

_DC_LOOP_CORE = r"""
            const device uint8_t* ws0 = (const device uint8_t*)w + (long)e * NN * in_w + (long)out_row * in_w + lane * PPT * 4;
            const device uint8_t* sc0 = scales + (long)e * NN * in_g + (long)out_row * in_g + lane / SSTEP;
            for (int tb = 0; tb < cnt; tb += TMAX) {
                int xo[TMAX];
                int pqs[TMAX];
                for (int t = 0; t < TMAX; t++) {
                    int pq = __PQ_EXPR__;
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
                    if constexpr (HOIST_DQ == 2) {
#pragma unroll
                        for (int r = 0; r < ROWS; r++) {
                            const device uint32_t* wp = (const device uint32_t*)(ws + r * in_w);
                            thread uint32_t rw[PPT];
#pragma unroll
                            for (int p = 0; p < PPT; p++) rw[p] = wp[p];
                            U sv = dequantize_scale<U, GS>((sc + r * in_g)[0]);
                            thread U d[VPT];
                            dequant_vals_u32<BITS, VPT>(rw, d);
#pragma unroll
                            for (int t = 0; t < TMAX; t++) {
                                if (tb + t < cnt) {
                                    thread U xt[VPT];
                                    load_vector<T, U, VPT>(x + xo[t] + k, xt);
                                    result[t][r] += dot_vals<BITS, VPT>(xt, d, sv);
                                }
                            }
                        }
                    } else if constexpr (HOIST_DQ == 1) {
                        thread T xt_bf[TMAX][VPT];
#pragma unroll
                        for (int t = 0; t < TMAX; t++) {
                            if (tb + t < cnt) {
                                const device T* xp = x + xo[t] + k;
#pragma unroll
                                for (int v = 0; v < VPT; v++) xt_bf[t][v] = xp[v];
                            }
                        }
#pragma unroll
                        for (int r = 0; r < ROWS; r++) {
                            const device uint32_t* wp = (const device uint32_t*)(ws + r * in_w);
                            thread uint32_t rw[PPT];
#pragma unroll
                            for (int p = 0; p < PPT; p++) rw[p] = wp[p];
                            U sv = dequantize_scale<U, GS>((sc + r * in_g)[0]);
                            thread U d[VPT];
                            dequant_vals_u32<BITS, VPT>(rw, d);
#pragma unroll
                            for (int t = 0; t < TMAX; t++) {
                                if (tb + t < cnt) {
                                    thread U xt[VPT];
#pragma unroll
                                    for (int v = 0; v < VPT; v++) xt[v] = static_cast<U>(xt_bf[t][v]);
                                    result[t][r] += dot_vals<BITS, VPT>(xt, d, sv);
                                }
                            }
                        }
                    } else {
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
                                    dequant_vals_u32<BITS, VPT>(rw[r], d);
                                    result[t][r] += dot_vals<BITS, VPT>(xt, d, sv[r]);
                                }
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
#pragma unroll
                for (int t = 0; t < TMAX; t++) {
#pragma unroll
                    for (int r = 0; r < ROWS; r++) {
                        U v = simd_sum(result[t][r]);
                        if (lane == 0 && tb + t < cnt) {
                            dn_s[pqs[t] * 4 + r] = float(static_cast<T>(v));
                        }
                    }
                }
            }
"""

_DC_TAIL = r"""
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int idx = int(ttid); idx < BB * 4; idx += 32 * NSG) {
        const int b = idx / 4;
        const int r = idx % 4;
        const int c = out_row + r;
        float ev[8], ww[8];
        for (int j = 0; j < 8; j++) {
            ev[j] = dn_s[(b * 8 + j) * 4 + r];
            ww[j] = wts[b * 8 + j];
        }
        bfloat16_t val = static_cast<bfloat16_t>(glue_moe_combine8(ev, ww));
        if constexpr (HAS_SH) val = val + shared[b * NN + c];
        if constexpr (HAS_RES) val = res[b * NN + c] + val;
        out[b * NN + c] = val;
    }
"""


def _dc_src(use_dedup: bool) -> str:
    if use_dedup:
        mid = r"""
    threadgroup float dn_s[NPAIR * 4];
    const int nu = n_uniq[0];
    for (int u = int(sg); u < nu; u += NSG) {
        const uint e = u_exp[u];
        const int off0 = u_off[u];
        const int cnt = u_cnt[u];
        __CORE__
    }
""".replace("__CORE__", _DC_LOOP_CORE.replace("__PQ_EXPR__", "(tb + t < cnt) ? (int)u_pqs[off0 + tb + t] : (int)u_pqs[off0]"))
    else:
        mid = r"""
    threadgroup uint ind_s[NPAIR];
    threadgroup uint pl_all[NSG * NPAIR];
    threadgroup float dn_s[NPAIR * 4];
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
            __CORE__
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
""".replace("__CORE__", _DC_LOOP_CORE.replace("__PQ_EXPR__", "(tb + t < cnt) ? (int)pl[tb + t] : P0"))
    return _DC_HDR + mid + _DC_TAIL


def _gu_kernel(use_dedup: bool = False):
    key = f"gu_d{int(use_dedup)}"
    k = _KERNELS.get(key)
    if k is None:
        in_names = (
            ["wg", "scg", "wu", "scu", "x", "u_exp", "u_off", "u_cnt", "u_pqs", "n_uniq"]
            if use_dedup
            else ["wg", "scg", "wu", "scu", "x", "inds"]
        )
        k = _KERNELS[key] = mx.fast.metal_kernel(
            name=f"bv3_moe_gu_sw_d{int(use_dedup)}",
            input_names=in_names,
            output_names=["out"],
            header=HEADER,
            source=_gu_src(use_dedup),
            ensure_row_contiguous=True,
        )
    return k


def _dc_kernel(use_dedup: bool = False):
    key = f"dc_d{int(use_dedup)}"
    k = _KERNELS.get(key)
    if k is None:
        in_names = (
            ["w", "scales", "x", "u_exp", "u_off", "u_cnt", "u_pqs", "n_uniq", "wts", "shared", "res"]
            if use_dedup
            else ["w", "scales", "x", "inds", "wts", "shared", "res"]
        )
        k = _KERNELS[key] = mx.fast.metal_kernel(
            name=f"bv3_moe_dc_d{int(use_dedup)}",
            input_names=in_names,
            output_names=["out"],
            header=HEADER,
            source=_dc_src(use_dedup),
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
    sh = shared if has_sh else mx.zeros((1,), dtype=mx.bfloat16)
    rs = residual if has_res else mx.zeros((1,), dtype=mx.bfloat16)
    (out,) = _comb8_kernel()(
        inputs=[dn, wts, sh, rs],
        template=[("NN", N), ("HAS_SH", int(has_sh)), ("HAS_RES", int(has_res))],
        grid=(N, B, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(B, N)],
        output_dtypes=[dn.dtype],
    )
    return out


def moe_gate_up_swiglu_rows(x, wg, sg, wu, su, inds, *, group_size=32, bits=4, tmax=None, nsg=None, rows=None, hoist_dq=None, dedup=None, adaptive=None):
    B, E = inds.shape
    K = x.shape[-1]
    N = wg.shape[1]
    assert 1 <= B <= 24 and E == 8 and x.shape == (B, K) and x.dtype == mx.bfloat16
    fast = (N % 8 == 0) and (K % ((32 // bits) * 2 * 32) == 0)
    if rows is None:
        rw = 4
    else:
        rw = int(rows)
    assert rw in (1, 2, 4) and N % rw == 0
    if tmax is None:
        tm = 2 if B >= 5 else _tmax(B, _GU_TMAX_CAP)
    else:
        tm = int(tmax)
    if nsg is None:
        nsg = 4 if B >= 5 else 2
    hq = 0 if hoist_dq is None else int(hoist_dq)
    if not fast or tm == 1:
        hq = 0
    ad = (int(B >= _GU_ADAPTIVE_MIN_B) if adaptive is None else int(bool(adaptive))) if rw == 4 else 0
    use_dedup = dedup is not None
    if use_dedup:
        u_exp, u_off, u_cnt, u_pqs, n_uniq = dedup
        inputs = [wg, sg, wu, su, x, u_exp, u_off, u_cnt, u_pqs, n_uniq]
    else:
        inds_u32 = inds if inds.dtype == mx.uint32 else inds.astype(mx.uint32)
        inputs = [wg, sg, wu, su, x, inds_u32]
    (out,) = _gu_kernel(use_dedup)(
        inputs=inputs,
        template=[("BB", B), ("GS", group_size), ("BITS", bits), ("FAST", fast), ("HOIST_DQ", hq), ("ADAPTIVE", ad), ("ROWS", rw), ("TMAX", tm), ("NSG", nsg), ("KK", K), ("NN", N)],
        grid=(32, nsg * ((N + rw * nsg - 1) // (rw * nsg)), B * E),
        threadgroup=(32, nsg, 1),
        output_shapes=[(B, E, N)],
        output_dtypes=[mx.bfloat16],
    )
    return out


def moe_down_combine_rows(h, wd, sd, inds, wts, *, shared=None, residual=None, group_size=32, bits=4, nsg=None, tmax=None, dedup=None, hoist_dq=None):
    B, E, K = h.shape
    N = wd.shape[1]
    assert 1 <= B <= 24 and E == 8 and h.dtype == mx.bfloat16
    if nsg is None:
        nsg = 4 if (B <= 2 or B >= 5) else 2
    pf = 32 // bits
    fast = (N % 8 == 0) and (K % (pf * 2 * 32) == 0)
    exact_blk = (K % (pf * 32) == 0)
    has_sh = shared is not None
    has_res = residual is not None
    sh = shared if has_sh else _ZERO_BF16
    rs = residual if has_res else _ZERO_BF16
    if tmax is None:
        tm = 4 if B >= 21 else (2 if B >= 8 else _tmax(B, _DC_TMAX_CAP))
    else:
        tm = int(tmax)
    if hoist_dq is None:
        hdq = int(B >= 8 and tm > 1)
    else:
        hdq = int(hoist_dq)
    if not (fast or exact_blk) or tm == 1:
        hdq = 0
    use_dedup = dedup is not None
    if use_dedup:
        u_exp, u_off, u_cnt, u_pqs, n_uniq = dedup
        inputs = [wd, sd, h, u_exp, u_off, u_cnt, u_pqs, n_uniq, wts, sh, rs]
    else:
        inds_u32 = inds if inds.dtype == mx.uint32 else inds.astype(mx.uint32)
        inputs = [wd, sd, h, inds_u32, wts, sh, rs]
    (out,) = _dc_kernel(use_dedup)(
        inputs=inputs,
        template=[
            ("BB", B), ("GS", group_size), ("BITS", bits), ("FAST", fast), ("EXACT_BLK", exact_blk),
            ("HOIST_DQ", hdq), ("TMAX", tm), ("NSG", nsg), ("KK", K), ("NN", N), ("HAS_SH", int(has_sh)), ("HAS_RES", int(has_res)),
        ],
        grid=(32, nsg * (N // 4), 1),
        threadgroup=(32, nsg, 1),
        output_shapes=[(B, N)],
        output_dtypes=[mx.bfloat16],
    )
    return out
