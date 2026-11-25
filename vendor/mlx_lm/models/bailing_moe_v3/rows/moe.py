"""Fused bit-exact MXFP4 MoE kernels for BailingSparseMoE (B = 1..24).

Phase A optimizations (strictly 0 bit diffs vs stock B=1 SwitchGLU + combine + shared + residual):
  - T1: Reverts monolithic `ADAPTIVE=1` static GPR ceiling & `dequant_vals_u32` regression.
  - T2: LPT `(cnt DESC, exp ASC)` packed CSR `moe_dedup_rows` (`u_meta`, `u_pqs`, `n_hot`, `n_uniq`),
        2-PSO zero-spill split (`cnt==1` single-pass vs `cnt>=2` hot) + lean single-PSO modes,
        fixed `moe_dc` `TMAX=2` at `B=2..7` & dedup support for `B=2..24`,
        `threadgroup atomic_uint next_u` work-stealing in `moe_dc`, `bfloat16_t dn_s`, capped `grid.z`.
  - T3: Optional runtime-on-load K-major `gu_packed` (`17408 B`/4-row tile, 0B pad) & `dn_packed`
        (`2720 B`/4-row tile, 0B pad) + unpack fallback for `SwitchGLU` prefill.
  - T4: Exact `FP4_PAIR_LUT[256]` (preserving `-0.0f` for nibble 8) + streaming 4-nibble `fp4_dot4`
        + aligned 128-bit `uint4` / 64-bit `bfloat4` / `uint2` loads.
  - T6: Optional `return_part_ss=True` on `moe_down_combine_rows` (exact 4-elem FMA for `mx.fast.rms_norm`).
  - T7: Optional fused `se.down_proj` (FP8 Zero-Mul-256 ALU, lean `R=2, T=2` tile) inside `moe_down_combine_rows`.
"""
import os

import mlx.core as mx

from .gather import mlx_header
from .glue import GLUE_HEADER
from .router import LPT_DEDUP_HELPER


def _fp4_pair_lut_metal() -> str:
    # Exact fp4_e2m1 values matching mlx/backend/metal/kernels/fp4.h (incl. -0.0f for nibble 8).
    vals = [
        "0.0f", "0.5f", "1.0f", "1.5f", "2.0f", "3.0f", "4.0f", "6.0f",
        "-0.0f", "-0.5f", "-1.0f", "-1.5f", "-2.0f", "-3.0f", "-4.0f", "-6.0f",
    ]
    items = [f"float2({vals[b & 0xF]}, {vals[(b >> 4) & 0xF]})" for b in range(256)]
    rows = [", ".join(items[j : j + 4]) for j in range(0, 256, 4)]
    return "constant float2 FP4_PAIR_LUT[256] = {\n  " + ",\n  ".join(rows) + "\n};\n"


_FP4_HELPERS = (
    _fp4_pair_lut_metal()
    + r"""
// Vectorized branch-free fp4_e2m1 4-nibble decode (matches mlx fp4.h bit-for-bit incl. -0.0f for nibble 8).
// Zero constant-cache serialization, zero threadgroup memory!
inline float4 fp4_decode4_alu(uint16_t w16) {
  const ushort4 nibs = ushort4(w16, w16 >> 4, w16 >> 8, w16 >> 12);
  half4 h = as_type<half4>((nibs & 7u) << 9) * half(16384.0h);
  h = as_type<half4>(as_type<ushort4>(h) | ((nibs & 8u) << 12));
  return float4(h);
}

inline float fp4_dot4_alu(uint16_t w16, bfloat4 xv) {
  const float4 d = fp4_decode4_alu(w16);
  return ((float(xv.x) * d.x + float(xv.y) * d.y) +
          float(xv.z) * d.z) + float(xv.w) * d.w;
}

inline float fp4_dot4_lut(uint16_t w16, bfloat4 xv, threadgroup const half2* lut_tg) {
  const float2 d01 = float2(lut_tg[w16 & 0xFFu]);
  const float2 d23 = float2(lut_tg[w16 >> 8]);
  return ((float(xv.x) * d01.x + float(xv.y) * d01.y) +
          float(xv.z) * d23.x) + float(xv.w) * d23.y;
}

inline float fp4_dot4_clut(uint16_t w16, bfloat4 xv) {
  const float2 d01 = FP4_PAIR_LUT[w16 & 0xFFu];
  const float2 d23 = FP4_PAIR_LUT[w16 >> 8];
  return ((float(xv.x) * d01.x + float(xv.y) * d01.y) +
          float(xv.z) * d23.x) + float(xv.w) * d23.y;
}

template <int DQ_MODE>
inline float4 fp4_decode4(uint16_t w16, threadgroup const half2* lut_tg) {
  if constexpr (DQ_MODE == 0) {
    return fp4_decode4_alu(w16);
  } else if constexpr (DQ_MODE == 1) {
    return float4(float2(lut_tg[w16 & 0xFFu]), float2(lut_tg[w16 >> 8]));
  } else {
    return float4(FP4_PAIR_LUT[w16 & 0xFFu], FP4_PAIR_LUT[w16 >> 8]);
  }
}

template <int DQ_MODE>
inline float fp4_dot4(uint16_t w16, bfloat4 xv, threadgroup const half2* lut_tg) {
  if constexpr (DQ_MODE == 0) {
    return fp4_dot4_alu(w16, xv);
  } else if constexpr (DQ_MODE == 1) {
    return fp4_dot4_lut(w16, xv, lut_tg);
  } else {
    return fp4_dot4_clut(w16, xv);
  }
}

template <int DQ_MODE>
inline float fp4_dot8_u4(uint32_t w32, uint4 xu, threadgroup const half2* lut_tg) {
  float acc = 0.0f;
  acc += fp4_dot4<DQ_MODE>(uint16_t(w32), as_type<bfloat4>(xu.xy), lut_tg);
  acc += fp4_dot4<DQ_MODE>(uint16_t(w32 >> 16), as_type<bfloat4>(xu.zw), lut_tg);
  return acc;
}

template <int DQ_MODE>
inline float fp4_dot16_u4(uint2 w64, uint4 xu0, uint4 xu1, threadgroup const half2* lut_tg) {
  float acc = 0.0f;
  acc += fp4_dot4<DQ_MODE>(uint16_t(w64.x), as_type<bfloat4>(xu0.xy), lut_tg);
  acc += fp4_dot4<DQ_MODE>(uint16_t(w64.x >> 16), as_type<bfloat4>(xu0.zw), lut_tg);
  acc += fp4_dot4<DQ_MODE>(uint16_t(w64.y), as_type<bfloat4>(xu1.xy), lut_tg);
  acc += fp4_dot4<DQ_MODE>(uint16_t(w64.y >> 16), as_type<bfloat4>(xu1.zw), lut_tg);
  return acc;
}

inline void init_fp4_tg_lut(threadgroup half2* lut_tg, uint ttid, uint n_thr) {
  for (uint i = ttid; i < 256u; i += n_thr) {
    lut_tg[i] = half2(FP4_PAIR_LUT[i]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Zero-Mul-256 exact F8_E4M3FN uchar4 * (sc * 256.0f) -> bfloat4 (matches E4M3_LUT[u8] * sc -> bf16 bit-for-bit).
inline bfloat4 fp8_decode4_bf(uchar4 pk, float sc256) {
  const ushort4 b4 = ushort4(pk);
  const ushort4 v4 = b4 & 127u;
  const ushort4 u4 = (v4 << 7) | (((v4 + 1u) >> 7) << 14) | ((b4 & 128u) << 8);
  return bfloat4(float4(as_type<half4>(u4)) * sc256);
}

// Single-pass gate+up+swiglu for cnt == 1 (T=1, R=4, FAST=true, K=4096): ~36 regs/thread, reads x ONCE!
template <typename T, int GS, int DQ_MODE>
inline void moe_gu_cnt1_singlepass(
    const device uint8_t* wg0,
    const device uint8_t* sg0,
    const device uint8_t* wu0,
    const device uint8_t* su0,
    const device T* x,
    const int pq0,
    const int out_row,
    const int NN,
    const int KK,
    const int in_w,
    const int in_g,
    const uint lane,
    threadgroup const half2* lut_tg,
    device T* out) {
  constexpr int VPT = 16;
  constexpr int BLOCK = 512;
  const int xo = (pq0 / 8) * KK + int(lane) * VPT;
  float rg[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float ru[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  const device uint8_t* wgp = wg0;
  const device uint8_t* sgp = sg0;
  const device uint8_t* wup = wu0;
  const device uint8_t* sup = su0;
  for (int k = 0; k < KK; k += BLOCK) {
    const device uint4* xp = (const device uint4*)(x + xo + k);
    const uint4 xu0 = xp[0];
    const uint4 xu1 = xp[1];
#pragma unroll
    for (int r = 0; r < 4; r++) {
      const int rr = (out_row + r < NN) ? r : 0;
      const uint2 rw_g = *(const device uint2*)(wgp + rr * in_w);
      const float sv_g = dequantize_scale<float, GS>((sgp + rr * in_g)[0]);
      rg[r] += sv_g * fp4_dot16_u4<DQ_MODE>(rw_g, xu0, xu1, lut_tg);

      const uint2 rw_u = *(const device uint2*)(wup + rr * in_w);
      const float sv_u = dequantize_scale<float, GS>((sup + rr * in_g)[0]);
      ru[r] += sv_u * fp4_dot16_u4<DQ_MODE>(rw_u, xu0, xu1, lut_tg);
    }
    wgp += 256;
    sgp += 16;
    wup += 256;
    sup += 16;
  }
#pragma unroll
  for (int r = 0; r < 4; r++) {
    const float vg = simd_sum(rg[r]);
    const float vu = simd_sum(ru[r]);
    if (lane == 0 && (out_row + r) < NN) {
      out[pq0 * NN + out_row + r] = glue_swiglu_bf16(static_cast<T>(vg), static_cast<T>(vu));
    }
  }
}

// Streaming 4-nibble hoisted-dequant gate/up pass for cnt >= 2 (FAST=true, BITS=4, K % 512 == 0).
// Never materializes float d[16] or float xt[16], and reloads pqs only on lane == 0 -> zero spills up to T=8!
template <typename T, int GS, int R_TILE, int T_TILE, bool WRITE_SWIGLU, int DQ_MODE>
inline void moe_gu_pass_stream(
    const device uint8_t* w0,
    const device uint8_t* s0,
    const device T* x,
    thread const int* xo,
    const int n_tok,
    const int r0,
    const int out_row,
    const int NN,
    const int KK,
    const int in_w,
    const int in_g,
    const uint lane,
    threadgroup const half2* lut_tg,
    thread T g_bf[T_TILE][R_TILE],
    const device uint* u_pqs,
    const int pq_base,
    device T* out) {
  constexpr int BLOCK = 512;
  float res[T_TILE][R_TILE];
#pragma unroll
  for (int t = 0; t < T_TILE; t++)
#pragma unroll
    for (int r = 0; r < R_TILE; r++) res[t][r] = 0.0f;

  const device uint8_t* wp = w0 + (long)r0 * in_w;
  const device uint8_t* sp = s0 + (long)r0 * in_g;
  for (int k = 0; k < KK; k += BLOCK) {
#pragma unroll
    for (int r = 0; r < R_TILE; r++) {
      const int rr = (out_row + r0 + r < NN) ? r : 0;
      const uint2 rw = *(const device uint2*)(wp + rr * in_w);
      const float sv = dequantize_scale<float, GS>((sp + rr * in_g)[0]);
      const uint16_t w16[4] = {
          uint16_t(rw.x), uint16_t(rw.x >> 16),
          uint16_t(rw.y), uint16_t(rw.y >> 16),
      };
      float acc[T_TILE];
#pragma unroll
      for (int t = 0; t < T_TILE; t++) acc[t] = 0.0f;
#pragma unroll
      for (int g = 0; g < 4; g++) {
        const float4 d = fp4_decode4<DQ_MODE>(w16[g], lut_tg);
#pragma unroll
        for (int t = 0; t < T_TILE; t++) {
          if (t < n_tok) {
            const bfloat4 xv = *(const device bfloat4*)(x + xo[t] + k + g * 4);
            acc[t] += ((float(xv.x) * d.x + float(xv.y) * d.y) +
                       float(xv.z) * d.z) + float(xv.w) * d.w;
          }
        }
      }
#pragma unroll
      for (int t = 0; t < T_TILE; t++) {
        if (t < n_tok) res[t][r] += sv * acc[t];
      }
    }
    wp += 256;
    sp += 16;
  }
#pragma unroll
  for (int t = 0; t < T_TILE; t++) {
    const int pq = (WRITE_SWIGLU && lane == 0 && t < n_tok) ? int(u_pqs[pq_base + t]) : 0;
#pragma unroll
    for (int r = 0; r < R_TILE; r++) {
      const float v = simd_sum(res[t][r]);
      if (lane == 0 && t < n_tok && (out_row + r0 + r) < NN) {
        if constexpr (WRITE_SWIGLU) {
          out[pq * NN + out_row + r0 + r] = glue_swiglu_bf16(g_bf[t][r], static_cast<T>(v));
        } else {
          g_bf[t][r] = static_cast<T>(v);
        }
      }
    }
  }
}

// Lean R=2, T=4 FP8 se.down_proj for 2 rows (r0 = 0 or 2), split into 2 parallel work-stealing jobs!
// Exact SN=32, TN=4 K-reduction order matching gemv_fp8_rows -> 0 bit diffs!
template <int BB, int KK>
inline void moe_se_down_fp8_rows2(
    const device uchar* se_w,
    const device float* se_sc,
    const device bfloat16_t* se_mid,
    const int out_row,
    const int r0,
    const uint lane,
    threadgroup bfloat16_t* sh_s) {
  constexpr int k_blocks = KK >> 7;
  const int row_blk = out_row >> 7;
  const device float* scp = se_sc + size_t(row_blk) * k_blocks;
  const device uchar* wp_r0 = se_w + size_t(out_row + r0) * KK + int(lane) * 4;
  const device uchar* wp_r1 = wp_r0 + KK;

  for (int tb = 0; tb < BB; tb += 4) {
    const int n_tok = min(4, BB - tb);
    float acc0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float acc1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int bn = int(lane) * 4;
    for (int i = 0; i < k_blocks; ++i) {
      const float sc256 = simd_broadcast_first(scp[i] * 256.0f);
      const bfloat4 w0 = fp8_decode4_bf(*(const device uchar4*)(wp_r0 + i * 128), sc256);
      const bfloat4 w1 = fp8_decode4_bf(*(const device uchar4*)(wp_r1 + i * 128), sc256);
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        if (t < n_tok) {
          const bfloat4 xt = *(const device bfloat4*)(se_mid + size_t(tb + t) * KK + bn);
          acc0[t] += float(w0.x) * float(xt.x);
          acc0[t] += float(w0.y) * float(xt.y);
          acc0[t] += float(w0.z) * float(xt.z);
          acc0[t] += float(w0.w) * float(xt.w);
          acc1[t] += float(w1.x) * float(xt.x);
          acc1[t] += float(w1.y) * float(xt.y);
          acc1[t] += float(w1.z) * float(xt.z);
          acc1[t] += float(w1.w) * float(xt.w);
        }
      }
      bn += 128;
    }
#pragma unroll
    for (int t = 0; t < 4; ++t) {
      if (t < n_tok) {
        float a0 = acc0[t], a1 = acc1[t];
#pragma unroll
        for (ushort sn = 16; sn >= 1; sn >>= 1) {
          a0 += simd_shuffle_down(a0, sn);
          a1 += simd_shuffle_down(a1, sn);
        }
        if (lane == 0) {
          sh_s[(tb + t) * 4 + r0] = static_cast<bfloat16_t>(a0);
          sh_s[(tb + t) * 4 + r0 + 1] = static_cast<bfloat16_t>(a1);
        }
      }
    }
  }
}
"""
)

HEADER = mlx_header() + GLUE_HEADER + _FP4_HELPERS
_KERNELS = {}
_GU_TMAX_CAP = int(os.environ.get("LINGSANTE_MOE_GU_TMAX_CAP", os.environ.get("LINGSANTE_MOE_TMAX_CAP", "4")))
_DC_TMAX_CAP = int(os.environ.get("LINGSANTE_MOE_DC_TMAX_CAP", "2"))
_ZERO_BF16 = mx.zeros((1,), dtype=mx.bfloat16)
_ZERO_U8 = mx.zeros((1,), dtype=mx.uint8)
_ZERO_F32 = mx.zeros((1,), dtype=mx.float32)


def _tmax(B, cap):
    for t in (1, 2, 4):
        if B <= t or t >= cap:
            return t
    return 4


def _cap_gz(B: int) -> int:
    # Cap grid.z to avoid launching 50-67% empty dead Z-slices on high B while keeping full GPU saturation.
    n = B * 8
    if B <= 4:
        return n
    if B <= 8:
        return min(n, 32)
    if B <= 16:
        return min(n, 48)
    return min(n, 64)


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
    for (int i = int(lane); i < NPAIR; i += 32) {
      ind_s[i] = inds[i];
    }
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
    """Deduplicate `(B, 8)` expert indices into LPT `(cnt DESC, exp ASC)` packed CSR worklist.

    Returns 5-tuple `(u_meta, u_pqs, n_hot, n_uniq, n_uniq)` so legacy callers indexing `dd[4]`
    for `n_uniq` continue working unchanged while MoE kernels bind only `(u_meta, u_pqs, n_hot, n_uniq)`.
    """
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
# 2. Gate + Up + SwiGLU Kernels (Unpacked: 1-PSO lean2/flat + True 2-PSO Split)
# ---------------------------------------------------------------------------

_GU_INLINE_SRC = r"""
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
    threadgroup half2 lut_tg[(DQ_MODE == 1) ? 256 : 1];
    if constexpr (DQ_MODE == 1) init_fp4_tg_lut(lut_tg, ttid, 32 * NSG);
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

    if constexpr (FAST && BITS == 4 && ROWS == 4) {
      if (cnt == 1) {
        moe_gu_cnt1_singlepass<T, GS, DQ_MODE>(wg0, sg0, wu0, su0, x, int(pl[0]), out_row, NN, KK, in_w, in_g, lane, lut_tg, out);
        return;
      }
    }

    for (int tb = 0; tb < cnt; tb += TMAX) {
        const int n_tok = min(TMAX, cnt - tb);
        int xo[TMAX];
        for (int t = 0; t < TMAX; t++) {
            int pq = (t < n_tok) ? (int)pl[tb + t] : q;
            xo[t] = (pq / 8) * KK + int(lane) * VPT;
        }
        T g_bf[TMAX][ROWS];
        if constexpr (FAST && BITS == 4) {
            // Pass 1: gate_proj
            {
                float res[TMAX][ROWS];
#pragma unroll
                for (int t = 0; t < TMAX; t++)
#pragma unroll
                    for (int r = 0; r < ROWS; r++) res[t][r] = 0.0f;
                const device uint8_t* wp = wg0;
                const device uint8_t* sp = sg0;
                for (int k = 0; k < KK; k += BLOCK) {
#pragma unroll
                    for (int r = 0; r < ROWS; r++) {
                        const int rr = (out_row + r < NN) ? r : 0;
                        const uint2 rw = *(const device uint2*)(wp + rr * in_w);
                        const float sv = dequantize_scale<float, GS>((sp + rr * in_g)[0]);
                        const uint16_t w16[4] = {uint16_t(rw.x), uint16_t(rw.x >> 16), uint16_t(rw.y), uint16_t(rw.y >> 16)};
                        float acc[TMAX];
#pragma unroll
                        for (int t = 0; t < TMAX; t++) acc[t] = 0.0f;
#pragma unroll
                        for (int g = 0; g < 4; g++) {
                            const float4 d = fp4_decode4<DQ_MODE>(w16[g], lut_tg);
#pragma unroll
                            for (int t = 0; t < TMAX; t++) {
                                if (t < n_tok) {
                                    const bfloat4 xv = *(const device bfloat4*)(x + xo[t] + k + g * 4);
                                    acc[t] += ((float(xv.x) * d.x + float(xv.y) * d.y) +
                                               float(xv.z) * d.z) + float(xv.w) * d.w;
                                }
                            }
                        }
#pragma unroll
                        for (int t = 0; t < TMAX; t++) {
                            if (t < n_tok) res[t][r] += sv * acc[t];
                        }
                    }
                    wp += 256;
                    sp += 16;
                }
#pragma unroll
                for (int t = 0; t < TMAX; t++)
#pragma unroll
                    for (int r = 0; r < ROWS; r++) g_bf[t][r] = static_cast<T>(simd_sum(res[t][r]));
            }
            // Pass 2: up_proj + fused swiglu
            {
                float res[TMAX][ROWS];
#pragma unroll
                for (int t = 0; t < TMAX; t++)
#pragma unroll
                    for (int r = 0; r < ROWS; r++) res[t][r] = 0.0f;
                const device uint8_t* wp = wu0;
                const device uint8_t* sp = su0;
                for (int k = 0; k < KK; k += BLOCK) {
#pragma unroll
                    for (int r = 0; r < ROWS; r++) {
                        const int rr = (out_row + r < NN) ? r : 0;
                        const uint2 rw = *(const device uint2*)(wp + rr * in_w);
                        const float sv = dequantize_scale<float, GS>((sp + rr * in_g)[0]);
                        const uint16_t w16[4] = {uint16_t(rw.x), uint16_t(rw.x >> 16), uint16_t(rw.y), uint16_t(rw.y >> 16)};
                        float acc[TMAX];
#pragma unroll
                        for (int t = 0; t < TMAX; t++) acc[t] = 0.0f;
#pragma unroll
                        for (int g = 0; g < 4; g++) {
                            const float4 d = fp4_decode4<DQ_MODE>(w16[g], lut_tg);
#pragma unroll
                            for (int t = 0; t < TMAX; t++) {
                                if (t < n_tok) {
                                    const bfloat4 xv = *(const device bfloat4*)(x + xo[t] + k + g * 4);
                                    acc[t] += ((float(xv.x) * d.x + float(xv.y) * d.y) +
                                               float(xv.z) * d.z) + float(xv.w) * d.w;
                                }
                            }
                        }
#pragma unroll
                        for (int t = 0; t < TMAX; t++) {
                            if (t < n_tok) res[t][r] += sv * acc[t];
                        }
                    }
                    wp += 256;
                    sp += 16;
                }
#pragma unroll
                for (int t = 0; t < TMAX; t++) {
                    const int pq = (lane == 0 && t < n_tok) ? int(pl[tb + t]) : 0;
#pragma unroll
                    for (int r = 0; r < ROWS; r++) {
                        const float vu = simd_sum(res[t][r]);
                        if (lane == 0 && t < n_tok && out_row + r < NN) {
                            out[pq * NN + out_row + r] = glue_swiglu_bf16(g_bf[t][r], static_cast<T>(vu));
                        }
                    }
                }
            }
        } else {
            // Generic fallback (!FAST or bits != 4)
            constexpr int KEND_SUB = FAST ? 0 : BLOCK;
            for (int pass = 0; pass < 2; ++pass) {
                thread U res[TMAX][ROWS];
                for (int t = 0; t < TMAX; t++)
                    for (int r = 0; r < ROWS; r++) res[t][r] = 0;
                const device uint8_t* wp = (pass == 0) ? wg0 : wu0;
                const device uint8_t* sp = (pass == 0) ? sg0 : su0;
                int k = 0;
                for (; k < KK - KEND_SUB; k += BLOCK) {
                    for (int t = 0; t < n_tok; t++) {
                        thread U xt[VPT];
                        load_vector<T, U, VPT>(x + xo[t] + k, xt);
                        for (int r = 0; r < ROWS; r++) {
                            if (out_row + r < NN) {
                                thread uint32_t rw[PPT];
                                const device uint32_t* wptr = (const device uint32_t*)(wp + r * in_w);
                                for (int p = 0; p < PPT; p++) rw[p] = wptr[p];
                                thread U d[VPT];
                                dequant_vals<BITS, VPT>(rw, d);
                                res[t][r] += dot_vals<BITS, VPT>(xt, d, dequantize_scale<U, GS>((sp + r * in_g)[0]));
                            }
                        }
                    }
                    wp += BLOCK * 4 / PF;
                    sp += BLOCK / GS;
                }
                if constexpr (!FAST) {
                    const int rem = clamp((int)(KK - k - int(lane) * VPT), 0, VPT);
                    if (rem > 0) {
                        for (int t = 0; t < n_tok; t++) {
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
                for (int t = 0; t < TMAX; t++) {
                    const int pq = (pass == 1 && lane == 0 && t < n_tok) ? int(pl[tb + t]) : 0;
                    for (int r = 0; r < ROWS; r++) {
                        U v = simd_sum(res[t][r]);
                        if (pass == 0) {
                            g_bf[t][r] = static_cast<T>(v);
                        } else if (lane == 0 && t < n_tok && out_row + r < NN) {
                            out[pq * NN + out_row + r] = glue_swiglu_bf16(g_bf[t][r], static_cast<T>(v));
                        }
                    }
                }
            }
        }
    }
"""

# Dedup-consuming single-PSO kernel with capped grid.z and MODE:
#   MODE == 0 ("flat"): uniform TMAX streaming 4-nibble (no cnt==1 branch)
#   MODE == 1 ("lean2"): cnt==1 single-pass (R=4, T=1) + cnt>=2 streaming (R=2, T=T_HOT) -> ~44 regs!
_GU_DEDUP_SRC = r"""
    constexpr int PF = 32 / BITS;      // 8
    constexpr int PPT = FAST ? 2 : 1;  // 2 for K=4096
    constexpr int VPT = PF * PPT;      // 16
    constexpr int SSTEP = GS / VPT;    // 2
    typedef bfloat16_t T;

    const uint3 tid = threadgroup_position_in_grid;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint ttid = sg * 32 + lane;
    threadgroup half2 lut_tg[(DQ_MODE == 1) ? 256 : 1];
    if constexpr (DQ_MODE == 1) init_fp4_tg_lut(lut_tg, ttid, 32 * NSG);
    const int out_row = tid.y * (ROWS * NSG) + sg * ROWS;
    if (out_row >= NN) return;

    const int in_w = KK * BITS / 8;
    const int in_g = KK / GS;
    const int nu = n_uniq[0];

    for (int q = int(tid.z); q < nu; q += GZ) {
      const uint meta = u_meta[q];
      const uint e = meta & 0x3FFu;
      const int off0 = int((meta >> 10) & 0xFFu);
      const int cnt = int((meta >> 18) & 0xFFu);

      const device uint8_t* wg0 = (const device uint8_t*)wg + (long)e * NN * in_w + (long)out_row * in_w + lane * PPT * 4;
      const device uint8_t* sg0 = scg + (long)e * NN * in_g + (long)out_row * in_g + lane / SSTEP;
      const device uint8_t* wu0 = (const device uint8_t*)wu + (long)e * NN * in_w + (long)out_row * in_w + lane * PPT * 4;
      const device uint8_t* su0 = scu + (long)e * NN * in_g + (long)out_row * in_g + lane / SSTEP;

      if constexpr (MODE == 1 && ROWS == 4) {
        if (cnt == 1) {
          moe_gu_cnt1_singlepass<T, GS, DQ_MODE>(wg0, sg0, wu0, su0, x, int(u_pqs[off0]), out_row, NN, KK, in_w, in_g, lane, lut_tg, out);
        } else {
          constexpr int RH = (TMAX >= 8) ? 1 : 2;
          for (int r0 = 0; r0 < 4; r0 += RH) {
            if (out_row + r0 >= NN) continue;
            for (int tb = 0; tb < cnt; tb += TMAX) {
              const int n_tok = min(TMAX, cnt - tb);
              int xo[TMAX];
#pragma unroll
              for (int t = 0; t < TMAX; t++) {
                const int pq = (t < n_tok) ? int(u_pqs[off0 + tb + t]) : int(u_pqs[off0]);
                xo[t] = (pq / 8) * KK + int(lane) * VPT;
              }
              T g_bf[TMAX][RH];
              moe_gu_pass_stream<T, GS, RH, TMAX, false, DQ_MODE>(wg0, sg0, x, xo, n_tok, r0, out_row, NN, KK, in_w, in_g, lane, lut_tg, g_bf, u_pqs, off0 + tb, out);
              moe_gu_pass_stream<T, GS, RH, TMAX, true, DQ_MODE>(wu0, su0, x, xo, n_tok, r0, out_row, NN, KK, in_w, in_g, lane, lut_tg, g_bf, u_pqs, off0 + tb, out);
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
            xo[t] = (pq / 8) * KK + int(lane) * VPT;
          }
          T g_bf[TMAX][ROWS];
          moe_gu_pass_stream<T, GS, ROWS, TMAX, false, DQ_MODE>(wg0, sg0, x, xo, n_tok, 0, out_row, NN, KK, in_w, in_g, lane, lut_tg, g_bf, u_pqs, off0 + tb, out);
          moe_gu_pass_stream<T, GS, ROWS, TMAX, true, DQ_MODE>(wu0, su0, x, xo, n_tok, 0, out_row, NN, KK, in_w, in_g, lane, lut_tg, g_bf, u_pqs, off0 + tb, out);
        }
      }
    }
"""

# True 2-PSO Split:
#   PSO 1 (cnt == 1 ONLY, u in [n_hot, n_uniq)): ultra-lean ~36-reg single-pass gate+up!
#   PSO 2 (cnt >= 2 ONLY, u in [0, n_hot)):      hot-only R_HOT x T_HOT, grid.z <= 4*B!
_GU_CNT1_SRC = r"""
    constexpr int PPT = 2;
    constexpr int VPT = 16;
    constexpr int SSTEP = GS / VPT;
    typedef bfloat16_t T;

    const uint3 tid = threadgroup_position_in_grid;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint ttid = sg * 32 + lane;
    threadgroup half2 lut_tg[(DQ_MODE == 1) ? 256 : 1];
    if constexpr (DQ_MODE == 1) init_fp4_tg_lut(lut_tg, ttid, 32 * NSG);
    const int out_row = tid.y * (4 * NSG) + sg * 4;
    if (out_row >= NN) return;

    const int nh = n_hot[0];
    const int nu = n_uniq[0];
    const int in_w = KK / 2;
    const int in_g = KK / GS;

    for (int q = nh + int(tid.z); q < nu; q += GZ) {
      const uint meta = u_meta[q];
      const uint e = meta & 0x3FFu;
      const int off0 = int((meta >> 10) & 0xFFu);
      const int pq0 = int(u_pqs[off0]);

      const device uint8_t* wg0 = (const device uint8_t*)wg + (long)e * NN * in_w + (long)out_row * in_w + lane * PPT * 4;
      const device uint8_t* sg0 = scg + (long)e * NN * in_g + (long)out_row * in_g + lane / SSTEP;
      const device uint8_t* wu0 = (const device uint8_t*)wu + (long)e * NN * in_w + (long)out_row * in_w + lane * PPT * 4;
      const device uint8_t* su0 = scu + (long)e * NN * in_g + (long)out_row * in_g + lane / SSTEP;
      moe_gu_cnt1_singlepass<T, GS, DQ_MODE>(wg0, sg0, wu0, su0, x, pq0, out_row, NN, KK, in_w, in_g, lane, lut_tg, out);
    }
"""

_GU_HOT_SRC = r"""
    constexpr int PPT = 2;
    constexpr int VPT = 16;
    constexpr int SSTEP = GS / VPT;
    typedef bfloat16_t T;

    const uint3 tid = threadgroup_position_in_grid;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint ttid = sg * 32 + lane;
    threadgroup half2 lut_tg[(DQ_MODE == 1) ? 256 : 1];
    if constexpr (DQ_MODE == 1) init_fp4_tg_lut(lut_tg, ttid, 32 * NSG);
    // Pre-converted float4 x tile: eliminates 32x bf16->fp32 conversions & shares 1 TG load across gate+up!
    threadgroup float4 x_tg[T_HOT][4][32];
    threadgroup int tok_row[T_HOT];

    const int out_row = tid.y * (R_HOT * NSG) + sg * R_HOT;
    const int nh = n_hot[0];
    const int in_w = KK / 2;
    const int in_g = KK / GS;

    for (int q = int(tid.z); q < nh; q += GZ) {
      const uint meta = u_meta[q];
      const uint e = meta & 0x3FFu;
      const int off0 = int((meta >> 10) & 0xFFu);
      const int cnt = int((meta >> 18) & 0xFFu);

      const device uint8_t* wg0 = (const device uint8_t*)wg + (long)e * NN * in_w + (long)min(out_row, NN - 1) * in_w + lane * PPT * 4;
      const device uint8_t* sg0 = scg + (long)e * NN * in_g + (long)min(out_row, NN - 1) * in_g + lane / SSTEP;
      const device uint8_t* wu0 = (const device uint8_t*)wu + (long)e * NN * in_w + (long)min(out_row, NN - 1) * in_w + lane * PPT * 4;
      const device uint8_t* su0 = scu + (long)e * NN * in_g + (long)min(out_row, NN - 1) * in_g + lane / SSTEP;

      for (int tb = 0; tb < cnt; tb += T_HOT) {
        const int n_tok = min(T_HOT, cnt - tb);
        if (int(ttid) < n_tok) {
          tok_row[ttid] = (int(u_pqs[off0 + tb + int(ttid)]) / 8) * KK;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float rg[T_HOT][R_HOT];
        float ru[T_HOT][R_HOT];
#pragma unroll
        for (int t = 0; t < T_HOT; t++)
#pragma unroll
          for (int r = 0; r < R_HOT; r++) { rg[t][r] = 0.0f; ru[t][r] = 0.0f; }

        const device uint8_t* wgp = wg0;
        const device uint8_t* sgp = sg0;
        const device uint8_t* wup = wu0;
        const device uint8_t* sup = su0;
        for (int k = 0; k < KK; k += 512) {
          // Cooperative coalesced 128-bit load + one-time bf16->fp32 conversion shared across NSG*R_HOT*2 rows!
          for (int idx = int(ttid); idx < n_tok * 64; idx += 32 * NSG) {
            const int t = idx >> 6;
            const int h = (idx >> 5) & 1;
            const int l = idx & 31;
            const uint4 raw = *(const device uint4*)(x + tok_row[t] + k + l * 16 + h * 8);
            x_tg[t][h * 2 + 0][l] = float4(as_type<bfloat4>(raw.xy));
            x_tg[t][h * 2 + 1][l] = float4(as_type<bfloat4>(raw.zw));
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);

          if (out_row < NN) {
#pragma unroll
            for (int r = 0; r < R_HOT; r++) {
              const int rr = (out_row + r < NN) ? r : 0;
              const uint2 rwg = *(const device uint2*)(wgp + rr * in_w);
              const float svg = dequantize_scale<float, GS>((sgp + rr * in_g)[0]);
              const uint2 rwu = *(const device uint2*)(wup + rr * in_w);
              const float svu = dequantize_scale<float, GS>((sup + rr * in_g)[0]);
              const uint16_t wg16[4] = {uint16_t(rwg.x), uint16_t(rwg.x >> 16), uint16_t(rwg.y), uint16_t(rwg.y >> 16)};
              const uint16_t wu16[4] = {uint16_t(rwu.x), uint16_t(rwu.x >> 16), uint16_t(rwu.y), uint16_t(rwu.y >> 16)};
              float ag[T_HOT], au[T_HOT];
#pragma unroll
              for (int t = 0; t < T_HOT; t++) { ag[t] = 0.0f; au[t] = 0.0f; }
#pragma unroll
              for (int g = 0; g < 4; g++) {
                const float4 dg = fp4_decode4<DQ_MODE>(wg16[g], lut_tg);
                const float4 du = fp4_decode4<DQ_MODE>(wu16[g], lut_tg);
#pragma unroll
                for (int t = 0; t < T_HOT; t++) {
                  if (t < n_tok) {
                    const float4 xf = x_tg[t][g][lane];
                    ag[t] += ((xf.x * dg.x + xf.y * dg.y) + xf.z * dg.z) + xf.w * dg.w;
                    au[t] += ((xf.x * du.x + xf.y * du.y) + xf.z * du.z) + xf.w * du.w;
                  }
                }
              }
#pragma unroll
              for (int t = 0; t < T_HOT; t++) {
                if (t < n_tok) {
                  rg[t][r] += svg * ag[t];
                  ru[t][r] += svu * au[t];
                }
              }
            }
          }
          wgp += 256; sgp += 16; wup += 256; sup += 16;
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (out_row < NN) {
#pragma unroll
          for (int t = 0; t < T_HOT; t++) {
            const int pq = (lane == 0 && t < n_tok) ? int(u_pqs[off0 + tb + t]) : 0;
#pragma unroll
            for (int r = 0; r < R_HOT; r++) {
              const float vg = simd_sum(rg[t][r]);
              const float vu = simd_sum(ru[t][r]);
              if (lane == 0 && t < n_tok && (out_row + r) < NN) {
                out[pq * NN + out_row + r] = glue_swiglu_bf16(static_cast<T>(vg), static_cast<T>(vu));
              }
            }
          }
        }
      }
    }
"""


# ---------------------------------------------------------------------------
# 3. Down + Combine + Shared + Residual (+ Optional T6 part_ss & T7 se_down)
# ---------------------------------------------------------------------------

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

    const uint3 tid = threadgroup_position_in_grid;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint ttid = sg * 32 + lane;
    const int out_row = tid.y * ROWS;
    const int in_w = KK * BITS / 8;
    const int in_g = KK / GS;
    threadgroup half2 lut_tg[(DQ_MODE == 1) ? 256 : 1];
"""

_DC_LOOP_CORE = r"""
            const device T* x_src = (SPLIT_H && (cnt >= 2)) ? x_hot : x;
            const device uint8_t* ws0 = (const device uint8_t*)w + (long)e * NN * in_w + (long)out_row * in_w + lane * PPT * 4;
            const device uint8_t* sc0 = scales + (long)e * NN * in_g + (long)out_row * in_g + lane / SSTEP;
            if constexpr (EXACT_BLK && !FAST && BITS == 4) {
              if (cnt == 1) {
                const int tb = 0;
                const int t = 0;
                const int pq0 = __PQ_EXPR__;
                const int xo0 = pq0 * KK + int(lane) * 8;
                float res0 = 0.0f, res1 = 0.0f, res2 = 0.0f, res3 = 0.0f;
                const device uint8_t* ws = ws0;
                const device uint8_t* sc = sc0;
                for (int k = 0; k < KK; k += 256) {
                  const uint4 xu = *(const device uint4*)(x_src + xo0 + k);
                  res0 += dequantize_scale<float, GS>(sc[0]) * fp4_dot8_u4<DQ_MODE>(*(const device uint32_t*)(ws), xu, lut_tg);
                  res1 += dequantize_scale<float, GS>(sc[in_g]) * fp4_dot8_u4<DQ_MODE>(*(const device uint32_t*)(ws + in_w), xu, lut_tg);
                  res2 += dequantize_scale<float, GS>(sc[2 * in_g]) * fp4_dot8_u4<DQ_MODE>(*(const device uint32_t*)(ws + 2 * in_w), xu, lut_tg);
                  res3 += dequantize_scale<float, GS>(sc[3 * in_g]) * fp4_dot8_u4<DQ_MODE>(*(const device uint32_t*)(ws + 3 * in_w), xu, lut_tg);
                  ws += 128;
                  sc += 8;
                }
                const float v0 = simd_sum(res0);
                const float v1 = simd_sum(res1);
                const float v2 = simd_sum(res2);
                const float v3 = simd_sum(res3);
                if (lane == 0) {
                  dn_s[pq0 * 4 + 0] = static_cast<T>(v0);
                  dn_s[pq0 * 4 + 1] = static_cast<T>(v1);
                  dn_s[pq0 * 4 + 2] = static_cast<T>(v2);
                  dn_s[pq0 * 4 + 3] = static_cast<T>(v3);
                }
              } else {
                for (int tb = 0; tb < cnt; tb += TMAX) {
                  const int n_tok = min(TMAX, cnt - tb);
                  int xo[TMAX];
#pragma unroll
                  for (int t = 0; t < TMAX; t++) {
                    const int pq = __PQ_EXPR__;
                    xo[t] = pq * KK + int(lane) * 8;
                  }
                  float result[TMAX][ROWS];
#pragma unroll
                  for (int t = 0; t < TMAX; t++)
#pragma unroll
                    for (int r = 0; r < ROWS; r++) result[t][r] = 0.0f;
                  const device uint8_t* ws = ws0;
                  const device uint8_t* sc = sc0;
                  for (int k = 0; k < KK; k += 256) {
                    uint4 xu[TMAX];
#pragma unroll
                    for (int t = 0; t < TMAX; t++) {
                      if (t < n_tok) xu[t] = *(const device uint4*)(x_src + xo[t] + k);
                    }
#pragma unroll
                    for (int r = 0; r < ROWS; r++) {
                      const uint32_t rw = *(const device uint32_t*)(ws + r * in_w);
                      const float sv = dequantize_scale<float, GS>((sc + r * in_g)[0]);
                      const float4 d01 = fp4_decode4<DQ_MODE>(uint16_t(rw), lut_tg);
                      const float4 d23 = fp4_decode4<DQ_MODE>(uint16_t(rw >> 16), lut_tg);
#pragma unroll
                      for (int t = 0; t < TMAX; t++) {
                        if (t < n_tok) {
                          const bfloat4 xv0 = as_type<bfloat4>(xu[t].xy);
                          const bfloat4 xv1 = as_type<bfloat4>(xu[t].zw);
                          float acc = 0.0f;
                          acc += ((float(xv0.x) * d01.x + float(xv0.y) * d01.y) +
                                  float(xv0.z) * d01.z) + float(xv0.w) * d01.w;
                          acc += ((float(xv1.x) * d23.x + float(xv1.y) * d23.y) +
                                  float(xv1.z) * d23.z) + float(xv1.w) * d23.w;
                          result[t][r] += sv * acc;
                        }
                      }
                    }
                    ws += 128;
                    sc += 8;
                  }
#pragma unroll
                  for (int t = 0; t < TMAX; t++) {
                    const int pq = (lane == 0 && t < n_tok) ? __PQ_EXPR__ : 0;
#pragma unroll
                    for (int r = 0; r < ROWS; r++) {
                      const float v = simd_sum(result[t][r]);
                      if (lane == 0 && t < n_tok) {
                        dn_s[pq * 4 + r] = static_cast<T>(v);
                      }
                    }
                  }
                }
              }
            } else {
              for (int tb = 0; tb < cnt; tb += TMAX) {
                const int n_tok = min(TMAX, cnt - tb);
                int xo[TMAX];
                for (int t = 0; t < TMAX; t++) {
                  const int pq = __PQ_EXPR__;
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
                  for (int t = 0; t < n_tok; t++) {
                    thread U xt[VPT];
                    load_vector<T, U, VPT>(x_src + xo[t] + k, xt);
                    for (int r = 0; r < ROWS; r++) {
                      thread uint32_t rw[PPT];
                      const device uint32_t* wp = (const device uint32_t*)(ws + r * in_w);
                      for (int p = 0; p < PPT; p++) rw[p] = wp[p];
                      thread U d[VPT];
                      dequant_vals<BITS, VPT>(rw, d);
                      result[t][r] += dot_vals<BITS, VPT>(xt, d, dequantize_scale<U, GS>((sc + r * in_g)[0]));
                    }
                  }
                  ws += BLOCK * 4 / PF;
                  sc += BLOCK / GS;
                }
                if constexpr (!(EXACT_BLK || FAST)) {
                  const int rem = clamp((int)(KK - k - int(lane) * VPT), 0, VPT);
                  if (rem > 0) {
                    for (int t = 0; t < n_tok; t++) {
                      thread U xt[VPT];
                      load_vector_safe<T, U, VPT>(x_src + xo[t] + k, xt, rem);
                      for (int r = 0; r < ROWS; r++) {
                        result[t][r] += qdot_safe<U, VPT, BITS>(ws + r * in_w, xt, dequantize_scale<U, GS>((sc + r * in_g)[0]), rem);
                      }
                    }
                  }
                }
                for (int t = 0; t < TMAX; t++) {
                  const int pq = (lane == 0 && t < n_tok) ? __PQ_EXPR__ : 0;
                  for (int r = 0; r < ROWS; r++) {
                    U v = simd_sum(result[t][r]);
                    if (lane == 0 && t < n_tok) dn_s[pq * 4 + r] = static_cast<T>(v);
                  }
                }
              }
            }
"""

_DC_TAIL_NO_SS = r"""
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int idx = int(ttid); idx < BB * 4; idx += 32 * NSG) {
      const int b = idx / 4;
      const int r = idx % 4;
      const int c = out_row + r;
      float ev[8], ww[8];
#pragma unroll
      for (int j = 0; j < 8; j++) {
        ev[j] = static_cast<float>(dn_s[(b * 8 + j) * 4 + r]);
        ww[j] = wts[b * 8 + j];
      }
      bfloat16_t val = static_cast<bfloat16_t>(glue_moe_combine8(ev, ww));
      if constexpr (HAS_SE_DOWN) {
        val = val + sh_s[b * 4 + r];
      } else if constexpr (HAS_SH) {
        val = val + shared[b * NN + c];
      }
      if constexpr (HAS_RES) val = res[b * NN + c] + val;
      out[b * NN + c] = val;
    }
"""

_DC_TAIL_WITH_SS = r"""
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Exact mx.fast.rms_norm 4-elem FMA lane: thread b handles r = 0..3 for row b (BB <= 24 < 32*NSG).
    for (int b = int(ttid); b < BB; b += 32 * NSG) {
      float ww[8];
#pragma unroll
      for (int j = 0; j < 8; j++) ww[j] = wts[b * 8 + j];
      float ss = 0.0f;
#pragma unroll
      for (int r = 0; r < 4; r++) {
        const int c = out_row + r;
        float ev[8];
#pragma unroll
        for (int j = 0; j < 8; j++) ev[j] = static_cast<float>(dn_s[(b * 8 + j) * 4 + r]);
        bfloat16_t val = static_cast<bfloat16_t>(glue_moe_combine8(ev, ww));
        if constexpr (HAS_SE_DOWN) {
          val = val + sh_s[b * 4 + r];
        } else if constexpr (HAS_SH) {
          val = val + shared[b * NN + c];
        }
        if constexpr (HAS_RES) val = res[b * NN + c] + val;
        out[b * NN + c] = val;
        const float xf = static_cast<float>(val);
        ss = fma(xf, xf, ss);
      }
      part_ss[b * (NN / 4) + int(tid.y)] = ss;
    }
"""


def _dc_tail(ret_part_ss: bool) -> str:
    return _DC_TAIL_WITH_SS if ret_part_ss else _DC_TAIL_NO_SS


def _dc_src(use_dedup: bool, has_se_down: bool = False, ret_part_ss: bool = False) -> str:
    sh_decl = "    threadgroup bfloat16_t sh_s[BB * 4];\n" if has_se_down else "    threadgroup bfloat16_t sh_s[1];\n"
    if use_dedup:
        se_branch = r"""
      if constexpr (HAS_SE_DOWN) {
        if (job < 2) {
          moe_se_down_fp8_rows2<BB, KK>(se_w, se_sc, se_mid, out_row, job * 2, lane, sh_s);
          continue;
        }
      }
      const int u = HAS_SE_DOWN ? (job - 2) : job;
"""
        mid = (
            sh_decl
            + r"""
    threadgroup bfloat16_t dn_s[NPAIR * 4];
    threadgroup atomic_uint next_u;
    if constexpr (DQ_MODE == 1) {
      for (uint i = ttid; i < 256u; i += 32u * NSG) lut_tg[i] = half2(FP4_PAIR_LUT[i]);
    }
    if constexpr (!(BB == 1 && !HAS_SE_DOWN)) {
      if (ttid == 0) atomic_store_explicit(&next_u, 0u, memory_order_relaxed);
    }
    if constexpr (DQ_MODE == 1 || !(BB == 1 && !HAS_SE_DOWN)) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int nu = n_uniq[0];
    if constexpr (BB == 1 && !HAS_SE_DOWN) {
      for (int u = int(sg); u < nu; u += NSG) {
        const uint meta = u_meta[u];
        const uint e = meta & 0x3FFu;
        const int off0 = int((meta >> 10) & 0xFFu);
        const int cnt = int((meta >> 18) & 0xFFu);
        __CORE__
      }
    } else {
      const int n_jobs = HAS_SE_DOWN ? (nu + 2) : nu;
      while (true) {
        uint j_grab = 0u;
        if (lane == 0) j_grab = atomic_fetch_add_explicit(&next_u, 1u, memory_order_relaxed);
        const int job = int(simd_broadcast_first(j_grab));
        if (job >= n_jobs) break;
"""
            + se_branch
            + r"""
        const uint meta = u_meta[u];
        const uint e = meta & 0x3FFu;
        const int off0 = int((meta >> 10) & 0xFFu);
        const int cnt = int((meta >> 18) & 0xFFu);
        __CORE__
      }
    }
"""
        ).replace("__CORE__", _DC_LOOP_CORE.replace("__PQ_EXPR__", "(tb + t < cnt) ? (int)u_pqs[off0 + tb + t] : (int)u_pqs[off0]"))
    else:
        se_inline = r"""
    if constexpr (HAS_SE_DOWN) {
      if (sg < 2) {
        moe_se_down_fp8_rows2<BB, KK>(se_w, se_sc, se_mid, out_row, int(sg) * 2, lane, sh_s);
      }
    }
"""
        mid = (
            sh_decl
            + r"""
    threadgroup uint ind_s[NPAIR];
    threadgroup uint pl_all[NSG * NPAIR];
    threadgroup bfloat16_t dn_s[NPAIR * 4];
    if constexpr (DQ_MODE == 1) {
      for (uint i = ttid; i < 256u; i += 32u * NSG) lut_tg[i] = half2(FP4_PAIR_LUT[i]);
    }
    for (int i = int(ttid); i < NPAIR; i += 32 * NSG) ind_s[i] = inds[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup uint* pl = pl_all + sg * NPAIR;
"""
            + se_inline
            + r"""
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
"""
        ).replace("__CORE__", _DC_LOOP_CORE.replace("__PQ_EXPR__", "(tb + t < cnt) ? (int)pl[tb + t] : P0"))
    return _DC_HDR + mid + _dc_tail(ret_part_ss)


# ---------------------------------------------------------------------------
# 4. T3: Runtime-on-Load K-Major Weight+Scale Repacking (0 Extra Bytes!)
# ---------------------------------------------------------------------------
# gu_packed: [E, N//4, 17408] u8 (for K=4096, GS=32, BITS=4):
#   Per (e, tile = r//4):
#     [0..511]     : gate scales, [k_blk=0..7][r=0..3][16 B]
#     [512..1023]  : up scales,   [k_blk=0..7][r=0..3][16 B]
#     [1024..17407]: 8 K-blocks x (1024 B gate_k [r=0..3][256 B] | 1024 B up_k [r=0..3][256 B])
#   Total = 17408 B = 136 * 128 B (exact 128B aligned, 0 bytes padding!).
#
# dn_packed: [E, N//4, 2720] u8 (for K=1280, GS=32, BITS=4):
#   Per (e, tile = r//4):
#     [0..159]     : scales, [k_blk=0..4][r=0..3][8 B]
#     [160..2719]  : 5 K-blocks x 512 B weights ([r=0..3][128 B])
#   Total = 2720 B (every 4 tiles = 10880 B = 85 * 128 B, 0 bytes padding!).

_PACK_GU_SRC = r"""
    // Grid: (32, 4 * ((N//4 + 3)//4), E), threadgroup: (32, 4, 1) -> 4 simdgroups pack 4 tiles/TG with uint4 scales!
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

_GU_PACKED_DEDUP_SRC = r"""
    typedef bfloat16_t T;
    const uint3 tid = threadgroup_position_in_grid;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint ttid = sg * 32 + lane;
    threadgroup half2 lut_tg[(DQ_MODE == 1) ? 256 : 1];
    if constexpr (DQ_MODE == 1) init_fp4_tg_lut(lut_tg, ttid, 32 * NSG);
    threadgroup float4 x_tg[TMAX][4][32];
    threadgroup int tok_row[TMAX];
    const int tile = int(tid.y) * NSG + int(sg);
    const int out_row = tile * 4;
    const int nu = n_uniq[0];
    const int n_tiles = NN / 4;
    const uint s_lane = lane >> 1;

    for (int q = int(tid.z); q < nu; q += GZ) {
      const uint meta = u_meta[q];
      const uint e = meta & 0x3FFu;
      const int off0 = int((meta >> 10) & 0xFFu);
      const int cnt = int((meta >> 18) & 0xFFu);

      const device uint8_t* base = gu_packed + ((long)e * n_tiles + min(tile, n_tiles - 1)) * 17408;
      if (cnt == 1) {
        if (out_row < NN) {
          const int pq0 = int(u_pqs[off0]);
          const int xo = (pq0 / 8) * 4096 + int(lane) * 16;
          float rg[4] = {0.0f, 0.0f, 0.0f, 0.0f};
          float ru[4] = {0.0f, 0.0f, 0.0f, 0.0f};
          for (int kb = 0; kb < 8; ++kb) {
            const device uint4* xp = (const device uint4*)(x + xo + kb * 512);
            const uint4 xu0 = xp[0];
            const uint4 xu1 = xp[1];
            const device uint8_t* sg_k = base + kb * 64 + s_lane;
            const device uint8_t* su_k = base + 512 + kb * 64 + s_lane;
            const device uint2* wg_k = (const device uint2*)(base + 1024 + kb * 2048) + lane;
            const device uint2* wu_k = (const device uint2*)(base + 2048 + kb * 2048) + lane;
#pragma unroll
            for (int r = 0; r < 4; ++r) {
              rg[r] += dequantize_scale<float, 32>(sg_k[r * 16]) * fp4_dot16_u4<DQ_MODE>(wg_k[r * 32], xu0, xu1, lut_tg);
              ru[r] += dequantize_scale<float, 32>(su_k[r * 16]) * fp4_dot16_u4<DQ_MODE>(wu_k[r * 32], xu0, xu1, lut_tg);
            }
          }
#pragma unroll
          for (int r = 0; r < 4; ++r) {
            const float vg = simd_sum(rg[r]);
            const float vu = simd_sum(ru[r]);
            if (lane == 0) out[pq0 * NN + out_row + r] = glue_swiglu_bf16(static_cast<T>(vg), static_cast<T>(vu));
          }
        }
      } else {
        for (int tb = 0; tb < cnt; tb += TMAX) {
          const int n_tok = min(TMAX, cnt - tb);
          if (int(ttid) < n_tok) {
            tok_row[ttid] = (int(u_pqs[off0 + tb + int(ttid)]) / 8) * 4096;
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);

          float rg[TMAX][4];
          float ru[TMAX][4];
#pragma unroll
          for (int t = 0; t < TMAX; t++)
#pragma unroll
            for (int r = 0; r < 4; r++) { rg[t][r] = 0.0f; ru[t][r] = 0.0f; }

          for (int kb = 0; kb < 8; ++kb) {
            for (int idx = int(ttid); idx < n_tok * 64; idx += 32 * NSG) {
              const int t = idx >> 6;
              const int h = (idx >> 5) & 1;
              const int l = idx & 31;
              const uint4 raw = *(const device uint4*)(x + tok_row[t] + kb * 512 + l * 16 + h * 8);
              x_tg[t][h * 2 + 0][l] = float4(as_type<bfloat4>(raw.xy));
              x_tg[t][h * 2 + 1][l] = float4(as_type<bfloat4>(raw.zw));
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (out_row < NN) {
              const device uint8_t* sg_k = base + kb * 64 + s_lane;
              const device uint8_t* su_k = base + 512 + kb * 64 + s_lane;
              const device uint2* wg_k = (const device uint2*)(base + 1024 + kb * 2048) + lane;
              const device uint2* wu_k = (const device uint2*)(base + 2048 + kb * 2048) + lane;
#pragma unroll
              for (int r = 0; r < 4; ++r) {
                const uint2 rwg = wg_k[r * 32];
                const float svg = dequantize_scale<float, 32>(sg_k[r * 16]);
                const uint2 rwu = wu_k[r * 32];
                const float svu = dequantize_scale<float, 32>(su_k[r * 16]);
                const uint16_t wg16[4] = {uint16_t(rwg.x), uint16_t(rwg.x >> 16), uint16_t(rwg.y), uint16_t(rwg.y >> 16)};
                const uint16_t wu16[4] = {uint16_t(rwu.x), uint16_t(rwu.x >> 16), uint16_t(rwu.y), uint16_t(rwu.y >> 16)};
                float ag[TMAX], au[TMAX];
#pragma unroll
                for (int t = 0; t < TMAX; t++) { ag[t] = 0.0f; au[t] = 0.0f; }
#pragma unroll
                for (int g = 0; g < 4; g++) {
                  const float4 dg = fp4_decode4<DQ_MODE>(wg16[g], lut_tg);
                  const float4 du = fp4_decode4<DQ_MODE>(wu16[g], lut_tg);
#pragma unroll
                  for (int t = 0; t < TMAX; t++) {
                    if (t < n_tok) {
                      const float4 xf = x_tg[t][g][lane];
                      ag[t] += ((xf.x * dg.x + xf.y * dg.y) + xf.z * dg.z) + xf.w * dg.w;
                      au[t] += ((xf.x * du.x + xf.y * du.y) + xf.z * du.z) + xf.w * du.w;
                    }
                  }
                }
#pragma unroll
                for (int t = 0; t < TMAX; t++) {
                  if (t < n_tok) {
                    rg[t][r] += svg * ag[t];
                    ru[t][r] += svu * au[t];
                  }
                }
              }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
          }

          if (out_row < NN) {
#pragma unroll
            for (int t = 0; t < TMAX; t++) {
              const int pq = (lane == 0 && t < n_tok) ? int(u_pqs[off0 + tb + t]) : 0;
#pragma unroll
              for (int r = 0; r < 4; r++) {
                const float vg = simd_sum(rg[t][r]);
                const float vu = simd_sum(ru[t][r]);
                if (lane == 0 && t < n_tok) {
                  out[pq * NN + out_row + r] = glue_swiglu_bf16(static_cast<T>(vg), static_cast<T>(vu));
                }
              }
            }
          }
        }
      }
    }
"""

_DC_PACKED_DEDUP_SRC = r"""
    constexpr int NPAIR = BB * 8;
    constexpr int ROWS = 4;
    typedef bfloat16_t T;

    const uint3 tid = threadgroup_position_in_grid;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint ttid = sg * 32 + lane;
    const int tile = int(tid.y);
    const int out_row = tile * 4;
    const int n_tiles = NN / 4;
    const uint s_lane = lane >> 2;

    threadgroup half2 lut_tg[(DQ_MODE == 1) ? 256 : 1];
    threadgroup bfloat16_t sh_s[HAS_SE_DOWN ? (BB * 4) : 1];
    threadgroup bfloat16_t dn_s[NPAIR * 4];
    threadgroup atomic_uint next_u;
    if constexpr (DQ_MODE == 1) {
      for (uint i = ttid; i < 256u; i += 32u * NSG) lut_tg[i] = half2(FP4_PAIR_LUT[i]);
    }
    if (ttid == 0) atomic_store_explicit(&next_u, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int nu = n_uniq[0];
    const int n_jobs = HAS_SE_DOWN ? (nu + 2) : nu;
    while (true) {
      uint j_grab = 0u;
      if (lane == 0) j_grab = atomic_fetch_add_explicit(&next_u, 1u, memory_order_relaxed);
      const int job = int(simd_broadcast_first(j_grab));
      if (job >= n_jobs) break;
      if constexpr (HAS_SE_DOWN) {
        if (job < 2) {
          moe_se_down_fp8_rows2<BB, 1280>(se_w, se_sc, se_mid, out_row, job * 2, lane, sh_s);
          continue;
        }
      }
      const int u = HAS_SE_DOWN ? (job - 2) : job;
      const uint meta = u_meta[u];
      const uint e = meta & 0x3FFu;
      const int off0 = int((meta >> 10) & 0xFFu);
      const int cnt = int((meta >> 18) & 0xFFu);
      const device uint8_t* base = dn_packed + ((long)e * n_tiles + tile) * 2720;

      if (cnt == 1) {
        const int pq0 = int(u_pqs[off0]);
        const int xo0 = pq0 * 1280 + int(lane) * 8;
        float res0 = 0.0f, res1 = 0.0f, res2 = 0.0f, res3 = 0.0f;
        for (int kb = 0; kb < 5; ++kb) {
          const uint4 xu = *(const device uint4*)(x + xo0 + kb * 256);
          const device uint8_t* sc_k = base + kb * 32 + s_lane;
          const device uint32_t* w_k = (const device uint32_t*)(base + 160 + kb * 512) + lane;
          res0 += dequantize_scale<float, 32>(sc_k[0]) * fp4_dot8_u4<DQ_MODE>(w_k[0], xu, lut_tg);
          res1 += dequantize_scale<float, 32>(sc_k[8]) * fp4_dot8_u4<DQ_MODE>(w_k[32], xu, lut_tg);
          res2 += dequantize_scale<float, 32>(sc_k[16]) * fp4_dot8_u4<DQ_MODE>(w_k[64], xu, lut_tg);
          res3 += dequantize_scale<float, 32>(sc_k[24]) * fp4_dot8_u4<DQ_MODE>(w_k[96], xu, lut_tg);
        }
        const float v0 = simd_sum(res0), v1 = simd_sum(res1), v2 = simd_sum(res2), v3 = simd_sum(res3);
        if (lane == 0) {
          dn_s[pq0 * 4 + 0] = static_cast<T>(v0);
          dn_s[pq0 * 4 + 1] = static_cast<T>(v1);
          dn_s[pq0 * 4 + 2] = static_cast<T>(v2);
          dn_s[pq0 * 4 + 3] = static_cast<T>(v3);
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
          float result[TMAX][4];
#pragma unroll
          for (int t = 0; t < TMAX; t++)
#pragma unroll
            for (int r = 0; r < 4; r++) result[t][r] = 0.0f;
          for (int kb = 0; kb < 5; ++kb) {
            uint4 xu[TMAX];
#pragma unroll
            for (int t = 0; t < TMAX; t++) {
              if (t < n_tok) xu[t] = *(const device uint4*)(x + xo[t] + kb * 256);
            }
            const device uint8_t* sc_k = base + kb * 32 + s_lane;
            const device uint32_t* w_k = (const device uint32_t*)(base + 160 + kb * 512) + lane;
#pragma unroll
            for (int r = 0; r < 4; r++) {
              const uint32_t rw = w_k[r * 32];
              const float sv = dequantize_scale<float, 32>(sc_k[r * 8]);
              const float4 d01 = fp4_decode4<DQ_MODE>(uint16_t(rw), lut_tg);
              const float4 d23 = fp4_decode4<DQ_MODE>(uint16_t(rw >> 16), lut_tg);
#pragma unroll
              for (int t = 0; t < TMAX; t++) {
                if (t < n_tok) {
                  const bfloat4 xv0 = as_type<bfloat4>(xu[t].xy);
                  const bfloat4 xv1 = as_type<bfloat4>(xu[t].zw);
                  float acc = 0.0f;
                  acc += ((float(xv0.x) * d01.x + float(xv0.y) * d01.y) +
                          float(xv0.z) * d01.z) + float(xv0.w) * d01.w;
                  acc += ((float(xv1.x) * d23.x + float(xv1.y) * d23.y) +
                          float(xv1.z) * d23.z) + float(xv1.w) * d23.w;
                  result[t][r] += sv * acc;
                }
              }
            }
          }
#pragma unroll
          for (int t = 0; t < TMAX; t++) {
            const int pq = (lane == 0 && t < n_tok) ? int(u_pqs[off0 + tb + t]) : 0;
#pragma unroll
            for (int r = 0; r < 4; r++) {
              const float v = simd_sum(result[t][r]);
              if (lane == 0 && t < n_tok) dn_s[pq * 4 + r] = static_cast<T>(v);
            }
          }
        }
      }
    }
"""


def _dc_packed_src(ret_part_ss: bool) -> str:
    return _DC_PACKED_DEDUP_SRC + _dc_tail(ret_part_ss)


# ---------------------------------------------------------------------------
# Kernel Builders & Public Python Wrappers
# ---------------------------------------------------------------------------

def _gu_kernel(use_dedup: bool = False):
    key = f"gu_d{int(use_dedup)}"
    k = _KERNELS.get(key)
    if k is None:
        in_names = (
            ["wg", "scg", "wu", "scu", "x", "u_meta", "u_pqs", "n_uniq"]
            if use_dedup
            else ["wg", "scg", "wu", "scu", "x", "inds"]
        )
        k = _KERNELS[key] = mx.fast.metal_kernel(
            name=f"bv3_moe_gu_sw_d{int(use_dedup)}",
            input_names=in_names,
            output_names=["out"],
            header=HEADER,
            source=_GU_DEDUP_SRC if use_dedup else _GU_INLINE_SRC,
            ensure_row_contiguous=True,
        )
    return k


def _gu_split_kernels():
    k1 = _KERNELS.get("gu_cnt1")
    k2 = _KERNELS.get("gu_hot")
    if k1 is None:
        k1 = _KERNELS["gu_cnt1"] = mx.fast.metal_kernel(
            name="bv3_moe_gu_cnt1",
            input_names=["wg", "scg", "wu", "scu", "x", "u_meta", "u_pqs", "n_hot", "n_uniq"],
            output_names=["out"],
            header=HEADER,
            source=_GU_CNT1_SRC,
            ensure_row_contiguous=True,
        )
        k2 = _KERNELS["gu_hot"] = mx.fast.metal_kernel(
            name="bv3_moe_gu_hot",
            input_names=["wg", "scg", "wu", "scu", "x", "u_meta", "u_pqs", "n_hot"],
            output_names=["out"],
            header=HEADER,
            source=_GU_HOT_SRC,
            ensure_row_contiguous=True,
        )
    return k1, k2


def _dc_kernel(use_dedup: bool = False, has_se_down: bool = False, ret_part_ss: bool = False):
    key = f"dc_d{int(use_dedup)}_se{int(has_se_down)}_ss{int(ret_part_ss)}"
    k = _KERNELS.get(key)
    if k is None:
        in_names = (
            ["w", "scales", "x", "x_hot", "u_meta", "u_pqs", "n_uniq", "wts", "shared", "res", "se_mid", "se_w", "se_sc"]
            if use_dedup
            else ["w", "scales", "x", "x_hot", "inds", "wts", "shared", "res", "se_mid", "se_w", "se_sc"]
        )
        out_names = ["out", "part_ss"] if ret_part_ss else ["out"]
        k = _KERNELS[key] = mx.fast.metal_kernel(
            name=f"bv3_moe_{key}",
            input_names=in_names,
            output_names=out_names,
            header=HEADER,
            source=_dc_src(use_dedup, has_se_down, ret_part_ss),
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
    dq_mode=1,
):
    """Fused gate_proj + up_proj + SwiGLU -> (B, 8, N) bfloat16 (0 bit diffs)."""
    del hoist_dq
    B, E = inds.shape
    K = x.shape[-1]
    N = wg.shape[1]
    assert 1 <= B <= 24 and E == 8 and x.shape == (B, K) and x.dtype == mx.bfloat16
    fast = (N % 8 == 0) and (K % ((32 // bits) * 2 * 32) == 0)
    rw = 4 if rows is None else int(rows)
    assert rw in (1, 2, 4) and N % rw == 0

    use_dedup = dedup is not None and fast and bits == 4
    if use_dedup:
        u_meta, u_pqs, n_hot, _n_u_dup, n_uniq = dedup
        # At B == 1 (8 distinct experts, n_hot == 0), dispatch ultra-lean cnt==1 PSO directly (~36 regs, 0 dead branches)!
        if B == 1 and rw == 4 and adaptive is None and tmax is None:
            k_cnt1, _ = _gu_split_kernels()
            nsg1 = 4 if nsg is None else int(nsg)
            gz1 = 8 if gz is None else int(gz)
            (out,) = k_cnt1(
                inputs=[wg, sg, wu, su, x, u_meta, u_pqs, n_hot, n_uniq],
                template=[("GS", group_size), ("DQ_MODE", int(dq_mode)), ("NSG", nsg1), ("GZ", gz1), ("KK", K), ("NN", N)],
                grid=(32, nsg1 * ((N + 4 * nsg1 - 1) // (4 * nsg1)), gz1),
                threadgroup=(32, nsg1, 1),
                output_shapes=[(B, E, N)],
                output_dtypes=[mx.bfloat16],
            )
            return out
        mode = (0 if (adaptive is not None and not bool(adaptive)) else 1) if rw == 4 else 0
        if tmax is None:
            tm = 1 if B == 1 else (2 if B <= 3 else 4)
        else:
            tm = int(tmax)
        if nsg is None:
            nsg = 4
        gz_val = _cap_gz(B) if gz is None else int(gz)
        (out,) = _gu_kernel(True)(
            inputs=[wg, sg, wu, su, x, u_meta, u_pqs, n_uniq],
            template=[
                ("GS", group_size), ("BITS", bits), ("FAST", fast), ("MODE", mode),
                ("DQ_MODE", int(dq_mode)), ("ROWS", rw), ("TMAX", tm), ("NSG", nsg),
                ("GZ", gz_val), ("KK", K), ("NN", N),
            ],
            grid=(32, nsg * ((N + rw * nsg - 1) // (rw * nsg)), gz_val),
            threadgroup=(32, nsg, 1),
            output_shapes=[(B, E, N)],
            output_dtypes=[mx.bfloat16],
        )
        return out

    if tmax is None:
        tm = 2 if B >= 5 else _tmax(B, _GU_TMAX_CAP)
    else:
        tm = int(tmax)
    if nsg is None:
        nsg = 4 if B >= 5 else 2
    inds_u32 = inds if inds.dtype == mx.uint32 else inds.astype(mx.uint32)
    (out,) = _gu_kernel(False)(
        inputs=[wg, sg, wu, su, x, inds_u32],
        template=[
            ("BB", B), ("GS", group_size), ("BITS", bits), ("FAST", fast),
            ("DQ_MODE", int(dq_mode)), ("ROWS", rw), ("TMAX", tm), ("NSG", nsg),
            ("KK", K), ("NN", N),
        ],
        grid=(32, nsg * ((N + rw * nsg - 1) // (rw * nsg)), B * E),
        threadgroup=(32, nsg, 1),
        output_shapes=[(B, E, N)],
        output_dtypes=[mx.bfloat16],
    )
    return out


def moe_gate_up_swiglu_split_rows(
    x,
    wg,
    sg,
    wu,
    su,
    dedup,
    *,
    group_size=32,
    t_hot=None,
    r_hot=None,
    nsg_cnt1=4,
    nsg_hot=8,
    gz_cnt1=None,
    gz_hot=None,
    dq_mode=1,
):
    """True 2-PSO zero-spill gate+up+SwiGLU: separate cnt==1 (~36 regs) & cnt>=2 (x_tg single-pass) PSOs.

    Returns `(h_cold, h_hot)` consumed directly by `moe_down_combine_rows((h_cold, h_hot), ...)` with 0 copy!
    """
    B, K = x.shape
    N = wg.shape[1]
    assert 1 <= B <= 24 and K % 512 == 0 and N % 8 == 0 and x.dtype == mx.bfloat16
    u_meta, u_pqs, n_hot, _n_u_dup, n_uniq = dedup
    k_cnt1, k_hot = _gu_split_kernels()

    th = 4 if t_hot is None else int(t_hot)
    rh = 2 if r_hot is None else int(r_hot)
    gz1 = _cap_gz(B) if gz_cnt1 is None else int(gz_cnt1)
    gz2 = min(4 * B, 24) if gz_hot is None else int(gz_hot)

    (h_cold,) = k_cnt1(
        inputs=[wg, sg, wu, su, x, u_meta, u_pqs, n_hot, n_uniq],
        template=[("GS", group_size), ("DQ_MODE", int(dq_mode)), ("NSG", nsg_cnt1), ("GZ", gz1), ("KK", K), ("NN", N)],
        grid=(32, nsg_cnt1 * ((N + 4 * nsg_cnt1 - 1) // (4 * nsg_cnt1)), gz1),
        threadgroup=(32, nsg_cnt1, 1),
        output_shapes=[(B, 8, N)],
        output_dtypes=[mx.bfloat16],
    )
    if B == 1:
        # Single token selects 8 distinct experts (n_hot == 0 guaranteed): skip k_hot dispatch completely!
        return (h_cold, h_cold)
    (h_hot,) = k_hot(
        inputs=[wg, sg, wu, su, x, u_meta, u_pqs, n_hot],
        template=[("GS", group_size), ("DQ_MODE", int(dq_mode)), ("R_HOT", rh), ("T_HOT", th), ("NSG", nsg_hot), ("GZ", gz2), ("KK", K), ("NN", N)],
        grid=(32, nsg_hot * ((N + rh * nsg_hot - 1) // (rh * nsg_hot)), gz2),
        threadgroup=(32, nsg_hot, 1),
        output_shapes=[(B, 8, N)],
        output_dtypes=[mx.bfloat16],
    )
    return (h_cold, h_hot)


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
    se_mid=None,
    se_w_fp8=None,
    se_sc=None,
    return_part_ss: bool = False,
    dq_mode=1,
):
    """Fused MoE down_proj + combine8 + optional shared/se_down + residual + optional T6 part_ss."""
    del hoist_dq
    split_h = isinstance(h, tuple)
    if split_h:
        h_cold, h_hot = h
    else:
        h_cold = h
        h_hot = h
    B, E, K = h_cold.shape
    N = wd.shape[1]
    assert 1 <= B <= 24 and E == 8 and h_cold.dtype == mx.bfloat16
    if nsg is None:
        nsg = 4
    pf = 32 // bits
    fast = (N % 8 == 0) and (K % (pf * 2 * 32) == 0)
    exact_blk = (K % (pf * 32) == 0)
    has_se_down = se_mid is not None
    if has_se_down:
        assert se_w_fp8 is not None and se_sc is not None and se_mid.shape == (B, K) and K % 128 == 0
    has_sh = (shared is not None) and not has_se_down
    has_res = residual is not None
    sh = shared if has_sh else _ZERO_BF16
    rs = residual if has_res else _ZERO_BF16
    sm = se_mid if has_se_down else _ZERO_BF16
    sw_f8 = se_w_fp8 if has_se_down else _ZERO_U8
    sc_f8 = se_sc if has_se_down else _ZERO_F32

    if tmax is None:
        tm = 1 if B == 1 else (2 if B <= 4 else 4)
    else:
        tm = int(tmax)

    use_dedup = dedup is not None
    if use_dedup:
        u_meta, u_pqs, _n_hot, _n_u_dup, n_uniq = dedup
        inputs = [wd, sd, h_cold, h_hot, u_meta, u_pqs, n_uniq, wts, sh, rs, sm, sw_f8, sc_f8]
    else:
        inds_u32 = inds if inds.dtype == mx.uint32 else inds.astype(mx.uint32)
        inputs = [wd, sd, h_cold, h_hot, inds_u32, wts, sh, rs, sm, sw_f8, sc_f8]

    out_shapes = [(B, N), (B, N // 4)] if return_part_ss else [(B, N)]
    out_dtypes = [mx.bfloat16, mx.float32] if return_part_ss else [mx.bfloat16]
    outs = _dc_kernel(use_dedup, has_se_down, return_part_ss)(
        inputs=inputs,
        template=[
            ("BB", B), ("GS", group_size), ("BITS", bits), ("FAST", fast), ("EXACT_BLK", exact_blk),
            ("DQ_MODE", int(dq_mode)), ("SPLIT_H", int(split_h)), ("TMAX", tm), ("NSG", nsg),
            ("KK", K), ("NN", N), ("HAS_SH", int(has_sh)), ("HAS_RES", int(has_res)),
            ("HAS_SE_DOWN", int(has_se_down)), ("RET_PART_SS", int(return_part_ss)),
        ],
        grid=(32, nsg * (N // 4), 1),
        threadgroup=(32, nsg, 1),
        output_shapes=out_shapes,
        output_dtypes=out_dtypes,
    )
    return (outs[0], outs[1]) if return_part_ss else outs[0]


# ---------------------------------------------------------------------------
# 5. T3: Runtime-on-Load Repacking Helpers & Packed Kernels
# ---------------------------------------------------------------------------

def _pack_gu_kernel():
    k = _KERNELS.get("pack_gu")
    if k is None:
        k = _KERNELS["pack_gu"] = mx.fast.metal_kernel(
            name="bv3_moe_pack_gu",
            input_names=["wg", "scg", "wu", "scu"],
            output_names=["out"],
            header=GLUE_HEADER,
            source=_PACK_GU_SRC,
            ensure_row_contiguous=True,
        )
    return k


def _unpack_gu_kernel():
    k = _KERNELS.get("unpack_gu")
    if k is None:
        k = _KERNELS["unpack_gu"] = mx.fast.metal_kernel(
            name="bv3_moe_unpack_gu",
            input_names=["packed"],
            output_names=["wg", "scg", "wu", "scu"],
            header=GLUE_HEADER,
            source=_UNPACK_GU_SRC,
            ensure_row_contiguous=True,
        )
    return k


def _pack_dn_kernel():
    k = _KERNELS.get("pack_dn")
    if k is None:
        k = _KERNELS["pack_dn"] = mx.fast.metal_kernel(
            name="bv3_moe_pack_dn",
            input_names=["w", "sc"],
            output_names=["out"],
            header=GLUE_HEADER,
            source=_PACK_DN_SRC,
            ensure_row_contiguous=True,
        )
    return k


def _unpack_dn_kernel():
    k = _KERNELS.get("unpack_dn")
    if k is None:
        k = _KERNELS["unpack_dn"] = mx.fast.metal_kernel(
            name="bv3_moe_unpack_dn",
            input_names=["packed"],
            output_names=["w", "sc"],
            header=GLUE_HEADER,
            source=_UNPACK_DN_SRC,
            ensure_row_contiguous=True,
        )
    return k


def _gu_packed_kernel():
    k = _KERNELS.get("gu_packed")
    if k is None:
        k = _KERNELS["gu_packed"] = mx.fast.metal_kernel(
            name="bv3_moe_gu_packed_d1",
            input_names=["gu_packed", "x", "u_meta", "u_pqs", "n_uniq"],
            output_names=["out"],
            header=HEADER,
            source=_GU_PACKED_DEDUP_SRC,
            ensure_row_contiguous=True,
        )
    return k


def _dc_packed_kernel(has_se_down: bool = False, ret_part_ss: bool = False):
    key = f"dc_packed_se{int(has_se_down)}_ss{int(ret_part_ss)}"
    k = _KERNELS.get(key)
    if k is None:
        out_names = ["out", "part_ss"] if ret_part_ss else ["out"]
        k = _KERNELS[key] = mx.fast.metal_kernel(
            name=f"bv3_moe_{key}",
            input_names=["dn_packed", "x", "u_meta", "u_pqs", "n_uniq", "wts", "shared", "res", "se_mid", "se_w", "se_sc"],
            output_names=out_names,
            header=HEADER,
            source=_dc_packed_src(ret_part_ss),
            ensure_row_contiguous=True,
        )
    return k


def pack_moe_gu_weights(wg: mx.array, scg: mx.array, wu: mx.array, scu: mx.array) -> mx.array:
    """Lossless runtime repack of MXFP4 K=4096 (wg, scg, wu, scu) -> (E, N//4, 17408) u8 (0B padding)."""
    E, N, K8 = wg.shape
    assert K8 == 512 and N % 4 == 0 and scg.shape == (E, N, 128) and wu.shape == wg.shape and scu.shape == scg.shape
    nt = N // 4
    (out,) = _pack_gu_kernel()(
        inputs=[wg, scg, wu, scu],
        template=[("NN", N)],
        grid=(32, 4 * ((nt + 3) // 4), E),
        threadgroup=(32, 4, 1),
        output_shapes=[(E, nt, 17408)],
        output_dtypes=[mx.uint8],
    )
    return out


def unpack_moe_gu_weights(gu_packed: mx.array) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Lossless inverse of `pack_moe_gu_weights` for SwitchGLU prefill fallback."""
    E, NT, BPT = gu_packed.shape
    assert BPT == 17408
    N = NT * 4
    return tuple(
        _unpack_gu_kernel()(
            inputs=[gu_packed],
            template=[("NN", N)],
            grid=(32, 4 * ((NT + 3) // 4), E),
            threadgroup=(32, 4, 1),
            output_shapes=[(E, N, 512), (E, N, 128), (E, N, 512), (E, N, 128)],
            output_dtypes=[mx.uint32, mx.uint8, mx.uint32, mx.uint8],
        )
    )


def pack_moe_dn_weights(wd: mx.array, scd: mx.array) -> mx.array:
    """Lossless runtime repack of MXFP4 K=1280 (wd, scd) -> (E, N//4, 2720) u8 (0B padding)."""
    E, N, K8 = wd.shape
    assert K8 == 160 and N % 4 == 0 and scd.shape == (E, N, 40)
    nt = N // 4
    (out,) = _pack_dn_kernel()(
        inputs=[wd, scd],
        template=[("NN", N)],
        grid=(32, 4 * ((nt + 3) // 4), E),
        threadgroup=(32, 4, 1),
        output_shapes=[(E, nt, 2720)],
        output_dtypes=[mx.uint8],
    )
    return out


def unpack_moe_dn_weights(dn_packed: mx.array) -> tuple[mx.array, mx.array]:
    """Lossless inverse of `pack_moe_dn_weights` for SwitchGLU prefill fallback."""
    E, NT, BPT = dn_packed.shape
    assert BPT == 2720
    N = NT * 4
    return tuple(
        _unpack_dn_kernel()(
            inputs=[dn_packed],
            template=[("NN", N)],
            grid=(32, 4 * ((NT + 3) // 4), E),
            threadgroup=(32, 4, 1),
            output_shapes=[(E, N, 160), (E, N, 40)],
            output_dtypes=[mx.uint32, mx.uint8],
        )
    )


def moe_gate_up_swiglu_packed_rows(x, gu_packed, dedup, *, tmax=None, nsg=4, gz=None, dq_mode=1):
    """K-major tile-packed gate+up+SwiGLU (1 buffer vs 4, 0B pad, 0 bit diffs)."""
    B, K = x.shape
    assert 1 <= B <= 24 and K == 4096 and x.dtype == mx.bfloat16
    N = gu_packed.shape[1] * 4
    tm = 4 if tmax is None else int(tmax)
    gz_val = _cap_gz(B) if gz is None else int(gz)
    u_meta, u_pqs, _n_hot, _n_u_dup, n_uniq = dedup
    (out,) = _gu_packed_kernel()(
        inputs=[gu_packed, x, u_meta, u_pqs, n_uniq],
        template=[("DQ_MODE", int(dq_mode)), ("TMAX", tm), ("NSG", nsg), ("GZ", gz_val), ("NN", N)],
        grid=(32, nsg * ((N // 4 + nsg - 1) // nsg), gz_val),
        threadgroup=(32, nsg, 1),
        output_shapes=[(B, 8, N)],
        output_dtypes=[mx.bfloat16],
    )
    return out


def moe_down_combine_packed_rows(
    h,
    dn_packed,
    dedup,
    wts,
    *,
    shared=None,
    residual=None,
    nsg=4,
    tmax=None,
    se_mid=None,
    se_w_fp8=None,
    se_sc=None,
    return_part_ss: bool = False,
    dq_mode=1,
):
    """K-major tile-packed down_proj + combine8 + optional se_down + residual + optional T6 part_ss."""
    B, E, K = h.shape
    assert 1 <= B <= 24 and E == 8 and K == 1280 and h.dtype == mx.bfloat16
    N = dn_packed.shape[1] * 4
    has_se_down = se_mid is not None
    has_sh = (shared is not None) and not has_se_down
    has_res = residual is not None
    sh = shared if has_sh else _ZERO_BF16
    rs = residual if has_res else _ZERO_BF16
    sm = se_mid if has_se_down else _ZERO_BF16
    sw_f8 = se_w_fp8 if has_se_down else _ZERO_U8
    sc_f8 = se_sc if has_se_down else _ZERO_F32
    tm = (1 if B == 1 else (2 if B <= 4 else 4)) if tmax is None else int(tmax)
    u_meta, u_pqs, _n_hot, _n_u_dup, n_uniq = dedup
    out_shapes = [(B, N), (B, N // 4)] if return_part_ss else [(B, N)]
    out_dtypes = [mx.bfloat16, mx.float32] if return_part_ss else [mx.bfloat16]
    outs = _dc_packed_kernel(has_se_down, return_part_ss)(
        inputs=[dn_packed, h, u_meta, u_pqs, n_uniq, wts, sh, rs, sm, sw_f8, sc_f8],
        template=[
            ("BB", B), ("DQ_MODE", int(dq_mode)), ("TMAX", tm), ("NSG", nsg), ("NN", N),
            ("HAS_SH", int(has_sh)), ("HAS_RES", int(has_res)),
            ("HAS_SE_DOWN", int(has_se_down)), ("RET_PART_SS", int(return_part_ss)),
        ],
        grid=(32, nsg * (N // 4), 1),
        threadgroup=(32, nsg, 1),
        output_shapes=out_shapes,
        output_dtypes=out_dtypes,
    )
    return (outs[0], outs[1]) if return_part_ss else outs[0]
