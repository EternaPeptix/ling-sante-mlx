"""Bit-exact Wave 4+ MLA decode/verify helpers & kernels for BailingMLA (9 of 72 LingSante layers).

1. `mla_prep_kv_rope_rows`: fuses optional q_nope split (zero-copy via `skip_q_nope=True`),
   q_rope split + optional `* attn_scale`, kv_latent/k_rope split, `kv_a_layernorm` (RMSNorm 512),
   and per-row interleaved (`traditional=True`) RoPE (base=6e6) in ONE launch.
2. `mla_multilinear_[fp8]_rows`: weights-read-once MultiLinear with `bfloat4` loads, Zero-Mul-256
   E4M3 ALU + `bfloat4 wf[TM]` (`bsg=1, tm=2` at `B=9..16`), optional `in_stride=192` zero-copy
   read from `q_full`, optional fused `* attn_scale` (`embed_q`), and optional fused head gate
   (`unembed_out`).
3. `mla_q_a_norm_from_partss_rows`: T6 `q_a_layernorm` (`N=1536`) consumer using producer `part_ss`.
4. `mla_sdpa_rows` / `mla_verify_layer_rows` / `mla_verify_varlen_rows`: bit-exact causal MLA SDPA.
"""
import inspect
import math

import mlx.core as mx

from .gemv import gemv_rows, gemv_rows_multi, gemv_rows_res
from .gemv_fp8 import gemv_lin, gemv_lin_multi, gemv_lin_res
from .glue import GLUE_HEADER, flit

_KERNELS = {}
_ZERO_BF16 = None
_SCALE_CACHE = {}


def _zero_bf16(dtype=mx.bfloat16) -> mx.array:
    global _ZERO_BF16
    if dtype == mx.bfloat16:
        if _ZERO_BF16 is None:
            _ZERO_BF16 = mx.zeros((1,), dtype=mx.bfloat16)
        return _ZERO_BF16
    return mx.zeros((1,), dtype=dtype)


def _scale_arr(scale: float, dtype=mx.bfloat16) -> mx.array:
    key = (float(scale), dtype)
    arr = _SCALE_CACHE.get(key)
    if arr is None:
        arr = mx.array([float(scale)], dtype=dtype)
        _SCALE_CACHE[key] = arr
    return arr


_PREP_SRC = r"""
    constexpr int NH = 64;
    constexpr int NOPE_D = 128;
    constexpr int ROPE_D = 64;
    constexpr int QHEAD_D = NOPE_D + ROPE_D;  // 192
    constexpr int Q_DIM = NH * QHEAD_D;       // 12288
    constexpr int KV_D = 512;
    constexpr int KVA_DIM = KV_D + ROPE_D;    // 576

    const uint lane = thread_index_in_simdgroup;
    const uint y = threadgroup_position_in_grid.y;  // 0..63 = q head, 64 = kv_a
    const uint b = threadgroup_position_in_grid.z;

    const float L = ROPE_SCALE * static_cast<float>(offsets[b]);
    const float d = static_cast<float>(lane) / 32.0f;
    const float inv_freq = metal::exp2(-d * LOG2_BASE);
    const float theta = L * inv_freq;
    const float ct = metal::fast::cos(theta);
    const float st = metal::fast::sin(theta);

    if (y < NH) {
      const device T* qh = q_full + size_t(b) * Q_DIM + size_t(y) * QHEAD_D;
      device T* qr = q_rope + (size_t(b) * NH + y) * ROPE_D;
      if constexpr (!SKIP_Q_NOPE) {
        device T* qn = q_nope + (size_t(b) * NH + y) * NOPE_D;
        *(device vec<T, 4>*)(qn + int(lane) * 4) = *(const device vec<T, 4>*)(qh + int(lane) * 4);
      }
      int r1 = 2 * int(lane);
      float x1 = static_cast<float>(qh[NOPE_D + r1]);
      float x2 = static_cast<float>(qh[NOPE_D + r1 + 1]);
      T v1 = static_cast<T>(x1 * ct - x2 * st);
      T v2 = static_cast<T>(x1 * st + x2 * ct);
      if constexpr (HAS_ATTN_SCALE) {
        const T sc = attn_scale[0];
        v1 = v1 * sc;
        v2 = v2 * sc;
      }
      qr[r1]     = v1;
      qr[r1 + 1] = v2;
    } else {
      const device T* kvh = kv_a_full + size_t(b) * KVA_DIM;
      device T* kl = kv_latent + size_t(b) * KV_D;
      device T* kr = k_rope + size_t(b) * ROPE_D;
      float inv = glue_rms_inv_sg<KV_D, true>(kvh, KV_EPS, lane);
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        int base = (g * 32 + int(lane)) * 4;
        vec<T, 4> xv = *(const device vec<T, 4>*)(kvh + base);
        vec<WT, 4> wv = *(const device vec<WT, 4>*)(kv_ln_w + base);
        vec<T, 4> ov;
        ov.x = glue_rms_apply<T, WT>(static_cast<float>(xv.x), inv, wv.x);
        ov.y = glue_rms_apply<T, WT>(static_cast<float>(xv.y), inv, wv.y);
        ov.z = glue_rms_apply<T, WT>(static_cast<float>(xv.z), inv, wv.z);
        ov.w = glue_rms_apply<T, WT>(static_cast<float>(xv.w), inv, wv.w);
        *(device vec<T, 4>*)(kl + base) = ov;
      }
      int r1 = 2 * int(lane);
      float x1 = static_cast<float>(kvh[KV_D + r1]);
      float x2 = static_cast<float>(kvh[KV_D + r1 + 1]);
      kr[r1]     = static_cast<T>(x1 * ct - x2 * st);
      kr[r1 + 1] = static_cast<T>(x1 * st + x2 * ct);
    }
"""

_GATE_SRC = r"""
    constexpr int NH = 64;
    constexpr int VD = 128;
    const uint lane = thread_index_in_simdgroup;
    const uint h = threadgroup_position_in_grid.y;
    const uint b = threadgroup_position_in_grid.z;
    const size_t off = (size_t(b) * NH + h) * VD + int(lane) * 4;
    const T g = static_cast<T>(glue_sigmoid(static_cast<float>(g_raw[size_t(b) * NH + h])));
    vec<T, 4> xv = *(const device vec<T, 4>*)(x + off);
    *(device vec<T, 4>*)(y + off) = vec<T, 4>(xv.x * g, xv.y * g, xv.z * g, xv.w * g);
"""

_QA_NORM_PARTSS_SRC = r"""
    // T6: Bit-exact N=1536 RMSNorm from producer-emitted part_ss (single launch, 1 TG/row).
    constexpr int N = 1536;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint b = threadgroup_position_in_grid.z;
    const uint ttid = sg * 32 + lane;  // 0..127

    constexpr int P_PER_ROW = (N / 128) * (128 / BLOCKM);
    threadgroup float tg_inv;
    if (sg == 0) {
      float inv = glue_rms_inv_from_partss<N, BLOCKM>(part_ss + size_t(b) * P_PER_ROW, QA_EPS, lane);
      if (lane == 0) tg_inv = inv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = tg_inv;

    const device T* xr = x + size_t(b) * N;
    device T* yr = y + size_t(b) * N;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
      int base = (i * 128 + int(ttid)) * 4;
      vec<T, 4> xv = *(const device vec<T, 4>*)(xr + base);
      vec<WT, 4> wv = *(const device vec<WT, 4>*)(w + base);
      vec<T, 4> ov;
      ov.x = glue_rms_apply<T, WT>(static_cast<float>(xv.x), inv, wv.x);
      ov.y = glue_rms_apply<T, WT>(static_cast<float>(xv.y), inv, wv.y);
      ov.z = glue_rms_apply<T, WT>(static_cast<float>(xv.z), inv, wv.z);
      ov.w = glue_rms_apply<T, WT>(static_cast<float>(xv.w), inv, wv.w);
      *(device vec<T, 4>*)(yr + base) = ov;
    }
"""

_CAUSAL_ADD_SOFTMAX_SRC = r"""
    // Exact MLX softmax_single_row (N_READS=4, fast::exp, 1/sum) + fused (qk + pe) + causal zero-pad (S_end <= 4096).
    constexpr int N_READS = 4;
    const uint lid = thread_position_in_threadgroup.x;
    const uint lsize = threads_per_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint nsg = lsize / 32;
    const uint row = threadgroup_position_in_grid.z;  // t * 64 + h
    const int t = int(row / 64);
    const int valid_len = S_base + t + 1;

    const device bfloat16_t* qk_r = qk + size_t(row) * S_end;
    const device bfloat16_t* pe_r = pe + size_t(row) * S_end;
    device bfloat16_t* pr_r = pr + size_t(row) * S_end;

    float ld[N_READS];
    float maxval = -INFINITY;
    const int base = int(lid) * N_READS;
#pragma unroll
    for (int i = 0; i < N_READS; ++i) {
      int idx = base + i;
      if (idx < valid_len) {
        bfloat16_t sc = qk_r[idx] + pe_r[idx];
        float v = static_cast<float>(sc);
        ld[i] = v;
        maxval = metal::max(maxval, v);
      } else {
        ld[i] = -INFINITY;
      }
    }

    threadgroup float local_max[32];
    maxval = simd_max(maxval);
    if (lane == 0) local_max[sg] = maxval;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
      float m = (lane < nsg) ? local_max[lane] : -INFINITY;
      m = simd_max(m);
      if (lane == 0) local_max[0] = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    maxval = local_max[0];

    float normalizer = 0.0f;
#pragma unroll
    for (int i = 0; i < N_READS; ++i) {
      int idx = base + i;
      float ex = (idx < valid_len) ? metal::fast::exp(ld[i] - maxval) : 0.0f;
      ld[i] = ex;
      normalizer += ex;
    }

    threadgroup float local_norm[32];
    normalizer = simd_sum(normalizer);
    if (lane == 0) local_norm[sg] = normalizer;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
      float s = (lane < nsg) ? local_norm[lane] : 0.0f;
      s = simd_sum(s);
      if (lane == 0) local_norm[0] = 1.0f / s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_norm = local_norm[0];

#pragma unroll
    for (int i = 0; i < N_READS; ++i) {
      int idx = base + i;
      if (idx < S_end) {
        pr_r[idx] = (idx < valid_len) ? bfloat16_t(ld[i] * inv_norm) : bfloat16_t(0.0f);
      }
    }
"""

_ADD_SOFTMAX_STRIDED_SRC = r"""
    // Single-token fused (qk[h, :valid_len] + pe[t, h, :valid_len]) + exact softmax_single_row (valid_len <= 4096).
    constexpr int N_READS = 4;
    const uint lid = thread_position_in_threadgroup.x;
    const uint lsize = threads_per_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint nsg = lsize / 32;
    const uint h = threadgroup_position_in_grid.z;

    const device bfloat16_t* qk_r = qk + size_t(h) * valid_len;
    const device bfloat16_t* pe_r = pe + size_t(t_idx * 64 + int(h)) * pe_stride;
    device bfloat16_t* pr_r = pr + size_t(h) * valid_len;

    float ld[N_READS];
    float maxval = -INFINITY;
    const int base = int(lid) * N_READS;
#pragma unroll
    for (int i = 0; i < N_READS; ++i) {
      int idx = base + i;
      if (idx < valid_len) {
        bfloat16_t sc = qk_r[idx] + pe_r[idx];
        float v = static_cast<float>(sc);
        ld[i] = v;
        maxval = metal::max(maxval, v);
      } else {
        ld[i] = -INFINITY;
      }
    }

    threadgroup float local_max[32];
    maxval = simd_max(maxval);
    if (lane == 0) local_max[sg] = maxval;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
      float m = (lane < nsg) ? local_max[lane] : -INFINITY;
      m = simd_max(m);
      if (lane == 0) local_max[0] = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    maxval = local_max[0];

    float normalizer = 0.0f;
#pragma unroll
    for (int i = 0; i < N_READS; ++i) {
      int idx = base + i;
      float ex = (idx < valid_len) ? metal::fast::exp(ld[i] - maxval) : 0.0f;
      ld[i] = ex;
      normalizer += ex;
    }

    threadgroup float local_norm[32];
    normalizer = simd_sum(normalizer);
    if (lane == 0) local_norm[sg] = normalizer;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
      float s = (lane < nsg) ? local_norm[lane] : 0.0f;
      s = simd_sum(s);
      if (lane == 0) local_norm[0] = 1.0f / s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_norm = local_norm[0];

#pragma unroll
    for (int i = 0; i < N_READS; ++i) {
      int idx = base + i;
      if (idx < valid_len) {
        pr_r[idx] = bfloat16_t(ld[i] * inv_norm);
      }
    }
"""


def _add_softmax_multi_src(lc: int) -> str:
    sel = [
        "    const device bfloat16_t* qk_r = qk0 + size_t(h) * valid_len;",
        "    device bfloat16_t* pr_r = pr0 + size_t(h) * valid_len;",
    ]
    for i in range(1, lc):
        kw = "if" if i == 1 else "else if"
        sel.append(
            f"    {kw} (t == {i}) {{ qk_r = qk{i} + size_t(h) * valid_len; pr_r = pr{i} + size_t(h) * valid_len; }}"
        )
    return (
        r"""
    constexpr int N_READS = 4;
    const uint lid = thread_position_in_threadgroup.x;
    const uint lsize = threads_per_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint nsg = lsize / 32;
    const uint row = threadgroup_position_in_grid.z;
    const int t = int(row / 64);
    const uint h = row % 64;
    const int valid_len = S_base + t + 1;
    const device bfloat16_t* pe_r = pe + size_t(row) * pe_stride;
"""
        + "\n".join(sel)
        + r"""

    float ld[N_READS];
    float maxval = -INFINITY;
    const int base = int(lid) * N_READS;
#pragma unroll
    for (int i = 0; i < N_READS; ++i) {
      int idx = base + i;
      if (idx < valid_len) {
        bfloat16_t sc = qk_r[idx] + pe_r[idx];
        float v = static_cast<float>(sc);
        ld[i] = v;
        maxval = metal::max(maxval, v);
      } else {
        ld[i] = -INFINITY;
      }
    }

    threadgroup float local_max[32];
    maxval = simd_max(maxval);
    if (lane == 0) local_max[sg] = maxval;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
      float m = (lane < nsg) ? local_max[lane] : -INFINITY;
      m = simd_max(m);
      if (lane == 0) local_max[0] = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    maxval = local_max[0];

    float normalizer = 0.0f;
#pragma unroll
    for (int i = 0; i < N_READS; ++i) {
      int idx = base + i;
      float ex = (idx < valid_len) ? metal::fast::exp(ld[i] - maxval) : 0.0f;
      ld[i] = ex;
      normalizer += ex;
    }

    threadgroup float local_norm[32];
    normalizer = simd_sum(normalizer);
    if (lane == 0) local_norm[sg] = normalizer;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
      float s = (lane < nsg) ? local_norm[lane] : 0.0f;
      s = simd_sum(s);
      if (lane == 0) local_norm[0] = 1.0f / s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_norm = local_norm[0];

#pragma unroll
    for (int i = 0; i < N_READS; ++i) {
      int idx = base + i;
      if (idx < valid_len) {
        pr_r[idx] = bfloat16_t(ld[i] * inv_norm);
      }
    }
"""
    )

_ML_SRC = r"""
    // Multi-head batched GEMV: out[b, h, 0, n] = x[b, h, 0, :KK] @ w[h, n, :].T
    // Uses exact MLX (bm=4, bn=1, sm=1, sn=32, tm=TM, tn=4) K-reduction order.
    constexpr int BM = 4;
    constexpr int TN = 4;
    constexpr int blockM = BM * TM;
    constexpr int blockN = 32 * TN;  // 128

    const uint lane = thread_index_in_simdgroup;
    const uint sg_all = simdgroup_index_in_threadgroup;
    const int sg = int(sg_all) / BSG;
    const int b_sg = int(sg_all) % BSG;
    const int b0 = b_sg * B_SUB;
    const uint tgx = threadgroup_position_in_grid.x;
    const uint h = threadgroup_position_in_grid.z;

    const int out_row = int(tgx) * blockM + sg * TM;
    if (out_row >= NN) return;

    const device T* mat = w + size_t(h) * NN * KK + size_t(out_row) * KK;
    int bn = int(lane) * TN;
    float res[B_SUB][TM];
#pragma unroll
    for (int bi = 0; bi < B_SUB; ++bi)
#pragma unroll
      for (int tm = 0; tm < TM; ++tm)
        res[bi][tm] = 0.0f;

    constexpr int n_iter = KK / blockN;
    for (int i = 0; i < n_iter; ++i) {
      vec<T, 4> inter[TM];
#pragma unroll
      for (int tm = 0; tm < TM; ++tm) {
        inter[tm] = *(const device vec<T, 4>*)(mat + size_t(tm) * KK + bn);
      }
#pragma unroll
      for (int bi = 0; bi < B_SUB; ++bi) {
        const int b = b0 + bi;
        if (b < BB) {
          float4 vc = static_cast<float4>(
              *(const device vec<T, 4>*)(x + (size_t(b) * NH + h) * IN_STRIDE + bn));
#pragma unroll
          for (int tm = 0; tm < TM; ++tm) {
            float4 wf = static_cast<float4>(inter[tm]);
            res[bi][tm] += wf.x * vc.x;
            res[bi][tm] += wf.y * vc.y;
            res[bi][tm] += wf.z * vc.z;
            res[bi][tm] += wf.w * vc.w;
          }
        }
      }
      bn += blockN;
    }

    const T sc_t = HAS_SCALE ? scale_arr[0] : static_cast<T>(1.0f);
#pragma unroll
    for (int bi = 0; bi < B_SUB; ++bi) {
      const int b = b0 + bi;
      if (b < BB) {
        T gate = static_cast<T>(1.0f);
        if constexpr (HAS_GATE) {
          gate = static_cast<T>(glue_sigmoid(static_cast<float>(g_raw[size_t(b) * NH + h])));
        }
#pragma unroll
        for (int tm = 0; tm < TM; ++tm) {
#pragma unroll
          for (ushort sn = 16; sn >= 1; sn >>= 1) {
            res[bi][tm] += simd_shuffle_down(res[bi][tm], sn);
          }
          if (lane == 0) {
            T v = static_cast<T>(res[bi][tm]);
            if constexpr (HAS_SCALE) v = v * sc_t;
            if constexpr (HAS_GATE) v = v * gate;
            out[(size_t(b) * NH + h) * NN + out_row + tm] = v;
          }
        }
      }
    }
"""

_ML_FP8_SRC = r"""
    // Low-TG contiguous looped FP8 MultiLinear (cuts 4096/1024 TGs -> <=512 TGs, hoists x at KK==128).
    constexpr int BM = 4;
    constexpr int TN = 4;
    constexpr int blockM = BM * TM;
    constexpr int blockN = 32 * TN;  // 128

    const uint lane = thread_index_in_simdgroup;
    const uint sg_all = simdgroup_index_in_threadgroup;
    const int sg = int(sg_all) / BSG;
    const int b_sg = int(sg_all) % BSG;
    const int b0 = b_sg * B_SUB;
    const uint tgx = threadgroup_position_in_grid.x;
    const uint n_tgx = threadgroups_per_grid.x;
    const uint h = threadgroup_position_in_grid.z;

    const int n_tiles = NN / blockM;
    const int tiles_per_tg = (n_tiles + int(n_tgx) - 1) / int(n_tgx);
    const int tile_beg = int(tgx) * tiles_per_tg;
    const int tile_end = min(n_tiles, tile_beg + tiles_per_tg);

    const device float* sc_h = sc_inv + size_t(h) * 4;
    const T sc_t = HAS_SCALE ? scale_arr[0] : static_cast<T>(1.0f);

    constexpr int n_iter = KK / blockN;
    for (int tile = tile_beg; tile < tile_end; ++tile) {
      const int out_row = tile * blockM + sg * TM;
      const device uchar* mat = w + size_t(h) * NN * KK + size_t(out_row) * KK;
      int bn = int(lane) * TN;
      float res[B_SUB][TM];
#pragma unroll
      for (int bi = 0; bi < B_SUB; ++bi)
#pragma unroll
        for (int tm = 0; tm < TM; ++tm)
          res[bi][tm] = 0.0f;

      float sc_n256 = 0.0f;
      if constexpr (SCALE_BY_N) {
        sc_n256 = simd_broadcast_first(sc_h[out_row >> 7] * 256.0f);
      }

      for (int i = 0; i < n_iter; ++i) {
        const float sc256 = SCALE_BY_N ? sc_n256 : simd_broadcast_first(sc_h[i] * 256.0f);
        bfloat4 wf[TM];
#pragma unroll
        for (int tm = 0; tm < TM; ++tm) {
          const uchar4 pk = *(const device uchar4*)(mat + size_t(tm) * KK + bn);
          wf[tm] = glue_e4m3x4_to_bf4(pk, sc256);
        }
#pragma unroll
        for (int bi = 0; bi < B_SUB; ++bi) {
          const int b = b0 + bi;
          if (b < BB) {
            float4 vc = static_cast<float4>(
                *(const device vec<T, 4>*)(x + (size_t(b) * NH + h) * IN_STRIDE + bn));
#pragma unroll
            for (int tm = 0; tm < TM; ++tm) {
              float4 w4 = float4(wf[tm]);
              res[bi][tm] += w4.x * vc.x;
              res[bi][tm] += w4.y * vc.y;
              res[bi][tm] += w4.z * vc.z;
              res[bi][tm] += w4.w * vc.w;
            }
          }
        }
        bn += blockN;
      }

#pragma unroll
      for (int bi = 0; bi < B_SUB; ++bi) {
        const int b = b0 + bi;
        if (b < BB) {
          T gate = static_cast<T>(1.0f);
          if constexpr (HAS_GATE) {
            gate = static_cast<T>(glue_sigmoid(static_cast<float>(g_raw[size_t(b) * NH + h])));
          }
#pragma unroll
          for (int tm = 0; tm < TM; ++tm) {
#pragma unroll
            for (ushort sn = 16; sn >= 1; sn >>= 1) {
              res[bi][tm] += simd_shuffle_down(res[bi][tm], sn);
            }
            if (lane == 0) {
              T v = static_cast<T>(res[bi][tm]);
              if constexpr (HAS_SCALE) v = v * sc_t;
              if constexpr (HAS_GATE) v = v * gate;
              out[(size_t(b) * NH + h) * NN + out_row + tm] = v;
            }
          }
        }
      }
    }
"""


def _prep_kernel(kv_eps: float, log2_base: float, rope_scale: float):
    key = ("prep", float(kv_eps), float(log2_base), float(rope_scale))
    k = _KERNELS.get(key)
    if k is None:
        hdr = GLUE_HEADER + f"""
#define KV_EPS {flit(kv_eps)}
#define LOG2_BASE {flit(log2_base)}
#define ROPE_SCALE {flit(rope_scale)}
"""
        k = _KERNELS[key] = mx.fast.metal_kernel(
            name="bv3_mla_prep_kv_rope",
            input_names=["q_full", "kv_a_full", "kv_ln_w", "offsets", "attn_scale"],
            output_names=["q_nope", "q_rope", "kv_latent", "k_rope"],
            header=hdr,
            source=_PREP_SRC,
            ensure_row_contiguous=True,
        )
    return k


def _gate_kernel():
    k = _KERNELS.get("gate")
    if k is None:
        k = _KERNELS["gate"] = mx.fast.metal_kernel(
            name="bv3_mla_gate",
            input_names=["x", "g_raw"],
            output_names=["y"],
            header=GLUE_HEADER,
            source=_GATE_SRC,
            ensure_row_contiguous=True,
        )
    return k


def _qa_norm_partss_kernel(qa_eps: float):
    key = ("qa_norm_partss", float(qa_eps))
    k = _KERNELS.get(key)
    if k is None:
        hdr = GLUE_HEADER + f"\n#define QA_EPS {flit(qa_eps)}\n"
        k = _KERNELS[key] = mx.fast.metal_kernel(
            name="bv3_mla_qa_norm_partss",
            input_names=["x", "part_ss", "w"],
            output_names=["y"],
            header=hdr,
            source=_QA_NORM_PARTSS_SRC,
            ensure_row_contiguous=True,
        )
    return k


def _causal_add_softmax_kernel():
    k = _KERNELS.get("causal_add_sm")
    if k is None:
        k = _KERNELS["causal_add_sm"] = mx.fast.metal_kernel(
            name="bv3_mla_causal_add_softmax",
            input_names=["qk", "pe", "S_end", "S_base"],
            output_names=["pr"],
            header=GLUE_HEADER,
            source=_CAUSAL_ADD_SOFTMAX_SRC,
            ensure_row_contiguous=True,
        )
    return k


def _add_softmax_strided_kernel():
    k = _KERNELS.get("add_sm_strided")
    if k is None:
        k = _KERNELS["add_sm_strided"] = mx.fast.metal_kernel(
            name="bv3_mla_add_softmax_strided",
            input_names=["qk", "pe", "valid_len", "pe_stride", "t_idx"],
            output_names=["pr"],
            header=GLUE_HEADER,
            source=_ADD_SOFTMAX_STRIDED_SRC,
            ensure_row_contiguous=True,
        )
    return k


def _add_softmax_multi_kernel(lc: int):
    key = ("add_sm_multi", lc)
    k = _KERNELS.get(key)
    if k is None:
        k = _KERNELS[key] = mx.fast.metal_kernel(
            name=f"bv3_mla_add_softmax_multi_{lc}",
            input_names=[f"qk{i}" for i in range(lc)] + ["pe", "S_base", "pe_stride"],
            output_names=[f"pr{i}" for i in range(lc)],
            header=GLUE_HEADER,
            source=_add_softmax_multi_src(lc),
            ensure_row_contiguous=True,
        )
    return k


def _ml_kernel():
    k = _KERNELS.get("ml")
    if k is None:
        k = _KERNELS["ml"] = mx.fast.metal_kernel(
            name="bv3_mla_multilinear",
            input_names=["x", "w", "g_raw", "scale_arr"],
            output_names=["out"],
            header=GLUE_HEADER,
            source=_ML_SRC,
            ensure_row_contiguous=True,
        )
    return k


def _ml_fp8_kernel():
    k = _KERNELS.get("ml_fp8")
    if k is None:
        k = _KERNELS["ml_fp8"] = mx.fast.metal_kernel(
            name="bv3_mla_multilinear_fp8",
            input_names=["x", "w", "sc_inv", "g_raw", "scale_arr"],
            output_names=["out"],
            header=GLUE_HEADER,
            source=_ML_FP8_SRC,
            ensure_row_contiguous=True,
        )
    return k


def _ml_cfg(B: int):
    # With bfloat4 wf[TM] (4 regs at TM=2) + Zero-Mul-256 ALU, B=9..16 fits in bsg=1, tm=2
    # (36 regs total, 1x weight reads instead of 2x with bsg=2).
    if B <= 4:
        return 1, B, 4
    if B <= 16:
        return 1, B, 2
    return 2, (B + 1) // 2, 1


def mla_q_a_norm_from_partss_rows(
    q_a: mx.array, part_ss: mx.array, norm_w: mx.array, eps: float, *, blockm: int
) -> mx.array:
    """T6: Bit-exact q_a_layernorm (N=1536) using producer-emitted `part_ss`."""
    B, N = q_a.shape
    assert N == 1536 and blockm in (4, 8, 16, 32, 64, 128)
    (y,) = _qa_norm_partss_kernel(eps)(
        inputs=[q_a, part_ss, norm_w],
        template=[("T", q_a.dtype), ("WT", norm_w.dtype), ("BLOCKM", int(blockm))],
        grid=(32, 4, B),
        threadgroup=(32, 4, 1),
        output_shapes=[(B, N)],
        output_dtypes=[q_a.dtype],
    )
    return y


def mla_multilinear_rows(
    x: mx.array,
    w: mx.array,
    g_raw: mx.array | None = None,
    *,
    in_stride: int = 0,
    attn_scale: float | None = None,
    cfg=None,
) -> mx.array:
    """Weights-read-once MultiLinear for (B, H, 1, K) @ (H, N, K).T, optional zero-copy stride, scale, gate."""
    H, N, K = w.shape
    if x.ndim == 2:
        assert in_stride > 0 and x.shape[1] == H * in_stride
        B = x.shape[0]
        stride = in_stride
    else:
        B, xH, L, xK = x.shape
        assert L == 1 and xH == H and xK == K
        stride = in_stride if in_stride > 0 else K
    assert 1 <= B <= 24 and K % 128 == 0 and N % 8 == 0 and stride % 4 == 0
    has_gate = g_raw is not None
    has_scale = attn_scale is not None
    gr = g_raw if has_gate else _zero_bf16(x.dtype)
    sc_a = _scale_arr(attn_scale, x.dtype) if has_scale else _zero_bf16(x.dtype)
    bsg, b_sub, tm = _ml_cfg(B) if cfg is None else cfg
    nt = N // (4 * tm)
    (out,) = _ml_kernel()(
        inputs=[x, w, gr, sc_a],
        template=[
            ("T", x.dtype),
            ("BB", B),
            ("BSG", bsg),
            ("B_SUB", b_sub),
            ("NH", H),
            ("NN", N),
            ("KK", K),
            ("IN_STRIDE", stride),
            ("TM", tm),
            ("HAS_GATE", int(has_gate)),
            ("HAS_SCALE", int(has_scale)),
        ],
        grid=(nt * 32, 4 * bsg, H),
        threadgroup=(32, 4 * bsg, 1),
        output_shapes=[(B, N * H) if has_gate else (B, H, 1, N)],
        output_dtypes=[x.dtype],
    )
    return out


def mla_multilinear_fp8_rows(
    x: mx.array,
    w_fp8: mx.array,
    w_sc: mx.array,
    *,
    scale_by_n: bool,
    g_raw: mx.array | None = None,
    in_stride: int = 0,
    attn_scale: float | None = None,
    cfg=None,
) -> mx.array:
    """Bit-exact FP8 MultiLinear reading uint8 weights + float32 scale_inv directly."""
    H, N, K = w_fp8.shape
    if x.ndim == 2:
        assert in_stride > 0 and x.shape[1] == H * in_stride
        B = x.shape[0]
        stride = in_stride
    else:
        B, xH, L, xK = x.shape
        assert L == 1 and xH == H and xK == K
        stride = in_stride if in_stride > 0 else K
    assert 1 <= B <= 24 and K % 128 == 0 and N % 8 == 0 and stride % 4 == 0
    has_gate = g_raw is not None
    has_scale = attn_scale is not None
    gr = g_raw if has_gate else _zero_bf16(x.dtype)
    sc_a = _scale_arr(attn_scale, x.dtype) if has_scale else _zero_bf16(x.dtype)
    bsg, b_sub, tm = _ml_cfg(B) if cfg is None else cfg
    nt = N // (4 * tm)
    gx = min(nt, 8)
    (out,) = _ml_fp8_kernel()(
        inputs=[x, w_fp8, w_sc, gr, sc_a],
        template=[
            ("T", x.dtype),
            ("BB", B),
            ("BSG", bsg),
            ("B_SUB", b_sub),
            ("NH", H),
            ("NN", N),
            ("KK", K),
            ("IN_STRIDE", stride),
            ("TM", tm),
            ("SCALE_BY_N", int(bool(scale_by_n))),
            ("HAS_GATE", int(has_gate)),
            ("HAS_SCALE", int(has_scale)),
        ],
        grid=(gx * 32, 4 * bsg, H),
        threadgroup=(32, 4 * bsg, 1),
        output_shapes=[(B, N * H) if has_gate else (B, H, 1, N)],
        output_dtypes=[x.dtype],
    )
    return out


def mla_multilinear_lin(
    x: mx.array,
    mod,
    g_raw: mx.array | None = None,
    *,
    in_stride: int = 0,
    attn_scale: float | None = None,
) -> mx.array:
    if hasattr(mod, "weight_fp8"):
        return mla_multilinear_fp8_rows(
            x,
            mod.weight_fp8,
            mod.weight_scale_inv,
            scale_by_n=mod.scale_by_n,
            g_raw=g_raw,
            in_stride=in_stride,
            attn_scale=attn_scale,
        )
    return mla_multilinear_rows(x, mod.weight, g_raw=g_raw, in_stride=in_stride, attn_scale=attn_scale)


def mla_prep_kv_rope_rows(
    q_full: mx.array,
    kv_a_full: mx.array,
    kv_ln_w: mx.array,
    offsets,
    *,
    kv_eps: float = 1e-6,
    base: float = 6000000.0,
    scale: float = 1.0,
    skip_q_nope: bool = False,
    attn_scale: float | None = None,
):
    B = q_full.shape[0]
    if not isinstance(offsets, mx.array):
        offsets = mx.array([int(offsets)] * B, dtype=mx.int32)
    elif offsets.dtype != mx.int32:
        offsets = offsets.astype(mx.int32)
    has_as = attn_scale is not None
    as_arr = _scale_arr(attn_scale, q_full.dtype) if has_as else _zero_bf16(q_full.dtype)
    qn_shape = (1,) if skip_q_nope else (B, 64, 1, 128)
    return _prep_kernel(kv_eps, math.log2(float(base)), float(scale))(
        inputs=[q_full, kv_a_full, kv_ln_w, offsets, as_arr],
        template=[
            ("T", q_full.dtype),
            ("WT", kv_ln_w.dtype),
            ("SKIP_Q_NOPE", int(skip_q_nope)),
            ("HAS_ATTN_SCALE", int(has_as)),
        ],
        grid=(32, 65, B),
        threadgroup=(32, 1, 1),
        output_shapes=[qn_shape, (B, 64, 1, 64), (B, 1, 1, 512), (B, 1, 1, 64)],
        output_dtypes=[q_full.dtype] * 4,
    )


def mla_gate_rows(x: mx.array, g_raw: mx.array) -> mx.array:
    B = x.shape[0]
    (y,) = _gate_kernel()(
        inputs=[x, g_raw],
        template=[("T", x.dtype)],
        grid=(32, 64, B),
        threadgroup=(32, 1, 1),
        output_shapes=[(B, 8192)],
        output_dtypes=[x.dtype],
    )
    return y


def _get_left_starts(cache, B: int):
    lp = getattr(cache, "left_padding", None)
    if lp is None:
        return [0] * B
    lp_id = id(lp)
    cached = getattr(cache, "_lingsante_lp_cache", None)
    if cached is not None and cached[0] == lp_id:
        return cached[1]
    vals = lp.tolist()
    cache._lingsante_lp_cache = (lp_id, vals)
    return vals


# Check once if gemv_lin_multi / gemv_lin / gemv_lin_res support T6 `part_ss` / `return_part_ss`
_GEMV_LIN_MULTI_PARAMS = set(inspect.signature(gemv_lin_multi).parameters)
_GEMV_LIN_PARAMS = set(inspect.signature(gemv_lin).parameters)
_GEMV_LIN_RES_PARAMS = set(inspect.signature(gemv_lin_res).parameters)


def mla_causal_add_softmax_rows(qk: mx.array, pe: mx.array, S_base: int) -> mx.array:
    """Bit-exact fused `(qk + pe)` + causal mask + `mx.softmax(..., precise=True)` for `S_end <= 4096`."""
    L64, S_end = qk.shape
    L = L64 // 64
    assert L64 == L * 64 and S_end <= 4096
    n_thr = min(1024, ((S_end + 3) // 4 + 31) // 32 * 32)
    (pr,) = _causal_add_softmax_kernel()(
        inputs=[qk, pe, int(S_end), int(S_base)],
        grid=(n_thr, 1, L64),
        threadgroup=(n_thr, 1, 1),
        output_shapes=[(L, 64, 1, S_end)],
        output_dtypes=[mx.bfloat16],
    )
    return pr


def mla_add_softmax_strided_rows(qk_t: mx.array, pe_bat: mx.array, t: int, valid_len: int, pe_stride: int) -> mx.array:
    """Bit-exact single-token fused `(qk_t + pe_bat[t, :, :valid_len])` + `mx.softmax` (`valid_len <= 4096`)."""
    n_thr = min(1024, ((valid_len + 3) // 4 + 31) // 32 * 32)
    (pr,) = _add_softmax_strided_kernel()(
        inputs=[qk_t.reshape(64, valid_len), pe_bat, int(valid_len), int(pe_stride), int(t)],
        grid=(n_thr, 1, 64),
        threadgroup=(n_thr, 1, 1),
        output_shapes=[(1, 64, 1, valid_len)],
        output_dtypes=[mx.bfloat16],
    )
    return pr


def mla_add_softmax_multi_rows(qk_list: list[mx.array], pe_bat: mx.array, S_base: int, pe_stride: int) -> list[mx.array]:
    """Single-launch fused `(qk_t + pe_bat[t])` + exact `softmax_single_row` across `2 <= lc <= 8` tokens."""
    lc = len(qk_list)
    S_end = S_base + lc
    assert 2 <= lc <= 8 and S_end <= 4096
    n_thr = min(1024, ((S_end + 3) // 4 + 31) // 32 * 32)
    outs = _add_softmax_multi_kernel(lc)(
        inputs=[qk_list[t].reshape(64, S_base + t + 1) for t in range(lc)] + [pe_bat, int(S_base), int(pe_stride)],
        grid=(n_thr, 1, lc * 64),
        threadgroup=(n_thr, 1, 1),
        output_shapes=[(1, 64, 1, S_base + t + 1) for t in range(lc)],
        output_dtypes=[mx.bfloat16] * lc,
    )
    return list(outs)


def _mla_q_kv_g_and_q_full(
    h_in: mx.array,
    attn,
    *,
    part_ss: mx.array | None = None,
    norm_w: mx.array | None = None,
    norm_eps: float = 1e-6,
):
    """Run q_a/kv_a/g -> q_a_layernorm -> q_b_proj, forwarding T6 `part_ss` when provided."""
    if part_ss is not None and "part_ss" in _GEMV_LIN_MULTI_PARAMS:
        q_a, kv_a_full, g_raw = gemv_lin_multi(
            h_in,
            [attn.q_a_proj, attn.kv_a_proj_with_mqa, attn.g_proj],
            part_ss=part_ss,
            norm_w=norm_w,
            norm_eps=norm_eps,
        )
    else:
        q_a, kv_a_full, g_raw = gemv_lin_multi(
            h_in, [attn.q_a_proj, attn.kv_a_proj_with_mqa, attn.g_proj]
        )
    q_a_norm = mx.fast.rms_norm(q_a, attn.q_a_layernorm.weight, attn.q_a_layernorm.eps)
    q_full = gemv_lin(q_a_norm, attn.q_b_proj)
    return q_full, kv_a_full, g_raw


def _mla_dense_res(gated: mx.array, dense_mod, res: mx.array, *, return_part_ss: bool = False):
    if return_part_ss and "return_part_ss" in _GEMV_LIN_RES_PARAMS:
        return gemv_lin_res(gated, dense_mod, res, return_part_ss=True)
    return gemv_lin_res(gated, dense_mod, res)


def mla_sdpa_rows(
    q_nope: mx.array,
    q_rope: mx.array,
    kv_latent_all: mx.array,
    k_rope_all: mx.array,
    scale: float | None,
    starts,
):
    """Exact MLA SDPA handling unpadded KVCache and heterogeneous left-padded BatchKVCache.

    When `scale is None`, `q_nope` and `q_rope` are already pre-multiplied by `bfloat16(attn.scale)`.
    """
    B = q_nope.shape[0]
    if B == 1:
        s = starts[0]
        kl = kv_latent_all if s == 0 else kv_latent_all[:, :, s:, :]
        kr = k_rope_all if s == 0 else k_rope_all[:, :, s:, :]
        qr = q_rope if scale is None else (q_rope * scale)
        qn = q_nope if scale is None else (q_nope * scale)
        pe = qr @ kr.swapaxes(-1, -2)
        qk = qn @ kl.swapaxes(-1, -2)
        S = kl.shape[2]
        if S <= 4096 and qk.dtype == mx.bfloat16:
            probs = mla_add_softmax_strided_rows(qk, pe.reshape(64, S), 0, S, S)
        else:
            probs = mx.softmax(qk + pe, axis=-1, precise=True)
        return probs @ kl

    # Batch > 1: run per-row on unpadded slices so softmax denominator & matmul M=64 match B=1 bit-for-bit.
    outs = []
    for b in range(B):
        s = starts[b]
        kl_b = kv_latent_all[b : b + 1] if s == 0 else kv_latent_all[b : b + 1, :, s:, :]
        kr_b = k_rope_all[b : b + 1] if s == 0 else k_rope_all[b : b + 1, :, s:, :]
        qr_b = q_rope[b : b + 1] if scale is None else (q_rope[b : b + 1] * scale)
        qn_b = q_nope[b : b + 1] if scale is None else (q_nope[b : b + 1] * scale)
        pe_b = qr_b @ kr_b.swapaxes(-1, -2)
        qk_b = qn_b @ kl_b.swapaxes(-1, -2)
        S_b = kl_b.shape[2]
        if S_b <= 4096 and qk_b.dtype == mx.bfloat16:
            pr_b = mla_add_softmax_strided_rows(qk_b, pe_b.reshape(64, S_b), 0, S_b, S_b)
        else:
            pr_b = mx.softmax(qk_b + pe_b, axis=-1, precise=True)
        outs.append(pr_b @ kl_b)
    return mx.concatenate(outs, axis=0)


def mla_sdpa_verify_rows(
    kl_buf: mx.array,
    kr_buf: mx.array,
    q_nope: mx.array,
    q_rope: mx.array,
    start_b: int,
    start_k: int,
    base_end: int,
    lc: int,
    scale: float | None = None,
) -> list[mx.array]:
    """Bit-exact batched-Steel causal MLA SDPA for a single stream of `lc` verify tokens.

    - Always batches `PE` (`(lc*64, 64) @ kr.T`, 0 diffs for all `S` and `lc=1..24`, 1x `kr` read).
    - For `S_base >= 512`, also batches `QK` (`(lc*64, 512) @ kl.T`, 0 diffs for all `S_base >= 512`
      and `lc=1..24`, 1x `kl` read) + single-launch fused `mla_causal_add_softmax_rows` (`S_end <= 4096`).
    - For `S_base < 512`, fuses `(qk_t + pe_t)` + exact softmax in 1 strided launch per token.
    """
    outs = []
    S_base = base_end - start_k
    S_end = S_base + lc
    if lc == 1:
        end = base_end + 1
        kl_t = kl_buf[:, :, start_k:end, :]
        kr_t = kr_buf[:, :, start_k:end, :]
        qr_b = q_rope[start_b : start_b + 1] if scale is None else (q_rope[start_b : start_b + 1] * scale)
        qn_b = q_nope[start_b : start_b + 1] if scale is None else (q_nope[start_b : start_b + 1] * scale)
        pe = qr_b @ kr_t.swapaxes(-1, -2)
        qk = qn_b @ kl_t.swapaxes(-1, -2)
        if S_end <= 4096 and qk.dtype == mx.bfloat16:
            pr = mla_add_softmax_strided_rows(qk, pe.reshape(64, S_end), 0, S_end, S_end)
        else:
            pr = mx.softmax(qk + pe, axis=-1, precise=True)
        outs.append(pr @ kl_t)
        return outs

    qr_c = q_rope[start_b : start_b + lc] if scale is None else (q_rope[start_b : start_b + lc] * scale)
    qn_c = q_nope[start_b : start_b + lc] if scale is None else (q_nope[start_b : start_b + lc] * scale)
    kr_full = kr_buf[0, 0, start_k : base_end + lc, :]
    pe_bat = qr_c.reshape(lc * 64, 64) @ kr_full.T  # (lc*64, S_end), 0 diffs for all S & lc

    if S_base >= 512 and qn_c.dtype == mx.bfloat16:
        kl_full = kl_buf[0, 0, start_k : base_end + lc, :]
        qk_bat = qn_c.reshape(lc * 64, 512) @ kl_full.T  # (lc*64, S_end), 0 diffs for all S_base >= 512
        if S_end <= 4096:
            pr_bat = mla_causal_add_softmax_rows(qk_bat, pe_bat, S_base)
            for t in range(lc):
                valid_len = S_base + t + 1
                end = base_end + t + 1
                outs.append(pr_bat[t : t + 1, :, :, :valid_len] @ kl_buf[:, :, start_k:end, :])
        else:
            sc_bat = (qk_bat + pe_bat).reshape(lc, 64, 1, S_end)
            for t in range(lc):
                valid_len = S_base + t + 1
                end = base_end + t + 1
                pr_t = mx.softmax(sc_bat[t : t + 1, :, :, :valid_len], axis=-1, precise=True)
                outs.append(pr_t @ kl_buf[:, :, start_k:end, :])
        return outs

    if 2 <= lc <= 8 and S_end <= 4096 and qn_c.dtype == mx.bfloat16:
        qks = []
        kls_t = []
        for t in range(lc):
            end = base_end + t + 1
            kl_t = kl_buf[:, :, start_k:end, :]
            kls_t.append(kl_t)
            qks.append(qn_c[t : t + 1] @ kl_t.swapaxes(-1, -2))
        prs = mla_add_softmax_multi_rows(qks, pe_bat, S_base, S_end)
        for t in range(lc):
            outs.append(prs[t] @ kls_t[t])
        return outs

    for t in range(lc):
        valid_len = S_base + t + 1
        end = base_end + t + 1
        kl_t = kl_buf[:, :, start_k:end, :]
        qk_t = qn_c[t : t + 1] @ kl_t.swapaxes(-1, -2)
        if valid_len <= 4096 and qk_t.dtype == mx.bfloat16:
            pr_t = mla_add_softmax_strided_rows(qk_t, pe_bat, t, valid_len, S_end)
        else:
            pe_t = pe_bat[t * 64 : (t + 1) * 64, :valid_len].reshape(1, 64, 1, valid_len)
            pr_t = mx.softmax(qk_t + pe_t, axis=-1, precise=True)
        outs.append(pr_t @ kl_t)
    return outs


def mla_layer_rows(
    h_in: mx.array,
    res: mx.array,
    attn,
    cache,
    *,
    part_ss: mx.array | None = None,
    norm_w: mx.array | None = None,
    norm_eps: float = 1e-6,
    return_part_ss: bool = False,
):
    """Full fused single-token (B <= 4) BailingMLA layer with residual add -> h_post (B, 4096)."""
    B = h_in.shape[0]
    q_full, kv_a_full, g_raw = _mla_q_kv_g_and_q_full(
        h_in, attn, part_ss=part_ss, norm_w=norm_w, norm_eps=norm_eps
    )
    _, q_rope, kv_latent, k_rope = mla_prep_kv_rope_rows(
        q_full,
        kv_a_full,
        attn.kv_a_layernorm.weight,
        cache.offset,
        kv_eps=attn.kv_a_layernorm.eps,
        base=float(getattr(attn.rope, "base", 6000000.0)),
        scale=float(getattr(attn.rope, "scale", 1.0)),
        skip_q_nope=True,
        attn_scale=attn.scale,
    )
    q_nope = mla_multilinear_lin(q_full, attn.embed_q, in_stride=192, attn_scale=attn.scale)
    kl_all, kr_all = cache.update_and_fetch(kv_latent, k_rope)
    starts = _get_left_starts(cache, B)
    out = mla_sdpa_rows(q_nope, q_rope, kl_all, kr_all, None, starts)
    gated = mla_multilinear_lin(out, attn.unembed_out, g_raw=g_raw)
    return _mla_dense_res(gated, attn.dense, res, return_part_ss=return_part_ss)


def _mla_verify_proj_prep(
    h_in: mx.array,
    attn,
    offsets: mx.array,
    *,
    part_ss: mx.array | None = None,
    norm_w: mx.array | None = None,
    norm_eps: float = 1e-6,
):
    q_full, kv_a_full, g_raw = _mla_q_kv_g_and_q_full(
        h_in, attn, part_ss=part_ss, norm_w=norm_w, norm_eps=norm_eps
    )
    _, q_rope, kv_latent, k_rope = mla_prep_kv_rope_rows(
        q_full,
        kv_a_full,
        attn.kv_a_layernorm.weight,
        offsets,
        kv_eps=attn.kv_a_layernorm.eps,
        base=float(getattr(attn.rope, "base", 6000000.0)),
        scale=float(getattr(attn.rope, "scale", 1.0)),
        skip_q_nope=True,
        attn_scale=attn.scale,
    )
    q_nope = mla_multilinear_lin(q_full, attn.embed_q, in_stride=192, attn_scale=attn.scale)
    return q_nope, q_rope, kv_latent, k_rope, g_raw


def _mla_append_causal_slice(
    outs: list,
    kl_buf: mx.array,
    kr_buf: mx.array,
    q_nope: mx.array,
    q_rope: mx.array,
    start_b: int,
    start_k: int,
    base_end: int,
    lc: int,
    scale: float | None,
):
    outs.extend(
        mla_sdpa_verify_rows(kl_buf, kr_buf, q_nope, q_rope, start_b, start_k, base_end, lc, scale)
    )


def mla_verify_layer_rows(
    h_in: mx.array,
    res: mx.array,
    attn,
    cache,
    *,
    C: int,
    L: int,
    offsets: mx.array | None = None,
    part_ss: mx.array | None = None,
    norm_w: mx.array | None = None,
    norm_eps: float = 1e-6,
    return_part_ss: bool = False,
):
    """Bit-exact causal multi-token (C * L <= 24) BailingMLA layer supporting KVCache, list[KVCache], BatchKVCache."""
    B = C * L
    assert h_in.shape[0] == B and 1 <= B <= 24

    if isinstance(cache, list):
        assert len(cache) == C
        offs_before = [int(c_i.offset) for c_i in cache]
        if offsets is None:
            offsets = mx.array([o + t for o in offs_before for t in range(L)], dtype=mx.int32)
    elif hasattr(cache, "left_padding"):
        if offsets is None:
            offsets = (cache.offset[:, None] + mx.arange(L, dtype=mx.int32)[None, :]).reshape(-1)
    else:
        assert C == 1
        off0 = int(cache.offset)
        if offsets is None:
            offsets = mx.arange(off0, off0 + L, dtype=mx.int32)

    q_nope, q_rope, kv_latent, k_rope, g_raw = _mla_verify_proj_prep(
        h_in, attn, offsets, part_ss=part_ss, norm_w=norm_w, norm_eps=norm_eps
    )

    kl_in = kv_latent.reshape(C, 1, L, 512)
    kr_in = k_rope.reshape(C, 1, L, 64)
    outs = []

    if isinstance(cache, list):
        for c in range(C):
            off_c = offs_before[c]
            kl_c, kr_c = cache[c].update_and_fetch(kl_in[c : c + 1], kr_in[c : c + 1])
            _mla_append_causal_slice(outs, kl_c, kr_c, q_nope, q_rope, c * L, 0, off_c, L, None)
    elif hasattr(cache, "left_padding"):
        starts = _get_left_starts(cache, C)
        kl_all, kr_all = cache.update_and_fetch(kl_in, kr_in)
        base_end = kl_all.shape[2] - L
        for c in range(C):
            _mla_append_causal_slice(
                outs,
                kl_all[c : c + 1],
                kr_all[c : c + 1],
                q_nope,
                q_rope,
                c * L,
                starts[c],
                base_end,
                L,
                None,
            )
    else:
        kl_all, kr_all = cache.update_and_fetch(kl_in, kr_in)
        _mla_append_causal_slice(outs, kl_all, kr_all, q_nope, q_rope, 0, 0, off0, L, None)

    out = mx.concatenate(outs, axis=0)
    gated = mla_multilinear_lin(out, attn.unembed_out, g_raw=g_raw)
    return _mla_dense_res(gated, attn.dense, res, return_part_ss=return_part_ss)


def mla_verify_varlen_rows(
    h_in: mx.array,
    res: mx.array,
    attn,
    caches: list,
    Ls: list[int],
    *,
    offsets: mx.array | None = None,
    part_ss: mx.array | None = None,
    norm_w: mx.array | None = None,
    norm_eps: float = 1e-6,
    return_part_ss: bool = False,
):
    """Bit-exact variable-length causal (sum(Ls) <= 24) BailingMLA layer over unpadded `list[KVCache]`."""
    C = len(Ls)
    B = sum(Ls)
    assert isinstance(caches, list) and len(caches) == C and 1 <= B <= 24 and h_in.shape[0] == B
    assert all(lc >= 1 for lc in Ls)

    offs_before = [int(c_i.offset) for c_i in caches]
    if offsets is None:
        offsets = mx.array([offs_before[c] + t for c in range(C) for t in range(Ls[c])], dtype=mx.int32)
    q_nope, q_rope, kv_latent, k_rope, g_raw = _mla_verify_proj_prep(
        h_in, attn, offsets, part_ss=part_ss, norm_w=norm_w, norm_eps=norm_eps
    )

    outs = []
    pos = 0
    for c, lc in enumerate(Ls):
        kl_c, kr_c = caches[c].update_and_fetch(
            kv_latent[pos : pos + lc].reshape(1, 1, lc, 512),
            k_rope[pos : pos + lc].reshape(1, 1, lc, 64),
        )
        _mla_append_causal_slice(outs, kl_c, kr_c, q_nope, q_rope, pos, 0, offs_before[c], lc, None)
        pos += lc

    out = mx.concatenate(outs, axis=0)
    gated = mla_multilinear_lin(out, attn.unembed_out, g_raw=g_raw)
    return _mla_dense_res(gated, attn.dense, res, return_part_ss=return_part_ss)
