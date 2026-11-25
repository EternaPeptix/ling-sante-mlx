"""Bit-exact native F8_E4M3 (128x128 F32 block scale_inv) GEMV and dequant kernels for BailingMoeV3.

Every element decode reproduces:
    static_cast<bfloat16_t>(E4M3_LUT[u8] * scale_inv[n // 128, k // 128])
using Zero-Mul-256 vectorized ushort4 ALU E4M3 decode (100% bit-identical across all 256 byte values
for scale_inv < 2^120) and accumulates in exact MLX GEMV K-reduction order (SN=32, TN=4 -> blockN=128)
with 0 bit diffs against bf16 gemv_rows / dequantized bf16 weights across B = 1..24.

Optimizations included:
  - Zero-Mul-256 vectorized E4M3 ALU + uniform<int> / simd_broadcast_first scales + bfloat4 wf.
  - Native bfloat4 (align 8) vector loads/stores replacing struct Bf4 (align 2).
  - Barrier-free private per-thread tg_acc overflow keeping bsg=1 (weights read once!) across B=1..24.
  - 80-core tail-wave tiling for N=4096 FP8 GEMVs (512/1024 TGs).
  - Fused mixed FP8+BF16 launchers:
      * `gemv_lin_multi`: auto-fuses KDA `[q,k,v,f,g,b_proj]` and MLA `[q_a,kv_a,g_proj]` into 1 launch!
      * `gemv_router_se_swiglu_rows`: fuses BF16 router GEMV + FP8 shared_experts SwiGLU into 1 launch!
  - Optional deterministic `part_ss` RMSNorm producer/consumer hooks (0 bit diffs vs `mx.fast.rms_norm`).
  - Optional runtime-on-load FP8 K-major tiling / merged qkvfg packing helpers (`pack_fp8_k_major`, `pack_qkvfg_fp8`).
"""
import os

import mlx.core as mx
import numpy as np

from .gemv import (
    MAX_B,
    _PART_SS_HEADER,
    _validate_pss_in,
    gemv_rows,
    gemv_rows_group,
    gemv_rows_res,
    gemv_swiglu_rows,
    router_gemv_rows,
)
from .glue import GLUE_HEADER, flit

_KERNELS = {}
_ENV_FP8_CFG_RAW = os.environ.get("LINGSANTE_FP8_CFG", "").strip()
_ENV_FP8_CFG = tuple(int(v) for v in _ENV_FP8_CFG_RAW.split(",")) if _ENV_FP8_CFG_RAW else None


def e4m3fn_table_np() -> np.ndarray:
    """IEEE/OCP F8_E4M3FN lookup table matching tools/convert.py bit-for-bit."""
    lut = np.zeros(256, dtype=np.float32)
    for i in range(256):
        s = -1.0 if (i & 0x80) else 1.0
        e = (i >> 3) & 0xF
        m = i & 7
        if e == 0xF and m == 7:
            v = np.nan
        elif e == 0:
            v = m / 8.0 * (2.0**-6)
        else:
            v = (1.0 + m / 8.0) * (2.0 ** (e - 7))
        lut[i] = np.float32(s * v)
    return lut


def _e4m3_lut_metal() -> str:
    lut = e4m3fn_table_np()
    items = []
    for i in range(256):
        v = float(lut[i])
        if np.isnan(v):
            items.append("as_type<float>(0x7fc00000u)")
        else:
            items.append(v.hex() + "f")
    rows = [", ".join(items[j : j + 8]) for j in range(0, 256, 8)]
    return "constant float E4M3_LUT[256] = {\n  " + ",\n  ".join(rows) + "\n};\n"


# Kept exported for external callers (e.g., rows/mla.py)
E4M3_LUT_METAL = _e4m3_lut_metal()

_HEADER = (
    GLUE_HEADER
    + _PART_SS_HEADER
    + r"""
METAL_FUNC bfloat4 bv3_fp8x4_to_bf4_sc256(uchar4 pk, float sc256) {
  ushort4 b4 = ushort4(pk);
  ushort4 v4 = b4 & 127;
  ushort4 u4 = (v4 << 7) | (((v4 + 1) >> 7) << 14) | ((b4 & 128) << 8);
  float4 f4 = float4(as_type<half4>(u4)) * sc256;
  return bfloat4(f4);
}

METAL_FUNC float4 bv3_fp8x4_to_f4_sc256(uchar4 pk, float sc256) {
  return float4(bv3_fp8x4_to_bf4_sc256(pk, sc256));
}

// Bit-exact BN=8 tail helper for small BF16 projections (KDA b_proj / MLA g_proj, N=64, K=4096)
// fused into the same launch as FP8 projections (qkvfg / q_a+kv_a).
// Uses TM=1 per TG (64 TGs total) and exact BN=8 ascending reduction (0+1+...+7) -> 0 bit diffs.
template <typename T, typename XT, int B, int NSG_TG, int KK, int PSS_IN>
METAL_FUNC void bailing_gemv_bf16_bn8_tail_run(
    const device bfloat16_t* mat,
    const device XT* in_vec,
    const device float* pss_in,
    const device bfloat16_t* norm_w,
    const float norm_eps,
    device T* out_vec,
    const int N,
    threadgroup float* tg_bn8,  // [B * 8]
    uint out_row,
    uint simd_gid,
    uint simd_lid) {
  if (int(out_row) >= N) return;
  constexpr int BN = 8;
  constexpr int blockN = BN * 32 * 4;  // 1024
  constexpr int n_iter = KK / blockN;
  const uniform<int> n_iter_u = make_uniform(n_iter);
  const device bfloat16_t* row_p = mat + size_t(out_row) * KK;

  constexpr int B_CHK = (B < 2) ? B : 2;
  for (int b_base = 0; b_base < B; b_base += B_CHK) {
    float inv_b[PSS_IN > 0 ? B_CHK : 1];
    if constexpr (PSS_IN > 0) {
#pragma unroll
      for (int bi = 0; bi < B_CHK; ++bi) {
        int b = b_base + bi;
        inv_b[bi] = (b < B) ? bv3_reduce_part_ss_inv<KK, PSS_IN>(pss_in + size_t(b) * PSS_IN, norm_eps, simd_lid) : 0.0f;
      }
    }

    for (int sgn = int(simd_gid); sgn < BN; sgn += NSG_TG) {
      int bn = (sgn * 32 + int(simd_lid)) * 4;
      float acc[B_CHK];
#pragma unroll
      for (int bi = 0; bi < B_CHK; ++bi) acc[bi] = 0.0f;

      for (int i = 0; i < n_iter_u; ++i) {
        bfloat4 nw_v;
        if constexpr (PSS_IN > 0) {
          nw_v = *(const device bfloat4*)(norm_w + bn);
        }
        float4 wv = float4(*(const device bfloat4*)(row_p + bn));
#pragma unroll
        for (int bi = 0; bi < B_CHK; ++bi) {
          int b = b_base + bi;
          if (b < B) {
            bfloat4 px = *(const device bfloat4*)(in_vec + size_t(b) * KK + bn);
            if constexpr (PSS_IN > 0) {
              px = bv3_apply_rms_bf4(px, nw_v, inv_b[bi]);
            }
            float4 vc = float4(px);
            float a = acc[bi];
            a += wv.x * vc.x;
            a += wv.y * vc.y;
            a += wv.z * vc.z;
            a += wv.w * vc.w;
            acc[bi] = a;
          }
        }
        bn += blockN;
      }

#pragma unroll
      for (int bi = 0; bi < B_CHK; ++bi) {
        int b = b_base + bi;
        if (b < B) {
          float a = acc[bi];
#pragma unroll
          for (ushort sn = 16; sn >= 1; sn >>= 1) {
            a += simd_shuffle_down(a, sn);
          }
          if (simd_lid == 0) {
            tg_bn8[b * BN + sgn] = a;
          }
        }
      }
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_gid == 0 && int(simd_lid) < B) {
    const int b = int(simd_lid);
    float tot = tg_bn8[b * BN + 0];
#pragma unroll
    for (int sgn = 1; sgn < BN; ++sgn) {
      tot += tg_bn8[b * BN + sgn];
    }
    out_vec[size_t(b) * N + out_row] = static_cast<T>(tot);
  }
}

// Bit-exact BN=1 BF16->F32 router tail helper fused with FP8 shared_experts SwiGLU.
template <typename XT, int B, int BM_M, int KK, int PSS_IN>
METAL_FUNC void bailing_router_bf16_tail_run(
    const device bfloat16_t* mat,
    const device XT* in_vec,
    const device float* pss_in,
    const device bfloat16_t* norm_w,
    const float norm_eps,
    device float* out_vec,
    const int N,
    uint t_loc,
    uint simd_gid,
    uint simd_lid) {
  const int out_row = int(t_loc) * BM_M + int(simd_gid);
  if (out_row >= N) return;
  constexpr int blockN = 128;
  constexpr int n_iter = KK / blockN;
  const uniform<int> n_iter_u = make_uniform(n_iter);
  const device bfloat16_t* row_p = mat + size_t(out_row) * KK;
  constexpr int B_CHK = (B < 2) ? B : 2;

  for (int b_base = 0; b_base < B; b_base += B_CHK) {
    float inv_b[PSS_IN > 0 ? B_CHK : 1];
    if constexpr (PSS_IN > 0) {
#pragma unroll
      for (int bi = 0; bi < B_CHK; ++bi) {
        int b = b_base + bi;
        inv_b[bi] = (b < B) ? bv3_reduce_part_ss_inv<KK, PSS_IN>(pss_in + size_t(b) * PSS_IN, norm_eps, simd_lid) : 0.0f;
      }
    }

    float acc[B_CHK];
#pragma unroll
    for (int bi = 0; bi < B_CHK; ++bi) acc[bi] = 0.0f;

    int bn = int(simd_lid) * 4;
    for (int i = 0; i < n_iter_u; ++i) {
      bfloat4 nw_v;
      if constexpr (PSS_IN > 0) {
        nw_v = *(const device bfloat4*)(norm_w + bn);
      }
      float4 wv = float4(*(const device bfloat4*)(row_p + bn));
#pragma unroll
      for (int bi = 0; bi < B_CHK; ++bi) {
        int b = b_base + bi;
        if (b < B) {
          bfloat4 px = *(const device bfloat4*)(in_vec + size_t(b) * KK + bn);
          if constexpr (PSS_IN > 0) {
            px = bv3_apply_rms_bf4(px, nw_v, inv_b[bi]);
          }
          float4 vc = float4(px);
          float a = acc[bi];
          a += wv.x * vc.x;
          a += wv.y * vc.y;
          a += wv.z * vc.z;
          a += wv.w * vc.w;
          acc[bi] = a;
        }
      }
      bn += blockN;
    }

#pragma unroll
    for (int bi = 0; bi < B_CHK; ++bi) {
      int b = b_base + bi;
      if (b < B) {
        float a = acc[bi];
#pragma unroll
        for (ushort sn = 16; sn >= 1; sn >>= 1) {
          a += simd_shuffle_down(a, sn);
        }
        if (simd_lid == 0) {
          out_vec[size_t(b) * N + out_row] = a;
        }
      }
    }
  }
}

template <
    typename T,
    typename XT,
    int B,
    int BSG,
    int B_SUB,
    int B_REG,
    int BM_M,
    int TM,
    int RES,
    int SWIGLU,
    int KK,
    int KMAJOR,
    int PSS_IN,
    int EMIT_PSS,
    int UNROLL,
    int WF_MODE>
METAL_FUNC void bailing_gemv_fp8_rows_run(
    const device uchar* __restrict mat,
    const device float* __restrict sc_mat,
    const device uchar* __restrict mat_up,
    const device float* __restrict sc_up,
    const device XT* __restrict in_vec,   // (B, KK) row-major
    const device T* __restrict res_vec,   // (B, N) row-major when RES == 1
    const device float* __restrict pss_in,
    const device bfloat16_t* __restrict norm_w,
    const float norm_eps,
    device T* __restrict out_vec,         // (B, N) row-major
    device float* __restrict pss_out,
    const int N,
    threadgroup float* tg_acc,
    threadgroup bfloat16_t* tg_y,
    uint tgx,
    uint simd_gid,
    uint simd_lid) {
  constexpr int TN = 4;
  constexpr int SN = 32;
  constexpr int blockN = SN * TN;  // 128 == FP8 scale block size!
  constexpr int blockM = BM_M * TM;
  constexpr int k_blocks = KK >> 7;
  constexpr int TOT_R = SWIGLU ? (2 * TM) : TM;
  constexpr int B_TG = B_SUB - B_REG;

  const int m_sg = int(simd_gid) / BSG;
  const int b_sg = int(simd_gid) % BSG;
  const int b0 = b_sg * B_SUB;

  int out_row = int(tgx) * blockM + m_sg * TM;
  const bool row_valid = (out_row < N);
  if constexpr (EMIT_PSS == 0) {
    if (!row_valid) return;
  }
  out_row = row_valid ? (out_row + TM <= N ? out_row : N - TM) : 0;

  const int row_blk = out_row >> 7;
  sc_mat += size_t(row_blk) * k_blocks;
  if constexpr (SWIGLU) sc_up += size_t(row_blk) * k_blocks;

  // When KMAJOR == 1, weights are packed as [N/4, K/128, 4, 128] (4-row tile contiguous per K-block).
  // When KMAJOR == 0, weights are standard row-major [N, K].
  const int tile4 = out_row >> 2;
  const int sub4 = out_row & 3;
  if constexpr (KMAJOR) {
    mat += (size_t(tile4) * k_blocks * 4 + sub4) * 128;
    if constexpr (SWIGLU) mat_up += (size_t(tile4) * k_blocks * 4 + sub4) * 128;
  } else {
    mat += size_t(out_row) * KK;
    if constexpr (SWIGLU) mat_up += size_t(out_row) * KK;
  }

  float inv_b[PSS_IN > 0 ? B_SUB : 1];
  if constexpr (PSS_IN > 0) {
#pragma unroll
    for (int bi = 0; bi < B_SUB; ++bi) {
      const int b = b0 + bi;
      inv_b[bi] = (b < B) ? bv3_reduce_part_ss_inv<KK, PSS_IN>(pss_in + size_t(b) * PSS_IN, norm_eps, simd_lid) : 0.0f;
    }
  }

  float result[B_REG > 0 ? B_REG : 1][TM];
  float result_up[(SWIGLU && B_REG > 0) ? B_REG : 1][SWIGLU ? TM : 1];
#pragma unroll
  for (int bi = 0; bi < B_REG; bi++) {
#pragma unroll
    for (int tm = 0; tm < TM; tm++) {
      result[bi][tm] = 0.0f;
      if constexpr (SWIGLU) result_up[bi][tm] = 0.0f;
    }
  }

  // Barrier-free private per-thread TG accumulator slice with stride-1 simd_lid (0 bank conflicts!)
  threadgroup float* my_tg = tg_acc;
  if constexpr (B_TG > 0) {
    my_tg = tg_acc + size_t(simd_gid) * (B_TG * TOT_R * 32) + simd_lid;
#pragma unroll
    for (int ti = 0; ti < B_TG * TOT_R; ++ti) {
      my_tg[ti * 32] = 0.0f;
    }
  }

  const uniform<int> k_blocks_u = make_uniform(k_blocks);
  int bn = int(simd_lid) * TN;
  const int lane4 = int(simd_lid) * 4;

  if (row_valid) {
    if constexpr (UNROLL > 1 && KMAJOR == 0 && PSS_IN == 0 && B_TG == 0 && sizeof(XT) == 2) {
      // Unrolled cold-DRAM fast path: prefetch UNROLL K-blocks of uchar4 weights before dequant/FMA,
      // and hoist float4 weights (WF_MODE 1: bi-outer float4 wf[TM]; WF_MODE 2: tm-outer single float4 wv;
      // WF_MODE 0: bi-outer bfloat4 wf[TM]). 100% bit-exact K-block reduction order!
      for (int i = 0; i < k_blocks_u; i += UNROLL) {
        uchar4 pk_u[UNROLL][TM];
        uchar4 pku_u[SWIGLU ? UNROLL : 1][SWIGLU ? TM : 1];
        float sc_u[UNROLL];
        float scu_u[SWIGLU ? UNROLL : 1];
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
          sc_u[u] = simd_broadcast_first(sc_mat[i + u] * 256.0f);
          if constexpr (SWIGLU) scu_u[u] = simd_broadcast_first(sc_up[i + u] * 256.0f);
#pragma unroll
          for (int tm = 0; tm < TM; ++tm) {
            const size_t mo = size_t(tm) * KK + bn + u * blockN;
            pk_u[u][tm] = *(const device uchar4*)(mat + mo);
            if constexpr (SWIGLU) pku_u[u][tm] = *(const device uchar4*)(mat_up + mo);
          }
        }
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
          const int bnu = bn + u * blockN;
          if constexpr (WF_MODE == 2 && !SWIGLU) {
            // tm-outer, single float4 wv live (4 regs, zero redundant bf16->f32 shifts, zero B=9..12 spill!)
#pragma unroll
            for (int tm = 0; tm < TM; ++tm) {
              float4 wv = bv3_fp8x4_to_f4_sc256(pk_u[u][tm], sc_u[u]);
#pragma unroll
              for (int bi = 0; bi < B_REG; ++bi) {
                const int b = b0 + bi;
                if (b < B) {
                  float4 vc = float4(*(const device bfloat4*)(in_vec + size_t(b) * KK + bnu));
                  float acc = result[bi][tm];
                  acc += wv.x * vc.x;
                  acc += wv.y * vc.y;
                  acc += wv.z * vc.z;
                  acc += wv.w * vc.w;
                  result[bi][tm] = acc;
                }
              }
            }
          } else if constexpr (WF_MODE == 1) {
            // Hoisted float4 wf[TM] (zero redundant bf16->f32 shifts inside bi loop for B<=8!)
            float4 wf[TM];
            float4 wf_u[SWIGLU ? TM : 1];
#pragma unroll
            for (int tm = 0; tm < TM; ++tm) {
              wf[tm] = bv3_fp8x4_to_f4_sc256(pk_u[u][tm], sc_u[u]);
              if constexpr (SWIGLU) wf_u[tm] = bv3_fp8x4_to_f4_sc256(pku_u[u][tm], scu_u[u]);
            }
#pragma unroll
            for (int bi = 0; bi < B_REG; ++bi) {
              const int b = b0 + bi;
              if (b < B) {
                float4 vc = float4(*(const device bfloat4*)(in_vec + size_t(b) * KK + bnu));
#pragma unroll
                for (int tm = 0; tm < TM; ++tm) {
                  float acc = result[bi][tm];
                  acc += wf[tm].x * vc.x;
                  acc += wf[tm].y * vc.y;
                  acc += wf[tm].z * vc.z;
                  acc += wf[tm].w * vc.w;
                  result[bi][tm] = acc;
                  if constexpr (SWIGLU) {
                    float accu = result_up[bi][tm];
                    accu += wf_u[tm].x * vc.x;
                    accu += wf_u[tm].y * vc.y;
                    accu += wf_u[tm].z * vc.z;
                    accu += wf_u[tm].w * vc.w;
                    result_up[bi][tm] = accu;
                  }
                }
              }
            }
          } else {
            // Low-register bfloat4 wf[TM]
            bfloat4 wf[TM];
            bfloat4 wf_u[SWIGLU ? TM : 1];
#pragma unroll
            for (int tm = 0; tm < TM; ++tm) {
              wf[tm] = bv3_fp8x4_to_bf4_sc256(pk_u[u][tm], sc_u[u]);
              if constexpr (SWIGLU) wf_u[tm] = bv3_fp8x4_to_bf4_sc256(pku_u[u][tm], scu_u[u]);
            }
#pragma unroll
            for (int bi = 0; bi < B_REG; ++bi) {
              const int b = b0 + bi;
              if (b < B) {
                float4 vc = float4(*(const device bfloat4*)(in_vec + size_t(b) * KK + bnu));
#pragma unroll
                for (int tm = 0; tm < TM; ++tm) {
                  float4 wv = float4(wf[tm]);
                  float acc = result[bi][tm];
                  acc += wv.x * vc.x;
                  acc += wv.y * vc.y;
                  acc += wv.z * vc.z;
                  acc += wv.w * vc.w;
                  result[bi][tm] = acc;
                  if constexpr (SWIGLU) {
                    float4 wvu = float4(wf_u[tm]);
                    float accu = result_up[bi][tm];
                    accu += wvu.x * vc.x;
                    accu += wvu.y * vc.y;
                    accu += wvu.z * vc.z;
                    accu += wvu.w * vc.w;
                    result_up[bi][tm] = accu;
                  }
                }
              }
            }
          }
        }
        bn += blockN * UNROLL;
      }
    } else {
      for (int i = 0; i < k_blocks_u; ++i) {
        const float sc256 = simd_broadcast_first(sc_mat[i] * 256.0f);
        const float sc256_u = SWIGLU ? simd_broadcast_first(sc_up[i] * 256.0f) : 0.0f;
        bfloat4 nw_v;
        if constexpr (PSS_IN > 0 && sizeof(XT) == 2) {
          nw_v = *(const device bfloat4*)(norm_w + bn);
        }

        bfloat4 wf[TM];
        bfloat4 wf_u[SWIGLU ? TM : 1];
#pragma unroll
        for (int tm = 0; tm < TM; tm++) {
          const size_t mo = KMAJOR
              ? (size_t(i) * 512 + size_t(tm) * 128 + lane4)
              : (size_t(tm) * KK + bn);
          const uchar4 pk = *(const device uchar4*)(mat + mo);
          wf[tm] = bv3_fp8x4_to_bf4_sc256(pk, sc256);
          if constexpr (SWIGLU) {
            const uchar4 pku = *(const device uchar4*)(mat_up + mo);
            wf_u[tm] = bv3_fp8x4_to_bf4_sc256(pku, sc256_u);
          }
        }

#pragma unroll
        for (int bi = 0; bi < B_REG; bi++) {
          const int b = b0 + bi;
          if (b < B) {
            float4 vc;
            if constexpr (sizeof(XT) == 2) {
              bfloat4 px = *(const device bfloat4*)(in_vec + size_t(b) * KK + bn);
              if constexpr (PSS_IN > 0) {
                px = bv3_apply_rms_bf4(px, nw_v, inv_b[bi]);
              }
              vc = float4(px);
            } else {
              vc = float4(
                  static_cast<float>(in_vec[size_t(b) * KK + bn + 0]),
                  static_cast<float>(in_vec[size_t(b) * KK + bn + 1]),
                  static_cast<float>(in_vec[size_t(b) * KK + bn + 2]),
                  static_cast<float>(in_vec[size_t(b) * KK + bn + 3]));
            }
#pragma unroll
            for (int tm = 0; tm < TM; tm++) {
              float4 wv = float4(wf[tm]);
              float acc = result[bi][tm];
              acc += wv.x * vc.x;
              acc += wv.y * vc.y;
              acc += wv.z * vc.z;
              acc += wv.w * vc.w;
              result[bi][tm] = acc;
              if constexpr (SWIGLU) {
                float4 wvu = float4(wf_u[tm]);
                float accu = result_up[bi][tm];
                accu += wvu.x * vc.x;
                accu += wvu.y * vc.y;
                accu += wvu.z * vc.z;
                accu += wvu.w * vc.w;
                result_up[bi][tm] = accu;
              }
            }
          }
        }
        if constexpr (B_TG > 0) {
#pragma unroll
          for (int ti = 0; ti < B_TG; ti++) {
            const int bi = B_REG + ti;
            const int b = b0 + bi;
            if (b < B) {
              float4 vc;
              if constexpr (sizeof(XT) == 2) {
                bfloat4 px = *(const device bfloat4*)(in_vec + size_t(b) * KK + bn);
                if constexpr (PSS_IN > 0) {
                  px = bv3_apply_rms_bf4(px, nw_v, inv_b[bi]);
                }
                vc = float4(px);
              } else {
                vc = float4(
                    static_cast<float>(in_vec[size_t(b) * KK + bn + 0]),
                    static_cast<float>(in_vec[size_t(b) * KK + bn + 1]),
                    static_cast<float>(in_vec[size_t(b) * KK + bn + 2]),
                    static_cast<float>(in_vec[size_t(b) * KK + bn + 3]));
              }
#pragma unroll
              for (int tm = 0; tm < TM; tm++) {
                float4 wv = float4(wf[tm]);
                const int slot = (ti * TOT_R + tm) * 32;
                float acc = my_tg[slot];
                acc += wv.x * vc.x;
                acc += wv.y * vc.y;
                acc += wv.z * vc.z;
                acc += wv.w * vc.w;
                my_tg[slot] = acc;
                if constexpr (SWIGLU) {
                  float4 wvu = float4(wf_u[tm]);
                  const int slot_u = (ti * TOT_R + TM + tm) * 32;
                  float accu = my_tg[slot_u];
                  accu += wvu.x * vc.x;
                  accu += wvu.y * vc.y;
                  accu += wvu.z * vc.z;
                  accu += wvu.w * vc.w;
                  my_tg[slot_u] = accu;
                }
              }
            }
          }
        }
        bn += blockN;
      }
    }
  }

  // Reduce & write register-resident rows
#pragma unroll
  for (int bi = 0; bi < B_REG; bi++) {
    const int b = b0 + bi;
    if (b < B) {
      T y_loc[TM];
#pragma unroll
      for (int tm = 0; tm < TM; tm++) {
        float acc = result[bi][tm];
#pragma unroll
        for (ushort sn = 16; sn >= 1; sn >>= 1) {
          acc += simd_shuffle_down(acc, sn);
        }
        float accu = 0.0f;
        if constexpr (SWIGLU) {
          accu = result_up[bi][tm];
#pragma unroll
          for (ushort sn = 16; sn >= 1; sn >>= 1) {
            accu += simd_shuffle_down(accu, sn);
          }
        }
        if (row_valid && simd_lid == 0) {
          T yv;
          if constexpr (SWIGLU) {
            yv = glue_swiglu_bf16(static_cast<bfloat16_t>(acc), static_cast<bfloat16_t>(accu));
          } else if constexpr (RES) {
            yv = static_cast<T>(acc) + res_vec[size_t(b) * N + out_row + tm];
          } else {
            yv = static_cast<T>(acc);
          }
          out_vec[size_t(b) * N + out_row + tm] = yv;
          y_loc[tm] = yv;
        }
      }
      if constexpr (EMIT_PSS == 1) {
        if (row_valid && simd_lid == 0) {
          if constexpr (TM == 4) {
            pss_out[size_t(b) * (N >> 2) + (out_row >> 2)] = bv3_fma_ss4(
                static_cast<bfloat16_t>(y_loc[0]),
                static_cast<bfloat16_t>(y_loc[1]),
                static_cast<bfloat16_t>(y_loc[2]),
                static_cast<bfloat16_t>(y_loc[3]));
          } else {
#pragma unroll
            for (int tm = 0; tm < TM; tm++) {
              tg_y[size_t(b) * blockM + m_sg * TM + tm] = static_cast<bfloat16_t>(y_loc[tm]);
            }
          }
        }
      }
    }
  }

  // Reduce & write TG-overflow rows (zero barriers needed for private tg_acc!)
  if constexpr (B_TG > 0) {
#pragma unroll
    for (int ti = 0; ti < B_TG; ++ti) {
      const int b = b0 + B_REG + ti;
      if (b < B) {
        T y_loc[TM];
#pragma unroll
        for (int tm = 0; tm < TM; ++tm) {
          float acc = my_tg[(ti * TOT_R + tm) * 32];
#pragma unroll
          for (ushort sn = 16; sn >= 1; sn >>= 1) {
            acc += simd_shuffle_down(acc, sn);
          }
          float accu = 0.0f;
          if constexpr (SWIGLU) {
            accu = my_tg[(ti * TOT_R + TM + tm) * 32];
#pragma unroll
            for (ushort sn = 16; sn >= 1; sn >>= 1) {
              accu += simd_shuffle_down(accu, sn);
            }
          }
          if (row_valid && simd_lid == 0) {
            T yv;
            if constexpr (SWIGLU) {
              yv = glue_swiglu_bf16(static_cast<bfloat16_t>(acc), static_cast<bfloat16_t>(accu));
            } else if constexpr (RES) {
              yv = static_cast<T>(acc) + res_vec[size_t(b) * N + out_row + tm];
            } else {
              yv = static_cast<T>(acc);
            }
            out_vec[size_t(b) * N + out_row + tm] = yv;
            y_loc[tm] = yv;
          }
        }
        if constexpr (EMIT_PSS == 1) {
          if (row_valid && simd_lid == 0) {
            if constexpr (TM == 4) {
              pss_out[size_t(b) * (N >> 2) + (out_row >> 2)] = bv3_fma_ss4(
                  static_cast<bfloat16_t>(y_loc[0]),
                  static_cast<bfloat16_t>(y_loc[1]),
                  static_cast<bfloat16_t>(y_loc[2]),
                  static_cast<bfloat16_t>(y_loc[3]));
            } else {
#pragma unroll
              for (int tm = 0; tm < TM; tm++) {
                tg_y[size_t(b) * blockM + m_sg * TM + tm] = static_cast<bfloat16_t>(y_loc[tm]);
              }
            }
          }
        }
      }
    }
  }

  // TG-staged deterministic 4-elem FMA + butterfly sum when EMIT_PSS == 1 && TM < 4
  if constexpr (EMIT_PSS == 1 && TM < 4) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_gid == 0) {
      constexpr int NC = blockM / 4;
      const int ntg = N / blockM;
      for (int b = 0; b < B; ++b) {
        float s = 0.0f;
        if (int(simd_lid) < NC) {
          const int off = b * blockM + int(simd_lid) * 4;
          s = bv3_fma_ss4(tg_y[off + 0], tg_y[off + 1], tg_y[off + 2], tg_y[off + 3]);
        }
        if constexpr (NC >= 2) s += simd_shuffle_xor(s, 1);
        if constexpr (NC >= 4) s += simd_shuffle_xor(s, 2);
        if constexpr (NC >= 8) s += simd_shuffle_xor(s, 4);
        if (simd_lid == 0) {
          pss_out[size_t(b) * ntg + tgx] = s;
        }
      }
    }
  }
}
"""
)

_DEQUANT_2D_SRC = r"""
    const uint col4 = thread_position_in_grid.x;
    const uint row = thread_position_in_grid.y;
    const uint k = col4 * 4;
    if (int(row) >= NN || int(k) >= KK) return;
    const uint sk = (KK + 127) >> 7;
    const float sc256 = scale_inv[(row >> 7) * sk + (k >> 7)] * 256.0f;
    const size_t w_off = KMAJOR
        ? ((((size_t(row >> 2) * sk + (k >> 7)) * 4 + (row & 3)) << 7) + (k & 127))
        : (size_t(row) * KK + k);
    const uchar4 pk = *(const device uchar4*)(w_fp8 + w_off);
    *(device bfloat4*)(out + size_t(row) * KK + k) = bv3_fp8x4_to_bf4_sc256(pk, sc256);
"""

_DEQUANT_ML_SRC = r"""
    const uint col4 = thread_position_in_grid.x;
    const uint row = thread_position_in_grid.y;
    const uint h = thread_position_in_grid.z;
    const uint k = col4 * 4;
    if (int(h) >= HH || int(row) >= NN || int(k) >= KK) return;
    const float sc256 = (SCALE_BY_N ? scale_inv[h * 4 + (row >> 7)] : scale_inv[h * 4 + (k >> 7)]) * 256.0f;
    const size_t off = (size_t(h) * NN + row) * KK + k;
    const uchar4 pk = *(const device uchar4*)(w_fp8 + off);
    *(device bfloat4*)(out + off) = bv3_fp8x4_to_bf4_sc256(pk, sc256);
"""


def _dq2_kernel():
    k = _KERNELS.get("dq2")
    if k is None:
        k = _KERNELS["dq2"] = mx.fast.metal_kernel(
            name="bv3_dequant_fp8_2d",
            input_names=["w_fp8", "scale_inv"],
            output_names=["out"],
            header=_HEADER,
            source=_DEQUANT_2D_SRC,
            ensure_row_contiguous=True,
        )
    return k


def _dqml_kernel():
    k = _KERNELS.get("dqml")
    if k is None:
        k = _KERNELS["dqml"] = mx.fast.metal_kernel(
            name="bv3_dequant_fp8_ml",
            input_names=["w_fp8", "scale_inv"],
            output_names=["out"],
            header=_HEADER,
            source=_DEQUANT_ML_SRC,
            ensure_row_contiguous=True,
        )
    return k


def dequant_fp8_to_bf16(w_fp8: mx.array, scale_inv: mx.array, *, k_major: bool = False) -> mx.array:
    """Exact FP8 E4M3 [N, K] * 128x128 F32 scale_inv -> bfloat16 [N, K]."""
    if k_major:
        N4, K128, R4, C128 = w_fp8.shape
        N, K = N4 * R4, K128 * C128
    else:
        N, K = w_fp8.shape
    assert w_fp8.dtype == mx.uint8 and scale_inv.dtype == mx.float32 and K % 4 == 0
    (out,) = _dq2_kernel()(
        inputs=[w_fp8, scale_inv],
        template=[("NN", N), ("KK", K), ("KMAJOR", int(bool(k_major)))],
        grid=(K // 4, N, 1),
        threadgroup=(64, 4, 1),
        output_shapes=[(N, K)],
        output_dtypes=[mx.bfloat16],
    )
    return out


def dequant_fp8_multilinear_to_bf16(w_fp8: mx.array, scale_inv: mx.array, scale_by_n: bool) -> mx.array:
    """Exact FP8 E4M3 [H, N, K] * [H, 4] F32 scale_inv -> bfloat16 [H, N, K]."""
    H, N, K = w_fp8.shape
    assert w_fp8.dtype == mx.uint8 and scale_inv.dtype == mx.float32 and K % 4 == 0
    (out,) = _dqml_kernel()(
        inputs=[w_fp8, scale_inv],
        template=[("HH", H), ("NN", N), ("KK", K), ("SCALE_BY_N", int(bool(scale_by_n)))],
        grid=(K // 4, N, H),
        threadgroup=(32, 8, 1),
        output_shapes=[(H, N, K)],
        output_dtypes=[mx.bfloat16],
    )
    return out


def pack_fp8_k_major(w_fp8: mx.array) -> mx.array:
    """Runtime-on-load bit-exact K-major 4-row tiling: [N, K] -> [N//4, K//128, 4, 128] u8 (0 extra bytes)."""
    N, K = w_fp8.shape
    assert w_fp8.dtype == mx.uint8 and N % 4 == 0 and K % 128 == 0
    return mx.contiguous(w_fp8.reshape(N // 4, 4, K // 128, 128).transpose(0, 2, 1, 3))


def pack_qkvfg_fp8(mods, *, k_major: bool = False):
    """Runtime-on-load merge of `[q, k, v, f, g]` FP8 weights & scales into 1 contiguous pair (0 extra bytes)."""
    w_cat = mx.concatenate([m.weight_fp8 for m in mods], axis=0)
    s_cat = mx.concatenate([m.weight_scale_inv for m in mods], axis=0)
    if k_major:
        w_cat = pack_fp8_k_major(w_cat)
    return w_cat, s_cat


def _fp8_cfg(N: int, K: int, B: int, is_swiglu: bool = False, emit_pss: bool = False):
    """Cold multi-layer DRAM tuned `(bsg, b_sub, bm_m, tm)` per B regime on M3 Ultra (`bsg=1` across B=1..16!)."""
    if _ENV_FP8_CFG is not None:
        bsg, bm_m, tm = _ENV_FP8_CFG
        b_sub = (B + bsg - 1) // bsg
        if N < tm:
            tm = 1
        return bsg, b_sub, bm_m, tm

    if is_swiglu:
        if N <= 2048:
            # shared_experts SwiGLU (N=1280, K=4096): higher TG count avoids 80-core underfill!
            if B <= 4:
                bsg, bm_m, tm = 1, 4, 2
            elif B <= 6:
                bsg, bm_m, tm = 1, 4, 1
            elif B <= 8:
                bsg, bm_m, tm = 1, 2, 1  # 640 TGs: 36.0us vs 47.4us (-0.78ms x68)
            elif B <= 16:
                bsg, bm_m, tm = 1, 4, 1
            else:
                bsg, bm_m, tm = 2, 4, 1
        else:
            # dense SwiGLU (N=10240, K=4096)
            if B <= 6:
                bsg, bm_m, tm = 1, 4, 1
            elif B <= 8:
                bsg, bm_m, tm = 1, 4, 2  # 138.6us vs 152.7us
            elif B <= 16:
                bsg, bm_m, tm = 1, 4, 1
            else:
                bsg, bm_m, tm = 2, 4, 1
    else:
        if N == 4096:
            if K >= 8192:
                # KDA/MLA o_proj (K=8192) & dense down_proj (K=10240)
                if B <= 5:
                    bsg, bm_m, tm = (1, 2, 2) if emit_pss else (1, 2, 1)  # 65-66us cold (vs 70-77us)
                elif B <= 8:
                    bsg, bm_m, tm = (1, 2, 2) if K >= 10240 else (1, 4, 2)  # 69.4us cold (vs 92.3us)
                elif B <= 12:
                    bsg, bm_m, tm = 1, 4, 2  # WF_MODE=2 tm-outer: 82.4us cold (vs 102-114us)
                elif B <= 16:
                    bsg, bm_m, tm = 1, 8, 2
                else:
                    bsg, bm_m, tm = 2, 4, 2
            else:
                # shared_experts down_proj (N=4096, K=1280)
                if B <= 4:
                    bsg, bm_m, tm = 1, 2, 2
                elif B <= 6:
                    bsg, bm_m, tm = (1, 2, 2) if emit_pss else (1, 2, 1)
                elif B <= 12:
                    bsg, bm_m, tm = 1, 4, 2
                elif B <= 16:
                    bsg, bm_m, tm = 1, 8, 2
                else:
                    bsg, bm_m, tm = 2, 4, 2
        else:
            if B <= 12:
                bsg, bm_m, tm = 1, 4, 2  # U=2 + WF_MODE=1 (B<=8) / WF_MODE=2 (B=9..12)
            elif B <= 16:
                bsg, bm_m, tm = 1, 8, 2  # 425us at B=16 (vs 483us bsg=2)
            else:
                bsg, bm_m, tm = 2, 4, 2
    if N < tm:
        tm = 1
    b_sub = (B + bsg - 1) // bsg
    return bsg, b_sub, bm_m, tm


def _fp8_unroll_wf(N: int, K: int, B: int, cfg, b_reg: int, is_swiglu: bool, k_major: bool, pss_in: int, x_dt):
    """Select `(unroll, wf_mode)` for the cold-DRAM unrolled fast path (0 bit diffs)."""
    if k_major or pss_in > 0 or x_dt != mx.bfloat16:
        return 1, 0
    bsg, b_sub, _, tm = cfg
    if bsg > 1 or b_sub != b_reg:
        return 1, 0
    k_blocks = K >> 7
    if is_swiglu:
        if B <= 12:
            unroll = 2
            wf_mode = 1 if (B <= 4 or (tm == 1 and B <= 8)) else 0
        else:
            return 1, 0
    else:
        if N == 4096 and K <= 2048 and B >= 7:
            return 1, 0
        if B <= 8:
            unroll = 4 if (tm == 1 and k_blocks % 4 == 0) else 2
            wf_mode = 1
        elif B <= 12:
            unroll = 2
            wf_mode = 2
        else:
            return 1, 0
    while unroll > 1 and (k_blocks % unroll != 0):
        unroll //= 2
    if unroll == 1:
        wf_mode = 0
    return unroll, wf_mode


def _b_reg_fp8(b_sub: int, tm: int, is_swiglu: bool) -> int:
    if is_swiglu:
        cap = 12 if tm == 1 else (6 if tm == 2 else 2)
    else:
        cap = 12 if tm <= 2 else 4
    return min(b_sub, cap)


def _tname(dt):
    return {mx.bfloat16: "bfloat16_t", mx.float32: "float", mx.float16: "half"}[dt]


def _can_coop(bsg: int, bm_m: int, tm: int, is_swiglu: bool = False) -> bool:
    return False


def _resolve_coop(cfg, is_swiglu: bool, coop: bool | None) -> bool:
    return False


def _build_fp8_group(
    out_dt,
    x_dt,
    B,
    G,
    Ns,
    K,
    cfg,
    has_res=False,
    is_swiglu=False,
    k_major=False,
    pss_in=0,
    emit_pss=False,
    norm_eps=1e-6,
    bf16_tail_n=0,
    router_tail_n=0,
    shared_x=False,
):
    km_i = int(bool(k_major))
    emit_i = int(bool(emit_pss))
    sx_i = int(bool(shared_x and G > 1))
    eps_f = float(norm_eps) if pss_in > 0 else 0.0
    bsg, b_sub, bm_m, tm = cfg
    b_reg = _b_reg_fp8(b_sub, tm, is_swiglu)
    unroll, wf_mode = _fp8_unroll_wf(max(Ns), K, B, cfg, b_reg, is_swiglu, k_major, pss_in, x_dt)
    key = (
        "fp8",
        out_dt,
        x_dt,
        B,
        G,
        tuple(Ns),
        K,
        cfg,
        has_res,
        is_swiglu,
        km_i,
        pss_in,
        emit_i,
        eps_f,
        bf16_tail_n,
        router_tail_n,
        sx_i,
        unroll,
        wf_mode,
    )
    if key in _KERNELS:
        return _KERNELS[key]
    b_tg = b_sub - b_reg
    nsg = bsg * bm_m
    blockM = bm_m * tm
    T, XT = _tname(out_dt), _tname(x_dt)
    tot_r = (2 * tm) if is_swiglu else tm
    tg_acc_elems = max(1, (nsg * b_tg * tot_r * 32) if b_tg > 0 else 1, (B * 8) if bf16_tail_n > 0 else 1)
    tg_y_elems = (B * blockM) if (emit_i and tm < 4) else 1
    eps_lit = flit(eps_f) if pss_in > 0 else "0.0f"
    pss_in_arg = "pss_in" if pss_in > 0 else "(const device float*)0"
    nw_arg = "norm_w" if pss_in > 0 else "(const device bfloat16_t*)0"
    pss_out_arg = "pss_out" if emit_i else "(device float*)0"

    lines = [
        f"threadgroup float tg_acc[{tg_acc_elems}];",
        f"threadgroup bfloat16_t tg_y[{tg_y_elems}];",
        "const uint t = threadgroup_position_in_grid.x;",
    ]
    start = 0
    if G == 1 and bf16_tail_n == 0 and router_tail_n == 0:
        N0 = Ns[0]
        nt = (N0 + blockM - 1) // blockM
        wup = "u0" if is_swiglu else "w0"
        sup = "su0" if is_swiglu else "s0"
        rv = "res" if has_res else "out0"
        lines.append(
            f"bailing_gemv_fp8_rows_run<{T}, {XT}, {B}, {bsg}, {b_sub}, {b_reg}, {bm_m}, {tm}, {int(has_res)}, {int(is_swiglu)}, {K}, {km_i}, {pss_in}, {emit_i}, {unroll}, {wf_mode}>"
            f"(w0, s0, {wup}, {sup}, x0, {rv}, {pss_in_arg}, {nw_arg}, {eps_lit}, out0, {pss_out_arg}, {N0}, tg_acc, tg_y, t, "
            f"simdgroup_index_in_threadgroup, thread_index_in_simdgroup);"
        )
        start = nt
    else:
        for g, N in enumerate(Ns):
            start += (N + blockM - 1) // blockM
        tail_front = 0
        if bf16_tail_n > 0:
            # Schedule small BF16 tail at FRONT of grid (t < bf16_tail_n) to eliminate straggler tail wave!
            tail_front = bf16_tail_n
            lines.append(f"if (t < {tail_front}) {{")
            lines.append(
                f"  bailing_gemv_bf16_bn8_tail_run<{T}, {XT}, {B}, {nsg}, {K}, {pss_in}>"
                f"(w_bf, x0, {pss_in_arg}, {nw_arg}, {eps_lit}, out_bf, {bf16_tail_n}, tg_acc, t, "
                f"simdgroup_index_in_threadgroup, thread_index_in_simdgroup);"
            )
            lines.append("  return;")
            lines.append("}")
            start += tail_front
        elif router_tail_n > 0:
            nt_r = (router_tail_n + nsg - 1) // nsg
            tail_front = nt_r
            lines.append(f"if (t < {tail_front}) {{")
            lines.append(
                f"  bailing_router_bf16_tail_run<{XT}, {B}, {nsg}, {K}, {pss_in}>"
                f"(w_rtr, x0, {pss_in_arg}, {nw_arg}, {eps_lit}, out_rtr, {router_tail_n}, t, "
                f"simdgroup_index_in_threadgroup, thread_index_in_simdgroup);"
            )
            lines.append("  return;")
            lines.append("}")
            start += tail_front

        t_expr = f"(t - {tail_front})" if tail_front > 0 else "t"
        if G == 1:
            N0 = Ns[0]
            wup = "u0" if is_swiglu else "w0"
            sup = "su0" if is_swiglu else "s0"
            rv = "res" if has_res else "out0"
            lines.append(
                f"bailing_gemv_fp8_rows_run<{T}, {XT}, {B}, {bsg}, {b_sub}, {b_reg}, {bm_m}, {tm}, {int(has_res)}, {int(is_swiglu)}, {K}, {km_i}, {pss_in}, {emit_i}, {unroll}, {wf_mode}>"
                f"(w0, s0, {wup}, {sup}, x0, {rv}, {pss_in_arg}, {nw_arg}, {eps_lit}, out0, {pss_out_arg}, {N0}, tg_acc, tg_y, {t_expr}, "
                f"simdgroup_index_in_threadgroup, thread_index_in_simdgroup);"
            )
        else:
            assert not emit_i, "emit_pss only supported for G == 1"
            lines.extend([
                f"const uint tf = {t_expr};",
                "const device uchar* wp = w0;",
                "const device float* sp = s0;",
                "const device uchar* up_p = w0;",
                "const device float* sup_p = s0;",
                f"const device {XT}* xp = x0;",
                f"device {T}* op = out0;",
                f"const device {T}* rp = out0;",
                f"int n_cur = {Ns[0]};",
                "uint t_loc = tf;",
            ])
            cur = 0
            for g, N in enumerate(Ns):
                nt = (N + blockM - 1) // blockM
                kw = "if" if g == 0 else "else if"
                wup_s = f"up_p = u{g}; sup_p = su{g}; " if is_swiglu else ""
                xp_s = "" if sx_i else f"xp = x{g}; "
                rv_s = "rp = res; " if has_res else f"rp = out{g}; "
                if g < len(Ns) - 1:
                    lines.append(
                        f"{kw} (tf < {cur + nt}) {{ wp = w{g}; sp = s{g}; {wup_s}{xp_s}{rv_s}op = out{g}; n_cur = {N}; t_loc = tf - {cur}; }}"
                    )
                else:
                    lines.append(
                        f"else {{ wp = w{g}; sp = s{g}; {wup_s}{xp_s}{rv_s}op = out{g}; n_cur = {N}; t_loc = tf - {cur}; }}"
                    )
                cur += nt
            xp_arg = "x0" if sx_i else "xp"
            lines.append(
                f"bailing_gemv_fp8_rows_run<{T}, {XT}, {B}, {bsg}, {b_sub}, {b_reg}, {bm_m}, {tm}, {int(has_res)}, {int(is_swiglu)}, {K}, {km_i}, {pss_in}, 0, {unroll}, {wf_mode}>"
                f"(wp, sp, up_p, sup_p, {xp_arg}, rp, {pss_in_arg}, {nw_arg}, {eps_lit}, op, {pss_out_arg}, n_cur, tg_acc, tg_y, t_loc, "
                f"simdgroup_index_in_threadgroup, thread_index_in_simdgroup);"
            )

    in_names = [f"w{g}" for g in range(G)] + [f"s{g}" for g in range(G)]
    if is_swiglu:
        in_names += [f"u{g}" for g in range(G)] + [f"su{g}" for g in range(G)]
    in_names += ["x0"] if sx_i else [f"x{g}" for g in range(G)]
    if has_res:
        in_names.append("res")
    if bf16_tail_n > 0:
        in_names.append("w_bf")
    if router_tail_n > 0:
        in_names.append("w_rtr")
    if pss_in > 0:
        in_names.extend(["pss_in", "norm_w"])
    out_names = [f"out{g}" for g in range(G)]
    if bf16_tail_n > 0:
        out_names.append("out_bf")
    if router_tail_n > 0:
        out_names.append("out_rtr")
    if emit_i:
        out_names.append("pss_out")
    tag = (
        f"{out_dt}_{x_dt}_b{B}_g{G}_k{K}_r{int(has_res)}_s{int(is_swiglu)}_km{km_i}"
        f"_pi{pss_in}_po{emit_i}_bt{bf16_tail_n}_rt{router_tail_n}_sx{sx_i}_u{unroll}_wm{wf_mode}_br{b_reg}_{'_'.join(map(str, Ns + list(cfg)))}"
    )
    tag = tag.replace("mlx.core.", "")
    kern = mx.fast.metal_kernel(
        name=f"bv3_gemv_fp8_{tag}",
        input_names=in_names,
        output_names=out_names,
        header=_HEADER,
        source="\n".join(lines),
        ensure_row_contiguous=True,
    )
    _KERNELS[key] = (kern, start)
    return _KERNELS[key]


def gemv_fp8_rows_group(
    xs,
    ws_fp8,
    scs,
    *,
    cfg=None,
    coop=None,
    k_major: bool = False,
    part_ss=None,
    norm_w=None,
    norm_eps: float = 1e-6,
):
    """Independent problems out_i = xs[i] @ bf16(fp8_i * sc_i).T."""
    del coop
    xs, ws_fp8, scs = list(xs), list(ws_fp8), list(scs)
    B, K = xs[0].shape
    dt = xs[0].dtype
    assert 1 <= B <= MAX_B and K % 128 == 0
    assert all(x.shape == (B, K) and x.dtype == dt for x in xs)
    if k_major:
        Ns = [w.shape[0] * 4 for w in ws_fp8]
        assert all(w.ndim == 4 and w.shape[1] * 128 == K and w.dtype == mx.uint8 for w in ws_fp8)
    else:
        Ns = [w.shape[0] for w in ws_fp8]
        assert all(w.ndim == 2 and w.shape[1] == K and w.dtype == mx.uint8 for w in ws_fp8)
    assert all(s.dtype == mx.float32 for s in scs)
    pss_in = _validate_pss_in(part_ss, norm_w, B, K)
    if cfg is None:
        cfgs = [_fp8_cfg(n, K, B, False) for n in Ns]
        if len(set(cfgs)) > 1:
            out = [None] * len(ws_fp8)
            for c in dict.fromkeys(cfgs):
                idx = [i for i, cc in enumerate(cfgs) if cc == c]
                for i, o in zip(
                    idx,
                    gemv_fp8_rows_group(
                        [xs[i] for i in idx],
                        [ws_fp8[i] for i in idx],
                        [scs[i] for i in idx],
                        cfg=c,
                        k_major=k_major,
                        part_ss=part_ss,
                        norm_w=norm_w,
                        norm_eps=norm_eps,
                    ),
                ):
                    out[i] = o
            return out
        cfg = cfgs[0]
    bsg, _, bm_m, tm = cfg
    assert all(n % tm == 0 for n in Ns), f"Ns={Ns} not divisible by tm={tm}"
    G = len(ws_fp8)
    shared_x = G > 1 and all(x is xs[0] for x in xs)
    kern, ntg = _build_fp8_group(
        dt, dt, B, G, Ns, K, cfg, k_major=k_major, pss_in=pss_in, norm_eps=norm_eps, shared_x=shared_x
    )
    nsg = bsg * bm_m
    x_in = [xs[0]] if shared_x else xs
    inputs = ws_fp8 + scs + x_in + ([part_ss, norm_w] if pss_in > 0 else [])
    return list(
        kern(
            inputs=inputs,
            grid=(ntg * 32, nsg, 1),
            threadgroup=(32, nsg, 1),
            output_shapes=[(B, n) for n in Ns],
            output_dtypes=[dt] * G,
        )
    )


def gemv_fp8_rows(
    x,
    w_fp8,
    sc,
    *,
    cfg=None,
    coop=None,
    k_major: bool = False,
    part_ss=None,
    norm_w=None,
    norm_eps: float = 1e-6,
    return_part_ss: bool = False,
):
    if not return_part_ss:
        return gemv_fp8_rows_group(
            [x],
            [w_fp8],
            [sc],
            cfg=cfg,
            coop=coop,
            k_major=k_major,
            part_ss=part_ss,
            norm_w=norm_w,
            norm_eps=norm_eps,
        )[0]
    B, K = x.shape
    N = (w_fp8.shape[0] * 4) if k_major else w_fp8.shape[0]
    dt = x.dtype
    assert 1 <= B <= MAX_B and K % 128 == 0 and N % 128 == 0
    pss_in = _validate_pss_in(part_ss, norm_w, B, K)
    if cfg is None:
        cfg = _fp8_cfg(N, K, B, False, emit_pss=True)
    bsg, _, bm_m, tm = cfg
    blockM = bm_m * tm
    assert N % blockM == 0 and blockM % 4 == 0
    pss_cols = (N // 4) if tm == 4 else (N // blockM)
    kern, ntg = _build_fp8_group(
        dt, dt, B, 1, [N], K, cfg, k_major=k_major, pss_in=pss_in, emit_pss=True, norm_eps=norm_eps
    )
    nsg = bsg * bm_m
    inputs = [w_fp8, sc, x] + ([part_ss, norm_w] if pss_in > 0 else [])
    out, pss_out = kern(
        inputs=inputs,
        grid=(ntg * 32, nsg, 1),
        threadgroup=(32, nsg, 1),
        output_shapes=[(B, N), (B, pss_cols)],
        output_dtypes=[dt, mx.float32],
    )
    return out, pss_out


def gemv_fp8_rows_res(
    x,
    w_fp8,
    sc,
    res,
    *,
    cfg=None,
    coop=None,
    k_major: bool = False,
    part_ss=None,
    norm_w=None,
    norm_eps: float = 1e-6,
    return_part_ss: bool = False,
):
    del coop
    B, K = x.shape
    N = (w_fp8.shape[0] * 4) if k_major else w_fp8.shape[0]
    dt = x.dtype
    assert 1 <= B <= MAX_B and K % 128 == 0 and res.shape == (B, N) and res.dtype == dt
    assert w_fp8.dtype == mx.uint8 and sc.dtype == mx.float32
    pss_in = _validate_pss_in(part_ss, norm_w, B, K)
    emit_i = bool(return_part_ss)
    if cfg is None:
        cfg = _fp8_cfg(N, K, B, False, emit_pss=emit_i)
    bsg, _, bm_m, tm = cfg
    blockM = bm_m * tm
    assert N % tm == 0
    if emit_i:
        assert N % 128 == 0 and N % blockM == 0 and blockM % 4 == 0
    kern, ntg = _build_fp8_group(
        dt, dt, B, 1, [N], K, cfg, has_res=True, k_major=k_major, pss_in=pss_in, emit_pss=emit_i, norm_eps=norm_eps
    )
    nsg = bsg * bm_m
    inputs = [w_fp8, sc, x, res] + ([part_ss, norm_w] if pss_in > 0 else [])
    if emit_i:
        pss_cols = (N // 4) if tm == 4 else (N // blockM)
        out, pss_out = kern(
            inputs=inputs,
            grid=(ntg * 32, nsg, 1),
            threadgroup=(32, nsg, 1),
            output_shapes=[(B, N), (B, pss_cols)],
            output_dtypes=[dt, mx.float32],
        )
        return out, pss_out
    return kern(
        inputs=inputs,
        grid=(ntg * 32, nsg, 1),
        threadgroup=(32, nsg, 1),
        output_shapes=[(B, N)],
        output_dtypes=[dt],
    )[0]


def gemv_fp8_swiglu_rows(
    x,
    wg_fp8,
    sc_g,
    wu_fp8,
    sc_u,
    *,
    cfg=None,
    coop=None,
    k_major: bool = False,
    part_ss=None,
    norm_w=None,
    norm_eps: float = 1e-6,
):
    del coop
    B, K = x.shape
    N = (wg_fp8.shape[0] * 4) if k_major else wg_fp8.shape[0]
    assert 1 <= B <= MAX_B and K % 128 == 0 and x.dtype == mx.bfloat16
    assert wg_fp8.dtype == wu_fp8.dtype == mx.uint8 and sc_g.dtype == sc_u.dtype == mx.float32
    pss_in = _validate_pss_in(part_ss, norm_w, B, K)
    if cfg is None:
        cfg = _fp8_cfg(N, K, B, True)
    bsg, _, bm_m, tm = cfg
    assert N % tm == 0
    kern, ntg = _build_fp8_group(
        mx.bfloat16,
        mx.bfloat16,
        B,
        1,
        [N],
        K,
        cfg,
        is_swiglu=True,
        k_major=k_major,
        pss_in=pss_in,
        norm_eps=norm_eps,
    )
    nsg = bsg * bm_m
    inputs = [wg_fp8, sc_g, wu_fp8, sc_u, x] + ([part_ss, norm_w] if pss_in > 0 else [])
    return kern(
        inputs=inputs,
        grid=(ntg * 32, nsg, 1),
        threadgroup=(32, nsg, 1),
        output_shapes=[(B, N)],
        output_dtypes=[mx.bfloat16],
    )[0]


def gemv_fp8_with_bf16_tail_rows(
    x,
    ws_fp8,
    scs,
    w_bf16,
    *,
    cfg=None,
    k_major: bool = False,
    part_ss=None,
    norm_w=None,
    norm_eps: float = 1e-6,
):
    """Fused single-launch FP8 multi-projection + small BN=8 BF16 tail (KDA qkvfg+b_proj, MLA q_a+kv_a+g_proj)."""
    ws_fp8, scs = list(ws_fp8), list(scs)
    B, K = x.shape
    dt = x.dtype
    N_bf = w_bf16.shape[0]
    assert 1 <= B <= MAX_B and K % 1024 == 0 and K >= 16 * N_bf and dt == mx.bfloat16 and w_bf16.dtype == mx.bfloat16
    Ns = [(w.shape[0] * 4) if k_major else w.shape[0] for w in ws_fp8]
    pss_in = _validate_pss_in(part_ss, norm_w, B, K)
    if cfg is None:
        cfg = _fp8_cfg(max(Ns), K, B, False)
    bsg, _, bm_m, tm = cfg
    assert all(n % tm == 0 for n in Ns)
    G = len(ws_fp8)
    shared_x = G > 1
    kern, ntg = _build_fp8_group(
        dt,
        dt,
        B,
        G,
        Ns,
        K,
        cfg,
        k_major=k_major,
        pss_in=pss_in,
        norm_eps=norm_eps,
        bf16_tail_n=N_bf,
        shared_x=shared_x,
    )
    nsg = bsg * bm_m
    inputs = ws_fp8 + scs + [x] + [w_bf16] + ([part_ss, norm_w] if pss_in > 0 else [])
    return list(
        kern(
            inputs=inputs,
            grid=(ntg * 32, nsg, 1),
            threadgroup=(32, nsg, 1),
            output_shapes=[(B, n) for n in Ns] + [(B, N_bf)],
            output_dtypes=[dt] * (G + 1),
        )
    )


def gemv_router_se_swiglu_rows(
    x,
    router_w,
    se_gate_mod,
    se_up_mod,
    *,
    cfg=None,
    part_ss=None,
    norm_w=None,
    norm_eps: float = 1e-6,
):
    """Fused single-launch BF16->F32 router_gemv + FP8 shared_experts SwiGLU (bit-exact)."""
    if not (_is_fp8_mod(se_gate_mod) and _is_fp8_mod(se_up_mod)):
        logits = router_gemv_rows(x, router_w, part_ss=part_ss, norm_w=norm_w, norm_eps=norm_eps)
        se_mid = gemv_swiglu_rows(
            x, se_gate_mod.weight, se_up_mod.weight, part_ss=part_ss, norm_w=norm_w, norm_eps=norm_eps
        )
        return logits, se_mid
    B, K = x.shape
    N_se = se_gate_mod.weight_fp8.shape[0]
    N_rtr = router_w.shape[0]
    assert 1 <= B <= MAX_B and K % 128 == 0 and x.dtype == mx.bfloat16 and router_w.dtype == mx.bfloat16
    assert router_w.shape[1] == K
    pss_in = _validate_pss_in(part_ss, norm_w, B, K)
    if cfg is None:
        cfg = _fp8_cfg(N_se, K, B, True)
    bsg, _, bm_m, tm = cfg
    assert N_se % tm == 0
    kern, ntg = _build_fp8_group(
        mx.bfloat16,
        mx.bfloat16,
        B,
        1,
        [N_se],
        K,
        cfg,
        is_swiglu=True,
        pss_in=pss_in,
        norm_eps=norm_eps,
        router_tail_n=N_rtr,
    )
    nsg = bsg * bm_m
    inputs = [
        se_gate_mod.weight_fp8,
        se_gate_mod.weight_scale_inv,
        se_up_mod.weight_fp8,
        se_up_mod.weight_scale_inv,
        x,
        router_w,
    ] + ([part_ss, norm_w] if pss_in > 0 else [])
    se_mid, logits = kern(
        inputs=inputs,
        grid=(ntg * 32, nsg, 1),
        threadgroup=(32, nsg, 1),
        output_shapes=[(B, N_se), (B, N_rtr)],
        output_dtypes=[mx.bfloat16, mx.float32],
    )
    return logits, se_mid


def gemv_kda_qkvfgb_merged_rows(
    x,
    w_qkvfg,
    s_qkvfg,
    w_b,
    *,
    cfg=None,
    part_ss=None,
    norm_w=None,
    norm_eps: float = 1e-6,
):
    """Single-launch KDA qkvfg+b_proj reading ONE merged `[40960, 4096]` FP8 weight + `[320, 32]` scale
    and emitting 5 contiguous `(B, 8192)` outputs + `(B, 64)` `b_proj` (zero copies into `kda_pre_seq_rows`).
    """
    B, K = x.shape
    N_tot = w_qkvfg.shape[0]
    N_each = N_tot // 5
    N_bf = w_b.shape[0]
    assert 1 <= B <= MAX_B and K == 4096 and N_tot == 5 * N_each and N_each % 128 == 0
    assert x.dtype == mx.bfloat16 and w_qkvfg.dtype == mx.uint8 and s_qkvfg.dtype == mx.float32 and w_b.dtype == mx.bfloat16
    pss_in = _validate_pss_in(part_ss, norm_w, B, K)
    if cfg is None:
        cfg = _fp8_cfg(N_tot, K, B, False)
    bsg, b_sub, bm_m, tm = cfg
    b_reg = _b_reg_fp8(b_sub, tm, False)
    b_tg = b_sub - b_reg
    unroll, wf_mode = _fp8_unroll_wf(N_tot, K, B, cfg, b_reg, False, False, pss_in, mx.bfloat16)
    nsg = bsg * bm_m
    blockM = bm_m * tm
    assert N_each % blockM == 0
    nt_each = N_each // blockM
    fp8_tgs = 5 * nt_each
    ntg = fp8_tgs + N_bf
    eps_f = float(norm_eps) if pss_in > 0 else 0.0
    key = ("kda_merged", B, N_each, N_bf, K, cfg, b_reg, pss_in, eps_f, unroll, wf_mode)
    kern = _KERNELS.get(key)
    if kern is None:
        tg_acc_elems = max(1, (nsg * b_tg * tm * 32) if b_tg > 0 else 1, B * 8)
        eps_lit = flit(eps_f) if pss_in > 0 else "0.0f"
        pss_in_arg = "pss_in" if pss_in > 0 else "(const device float*)0"
        nw_arg = "norm_w" if pss_in > 0 else "(const device bfloat16_t*)0"
        lines = [
            f"threadgroup float tg_acc[{tg_acc_elems}];",
            "threadgroup bfloat16_t tg_y[1];",
            "const uint t = threadgroup_position_in_grid.x;",
            f"if (t < {N_bf}) {{",
            f"  bailing_gemv_bf16_bn8_tail_run<bfloat16_t, bfloat16_t, {B}, {nsg}, {K}, {pss_in}>"
            f"(w_bf, x0, {pss_in_arg}, {nw_arg}, {eps_lit}, out_bf, {N_bf}, tg_acc, t, "
            f"simdgroup_index_in_threadgroup, thread_index_in_simdgroup);",
            "  return;",
            "}",
            f"const uint tf = t - {N_bf};",
            f"const uint seg = tf / {nt_each};",
            f"const uint t_loc = tf - seg * {nt_each};",
            f"const device uchar* wp = w_cat + size_t(seg) * {N_each * K};",
            f"const device float* sp = s_cat + size_t(seg) * {(N_each // 128) * (K // 128)};",
            "device bfloat16_t* op = out0;",
            "if (seg == 1) op = out1;",
            "else if (seg == 2) op = out2;",
            "else if (seg == 3) op = out3;",
            "else if (seg == 4) op = out4;",
            f"bailing_gemv_fp8_rows_run<bfloat16_t, bfloat16_t, {B}, {bsg}, {b_sub}, {b_reg}, {bm_m}, {tm}, 0, 0, {K}, 0, {pss_in}, 0, {unroll}, {wf_mode}>"
            f"(wp, sp, wp, sp, x0, op, {pss_in_arg}, {nw_arg}, {eps_lit}, op, (device float*)0, {N_each}, tg_acc, tg_y, t_loc, "
            f"simdgroup_index_in_threadgroup, thread_index_in_simdgroup);",
        ]
        in_names = ["w_cat", "s_cat", "x0", "w_bf"] + (["pss_in", "norm_w"] if pss_in > 0 else [])
        out_names = ["out0", "out1", "out2", "out3", "out4", "out_bf"]
        tag = f"b{B}_n{N_each}_pi{pss_in}_u{unroll}_wm{wf_mode}_br{b_reg}_{'_'.join(map(str, cfg))}"
        kern = _KERNELS[key] = mx.fast.metal_kernel(
            name=f"bv3_kda_qkvfgb_m_{tag}",
            input_names=in_names,
            output_names=out_names,
            header=_HEADER,
            source="\n".join(lines),
            ensure_row_contiguous=True,
        )
    inputs = [w_qkvfg, s_qkvfg, x, w_b] + ([part_ss, norm_w] if pss_in > 0 else [])
    return list(
        kern(
            inputs=inputs,
            grid=(ntg * 32, nsg, 1),
            threadgroup=(32, nsg, 1),
            output_shapes=[(B, N_each)] * 5 + [(B, N_bf)],
            output_dtypes=[mx.bfloat16] * 6,
        )
    )


def attach_merged_kda_qkvfg(model) -> int:
    """Runtime-on-load merge of KDA `[q,k,v,f,g]` FP8 weights into `attn._w_qkvfg`/`attn._s_qkvfg` (0 extra RAM)."""
    merged = 0
    for layer in model.model.layers:
        if not getattr(layer, "is_linear", False):
            continue
        attn = layer.attention
        mods = [attn.q_proj, attn.k_proj, attn.v_proj, attn.f_proj, attn.g_proj]
        if not all(_is_fp8_mod(m) for m in mods):
            continue
        w_cat, s_cat = pack_qkvfg_fp8(mods, k_major=False)
        mx.eval(w_cat, s_cat)
        attn._w_qkvfg = w_cat
        attn._s_qkvfg = s_cat
        # Zero-extra-RAM: replace per-proj tensors with contiguous slices of the merged buffer!
        n_each = mods[0].weight_fp8.shape[0]
        sn_each = mods[0].weight_scale_inv.shape[0]
        for i, m in enumerate(mods):
            m.weight_fp8 = w_cat[i * n_each : (i + 1) * n_each]
            m.weight_scale_inv = s_cat[i * sn_each : (i + 1) * sn_each]
        merged += 1
    return merged


def _is_fp8_mod(mod) -> bool:
    return hasattr(mod, "weight_fp8")


def gemv_lin(x: mx.array, mod, *, part_ss=None, norm_w=None, norm_eps: float = 1e-6, return_part_ss: bool = False):
    if _is_fp8_mod(mod):
        return gemv_fp8_rows(
            x,
            mod.weight_fp8,
            mod.weight_scale_inv,
            k_major=bool(getattr(mod, "_k_major", False)),
            part_ss=part_ss,
            norm_w=norm_w,
            norm_eps=norm_eps,
            return_part_ss=return_part_ss,
        )
    return gemv_rows(
        x,
        mod.weight,
        part_ss=part_ss,
        norm_w=norm_w,
        norm_eps=norm_eps,
        return_part_ss=return_part_ss,
    )


def gemv_lin_res(
    x: mx.array, mod, res: mx.array, *, part_ss=None, norm_w=None, norm_eps: float = 1e-6, return_part_ss: bool = False
):
    if _is_fp8_mod(mod):
        return gemv_fp8_rows_res(
            x,
            mod.weight_fp8,
            mod.weight_scale_inv,
            res,
            k_major=bool(getattr(mod, "_k_major", False)),
            part_ss=part_ss,
            norm_w=norm_w,
            norm_eps=norm_eps,
            return_part_ss=return_part_ss,
        )
    return gemv_rows_res(
        x,
        mod.weight,
        res,
        part_ss=part_ss,
        norm_w=norm_w,
        norm_eps=norm_eps,
        return_part_ss=return_part_ss,
    )


def gemv_lin_swiglu(x: mx.array, gate_mod, up_mod, *, part_ss=None, norm_w=None, norm_eps: float = 1e-6):
    if _is_fp8_mod(gate_mod) and _is_fp8_mod(up_mod):
        return gemv_fp8_swiglu_rows(
            x,
            gate_mod.weight_fp8,
            gate_mod.weight_scale_inv,
            up_mod.weight_fp8,
            up_mod.weight_scale_inv,
            k_major=bool(getattr(gate_mod, "_k_major", False)),
            part_ss=part_ss,
            norm_w=norm_w,
            norm_eps=norm_eps,
        )
    return gemv_swiglu_rows(
        x, gate_mod.weight, up_mod.weight, part_ss=part_ss, norm_w=norm_w, norm_eps=norm_eps
    )


def gemv_lin_multi(x: mx.array, mods, *, part_ss=None, norm_w=None, norm_eps: float = 1e-6) -> list[mx.array]:
    mods = list(mods)
    fp8_idx = [i for i, m in enumerate(mods) if _is_fp8_mod(m)]
    bf_idx = [i for i, m in enumerate(mods) if not _is_fp8_mod(m)]
    # Fast fused path: FP8 group + 1 small BN=8 BF16 tail (KDA qkvfg+b_proj, MLA q_a+kv_a+g_proj)
    if (
        fp8_idx
        and len(bf_idx) == 1
        and x.dtype == mx.bfloat16
        and x.shape[1] % 1024 == 0
        and mods[bf_idx[0]].weight.dtype == mx.bfloat16
        and x.shape[1] >= 16 * mods[bf_idx[0]].weight.shape[0]
    ):
        km_flags = {bool(getattr(mods[i], "_k_major", False)) for i in fp8_idx}
        if len(km_flags) == 1:
            fused_outs = gemv_fp8_with_bf16_tail_rows(
                x,
                [mods[i].weight_fp8 for i in fp8_idx],
                [mods[i].weight_scale_inv for i in fp8_idx],
                mods[bf_idx[0]].weight,
                k_major=next(iter(km_flags)),
                part_ss=part_ss,
                norm_w=norm_w,
                norm_eps=norm_eps,
            )
            out = [None] * len(mods)
            for i, o in zip(fp8_idx, fused_outs[:-1]):
                out[i] = o
            out[bf_idx[0]] = fused_outs[-1]
            return out

    out = [None] * len(mods)
    if fp8_idx:
        fp8_outs = gemv_fp8_rows_group(
            [x] * len(fp8_idx),
            [mods[i].weight_fp8 for i in fp8_idx],
            [mods[i].weight_scale_inv for i in fp8_idx],
            k_major=bool(getattr(mods[fp8_idx[0]], "_k_major", False)),
            part_ss=part_ss,
            norm_w=norm_w,
            norm_eps=norm_eps,
        )
        for i, o in zip(fp8_idx, fp8_outs):
            out[i] = o
    if bf_idx:
        bf_outs = gemv_rows_group(
            [x] * len(bf_idx),
            [mods[i].weight for i in bf_idx],
            part_ss=part_ss,
            norm_w=norm_w,
            norm_eps=norm_eps,
        )
        for i, o in zip(bf_idx, bf_outs):
            out[i] = o
    return out
