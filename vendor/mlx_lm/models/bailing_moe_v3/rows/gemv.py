"""Exact multi-row bf16 / f32 GEMV row kernels for BailingMoeV3 (B = 1..24).

Row b of a B-row launch is bitwise identical to MLX 0.32.4 M == 1 (`x[b:b+1] @ w.T`,
which dispatches `GEMVKernel::run` in `mlx/backend/metal/kernels/gemv.h`), and every
weight matrix is read once across all B rows.

Also provides:
  - `gemv_rows_res(x, w, res)`: exact `gemv_rows(x, w) + res` (bf16)
  - `gemv_swiglu_rows(x, w_gate, w_up)`: exact `swiglu(gemv(x, w_gate), gemv(x, w_up))` (bf16)
  - `router_gemv_rows(x_bf16, w_bf16)`: exact `x.astype(f32) @ w.astype(f32).T` (reads w in bf16!)
  - Optional deterministic `part_ss` RMSNorm producer/consumer hooks (0 bit diffs vs `mx.fast.rms_norm`).
"""
import os

import mlx.core as mx

from .glue import GLUE_HEADER, flit

MAX_B = 24
_KERNELS = {}
_GEMV_BM_LARGE = int(os.environ.get("LINGSANTE_GEMV_BM", "8"))
_ZERO_F32 = mx.zeros((1,), dtype=mx.float32)
_ZERO_BF16 = mx.zeros((1,), dtype=mx.bfloat16)

_PART_SS_HEADER = r"""
template <int KK, int P>
METAL_FUNC float bv3_reduce_part_ss_inv(const device float* pss_row, float eps, uint lane) {
  GL_NOFMA
  constexpr int NSG_RMS = KK / 128;
  constexpr int PER_G = P / NSG_RMS;
  constexpr int G_PER_WAVE = 32 / PER_G;
  float grp_v = 0.0f;
  if constexpr (PER_G == 1) {
    grp_v = (int(lane) < NSG_RMS) ? pss_row[lane] : 0.0f;
  } else {
    for (int g0 = 0; g0 < NSG_RMS; g0 += G_PER_WAVE) {
      int local_g = int(lane) / PER_G;
      int g = g0 + local_g;
      int idx = g * PER_G + (int(lane) & (PER_G - 1));
      float v = (g < NSG_RMS) ? pss_row[idx] : 0.0f;
      if constexpr (PER_G >= 2)  v += simd_shuffle_xor(v, 1);
      if constexpr (PER_G >= 4)  v += simd_shuffle_xor(v, 2);
      if constexpr (PER_G >= 8)  v += simd_shuffle_xor(v, 4);
      if constexpr (PER_G >= 16) v += simd_shuffle_xor(v, 8);
      if constexpr (PER_G >= 32) v += simd_shuffle_xor(v, 16);
      if constexpr (G_PER_WAVE == 1) {
        if (int(lane) == g0) grp_v = v;
      } else {
#pragma unroll
        for (int k = 0; k < G_PER_WAVE; ++k) {
          int gk = g0 + k;
          if (gk < NSG_RMS) {
            float gv = simd_broadcast(v, ushort(k * PER_G));
            if (int(lane) == gk) grp_v = gv;
          }
        }
      }
    }
  }
  float tot = simd_sum(grp_v);
  return metal::precise::rsqrt(tot / float(KK) + eps);
}

METAL_FUNC float bv3_fma_ss4(bfloat16_t y0, bfloat16_t y1, bfloat16_t y2, bfloat16_t y3) {
  GL_NOFMA
  float f0 = static_cast<float>(y0);
  float f1 = static_cast<float>(y1);
  float f2 = static_cast<float>(y2);
  float f3 = static_cast<float>(y3);
  float acc = fma(f0, f0, 0.0f);
  acc = fma(f1, f1, acc);
  acc = fma(f2, f2, acc);
  acc = fma(f3, f3, acc);
  return acc;
}

METAL_FUNC bfloat4 bv3_apply_rms_bf4(bfloat4 px, bfloat4 nw, float inv) {
  GL_NOFMA
  float4 xf = float4(px);
  return bfloat4(
      nw.x * static_cast<bfloat16_t>(xf.x * inv),
      nw.y * static_cast<bfloat16_t>(xf.y * inv),
      nw.z * static_cast<bfloat16_t>(xf.z * inv),
      nw.w * static_cast<bfloat16_t>(xf.w * inv));
}
"""

_HEADER = (
    GLUE_HEADER
    + _PART_SS_HEADER
    + r"""
template <
    typename T,
    typename WT,
    typename XT,
    int B,
    int BSG,
    int B_SUB,
    int B_REG,
    int BM,
    int BN,
    int SM,
    int SN,
    int TM,
    int TN,
    int RES,
    int SWIGLU,
    int KK,
    int PSS_IN,
    int EMIT_PSS>
METAL_FUNC void bailing_gemv_rows_run(
    const device WT* __restrict mat,
    const device WT* __restrict mat_up,
    const device XT* __restrict in_vec,   // (B, KK) row-major
    const device T* __restrict res_vec,   // (B, N) row-major when RES == 1
    const device float* __restrict pss_in,
    const device bfloat16_t* __restrict norm_w,
    const float norm_eps,
    device T* __restrict out_vec,         // (B, N) row-major
    device float* __restrict pss_out,
    const int N,
    threadgroup float* tgp_memory,
    threadgroup bfloat16_t* tg_y,
    uint tgx,
    uint simd_gid,
    uint simd_lid) {
  constexpr int threadsM = BM * SM;
  constexpr int threadsN = BN * SN;
  constexpr int blockM = threadsM * TM;
  constexpr int blockN = threadsN * TN;
  constexpr int B_TG = B_SUB - B_REG;
  constexpr int TOT_R = SWIGLU ? (2 * TM) : TM;
  constexpr bool USE_BF4_W = (TN == 4 && sizeof(WT) == 2 && (KK % 4 == 0));
  constexpr bool USE_BF4_X = (TN == 4 && sizeof(XT) == 2 && (KK % 4 == 0));

  const int thrM = SN != 32 ? int(simd_lid) / SN : 0;
  const int thrN = SN != 32 ? int(simd_lid) % SN : int(simd_lid);
  const int sgN = BN != 1 ? int(simd_gid % BN) : 0;
  const int b_sg = (BN == 1 && BSG > 1) ? int(simd_gid % BSG) : 0;
  const int b0 = b_sg * B_SUB;
  const int simdM = (BN != 1) ? SM * int(simd_gid / BN) : (BSG > 1 ? SM * int(simd_gid / BSG) : int(SM * simd_gid));
  const int simdN = BN != 1 ? SN * int(simd_gid % BN) : 0;
  const int bm = (simdM + thrM) * TM;
  int bn = (simdN + thrN) * TN;

  int out_row = int(tgx) * blockM + bm;
  const bool row_valid = (out_row < N);
  if constexpr (BN == 1 && EMIT_PSS == 0) {
    if (!row_valid) return;
  }
  out_row = row_valid ? (out_row + TM <= N ? out_row : N - TM) : 0;
  mat += size_t(out_row) * KK;
  if constexpr (SWIGLU) mat_up += size_t(out_row) * KK;

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

  threadgroup float* my_tg = tgp_memory;
  if constexpr (BN == 1 && B_TG > 0) {
    my_tg = tgp_memory + size_t(simd_gid) * (B_TG * TOT_R * 32) + simd_lid;
#pragma unroll
    for (int ti = 0; ti < B_TG * TOT_R; ++ti) {
      my_tg[ti * 32] = 0.0f;
    }
  }

  constexpr int n_iter = KK / blockN;
  constexpr int last_iter = blockN * n_iter;
  constexpr int leftover = KK - last_iter;
  const uniform<int> n_iter_u = make_uniform(n_iter);

  if (row_valid) {
    for (int i = 0; i < n_iter_u; ++i) {
      bfloat4 nw_v;
      if constexpr (PSS_IN > 0 && USE_BF4_X) {
        nw_v = *(const device bfloat4*)(norm_w + bn);
      }

      if constexpr (USE_BF4_W) {
        bfloat4 inter[TM];
        bfloat4 inter_up[SWIGLU ? TM : 1];
#pragma unroll
        for (int tm = 0; tm < TM; tm++) {
          const size_t mo = size_t(tm) * KK + bn;
          inter[tm] = *(const device bfloat4*)(mat + mo);
          if constexpr (SWIGLU) {
            inter_up[tm] = *(const device bfloat4*)(mat_up + mo);
          }
        }
#pragma unroll
        for (int bi = 0; bi < B_REG; bi++) {
          const int b = b0 + bi;
          if (b < B) {
            float4 vc;
            if constexpr (USE_BF4_X) {
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
              float4 wv = float4(inter[tm]);
              float acc = result[bi][tm];
              acc += wv.x * vc.x;
              acc += wv.y * vc.y;
              acc += wv.z * vc.z;
              acc += wv.w * vc.w;
              result[bi][tm] = acc;
              if constexpr (SWIGLU) {
                float4 wvu = float4(inter_up[tm]);
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
        if constexpr (BN == 1 && B_TG > 0) {
#pragma unroll
          for (int ti = 0; ti < B_TG; ti++) {
            const int bi = B_REG + ti;
            const int b = b0 + bi;
            if (b < B) {
              float4 vc;
              if constexpr (USE_BF4_X) {
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
                float4 wv = float4(inter[tm]);
                const int slot = (ti * TOT_R + tm) * 32;
                float acc = my_tg[slot];
                acc += wv.x * vc.x;
                acc += wv.y * vc.y;
                acc += wv.z * vc.z;
                acc += wv.w * vc.w;
                my_tg[slot] = acc;
                if constexpr (SWIGLU) {
                  float4 wvu = float4(inter_up[tm]);
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
      } else {
        WT inter[TM][TN];
        WT inter_up[SWIGLU ? TM : 1][SWIGLU ? TN : 1];
#pragma unroll
        for (int tm = 0; tm < TM; tm++) {
          const size_t mo = size_t(tm) * KK + bn;
#pragma unroll
          for (int tn = 0; tn < TN; tn++) inter[tm][tn] = mat[mo + tn];
          if constexpr (SWIGLU) {
#pragma unroll
            for (int tn = 0; tn < TN; tn++) inter_up[tm][tn] = mat_up[mo + tn];
          }
        }
#pragma unroll
        for (int bi = 0; bi < B_REG; bi++) {
          const int b = b0 + bi;
          if (b < B) {
            float vc[TN];
#pragma unroll
            for (int tn = 0; tn < TN; tn++) {
              float xv = static_cast<float>(in_vec[size_t(b) * KK + bn + tn]);
              if constexpr (PSS_IN > 0) {
                xv = static_cast<float>(glue_rms_apply<bfloat16_t, bfloat16_t>(xv, inv_b[bi], norm_w[bn + tn]));
              }
              vc[tn] = xv;
            }
#pragma unroll
            for (int tm = 0; tm < TM; tm++) {
#pragma unroll
              for (int tn = 0; tn < TN; tn++) {
                result[bi][tm] += static_cast<float>(inter[tm][tn]) * vc[tn];
                if constexpr (SWIGLU) {
                  result_up[bi][tm] += static_cast<float>(inter_up[tm][tn]) * vc[tn];
                }
              }
            }
          }
        }
        if constexpr (BN == 1 && B_TG > 0) {
#pragma unroll
          for (int ti = 0; ti < B_TG; ti++) {
            const int bi = B_REG + ti;
            const int b = b0 + bi;
            if (b < B) {
              float vc[TN];
#pragma unroll
              for (int tn = 0; tn < TN; tn++) {
                float xv = static_cast<float>(in_vec[size_t(b) * KK + bn + tn]);
                if constexpr (PSS_IN > 0) {
                  xv = static_cast<float>(glue_rms_apply<bfloat16_t, bfloat16_t>(xv, inv_b[bi], norm_w[bn + tn]));
                }
                vc[tn] = xv;
              }
#pragma unroll
              for (int tm = 0; tm < TM; tm++) {
                const int slot = (ti * TOT_R + tm) * 32;
                float acc = my_tg[slot];
#pragma unroll
                for (int tn = 0; tn < TN; tn++) acc += static_cast<float>(inter[tm][tn]) * vc[tn];
                my_tg[slot] = acc;
                if constexpr (SWIGLU) {
                  const int slot_u = (ti * TOT_R + TM + tm) * 32;
                  float accu = my_tg[slot_u];
#pragma unroll
                  for (int tn = 0; tn < TN; tn++) accu += static_cast<float>(inter_up[tm][tn]) * vc[tn];
                  my_tg[slot_u] = accu;
                }
              }
            }
          }
        }
      }
      bn += blockN;
    }

    if constexpr (leftover > 0) {
      WT inter[TM][TN];
      WT inter_up[SWIGLU ? TM : 1][SWIGLU ? TN : 1];
#pragma unroll
      for (int tm = 0; tm < TM; tm++) {
#pragma unroll
        for (int tn = 0; tn < TN; tn++) {
          inter[tm][tn] = (bn + tn < KK) ? mat[size_t(tm) * KK + bn + tn] : WT(0);
          if constexpr (SWIGLU) {
            inter_up[tm][tn] = (bn + tn < KK) ? mat_up[size_t(tm) * KK + bn + tn] : WT(0);
          }
        }
      }
#pragma unroll
      for (int bi = 0; bi < B_REG; bi++) {
        const int b = b0 + bi;
        if (b < B) {
          float vc[TN];
#pragma unroll
          for (int tn = 0; tn < TN; tn++) {
            float xv = (bn + tn < KK) ? static_cast<float>(in_vec[size_t(b) * KK + bn + tn]) : 0.0f;
            if constexpr (PSS_IN > 0) {
              if (bn + tn < KK) {
                xv = static_cast<float>(glue_rms_apply<bfloat16_t, bfloat16_t>(xv, inv_b[bi], norm_w[bn + tn]));
              }
            }
            vc[tn] = xv;
          }
#pragma unroll
          for (int tm = 0; tm < TM; tm++) {
#pragma unroll
            for (int tn = 0; tn < TN; tn++) {
              result[bi][tm] += static_cast<float>(inter[tm][tn]) * vc[tn];
              if constexpr (SWIGLU) {
                result_up[bi][tm] += static_cast<float>(inter_up[tm][tn]) * vc[tn];
              }
            }
          }
        }
      }
      if constexpr (BN == 1 && B_TG > 0) {
#pragma unroll
        for (int ti = 0; ti < B_TG; ti++) {
          const int bi = B_REG + ti;
          const int b = b0 + bi;
          if (b < B) {
            float vc[TN];
#pragma unroll
            for (int tn = 0; tn < TN; tn++) {
              float xv = (bn + tn < KK) ? static_cast<float>(in_vec[size_t(b) * KK + bn + tn]) : 0.0f;
              if constexpr (PSS_IN > 0) {
                if (bn + tn < KK) {
                  xv = static_cast<float>(glue_rms_apply<bfloat16_t, bfloat16_t>(xv, inv_b[bi], norm_w[bn + tn]));
                }
              }
              vc[tn] = xv;
            }
#pragma unroll
            for (int tm = 0; tm < TM; tm++) {
              const int slot = (ti * TOT_R + tm) * 32;
              float acc = my_tg[slot];
#pragma unroll
              for (int tn = 0; tn < TN; tn++) acc += static_cast<float>(inter[tm][tn]) * vc[tn];
              my_tg[slot] = acc;
              if constexpr (SWIGLU) {
                const int slot_u = (ti * TOT_R + TM + tm) * 32;
                float accu = my_tg[slot_u];
#pragma unroll
                for (int tn = 0; tn < TN; tn++) accu += static_cast<float>(inter_up[tm][tn]) * vc[tn];
                my_tg[slot_u] = accu;
              }
            }
          }
        }
      }
    }
  }

  // Simdgroup reduction for register-resident rows
#pragma unroll
  for (int bi = 0; bi < B_REG; bi++) {
#pragma unroll
    for (int tm = 0; tm < TM; tm++) {
#pragma unroll
      for (ushort sn = (SN / 2); sn >= 1; sn >>= 1) {
        result[bi][tm] += simd_shuffle_down(result[bi][tm], sn);
        if constexpr (SWIGLU) result_up[bi][tm] += simd_shuffle_down(result_up[bi][tm], sn);
      }
    }
  }

  // Non-divergent single-barrier BN > 1 reduction (fixes Metal UB in divergent if (thrN == 0))
  if constexpr (BN > 1) {
    if (row_valid && thrN == 0) {
#pragma unroll
      for (int bi = 0; bi < B_REG; bi++) {
        const int b = b0 + bi;
        if (b < B) {
          threadgroup float* tgp_results = tgp_memory + (b * BN + sgN) * (blockM + TM) + bm;
#pragma unroll
          for (int tm = 0; tm < TM; tm++) tgp_results[tm] = result[bi][tm];
          if constexpr (SWIGLU) {
            threadgroup float* tgp_up = tgp_memory + (B * BN + b * BN + sgN) * (blockM + TM) + bm;
#pragma unroll
            for (int tm = 0; tm < TM; tm++) tgp_up[tm] = result_up[bi][tm];
          }
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row_valid && thrN == 0 && sgN == 0) {
#pragma unroll
      for (int bi = 0; bi < B_REG; bi++) {
        const int b = b0 + bi;
        if (b < B) {
          threadgroup float* tgp_results = tgp_memory + (b * BN) * (blockM + TM) + bm;
          for (int sgn = 1; sgn < BN; sgn++) {
#pragma unroll
            for (int tm = 0; tm < TM; tm++) {
              result[bi][tm] += tgp_results[sgn * (blockM + TM) + tm];
            }
          }
          if constexpr (SWIGLU) {
            threadgroup float* tgp_up = tgp_memory + (B * BN + b * BN) * (blockM + TM) + bm;
            for (int sgn = 1; sgn < BN; sgn++) {
#pragma unroll
              for (int tm = 0; tm < TM; tm++) {
                result_up[bi][tm] += tgp_up[sgn * (blockM + TM) + tm];
              }
            }
          }
        }
      }
    }
  }

  // Write register-resident outputs (and optional part_ss)
  if (row_valid && simdN == 0 && thrN == 0) {
#pragma unroll
    for (int bi = 0; bi < B_REG; bi++) {
      const int b = b0 + bi;
      if (b < B) {
        T y_loc[TM];
#pragma unroll
        for (int tm = 0; tm < TM; tm++) {
          T yv;
          if constexpr (SWIGLU) {
            bfloat16_t g = static_cast<bfloat16_t>(result[bi][tm]);
            bfloat16_t u = static_cast<bfloat16_t>(result_up[bi][tm]);
            yv = glue_swiglu_bf16(g, u);
          } else if constexpr (RES) {
            yv = static_cast<T>(result[bi][tm]) + res_vec[size_t(b) * N + out_row + tm];
          } else {
            yv = static_cast<T>(result[bi][tm]);
          }
          out_vec[size_t(b) * N + out_row + tm] = yv;
          y_loc[tm] = yv;
        }
        if constexpr (EMIT_PSS == 1) {
          if constexpr (TM == 4) {
            pss_out[size_t(b) * (N >> 2) + (out_row >> 2)] = bv3_fma_ss4(
                static_cast<bfloat16_t>(y_loc[0]),
                static_cast<bfloat16_t>(y_loc[1]),
                static_cast<bfloat16_t>(y_loc[2]),
                static_cast<bfloat16_t>(y_loc[3]));
          } else {
#pragma unroll
            for (int tm = 0; tm < TM; tm++) {
              tg_y[size_t(b) * blockM + bm + tm] = static_cast<bfloat16_t>(y_loc[tm]);
            }
          }
        }
      }
    }
  }

  // Reduce & write TG-overflow rows (only when BN == 1 && B_TG > 0)
  if constexpr (BN == 1 && B_TG > 0) {
#pragma unroll
    for (int ti = 0; ti < B_TG; ++ti) {
      const int b = b0 + B_REG + ti;
      if (b < B) {
        T y_loc[TM];
#pragma unroll
        for (int tm = 0; tm < TM; ++tm) {
          float acc = my_tg[(ti * TOT_R + tm) * 32];
#pragma unroll
          for (ushort sn = (SN / 2); sn >= 1; sn >>= 1) {
            acc += simd_shuffle_down(acc, sn);
          }
          float accu = 0.0f;
          if constexpr (SWIGLU) {
            accu = my_tg[(ti * TOT_R + TM + tm) * 32];
#pragma unroll
            for (ushort sn = (SN / 2); sn >= 1; sn >>= 1) {
              accu += simd_shuffle_down(accu, sn);
            }
          }
          if (row_valid && thrN == 0) {
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
          if (row_valid && thrN == 0) {
            if constexpr (TM == 4) {
              pss_out[size_t(b) * (N >> 2) + (out_row >> 2)] = bv3_fma_ss4(
                  static_cast<bfloat16_t>(y_loc[0]),
                  static_cast<bfloat16_t>(y_loc[1]),
                  static_cast<bfloat16_t>(y_loc[2]),
                  static_cast<bfloat16_t>(y_loc[3]));
            } else {
#pragma unroll
              for (int tm = 0; tm < TM; tm++) {
                tg_y[size_t(b) * blockM + bm + tm] = static_cast<bfloat16_t>(y_loc[tm]);
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


def _cfg(N: int, K: int, B: int = 1, is_swiglu: bool = False):
    """Exact block config MLX 0.32.4 picks for M == 1, tuned for B=1..24 (bsg=1 up to B=16!)."""
    tm, tn, sm, sn, bm, bn, bsg = 4, 4, 1, 32, 1, 1, 1
    bm = _GEMV_BM_LARGE if N >= 4096 else 4
    sn = 32
    if K <= 64:
        bm, sm, sn = 1, 8, 4
    elif K >= 16 * N:
        bm, bn = 1, 8
    if is_swiglu:
        if B >= 5:
            tm = 1
        if B >= 17 and bn == 1 and sm == 1 and sn == 32:
            bsg = 2
            bm = 4
    else:
        if N >= 100000:
            # lm_head (157184 x 4096): bm=8, bsg=1, tm=2 hits 706-729 GB/s across B=1..16!
            bm = 8 if B <= 16 else 4
            tm = 2
            bsg = 1 if B <= 16 else 2
        elif 5 <= B <= 16:
            if bn == 1 and sm == 1 and sn == 32:
                bsg = 1
                tm = 2
            else:
                tm = 1 if B >= 13 else 2
        elif B >= 17:
            if bn == 1 and sm == 1 and sn == 32:
                bsg = 2
                tm = 2
                bm = 4
            else:
                tm = 1
    tm = 1 if N < tm else tm
    return bm, bn, sm, sn, tm, tn, bsg


def _router_cfg(N: int, K: int, B: int):
    """80-core tuned router GEMV config for N=512, K=4096, BN=1."""
    if N == 512 and K == 4096:
        if B <= 4:
            return 4, 1, 1, 32, 4, 4, 1
        if B <= 12:
            return 2, 1, 1, 32, 2, 4, 1  # ntg=128
        return 2, 1, 1, 32, 1, 4, 1      # ntg=256 (9.7us vs 15.9us at B=16)
    return _cfg(N, K, B, False)


def _b_reg_for(B_SUB: int, bn: int, tm: int, is_swiglu: bool) -> int:
    if bn > 1:
        return B_SUB
    cap = 12 if is_swiglu else (16 if tm <= 2 else 8)
    return min(B_SUB, cap)


def _tname(dt):
    return {mx.bfloat16: "bfloat16_t", mx.float32: "float", mx.float16: "half"}[dt]


def _build_group(
    out_dt,
    w_dt,
    x_dt,
    B,
    G,
    Ns,
    K,
    cfg,
    has_res=False,
    is_swiglu=False,
    pss_in=0,
    emit_pss=False,
    norm_eps=1e-6,
):
    emit_i = int(bool(emit_pss))
    eps_f = float(norm_eps) if pss_in > 0 else 0.0
    key = (out_dt, w_dt, x_dt, B, G, tuple(Ns), K, cfg, has_res, is_swiglu, pss_in, emit_i, eps_f)
    if key in _KERNELS:
        return _KERNELS[key]
    bm, bn, sm, sn, tm, tn, bsg = cfg
    b_sub = (B + bsg - 1) // bsg
    b_reg = _b_reg_for(b_sub, bn, tm, is_swiglu)
    b_tg = b_sub - b_reg
    blockM = bm * sm * tm
    tot_r = (2 * tm) if is_swiglu else tm
    if bn > 1:
        tg_floats = (2 if is_swiglu else 1) * B * bn * (blockM + tm)
    elif b_tg > 0:
        tg_floats = (bm * bsg) * b_tg * tot_r * 32
    else:
        tg_floats = 1
    tg_y_elems = (B * blockM) if (emit_i and tm < 4) else 1
    T, WT, XT = _tname(out_dt), _tname(w_dt), _tname(x_dt)
    eps_lit = flit(eps_f) if pss_in > 0 else "0.0f"
    lines = [
        f"threadgroup float tgp_mem[{tg_floats}];",
        f"threadgroup bfloat16_t tg_y[{tg_y_elems}];",
        "const uint t = threadgroup_position_in_grid.x;",
    ]
    pss_in_arg = "pss_in" if pss_in > 0 else "(const device float*)0"
    nw_arg = "norm_w" if pss_in > 0 else "(const device bfloat16_t*)0"
    pss_out_arg = "pss_out" if emit_i else "(device float*)0"
    if G == 1:
        N0 = Ns[0]
        nt = (N0 + blockM - 1) // blockM
        wup = "u0" if is_swiglu else "w0"
        rv = "res" if has_res else "out0"
        lines.append(
            f"bailing_gemv_rows_run<{T}, {WT}, {XT}, {B}, {bsg}, {b_sub}, {b_reg}, {bm}, {bn}, {sm}, {sn}, {tm}, {tn}, {int(has_res)}, {int(is_swiglu)}, {K}, {pss_in}, {emit_i}>"
            f"(w0, {wup}, x0, {rv}, {pss_in_arg}, {nw_arg}, {eps_lit}, out0, {pss_out_arg}, {N0}, tgp_mem, tg_y, t, "
            f"simdgroup_index_in_threadgroup, thread_index_in_simdgroup);"
        )
        start = nt
    else:
        assert not emit_i, "emit_pss only supported for G == 1"
        lines.extend([
            f"const device {WT}* wp = w0;",
            f"const device {WT}* up_p = w0;",
            f"const device {XT}* xp = x0;",
            f"device {T}* op = out0;",
            f"const device {T}* rp = out0;",
            f"int n_cur = {Ns[0]};",
            "uint t_loc = t;",
        ])
        start = 0
        for g, N in enumerate(Ns):
            nt = (N + blockM - 1) // blockM
            kw = "if" if g == 0 else "else if"
            wup_s = f"up_p = u{g}; " if is_swiglu else ""
            rv_s = "rp = res; " if has_res else f"rp = out{g}; "
            if g < len(Ns) - 1:
                lines.append(
                    f"{kw} (t < {start + nt}) {{ wp = w{g}; {wup_s}xp = x{g}; {rv_s}op = out{g}; n_cur = {N}; t_loc = t - {start}; }}"
                )
            else:
                lines.append(
                    f"else {{ wp = w{g}; {wup_s}xp = x{g}; {rv_s}op = out{g}; n_cur = {N}; t_loc = t - {start}; }}"
                )
            start += nt
        lines.append(
            f"bailing_gemv_rows_run<{T}, {WT}, {XT}, {B}, {bsg}, {b_sub}, {b_reg}, {bm}, {bn}, {sm}, {sn}, {tm}, {tn}, {int(has_res)}, {int(is_swiglu)}, {K}, {pss_in}, 0>"
            f"(wp, up_p, xp, rp, {pss_in_arg}, {nw_arg}, {eps_lit}, op, {pss_out_arg}, n_cur, tgp_mem, tg_y, t_loc, "
            f"simdgroup_index_in_threadgroup, thread_index_in_simdgroup);"
        )
    in_names = [f"w{g}" for g in range(G)]
    if is_swiglu:
        in_names += [f"u{g}" for g in range(G)]
    in_names += [f"x{g}" for g in range(G)]
    if has_res:
        in_names.append("res")
    if pss_in > 0:
        in_names.extend(["pss_in", "norm_w"])
    out_names = [f"out{g}" for g in range(G)]
    if emit_i:
        out_names.append("pss_out")
    tag = (
        f"{out_dt}_{w_dt}_{x_dt}_b{B}_g{G}_k{K}_r{int(has_res)}_s{int(is_swiglu)}"
        f"_pi{pss_in}_po{emit_i}_br{b_reg}_{'_'.join(map(str, Ns + list(cfg)))}"
    )
    tag = tag.replace("mlx.core.", "")
    kern = mx.fast.metal_kernel(
        name=f"bv3_gemv_{tag}",
        input_names=in_names,
        output_names=out_names,
        header=_HEADER,
        source="\n".join(lines),
        ensure_row_contiguous=True,
    )
    _KERNELS[key] = (kern, start)
    return _KERNELS[key]


def _validate_pss_in(part_ss, norm_w, B: int, K: int) -> int:
    if part_ss is None:
        return 0
    assert norm_w is not None and norm_w.shape == (K,) and norm_w.dtype == mx.bfloat16
    assert part_ss.ndim == 2 and part_ss.shape[0] == B and part_ss.dtype == mx.float32
    P = int(part_ss.shape[1])
    assert K % 128 == 0
    nsg_rms = K // 128
    assert P % nsg_rms == 0 and (P // nsg_rms) in (1, 2, 4, 8, 16, 32), f"invalid part_ss P={P} for K={K}"
    return P


def gemv_rows_group(xs, ws, *, cfg=None, part_ss=None, norm_w=None, norm_eps: float = 1e-6):
    """Independent problems out_i = xs[i] @ ws[i].T; xs[i] (B, K), ws[i] (N_i, K) same dtype & K."""
    xs, ws = list(xs), list(ws)
    B, K = xs[0].shape
    dt = ws[0].dtype
    assert 1 <= B <= MAX_B
    assert all(x.shape == (B, K) and x.dtype == dt for x in xs)
    assert all(w.ndim == 2 and w.shape[1] == K and w.dtype == dt for w in ws)
    pss_in = _validate_pss_in(part_ss, norm_w, B, K)
    if cfg is None:
        cfgs = [_cfg(w.shape[0], K, B, False) for w in ws]
        if len(set(cfgs)) > 1:
            out = [None] * len(ws)
            for c in dict.fromkeys(cfgs):
                idx = [i for i, cc in enumerate(cfgs) if cc == c]
                for i, o in zip(
                    idx,
                    gemv_rows_group(
                        [xs[i] for i in idx],
                        [ws[i] for i in idx],
                        cfg=c,
                        part_ss=part_ss,
                        norm_w=norm_w,
                        norm_eps=norm_eps,
                    ),
                ):
                    out[i] = o
            return out
        cfg = cfgs[0]
    Ns = [w.shape[0] for w in ws]
    G = len(ws)
    kern, ntg = _build_group(dt, dt, dt, B, G, Ns, K, cfg, pss_in=pss_in, norm_eps=norm_eps)
    bm, bn, sm, sn, tm, tn, bsg = cfg
    y_dim = bn * bsg
    inputs = list(ws) + list(xs) + ([part_ss, norm_w] if pss_in > 0 else [])
    return list(
        kern(
            inputs=inputs,
            grid=(ntg * 32, y_dim, bm),
            threadgroup=(32, y_dim, bm),
            output_shapes=[(B, n) for n in Ns],
            output_dtypes=[dt] * G,
        )
    )


def gemv_rows(
    x,
    w,
    *,
    cfg=None,
    part_ss=None,
    norm_w=None,
    norm_eps: float = 1e-6,
    return_part_ss: bool = False,
):
    if not return_part_ss:
        return gemv_rows_group([x], [w], cfg=cfg, part_ss=part_ss, norm_w=norm_w, norm_eps=norm_eps)[0]
    B, K = x.shape
    N = w.shape[0]
    dt = w.dtype
    assert 1 <= B <= MAX_B and x.dtype == dt and w.shape[1] == K and N % 128 == 0
    pss_in = _validate_pss_in(part_ss, norm_w, B, K)
    if cfg is None:
        cfg = _cfg(N, K, B, False)
    bm, bn, sm, sn, tm, tn, bsg = cfg
    blockM = bm * sm * tm
    assert bn == 1 and N % blockM == 0 and blockM % 4 == 0
    pss_cols = (N // 4) if tm == 4 else (N // blockM)
    kern, ntg = _build_group(
        dt, dt, dt, B, 1, [N], K, cfg, pss_in=pss_in, emit_pss=True, norm_eps=norm_eps
    )
    y_dim = bn * bsg
    inputs = [w, x] + ([part_ss, norm_w] if pss_in > 0 else [])
    out, pss_out = kern(
        inputs=inputs,
        grid=(ntg * 32, y_dim, bm),
        threadgroup=(32, y_dim, bm),
        output_shapes=[(B, N), (B, pss_cols)],
        output_dtypes=[dt, mx.float32],
    )
    return out, pss_out


def gemv_rows_multi(x, ws, *, part_ss=None, norm_w=None, norm_eps: float = 1e-6):
    return gemv_rows_group([x] * len(ws), list(ws), part_ss=part_ss, norm_w=norm_w, norm_eps=norm_eps)


def gemv_rows_res(
    x,
    w,
    res,
    *,
    cfg=None,
    part_ss=None,
    norm_w=None,
    norm_eps: float = 1e-6,
    return_part_ss: bool = False,
):
    """Exact gemv_rows(x, w) + res for (B, K) bf16, (N, K) bf16, (B, N) bf16."""
    B, K = x.shape
    N = w.shape[0]
    dt = w.dtype
    assert 1 <= B <= MAX_B and x.dtype == dt and res.shape == (B, N) and res.dtype == dt and w.shape[1] == K
    pss_in = _validate_pss_in(part_ss, norm_w, B, K)
    if cfg is None:
        cfg = _cfg(N, K, B, False)
    bm, bn, sm, sn, tm, tn, bsg = cfg
    blockM = bm * sm * tm
    emit_i = bool(return_part_ss)
    if emit_i:
        assert bn == 1 and N % 128 == 0 and N % blockM == 0 and blockM % 4 == 0
    kern, ntg = _build_group(
        dt, dt, dt, B, 1, [N], K, cfg, has_res=True, pss_in=pss_in, emit_pss=emit_i, norm_eps=norm_eps
    )
    y_dim = bn * bsg
    inputs = [w, x, res] + ([part_ss, norm_w] if pss_in > 0 else [])
    if emit_i:
        pss_cols = (N // 4) if tm == 4 else (N // blockM)
        out, pss_out = kern(
            inputs=inputs,
            grid=(ntg * 32, y_dim, bm),
            threadgroup=(32, y_dim, bm),
            output_shapes=[(B, N), (B, pss_cols)],
            output_dtypes=[dt, mx.float32],
        )
        return out, pss_out
    return kern(
        inputs=inputs,
        grid=(ntg * 32, y_dim, bm),
        threadgroup=(32, y_dim, bm),
        output_shapes=[(B, N)],
        output_dtypes=[dt],
    )[0]


def gemv_swiglu_rows(x, w_gate, w_up, *, cfg=None, part_ss=None, norm_w=None, norm_eps: float = 1e-6):
    """Exact swiglu(x @ w_gate.T, x @ w_up.T) in one launch for (B, K) bf16."""
    B, K = x.shape
    N = w_gate.shape[0]
    assert 1 <= B <= MAX_B and x.dtype == w_gate.dtype == w_up.dtype == mx.bfloat16
    assert w_up.shape == (N, K) and w_gate.shape[1] == K
    pss_in = _validate_pss_in(part_ss, norm_w, B, K)
    if cfg is None:
        cfg = _cfg(N, K, B, True)
    kern, ntg = _build_group(
        mx.bfloat16, mx.bfloat16, mx.bfloat16, B, 1, [N], K, cfg, is_swiglu=True, pss_in=pss_in, norm_eps=norm_eps
    )
    bm, bn, sm, sn, tm, tn, bsg = cfg
    y_dim = bn * bsg
    inputs = [w_gate, w_up, x] + ([part_ss, norm_w] if pss_in > 0 else [])
    return kern(
        inputs=inputs,
        grid=(ntg * 32, y_dim, bm),
        threadgroup=(32, y_dim, bm),
        output_shapes=[(B, N)],
        output_dtypes=[mx.bfloat16],
    )[0]


def router_gemv_rows(x_bf16, w_bf16, *, cfg=None, part_ss=None, norm_w=None, norm_eps: float = 1e-6):
    """Exact x.astype(float32) @ w.astype(float32).T reading bf16 weights directly."""
    B, K = x_bf16.shape
    N = w_bf16.shape[0]
    assert 1 <= B <= MAX_B and x_bf16.dtype == mx.bfloat16 and w_bf16.dtype == mx.bfloat16 and w_bf16.shape[1] == K
    pss_in = _validate_pss_in(part_ss, norm_w, B, K)
    if cfg is None:
        cfg = _router_cfg(N, K, B)
    kern, ntg = _build_group(
        mx.float32, mx.bfloat16, mx.bfloat16, B, 1, [N], K, cfg, pss_in=pss_in, norm_eps=norm_eps
    )
    bm, bn, sm, sn, tm, tn, bsg = cfg
    y_dim = bn * bsg
    inputs = [w_bf16, x_bf16] + ([part_ss, norm_w] if pss_in > 0 else [])
    return kern(
        inputs=inputs,
        grid=(ntg * 32, y_dim, bm),
        threadgroup=(32, y_dim, bm),
        output_shapes=[(B, N)],
        output_dtypes=[mx.float32],
    )[0]
