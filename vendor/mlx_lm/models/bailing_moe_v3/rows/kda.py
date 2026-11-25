"""Bit-exact fused KDA decode, multi-token/varlen verify, and lazy-commit kernels (63 of 72 LingSante layers).

Optimizations (strictly 0 bit diffs vs stock B=1 BailingKDA):
1. `kda_pre_rows`, `kda_pre_seq_rows`, `kda_pre_varlen_rows`:
   - (h, t)-parallel FIR Conv1d(K=4) + silu + normalize_qk + compute_lower_bound_g + sigmoid(b),
     launching `64 * B` TGs (saturating all 80 M3 Ultra cores) with `bfloat4`/`float4` vector I/O.
   - Optional `emit_hist=False` drops `hist0..2` allocations; `kda_conv_commit_rows` slices conv
     cache directly from `(cs0..2, q_raw, k_raw, v_raw, n_consumed)`.
2. `kda_gdn_norm_gate_rows`, `kda_gdn_verify_seq[_split]_rows`, `kda_gdn_verify_varlen[_split]_rows`,
   `kda_gdn_commit_and_verify_seq[_split]_rows`, `kda_gdn_commit_and_verify_varlen[_split]_rows`,
   `kda_gdn_advance[_varlen][_split]_only`, `kda_o_norm_gate_rows`:
   - Zero-spill (~50 regs vs 136): stages `q, k, g` in threadgroup memory (`1.5 KB`).
   - 128-bit `8x float4` state load/store (`st_off` is 128B-aligned).
   - `NSG=4` (`256` TGs at `C=1` = 3.2 TGs/core on 80-core M3 Ultra) + deferred `kda_o_norm_gate_rows`.
   - Lazy fused commit+verify replays previous cycle's `prev_nc` steps and verifies current `L`/`Ls`
     steps in a single state read + single committed-state write (halving DRAM state traffic on rollback).
"""
import mlx.core as mx

from .glue import GLUE_HEADER, flit

_KERNELS = {}
_ZERO_U8 = None


def _dummy_u8() -> mx.array:
    global _ZERO_U8
    if _ZERO_U8 is None:
        _ZERO_U8 = mx.zeros((1,), dtype=mx.uint8)
    return _ZERO_U8


_KDA_HELPER_HDR = (
    GLUE_HEADER
    + r"""
inline float4 kda_load_bf4_f(const device bfloat16_t* p) {
  bfloat4 v = *(const device bfloat4*)p;
  return float4(v);
}

inline void kda_store_f_bf4(device bfloat16_t* p, float4 v) {
  *(device bfloat4*)p = bfloat4(v);
}
"""
)


# Single-step KDA pre-projection (fastdecode.py, B=1..4)
_PRE_SRC = r"""
    constexpr int NH = 64;
    constexpr int HD = 128;
    constexpr int PROJ = NH * HD;  // 8192
    const uint lane = thread_index_in_simdgroup;
    const uint h = threadgroup_position_in_grid.y;
    const uint b = threadgroup_position_in_grid.z;
    const size_t row_off = size_t(b) * PROJ + size_t(h) * HD;
    const int c0 = int(lane) * 4;

    bool active = true;
    if constexpr (HAS_MASK) {
      active = (mask[b] != 0);
    }

    const device T* cs0_r = cs0 + size_t(b) * 3 * PROJ + size_t(h) * HD;
    const device T* cs1_r = cs1 + size_t(b) * 3 * PROJ + size_t(h) * HD;
    const device T* cs2_r = cs2 + size_t(b) * 3 * PROJ + size_t(h) * HD;

    device T* ns0_r = ns0 + size_t(b) * 3 * PROJ + size_t(h) * HD;
    device T* ns1_r = ns1 + size_t(b) * 3 * PROJ + size_t(h) * HD;
    device T* ns2_r = ns2 + size_t(b) * 3 * PROJ + size_t(h) * HD;

    const float expA = metal::precise::exp(static_cast<float>(A_log[h]));
    if (lane == 0) {
      beta_out[size_t(b) * NH + h] = glue_sigmoid(static_cast<float>(b_raw[size_t(b) * NH + h]));
    }

    bfloat4 qx4 = active ? *(const device bfloat4*)(q_raw + row_off + c0) : bfloat4(0.0f);
    bfloat4 qp0 = *(const device bfloat4*)(cs0_r + c0);
    bfloat4 qp1 = *(const device bfloat4*)(cs0_r + PROJ + c0);
    bfloat4 qp2 = *(const device bfloat4*)(cs0_r + 2 * PROJ + c0);
    *(device bfloat4*)(ns0_r + c0) = qp1;
    *(device bfloat4*)(ns0_r + PROJ + c0) = qp2;
    *(device bfloat4*)(ns0_r + 2 * PROJ + c0) = qx4;

    bfloat4 kx4 = active ? *(const device bfloat4*)(k_raw + row_off + c0) : bfloat4(0.0f);
    bfloat4 kp0 = *(const device bfloat4*)(cs1_r + c0);
    bfloat4 kp1 = *(const device bfloat4*)(cs1_r + PROJ + c0);
    bfloat4 kp2 = *(const device bfloat4*)(cs1_r + 2 * PROJ + c0);
    *(device bfloat4*)(ns1_r + c0) = kp1;
    *(device bfloat4*)(ns1_r + PROJ + c0) = kp2;
    *(device bfloat4*)(ns1_r + 2 * PROJ + c0) = kx4;

    bfloat4 vx4 = active ? *(const device bfloat4*)(v_raw + row_off + c0) : bfloat4(0.0f);
    bfloat4 vp0 = *(const device bfloat4*)(cs2_r + c0);
    bfloat4 vp1 = *(const device bfloat4*)(cs2_r + PROJ + c0);
    bfloat4 vp2 = *(const device bfloat4*)(cs2_r + 2 * PROJ + c0);
    *(device bfloat4*)(ns2_r + c0) = vp1;
    *(device bfloat4*)(ns2_r + PROJ + c0) = vp2;
    *(device bfloat4*)(ns2_r + 2 * PROJ + c0) = vx4;

    bfloat4 f4 = *(const device bfloat4*)(f_raw + row_off + c0);
    float4 dt4 = *(const device float4*)(dt_bias + size_t(h) * HD + c0);

    bfloat4 q_silu, k_silu, v_silu;
    float4 g4;
    float q_acc = 0.0f;
    float k_acc = 0.0f;

#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const size_t gc = size_t(h) * HD + c0 + i;

      bfloat4 wq4 = *(const device bfloat4*)(wq + gc * 4);
      float qconv = 0.0f;
      qconv += static_cast<float>(qp0[i]) * static_cast<float>(wq4[0]);
      qconv += static_cast<float>(qp1[i]) * static_cast<float>(wq4[1]);
      qconv += static_cast<float>(qp2[i]) * static_cast<float>(wq4[2]);
      qconv += static_cast<float>(qx4[i]) * static_cast<float>(wq4[3]);
      T qcb = static_cast<T>(qconv);
      q_silu[i] = qcb * glue_sigmoid_bf16(qcb);
      float qf = static_cast<float>(q_silu[i]);
      q_acc = fma(qf, qf, q_acc);

      bfloat4 wk4 = *(const device bfloat4*)(wk + gc * 4);
      float kconv = 0.0f;
      kconv += static_cast<float>(kp0[i]) * static_cast<float>(wk4[0]);
      kconv += static_cast<float>(kp1[i]) * static_cast<float>(wk4[1]);
      kconv += static_cast<float>(kp2[i]) * static_cast<float>(wk4[2]);
      kconv += static_cast<float>(kx4[i]) * static_cast<float>(wk4[3]);
      T kcb = static_cast<T>(kconv);
      k_silu[i] = kcb * glue_sigmoid_bf16(kcb);
      float kf = static_cast<float>(k_silu[i]);
      k_acc = fma(kf, kf, k_acc);

      bfloat4 wv4 = *(const device bfloat4*)(wv + gc * 4);
      float vconv = 0.0f;
      vconv += static_cast<float>(vp0[i]) * static_cast<float>(wv4[0]);
      vconv += static_cast<float>(vp1[i]) * static_cast<float>(wv4[1]);
      vconv += static_cast<float>(vp2[i]) * static_cast<float>(wv4[2]);
      vconv += static_cast<float>(vx4[i]) * static_cast<float>(wv4[3]);
      T vcb = static_cast<T>(vconv);
      v_silu[i] = vcb * glue_sigmoid_bf16(vcb);

      float fval = static_cast<float>(f4[i]) + dt4[i];
      g4[i] = metal::precise::exp(LOWER_BOUND * glue_sigmoid(expA * fval));
    }

    *(device bfloat4*)(v_out + row_off + c0) = v_silu;
    *(device float4*)(g_out + row_off + c0) = g4;

    float q_tot = simd_sum(q_acc);
    float k_tot = simd_sum(k_acc);
    float q_inv = metal::precise::rsqrt(q_tot / 128.0f + RMS_EPS);
    float k_inv = metal::precise::rsqrt(k_tot / 128.0f + RMS_EPS);
    T q_sc = static_cast<T>(INV_SCALE_SQ);
    T k_sc = static_cast<T>(INV_SCALE);

    bfloat4 qo4, ko4;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      qo4[i] = q_sc * static_cast<T>(static_cast<float>(q_silu[i]) * q_inv);
      ko4[i] = k_sc * static_cast<T>(static_cast<float>(k_silu[i]) * k_inv);
    }
    *(device bfloat4*)(q_out + row_off + c0) = qo4;
    *(device bfloat4*)(k_out + row_off + c0) = ko4;
"""


# Deferred head-wise o_norm * sigmoid(g_raw) — grid=(32, 64, B), tg=(32, 1, 1)
_ONORM_GATE_SRC = r"""
    constexpr int NH = 64;
    constexpr int HD = 128;
    constexpr int PROJ = NH * HD;
    const uint lane = thread_index_in_simdgroup;
    const uint h = threadgroup_position_in_grid.y;
    const uint b = threadgroup_position_in_grid.z;
    const size_t head_off = size_t(b) * PROJ + size_t(h) * HD;
    const int c0 = int(lane) * 4;

    bfloat4 y4 = *(const device bfloat4*)(y_raw + head_off + c0);
    float4 yf = float4(y4);
    float acc = 0.0f;
    acc = fma(yf.x, yf.x, acc);
    acc = fma(yf.y, yf.y, acc);
    acc = fma(yf.z, yf.z, acc);
    acc = fma(yf.w, yf.w, acc);
    float inv = metal::precise::rsqrt(simd_sum(acc) / 128.0f + O_EPS);

    bfloat4 w4 = *(const device bfloat4*)(o_norm_w + c0);
    bfloat4 g4 = *(const device bfloat4*)(g_raw + head_off + c0);
    bfloat4 out4;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      T normed = static_cast<T>(w4[i]) * static_cast<T>(yf[i] * inv);
      T gate = static_cast<T>(glue_sigmoid(static_cast<float>(g4[i])));
      out4[i] = normed * gate;
    }
    *(device bfloat4*)(out_vec + head_off + c0) = out4;
"""


# Zero-spill, zero-TG single-step GDN (fastdecode.py): direct bfloat4/float4 loads + 8x float4 state I/O
_GDN_NOTG_SRC = r"""
    constexpr int D = 128;
    constexpr int H = 64;
    constexpr int PROJ = H * D;
    constexpr int LPR = 4;
    constexpr int RPS = 32 / LPR;  // 8 rows/sg
    constexpr int VPL = D / LPR;   // 32 elems/lane
    constexpr int PPL = VPL / 4;   // 8 4-elem chains

    const uint lane = thread_index_in_simdgroup;
    const uint n = threadgroup_position_in_grid.z;  // b * H + hv
    const uint b = n / H;
    const uint hv = n % H;
    const uint row_in_sg = lane / LPR;
    const uint lane_in_row = lane & (LPR - 1);

    const size_t head_off = size_t(b) * PROJ + size_t(hv) * D;
    const float bt = beta[size_t(b) * H + hv];
    const uint dv_idx = uint(threadgroup_position_in_grid.y) * RPS + row_in_sg;
    const uint k_base = lane_in_row * VPL;
    const size_t st_off = (size_t(n) * D + dv_idx) * D + k_base;
    const device float4* i_st4 = (const device float4*)(state_in + st_off);
    device float4* o_st4 = (device float4*)(state_out + st_off);
    const device bfloat4* k4p = (const device bfloat4*)(k + head_off + k_base);
    const device bfloat4* q4p = (const device bfloat4*)(q + head_off + k_base);
    const device float4* g4p = (const device float4*)(g + head_off + k_base);

    float4 st[PPL];
    float part[PPL];
#pragma unroll
    for (int pb = 0; pb < PPL; ++pb) {
      float4 s4 = i_st4[pb] * g4p[pb];
      st[pb] = s4;
      float4 k4 = float4(k4p[pb]);
      float acc = 0.0f;
      acc += s4.x * k4.x;
      acc += s4.y * k4.y;
      acc += s4.z * k4.z;
      acc += s4.w * k4.w;
      part[pb] = acc;
    }
    float kv_mem =
        ((part[0] + part[1]) + (part[2] + part[3])) +
        ((part[4] + part[5]) + (part[6] + part[7]));
    kv_mem += simd_shuffle_xor(kv_mem, 1);
    kv_mem += simd_shuffle_xor(kv_mem, 2);

    float delta = (static_cast<float>(v[head_off + dv_idx]) - kv_mem) * bt;

#pragma unroll
    for (int pb = 0; pb < PPL; ++pb) {
      float4 s4 = st[pb] + float4(k4p[pb]) * delta;
      o_st4[pb] = s4;
      float4 q4 = float4(q4p[pb]);
      float acc = 0.0f;
      acc += s4.x * q4.x;
      acc += s4.y * q4.y;
      acc += s4.z * q4.z;
      acc += s4.w * q4.w;
      part[pb] = acc;
    }
    float out_v =
        ((part[0] + part[1]) + (part[2] + part[3])) +
        ((part[4] + part[5]) + (part[6] + part[7]));
    out_v += simd_shuffle_xor(out_v, 1);
    out_v += simd_shuffle_xor(out_v, 2);

    if (lane_in_row == 0) {
      out_vec[head_off + dv_idx] = static_cast<T>(out_v);
    }
"""

# Zero-spill single-step GDN (fastdecode.py): TG-staged q/k/g + 8x float4 state I/O
_GDN_FUSED_SRC = r"""
    constexpr int D = 128;
    constexpr int H = 64;
    constexpr int PROJ = H * D;
    constexpr int LPR = 4;
    constexpr int RPS = 32 / LPR;  // 8 rows/sg
    constexpr int VPL = D / LPR;   // 32 elems/lane
    constexpr int PPL = VPL / 4;   // 8 4-elem chains
    constexpr int RPTG = RPS * NSG;

    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint ttid = sg * 32 + lane;
    const uint n = threadgroup_position_in_grid.z;  // b * H + hv
    const uint b = n / H;
    const uint hv = n % H;
    const uint row_in_sg = lane / LPR;
    const uint lane_in_row = lane & (LPR - 1);

    const size_t head_off = size_t(b) * PROJ + size_t(hv) * D;
    const float bt = beta[size_t(b) * H + hv];

    threadgroup float tg_q[D];
    threadgroup float tg_k[D];
    threadgroup float tg_g[D];
    threadgroup T y_tg[FUSED ? D : 1];

    for (uint i = ttid; i < D; i += NSG * 32) {
      tg_q[i] = static_cast<float>(q[head_off + i]);
      tg_k[i] = static_cast<float>(k[head_off + i]);
      tg_g[i] = g[head_off + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint dv_idx = uint(threadgroup_position_in_grid.y) * RPTG + sg * RPS + row_in_sg;
    const uint k_base = lane_in_row * VPL;
    const size_t st_off = (size_t(n) * D + dv_idx) * D + k_base;
    const device float4* i_st4 = (const device float4*)(state_in + st_off);
    device float4* o_st4 = (device float4*)(state_out + st_off);

    float4 st[PPL];
    float part[PPL];
#pragma unroll
    for (int pb = 0; pb < PPL; ++pb) {
      float4 s4 = i_st4[pb];
      float4 g4 = *(const threadgroup float4*)(tg_g + k_base + pb * 4);
      float4 k4 = *(const threadgroup float4*)(tg_k + k_base + pb * 4);
      s4 = s4 * g4;
      st[pb] = s4;
      float acc = 0.0f;
      acc += s4.x * k4.x;
      acc += s4.y * k4.y;
      acc += s4.z * k4.z;
      acc += s4.w * k4.w;
      part[pb] = acc;
    }
    float kv_mem =
        ((part[0] + part[1]) + (part[2] + part[3])) +
        ((part[4] + part[5]) + (part[6] + part[7]));
    kv_mem += simd_shuffle_xor(kv_mem, 1);
    kv_mem += simd_shuffle_xor(kv_mem, 2);

    float delta = (static_cast<float>(v[head_off + dv_idx]) - kv_mem) * bt;

#pragma unroll
    for (int pb = 0; pb < PPL; ++pb) {
      float4 k4 = *(const threadgroup float4*)(tg_k + k_base + pb * 4);
      float4 q4 = *(const threadgroup float4*)(tg_q + k_base + pb * 4);
      float4 s4 = st[pb] + k4 * delta;
      o_st4[pb] = s4;
      float acc = 0.0f;
      acc += s4.x * q4.x;
      acc += s4.y * q4.y;
      acc += s4.z * q4.z;
      acc += s4.w * q4.w;
      part[pb] = acc;
    }
    float out_v =
        ((part[0] + part[1]) + (part[2] + part[3])) +
        ((part[4] + part[5]) + (part[6] + part[7]));
    out_v += simd_shuffle_xor(out_v, 1);
    out_v += simd_shuffle_xor(out_v, 2);

    if (lane_in_row == 0) {
      if constexpr (FUSED) {
        y_tg[dv_idx] = static_cast<T>(out_v);
      } else {
        out_vec[head_off + dv_idx] = static_cast<T>(out_v);
      }
    }

    if constexpr (FUSED) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sg == 0) {
        const int c0 = int(lane) * 4;
        float acc = 0.0f;
        float vals[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          vals[i] = static_cast<float>(y_tg[c0 + i]);
          acc = fma(vals[i], vals[i], acc);
        }
        float inv = metal::precise::rsqrt(simd_sum(acc) / 128.0f + O_EPS);
        bfloat4 w4 = *(const device bfloat4*)(o_norm_w + c0);
        bfloat4 g4 = *(const device bfloat4*)(g_raw + head_off + c0);
        bfloat4 out4;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          T normed = static_cast<T>(w4[i]) * static_cast<T>(vals[i] * inv);
          T gate = static_cast<T>(glue_sigmoid(static_cast<float>(g4[i])));
          out4[i] = normed * gate;
        }
        *(device bfloat4*)(out_vec + head_off + c0) = out4;
      }
    }
"""


# (h, t)-parallel FIR Conv1d(K=4) + silu + normalize_qk + lower_bound_g + beta
# Grid: (32, 64, B), threadgroup: (32, 1, 1). Supports both uniform (C, L) and varlen Ls!
def _pre_parallel_src(varlen: bool, emit_hist: bool) -> str:
    idx_setup = (
        """
    const uint b = threadgroup_position_in_grid.z;
    const int c = stream_id[b];
    const int t = step_in_stream[b];
"""
        if varlen
        else """
    const uint b = threadgroup_position_in_grid.z;
    const int c = int(b) / LL;
    const int t = int(b) % LL;
"""
    )
    hist_write = (
        """
    device T* hist0_r = hist0 + size_t(c) * (3 + LL) * PROJ + h_off;
    device T* hist1_r = hist1 + size_t(c) * (3 + LL) * PROJ + h_off;
    device T* hist2_r = hist2 + size_t(c) * (3 + LL) * PROJ + h_off;
    if (t == 0) {
      *(device bfloat4*)(hist0_r + c0) = *(const device bfloat4*)(cs0_r + c0);
      *(device bfloat4*)(hist0_r + PROJ + c0) = *(const device bfloat4*)(cs0_r + PROJ + c0);
      *(device bfloat4*)(hist0_r + 2 * PROJ + c0) = *(const device bfloat4*)(cs0_r + 2 * PROJ + c0);
      *(device bfloat4*)(hist1_r + c0) = *(const device bfloat4*)(cs1_r + c0);
      *(device bfloat4*)(hist1_r + PROJ + c0) = *(const device bfloat4*)(cs1_r + PROJ + c0);
      *(device bfloat4*)(hist1_r + 2 * PROJ + c0) = *(const device bfloat4*)(cs1_r + 2 * PROJ + c0);
      *(device bfloat4*)(hist2_r + c0) = *(const device bfloat4*)(cs2_r + c0);
      *(device bfloat4*)(hist2_r + PROJ + c0) = *(const device bfloat4*)(cs2_r + PROJ + c0);
      *(device bfloat4*)(hist2_r + 2 * PROJ + c0) = *(const device bfloat4*)(cs2_r + 2 * PROJ + c0);
    }
    *(device bfloat4*)(hist0_r + size_t(3 + t) * PROJ + c0) = qx4;
    *(device bfloat4*)(hist1_r + size_t(3 + t) * PROJ + c0) = kx4;
    *(device bfloat4*)(hist2_r + size_t(3 + t) * PROJ + c0) = vx4;
"""
        if emit_hist
        else ""
    )
    return (
        r"""
    constexpr int NH = 64;
    constexpr int HD = 128;
    constexpr int PROJ = NH * HD;  // 8192
    const uint lane = thread_index_in_simdgroup;
    const uint h = threadgroup_position_in_grid.y;
    const size_t h_off = size_t(h) * HD;
    const int c0 = int(lane) * 4;
"""
        + idx_setup
        + r"""
    const size_t row_off = size_t(b) * PROJ + h_off;
    const device T* cs0_r = cs0 + size_t(c) * 3 * PROJ + h_off;
    const device T* cs1_r = cs1 + size_t(c) * 3 * PROJ + h_off;
    const device T* cs2_r = cs2 + size_t(c) * 3 * PROJ + h_off;

    if (lane == 0) {
      beta_out[size_t(b) * NH + h] = glue_sigmoid(static_cast<float>(b_raw[size_t(b) * NH + h]));
    }

    bfloat4 qx4 = *(const device bfloat4*)(q_raw + row_off + c0);
    bfloat4 kx4 = *(const device bfloat4*)(k_raw + row_off + c0);
    bfloat4 vx4 = *(const device bfloat4*)(v_raw + row_off + c0);

    // Direct 4-tap FIR lookup: tap j in {0,1,2} maps to t+j-3 (<0 -> cs[3+(t+j-3)], >=0 -> raw[b+j-3])
    bfloat4 qp0 = (t < 3) ? *(const device bfloat4*)(cs0_r + size_t(t) * PROJ + c0)
                          : *(const device bfloat4*)(q_raw + size_t(b - 3) * PROJ + h_off + c0);
    bfloat4 qp1 = (t < 2) ? *(const device bfloat4*)(cs0_r + size_t(t + 1) * PROJ + c0)
                          : *(const device bfloat4*)(q_raw + size_t(b - 2) * PROJ + h_off + c0);
    bfloat4 qp2 = (t < 1) ? *(const device bfloat4*)(cs0_r + size_t(t + 2) * PROJ + c0)
                          : *(const device bfloat4*)(q_raw + size_t(b - 1) * PROJ + h_off + c0);

    bfloat4 kp0 = (t < 3) ? *(const device bfloat4*)(cs1_r + size_t(t) * PROJ + c0)
                          : *(const device bfloat4*)(k_raw + size_t(b - 3) * PROJ + h_off + c0);
    bfloat4 kp1 = (t < 2) ? *(const device bfloat4*)(cs1_r + size_t(t + 1) * PROJ + c0)
                          : *(const device bfloat4*)(k_raw + size_t(b - 2) * PROJ + h_off + c0);
    bfloat4 kp2 = (t < 1) ? *(const device bfloat4*)(cs1_r + size_t(t + 2) * PROJ + c0)
                          : *(const device bfloat4*)(k_raw + size_t(b - 1) * PROJ + h_off + c0);

    bfloat4 vp0 = (t < 3) ? *(const device bfloat4*)(cs2_r + size_t(t) * PROJ + c0)
                          : *(const device bfloat4*)(v_raw + size_t(b - 3) * PROJ + h_off + c0);
    bfloat4 vp1 = (t < 2) ? *(const device bfloat4*)(cs2_r + size_t(t + 1) * PROJ + c0)
                          : *(const device bfloat4*)(v_raw + size_t(b - 2) * PROJ + h_off + c0);
    bfloat4 vp2 = (t < 1) ? *(const device bfloat4*)(cs2_r + size_t(t + 2) * PROJ + c0)
                          : *(const device bfloat4*)(v_raw + size_t(b - 1) * PROJ + h_off + c0);
"""
        + hist_write
        + r"""
    const float expA = metal::precise::exp(static_cast<float>(A_log[h]));
    bfloat4 f4 = *(const device bfloat4*)(f_raw + row_off + c0);
    float4 dt4 = *(const device float4*)(dt_bias + h_off + c0);

    bfloat4 q_silu, k_silu, v_silu;
    float4 g4;
    float q_acc = 0.0f;
    float k_acc = 0.0f;

#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const size_t gc = h_off + c0 + i;

      bfloat4 wq4 = *(const device bfloat4*)(wq + gc * 4);
      float qconv = 0.0f;
      qconv += static_cast<float>(qp0[i]) * static_cast<float>(wq4[0]);
      qconv += static_cast<float>(qp1[i]) * static_cast<float>(wq4[1]);
      qconv += static_cast<float>(qp2[i]) * static_cast<float>(wq4[2]);
      qconv += static_cast<float>(qx4[i]) * static_cast<float>(wq4[3]);
      T qcb = static_cast<T>(qconv);
      q_silu[i] = qcb * glue_sigmoid_bf16(qcb);
      float qf = static_cast<float>(q_silu[i]);
      q_acc = fma(qf, qf, q_acc);

      bfloat4 wk4 = *(const device bfloat4*)(wk + gc * 4);
      float kconv = 0.0f;
      kconv += static_cast<float>(kp0[i]) * static_cast<float>(wk4[0]);
      kconv += static_cast<float>(kp1[i]) * static_cast<float>(wk4[1]);
      kconv += static_cast<float>(kp2[i]) * static_cast<float>(wk4[2]);
      kconv += static_cast<float>(kx4[i]) * static_cast<float>(wk4[3]);
      T kcb = static_cast<T>(kconv);
      k_silu[i] = kcb * glue_sigmoid_bf16(kcb);
      float kf = static_cast<float>(k_silu[i]);
      k_acc = fma(kf, kf, k_acc);

      bfloat4 wv4 = *(const device bfloat4*)(wv + gc * 4);
      float vconv = 0.0f;
      vconv += static_cast<float>(vp0[i]) * static_cast<float>(wv4[0]);
      vconv += static_cast<float>(vp1[i]) * static_cast<float>(wv4[1]);
      vconv += static_cast<float>(vp2[i]) * static_cast<float>(wv4[2]);
      vconv += static_cast<float>(vx4[i]) * static_cast<float>(wv4[3]);
      T vcb = static_cast<T>(vconv);
      v_silu[i] = vcb * glue_sigmoid_bf16(vcb);

      float fval = static_cast<float>(f4[i]) + dt4[i];
      g4[i] = metal::precise::exp(LOWER_BOUND * glue_sigmoid(expA * fval));
    }

    *(device bfloat4*)(v_out + row_off + c0) = v_silu;
    *(device float4*)(g_out + row_off + c0) = g4;

    float q_tot = simd_sum(q_acc);
    float k_tot = simd_sum(k_acc);
    float q_inv = metal::precise::rsqrt(q_tot / 128.0f + RMS_EPS);
    float k_inv = metal::precise::rsqrt(k_tot / 128.0f + RMS_EPS);
    const T q_sc = static_cast<T>(INV_SCALE_SQ);
    const T k_sc = static_cast<T>(INV_SCALE);

    bfloat4 qo4, ko4;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      qo4[i] = q_sc * static_cast<T>(static_cast<float>(q_silu[i]) * q_inv);
      ko4[i] = k_sc * static_cast<T>(static_cast<float>(k_silu[i]) * k_inv);
    }
    *(device bfloat4*)(q_out + row_off + c0) = qo4;
    *(device bfloat4*)(k_out + row_off + c0) = ko4;
"""
    )


# Unified zero-spill GDN verify (+ optional lazy commit + optional varlen + optional split states)
# Supports both zero-TG/zero-barrier `no_tg=True` (nsg=1, 1024 TGs at C=1, fastest on M3 Ultra)
# and TG-staged `no_tg=False` (nsg in {2, 4}).
def _gdn_verify_src(split_c: int, has_prev: bool, write_end: bool, varlen: bool, no_tg: bool = True) -> str:
    head = r"""
    constexpr int D = 128;
    constexpr int H = 64;
    constexpr int PROJ = H * D;
    constexpr int LPR = 4;
    constexpr int RPS = 32 / LPR;  // 8 rows/sg
    constexpr int VPL = D / LPR;   // 32 elems/lane
    constexpr int PPL = VPL / 4;   // 8 4-elem chains
    constexpr int RPTG = RPS * NSG;

    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint ttid = sg * 32 + lane;
    const uint n = threadgroup_position_in_grid.z;  // c * H + hv
    const uint c = n / H;
    const uint hv = n % H;
    const uint row_in_sg = lane / LPR;
    const uint lane_in_row = lane & (LPR - 1);
    const uint dv_idx = uint(threadgroup_position_in_grid.y) * RPTG + sg * RPS + row_in_sg;
    const uint k_base = lane_in_row * VPL;
"""
    if split_c == 0:
        ptr = """
    const size_t st_off = (size_t(n) * D + dv_idx) * D + k_base;
    const device float4* i_st4 = (const device float4*)(state_in + st_off);
"""
        if has_prev:
            ptr += "    device float4* c_st4 = (device float4*)(state_commit + st_off);\n"
        if write_end:
            ptr += "    device float4* o_st4 = (device float4*)(state_out + st_off);\n"
    else:
        ptr = f"""
    const size_t st_off = (size_t(hv) * D + dv_idx) * D + k_base;
    const device float4* i_st4 = (const device float4*)(si0 + st_off);
"""
        if has_prev:
            ptr += "    device float4* c_st4 = (device float4*)(sc0 + st_off);\n"
        if write_end:
            ptr += "    device float4* o_st4 = (device float4*)(so0 + st_off);\n"
        for i in range(1, split_c):
            kw = "if" if i == 1 else "else if"
            sets = [f"i_st4 = (const device float4*)(si{i} + st_off);"]
            if has_prev:
                sets.append(f"c_st4 = (device float4*)(sc{i} + st_off);")
            if write_end:
                sets.append(f"o_st4 = (device float4*)(so{i} + st_off);")
            ptr += f"    {kw} (c == {i}) {{ {' '.join(sets)} }}\n"

    load_st = r"""
    float4 st[PPL];
#pragma unroll
    for (int pb = 0; pb < PPL; ++pb) {
      st[pb] = i_st4[pb];
    }
""" + (
        ""
        if no_tg
        else r"""
    threadgroup float tg_q[D];
    threadgroup float tg_k[D];
    threadgroup float tg_g[D];
"""
    )
    prev_block = ""
    if has_prev:
        if no_tg:
            prev_block = r"""
    const int p_nc = prev_nc[c];
    const int p_off = prev_off[c];
    for (int t = 0; t < p_nc; ++t) {
      const size_t pb_idx = size_t(p_off + t);
      const size_t head_off = pb_idx * PROJ + size_t(hv) * D;
      const float bt = prev_beta[pb_idx * H + hv];
      const device bfloat4* k4p = (const device bfloat4*)(prev_k + head_off + k_base);
      const device float4* g4p = (const device float4*)(prev_g + head_off + k_base);

      float part[PPL];
#pragma unroll
      for (int pb = 0; pb < PPL; ++pb) {
        float4 s4 = st[pb] * g4p[pb];
        st[pb] = s4;
        float4 k4 = float4(k4p[pb]);
        float acc = 0.0f;
        acc += s4.x * k4.x;
        acc += s4.y * k4.y;
        acc += s4.z * k4.z;
        acc += s4.w * k4.w;
        part[pb] = acc;
      }
      float kv_mem =
          ((part[0] + part[1]) + (part[2] + part[3])) +
          ((part[4] + part[5]) + (part[6] + part[7]));
      kv_mem += simd_shuffle_xor(kv_mem, 1);
      kv_mem += simd_shuffle_xor(kv_mem, 2);

      float delta = (static_cast<float>(prev_v[head_off + dv_idx]) - kv_mem) * bt;

#pragma unroll
      for (int pb = 0; pb < PPL; ++pb) {
        st[pb] = st[pb] + float4(k4p[pb]) * delta;
      }
    }

#pragma unroll
    for (int pb = 0; pb < PPL; ++pb) {
      c_st4[pb] = st[pb];
    }
"""
        else:
            prev_block = r"""
    const int p_nc = prev_nc[c];
    const int p_off = prev_off[c];
    for (int t = 0; t < p_nc; ++t) {
      const size_t pb_idx = size_t(p_off + t);
      const size_t head_off = pb_idx * PROJ + size_t(hv) * D;
      const float bt = prev_beta[pb_idx * H + hv];
      for (uint i = ttid; i < D; i += NSG * 32) {
        tg_k[i] = static_cast<float>(prev_k[head_off + i]);
        tg_g[i] = prev_g[head_off + i];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      float part[PPL];
#pragma unroll
      for (int pb = 0; pb < PPL; ++pb) {
        float4 g4 = *(const threadgroup float4*)(tg_g + k_base + pb * 4);
        float4 k4 = *(const threadgroup float4*)(tg_k + k_base + pb * 4);
        float4 s4 = st[pb] * g4;
        st[pb] = s4;
        float acc = 0.0f;
        acc += s4.x * k4.x;
        acc += s4.y * k4.y;
        acc += s4.z * k4.z;
        acc += s4.w * k4.w;
        part[pb] = acc;
      }
      float kv_mem =
          ((part[0] + part[1]) + (part[2] + part[3])) +
          ((part[4] + part[5]) + (part[6] + part[7]));
      kv_mem += simd_shuffle_xor(kv_mem, 1);
      kv_mem += simd_shuffle_xor(kv_mem, 2);

      float delta = (static_cast<float>(prev_v[head_off + dv_idx]) - kv_mem) * bt;

#pragma unroll
      for (int pb = 0; pb < PPL; ++pb) {
        float4 k4 = *(const threadgroup float4*)(tg_k + k_base + pb * 4);
        st[pb] = st[pb] + k4 * delta;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

#pragma unroll
    for (int pb = 0; pb < PPL; ++pb) {
      c_st4[pb] = st[pb];
    }
"""

    cur_bounds = (
        """
    const int cur_off = stream_off[c];
    const int cur_L = stream_L[c];
"""
        if varlen
        else """
    const int cur_off = int(c) * LL;
    const int cur_L = LL;
"""
    )

    cur_block = (
        r"""
    for (int t = 0; t < cur_L; ++t) {
      const size_t b = size_t(cur_off + t);
      const size_t head_off = b * PROJ + size_t(hv) * D;
      const float bt = beta[b * H + hv];
      const device bfloat4* k4p = (const device bfloat4*)(k + head_off + k_base);
      const device bfloat4* q4p = (const device bfloat4*)(q + head_off + k_base);
      const device float4* g4p = (const device float4*)(g + head_off + k_base);

      float part[PPL];
#pragma unroll
      for (int pb = 0; pb < PPL; ++pb) {
        float4 s4 = st[pb] * g4p[pb];
        st[pb] = s4;
        float4 k4 = float4(k4p[pb]);
        float acc = 0.0f;
        acc += s4.x * k4.x;
        acc += s4.y * k4.y;
        acc += s4.z * k4.z;
        acc += s4.w * k4.w;
        part[pb] = acc;
      }
      float kv_mem =
          ((part[0] + part[1]) + (part[2] + part[3])) +
          ((part[4] + part[5]) + (part[6] + part[7]));
      kv_mem += simd_shuffle_xor(kv_mem, 1);
      kv_mem += simd_shuffle_xor(kv_mem, 2);

      float delta = (static_cast<float>(v[head_off + dv_idx]) - kv_mem) * bt;

#pragma unroll
      for (int pb = 0; pb < PPL; ++pb) {
        float4 s4 = st[pb] + float4(k4p[pb]) * delta;
        st[pb] = s4;
        float4 q4 = float4(q4p[pb]);
        float acc = 0.0f;
        acc += s4.x * q4.x;
        acc += s4.y * q4.y;
        acc += s4.z * q4.z;
        acc += s4.w * q4.w;
        part[pb] = acc;
      }
      float out_v =
          ((part[0] + part[1]) + (part[2] + part[3])) +
          ((part[4] + part[5]) + (part[6] + part[7]));
      out_v += simd_shuffle_xor(out_v, 1);
      out_v += simd_shuffle_xor(out_v, 2);

      if (lane_in_row == 0) {
        y_raw[head_off + dv_idx] = static_cast<T>(out_v);
      }
    }
"""
        if no_tg
        else r"""
    for (int t = 0; t < cur_L; ++t) {
      const size_t b = size_t(cur_off + t);
      const size_t head_off = b * PROJ + size_t(hv) * D;
      const float bt = beta[b * H + hv];
      for (uint i = ttid; i < D; i += NSG * 32) {
        tg_q[i] = static_cast<float>(q[head_off + i]);
        tg_k[i] = static_cast<float>(k[head_off + i]);
        tg_g[i] = g[head_off + i];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      float part[PPL];
#pragma unroll
      for (int pb = 0; pb < PPL; ++pb) {
        float4 g4 = *(const threadgroup float4*)(tg_g + k_base + pb * 4);
        float4 k4 = *(const threadgroup float4*)(tg_k + k_base + pb * 4);
        float4 s4 = st[pb] * g4;
        st[pb] = s4;
        float acc = 0.0f;
        acc += s4.x * k4.x;
        acc += s4.y * k4.y;
        acc += s4.z * k4.z;
        acc += s4.w * k4.w;
        part[pb] = acc;
      }
      float kv_mem =
          ((part[0] + part[1]) + (part[2] + part[3])) +
          ((part[4] + part[5]) + (part[6] + part[7]));
      kv_mem += simd_shuffle_xor(kv_mem, 1);
      kv_mem += simd_shuffle_xor(kv_mem, 2);

      float delta = (static_cast<float>(v[head_off + dv_idx]) - kv_mem) * bt;

#pragma unroll
      for (int pb = 0; pb < PPL; ++pb) {
        float4 k4 = *(const threadgroup float4*)(tg_k + k_base + pb * 4);
        float4 q4 = *(const threadgroup float4*)(tg_q + k_base + pb * 4);
        float4 s4 = st[pb] + k4 * delta;
        st[pb] = s4;
        float acc = 0.0f;
        acc += s4.x * q4.x;
        acc += s4.y * q4.y;
        acc += s4.z * q4.z;
        acc += s4.w * q4.w;
        part[pb] = acc;
      }
      float out_v =
          ((part[0] + part[1]) + (part[2] + part[3])) +
          ((part[4] + part[5]) + (part[6] + part[7]));
      out_v += simd_shuffle_xor(out_v, 1);
      out_v += simd_shuffle_xor(out_v, 2);

      if (lane_in_row == 0) {
        y_raw[head_off + dv_idx] = static_cast<T>(out_v);
      }
      if (t + 1 < cur_L) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
    }
"""
    )
    end_store = (
        r"""
#pragma unroll
    for (int pb = 0; pb < PPL; ++pb) {
      o_st4[pb] = st[pb];
    }
"""
        if write_end
        else ""
    )
    return head + ptr + load_st + prev_block + cur_bounds + cur_block + end_store


# Zero-spill advance-only (zero-TG/zero-barrier `no_tg=True` default + 8x float4 state I/O + varlen support)
def _gdn_adv_src(split_m: int = 0, varlen: bool = False, no_tg: bool = True) -> str:
    head = r"""
    constexpr int D = 128;
    constexpr int H = 64;
    constexpr int PROJ = H * D;
    constexpr int LPR = 4;
    constexpr int RPS = 32 / LPR;  // 8 rows/sg
    constexpr int VPL = D / LPR;   // 32 elems/lane
    constexpr int PPL = VPL / 4;

    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint ttid = sg * 32 + lane;
    const uint n = threadgroup_position_in_grid.z;
    const uint row_in_sg = lane / LPR;
    const uint lane_in_row = lane & (LPR - 1);
    const uint dv_idx = uint(threadgroup_position_in_grid.y) * (RPS * NSG) + sg * RPS + row_in_sg;
    const uint k_base = lane_in_row * VPL;
"""
    if split_m == 0:
        off_expr = "stream_off[c]" if varlen else "int(c) * LL"
        sel = f"""
    const uint c = n / H;
    const uint hv = n % H;
    const size_t st_off = (size_t(n) * D + dv_idx) * D + k_base;
    const device float4* i_st4 = (const device float4*)(state_in + st_off);
    device float4* o_st4 = (device float4*)(state_out + st_off);
    const int nc = n_consumed[c];
    const int base_off = {off_expr};
"""
    else:
        off_expr = "stream_off[r]" if varlen else "c * LL"
        branches = "\n".join(
            f"    {'if' if i == 1 else 'else if'} (r == {i}) {{ i_st4 = (const device float4*)(si{i} + st_off); o_st4 = (device float4*)(so{i} + st_off); }}"
            for i in range(1, split_m)
        )
        sel = f"""
    const uint r = n / H;
    const uint hv = n % H;
    const int c = stream_ids[r];
    const int nc = n_consumed[r];
    const int base_off = {off_expr};
    const size_t st_off = (size_t(hv) * D + dv_idx) * D + k_base;
    const device float4* i_st4 = (const device float4*)(si0 + st_off);
    device float4* o_st4 = (device float4*)(so0 + st_off);
{branches}
"""
    body = (
        r"""
    float4 st[PPL];
#pragma unroll
    for (int pb = 0; pb < PPL; ++pb) {
      st[pb] = i_st4[pb];
    }

    for (int t = 0; t < nc; ++t) {
      const size_t b = size_t(base_off + t);
      const size_t head_off = b * PROJ + size_t(hv) * D;
      const float bt = beta[b * H + hv];
      const device bfloat4* k4p = (const device bfloat4*)(k + head_off + k_base);
      const device float4* g4p = (const device float4*)(g + head_off + k_base);

      float part[PPL];
#pragma unroll
      for (int pb = 0; pb < PPL; ++pb) {
        float4 s4 = st[pb] * g4p[pb];
        st[pb] = s4;
        float4 k4 = float4(k4p[pb]);
        float acc = 0.0f;
        acc += s4.x * k4.x;
        acc += s4.y * k4.y;
        acc += s4.z * k4.z;
        acc += s4.w * k4.w;
        part[pb] = acc;
      }
      float kv_mem =
          ((part[0] + part[1]) + (part[2] + part[3])) +
          ((part[4] + part[5]) + (part[6] + part[7]));
      kv_mem += simd_shuffle_xor(kv_mem, 1);
      kv_mem += simd_shuffle_xor(kv_mem, 2);

      float delta = (static_cast<float>(v[head_off + dv_idx]) - kv_mem) * bt;

#pragma unroll
      for (int pb = 0; pb < PPL; ++pb) {
        st[pb] = st[pb] + float4(k4p[pb]) * delta;
      }
    }

#pragma unroll
    for (int pb = 0; pb < PPL; ++pb) {
      o_st4[pb] = st[pb];
    }
"""
        if no_tg
        else r"""
    float4 st[PPL];
#pragma unroll
    for (int pb = 0; pb < PPL; ++pb) {
      st[pb] = i_st4[pb];
    }

    threadgroup float tg_k[D];
    threadgroup float tg_g[D];

    for (int t = 0; t < nc; ++t) {
      const size_t b = size_t(base_off + t);
      const size_t head_off = b * PROJ + size_t(hv) * D;
      const float bt = beta[b * H + hv];
      for (uint i = ttid; i < D; i += NSG * 32) {
        tg_k[i] = static_cast<float>(k[head_off + i]);
        tg_g[i] = g[head_off + i];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      float part[PPL];
#pragma unroll
      for (int pb = 0; pb < PPL; ++pb) {
        float4 g4 = *(const threadgroup float4*)(tg_g + k_base + pb * 4);
        float4 k4 = *(const threadgroup float4*)(tg_k + k_base + pb * 4);
        float4 s4 = st[pb] * g4;
        st[pb] = s4;
        float acc = 0.0f;
        acc += s4.x * k4.x;
        acc += s4.y * k4.y;
        acc += s4.z * k4.z;
        acc += s4.w * k4.w;
        part[pb] = acc;
      }
      float kv_mem =
          ((part[0] + part[1]) + (part[2] + part[3])) +
          ((part[4] + part[5]) + (part[6] + part[7]));
      kv_mem += simd_shuffle_xor(kv_mem, 1);
      kv_mem += simd_shuffle_xor(kv_mem, 2);

      float delta = (static_cast<float>(v[head_off + dv_idx]) - kv_mem) * bt;

#pragma unroll
      for (int pb = 0; pb < PPL; ++pb) {
        float4 k4 = *(const threadgroup float4*)(tg_k + k_base + pb * 4);
        st[pb] = st[pb] + k4 * delta;
      }
      if (t + 1 < nc) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
    }

#pragma unroll
    for (int pb = 0; pb < PPL; ++pb) {
      o_st4[pb] = st[pb];
    }
"""
    )
    return head + sel + body


_SLICE_CONV_SRC = r"""
    constexpr int PROJ = 8192;
    const uint x4 = thread_position_in_grid.x;
    if (x4 >= (PROJ / 4)) return;
    const uint j = threadgroup_position_in_grid.y;
    const uint c = threadgroup_position_in_grid.z;
    const int nc = n_consumed[c];
    const size_t src_off = (size_t(c) * (3 + LL) + size_t(nc + int(j))) * (PROJ / 4) + x4;
    const size_t dst_off = (size_t(c) * 3 + size_t(j)) * (PROJ / 4) + x4;
    ((device uint2*)ns0)[dst_off] = ((const device uint2*)hist0)[src_off];
    ((device uint2*)ns1)[dst_off] = ((const device uint2*)hist1)[src_off];
    ((device uint2*)ns2)[dst_off] = ((const device uint2*)hist2)[src_off];
"""

# Direct conv-state commit from (cs0..2, q_raw, k_raw, v_raw, stream_off, n_consumed) — no hist0..2 needed!
_CONV_COMMIT_SRC = r"""
    constexpr int PROJ4 = 8192 / 4;
    const uint x4 = thread_position_in_grid.x;
    if (x4 >= PROJ4) return;
    const uint j = threadgroup_position_in_grid.y;
    const uint c = threadgroup_position_in_grid.z;
    const int nc = n_consumed[c];
    const int pos = nc + int(j);
    const size_t dst_off = (size_t(c) * 3 + size_t(j)) * PROJ4 + x4;
    if (pos < 3) {
      const size_t src_off = (size_t(c) * 3 + size_t(pos)) * PROJ4 + x4;
      ((device uint2*)ns0)[dst_off] = ((const device uint2*)cs0)[src_off];
      ((device uint2*)ns1)[dst_off] = ((const device uint2*)cs1)[src_off];
      ((device uint2*)ns2)[dst_off] = ((const device uint2*)cs2)[src_off];
    } else {
      const size_t row = size_t(stream_off[c] + (pos - 3));
      const size_t src_off = row * PROJ4 + x4;
      ((device uint2*)ns0)[dst_off] = ((const device uint2*)q_raw)[src_off];
      ((device uint2*)ns1)[dst_off] = ((const device uint2*)k_raw)[src_off];
      ((device uint2*)ns2)[dst_off] = ((const device uint2*)v_raw)[src_off];
    }
"""


def _pre_kernel(lower_bound: float, rms_eps: float, inv_scale: float, has_mask: bool):
    key = ("pre", float(lower_bound), float(rms_eps), float(inv_scale), has_mask)
    k = _KERNELS.get(key)
    if k is None:
        hdr = _KDA_HELPER_HDR + f"""
#define LOWER_BOUND {flit(lower_bound)}
#define RMS_EPS {flit(rms_eps)}
#define INV_SCALE {flit(inv_scale)}
#define INV_SCALE_SQ {flit(inv_scale * inv_scale)}
"""
        in_names = ["q_raw", "k_raw", "v_raw", "f_raw", "b_raw", "cs0", "cs1", "cs2", "wq", "wk", "wv", "A_log", "dt_bias", "mask"]
        k = _KERNELS[key] = mx.fast.metal_kernel(
            name=f"bv3_kda_pre_m{int(has_mask)}",
            input_names=in_names,
            output_names=["q_out", "k_out", "v_out", "g_out", "beta_out", "ns0", "ns1", "ns2"],
            header=hdr,
            source=_PRE_SRC,
            ensure_row_contiguous=True,
        )
    return k


def _onorm_gate_kernel(eps: float):
    key = ("onorm_gate", float(eps))
    k = _KERNELS.get(key)
    if k is None:
        hdr = _KDA_HELPER_HDR + f"#define O_EPS {flit(eps)}\n"
        k = _KERNELS[key] = mx.fast.metal_kernel(
            name="bv3_kda_onorm_gate",
            input_names=["y_raw", "g_raw", "o_norm_w"],
            output_names=["out_vec"],
            header=hdr,
            source=_ONORM_GATE_SRC,
            ensure_row_contiguous=True,
        )
    return k


def _gdn_notg_kernel():
    k = _KERNELS.get("gdn_notg")
    if k is None:
        k = _KERNELS["gdn_notg"] = mx.fast.metal_kernel(
            name="bv3_kda_gdn_notg",
            input_names=["q", "k", "v", "g", "beta", "state_in"],
            output_names=["out_vec", "state_out"],
            header=_KDA_HELPER_HDR,
            source=_GDN_NOTG_SRC,
            ensure_row_contiguous=True,
        )
    return k


def _gdn_kernel(eps: float):
    key = ("gdn", float(eps))
    k = _KERNELS.get(key)
    if k is None:
        hdr = _KDA_HELPER_HDR + f"#define O_EPS {flit(eps)}\n"
        k = _KERNELS[key] = mx.fast.metal_kernel(
            name="bv3_kda_gdn_norm_gate",
            input_names=["q", "k", "v", "g", "beta", "state_in", "g_raw", "o_norm_w"],
            output_names=["out_vec", "state_out"],
            header=hdr,
            source=_GDN_FUSED_SRC,
            ensure_row_contiguous=True,
        )
    return k


def _pre_parallel_kernel(lower_bound: float, rms_eps: float, inv_scale: float, varlen: bool, emit_hist: bool):
    key = ("pre_par", float(lower_bound), float(rms_eps), float(inv_scale), varlen, emit_hist)
    k = _KERNELS.get(key)
    if k is None:
        hdr = _KDA_HELPER_HDR + f"""
#define LOWER_BOUND {flit(lower_bound)}
#define RMS_EPS {flit(rms_eps)}
#define INV_SCALE {flit(inv_scale)}
#define INV_SCALE_SQ {flit(inv_scale * inv_scale)}
"""
        in_names = ["q_raw", "k_raw", "v_raw", "f_raw", "b_raw", "cs0", "cs1", "cs2", "wq", "wk", "wv", "A_log", "dt_bias"]
        if varlen:
            in_names += ["stream_id", "step_in_stream"]
        out_names = ["q_out", "k_out", "v_out", "g_out", "beta_out"]
        if emit_hist:
            out_names += ["hist0", "hist1", "hist2"]
        k = _KERNELS[key] = mx.fast.metal_kernel(
            name=f"bv3_kda_pre_par_v{int(varlen)}_h{int(emit_hist)}",
            input_names=in_names,
            output_names=out_names,
            header=hdr,
            source=_pre_parallel_src(varlen, emit_hist),
            ensure_row_contiguous=True,
        )
    return k


def _gdn_verify_kernel(split_c: int, has_prev: bool, write_end: bool, varlen: bool, no_tg: bool = True):
    key = ("gdn_ver", split_c, has_prev, write_end, varlen, no_tg)
    k = _KERNELS.get(key)
    if k is None:
        in_names = ["q", "k", "v", "g", "beta"]
        if split_c == 0:
            in_names.append("state_in")
        else:
            in_names += [f"si{i}" for i in range(split_c)]
        if has_prev:
            in_names += ["prev_k", "prev_v", "prev_g", "prev_beta", "prev_off", "prev_nc"]
        if varlen:
            in_names += ["stream_off", "stream_L"]
        out_names = ["y_raw"]
        if has_prev:
            if split_c == 0:
                out_names.append("state_commit")
            else:
                out_names += [f"sc{i}" for i in range(split_c)]
        if write_end:
            if split_c == 0:
                out_names.append("state_out")
            else:
                out_names += [f"so{i}" for i in range(split_c)]
        k = _KERNELS[key] = mx.fast.metal_kernel(
            name=f"bv3_kda_gdn_ver_c{split_c}_p{int(has_prev)}_e{int(write_end)}_v{int(varlen)}_nt{int(no_tg)}",
            input_names=in_names,
            output_names=out_names,
            header=_KDA_HELPER_HDR,
            source=_gdn_verify_src(split_c, has_prev, write_end, varlen, no_tg=no_tg),
            ensure_row_contiguous=True,
        )
    return k


def _gdn_adv_kernel(split_m: int = 0, varlen: bool = False, no_tg: bool = True):
    key = ("gdn_adv", split_m, varlen, no_tg)
    k = _KERNELS.get(key)
    if k is None:
        if split_m == 0:
            in_names = ["k", "v", "g", "beta", "state_in", "n_consumed"]
            if varlen:
                in_names.append("stream_off")
            out_names = ["state_out"]
        else:
            in_names = ["k", "v", "g", "beta"] + [f"si{i}" for i in range(split_m)] + ["stream_ids", "n_consumed"]
            if varlen:
                in_names.append("stream_off")
            out_names = [f"so{i}" for i in range(split_m)]
        k = _KERNELS[key] = mx.fast.metal_kernel(
            name=f"bv3_kda_gdn_adv_m{split_m}_v{int(varlen)}_nt{int(no_tg)}",
            input_names=in_names,
            output_names=out_names,
            header=_KDA_HELPER_HDR,
            source=_gdn_adv_src(split_m, varlen, no_tg=no_tg),
            ensure_row_contiguous=True,
        )
    return k


def _slice_conv_kernel():
    k = _KERNELS.get("slice_conv")
    if k is None:
        k = _KERNELS["slice_conv"] = mx.fast.metal_kernel(
            name="bv3_kda_slice_conv",
            input_names=["hist0", "hist1", "hist2", "n_consumed"],
            output_names=["ns0", "ns1", "ns2"],
            header=GLUE_HEADER,
            source=_SLICE_CONV_SRC,
            ensure_row_contiguous=True,
        )
    return k


def _conv_commit_kernel():
    k = _KERNELS.get("conv_commit")
    if k is None:
        k = _KERNELS["conv_commit"] = mx.fast.metal_kernel(
            name="bv3_kda_conv_commit",
            input_names=["cs0", "cs1", "cs2", "q_raw", "k_raw", "v_raw", "stream_off", "n_consumed"],
            output_names=["ns0", "ns1", "ns2"],
            header=GLUE_HEADER,
            source=_CONV_COMMIT_SRC,
            ensure_row_contiguous=True,
        )
    return k


def kda_o_norm_gate_rows(y_raw: mx.array, g_raw: mx.array, o_norm_w: mx.array, *, eps: float = 1e-6) -> mx.array:
    """Bit-exact head-wise RMSNorm(128) * sigmoid(g_raw) over (B, 8192) -> (B, 8192)."""
    B = y_raw.shape[0]
    (out,) = _onorm_gate_kernel(eps)(
        inputs=[y_raw, g_raw, o_norm_w],
        template=[("T", y_raw.dtype), ("WT", o_norm_w.dtype)],
        grid=(32, 64, B),
        threadgroup=(32, 1, 1),
        output_shapes=[(B, 8192)],
        output_dtypes=[y_raw.dtype],
    )
    return out


def kda_pre_rows(
    q_raw: mx.array,
    k_raw: mx.array,
    v_raw: mx.array,
    f_raw: mx.array,
    b_raw: mx.array,
    cs0: mx.array,
    cs1: mx.array,
    cs2: mx.array,
    wq: mx.array,
    wk: mx.array,
    wv: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
    *,
    lower_bound: float = -5.0,
    eps: float = 1e-6,
    mask: mx.array | None = None,
):
    B = q_raw.shape[0]
    inv_scale = 128.0 ** -0.5
    rms_eps = eps * (inv_scale * inv_scale)
    has_mask = mask is not None
    m_arr = mask.astype(mx.uint8) if has_mask else _dummy_u8()
    inputs = [q_raw, k_raw, v_raw, f_raw, b_raw, cs0, cs1, cs2, wq, wk, wv, A_log, dt_bias, m_arr]
    return _pre_kernel(lower_bound, rms_eps, inv_scale, has_mask)(
        inputs=inputs,
        template=[
            ("T", q_raw.dtype),
            ("AT", A_log.dtype),
            ("DT", dt_bias.dtype),
            ("HAS_MASK", int(has_mask)),
        ],
        grid=(32, 64, B),
        threadgroup=(32, 1, 1),
        output_shapes=[
            (B, 8192), (B, 8192), (B, 8192),
            (B, 8192), (B, 64),
            (B, 3, 8192), (B, 3, 8192), (B, 3, 8192),
        ],
        output_dtypes=[
            q_raw.dtype, q_raw.dtype, q_raw.dtype,
            mx.float32, mx.float32,
            q_raw.dtype, q_raw.dtype, q_raw.dtype,
        ],
    )


def kda_gdn_norm_gate_rows(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state_in: mx.array,
    g_raw: mx.array,
    o_norm_w: mx.array,
    *,
    eps: float = 1e-6,
    fused: bool = True,
    nsg: int = 1,
):
    B = q.shape[0]
    assert 1 <= B <= 4
    if nsg == 1:
        y_raw, state_out = _gdn_notg_kernel()(
            inputs=[q, k, v, g, beta, state_in],
            template=[("T", q.dtype), ("StT", state_in.dtype)],
            grid=(32, 16, B * 64),
            threadgroup=(32, 1, 1),
            output_shapes=[(B, 8192), state_in.shape],
            output_dtypes=[q.dtype, state_in.dtype],
        )
        return (kda_o_norm_gate_rows(y_raw, g_raw, o_norm_w, eps=eps), state_out) if fused else (y_raw, state_out)
    if fused:
        # Defer o_norm to 64*B-TG kernel when nsg < 16 so all 80 M3 Ultra cores stay saturated.
        if nsg < 16:
            assert 128 % (8 * nsg) == 0
            y_raw, state_out = _gdn_kernel(eps)(
                inputs=[q, k, v, g, beta, state_in, g_raw, o_norm_w],
                template=[
                    ("T", q.dtype),
                    ("StT", state_in.dtype),
                    ("WT", o_norm_w.dtype),
                    ("NSG", nsg),
                    ("FUSED", 0),
                ],
                grid=(32, 16, B * 64),
                threadgroup=(32, nsg, 1),
                output_shapes=[(B, 8192), state_in.shape],
                output_dtypes=[q.dtype, state_in.dtype],
            )
            return kda_o_norm_gate_rows(y_raw, g_raw, o_norm_w, eps=eps), state_out
        gy = 16
        out_vec, state_out = _gdn_kernel(eps)(
            inputs=[q, k, v, g, beta, state_in, g_raw, o_norm_w],
            template=[
                ("T", q.dtype),
                ("StT", state_in.dtype),
                ("WT", o_norm_w.dtype),
                ("NSG", 16),
                ("FUSED", 1),
            ],
            grid=(32, gy, B * 64),
            threadgroup=(32, 16, 1),
            output_shapes=[(B, 8192), state_in.shape],
            output_dtypes=[q.dtype, state_in.dtype],
        )
        return out_vec, state_out

    assert 128 % (8 * nsg) == 0
    out_vec, state_out = _gdn_kernel(eps)(
        inputs=[q, k, v, g, beta, state_in, g_raw, o_norm_w],
        template=[
            ("T", q.dtype),
            ("StT", state_in.dtype),
            ("WT", o_norm_w.dtype),
            ("NSG", nsg),
            ("FUSED", 0),
        ],
        grid=(32, 16, B * 64),
        threadgroup=(32, nsg, 1),
        output_shapes=[(B, 8192), state_in.shape],
        output_dtypes=[q.dtype, state_in.dtype],
    )
    return out_vec, state_out


def kda_pre_seq_rows(
    q_raw: mx.array,
    k_raw: mx.array,
    v_raw: mx.array,
    f_raw: mx.array,
    b_raw: mx.array,
    cs0: mx.array,
    cs1: mx.array,
    cs2: mx.array,
    wq: mx.array,
    wk: mx.array,
    wv: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
    *,
    C: int,
    L: int,
    lower_bound: float = -5.0,
    eps: float = 1e-6,
    emit_hist: bool = True,
):
    B = C * L
    assert q_raw.shape == (B, 8192) and cs0.shape == (C, 3, 8192)
    inv_scale = 128.0 ** -0.5
    rms_eps = eps * (inv_scale * inv_scale)
    out_shapes = [(B, 8192), (B, 8192), (B, 8192), (B, 8192), (B, 64)]
    out_dtypes = [q_raw.dtype, q_raw.dtype, q_raw.dtype, mx.float32, mx.float32]
    if emit_hist:
        out_shapes += [(C, 3 + L, 8192), (C, 3 + L, 8192), (C, 3 + L, 8192)]
        out_dtypes += [q_raw.dtype, q_raw.dtype, q_raw.dtype]
    return _pre_parallel_kernel(lower_bound, rms_eps, inv_scale, varlen=False, emit_hist=emit_hist)(
        inputs=[q_raw, k_raw, v_raw, f_raw, b_raw, cs0, cs1, cs2, wq, wk, wv, A_log, dt_bias],
        template=[
            ("T", q_raw.dtype),
            ("AT", A_log.dtype),
            ("DT", dt_bias.dtype),
            ("LL", L),
        ],
        grid=(32, 64, B),
        threadgroup=(32, 1, 1),
        output_shapes=out_shapes,
        output_dtypes=out_dtypes,
    )


def kda_pre_varlen_rows(
    q_raw: mx.array,
    k_raw: mx.array,
    v_raw: mx.array,
    f_raw: mx.array,
    b_raw: mx.array,
    cs0: mx.array,
    cs1: mx.array,
    cs2: mx.array,
    wq: mx.array,
    wk: mx.array,
    wv: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
    Ls: list[int],
    *,
    lower_bound: float = -5.0,
    eps: float = 1e-6,
    stream_id: mx.array | None = None,
    step_in_stream: mx.array | None = None,
):
    """(h, b)-parallel varlen KDA pre-projection for heterogeneous stream lengths `Ls` (drops hist0..2)."""
    C = len(Ls)
    B = sum(Ls)
    assert q_raw.shape == (B, 8192) and cs0.shape == (C, 3, 8192) and all(lc >= 1 for lc in Ls)
    if stream_id is None:
        stream_id = mx.array([c for c, lc in enumerate(Ls) for _ in range(lc)], dtype=mx.int32)
    if step_in_stream is None:
        step_in_stream = mx.array([t for lc in Ls for t in range(lc)], dtype=mx.int32)
    inv_scale = 128.0 ** -0.5
    rms_eps = eps * (inv_scale * inv_scale)
    return _pre_parallel_kernel(lower_bound, rms_eps, inv_scale, varlen=True, emit_hist=False)(
        inputs=[
            q_raw, k_raw, v_raw, f_raw, b_raw,
            cs0, cs1, cs2, wq, wk, wv, A_log, dt_bias,
            stream_id, step_in_stream,
        ],
        template=[
            ("T", q_raw.dtype),
            ("AT", A_log.dtype),
            ("DT", dt_bias.dtype),
        ],
        grid=(32, 64, B),
        threadgroup=(32, 1, 1),
        output_shapes=[(B, 8192), (B, 8192), (B, 8192), (B, 8192), (B, 64)],
        output_dtypes=[q_raw.dtype, q_raw.dtype, q_raw.dtype, mx.float32, mx.float32],
    )


def _to_i32(x) -> mx.array:
    if isinstance(x, mx.array):
        return x if x.dtype == mx.int32 else x.astype(mx.int32)
    return mx.array(x, dtype=mx.int32)


def _varlen_offs_ls(Ls: list[int]):
    offs = []
    pos = 0
    for lc in Ls:
        offs.append(pos)
        pos += lc
    return mx.array(offs, dtype=mx.int32), mx.array(Ls, dtype=mx.int32)


def kda_gdn_verify_seq_rows(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state_in: mx.array,
    g_raw: mx.array,
    o_norm_w: mx.array,
    *,
    C: int,
    L: int,
    eps: float = 1e-6,
    nsg: int = 1,
    write_end: bool = True,
):
    B = C * L
    assert q.shape == (B, 8192) and state_in.shape == (C, 64, 128, 128)
    assert state_in.dtype == mx.float32 and 128 % (8 * nsg) == 0
    out_shapes = [(B, 8192)] + ([state_in.shape] if write_end else [])
    out_dtypes = [q.dtype] + ([state_in.dtype] if write_end else [])
    outs = _gdn_verify_kernel(0, has_prev=False, write_end=write_end, varlen=False, no_tg=(nsg == 1))(
        inputs=[q, k, v, g, beta, state_in],
        template=[("T", q.dtype), ("StT", state_in.dtype), ("NSG", nsg), ("LL", L)],
        grid=(32, 16, C * 64),
        threadgroup=(32, nsg, 1),
        output_shapes=out_shapes,
        output_dtypes=out_dtypes,
    )
    ov = kda_o_norm_gate_rows(outs[0], g_raw, o_norm_w, eps=eps)
    return (ov, outs[1]) if write_end else ov


def kda_gdn_verify_seq_split_rows(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    states_in: list[mx.array],
    g_raw: mx.array,
    o_norm_w: mx.array,
    *,
    C: int,
    L: int,
    eps: float = 1e-6,
    nsg: int = 1,
    write_end: bool = True,
):
    """Per-stream split `(1, 64, 128, 128)` GDN verify so accepted streams commit with zero copy."""
    B = C * L
    assert 2 <= C <= 4 and len(states_in) == C and q.shape == (B, 8192)
    assert all(s.shape == (1, 64, 128, 128) and s.dtype == mx.float32 for s in states_in)
    assert 128 % (8 * nsg) == 0
    out_shapes = [(B, 8192)] + ([(1, 64, 128, 128)] * C if write_end else [])
    out_dtypes = [q.dtype] + ([mx.float32] * C if write_end else [])
    outs = _gdn_verify_kernel(C, has_prev=False, write_end=write_end, varlen=False, no_tg=(nsg == 1))(
        inputs=[q, k, v, g, beta, *states_in],
        template=[("T", q.dtype), ("StT", mx.float32), ("NSG", nsg), ("LL", L)],
        grid=(32, 16, C * 64),
        threadgroup=(32, nsg, 1),
        output_shapes=out_shapes,
        output_dtypes=out_dtypes,
    )
    ov = kda_o_norm_gate_rows(outs[0], g_raw, o_norm_w, eps=eps)
    return (ov, list(outs[1:])) if write_end else ov


def kda_gdn_verify_varlen_rows(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state_in: mx.array,
    g_raw: mx.array,
    o_norm_w: mx.array,
    Ls: list[int],
    *,
    eps: float = 1e-6,
    nsg: int = 1,
    write_end: bool = True,
    stream_off: mx.array | None = None,
    stream_L: mx.array | None = None,
):
    """Variable-length GDN verify for `(C,)` streams with lengths `Ls` (`sum(Ls) <= 24`)."""
    C = len(Ls)
    B = sum(Ls)
    assert q.shape == (B, 8192) and state_in.shape == (C, 64, 128, 128) and state_in.dtype == mx.float32
    if stream_off is None or stream_L is None:
        stream_off, stream_L = _varlen_offs_ls(Ls)
    out_shapes = [(B, 8192)] + ([state_in.shape] if write_end else [])
    out_dtypes = [q.dtype] + ([state_in.dtype] if write_end else [])
    outs = _gdn_verify_kernel(0, has_prev=False, write_end=write_end, varlen=True, no_tg=(nsg == 1))(
        inputs=[q, k, v, g, beta, state_in, stream_off, stream_L],
        template=[("T", q.dtype), ("StT", state_in.dtype), ("NSG", nsg)],
        grid=(32, 16, C * 64),
        threadgroup=(32, nsg, 1),
        output_shapes=out_shapes,
        output_dtypes=out_dtypes,
    )
    ov = kda_o_norm_gate_rows(outs[0], g_raw, o_norm_w, eps=eps)
    return (ov, outs[1]) if write_end else ov


def kda_gdn_verify_varlen_split_rows(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    states_in: list[mx.array],
    g_raw: mx.array,
    o_norm_w: mx.array,
    Ls: list[int],
    *,
    eps: float = 1e-6,
    nsg: int = 1,
    write_end: bool = True,
    stream_off: mx.array | None = None,
    stream_L: mx.array | None = None,
):
    """Variable-length split-state GDN verify for `2 <= C <= 4` streams."""
    C = len(Ls)
    B = sum(Ls)
    assert 2 <= C <= 4 and len(states_in) == C and q.shape == (B, 8192)
    assert all(s.shape == (1, 64, 128, 128) and s.dtype == mx.float32 for s in states_in)
    if stream_off is None or stream_L is None:
        stream_off, stream_L = _varlen_offs_ls(Ls)
    out_shapes = [(B, 8192)] + ([(1, 64, 128, 128)] * C if write_end else [])
    out_dtypes = [q.dtype] + ([mx.float32] * C if write_end else [])
    outs = _gdn_verify_kernel(C, has_prev=False, write_end=write_end, varlen=True, no_tg=(nsg == 1))(
        inputs=[q, k, v, g, beta, *states_in, stream_off, stream_L],
        template=[("T", q.dtype), ("StT", mx.float32), ("NSG", nsg)],
        grid=(32, 16, C * 64),
        threadgroup=(32, nsg, 1),
        output_shapes=out_shapes,
        output_dtypes=out_dtypes,
    )
    ov = kda_o_norm_gate_rows(outs[0], g_raw, o_norm_w, eps=eps)
    return (ov, list(outs[1:])) if write_end else ov


def kda_gdn_commit_and_verify_seq_rows(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    prev_state_in: mx.array,
    prev_k: mx.array,
    prev_v: mx.array,
    prev_g: mx.array,
    prev_beta: mx.array,
    prev_off,
    prev_nc,
    g_raw: mx.array,
    o_norm_w: mx.array,
    *,
    C: int,
    L: int,
    eps: float = 1e-6,
    nsg: int = 1,
    write_end: bool = False,
):
    """Lazy fused KDA commit+verify: replays `prev_nc` steps -> writes `state_committed`, then verifies `L` steps.

    With `write_end=False` (recommended), reads `prev_state_in` ONCE and writes `state_committed` ONCE
    (528 MB total vs 1,056 MB for separate commit + verify). Returns `(ov, state_committed)` or
    `(ov, state_committed, state_end)` when `write_end=True`.
    """
    B = C * L
    assert q.shape == (B, 8192) and prev_state_in.shape == (C, 64, 128, 128) and prev_state_in.dtype == mx.float32
    p_off = _to_i32(prev_off)
    p_nc = _to_i32(prev_nc)
    out_shapes = [(B, 8192), prev_state_in.shape] + ([prev_state_in.shape] if write_end else [])
    out_dtypes = [q.dtype, mx.float32] + ([mx.float32] if write_end else [])
    outs = _gdn_verify_kernel(0, has_prev=True, write_end=write_end, varlen=False, no_tg=(nsg == 1))(
        inputs=[q, k, v, g, beta, prev_state_in, prev_k, prev_v, prev_g, prev_beta, p_off, p_nc],
        template=[("T", q.dtype), ("StT", mx.float32), ("NSG", nsg), ("LL", L)],
        grid=(32, 16, C * 64),
        threadgroup=(32, nsg, 1),
        output_shapes=out_shapes,
        output_dtypes=out_dtypes,
    )
    ov = kda_o_norm_gate_rows(outs[0], g_raw, o_norm_w, eps=eps)
    return (ov, outs[1], outs[2]) if write_end else (ov, outs[1])


def kda_gdn_commit_and_verify_seq_split_rows(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    prev_states_in: list[mx.array],
    prev_k: mx.array,
    prev_v: mx.array,
    prev_g: mx.array,
    prev_beta: mx.array,
    prev_off,
    prev_nc,
    g_raw: mx.array,
    o_norm_w: mx.array,
    *,
    C: int,
    L: int,
    eps: float = 1e-6,
    nsg: int = 1,
    write_end: bool = False,
):
    """Split-state lazy fused KDA commit+verify for `2 <= C <= 4` streams."""
    B = C * L
    assert 2 <= C <= 4 and len(prev_states_in) == C and q.shape == (B, 8192)
    assert all(s.shape == (1, 64, 128, 128) and s.dtype == mx.float32 for s in prev_states_in)
    p_off = _to_i32(prev_off)
    p_nc = _to_i32(prev_nc)
    out_shapes = [(B, 8192)] + [(1, 64, 128, 128)] * C + ([(1, 64, 128, 128)] * C if write_end else [])
    out_dtypes = [q.dtype] + [mx.float32] * C + ([mx.float32] * C if write_end else [])
    outs = _gdn_verify_kernel(C, has_prev=True, write_end=write_end, varlen=False, no_tg=(nsg == 1))(
        inputs=[q, k, v, g, beta, *prev_states_in, prev_k, prev_v, prev_g, prev_beta, p_off, p_nc],
        template=[("T", q.dtype), ("StT", mx.float32), ("NSG", nsg), ("LL", L)],
        grid=(32, 16, C * 64),
        threadgroup=(32, nsg, 1),
        output_shapes=out_shapes,
        output_dtypes=out_dtypes,
    )
    ov = kda_o_norm_gate_rows(outs[0], g_raw, o_norm_w, eps=eps)
    s_commit = list(outs[1 : 1 + C])
    return (ov, s_commit, list(outs[1 + C :])) if write_end else (ov, s_commit)


def kda_gdn_commit_and_verify_varlen_rows(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    prev_state_in: mx.array,
    prev_k: mx.array,
    prev_v: mx.array,
    prev_g: mx.array,
    prev_beta: mx.array,
    prev_off,
    prev_nc,
    g_raw: mx.array,
    o_norm_w: mx.array,
    Ls: list[int],
    *,
    eps: float = 1e-6,
    nsg: int = 1,
    write_end: bool = False,
    stream_off: mx.array | None = None,
    stream_L: mx.array | None = None,
):
    """Variable-length lazy fused KDA commit+verify."""
    C = len(Ls)
    B = sum(Ls)
    assert q.shape == (B, 8192) and prev_state_in.shape == (C, 64, 128, 128) and prev_state_in.dtype == mx.float32
    p_off = _to_i32(prev_off)
    p_nc = _to_i32(prev_nc)
    if stream_off is None or stream_L is None:
        stream_off, stream_L = _varlen_offs_ls(Ls)
    out_shapes = [(B, 8192), prev_state_in.shape] + ([prev_state_in.shape] if write_end else [])
    out_dtypes = [q.dtype, mx.float32] + ([mx.float32] if write_end else [])
    outs = _gdn_verify_kernel(0, has_prev=True, write_end=write_end, varlen=True, no_tg=(nsg == 1))(
        inputs=[q, k, v, g, beta, prev_state_in, prev_k, prev_v, prev_g, prev_beta, p_off, p_nc, stream_off, stream_L],
        template=[("T", q.dtype), ("StT", mx.float32), ("NSG", nsg)],
        grid=(32, 16, C * 64),
        threadgroup=(32, nsg, 1),
        output_shapes=out_shapes,
        output_dtypes=out_dtypes,
    )
    ov = kda_o_norm_gate_rows(outs[0], g_raw, o_norm_w, eps=eps)
    return (ov, outs[1], outs[2]) if write_end else (ov, outs[1])


def kda_gdn_commit_and_verify_varlen_split_rows(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    prev_states_in: list[mx.array],
    prev_k: mx.array,
    prev_v: mx.array,
    prev_g: mx.array,
    prev_beta: mx.array,
    prev_off,
    prev_nc,
    g_raw: mx.array,
    o_norm_w: mx.array,
    Ls: list[int],
    *,
    eps: float = 1e-6,
    nsg: int = 1,
    write_end: bool = False,
    stream_off: mx.array | None = None,
    stream_L: mx.array | None = None,
):
    """Variable-length split-state lazy fused KDA commit+verify for `2 <= C <= 4`."""
    C = len(Ls)
    B = sum(Ls)
    assert 2 <= C <= 4 and len(prev_states_in) == C and q.shape == (B, 8192)
    assert all(s.shape == (1, 64, 128, 128) and s.dtype == mx.float32 for s in prev_states_in)
    p_off = _to_i32(prev_off)
    p_nc = _to_i32(prev_nc)
    if stream_off is None or stream_L is None:
        stream_off, stream_L = _varlen_offs_ls(Ls)
    out_shapes = [(B, 8192)] + [(1, 64, 128, 128)] * C + ([(1, 64, 128, 128)] * C if write_end else [])
    out_dtypes = [q.dtype] + [mx.float32] * C + ([mx.float32] * C if write_end else [])
    outs = _gdn_verify_kernel(C, has_prev=True, write_end=write_end, varlen=True, no_tg=(nsg == 1))(
        inputs=[q, k, v, g, beta, *prev_states_in, prev_k, prev_v, prev_g, prev_beta, p_off, p_nc, stream_off, stream_L],
        template=[("T", q.dtype), ("StT", mx.float32), ("NSG", nsg)],
        grid=(32, 16, C * 64),
        threadgroup=(32, nsg, 1),
        output_shapes=out_shapes,
        output_dtypes=out_dtypes,
    )
    ov = kda_o_norm_gate_rows(outs[0], g_raw, o_norm_w, eps=eps)
    s_commit = list(outs[1 : 1 + C])
    return (ov, s_commit, list(outs[1 + C :])) if write_end else (ov, s_commit)


def kda_gdn_advance_only(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state_in: mx.array,
    n_consumed,
    *,
    C: int,
    L: int = 0,
    nsg: int = 1,
    stream_off: mx.array | None = None,
):
    del q
    assert state_in.shape == (C, 64, 128, 128) and state_in.dtype == mx.float32
    assert 128 % (8 * nsg) == 0
    nc_arr = _to_i32(n_consumed)
    varlen = stream_off is not None
    inputs = [k, v, g, beta, state_in, nc_arr] + ([_to_i32(stream_off)] if varlen else [])
    tpl = [("T", k.dtype), ("StT", state_in.dtype), ("NSG", nsg)] + ([] if varlen else [("LL", L)])
    (state_out,) = _gdn_adv_kernel(0, varlen=varlen, no_tg=(nsg == 1))(
        inputs=inputs,
        template=tpl,
        grid=(32, 16, C * 64),
        threadgroup=(32, nsg, 1),
        output_shapes=[state_in.shape],
        output_dtypes=[state_in.dtype],
    )
    return state_out


def kda_gdn_advance_split_only(
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    states_in: list[mx.array],
    stream_ids,
    n_consumed,
    *,
    L: int = 0,
    nsg: int = 1,
    stream_off: mx.array | None = None,
) -> list[mx.array]:
    """Single-launch GDN replay for `M` rejected split-state streams (`1 <= M <= 4`), uniform or varlen."""
    M = len(states_in)
    assert 1 <= M <= 4 and 128 % (8 * nsg) == 0
    assert all(s.shape == (1, 64, 128, 128) and s.dtype == mx.float32 for s in states_in)
    sids = _to_i32(stream_ids)
    ncs = _to_i32(n_consumed)
    varlen = stream_off is not None
    inputs = [k, v, g, beta, *states_in, sids, ncs] + ([_to_i32(stream_off)] if varlen else [])
    tpl = [("T", k.dtype), ("StT", mx.float32), ("NSG", nsg)] + ([] if varlen else [("LL", L)])
    outs = _gdn_adv_kernel(M, varlen=varlen, no_tg=(nsg == 1))(
        inputs=inputs,
        template=tpl,
        grid=(32, 16, M * 64),
        threadgroup=(32, nsg, 1),
        output_shapes=[(1, 64, 128, 128)] * M,
        output_dtypes=[mx.float32] * M,
    )
    return list(outs)


def kda_slice_conv_rows(hist0: mx.array, hist1: mx.array, hist2: mx.array, n_consumed):
    C, T_tot, P = hist0.shape
    L = T_tot - 3
    assert P == 8192 and L >= 1
    nc_arr = _to_i32(n_consumed)
    return _slice_conv_kernel()(
        inputs=[hist0, hist1, hist2, nc_arr],
        template=[("LL", L)],
        grid=(2048, 3, C),
        threadgroup=(256, 1, 1),
        output_shapes=[(C, 3, 8192)] * 3,
        output_dtypes=[hist0.dtype] * 3,
    )


def kda_conv_commit_rows(
    cs0: mx.array,
    cs1: mx.array,
    cs2: mx.array,
    q_raw: mx.array,
    k_raw: mx.array,
    v_raw: mx.array,
    n_consumed,
    *,
    L: int | None = None,
    stream_off: mx.array | None = None,
):
    """Commit `(C, 3, 8192)` conv caches directly from `(cs0..2, q_raw, k_raw, v_raw, n_consumed)` without `hist0..2`."""
    C = cs0.shape[0]
    nc_arr = _to_i32(n_consumed)
    if stream_off is None:
        assert L is not None and L >= 1
        off_arr = mx.array([c * L for c in range(C)], dtype=mx.int32)
    else:
        off_arr = _to_i32(stream_off)
    return _conv_commit_kernel()(
        inputs=[cs0, cs1, cs2, q_raw, k_raw, v_raw, off_arr, nc_arr],
        grid=(2048, 3, C),
        threadgroup=(256, 1, 1),
        output_shapes=[(C, 3, 8192)] * 3,
        output_dtypes=[cs0.dtype] * 3,
    )
