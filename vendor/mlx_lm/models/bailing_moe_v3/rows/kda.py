"""Bit-exact fused KDA decode kernels (63 of 72 LingSante layers, B = 1..4).

1. `kda_pre_rows`: fuses q/k/v causal depthwise Conv1d(K=4) + cache state shift + silu +
   `normalize_qk` (inv_scale=128**-0.5, eps=1e-6) + `compute_lower_bound_g` + `sigmoid(b)`
   across (B, 64 heads, 128 head_dim) in ONE launch.
2. `kda_gdn_norm_gate_rows`: packed vector-gate Gated Delta step (`8` rows/simdgroup,
   `Dk=Dv=128`, exact ascending butterfly tree matching `simd_sum` on Apple GPU) fused with
   head-wise `o_norm` (`RMSNorm(128, eps)`) and `sigmoid(g_proj)` gating in ONE launch.
"""
import mlx.core as mx

from .glue import GLUE_HEADER, flit

_KERNELS = {}

_PRE_SRC = r"""
    constexpr int NH = 64;
    constexpr int HD = 128;
    constexpr int PROJ = NH * HD;  // 8192
    const uint lane = thread_index_in_simdgroup;
    const uint h = threadgroup_position_in_grid.y;
    const uint b = threadgroup_position_in_grid.z;
    const size_t row_off = size_t(b) * PROJ + size_t(h) * HD;

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

    T q_silu[4], k_silu[4];
    float q_acc = 0.0f;
    float k_acc = 0.0f;

#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int c = int(lane) * 4 + i;
      const size_t gc = size_t(h) * HD + c;

      // q conv1d + shift + silu
      T qx = active ? q_raw[row_off + c] : static_cast<T>(0.0f);
      T qp0 = cs0_r[c], qp1 = cs0_r[PROJ + c], qp2 = cs0_r[2 * PROJ + c];
      ns0_r[c] = qp1;
      ns0_r[PROJ + c] = qp2;
      ns0_r[2 * PROJ + c] = qx;
      float qconv = 0.0f;
      qconv += static_cast<float>(qp0) * static_cast<float>(wq[gc * 4 + 0]);
      qconv += static_cast<float>(qp1) * static_cast<float>(wq[gc * 4 + 1]);
      qconv += static_cast<float>(qp2) * static_cast<float>(wq[gc * 4 + 2]);
      qconv += static_cast<float>(qx)  * static_cast<float>(wq[gc * 4 + 3]);
      T qcb = static_cast<T>(qconv);
      q_silu[i] = qcb * glue_sigmoid_bf16(qcb);
      float qf = static_cast<float>(q_silu[i]);
      q_acc = fma(qf, qf, q_acc);

      // k conv1d + shift + silu
      T kx = active ? k_raw[row_off + c] : static_cast<T>(0.0f);
      T kp0 = cs1_r[c], kp1 = cs1_r[PROJ + c], kp2 = cs1_r[2 * PROJ + c];
      ns1_r[c] = kp1;
      ns1_r[PROJ + c] = kp2;
      ns1_r[2 * PROJ + c] = kx;
      float kconv = 0.0f;
      kconv += static_cast<float>(kp0) * static_cast<float>(wk[gc * 4 + 0]);
      kconv += static_cast<float>(kp1) * static_cast<float>(wk[gc * 4 + 1]);
      kconv += static_cast<float>(kp2) * static_cast<float>(wk[gc * 4 + 2]);
      kconv += static_cast<float>(kx)  * static_cast<float>(wk[gc * 4 + 3]);
      T kcb = static_cast<T>(kconv);
      k_silu[i] = kcb * glue_sigmoid_bf16(kcb);
      float kf = static_cast<float>(k_silu[i]);
      k_acc = fma(kf, kf, k_acc);

      // v conv1d + shift + silu
      T vx = active ? v_raw[row_off + c] : static_cast<T>(0.0f);
      T vp0 = cs2_r[c], vp1 = cs2_r[PROJ + c], vp2 = cs2_r[2 * PROJ + c];
      ns2_r[c] = vp1;
      ns2_r[PROJ + c] = vp2;
      ns2_r[2 * PROJ + c] = vx;
      float vconv = 0.0f;
      vconv += static_cast<float>(vp0) * static_cast<float>(wv[gc * 4 + 0]);
      vconv += static_cast<float>(vp1) * static_cast<float>(wv[gc * 4 + 1]);
      vconv += static_cast<float>(vp2) * static_cast<float>(wv[gc * 4 + 2]);
      vconv += static_cast<float>(vx)  * static_cast<float>(wv[gc * 4 + 3]);
      T vcb = static_cast<T>(vconv);
      v_out[row_off + c] = vcb * glue_sigmoid_bf16(vcb);

      // compute_lower_bound_g
      float fval = static_cast<float>(f_raw[row_off + c]) + static_cast<float>(dt_bias[gc]);
      g_out[row_off + c] = metal::precise::exp(LOWER_BOUND * glue_sigmoid(expA * fval));
    }

    float q_tot = simd_sum(q_acc);
    float k_tot = simd_sum(k_acc);
    float q_inv = metal::precise::rsqrt(q_tot / 128.0f + RMS_EPS);
    float k_inv = metal::precise::rsqrt(k_tot / 128.0f + RMS_EPS);
    T q_sc = static_cast<T>(INV_SCALE_SQ);
    T k_sc = static_cast<T>(INV_SCALE);

#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int c = int(lane) * 4 + i;
      q_out[row_off + c] = q_sc * static_cast<T>(static_cast<float>(q_silu[i]) * q_inv);
      k_out[row_off + c] = k_sc * static_cast<T>(static_cast<float>(k_silu[i]) * k_inv);
    }
"""

# Packed vector-gate GDN (8 rows/simdgroup, Dk=Dv=128) + optional fused o_norm & sigmoid(g_raw)
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
    const uint n = threadgroup_position_in_grid.z;  // b * H + hv
    const uint b = n / H;
    const uint hv = n % H;
    const uint row_in_sg = lane / LPR;
    const uint lane_in_row = lane & (LPR - 1);

    const size_t head_off = size_t(b) * PROJ + size_t(hv) * D;
    const device T* q_ = q + head_off + lane_in_row * VPL;
    const device T* k_ = k + head_off + lane_in_row * VPL;
    const device T* v_ = v + head_off;
    const device float* g_ = g + head_off + lane_in_row * VPL;
    const float bt = beta[size_t(b) * H + hv];

    float ql[VPL], kl[VPL], gl[VPL];
#pragma unroll
    for (int i = 0; i < VPL; ++i) {
      ql[i] = static_cast<float>(q_[i]);
      kl[i] = static_cast<float>(k_[i]);
      gl[i] = g_[i];
    }

    threadgroup T y_tg[FUSED ? D : 1];

    for (uint dv0 = uint(threadgroup_position_in_grid.y) * RPTG; dv0 < uint(threadgroup_position_in_grid.y + 1) * RPTG; dv0 += RPTG) {
      const uint dv_idx = dv0 + sg * RPS + row_in_sg;
      const size_t st_off = (size_t(n) * D + dv_idx) * D + lane_in_row * VPL;
      const device StT* i_st = state_in + st_off;
      device StT* o_st = state_out + st_off;

      float st[VPL];
      float part[PPL];
#pragma unroll
      for (int pb = 0; pb < PPL; ++pb) {
        float acc = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          int e = pb * 4 + i;
          st[e] = static_cast<float>(i_st[e]) * gl[e];
          acc += st[e] * kl[e];
        }
        part[pb] = acc;
      }
      float kv_mem =
          ((part[0] + part[1]) + (part[2] + part[3])) +
          ((part[4] + part[5]) + (part[6] + part[7]));
      kv_mem += simd_shuffle_xor(kv_mem, 1);
      kv_mem += simd_shuffle_xor(kv_mem, 2);

      float delta = (static_cast<float>(v_[dv_idx]) - kv_mem) * bt;

#pragma unroll
      for (int pb = 0; pb < PPL; ++pb) {
        float acc = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          int e = pb * 4 + i;
          st[e] = st[e] + kl[e] * delta;
          acc += st[e] * ql[e];
        }
        part[pb] = acc;
      }
      float out_v =
          ((part[0] + part[1]) + (part[2] + part[3])) +
          ((part[4] + part[5]) + (part[6] + part[7]));
      out_v += simd_shuffle_xor(out_v, 1);
      out_v += simd_shuffle_xor(out_v, 2);

#pragma unroll
      for (int i = 0; i < VPL; ++i) {
        o_st[i] = static_cast<StT>(st[i]);
      }

      if (lane_in_row == 0) {
        if constexpr (FUSED) {
          y_tg[dv_idx] = static_cast<T>(out_v);
        } else {
          out_vec[head_off + dv_idx] = static_cast<T>(out_v);
        }
      }
    }

    if constexpr (FUSED) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sg == 0) {
        float acc = 0.0f;
        float vals[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          int c = int(lane) * 4 + i;
          vals[i] = static_cast<float>(y_tg[c]);
          acc = fma(vals[i], vals[i], acc);
        }
        float inv = metal::precise::rsqrt(simd_sum(acc) / 128.0f + O_EPS);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          int c = int(lane) * 4 + i;
          T normed = static_cast<T>(o_norm_w[c]) * static_cast<T>(vals[i] * inv);
          T gate = static_cast<T>(glue_sigmoid(static_cast<float>(g_raw[head_off + c])));
          out_vec[head_off + c] = normed * gate;
        }
      }
    }
"""


def _pre_kernel(lower_bound: float, rms_eps: float, inv_scale: float, has_mask: bool):
    key = ("pre", float(lower_bound), float(rms_eps), float(inv_scale), has_mask)
    k = _KERNELS.get(key)
    if k is None:
        hdr = GLUE_HEADER + f"""
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


def _gdn_kernel(eps: float):
    key = ("gdn", float(eps))
    k = _KERNELS.get(key)
    if k is None:
        hdr = GLUE_HEADER + f"#define O_EPS {flit(eps)}\n"
        k = _KERNELS[key] = mx.fast.metal_kernel(
            name="bv3_kda_gdn_norm_gate",
            input_names=["q", "k", "v", "g", "beta", "state_in", "g_raw", "o_norm_w"],
            output_names=["out_vec", "state_out"],
            header=hdr,
            source=_GDN_FUSED_SRC,
            ensure_row_contiguous=True,
        )
    return k


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
    m_arr = mask.astype(mx.uint8) if has_mask else mx.zeros((1,), dtype=mx.uint8)
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
    nsg: int = 16,
):
    B = q.shape[0]
    assert 1 <= B <= 4
    if fused:
        nsg = 16
        gy = nsg
    else:
        assert 128 % (8 * nsg) == 0
        gy = 128 // 8
    out_vec, state_out = _gdn_kernel(eps)(
        inputs=[q, k, v, g, beta, state_in, g_raw, o_norm_w],
        template=[
            ("T", q.dtype),
            ("StT", state_in.dtype),
            ("WT", o_norm_w.dtype),
            ("NSG", nsg),
            ("FUSED", int(fused)),
        ],
        grid=(32, gy, B * 64),
        threadgroup=(32, nsg, 1),
        output_shapes=[(B, 8192), state_in.shape],
        output_dtypes=[q.dtype, state_in.dtype],
    )
    return out_vec, state_out
