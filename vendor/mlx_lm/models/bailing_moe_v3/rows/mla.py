"""Bit-exact Wave 4 MLA decode helpers & kernels for BailingMLA (9 of 72 LingSante layers, B = 1..4).

1. `mla_prep_kv_rope_rows`: fuses q_nope/q_rope split, kv_latent/k_rope split, `kv_a_layernorm`
   (RMSNorm 512), and per-row interleaved (`traditional=True`) RoPE (base=6e6) in ONE launch.
2. `mla_sdpa_rows`: per-row unpadded MLA SDPA handling `KVCache` & heterogeneous-length
   `BatchKVCache` with cached Python `left_padding` starts (zero GPU->CPU sync in decode).
3. `mla_gate_rows`: head-wise sigmoid gate + flatten in ONE tiny launch before `gemv_rows_res`.
"""
import math

import mlx.core as mx

from .gemv import gemv_rows, gemv_rows_multi, gemv_rows_res
from .gemv_fp8 import E4M3_LUT_METAL, gemv_lin, gemv_lin_multi, gemv_lin_res
from .glue import GLUE_HEADER, flit

_KERNELS = {}

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
      device T* qn = q_nope + (size_t(b) * NH + y) * NOPE_D;
      device T* qr = q_rope + (size_t(b) * NH + y) * ROPE_D;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        int idx = int(lane) * 4 + i;
        qn[idx] = qh[idx];
      }
      int r1 = 2 * int(lane);
      float x1 = static_cast<float>(qh[NOPE_D + r1]);
      float x2 = static_cast<float>(qh[NOPE_D + r1 + 1]);
      qr[r1]     = static_cast<T>(x1 * ct - x2 * st);
      qr[r1 + 1] = static_cast<T>(x1 * st + x2 * ct);
    } else {
      const device T* kvh = kv_a_full + size_t(b) * KVA_DIM;
      device T* kl = kv_latent + size_t(b) * KV_D;
      device T* kr = k_rope + size_t(b) * ROPE_D;
      float inv = glue_rms_inv_sg<KV_D, true>(kvh, KV_EPS, lane);
#pragma unroll
      for (int g = 0; g < 4; ++g) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          int idx = (g * 32 + int(lane)) * 4 + i;
          kl[idx] = glue_rms_apply<T, WT>(static_cast<float>(kvh[idx]), inv, kv_ln_w[idx]);
        }
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
    const size_t off = (size_t(b) * NH + h) * VD;
    const T g = static_cast<T>(glue_sigmoid(static_cast<float>(g_raw[size_t(b) * NH + h])));
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int c = int(lane) * 4 + i;
      y[off + c] = x[off + c] * g;
    }
"""

_ML_SRC = r"""
    // Multi-head batched GEMV: out[b, h, 0, n] = x[b, h, 0, :] @ w[h, n, :].T
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
      T inter[TM][4];
#pragma unroll
      for (int tm = 0; tm < TM; ++tm) {
        const device ushort* pw = (const device ushort*)(mat + size_t(tm) * KK + bn);
#pragma unroll
        for (int tn = 0; tn < 4; ++tn) inter[tm][tn] = as_type<bfloat16_t>(pw[tn]);
      }
#pragma unroll
      for (int bi = 0; bi < B_SUB; ++bi) {
        const int b = b0 + bi;
        if (b < BB) {
          float vc[TN];
          const device ushort* px = (const device ushort*)(x + (size_t(b) * NH + h) * KK + bn);
#pragma unroll
          for (int tn = 0; tn < 4; ++tn) vc[tn] = static_cast<float>(as_type<bfloat16_t>(px[tn]));
#pragma unroll
          for (int tm = 0; tm < TM; ++tm)
#pragma unroll
            for (int tn = 0; tn < 4; ++tn)
              res[bi][tm] += static_cast<float>(inter[tm][tn]) * vc[tn];
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
            if constexpr (HAS_GATE) v = v * gate;
            out[(size_t(b) * NH + h) * NN + out_row + tm] = v;
          }
        }
      }
    }
"""

_ML_FP8_SRC = r"""
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

    const device uchar* mat = w + size_t(h) * NN * KK + size_t(out_row) * KK;
    const device float* sc_h = sc_inv + size_t(h) * 4;
    int bn = int(lane) * TN;
    float res[B_SUB][TM];
#pragma unroll
    for (int bi = 0; bi < B_SUB; ++bi)
#pragma unroll
      for (int tm = 0; tm < TM; ++tm)
        res[bi][tm] = 0.0f;

    constexpr int n_iter = KK / blockN;
    for (int i = 0; i < n_iter; ++i) {
      const float sc = SCALE_BY_N ? sc_h[out_row >> 7] : sc_h[i];
      float wf[TM][4];
#pragma unroll
      for (int tm = 0; tm < TM; ++tm) {
        const uchar4 pk = *(const device uchar4*)(mat + size_t(tm) * KK + bn);
        wf[tm][0] = static_cast<float>(static_cast<bfloat16_t>(E4M3_LUT[pk.x] * sc));
        wf[tm][1] = static_cast<float>(static_cast<bfloat16_t>(E4M3_LUT[pk.y] * sc));
        wf[tm][2] = static_cast<float>(static_cast<bfloat16_t>(E4M3_LUT[pk.z] * sc));
        wf[tm][3] = static_cast<float>(static_cast<bfloat16_t>(E4M3_LUT[pk.w] * sc));
      }
#pragma unroll
      for (int bi = 0; bi < B_SUB; ++bi) {
        const int b = b0 + bi;
        if (b < BB) {
          float vc[TN];
          const device ushort* px = (const device ushort*)(x + (size_t(b) * NH + h) * KK + bn);
#pragma unroll
          for (int tn = 0; tn < 4; ++tn) vc[tn] = static_cast<float>(as_type<bfloat16_t>(px[tn]));
#pragma unroll
          for (int tm = 0; tm < TM; ++tm)
#pragma unroll
            for (int tn = 0; tn < 4; ++tn)
              res[bi][tm] += wf[tm][tn] * vc[tn];
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
            if constexpr (HAS_GATE) v = v * gate;
            out[(size_t(b) * NH + h) * NN + out_row + tm] = v;
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
            input_names=["q_full", "kv_a_full", "kv_ln_w", "offsets"],
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


def _ml_kernel():
    k = _KERNELS.get("ml")
    if k is None:
        k = _KERNELS["ml"] = mx.fast.metal_kernel(
            name="bv3_mla_multilinear",
            input_names=["x", "w", "g_raw"],
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
            input_names=["x", "w", "sc_inv", "g_raw"],
            output_names=["out"],
            header=GLUE_HEADER + "\n" + E4M3_LUT_METAL,
            source=_ML_FP8_SRC,
            ensure_row_contiguous=True,
        )
    return k


def _ml_cfg(B: int):
    if B <= 4:
        return 1, B, 4
    if B <= 8:
        return 1, B, 2
    if B <= 16:
        return 2, (B + 1) // 2, 2
    return 3, (B + 2) // 3, 2


def mla_multilinear_rows(x: mx.array, w: mx.array, g_raw: mx.array | None = None) -> mx.array:
    """Weights-read-once MultiLinear for (B, H, 1, K) @ (H, N, K).T, optional fused head gate."""
    B, H, L, K = x.shape
    N = w.shape[1]
    assert L == 1 and 1 <= B <= 24 and w.shape == (H, N, K) and K % 128 == 0 and N % 8 == 0
    has_gate = g_raw is not None
    gr = g_raw if has_gate else mx.zeros((1,), dtype=x.dtype)
    bsg, b_sub, tm = _ml_cfg(B)
    nt = N // (4 * tm)
    (out,) = _ml_kernel()(
        inputs=[x, w, gr],
        template=[
            ("T", x.dtype),
            ("BB", B),
            ("BSG", bsg),
            ("B_SUB", b_sub),
            ("NH", H),
            ("NN", N),
            ("KK", K),
            ("TM", tm),
            ("HAS_GATE", int(has_gate)),
        ],
        grid=(nt * 32, 4 * bsg, H),
        threadgroup=(32, 4 * bsg, 1),
        output_shapes=[(B, N * H) if has_gate else (B, H, 1, N)],
        output_dtypes=[x.dtype],
    )
    return out


def mla_multilinear_fp8_rows(
    x: mx.array, w_fp8: mx.array, w_sc: mx.array, *, scale_by_n: bool, g_raw: mx.array | None = None
) -> mx.array:
    """Bit-exact FP8 MultiLinear reading uint8 weights + float32 scale_inv directly."""
    B, H, L, K = x.shape
    N = w_fp8.shape[1]
    assert L == 1 and 1 <= B <= 24 and w_fp8.shape == (H, N, K) and K % 128 == 0 and N % 8 == 0
    has_gate = g_raw is not None
    gr = g_raw if has_gate else mx.zeros((1,), dtype=x.dtype)
    bsg, b_sub, tm = _ml_cfg(B)
    nt = N // (4 * tm)
    (out,) = _ml_fp8_kernel()(
        inputs=[x, w_fp8, w_sc, gr],
        template=[
            ("T", x.dtype),
            ("BB", B),
            ("BSG", bsg),
            ("B_SUB", b_sub),
            ("NH", H),
            ("NN", N),
            ("KK", K),
            ("TM", tm),
            ("SCALE_BY_N", int(bool(scale_by_n))),
            ("HAS_GATE", int(has_gate)),
        ],
        grid=(nt * 32, 4 * bsg, H),
        threadgroup=(32, 4 * bsg, 1),
        output_shapes=[(B, N * H) if has_gate else (B, H, 1, N)],
        output_dtypes=[x.dtype],
    )
    return out


def mla_multilinear_lin(x: mx.array, mod, g_raw: mx.array | None = None) -> mx.array:
    if hasattr(mod, "weight_fp8"):
        return mla_multilinear_fp8_rows(
            x, mod.weight_fp8, mod.weight_scale_inv, scale_by_n=mod.scale_by_n, g_raw=g_raw
        )
    return mla_multilinear_rows(x, mod.weight, g_raw=g_raw)


def mla_prep_kv_rope_rows(
    q_full: mx.array,
    kv_a_full: mx.array,
    kv_ln_w: mx.array,
    offsets,
    *,
    kv_eps: float = 1e-6,
    base: float = 6000000.0,
    scale: float = 1.0,
):
    B = q_full.shape[0]
    if not isinstance(offsets, mx.array):
        offsets = mx.array([int(offsets)] * B, dtype=mx.int32)
    elif offsets.dtype != mx.int32:
        offsets = offsets.astype(mx.int32)
    return _prep_kernel(kv_eps, math.log2(float(base)), float(scale))(
        inputs=[q_full, kv_a_full, kv_ln_w, offsets],
        template=[("T", q_full.dtype), ("WT", kv_ln_w.dtype)],
        grid=(32, 65, B),
        threadgroup=(32, 1, 1),
        output_shapes=[(B, 64, 1, 128), (B, 64, 1, 64), (B, 1, 1, 512), (B, 1, 1, 64)],
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


def mla_sdpa_rows(q_nope: mx.array, q_rope: mx.array, kv_latent_all: mx.array, k_rope_all: mx.array, scale: float, starts):
    """Exact MLA SDPA handling unpadded KVCache and heterogeneous left-padded BatchKVCache."""
    B = q_nope.shape[0]
    if B == 1:
        s = starts[0]
        kl = kv_latent_all if s == 0 else kv_latent_all[:, :, s:, :]
        kr = k_rope_all if s == 0 else k_rope_all[:, :, s:, :]
        pe = (q_rope * scale) @ kr.swapaxes(-1, -2)
        scores = (q_nope * scale) @ kl.swapaxes(-1, -2) + pe
        probs = mx.softmax(scores, axis=-1, precise=True)
        return probs @ kl

    # Batch > 1: run per-row on unpadded slices so softmax denominator & matmul M=64 match B=1 bit-for-bit.
    if all(s == 0 for s in starts):
        pe_all = (q_rope * scale) @ k_rope_all.swapaxes(-1, -2)
        sc_all = (q_nope * scale) @ kv_latent_all.swapaxes(-1, -2) + pe_all
        # Check at runtime if batched matmul/softmax matches? Remember probe showed batched padded differed by 9 bits even at lp=0!
        # Slice per row for strict bit-exactness:
        outs = []
        for b in range(B):
            kl_b = kv_latent_all[b : b + 1]
            kr_b = k_rope_all[b : b + 1]
            pe_b = (q_rope[b : b + 1] * scale) @ kr_b.swapaxes(-1, -2)
            sc_b = (q_nope[b : b + 1] * scale) @ kl_b.swapaxes(-1, -2) + pe_b
            pr_b = mx.softmax(sc_b, axis=-1, precise=True)
            outs.append(pr_b @ kl_b)
        return mx.concatenate(outs, axis=0)

    outs = []
    for b in range(B):
        s = starts[b]
        kl_b = kv_latent_all[b : b + 1] if s == 0 else kv_latent_all[b : b + 1, :, s:, :]
        kr_b = k_rope_all[b : b + 1] if s == 0 else k_rope_all[b : b + 1, :, s:, :]
        pe_b = (q_rope[b : b + 1] * scale) @ kr_b.swapaxes(-1, -2)
        sc_b = (q_nope[b : b + 1] * scale) @ kl_b.swapaxes(-1, -2) + pe_b
        pr_b = mx.softmax(sc_b, axis=-1, precise=True)
        outs.append(pr_b @ kl_b)
    return mx.concatenate(outs, axis=0)


def mla_layer_rows(h_in: mx.array, res: mx.array, attn, cache):
    """Full fused single-token (B <= 4) BailingMLA layer with residual add -> h_post (B, 4096)."""
    B = h_in.shape[0]
    q_a, kv_a_full, g_raw = gemv_lin_multi(
        h_in, [attn.q_a_proj, attn.kv_a_proj_with_mqa, attn.g_proj]
    )
    q_a_norm = mx.fast.rms_norm(q_a, attn.q_a_layernorm.weight, attn.q_a_layernorm.eps)
    q_full = gemv_lin(q_a_norm, attn.q_b_proj)
    q_nope, q_rope, kv_latent, k_rope = mla_prep_kv_rope_rows(
        q_full,
        kv_a_full,
        attn.kv_a_layernorm.weight,
        cache.offset,
        kv_eps=attn.kv_a_layernorm.eps,
        base=float(getattr(attn.rope, "base", 6000000.0)),
        scale=float(getattr(attn.rope, "scale", 1.0)),
    )
    q_nope = mla_multilinear_lin(q_nope, attn.embed_q)
    kl_all, kr_all = cache.update_and_fetch(kv_latent, k_rope)
    starts = _get_left_starts(cache, B)
    out = mla_sdpa_rows(q_nope, q_rope, kl_all, kr_all, attn.scale, starts)
    gated = mla_multilinear_lin(out, attn.unembed_out, g_raw=g_raw)
    return gemv_lin_res(gated, attn.dense, res)


def _mla_verify_proj_prep(h_in: mx.array, attn, offsets: mx.array):
    q_a, kv_a_full, g_raw = gemv_lin_multi(
        h_in, [attn.q_a_proj, attn.kv_a_proj_with_mqa, attn.g_proj]
    )
    q_a_norm = mx.fast.rms_norm(q_a, attn.q_a_layernorm.weight, attn.q_a_layernorm.eps)
    q_full = gemv_lin(q_a_norm, attn.q_b_proj)
    q_nope, q_rope, kv_latent, k_rope = mla_prep_kv_rope_rows(
        q_full,
        kv_a_full,
        attn.kv_a_layernorm.weight,
        offsets,
        kv_eps=attn.kv_a_layernorm.eps,
        base=float(getattr(attn.rope, "base", 6000000.0)),
        scale=float(getattr(attn.rope, "scale", 1.0)),
    )
    q_nope = mla_multilinear_lin(q_nope, attn.embed_q)
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
    scale: float,
):
    for t in range(lc):
        b = start_b + t
        end = base_end + t + 1
        kl_t = kl_buf[:, :, start_k:end, :]
        kr_t = kr_buf[:, :, start_k:end, :]
        pe = (q_rope[b : b + 1] * scale) @ kr_t.swapaxes(-1, -2)
        sc = (q_nope[b : b + 1] * scale) @ kl_t.swapaxes(-1, -2) + pe
        pr = mx.softmax(sc, axis=-1, precise=True)
        outs.append(pr @ kl_t)


def mla_verify_layer_rows(h_in: mx.array, res: mx.array, attn, cache, *, C: int, L: int):
    """Bit-exact causal multi-token (C * L <= 24) BailingMLA layer supporting KVCache, list[KVCache], BatchKVCache."""
    B = C * L
    assert h_in.shape[0] == B and 1 <= B <= 24

    if isinstance(cache, list):
        assert len(cache) == C
        offs_before = [int(c_i.offset) for c_i in cache]
        offsets = mx.array([o + t for o in offs_before for t in range(L)], dtype=mx.int32)
    elif hasattr(cache, "left_padding"):
        offsets = (cache.offset[:, None] + mx.arange(L, dtype=mx.int32)[None, :]).reshape(-1)
    else:
        assert C == 1
        off0 = int(cache.offset)
        offsets = mx.arange(off0, off0 + L, dtype=mx.int32)

    q_nope, q_rope, kv_latent, k_rope, g_raw = _mla_verify_proj_prep(h_in, attn, offsets)

    kl_in = kv_latent.reshape(C, 1, L, 512)
    kr_in = k_rope.reshape(C, 1, L, 64)
    scale = attn.scale
    outs = []

    if isinstance(cache, list):
        for c in range(C):
            off_c = offs_before[c]
            kl_c, kr_c = cache[c].update_and_fetch(kl_in[c : c + 1], kr_in[c : c + 1])
            _mla_append_causal_slice(outs, kl_c, kr_c, q_nope, q_rope, c * L, 0, off_c, L, scale)
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
                scale,
            )
    else:
        kl_all, kr_all = cache.update_and_fetch(kl_in, kr_in)
        _mla_append_causal_slice(outs, kl_all, kr_all, q_nope, q_rope, 0, 0, off0, L, scale)

    out = mx.concatenate(outs, axis=0)
    gated = mla_multilinear_lin(out, attn.unembed_out, g_raw=g_raw)
    return gemv_lin_res(gated, attn.dense, res)


def mla_verify_varlen_rows(h_in: mx.array, res: mx.array, attn, caches: list, Ls: list[int]):
    """Bit-exact variable-length causal (sum(Ls) <= 24) BailingMLA layer over unpadded `list[KVCache]`."""
    C = len(Ls)
    B = sum(Ls)
    assert isinstance(caches, list) and len(caches) == C and 1 <= B <= 24 and h_in.shape[0] == B
    assert all(lc >= 1 for lc in Ls)

    offs_before = [int(c_i.offset) for c_i in caches]
    offsets = mx.array([offs_before[c] + t for c in range(C) for t in range(Ls[c])], dtype=mx.int32)
    q_nope, q_rope, kv_latent, k_rope, g_raw = _mla_verify_proj_prep(h_in, attn, offsets)

    scale = attn.scale
    outs = []
    pos = 0
    for c, lc in enumerate(Ls):
        kl_c, kr_c = caches[c].update_and_fetch(
            kv_latent[pos : pos + lc].reshape(1, 1, lc, 512),
            k_rope[pos : pos + lc].reshape(1, 1, lc, 64),
        )
        _mla_append_causal_slice(outs, kl_c, kr_c, q_nope, q_rope, pos, 0, offs_before[c], lc, scale)
        pos += lc

    out = mx.concatenate(outs, axis=0)
    gated = mla_multilinear_lin(out, attn.unembed_out, g_raw=g_raw)
    return gemv_lin_res(gated, attn.dense, res)
