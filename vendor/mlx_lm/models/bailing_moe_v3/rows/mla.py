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
    // Uses exact MLX (bm=4, bn=1, sm=1, sn=32, tm=4, tn=4) K-reduction order.
    constexpr int BM = 4;
    constexpr int TM = 4;
    constexpr int TN = 4;
    constexpr int blockM = BM * TM;  // 16
    constexpr int blockN = 32 * TN;  // 128

    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint tgx = threadgroup_position_in_grid.x;
    const uint h = threadgroup_position_in_grid.z;

    const int out_row = int(tgx) * blockM + int(sg) * TM;
    if (out_row >= NN) return;

    const device T* mat = w + size_t(h) * NN * KK + size_t(out_row) * KK;
    int bn = int(lane) * TN;
    float res[BB][TM];
#pragma unroll
    for (int b = 0; b < BB; ++b)
#pragma unroll
      for (int tm = 0; tm < TM; ++tm)
        res[b][tm] = 0.0f;

    constexpr int n_iter = KK / blockN;
    for (int i = 0; i < n_iter; ++i) {
      float vc[BB][TN];
#pragma unroll
      for (int b = 0; b < BB; ++b) {
        const device ushort* px = (const device ushort*)(x + (size_t(b) * NH + h) * KK + bn);
#pragma unroll
        for (int tn = 0; tn < 4; ++tn) vc[b][tn] = static_cast<float>(as_type<bfloat16_t>(px[tn]));
      }
#pragma unroll
      for (int tm = 0; tm < TM; ++tm) {
        const device ushort* pw = (const device ushort*)(mat + size_t(tm) * KK + bn);
        T inter[4];
#pragma unroll
        for (int tn = 0; tn < 4; ++tn) inter[tn] = as_type<bfloat16_t>(pw[tn]);
#pragma unroll
        for (int b = 0; b < BB; ++b)
#pragma unroll
          for (int tn = 0; tn < 4; ++tn)
            res[b][tm] += static_cast<float>(inter[tn]) * vc[b][tn];
      }
      bn += blockN;
    }

#pragma unroll
    for (int b = 0; b < BB; ++b) {
      T gate = static_cast<T>(1.0f);
      if constexpr (HAS_GATE) {
        gate = static_cast<T>(glue_sigmoid(static_cast<float>(g_raw[size_t(b) * NH + h])));
      }
#pragma unroll
      for (int tm = 0; tm < TM; ++tm) {
#pragma unroll
        for (ushort sn = 16; sn >= 1; sn >>= 1) {
          res[b][tm] += simd_shuffle_down(res[b][tm], sn);
        }
        if (lane == 0) {
          T v = static_cast<T>(res[b][tm]);
          if constexpr (HAS_GATE) v = v * gate;
          out[(size_t(b) * NH + h) * NN + out_row + tm] = v;
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


def mla_multilinear_rows(x: mx.array, w: mx.array, g_raw: mx.array | None = None) -> mx.array:
    """Weights-read-once MultiLinear for (B, H, 1, K) @ (H, N, K).T, optional fused head gate."""
    B, H, L, K = x.shape
    N = w.shape[1]
    assert L == 1 and 1 <= B <= 4 and w.shape == (H, N, K) and K % 128 == 0 and N % 16 == 0
    has_gate = g_raw is not None
    gr = g_raw if has_gate else mx.zeros((1,), dtype=x.dtype)
    nt = N // 16
    (out,) = _ml_kernel()(
        inputs=[x, w, gr],
        template=[("T", x.dtype), ("BB", B), ("NH", H), ("NN", N), ("KK", K), ("HAS_GATE", int(has_gate))],
        grid=(nt * 32, 4, H),
        threadgroup=(32, 4, 1),
        output_shapes=[(B, N * H) if has_gate else (B, H, 1, N)],
        output_dtypes=[x.dtype],
    )
    return out


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
    q_a, kv_a_full, g_raw = gemv_rows_multi(
        h_in, [attn.q_a_proj.weight, attn.kv_a_proj_with_mqa.weight, attn.g_proj.weight]
    )
    q_a_norm = mx.fast.rms_norm(q_a, attn.q_a_layernorm.weight, attn.q_a_layernorm.eps)
    q_full = gemv_rows(q_a_norm, attn.q_b_proj.weight)
    q_nope, q_rope, kv_latent, k_rope = mla_prep_kv_rope_rows(
        q_full,
        kv_a_full,
        attn.kv_a_layernorm.weight,
        cache.offset,
        kv_eps=attn.kv_a_layernorm.eps,
        base=float(getattr(attn.rope, "base", 6000000.0)),
        scale=float(getattr(attn.rope, "scale", 1.0)),
    )
    q_nope = mla_multilinear_rows(q_nope, attn.embed_q.weight)
    kl_all, kr_all = cache.update_and_fetch(kv_latent, k_rope)
    starts = _get_left_starts(cache, B)
    out = mla_sdpa_rows(q_nope, q_rope, kl_all, kr_all, attn.scale, starts)
    gated = mla_multilinear_rows(out, attn.unembed_out.weight, g_raw=g_raw)
    return gemv_rows_res(gated, attn.dense.weight, res)
