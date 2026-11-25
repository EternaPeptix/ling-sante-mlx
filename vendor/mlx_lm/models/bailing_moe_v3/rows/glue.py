"""Exact Metal device helpers ("glue") for LingSante BailingMoeV3 row kernels.

All functions reproduce MLX 0.32.4 arithmetic bit-for-bit (GL_NOFMA disables compiler FP contraction
except where MLX's own metallib kernels contract, i.e. explicit fma() in rms_norm).
"""
import mlx.core as mx
import numpy as np

__all__ = ["GLUE_HEADER", "flit"]


def flit(x) -> str:
    """Float literal parsed by Metal to exact float32(x)."""
    s = "%.9g" % float(np.float32(x))
    if "." not in s and "e" not in s and "inf" not in s and "nan" not in s:
        s += ".0"
    return s + "f"


GLUE_HEADER = r"""
#define GL_NOFMA _Pragma("clang fp contract(off)")

// Unary ops matching MLX kernels/unary_ops.h
inline float glue_sigmoid(float x) {
  GL_NOFMA
  float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(x)));
  return (x < 0.0f) ? y : (1.0f - y);
}
inline float glue_tanh(float x) { return metal::precise::tanh(x); }
inline float glue_exp(float x) { return metal::precise::exp(x); }
inline float glue_log(float x) { return metal::precise::log(x); }
inline float glue_sqrt(float x) { return metal::precise::sqrt(x); }
inline float glue_min(float x, float y) { return metal::isnan(x) ? x : (x < y ? x : y); }
inline float glue_max(float x, float y) { return metal::isnan(x) ? x : (x > y ? x : y); }

// nn.silu(bf16) * bf16 (mlx_lm.layers.common.swiglu):
// MLX Sigmoid::operator()(bfloat16_t): abs/exp return bfloat16_t (bf16_math.h), 1 + ex and 1 / ... and 1 - y in bfloat16_t.
inline bfloat16_t glue_sigmoid_bf16(bfloat16_t x) {
  GL_NOFMA
  bfloat16_t ax = static_cast<bfloat16_t>(metal::abs(static_cast<float>(x)));
  bfloat16_t ex = static_cast<bfloat16_t>(metal::precise::exp(static_cast<float>(ax)));
  bfloat16_t one = static_cast<bfloat16_t>(1.0f);
  bfloat16_t y = one / (one + ex);
  return (x < static_cast<bfloat16_t>(0.0f)) ? y : (one - y);
}

inline bfloat16_t glue_swiglu_bf16(bfloat16_t g, bfloat16_t u) {
  GL_NOFMA
  bfloat16_t s = glue_sigmoid_bf16(g);
  bfloat16_t sil = g * s;
  return sil * u;
}

// mx.fast.rms_norm (kernels/rms_norm.metal rms_single_row, N <= 4096, RF=true uses fma)
template <int N, bool RF, typename P>
inline float glue_rms_inv_sg(P x, float eps, uint lane) {
  GL_NOFMA
  constexpr int NSG = (((N + 3) / 4 + 31) / 32);
  float v = 0.0f;
  for (int g = 0; g < NSG; g++) {
    float acc = 0.0f;
    for (int i = 0; i < 4; i++) {
      int idx = (g * 32 + int(lane)) * 4 + i;
      float xi = (idx < N) ? float(x[idx]) : 0.0f;
      acc = RF ? fma(xi, xi, acc) : (acc + xi * xi);
    }
    acc = simd_sum(acc);
    if (lane == uint(g)) v = acc;
  }
  float tot = simd_sum(v);
  return metal::precise::rsqrt(tot / float(N) + eps);
}

template <int N, int TG, bool RF, typename P>
inline float glue_rms_inv_tg(P x, float eps, threadgroup float* part, uint lid) {
  GL_NOFMA
  constexpr int NSG = (((N + 3) / 4 + 31) / 32);
  uint lane = lid & 31, sg = lid >> 5;
  for (int g = sg; g < NSG; g += TG / 32) {
    float acc = 0.0f;
    for (int i = 0; i < 4; i++) {
      int idx = (g * 32 + int(lane)) * 4 + i;
      float xi = (idx < N) ? float(x[idx]) : 0.0f;
      acc = RF ? fma(xi, xi, acc) : (acc + xi * xi);
    }
    acc = simd_sum(acc);
    if (lane == 0) part[g] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float tot = simd_sum(lane < uint(NSG) ? part[lane] : 0.0f);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return metal::precise::rsqrt(tot / float(N) + eps);
}

template <typename T, typename W>
inline T glue_rms_apply(float xi, float inv, W w) {
  return static_cast<T>(w) * static_cast<T>(xi * inv);
}

template <typename T>
inline T glue_rms_apply_nw(float xi, float inv) {
  return static_cast<T>(xi * inv);
}

// MoE combine over K=8 experts: (routed * weights[..., None]).sum(axis=-2).astype(bf16)
// Note: in BailingSparseMoE, routed is bf16, scores is f32 -> promoted to f32 product, col_reduce_small left-fold.
inline float glue_moe_combine8(thread const float* es, thread const float* w) {
  GL_NOFMA
  float t = (es[0] * w[0]) + 0.0f;
  for (int j = 1; j < 8; j++) {
    float p = es[j] * w[j];
    t = p + t;
  }
  return t;
}

// Zero-Multiply-256 vectorized F8_E4M3FN x4 -> bfloat4 decode (sc256 = sc * 256.0f).
// Bit-exact across all 256 E4M3FN codes (incl. signed zero, subnormals, normals, NaN).
inline bfloat4 glue_e4m3x4_to_bf4(uchar4 pk, float sc256) {
  ushort4 b4 = ushort4(pk);
  ushort4 v4 = b4 & ushort4(127);
  ushort4 u4 = (v4 << 7) | (((v4 + ushort4(1)) >> 7) << 14) | ((b4 & ushort4(128)) << 8);
  return bfloat4(float4(as_type<half4>(u4)) * sc256);
}

// T6: Exact single-lane 4-element FMA partial sum for mx.fast.rms_norm (N <= 4096, RF=true).
inline float glue_rms_lane4_ss(float x0, float x1, float x2, float x3) {
  GL_NOFMA
  float acc = 0.0f;
  acc = fma(x0, x0, acc);
  acc = fma(x1, x1, acc);
  acc = fma(x2, x2, acc);
  acc = fma(x3, x3, acc);
  return acc;
}

// T6: Reconstruct bit-exact mx.fast.rms_norm inv_rms from producer-emitted butterfly partial sums.
// BLOCKM in {4, 8, 16, 32, 64, 128} is the number of contiguous output elements per producer TG
// (so P = 128 / BLOCKM partial floats per 128-element rms_norm group, N % 128 == 0, NSG = N / 128 <= 32).
template <int N, int BLOCKM>
inline float glue_rms_inv_from_partss(const device float* part_ss, float eps, uint lane) {
  GL_NOFMA
  constexpr int NSG = N / 128;
  constexpr int P = 128 / BLOCKM;
  float v = 0.0f;
  if constexpr (P == 1) {
    if (lane < uint(NSG)) {
      v = part_ss[lane];
    }
  } else if constexpr (P == 2) {
    if (lane < uint(NSG)) {
      float2 a = *(const device float2*)(part_ss + lane * 2);
      v = a.x + a.y;
    }
  } else if constexpr (P == 4) {
    if (lane < uint(NSG)) {
      float4 a = *(const device float4*)(part_ss + lane * 4);
      v = (a.x + a.y) + (a.z + a.w);
    }
  } else if constexpr (P == 8) {
    if (lane < uint(NSG)) {
      float4 a = *(const device float4*)(part_ss + lane * 8);
      float4 b = *(const device float4*)(part_ss + lane * 8 + 4);
      v = ((a.x + a.y) + (a.z + a.w)) + ((b.x + b.y) + (b.z + b.w));
    }
  } else if constexpr (P == 16) {
    if (lane < uint(NSG)) {
      const device float4* p4 = (const device float4*)(part_ss + lane * 16);
      float4 a = p4[0], b = p4[1], c = p4[2], d = p4[3];
      float s0 = ((a.x + a.y) + (a.z + a.w)) + ((b.x + b.y) + (b.z + b.w));
      float s1 = ((c.x + c.y) + (c.z + c.w)) + ((d.x + d.y) + (d.z + d.w));
      v = s0 + s1;
    }
  } else if constexpr (P == 32) {
    for (int g = 0; g < NSG; ++g) {
      float s = simd_sum(part_ss[g * 32 + int(lane)]);
      if (lane == uint(g)) v = s;
    }
  }
  float tot = simd_sum(v);
  return metal::precise::rsqrt(tot / float(N) + eps);
}
"""
