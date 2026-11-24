"""Exact group-limited router kernel for BailingGate (512 experts = 8 groups x 64, topk_group=4, top_k=8).

Reproduces `_group_expert_select` bit-for-bit including exact index order (which matches
MLX's stable `argpartition` -> `gpu_merge_sort`) and unbiased score normalization.
Can also fuse router GEMV + routing into 2 tiny launches (avoiding w.astype(f32) and ~15 routing ops).
"""
import mlx.core as mx

from .gemv import router_gemv_rows
from .glue import GLUE_HEADER, flit

_KERNELS = {}

_SRC = r"""
    GL_NOFMA
    // One 32-lane simdgroup processes one token row `b` (B <= 8 -> B simdgroups or 1 simdgroup per threadgroup).
    // Grid: (32, B, 1), threadgroup: (32, 1, 1).
    constexpr int NE = 512;
    constexpr int NG = 8;
    constexpr int GSZ = 64;
    constexpr int K = 8;
    constexpr int R = NE / 32;  // 16 elements per lane
    const uint b = threadgroup_position_in_grid.y;
    const uint lane = thread_index_in_simdgroup;
    const device float* row = logits + size_t(b) * NE;

    float score[R];
    float biased[R];
    for (int i = 0; i < R; i++) {
      int e = int(lane) + 32 * i;
      float x = row[e];
      float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(x)));
      score[i] = (x < 0.0f) ? y : (1.0f - y);
      biased[i] = score[i] + bias[e];
    }

    // Store scores & biased in threadgroup memory for clean group-wise top-2 & top-8 selection.
    threadgroup float tg_score[NE];
    threadgroup float tg_biased[NE];
    for (int i = 0; i < R; i++) {
      int e = int(lane) + 32 * i;
      tg_score[e] = score[i];
      tg_biased[e] = biased[i];
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // Group g = 0..7 has 64 elements: each lane handles 2 elements per group? Or 32 lanes handle 1 group at a time (2 elements/lane)!
    // Actually each lane handles 2 elements of group g -> 2 reductions for top-2 in group g!
    threadgroup float tg_gscore[NG];
    for (int g = 0; g < NG; g++) {
      int e0 = g * GSZ + int(lane);
      int e1 = e0 + 32;
      float v0 = tg_biased[e0];
      float v1 = tg_biased[e1];
      // Pick top1 (largest value, tie -> smaller index)
      bool pick0 = (v0 > v1) || (v0 == v1 && e0 < e1);
      float best_v = pick0 ? v0 : v1;
      int best_e = pick0 ? e0 : e1;
      float top1 = simd_max(best_v);
      int win1 = simd_min((best_v == top1) ? best_e : NE);

      // Mask win1, pick top2
      if (e0 == win1) v0 = -INFINITY;
      if (e1 == win1) v1 = -INFINITY;
      pick0 = (v0 > v1) || (v0 == v1 && e0 < e1);
      best_v = pick0 ? v0 : v1;
      best_e = pick0 ? e0 : e1;
      float top2 = simd_max(best_v);
      if (lane == 0) {
        tg_gscore[g] = top1 + top2;
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // Determine which 4 groups are masked (the 4 smallest tg_gscore; ties break by smaller group index as in argpartition ascending)
    threadgroup int tg_keep[NG];
    if (lane < NG) tg_keep[lane] = 1;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
      float gs[NG];
      for (int g = 0; g < NG; g++) gs[g] = tg_gscore[g];
      for (int m = 0; m < (NG - TOPK_GROUP); m++) {
        int min_g = 0;
        float min_v = INFINITY;
        for (int g = 0; g < NG; g++) {
          if (tg_keep[g] && (gs[g] < min_v)) {
            min_v = gs[g];
            min_g = g;
          }
        }
        tg_keep[min_g] = 0;
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // Mask out dropped groups in biased registers
    for (int i = 0; i < R; i++) {
      int e = int(lane) + 32 * i;
      int g = e / GSZ;
      if (!tg_keep[g]) biased[i] = -INFINITY;
    }

    // Select top-K=8 experts in descending order (ties break by smaller expert index -> matches argpartition(-routing_scores))
    float w[K];
    for (int k = 0; k < K; k++) {
      float best = -INFINITY;
      int best_e = NE;
      for (int i = 0; i < R; i++) {
        int e = int(lane) + 32 * i;
        if (biased[i] > best || (biased[i] == best && e < best_e)) {
          best = biased[i];
          best_e = e;
        }
      }
      float top = simd_max(best);
      int winner = simd_min((best == top) ? best_e : NE);
      float s = 0.0f;
      for (int i = 0; i < R; i++) {
        if (int(lane) + 32 * i == winner) {
          s = score[i];
          biased[i] = -INFINITY;
        }
      }
      w[k] = simd_sum(s);
      if (lane == 0) {
        inds[size_t(b) * K + k] = uint(winner);
      }
    }

    float total = 0.0f;
    for (int k = 0; k < K; k++) {
      total = w[k] + total;
    }
    if (lane == 0) {
      float denom = total + EPS;
      for (int k = 0; k < K; k++) {
        weights[size_t(b) * K + k] = (w[k] / denom) * SCALE;
      }
    }
"""


def _kernel(topk_group: int, route_scale: float, eps: float):
    key = (topk_group, float(route_scale), float(eps))
    k = _KERNELS.get(key)
    if k is None:
        hdr = GLUE_HEADER + f"""
#define TOPK_GROUP {topk_group}
#define SCALE {flit(route_scale)}
#define EPS {flit(eps)}
"""
        k = _KERNELS[key] = mx.fast.metal_kernel(
            name=f"bv3_group_router_tg{topk_group}",
            input_names=["logits", "bias"],
            output_names=["inds", "weights"],
            header=hdr,
            source=_SRC,
            ensure_row_contiguous=True,
        )
    return k


def group_router_rows(
    logits: mx.array,
    expert_bias: mx.array,
    *,
    top_k: int = 8,
    n_group: int = 8,
    topk_group: int = 4,
    routed_scaling_factor: float = 2.5,
    eps: float = 1e-20,
):
    assert logits.ndim == 2 and logits.shape[1] == 512 and logits.dtype == mx.float32
    assert expert_bias.shape == (512,) and expert_bias.dtype == mx.float32
    assert top_k == 8 and n_group == 8
    B = logits.shape[0]
    inds, wts = _kernel(topk_group, routed_scaling_factor, eps)(
        inputs=[logits, expert_bias],
        grid=(32, B, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(B, top_k), (B, top_k)],
        output_dtypes=[mx.uint32, mx.float32],
    )
    return inds, wts


def gate_rows(x_bf16: mx.array, w_bf16: mx.array, expert_bias: mx.array, **kw):
    """Full BailingGate on (B, K) bf16 -> (inds (B, 8) u32, weights (B, 8) f32)."""
    logits = router_gemv_rows(x_bf16, w_bf16)
    return group_router_rows(logits, expert_bias, **kw)
