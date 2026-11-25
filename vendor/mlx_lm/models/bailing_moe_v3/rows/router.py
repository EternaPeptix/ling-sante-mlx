"""Exact register-only group-limited router & fused router+LPT-dedup kernels for BailingGate.

512 experts = 8 groups x 64, topk_group = 4, top_k = 8.
Reproduces `_group_expert_select` bit-for-bit (0 diffs vs baseline.json), with:
  - Zero threadgroup memory and zero barriers in `group_router_rows` (drops dead tg_score[512] & tg_biased[512]).
  - `group_router_dedup_rows` fusing group routing + LPT `(cnt DESC, exp ASC)` packed CSR dedup into 1 launch.
"""
import mlx.core as mx

from .gemv import router_gemv_rows
from .glue import GLUE_HEADER, flit

_KERNELS = {}

# Shared LPT (cnt DESC, exp ASC) packed CSR dedup body over `threadgroup uint ind_s[NPAIR]`.
# Executed by a single 32-lane simdgroup. Writes:
#   u_meta[0..uniq-1]: exp (10b) | (off << 10) (8b) | (cnt << 18) (8b), sorted (cnt DESC, exp ASC)
#   u_pqs[0..NPAIR-1]: flat (b*8 + j) indices grouped per unique expert
#   n_hot[0]: count of unique experts with cnt >= 2 (living at u in [0, n_hot))
#   n_uniq[0]: total unique expert count (cnt == 1 experts live at u in [n_hot, n_uniq), exp ASC)
LPT_DEDUP_HELPER = r"""
template <int NPAIR>
inline void lpt_dedup_sg(
    threadgroup const uint* ind_s,
    threadgroup uchar* seen,
    threadgroup uint* raw_key,
    threadgroup uint* raw_meta,
    device uint* u_meta,
    device uint* u_pqs,
    device int* n_hot,
    device int* n_uniq,
    uint lane) {
  for (int i = int(lane); i < NPAIR; i += 32) {
    seen[i] = 0;
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);

  int uniq = 0;
  int total_out = 0;
  int hot = 0;
  for (int q = 0; q < NPAIR; ++q) {
    if (seen[q]) continue;
    const uint e = ind_s[q];
    int cnt = 0;
    for (int base = 0; base < NPAIR; base += 32) {
      int p = base + int(lane);
      bool match = (p >= q) && (p < NPAIR) && (ind_s[p] == e);
      if (match) seen[p] = 1;
      int take = match ? 1 : 0;
      int pre = simd_prefix_exclusive_sum(take);
      if (take) u_pqs[total_out + cnt + pre] = (uint)p;
      cnt += simd_sum(take);
    }
    if (lane == 0) {
      // Strictly unique sort key (all experts distinct): (NPAIR - cnt) ASC (= cnt DESC), then exp ASC.
      raw_key[uniq] = (uint(NPAIR - cnt) << 10) | (e & 0x3FFu);
      raw_meta[uniq] = (e & 0x3FFu) | (uint(total_out) << 10) | (uint(cnt) << 18);
      if (cnt >= 2) hot += 1;
    }
    uniq += 1;
    total_out += cnt;
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Parallel O(uniq) rank sort across all 32 lanes (all keys strictly distinct -> zero tie ambiguity).
  for (int i = int(lane); i < uniq; i += 32) {
    const uint ki = raw_key[i];
    int rank = 0;
    for (int j = 0; j < uniq; ++j) {
      rank += (raw_key[j] < ki) ? 1 : 0;
    }
    u_meta[rank] = raw_meta[i];
  }
  if (lane == 0) {
    n_hot[0] = hot;
    n_uniq[0] = uniq;
  }
}
"""

_ROUTER_CORE_HELPER = r"""
template <bool WRITE_TG>
inline void group_router_row_sg(
    const device float* logits,
    const device float* bias,
    device uint* inds,
    device float* weights,
    threadgroup uint* ind_s_row,
    const uint b,
    const uint lane) {
  GL_NOFMA
  constexpr int NE = 512;
  constexpr int NG = 8;
  constexpr int GSZ = 64;
  constexpr int K = 8;
  constexpr int R = NE / 32;  // 16 elements per lane: biased[2*g] = g*64+lane, biased[2*g+1] = g*64+32+lane
  const device float* row = logits + size_t(b) * NE;

  float score[R];
  float biased[R];
#pragma unroll
  for (int i = 0; i < R; i++) {
    int e = int(lane) + 32 * i;
    float x = row[e];
    float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(x)));
    score[i] = (x < 0.0f) ? y : (1.0f - y);
    biased[i] = score[i] + bias[e];
  }

  // Group top-2 directly from registers (zero threadgroup memory, zero barriers).
  float gs[NG];
#pragma unroll
  for (int g = 0; g < NG; g++) {
    int e0 = g * GSZ + int(lane);
    int e1 = e0 + 32;
    float v0 = biased[2 * g];
    float v1 = biased[2 * g + 1];
    bool pick0 = (v0 > v1) || (v0 == v1 && e0 < e1);
    float best_v = pick0 ? v0 : v1;
    int best_e = pick0 ? e0 : e1;
    float top1 = simd_max(best_v);
    int win1 = simd_min((best_v == top1) ? best_e : NE);

    if (e0 == win1) v0 = -INFINITY;
    if (e1 == win1) v1 = -INFINITY;
    pick0 = (v0 > v1) || (v0 == v1 && e0 < e1);
    best_v = pick0 ? v0 : v1;
    float top2 = simd_max(best_v);
    gs[g] = top1 + top2;
  }

  // Mask out the (NG - TOPK_GROUP) smallest groups (ties break by smaller group index).
  uint keep_mask = 0xFFu;
#pragma unroll
  for (int m = 0; m < (NG - TOPK_GROUP); m++) {
    int min_g = 0;
    float min_v = INFINITY;
#pragma unroll
    for (int g = 0; g < NG; g++) {
      if (((keep_mask >> g) & 1u) && (gs[g] < min_v)) {
        min_v = gs[g];
        min_g = g;
      }
    }
    keep_mask &= ~(1u << min_g);
  }

#pragma unroll
  for (int g = 0; g < NG; g++) {
    if (!((keep_mask >> g) & 1u)) {
      biased[2 * g] = -INFINITY;
      biased[2 * g + 1] = -INFINITY;
    }
  }

  // Select top-K=8 experts in descending order (ties break by smaller expert index).
  float w[K];
#pragma unroll
  for (int k = 0; k < K; k++) {
    float best = -INFINITY;
    int best_e = NE;
#pragma unroll
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
#pragma unroll
    for (int i = 0; i < R; i++) {
      if (int(lane) + 32 * i == winner) {
        s = score[i];
        biased[i] = -INFINITY;
      }
    }
    w[k] = simd_sum(s);
    if (lane == 0) {
      inds[size_t(b) * K + k] = uint(winner);
      if constexpr (WRITE_TG) {
        ind_s_row[k] = uint(winner);
      }
    }
  }

  float total = 0.0f;
#pragma unroll
  for (int k = 0; k < K; k++) {
    total = w[k] + total;
  }
  if (lane == 0) {
    float denom = total + EPS;
#pragma unroll
    for (int k = 0; k < K; k++) {
      weights[size_t(b) * K + k] = (w[k] / denom) * SCALE;
    }
  }
}
"""

_SRC = r"""
    const uint b = threadgroup_position_in_grid.y;
    const uint lane = thread_index_in_simdgroup;
    group_router_row_sg<false>(logits, bias, inds, weights, (threadgroup uint*)nullptr, b, lane);
"""

_SRC_DEDUP = r"""
    constexpr int NPAIR = BB * 8;
    const uint b = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    threadgroup uint ind_s[NPAIR];
    threadgroup uchar seen[NPAIR];
    threadgroup uint raw_key[NPAIR];
    threadgroup uint raw_meta[NPAIR];
    if (int(b) < BB) {
      group_router_row_sg<true>(logits, bias, inds, weights, ind_s + b * 8, b, lane);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (b == 0) {
      lpt_dedup_sg<NPAIR>(ind_s, seen, raw_key, raw_meta, u_meta, u_pqs, n_hot, n_uniq, lane);
    }
"""


def _kernel(topk_group: int, route_scale: float, eps: float, with_dedup: bool = False):
    key = (topk_group, float(route_scale), float(eps), bool(with_dedup))
    k = _KERNELS.get(key)
    if k is None:
        hdr = (
            GLUE_HEADER
            + f"""
#define TOPK_GROUP {topk_group}
#define SCALE {flit(route_scale)}
#define EPS {flit(eps)}
"""
            + _ROUTER_CORE_HELPER
            + (LPT_DEDUP_HELPER if with_dedup else "")
        )
        if with_dedup:
            k = _KERNELS[key] = mx.fast.metal_kernel(
                name=f"bv3_group_router_dedup_tg{topk_group}",
                input_names=["logits", "bias"],
                output_names=["inds", "weights", "u_meta", "u_pqs", "n_hot", "n_uniq"],
                header=hdr,
                source=_SRC_DEDUP,
                ensure_row_contiguous=True,
            )
        else:
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
    inds, wts = _kernel(topk_group, routed_scaling_factor, eps, with_dedup=False)(
        inputs=[logits, expert_bias],
        grid=(32, B, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(B, top_k), (B, top_k)],
        output_dtypes=[mx.uint32, mx.float32],
    )
    return inds, wts


def group_router_dedup_rows(
    logits: mx.array,
    expert_bias: mx.array,
    *,
    top_k: int = 8,
    n_group: int = 8,
    topk_group: int = 4,
    routed_scaling_factor: float = 2.5,
    eps: float = 1e-20,
):
    """Fused register-only group router + LPT `(cnt DESC, exp ASC)` packed CSR dedup in 1 launch."""
    assert logits.ndim == 2 and logits.shape[1] == 512 and logits.dtype == mx.float32
    assert expert_bias.shape == (512,) and expert_bias.dtype == mx.float32
    assert top_k == 8 and n_group == 8
    B = logits.shape[0]
    assert 1 <= B <= 24
    NPAIR = B * top_k
    inds, wts, u_meta, u_pqs, n_hot, n_uniq = _kernel(topk_group, routed_scaling_factor, eps, with_dedup=True)(
        inputs=[logits, expert_bias],
        template=[("BB", B)],
        grid=(32, B, 1),
        threadgroup=(32, B, 1),
        output_shapes=[(B, top_k), (B, top_k), (NPAIR,), (NPAIR,), (1,), (1,)],
        output_dtypes=[mx.uint32, mx.float32, mx.uint32, mx.uint32, mx.int32, mx.int32],
    )
    return inds, wts, (u_meta, u_pqs, n_hot, n_uniq, n_uniq)


def gate_rows(x_bf16: mx.array, w_bf16: mx.array, expert_bias: mx.array, **kw):
    """Full BailingGate on (B, K) bf16 -> (inds (B, 8) u32, weights (B, 8) f32)."""
    logits = router_gemv_rows(x_bf16, w_bf16)
    return group_router_rows(logits, expert_bias, **kw)


def gate_dedup_rows(x_bf16: mx.array, w_bf16: mx.array, expert_bias: mx.array, **kw):
    """Full BailingGate + LPT packed CSR dedup -> (inds, weights, dedup)."""
    logits = router_gemv_rows(x_bf16, w_bf16)
    return group_router_dedup_rows(logits, expert_bias, **kw)
