"""Bit-exact row-local RMSNorm and fused residual-add + RMSNorm kernels (B = 1..4)."""
import mlx.core as mx

from .glue import GLUE_HEADER

_KERNELS = {}

_RMS_SRC = r"""
    const int b = int(threadgroup_position_in_grid.y);
    const uint lid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    threadgroup float part[32];

    const device T* xr = x + size_t(b) * NN;
    device T* yr = y + size_t(b) * NN;

    float inv;
    if constexpr (TG == 32) {
      inv = glue_rms_inv_sg<NN, true>(xr, EPS, lane);
    } else {
      inv = glue_rms_inv_tg<NN, TG, true>(xr, EPS, part, lid);
    }
    for (int i = int(lid); i < NN; i += TG) {
      yr[i] = glue_rms_apply<T, WT>(static_cast<float>(xr[i]), inv, w[i]);
    }
"""

_ADD_RMS_SRC = r"""
    const int b = int(threadgroup_position_in_grid.y);
    const uint lid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    threadgroup float part[32];
    const size_t off = size_t(b) * NN;

    for (int i = int(lid); i < NN; i += TG) {
      h_out[off + i] = x[off + i] + add_in[off + i];
    }
    if constexpr (TG > 32) {
      threadgroup_barrier(mem_flags::mem_device);
    } else {
      simdgroup_barrier(mem_flags::mem_device);
    }

    const device T* hr = h_out + off;
    float inv;
    if constexpr (TG == 32) {
      inv = glue_rms_inv_sg<NN, true>(hr, EPS, lane);
    } else {
      inv = glue_rms_inv_tg<NN, TG, true>(hr, EPS, part, lid);
    }
    for (int i = int(lid); i < NN; i += TG) {
      y[off + i] = glue_rms_apply<T, WT>(static_cast<float>(hr[i]), inv, w[i]);
    }
"""


def _tname(dt):
    return {mx.bfloat16: "bfloat16_t", mx.float32: "float", mx.float16: "half"}[dt]


def _get_rms(dt, wt, tg, N, eps):
    key = ("rms", dt, wt, tg, N, float(eps))
    k = _KERNELS.get(key)
    if k is None:
        hdr = GLUE_HEADER + f"#define EPS {float(eps):.9g}f\n"
        k = _KERNELS[key] = mx.fast.metal_kernel(
            name=f"bv3_rms_{N}_{tg}",
            input_names=["x", "w"],
            output_names=["y"],
            header=hdr,
            source=_RMS_SRC,
            ensure_row_contiguous=True,
        )
    return k


def _get_add_rms(dt, wt, tg, N, eps):
    key = ("add_rms", dt, wt, tg, N, float(eps))
    k = _KERNELS.get(key)
    if k is None:
        hdr = GLUE_HEADER + f"#define EPS {float(eps):.9g}f\n"
        k = _KERNELS[key] = mx.fast.metal_kernel(
            name=f"bv3_add_rms_{N}_{tg}",
            input_names=["x", "add_in", "w"],
            output_names=["h_out", "y"],
            header=hdr,
            source=_ADD_RMS_SRC,
            ensure_row_contiguous=True,
        )
    return k


def _tg_for(N: int) -> int:
    return 32 if N <= 512 else 256


def rms_norm_rows(x: mx.array, weight: mx.array, eps: float = 1e-6) -> mx.array:
    """Bit-exact mx.fast.rms_norm(x, weight, eps) for 2D (B, N)."""
    assert x.ndim == 2
    B, N = x.shape
    tg = _tg_for(N)
    (y,) = _get_rms(x.dtype, weight.dtype, tg, N, eps)(
        inputs=[x, weight],
        template=[("T", x.dtype), ("WT", weight.dtype), ("NN", N), ("TG", tg)],
        grid=(tg, B, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(B, N)],
        output_dtypes=[x.dtype],
    )
    return y


def add_rms_norm_rows(x: mx.array, add: mx.array, weight: mx.array, eps: float = 1e-6):
    """Return (x + add, rms_norm(x + add, weight, eps)) in 1 launch for 2D (B, N)."""
    assert x.ndim == 2 and x.shape == add.shape
    B, N = x.shape
    tg = _tg_for(N)
    h_out, y = _get_add_rms(x.dtype, weight.dtype, tg, N, eps)(
        inputs=[x, add, weight],
        template=[("T", x.dtype), ("WT", weight.dtype), ("NN", N), ("TG", tg)],
        grid=(tg, B, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(B, N), (B, N)],
        output_dtypes=[x.dtype, x.dtype],
    )
    return h_out, y
