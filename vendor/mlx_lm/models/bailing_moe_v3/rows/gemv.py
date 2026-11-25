"""Exact multi-row bf16 / f32 GEMV row kernels for BailingMoeV3 (B = 1..4).

Row b of a B-row launch is bitwise identical to MLX 0.32.4 M == 1 (`x[b:b+1] @ w.T`,
which dispatches `GEMVKernel::run` in `mlx/backend/metal/kernels/gemv.h`), and every
weight matrix is read once across all B rows.

Also provides:
  - `gemv_rows_res(x, w, res)`: exact `gemv_rows(x, w) + res` (bf16)
  - `gemv_swiglu_rows(x, w_gate, w_up)`: exact `swiglu(gemv(x, w_gate), gemv(x, w_up))` (bf16)
  - `router_gemv_rows(x_bf16, w_bf16)`: exact `x.astype(f32) @ w.astype(f32).T` (reads w in bf16!)
"""
import os

import mlx.core as mx

from .glue import GLUE_HEADER

MAX_B = 24
_KERNELS = {}
_GEMV_BM_LARGE = int(os.environ.get("LINGSANTE_GEMV_BM", "8"))

_HEADER = (
    GLUE_HEADER
    + r"""
struct Bf4 { ushort v[4]; };

template <typename T, typename WT, typename XT, int B, int BSG, int B_SUB, int BM, int BN, int SM, int SN, int TM, int TN, int RES, int SWIGLU, int KK>
METAL_FUNC void bailing_gemv_rows_run(
    const device WT* mat,
    const device WT* mat_up,
    const device XT* in_vec,   // (B, KK) row-major
    const device T* res_vec,   // (B, N) row-major when RES == 1
    device T* out_vec,         // (B, N) row-major
    const int N,
    threadgroup float* tgp_memory,
    uint tgx,
    uint simd_gid,
    uint simd_lid) {
  constexpr int threadsM = BM * SM;
  constexpr int threadsN = BN * SN;
  constexpr int blockM = threadsM * TM;
  constexpr int blockN = threadsN * TN;

  float result[B_SUB][TM];
  float result_up[SWIGLU ? B_SUB : 1][SWIGLU ? TM : 1];
#pragma unroll
  for (int bi = 0; bi < B_SUB; bi++) {
#pragma unroll
    for (int tm = 0; tm < TM; tm++) {
      result[bi][tm] = 0.0f;
      if constexpr (SWIGLU) result_up[bi][tm] = 0.0f;
    }
  }

  const int thrM = SN != 32 ? int(simd_lid) / SN : 0;
  const int thrN = SN != 32 ? int(simd_lid) % SN : int(simd_lid);
  const int sgN = BN != 1 ? int(simd_gid % BN) : 0;
  const int b_sg = (BN == 1 && BSG > 1) ? int(simd_gid % BSG) : 0;
  const int b0 = b_sg * B_SUB;
  const int simdM = (BN != 1) ? SM * int(simd_gid / BN) : (BSG > 1 ? SM * int(simd_gid / BSG) : int(SM * simd_gid));
  const int simdN = BN != 1 ? SN * int(simd_gid % BN) : 0;
  int bm = (simdM + thrM) * TM;
  int bn = (simdN + thrN) * TN;

  int out_row = int(tgx) * blockM + bm;
  if (out_row >= N) return;
  out_row = out_row + TM <= N ? out_row : N - TM;
  mat += size_t(out_row) * KK;
  if constexpr (SWIGLU) mat_up += size_t(out_row) * KK;

  constexpr int n_iter = KK / blockN;
  constexpr int last_iter = blockN * n_iter;
  constexpr int leftover = KK - last_iter;

  for (int i = 0; i < n_iter; ++i) {
    WT inter[TM][TN];
    WT inter_up[SWIGLU ? TM : 1][SWIGLU ? TN : 1];
#pragma unroll
    for (int tm = 0; tm < TM; tm++) {
      const size_t mo = size_t(tm) * KK + bn;
      if constexpr (TN == 4 && sizeof(WT) == 2) {
        Bf4 pk = *(const device Bf4*)(mat + mo);
#pragma unroll
        for (int tn = 0; tn < 4; tn++) inter[tm][tn] = as_type<bfloat16_t>(pk.v[tn]);
      } else {
#pragma unroll
        for (int tn = 0; tn < TN; tn++) inter[tm][tn] = mat[mo + tn];
      }
      if constexpr (SWIGLU) {
        if constexpr (TN == 4 && sizeof(WT) == 2) {
          Bf4 pk = *(const device Bf4*)(mat_up + mo);
#pragma unroll
          for (int tn = 0; tn < 4; tn++) inter_up[tm][tn] = as_type<bfloat16_t>(pk.v[tn]);
        } else {
#pragma unroll
          for (int tn = 0; tn < TN; tn++) inter_up[tm][tn] = mat_up[mo + tn];
        }
      }
    }
#pragma unroll
    for (int bi = 0; bi < B_SUB; bi++) {
      const int b = b0 + bi;
      if (b < B) {
        float vc[TN];
        if constexpr (TN == 4 && sizeof(XT) == 2) {
          Bf4 px = *(const device Bf4*)(in_vec + size_t(b) * KK + bn);
#pragma unroll
          for (int tn = 0; tn < 4; tn++) vc[tn] = static_cast<float>(as_type<bfloat16_t>(px.v[tn]));
        } else {
#pragma unroll
          for (int tn = 0; tn < TN; tn++) vc[tn] = static_cast<float>(in_vec[size_t(b) * KK + bn + tn]);
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
    for (int bi = 0; bi < B_SUB; bi++) {
      const int b = b0 + bi;
      if (b < B) {
        float vc[TN];
#pragma unroll
        for (int tn = 0; tn < TN; tn++) {
          vc[tn] = (bn + tn < KK) ? static_cast<float>(in_vec[size_t(b) * KK + bn + tn]) : 0.0f;
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
  }

#pragma unroll
  for (int bi = 0; bi < B_SUB; bi++) {
#pragma unroll
    for (int tm = 0; tm < TM; tm++) {
#pragma unroll
      for (ushort sn = (SN / 2); sn >= 1; sn >>= 1) {
        result[bi][tm] += simd_shuffle_down(result[bi][tm], sn);
        if constexpr (SWIGLU) result_up[bi][tm] += simd_shuffle_down(result_up[bi][tm], sn);
      }
    }
  }

  if (BN > 1) {
    for (int bi = 0; bi < B_SUB; bi++) {
      const int b = b0 + bi;
      if (b < B) {
        threadgroup float* tgp_results = tgp_memory + (b * BN + sgN) * (blockM + TM) + bm;
        if (thrN == 0) {
          for (int tm = 0; tm < TM; tm++) tgp_results[tm] = result[bi][tm];
          threadgroup_barrier(mem_flags::mem_none);
          if (sgN == 0) {
            for (int sgn = 1; sgn < BN; sgn++) {
              for (int tm = 0; tm < TM; tm++) {
                result[bi][tm] += tgp_results[sgn * (blockM + TM) + tm];
              }
            }
          }
        }
        if constexpr (SWIGLU) {
          threadgroup float* tgp_up = tgp_memory + (B * BN + b * BN + sgN) * (blockM + TM) + bm;
          if (thrN == 0) {
            for (int tm = 0; tm < TM; tm++) tgp_up[tm] = result_up[bi][tm];
            threadgroup_barrier(mem_flags::mem_none);
            if (sgN == 0) {
              for (int sgn = 1; sgn < BN; sgn++) {
                for (int tm = 0; tm < TM; tm++) {
                  result_up[bi][tm] += tgp_up[sgn * (blockM + TM) + tm];
                }
              }
            }
          }
        }
      }
    }
  }

  if (simdN == 0 && thrN == 0) {
    for (int bi = 0; bi < B_SUB; bi++) {
      const int b = b0 + bi;
      if (b < B) {
        for (int tm = 0; tm < TM; tm++) {
          if constexpr (SWIGLU) {
            bfloat16_t g = static_cast<bfloat16_t>(result[bi][tm]);
            bfloat16_t u = static_cast<bfloat16_t>(result_up[bi][tm]);
            out_vec[size_t(b) * N + out_row + tm] = glue_swiglu_bf16(g, u);
          } else if constexpr (RES) {
            out_vec[size_t(b) * N + out_row + tm] =
                static_cast<T>(result[bi][tm]) + res_vec[size_t(b) * N + out_row + tm];
          } else {
            out_vec[size_t(b) * N + out_row + tm] = static_cast<T>(result[bi][tm]);
          }
        }
      }
    }
  }
}
"""
)


def _cfg(N: int, K: int, B: int = 1, is_swiglu: bool = False):
    """Exact block config MLX 0.32.4 picks for M == 1, plus BSG splitting for B >= 13 to avoid spills."""
    tm, tn, sm, sn, bm, bn, bsg = 4, 4, 1, 32, 1, 1, 1
    bm = (4 if B >= 14 else _GEMV_BM_LARGE) if N >= 4096 else 4
    sn = 32
    if K <= 64:
        bm, sm, sn = 1, 8, 4
    elif K >= 16 * N:
        bm, bn = 1, 8
    if is_swiglu:
        if B >= 5:
            tm = 1
    else:
        if 5 <= B <= 12:
            tm = 2
        elif B >= 13:
            if bn == 1 and sm == 1 and sn == 32:
                bsg = 2
                tm = 2
            else:
                tm = 1
    tm = 1 if N < tm else tm
    return bm, bn, sm, sn, tm, tn, bsg


def _tname(dt):
    return {mx.bfloat16: "bfloat16_t", mx.float32: "float", mx.float16: "half"}[dt]


def _build_group(out_dt, w_dt, x_dt, B, G, Ns, K, cfg, has_res=False, is_swiglu=False):
    key = (out_dt, w_dt, x_dt, B, G, tuple(Ns), K, cfg, has_res, is_swiglu)
    if key in _KERNELS:
        return _KERNELS[key]
    bm, bn, sm, sn, tm, tn, bsg = cfg
    b_sub = (B + bsg - 1) // bsg
    blockM = bm * sm * tm
    tg_floats = (2 if is_swiglu else 1) * B * bn * (blockM + tm) if bn > 1 else 1
    T, WT, XT = _tname(out_dt), _tname(w_dt), _tname(x_dt)
    lines = [
        f"threadgroup float tgp_mem[{tg_floats}];",
        "const uint t = threadgroup_position_in_grid.x;",
    ]
    if G == 1:
        N0 = Ns[0]
        nt = (N0 + blockM - 1) // blockM
        wup = "u0" if is_swiglu else "w0"
        rv = "res" if has_res else "out0"
        lines.append(
            f"bailing_gemv_rows_run<{T}, {WT}, {XT}, {B}, {bsg}, {b_sub}, {bm}, {bn}, {sm}, {sn}, {tm}, {tn}, {int(has_res)}, {int(is_swiglu)}, {K}>"
            f"(w0, {wup}, x0, {rv}, out0, {N0}, tgp_mem, t, "
            f"simdgroup_index_in_threadgroup, thread_index_in_simdgroup);"
        )
        start = nt
    else:
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
            f"bailing_gemv_rows_run<{T}, {WT}, {XT}, {B}, {bsg}, {b_sub}, {bm}, {bn}, {sm}, {sn}, {tm}, {tn}, {int(has_res)}, {int(is_swiglu)}, {K}>"
            f"(wp, up_p, xp, rp, op, n_cur, tgp_mem, t_loc, "
            f"simdgroup_index_in_threadgroup, thread_index_in_simdgroup);"
        )
    in_names = [f"w{g}" for g in range(G)]
    if is_swiglu:
        in_names += [f"u{g}" for g in range(G)]
    in_names += [f"x{g}" for g in range(G)]
    if has_res:
        in_names.append("res")
    out_names = [f"out{g}" for g in range(G)]
    tag = f"{out_dt}_{w_dt}_{x_dt}_b{B}_g{G}_k{K}_r{int(has_res)}_s{int(is_swiglu)}_{'_'.join(map(str, Ns + list(cfg)))}"
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


def gemv_rows_group(xs, ws, *, cfg=None):
    """Independent problems out_i = xs[i] @ ws[i].T; xs[i] (B, K), ws[i] (N_i, K) same dtype & K."""
    xs, ws = list(xs), list(ws)
    B, K = xs[0].shape
    dt = ws[0].dtype
    assert 1 <= B <= MAX_B
    assert all(x.shape == (B, K) and x.dtype == dt for x in xs)
    assert all(w.ndim == 2 and w.shape[1] == K and w.dtype == dt for w in ws)
    if cfg is None:
        cfgs = [_cfg(w.shape[0], K, B, False) for w in ws]
        if len(set(cfgs)) > 1:
            out = [None] * len(ws)
            for c in dict.fromkeys(cfgs):
                idx = [i for i, cc in enumerate(cfgs) if cc == c]
                for i, o in zip(idx, gemv_rows_group([xs[i] for i in idx], [ws[i] for i in idx], cfg=c)):
                    out[i] = o
            return out
        cfg = cfgs[0]
    Ns = [w.shape[0] for w in ws]
    G = len(ws)
    kern, ntg = _build_group(dt, dt, dt, B, G, Ns, K, cfg)
    bm, bn, sm, sn, tm, tn, bsg = cfg
    y_dim = bn * bsg
    return list(
        kern(
            inputs=list(ws) + list(xs),
            grid=(ntg * 32, y_dim, bm),
            threadgroup=(32, y_dim, bm),
            output_shapes=[(B, n) for n in Ns],
            output_dtypes=[dt] * G,
        )
    )


def gemv_rows(x, w, *, cfg=None):
    return gemv_rows_group([x], [w], cfg=cfg)[0]


def gemv_rows_multi(x, ws):
    return gemv_rows_group([x] * len(ws), list(ws))


def gemv_rows_res(x, w, res):
    """Exact gemv_rows(x, w) + res for (B, K) bf16, (N, K) bf16, (B, N) bf16."""
    B, K = x.shape
    N = w.shape[0]
    dt = w.dtype
    assert 1 <= B <= MAX_B and x.dtype == dt and res.shape == (B, N) and res.dtype == dt and w.shape[1] == K
    cfg = _cfg(N, K, B, False)
    kern, ntg = _build_group(dt, dt, dt, B, 1, [N], K, cfg, has_res=True)
    bm, bn, sm, sn, tm, tn, bsg = cfg
    y_dim = bn * bsg
    return kern(
        inputs=[w, x, res],
        grid=(ntg * 32, y_dim, bm),
        threadgroup=(32, y_dim, bm),
        output_shapes=[(B, N)],
        output_dtypes=[dt],
    )[0]


def gemv_swiglu_rows(x, w_gate, w_up):
    """Exact swiglu(x @ w_gate.T, x @ w_up.T) in one launch for (B, K) bf16."""
    B, K = x.shape
    N = w_gate.shape[0]
    assert 1 <= B <= MAX_B and x.dtype == w_gate.dtype == w_up.dtype == mx.bfloat16
    assert w_up.shape == (N, K) and w_gate.shape[1] == K
    cfg = _cfg(N, K, B, True)
    kern, ntg = _build_group(mx.bfloat16, mx.bfloat16, mx.bfloat16, B, 1, [N], K, cfg, is_swiglu=True)
    bm, bn, sm, sn, tm, tn, bsg = cfg
    y_dim = bn * bsg
    return kern(
        inputs=[w_gate, w_up, x],
        grid=(ntg * 32, y_dim, bm),
        threadgroup=(32, y_dim, bm),
        output_shapes=[(B, N)],
        output_dtypes=[mx.bfloat16],
    )[0]


def router_gemv_rows(x_bf16, w_bf16):
    """Exact x.astype(float32) @ w.astype(float32).T reading bf16 weights directly."""
    B, K = x_bf16.shape
    N = w_bf16.shape[0]
    assert 1 <= B <= MAX_B and x_bf16.dtype == mx.bfloat16 and w_bf16.dtype == mx.bfloat16 and w_bf16.shape[1] == K
    cfg = _cfg(N, K, B, False)
    kern, ntg = _build_group(mx.float32, mx.bfloat16, mx.bfloat16, B, 1, [N], K, cfg)
    bm, bn, sm, sn, tm, tn, bsg = cfg
    y_dim = bn * bsg
    return kern(
        inputs=[w_bf16, x_bf16],
        grid=(ntg * 32, y_dim, bm),
        threadgroup=(32, y_dim, bm),
        output_shapes=[(B, N)],
        output_dtypes=[mx.float32],
    )[0]
