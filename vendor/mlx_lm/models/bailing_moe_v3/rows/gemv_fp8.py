"""Bit-exact native F8_E4M3 (128x128 F32 block scale_inv) GEMV and dequant kernels for BailingMoeV3.

Every element decode reproduces:
    static_cast<bfloat16_t>(E4M3_LUT[u8] * scale_inv[n // 128, k // 128])
and accumulates in exact MLX GEMV K-reduction order (SN=32, TN=4 -> blockN=128) with
0 bit diffs against bf16 gemv_rows / switchless dequantized bf16 weights across B = 1..24.
"""
import os

import mlx.core as mx
import numpy as np

from .gemv import MAX_B, gemv_rows, gemv_rows_group, gemv_rows_res, gemv_swiglu_rows
from .glue import GLUE_HEADER

_KERNELS = {}


def e4m3fn_table_np() -> np.ndarray:
    """IEEE/OCP F8_E4M3FN lookup table matching tools/convert.py bit-for-bit."""
    lut = np.zeros(256, dtype=np.float32)
    for i in range(256):
        s = -1.0 if (i & 0x80) else 1.0
        e = (i >> 3) & 0xF
        m = i & 7
        if e == 0xF and m == 7:
            v = np.nan
        elif e == 0:
            v = m / 8.0 * (2.0**-6)
        else:
            v = (1.0 + m / 8.0) * (2.0 ** (e - 7))
        lut[i] = np.float32(s * v)
    return lut


def _e4m3_lut_metal() -> str:
    lut = e4m3fn_table_np()
    items = []
    for i in range(256):
        v = float(lut[i])
        if np.isnan(v):
            items.append("as_type<float>(0x7fc00000u)")
        else:
            items.append(v.hex() + "f")
    rows = [", ".join(items[j : j + 8]) for j in range(0, 256, 8)]
    return "constant float E4M3_LUT[256] = {\n  " + ",\n  ".join(rows) + "\n};\n"


E4M3_LUT_METAL = _e4m3_lut_metal()

_HEADER = (
    GLUE_HEADER
    + "\n"
    + E4M3_LUT_METAL
    + r"""
struct Bf4 { ushort v[4]; };

template <typename T, typename XT, int B, int BSG, int B_SUB, int BM_M, int TM, int RES, int SWIGLU, int COOP, int KK>
METAL_FUNC void bailing_gemv_fp8_rows_run(
    const device uchar* mat,
    const device float* sc_mat,
    const device uchar* mat_up,
    const device float* sc_up,
    const device XT* in_vec,   // (B, KK) row-major
    const device T* res_vec,   // (B, N) row-major when RES == 1
    device T* out_vec,         // (B, N) row-major
    const int N,
    threadgroup bfloat16_t* w_tg,
    uint tgx,
    uint simd_gid,
    uint simd_lid) {
  constexpr int TN = 4;
  constexpr int SN = 32;
  constexpr int blockN = SN * TN;  // 128 == FP8 scale block size!
  constexpr int blockM = BM_M * TM;
  constexpr int k_blocks = KK >> 7;
  constexpr int TOT_R = SWIGLU ? (2 * TM) : TM;

  const int m_sg = int(simd_gid) / BSG;
  const int b_sg = int(simd_gid) % BSG;
  const int b0 = b_sg * B_SUB;

  int out_row = int(tgx) * blockM + m_sg * TM;
  const bool row_valid = (out_row < N);
  out_row = row_valid ? (out_row + TM <= N ? out_row : N - TM) : 0;

  // 128 % TM == 0 and out_row is a multiple of TM (when N % TM == 0), so all TM rows share one scale row.
  const int row_blk = out_row >> 7;
  sc_mat += size_t(row_blk) * k_blocks;
  if constexpr (SWIGLU) sc_up += size_t(row_blk) * k_blocks;

  mat += size_t(out_row) * KK;
  if constexpr (SWIGLU) mat_up += size_t(out_row) * KK;

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

  int bn = int(simd_lid) * TN;
  for (int i = 0; i < k_blocks; ++i) {
    const float sc = sc_mat[i];
    const float sc_u = SWIGLU ? sc_up[i] : 0.0f;
    float wf[TM][4];
    float wf_u[SWIGLU ? TM : 1][SWIGLU ? 4 : 1];
    if constexpr (COOP) {
      threadgroup bfloat16_t* my_w = w_tg + (size_t(i & 1) * BM_M + m_sg) * (TOT_R * 128);
      for (int ridx = b_sg; ridx < TOT_R; ridx += BSG) {
        const bool is_u = SWIGLU && (ridx >= TM);
        const int tm = is_u ? (ridx - TM) : ridx;
        const device uchar* mp = is_u ? mat_up : mat;
        const float sc_r = is_u ? sc_u : sc;
        const uchar4 pk = *(const device uchar4*)(mp + size_t(tm) * KK + bn);
        const int off = ridx * 128 + int(simd_lid) * 4;
        my_w[off + 0] = static_cast<bfloat16_t>(E4M3_LUT[pk.x] * sc_r);
        my_w[off + 1] = static_cast<bfloat16_t>(E4M3_LUT[pk.y] * sc_r);
        my_w[off + 2] = static_cast<bfloat16_t>(E4M3_LUT[pk.z] * sc_r);
        my_w[off + 3] = static_cast<bfloat16_t>(E4M3_LUT[pk.w] * sc_r);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
      for (int tm = 0; tm < TM; tm++) {
        const int off = tm * 128 + int(simd_lid) * 4;
        wf[tm][0] = static_cast<float>(my_w[off + 0]);
        wf[tm][1] = static_cast<float>(my_w[off + 1]);
        wf[tm][2] = static_cast<float>(my_w[off + 2]);
        wf[tm][3] = static_cast<float>(my_w[off + 3]);
        if constexpr (SWIGLU) {
          const int off_u = (TM + tm) * 128 + int(simd_lid) * 4;
          wf_u[tm][0] = static_cast<float>(my_w[off_u + 0]);
          wf_u[tm][1] = static_cast<float>(my_w[off_u + 1]);
          wf_u[tm][2] = static_cast<float>(my_w[off_u + 2]);
          wf_u[tm][3] = static_cast<float>(my_w[off_u + 3]);
        }
      }
    } else {
#pragma unroll
      for (int tm = 0; tm < TM; tm++) {
        const size_t mo = size_t(tm) * KK + bn;
        uchar4 pk = *(const device uchar4*)(mat + mo);
        wf[tm][0] = static_cast<float>(static_cast<bfloat16_t>(E4M3_LUT[pk.x] * sc));
        wf[tm][1] = static_cast<float>(static_cast<bfloat16_t>(E4M3_LUT[pk.y] * sc));
        wf[tm][2] = static_cast<float>(static_cast<bfloat16_t>(E4M3_LUT[pk.z] * sc));
        wf[tm][3] = static_cast<float>(static_cast<bfloat16_t>(E4M3_LUT[pk.w] * sc));
        if constexpr (SWIGLU) {
          uchar4 pku = *(const device uchar4*)(mat_up + mo);
          wf_u[tm][0] = static_cast<float>(static_cast<bfloat16_t>(E4M3_LUT[pku.x] * sc_u));
          wf_u[tm][1] = static_cast<float>(static_cast<bfloat16_t>(E4M3_LUT[pku.y] * sc_u));
          wf_u[tm][2] = static_cast<float>(static_cast<bfloat16_t>(E4M3_LUT[pku.z] * sc_u));
          wf_u[tm][3] = static_cast<float>(static_cast<bfloat16_t>(E4M3_LUT[pku.w] * sc_u));
        }
      }
    }
#pragma unroll
    for (int bi = 0; bi < B_SUB; bi++) {
      const int b = b0 + bi;
      if (b < B) {
        float vc[4];
        if constexpr (sizeof(XT) == 2) {
          Bf4 px = *(const device Bf4*)(in_vec + size_t(b) * KK + bn);
#pragma unroll
          for (int tn = 0; tn < 4; tn++) vc[tn] = static_cast<float>(as_type<bfloat16_t>(px.v[tn]));
        } else {
#pragma unroll
          for (int tn = 0; tn < 4; tn++) vc[tn] = static_cast<float>(in_vec[size_t(b) * KK + bn + tn]);
        }
#pragma unroll
        for (int tm = 0; tm < TM; tm++) {
#pragma unroll
          for (int tn = 0; tn < 4; tn++) {
            result[bi][tm] += wf[tm][tn] * vc[tn];
            if constexpr (SWIGLU) {
              result_up[bi][tm] += wf_u[tm][tn] * vc[tn];
            }
          }
        }
      }
    }
    bn += blockN;
  }

#pragma unroll
  for (int bi = 0; bi < B_SUB; bi++) {
#pragma unroll
    for (int tm = 0; tm < TM; tm++) {
#pragma unroll
      for (ushort sn = 16; sn >= 1; sn >>= 1) {
        result[bi][tm] += simd_shuffle_down(result[bi][tm], sn);
        if constexpr (SWIGLU) result_up[bi][tm] += simd_shuffle_down(result_up[bi][tm], sn);
      }
    }
  }

  if (row_valid && simd_lid == 0) {
#pragma unroll
    for (int bi = 0; bi < B_SUB; bi++) {
      const int b = b0 + bi;
      if (b < B) {
#pragma unroll
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

_DEQUANT_2D_SRC = r"""
    const uint col4 = thread_position_in_grid.x;
    const uint row = thread_position_in_grid.y;
    const uint k = col4 * 4;
    if (int(row) >= NN || int(k) >= KK) return;
    const uint sk = (KK + 127) >> 7;
    const float sc = scale_inv[(row >> 7) * sk + (k >> 7)];
    const uchar4 pk = *(const device uchar4*)(w_fp8 + size_t(row) * KK + k);
    Bf4 out_pk;
    out_pk.v[0] = as_type<ushort>(static_cast<bfloat16_t>(E4M3_LUT[pk.x] * sc));
    out_pk.v[1] = as_type<ushort>(static_cast<bfloat16_t>(E4M3_LUT[pk.y] * sc));
    out_pk.v[2] = as_type<ushort>(static_cast<bfloat16_t>(E4M3_LUT[pk.z] * sc));
    out_pk.v[3] = as_type<ushort>(static_cast<bfloat16_t>(E4M3_LUT[pk.w] * sc));
    *(device Bf4*)(out + size_t(row) * KK + k) = out_pk;
"""

_DEQUANT_ML_SRC = r"""
    const uint col4 = thread_position_in_grid.x;
    const uint row = thread_position_in_grid.y;
    const uint h = thread_position_in_grid.z;
    const uint k = col4 * 4;
    if (int(h) >= HH || int(row) >= NN || int(k) >= KK) return;
    const float sc = SCALE_BY_N ? scale_inv[h * 4 + (row >> 7)] : scale_inv[h * 4 + (k >> 7)];
    const size_t off = (size_t(h) * NN + row) * KK + k;
    const uchar4 pk = *(const device uchar4*)(w_fp8 + off);
    Bf4 out_pk;
    out_pk.v[0] = as_type<ushort>(static_cast<bfloat16_t>(E4M3_LUT[pk.x] * sc));
    out_pk.v[1] = as_type<ushort>(static_cast<bfloat16_t>(E4M3_LUT[pk.y] * sc));
    out_pk.v[2] = as_type<ushort>(static_cast<bfloat16_t>(E4M3_LUT[pk.z] * sc));
    out_pk.v[3] = as_type<ushort>(static_cast<bfloat16_t>(E4M3_LUT[pk.w] * sc));
    *(device Bf4*)(out + off) = out_pk;
"""


def _dq2_kernel():
    k = _KERNELS.get("dq2")
    if k is None:
        k = _KERNELS["dq2"] = mx.fast.metal_kernel(
            name="bv3_dequant_fp8_2d",
            input_names=["w_fp8", "scale_inv"],
            output_names=["out"],
            header=_HEADER,
            source=_DEQUANT_2D_SRC,
            ensure_row_contiguous=True,
        )
    return k


def _dqml_kernel():
    k = _KERNELS.get("dqml")
    if k is None:
        k = _KERNELS["dqml"] = mx.fast.metal_kernel(
            name="bv3_dequant_fp8_ml",
            input_names=["w_fp8", "scale_inv"],
            output_names=["out"],
            header=_HEADER,
            source=_DEQUANT_ML_SRC,
            ensure_row_contiguous=True,
        )
    return k


def dequant_fp8_to_bf16(w_fp8: mx.array, scale_inv: mx.array) -> mx.array:
    """Exact FP8 E4M3 [N, K] * 128x128 F32 scale_inv -> bfloat16 [N, K]."""
    N, K = w_fp8.shape
    assert w_fp8.dtype == mx.uint8 and scale_inv.dtype == mx.float32 and K % 4 == 0
    (out,) = _dq2_kernel()(
        inputs=[w_fp8, scale_inv],
        template=[("NN", N), ("KK", K)],
        grid=(K // 4, N, 1),
        threadgroup=(64, 4, 1),
        output_shapes=[(N, K)],
        output_dtypes=[mx.bfloat16],
    )
    return out


def dequant_fp8_multilinear_to_bf16(w_fp8: mx.array, scale_inv: mx.array, scale_by_n: bool) -> mx.array:
    """Exact FP8 E4M3 [H, N, K] * [H, 4] F32 scale_inv -> bfloat16 [H, N, K]."""
    H, N, K = w_fp8.shape
    assert w_fp8.dtype == mx.uint8 and scale_inv.dtype == mx.float32 and K % 4 == 0
    (out,) = _dqml_kernel()(
        inputs=[w_fp8, scale_inv],
        template=[("HH", H), ("NN", N), ("KK", K), ("SCALE_BY_N", int(bool(scale_by_n)))],
        grid=(K // 4, N, H),
        threadgroup=(32, 8, 1),
        output_shapes=[(H, N, K)],
        output_dtypes=[mx.bfloat16],
    )
    return out


def _fp8_cfg(N: int, K: int, B: int, is_swiglu: bool = False):
    """Empirically tuned `(bsg, b_sub, bm_m, tm)` per B regime on M3 Ultra."""
    env = os.environ.get("LINGSANTE_FP8_CFG")
    if env:
        bsg, bm_m, tm = (int(v) for v in env.split(","))
        b_sub = (B + bsg - 1) // bsg
        if N < tm:
            tm = 1
        return bsg, b_sub, bm_m, tm

    if is_swiglu:
        if B <= 2:
            bsg, bm_m, tm = 1, 4, 4
        elif B <= 4:
            bsg, bm_m, tm = 1, 4, 2
        elif B <= 8:
            bsg, bm_m, tm = 1, 4, 1
        elif B == 9:
            bsg, bm_m, tm = 2, 4, 1
        elif B <= 12:
            bsg, bm_m, tm = 2, 2, 1
        elif B <= 16:
            bsg, bm_m, tm = 2, 4, 1
        elif B <= 20:
            bsg, bm_m, tm = 3, 2, 1
        else:
            bsg, bm_m, tm = 4, 2, 1
    else:
        bm_large = 8 if N >= 4096 else 4
        if B <= 4:
            bsg, bm_m, tm = 1, bm_large, 4
        elif B <= 8:
            bsg, bm_m, tm = 1, bm_large, 2
        elif B <= 10:
            bsg, bm_m, tm = 2, 2, 2
        elif B <= 16:
            bsg, bm_m, tm = 2, (2 if K > N else 4), 2
        else:
            bsg, bm_m, tm = (3 if K > N else 4), 2, 2
    if N < tm:
        tm = 1
    b_sub = (B + bsg - 1) // bsg
    return bsg, b_sub, bm_m, tm


def _tname(dt):
    return {mx.bfloat16: "bfloat16_t", mx.float32: "float", mx.float16: "half"}[dt]


def _can_coop(bsg: int, bm_m: int, tm: int, is_swiglu: bool = False) -> bool:
    tot_r = (2 * tm) if is_swiglu else tm
    return bsg > 1 and (tot_r % bsg == 0) and (bm_m * tot_r * 512 <= 16384)


def _resolve_coop(cfg, is_swiglu: bool, coop: bool | None) -> bool:
    bsg, _, bm_m, tm = cfg
    if coop is not None:
        use_c = bool(coop)
        assert not use_c or _can_coop(bsg, bm_m, tm, is_swiglu), (
            f"COOP unsupported for cfg={cfg}, is_swiglu={is_swiglu}"
        )
        return use_c
    return False


def _build_fp8_group(out_dt, x_dt, B, G, Ns, K, cfg, has_res=False, is_swiglu=False, coop=False):
    use_coop = int(bool(coop))
    key = ("fp8", out_dt, x_dt, B, G, tuple(Ns), K, cfg, has_res, is_swiglu, use_coop)
    if key in _KERNELS:
        return _KERNELS[key]
    bsg, b_sub, bm_m, tm = cfg
    blockM = bm_m * tm
    T, XT = _tname(out_dt), _tname(x_dt)
    tot_r = (2 * tm) if is_swiglu else tm
    tg_elems = (2 * bm_m * tot_r * 128) if use_coop else 1
    lines = [
        f"threadgroup bfloat16_t w_tg[{tg_elems}];",
        "const uint t = threadgroup_position_in_grid.x;",
    ]
    if G == 1:
        N0 = Ns[0]
        nt = (N0 + blockM - 1) // blockM
        wup = "u0" if is_swiglu else "w0"
        sup = "su0" if is_swiglu else "s0"
        rv = "res" if has_res else "out0"
        lines.append(
            f"bailing_gemv_fp8_rows_run<{T}, {XT}, {B}, {bsg}, {b_sub}, {bm_m}, {tm}, {int(has_res)}, {int(is_swiglu)}, {use_coop}, {K}>"
            f"(w0, s0, {wup}, {sup}, x0, {rv}, out0, {N0}, w_tg, t, "
            f"simdgroup_index_in_threadgroup, thread_index_in_simdgroup);"
        )
        start = nt
    else:
        lines.extend([
            "const device uchar* wp = w0;",
            "const device float* sp = s0;",
            "const device uchar* up_p = w0;",
            "const device float* sup_p = s0;",
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
            wup_s = f"up_p = u{g}; sup_p = su{g}; " if is_swiglu else ""
            rv_s = "rp = res; " if has_res else f"rp = out{g}; "
            if g < len(Ns) - 1:
                lines.append(
                    f"{kw} (t < {start + nt}) {{ wp = w{g}; sp = s{g}; {wup_s}xp = x{g}; {rv_s}op = out{g}; n_cur = {N}; t_loc = t - {start}; }}"
                )
            else:
                lines.append(
                    f"else {{ wp = w{g}; sp = s{g}; {wup_s}xp = x{g}; {rv_s}op = out{g}; n_cur = {N}; t_loc = t - {start}; }}"
                )
            start += nt
        lines.append(
            f"bailing_gemv_fp8_rows_run<{T}, {XT}, {B}, {bsg}, {b_sub}, {bm_m}, {tm}, {int(has_res)}, {int(is_swiglu)}, {use_coop}, {K}>"
            f"(wp, sp, up_p, sup_p, xp, rp, op, n_cur, w_tg, t_loc, "
            f"simdgroup_index_in_threadgroup, thread_index_in_simdgroup);"
        )
    in_names = [f"w{g}" for g in range(G)] + [f"s{g}" for g in range(G)]
    if is_swiglu:
        in_names += [f"u{g}" for g in range(G)] + [f"su{g}" for g in range(G)]
    in_names += [f"x{g}" for g in range(G)]
    if has_res:
        in_names.append("res")
    out_names = [f"out{g}" for g in range(G)]
    tag = f"{out_dt}_{x_dt}_b{B}_g{G}_k{K}_r{int(has_res)}_s{int(is_swiglu)}_c{use_coop}_{'_'.join(map(str, Ns + list(cfg)))}"
    tag = tag.replace("mlx.core.", "")
    kern = mx.fast.metal_kernel(
        name=f"bv3_gemv_fp8_{tag}",
        input_names=in_names,
        output_names=out_names,
        header=_HEADER,
        source="\n".join(lines),
        ensure_row_contiguous=True,
    )
    _KERNELS[key] = (kern, start)
    return _KERNELS[key]


def gemv_fp8_rows_group(xs, ws_fp8, scs, *, cfg=None, coop=None):
    """Independent problems out_i = xs[i] @ bf16(fp8_i * sc_i).T."""
    xs, ws_fp8, scs = list(xs), list(ws_fp8), list(scs)
    B, K = xs[0].shape
    dt = xs[0].dtype
    assert 1 <= B <= MAX_B and K % 128 == 0
    assert all(x.shape == (B, K) and x.dtype == dt for x in xs)
    assert all(w.ndim == 2 and w.shape[1] == K and w.dtype == mx.uint8 for w in ws_fp8)
    assert all(s.dtype == mx.float32 for s in scs)
    if cfg is None:
        cfgs = [_fp8_cfg(w.shape[0], K, B, False) for w in ws_fp8]
        if len(set(cfgs)) > 1:
            out = [None] * len(ws_fp8)
            for c in dict.fromkeys(cfgs):
                idx = [i for i, cc in enumerate(cfgs) if cc == c]
                for i, o in zip(
                    idx,
                    gemv_fp8_rows_group(
                        [xs[i] for i in idx], [ws_fp8[i] for i in idx], [scs[i] for i in idx], cfg=c, coop=coop
                    ),
                ):
                    out[i] = o
            return out
        cfg = cfgs[0]
    bsg, _, bm_m, tm = cfg
    use_coop = _resolve_coop(cfg, False, coop)
    Ns = [w.shape[0] for w in ws_fp8]
    assert all(n % tm == 0 for n in Ns), f"Ns={Ns} not divisible by tm={tm}"
    G = len(ws_fp8)
    kern, ntg = _build_fp8_group(dt, dt, B, G, Ns, K, cfg, coop=use_coop)
    nsg = bsg * bm_m
    return list(
        kern(
            inputs=ws_fp8 + scs + xs,
            grid=(ntg * 32, nsg, 1),
            threadgroup=(32, nsg, 1),
            output_shapes=[(B, n) for n in Ns],
            output_dtypes=[dt] * G,
        )
    )


def gemv_fp8_rows(x, w_fp8, sc, *, cfg=None, coop=None):
    return gemv_fp8_rows_group([x], [w_fp8], [sc], cfg=cfg, coop=coop)[0]


def gemv_fp8_rows_res(x, w_fp8, sc, res, *, cfg=None, coop=None):
    B, K = x.shape
    N = w_fp8.shape[0]
    dt = x.dtype
    assert 1 <= B <= MAX_B and K % 128 == 0 and res.shape == (B, N) and res.dtype == dt
    assert w_fp8.dtype == mx.uint8 and sc.dtype == mx.float32 and w_fp8.shape[1] == K
    if cfg is None:
        cfg = _fp8_cfg(N, K, B, False)
    bsg, _, bm_m, tm = cfg
    use_coop = _resolve_coop(cfg, False, coop)
    assert N % tm == 0
    kern, ntg = _build_fp8_group(dt, dt, B, 1, [N], K, cfg, has_res=True, coop=use_coop)
    nsg = bsg * bm_m
    return kern(
        inputs=[w_fp8, sc, x, res],
        grid=(ntg * 32, nsg, 1),
        threadgroup=(32, nsg, 1),
        output_shapes=[(B, N)],
        output_dtypes=[dt],
    )[0]


def gemv_fp8_swiglu_rows(x, wg_fp8, sc_g, wu_fp8, sc_u, *, cfg=None, coop=None):
    B, K = x.shape
    N = wg_fp8.shape[0]
    assert 1 <= B <= MAX_B and K % 128 == 0 and x.dtype == mx.bfloat16
    assert wg_fp8.dtype == wu_fp8.dtype == mx.uint8 and sc_g.dtype == sc_u.dtype == mx.float32
    assert wg_fp8.shape == wu_fp8.shape == (N, K)
    if cfg is None:
        cfg = _fp8_cfg(N, K, B, True)
    bsg, _, bm_m, tm = cfg
    use_coop = _resolve_coop(cfg, True, coop)
    assert N % tm == 0
    kern, ntg = _build_fp8_group(mx.bfloat16, mx.bfloat16, B, 1, [N], K, cfg, is_swiglu=True, coop=use_coop)
    nsg = bsg * bm_m
    return kern(
        inputs=[wg_fp8, sc_g, wu_fp8, sc_u, x],
        grid=(ntg * 32, nsg, 1),
        threadgroup=(32, nsg, 1),
        output_shapes=[(B, N)],
        output_dtypes=[mx.bfloat16],
    )[0]


def _is_fp8_mod(mod) -> bool:
    return hasattr(mod, "weight_fp8")


def gemv_lin(x: mx.array, mod) -> mx.array:
    if _is_fp8_mod(mod):
        return gemv_fp8_rows(x, mod.weight_fp8, mod.weight_scale_inv)
    return gemv_rows(x, mod.weight)


def gemv_lin_res(x: mx.array, mod, res: mx.array) -> mx.array:
    if _is_fp8_mod(mod):
        return gemv_fp8_rows_res(x, mod.weight_fp8, mod.weight_scale_inv, res)
    return gemv_rows_res(x, mod.weight, res)


def gemv_lin_swiglu(x: mx.array, gate_mod, up_mod) -> mx.array:
    if _is_fp8_mod(gate_mod) and _is_fp8_mod(up_mod):
        return gemv_fp8_swiglu_rows(
            x,
            gate_mod.weight_fp8,
            gate_mod.weight_scale_inv,
            up_mod.weight_fp8,
            up_mod.weight_scale_inv,
        )
    return gemv_swiglu_rows(x, gate_mod.weight, up_mod.weight)


def gemv_lin_multi(x: mx.array, mods) -> list[mx.array]:
    mods = list(mods)
    fp8_idx = [i for i, m in enumerate(mods) if _is_fp8_mod(m)]
    bf_idx = [i for i, m in enumerate(mods) if not _is_fp8_mod(m)]
    out = [None] * len(mods)
    if fp8_idx:
        fp8_outs = gemv_fp8_rows_group(
            [x] * len(fp8_idx),
            [mods[i].weight_fp8 for i in fp8_idx],
            [mods[i].weight_scale_inv for i in fp8_idx],
        )
        for i, o in zip(fp8_idx, fp8_outs):
            out[i] = o
    if bf_idx:
        bf_outs = gemv_rows_group([x] * len(bf_idx), [mods[i].weight for i in bf_idx])
        for i, o in zip(bf_idx, bf_outs):
            out[i] = o
    return out
