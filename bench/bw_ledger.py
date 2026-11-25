"""Cold-DRAM silicon bandwidth ceiling & full-model component bandwidth ledger for LingSante on 512S1.

Measures:
  1. Raw >=1 GB cold Metal uint4 DRAM bandwidth ceiling (read-only & read+write) on 512S1.
  2. Exact cold-DRAM per-component ms & GB/s breakdown across ALL real model layers at
     C1_L1, C1_L5, C1_L8, and C4_L3 (zero SLC-warm illusions), plus e2e vs sum(components) bubble gap.

Usage:
  python bench/bw_ledger.py [--regimes C1_L1,C1_L5,C1_L8,C4_L3] [--out ~/science/runs/bw_ledger.json]
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(_root / "vendor"), str(_root)]

import mlx_lm  # noqa: E402,F401
import mlx.core as mx  # noqa: E402
from mlx_lm import load  # noqa: E402
from mlx_lm.models.bailing_moe_v3.mtp import BailingMTPLayer  # noqa: E402
from mlx_lm.models.bailing_moe_v3.rows.gemv_fp8 import (  # noqa: E402
    gemv_lin,
    gemv_lin_multi,
    gemv_lin_res,
    gemv_lin_swiglu,
    gemv_router_se_swiglu_rows,
)
from mlx_lm.models.bailing_moe_v3.rows.kda import (  # noqa: E402
    kda_fused_verify_rows,
    kda_gdn_commit_and_verify_seq_rows,
    kda_gdn_commit_and_verify_seq_split_rows,
    kda_gdn_verify_seq_rows,
    kda_gdn_verify_seq_split_rows,
    kda_pre_seq_rows,
)
from mlx_lm.models.bailing_moe_v3.rows.mla import mla_verify_layer_rows  # noqa: E402
from mlx_lm.models.bailing_moe_v3.rows.moe import (  # noqa: E402
    moe_down_combine_packed_rows,
    moe_down_combine_rows,
    moe_gate_up_swiglu_packed_rows,
    moe_gate_up_swiglu_rows,
)
from mlx_lm.models.bailing_moe_v3.rows.router import group_router_dedup_rows, group_router_rows  # noqa: E402
from mlx_lm.models.bailing_moe_v3.verify_decode import fast_verify_forward  # noqa: E402
from mlx_lm.utils import maybe_set_recommended_wired_limit  # noqa: E402

from bench.exact import DEFAULT_MODEL  # noqa: E402
from bench.profile_verify_c1 import _trim_mla_rb, build_realistic_verify_input  # noqa: E402

_RAW_READ_SRC = r"""
    // Contiguous-per-TG streaming uint4 read (maximum DRAM page-row locality on Apple Silicon).
    const uint lid = thread_position_in_threadgroup.x;
    const uint tg = threadgroup_position_in_grid.x;
    const uint ntg = threadgroups_per_grid.x;
    const uint tgp = threads_per_threadgroup.x;
    const size_t n4 = size_t(params[0]);
    const size_t per_tg = (n4 + ntg - 1) / ntg;
    const size_t beg = size_t(tg) * per_tg;
    const size_t end = min(n4, beg + per_tg);
    const device uint4* p = (const device uint4*)buf;
    uint4 acc0 = uint4(0), acc1 = uint4(0), acc2 = uint4(0), acc3 = uint4(0);
    size_t i = beg + size_t(lid);
    const size_t step = size_t(tgp) * 4;
    for (; i + size_t(tgp) * 3 < end; i += step) {
      acc0 ^= p[i];
      acc1 ^= p[i + tgp];
      acc2 ^= p[i + size_t(tgp) * 2];
      acc3 ^= p[i + size_t(tgp) * 3];
    }
    for (; i < end; i += tgp) {
      acc0 ^= p[i];
    }
    uint4 a = (acc0 ^ acc1) ^ (acc2 ^ acc3);
    if (tg == 0 && lid < 256) {
      out[lid] = a.x ^ a.y ^ a.z ^ a.w;
    }
"""

_RAW_COPY_SRC = r"""
    const uint lid = thread_position_in_threadgroup.x;
    const uint tg = threadgroup_position_in_grid.x;
    const uint ntg = threadgroups_per_grid.x;
    const uint tgp = threads_per_threadgroup.x;
    const size_t n4 = size_t(params[0]);
    const size_t per_tg = (n4 + ntg - 1) / ntg;
    const size_t beg = size_t(tg) * per_tg;
    const size_t end = min(n4, beg + per_tg);
    const device uint4* src = (const device uint4*)buf_in;
    device uint4* dst = (device uint4*)buf_out;
    for (size_t i = beg + size_t(lid); i < end; i += tgp) {
      dst[i] = src[i];
    }
"""


def _bench_median_ms(fn, warm: int = 2, iters: int = 7) -> float:
    for _ in range(warm):
        res = fn()
        mx.eval(*res) if isinstance(res, (list, tuple)) else mx.eval(res)
    mx.synchronize()
    dts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        res = fn()
        mx.eval(*res) if isinstance(res, (list, tuple)) else mx.eval(res)
        dts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(dts)


def _bench_pipe_ms(fn, reps: int = 4) -> float:
    """Zero-launch-overhead GPU timing via chained (4 - 1) / 3 differential."""
    mx.eval(fn())
    mx.synchronize()
    dts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(fn())
        t1 = time.perf_counter()
        mx.eval([fn() for _ in range(4)])
        t2 = time.perf_counter()
        dts.append(((t2 - t1) - (t1 - t0)) * 1e3 / 3.0)
    return statistics.median(dts)


def measure_silicon_ceiling() -> dict:
    """Measure raw cold Metal uint4 DRAM read-only & read+write ceiling on >=1 GB buffers."""
    print("\n=== 1. Raw Silicon Cold-DRAM Bandwidth Ceiling (Metal uint4, contiguous-per-TG) ===", flush=True)
    k_read = mx.fast.metal_kernel(
        name="bw_raw_read_u4_contig",
        input_names=["buf", "params"],
        output_names=["out"],
        source=_RAW_READ_SRC,
        ensure_row_contiguous=True,
    )
    k_copy = mx.fast.metal_kernel(
        name="bw_raw_copy_u4_contig",
        input_names=["buf_in", "params"],
        output_names=["buf_out"],
        source=_RAW_COPY_SRC,
        ensure_row_contiguous=True,
    )

    out = {}
    for gb in (1.0, 2.0, 4.0):
        n_u32 = int(gb * (1 << 30)) // 4
        n_u4 = n_u32 // 4
        buf = mx.arange(n_u32, dtype=mx.uint32)
        params = mx.array([n_u4], dtype=mx.uint32)
        mx.eval(buf, params)
        best_gbs = 0.0
        best_cfg = None
        for ntg in (160, 320, 640, 1280, 2560):
            nt = ntg * 256
            fn = lambda nt=nt: k_read(
                inputs=[buf, params],
                grid=(nt, 1, 1),
                threadgroup=(256, 1, 1),
                output_shapes=[(256,)],
                output_dtypes=[mx.uint32],
            )[0]
            ms = _bench_pipe_ms(fn, reps=4)
            gbs = (n_u32 * 4 / 1e9) / (ms / 1e3)
            if gbs > best_gbs:
                best_gbs = gbs
                best_cfg = (ntg, ms)
        out[f"read_{gb:.0f}GiB_GBs"] = round(best_gbs, 1)
        out[f"read_{gb:.0f}GiB_ms"] = round(best_cfg[1], 3)
        print(
            f"  uint4 read-only {gb:.0f} GiB ({n_u32*4/1e9:.3f} GB): "
            f"{best_cfg[1]:.3f} ms = {best_gbs:.1f} GB/s (pipelined, best ntg={best_cfg[0]})",
            flush=True,
        )
        del buf
        mx.clear_cache()

    # Read + Write copy (1 GiB in + 1 GiB out = 2.147 GB traffic, matching KDA state R+W)
    n_u32 = (1 << 30) // 4
    n_u4 = n_u32 // 4
    buf = mx.arange(n_u32, dtype=mx.uint32)
    params = mx.array([n_u4], dtype=mx.uint32)
    mx.eval(buf, params)
    best_rw = 0.0
    best_rw_ms = 0.0
    for ntg in (160, 320, 640, 1280):
        nt = ntg * 256
        fn = lambda nt=nt: k_copy(
            inputs=[buf, params],
            grid=(nt, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(n_u32,)],
            output_dtypes=[mx.uint32],
        )[0]
        ms = _bench_pipe_ms(fn, reps=4)
        gbs = (2 * n_u32 * 4 / 1e9) / (ms / 1e3)
        if gbs > best_rw:
            best_rw = gbs
            best_rw_ms = ms
    out["rw_1GiB_GBs"] = round(best_rw, 1)
    out["rw_1GiB_ms"] = round(best_rw_ms, 3)
    print(
        f"  uint4 read+write 1+1 GiB ({2*n_u32*4/1e9:.3f} GB): "
        f"{best_rw_ms:.3f} ms = {best_rw:.1f} GB/s (pipelined)",
        flush=True,
    )
    del buf
    mx.clear_cache()
    return out


def _fp8_lin_bytes(mod) -> int:
    if hasattr(mod, "weight_fp8"):
        return mod.weight_fp8.nbytes + mod.weight_scale_inv.nbytes
    return mod.weight.nbytes


def _capture_all_layers(model, inp: mx.array, vcache, C: int, L: int):
    """Run 1 exact pass capturing per-layer activations and routing across all 72 layers."""
    B = C * L
    inner = model.model
    layers = inner.layers
    x = inner.word_embeddings(inp.reshape(-1))
    acts = []
    uniq_counts = []

    for li, (layer, c) in enumerate(zip(layers, vcache)):
        rec = {"li": li, "is_linear": layer.is_linear, "is_moe": hasattr(layer.mlp, "switch_mlp"), "x": x}
        h_in = mx.fast.rms_norm(x, layer.input_layernorm.weight, layer.input_layernorm.eps)
        rec["h_in"] = h_in
        if layer.is_linear:
            attn = layer.attention
            q_r, k_r, v_r, f_r, g_r, b_r = gemv_lin_multi(
                h_in, [attn.q_proj, attn.k_proj, attn.v_proj, attn.f_proj, attn.g_proj, attn.b_proj]
            )
            q, k, v, g, beta = kda_pre_seq_rows(
                q_r, k_r, v_r, f_r, b_r, c[0], c[1], c[2],
                attn.q_conv1d.weight, attn.k_conv1d.weight, attn.v_conv1d.weight,
                attn.A_log, attn.dt_bias, C=C, L=L, lower_bound=attn.lower_bound, eps=1e-6, emit_hist=False,
            )
            is_split = len(c.cache) > 4
            if is_split:
                ov, _ = kda_gdn_verify_seq_split_rows(
                    q, k, v, g, beta, list(c.cache[3:]), g_r, attn.o_norm.weight, C=C, L=L, eps=attn.o_norm.eps, write_end=True
                )
            else:
                ov, _ = kda_gdn_verify_seq_rows(
                    q, k, v, g, beta, c[3], g_r, attn.o_norm.weight, C=C, L=L, eps=attn.o_norm.eps, write_end=True
                )
            h_post = gemv_lin_res(ov, attn.o_proj, x)
            rec.update({"q_r": q_r, "k_r": k_r, "v_r": v_r, "f_r": f_r, "g_r": g_r, "b_r": b_r,
                        "q": q, "k": k, "v": v, "g": g, "beta": beta, "ov": ov, "h_post": h_post})
        else:
            h_post = mla_verify_layer_rows(h_in, x, layer.attention, c, C=C, L=L)
            if isinstance(c, list):
                for ci in c:
                    ci.trim(L)
            else:
                c.trim(L)
            rec["h_post"] = h_post

        mlp_in = mx.fast.rms_norm(h_post, layer.post_attention_layernorm.weight, layer.post_attention_layernorm.eps)
        rec["mlp_in"] = mlp_in
        mlp = layer.mlp
        if hasattr(mlp, "switch_mlp"):
            gate = mlp.gate
            se = mlp.shared_experts
            logits_r, se_mid = gemv_router_se_swiglu_rows(mlp_in, gate.weight, se.gate_proj, se.up_proj)
            if B >= 2:
                inds, wts, dedup = group_router_dedup_rows(
                    logits_r, gate.expert_bias, top_k=gate.top_k, n_group=gate.n_group,
                    topk_group=gate.topk_group, routed_scaling_factor=gate.routed_scaling_factor,
                )
            else:
                inds, wts = group_router_rows(
                    logits_r, gate.expert_bias, top_k=gate.top_k, n_group=gate.n_group,
                    topk_group=gate.topk_group, routed_scaling_factor=gate.routed_scaling_factor,
                )
                dedup = None
            sw = mlp.switch_mlp
            if hasattr(sw, "_gu_packed"):
                h_mid = moe_gate_up_swiglu_packed_rows(mlp_in, sw._gu_packed, inds, dedup=dedup)
                sh = gemv_lin(se_mid, se.down_proj)
                x = moe_down_combine_packed_rows(
                    h_mid, sw._dn_packed, inds, wts, shared=sh, residual=h_post, dedup=dedup
                )
            else:
                h_mid = moe_gate_up_swiglu_rows(
                    mlp_in, sw.gate_proj.weight, sw.gate_proj.scales, sw.up_proj.weight, sw.up_proj.scales, inds, dedup=dedup
                )
                sh = gemv_lin(se_mid, se.down_proj)
                x = moe_down_combine_rows(
                    h_mid, sw.down_proj.weight, sw.down_proj.scales, inds, wts, shared=sh, residual=h_post, dedup=dedup
                )
            mx.eval(x, inds)
            uniq_counts.append(len(set(inds.reshape(-1).tolist())))
            rec.update({"logits_r": logits_r, "se_mid": se_mid, "inds": inds, "wts": wts,
                        "dedup": dedup, "h_mid": h_mid, "sh": sh})
        else:
            mid = gemv_lin_swiglu(mlp_in, mlp.gate_proj, mlp.up_proj)
            x = gemv_lin_res(mid, mlp.down_proj, h_post)
            mx.eval(x)
            rec["mid"] = mid
        acts.append(rec)

    hidden = mx.fast.rms_norm(x, inner.norm.weight, inner.norm.eps)
    mx.eval(hidden)
    return acts, hidden, x, uniq_counts


def measure_regime_ledger(model, mtp, tok, C: int, L: int) -> dict:
    B = C * L
    vcache, inp = build_realistic_verify_input(model, mtp, tok, C=C, K=L - 1)
    acts, hidden, x_last, uniq_counts = _capture_all_layers(model, inp, vcache, C, L)
    inner = model.model
    layers = inner.layers
    mean_u = sum(uniq_counts) / max(1, len(uniq_counts))
    total_u = sum(uniq_counts)

    # 1. Full e2e (write_end=True, matching profile_verify_c1 --e2e-only)
    def run_e2e():
        lg, rb = fast_verify_forward(model, inp, vcache, return_hidden=False, return_rollback=True, lazy_kda=False)
        _trim_mla_rb(rb, L)
        return lg

    e2e_ms = _bench_median_ms(run_e2e, warm=3, iters=7)

    # 2. Cold multi-layer component benchmarks (each runs across ALL its layers in the model!)
    kda_idx = [i for i, l in enumerate(layers) if l.is_linear]
    mla_idx = [i for i, l in enumerate(layers) if not l.is_linear]
    dense_idx = [i for i, l in enumerate(layers) if not hasattr(l.mlp, "switch_mlp")]
    moe_idx = [i for i, l in enumerate(layers) if hasattr(l.mlp, "switch_mlp")]

    # (a) KDA qkvfg+b (63 layers)
    def run_kda_qkv():
        outs = []
        for i in kda_idx:
            a = layers[i].attention
            outs.extend(gemv_lin_multi(acts[i]["h_in"], [a.q_proj, a.k_proj, a.v_proj, a.f_proj, a.g_proj, a.b_proj]))
        return outs

    bytes_kda_qkv = sum(
        sum(_fp8_lin_bytes(m) for m in (layers[i].attention.q_proj, layers[i].attention.k_proj,
                                        layers[i].attention.v_proj, layers[i].attention.f_proj,
                                        layers[i].attention.g_proj, layers[i].attention.b_proj))
        for i in kda_idx
    )
    ms_kda_qkv = _bench_median_ms(run_kda_qkv, warm=2, iters=5)

    # (b) KDA o_proj (63 layers)
    def run_kda_o():
        return [gemv_lin_res(acts[i]["ov"], layers[i].attention.o_proj, acts[i]["x"]) for i in kda_idx]

    bytes_kda_o = sum(_fp8_lin_bytes(layers[i].attention.o_proj) for i in kda_idx)
    ms_kda_o = _bench_median_ms(run_kda_o, warm=2, iters=5)

    # (c) KDA pre_seq (63 layers)
    def run_kda_pre():
        outs = []
        for i in kda_idx:
            a = layers[i].attention
            c = vcache[i]
            r = acts[i]
            outs.extend(kda_pre_seq_rows(
                r["q_r"], r["k_r"], r["v_r"], r["f_r"], r["b_r"], c[0], c[1], c[2],
                a.q_conv1d.weight, a.k_conv1d.weight, a.v_conv1d.weight,
                a.A_log, a.dt_bias, C=C, L=L, lower_bound=a.lower_bound, eps=1e-6, emit_hist=False,
            ))
        return outs

    bytes_kda_pre = len(kda_idx) * (B * (5 * 8192 * 2 + 3 * 8192 * 2 + 8192 * 4) + C * 3 * 3 * 8192 * 2)
    ms_kda_pre = _bench_median_ms(run_kda_pre, warm=2, iters=5)

    # (d) KDA gdn_verify (63 layers, write_end=True matching e2e, and write_end=False / lazy commit+verify)
    is_split = len(vcache[kda_idx[0]].cache) > 4

    def run_kda_gdn(write_end=True):
        outs = []
        for i in kda_idx:
            a = layers[i].attention
            c = vcache[i]
            r = acts[i]
            if is_split:
                res = kda_gdn_verify_seq_split_rows(
                    r["q"], r["k"], r["v"], r["g"], r["beta"], list(c.cache[3:]), r["g_r"],
                    a.o_norm.weight, C=C, L=L, eps=a.o_norm.eps, write_end=write_end,
                )
                outs.append(res[0] if write_end else res)
                if write_end:
                    outs.extend(res[1])
            else:
                res = kda_gdn_verify_seq_rows(
                    r["q"], r["k"], r["v"], r["g"], r["beta"], c[3], r["g_r"],
                    a.o_norm.weight, C=C, L=L, eps=a.o_norm.eps, write_end=write_end,
                )
                outs.extend(res if write_end else [res])
        return outs

    bytes_kda_gdn_rw = len(kda_idx) * (2 * C * 64 * 128 * 128 * 4)  # state read + state write
    bytes_kda_gdn_ro = len(kda_idx) * (C * 64 * 128 * 128 * 4)      # state read only
    ms_kda_gdn_rw = _bench_median_ms(lambda: run_kda_gdn(True), warm=2, iters=5)
    ms_kda_gdn_ro = _bench_median_ms(lambda: run_kda_gdn(False), warm=2, iters=5)

    # Also measure lazy commit+verify (realistic spec cycle: prev_nc=min(L,3))
    p_off = mx.array([ci * L for ci in range(C)], dtype=mx.int32)
    p_nc = mx.array([min(L, 3)] * C, dtype=mx.int32)
    mx.eval(p_off, p_nc)

    def run_kda_lazy_cv():
        outs = []
        for i in kda_idx:
            a = layers[i].attention
            c = vcache[i]
            r = acts[i]
            if is_split:
                ov, sc = kda_gdn_commit_and_verify_seq_split_rows(
                    r["q"], r["k"], r["v"], r["g"], r["beta"], list(c.cache[3:]),
                    r["k"], r["v"], r["g"], r["beta"], p_off, p_nc,
                    r["g_r"], a.o_norm.weight, C=C, L=L, eps=a.o_norm.eps, write_end=False,
                )
                outs.append(ov)
                outs.extend(sc)
            else:
                ov, sc = kda_gdn_commit_and_verify_seq_rows(
                    r["q"], r["k"], r["v"], r["g"], r["beta"], c[3],
                    r["k"], r["v"], r["g"], r["beta"], p_off, p_nc,
                    r["g_r"], a.o_norm.weight, C=C, L=L, eps=a.o_norm.eps, write_end=False,
                )
                outs.extend([ov, sc])
        return outs

    ms_kda_lazy_cv = _bench_median_ms(run_kda_lazy_cv, warm=2, iters=5)

    def run_kda_fused():
        outs = []
        for i in kda_idx:
            a = layers[i].attention
            c = vcache[i]
            r = acts[i]
            s_in = list(c.cache[3:]) if is_split else c[3]
            ov, _, se, *_ = kda_fused_verify_rows(
                r["q_r"], r["k_r"], r["v_r"], r["f_r"], r["g_r"], r["b_r"],
                c[0], c[1], c[2],
                a.q_conv1d.weight, a.k_conv1d.weight, a.v_conv1d.weight,
                a.A_log, a.dt_bias, s_in, a.o_norm.weight,
                C=C, L=L, lower_bound=a.lower_bound, eps=1e-6, o_eps=a.o_norm.eps,
                write_end=True, emit_kvgb=(L > 1),
            )
            outs.append(ov)
            outs.extend(se if is_split else [se])
        return outs

    ms_kda_fused = _bench_median_ms(run_kda_fused, warm=2, iters=5)

    # (e) MLA full (9 layers)
    def run_mla():
        outs = []
        for i in mla_idx:
            c = vcache[i]
            outs.append(mla_verify_layer_rows(acts[i]["h_in"], acts[i]["x"], layers[i].attention, c, C=C, L=L))
            if isinstance(c, list):
                for ci in c:
                    ci.trim(L)
            else:
                c.trim(L)
        return outs

    bytes_mla = sum(
        sum(_fp8_lin_bytes(m) for m in (
            layers[i].attention.q_a_proj, layers[i].attention.kv_a_proj_with_mqa,
            layers[i].attention.g_proj, layers[i].attention.q_b_proj,
            layers[i].attention.embed_q, layers[i].attention.unembed_out,
            layers[i].attention.dense,
        ))
        for i in mla_idx
    )
    ms_mla = _bench_median_ms(run_mla, warm=2, iters=5)

    # (f) Dense MLPs (4 layers)
    def run_dense():
        outs = []
        for i in dense_idx:
            mlp = layers[i].mlp
            mid = gemv_lin_swiglu(acts[i]["mlp_in"], mlp.gate_proj, mlp.up_proj)
            outs.append(gemv_lin_res(mid, mlp.down_proj, acts[i]["h_post"]))
        return outs

    bytes_dense = sum(
        _fp8_lin_bytes(layers[i].mlp.gate_proj)
        + _fp8_lin_bytes(layers[i].mlp.up_proj)
        + _fp8_lin_bytes(layers[i].mlp.down_proj)
        for i in dense_idx
    )
    ms_dense = _bench_median_ms(run_dense, warm=2, iters=5)

    # (g) Router + Shared SwiGLU (68 layers)
    def run_rtr_se_sw():
        outs = []
        for i in moe_idx:
            mlp = layers[i].mlp
            se = mlp.shared_experts
            outs.extend(gemv_router_se_swiglu_rows(acts[i]["mlp_in"], mlp.gate.weight, se.gate_proj, se.up_proj))
        return outs

    bytes_rtr_se_sw = sum(
        layers[i].mlp.gate.weight.nbytes
        + _fp8_lin_bytes(layers[i].mlp.shared_experts.gate_proj)
        + _fp8_lin_bytes(layers[i].mlp.shared_experts.up_proj)
        for i in moe_idx
    )
    ms_rtr_se_sw = _bench_median_ms(run_rtr_se_sw, warm=2, iters=5)

    # (h) Router select + dedup (68 layers)
    def run_rtr_sel():
        outs = []
        for i in moe_idx:
            g = layers[i].mlp.gate
            if B >= 2:
                inds, wts, dd = group_router_dedup_rows(
                    acts[i]["logits_r"], g.expert_bias, top_k=g.top_k, n_group=g.n_group,
                    topk_group=g.topk_group, routed_scaling_factor=g.routed_scaling_factor,
                )
                outs.extend([inds, wts, dd[0], dd[1]])
            else:
                outs.extend(group_router_rows(
                    acts[i]["logits_r"], g.expert_bias, top_k=g.top_k, n_group=g.n_group,
                    topk_group=g.topk_group, routed_scaling_factor=g.routed_scaling_factor,
                ))
        return outs

    ms_rtr_sel = _bench_median_ms(run_rtr_sel, warm=2, iters=5)

    # (i) Shared Down (68 layers)
    def run_se_dn():
        return [gemv_lin(acts[i]["se_mid"], layers[i].mlp.shared_experts.down_proj) for i in moe_idx]

    bytes_se_dn = sum(_fp8_lin_bytes(layers[i].mlp.shared_experts.down_proj) for i in moe_idx)
    ms_se_dn = _bench_median_ms(run_se_dn, warm=2, iters=5)

    # (j) MoE gate_up_swiglu (68 layers)
    def run_moe_gu():
        outs = []
        for i in moe_idx:
            sw = layers[i].mlp.switch_mlp
            r = acts[i]
            if hasattr(sw, "_gu_packed"):
                outs.append(moe_gate_up_swiglu_packed_rows(r["mlp_in"], sw._gu_packed, r["inds"], dedup=r["dedup"]))
            else:
                outs.append(moe_gate_up_swiglu_rows(
                    r["mlp_in"], sw.gate_proj.weight, sw.gate_proj.scales,
                    sw.up_proj.weight, sw.up_proj.scales, r["inds"], dedup=r["dedup"],
                ))
        return outs

    # MXFP4: 0.5 B/elem weights + 1/32 B/elem scales = 17/32 = 0.53125 B/elem
    bytes_moe_gu = int(total_u * 2 * 1280 * 4096 * 17 // 32)
    ms_moe_gu = _bench_median_ms(run_moe_gu, warm=2, iters=5)

    # (k) MoE down_combine (68 layers)
    def run_moe_dc():
        outs = []
        for i in moe_idx:
            sw = layers[i].mlp.switch_mlp
            r = acts[i]
            if hasattr(sw, "_dn_packed"):
                outs.append(moe_down_combine_packed_rows(
                    r["h_mid"], sw._dn_packed, r["inds"], r["wts"],
                    shared=r["sh"], residual=r["h_post"], dedup=r["dedup"],
                ))
            else:
                outs.append(moe_down_combine_rows(
                    r["h_mid"], sw.down_proj.weight, sw.down_proj.scales,
                    r["inds"], r["wts"], shared=r["sh"], residual=r["h_post"], dedup=r["dedup"],
                ))
        return outs

    bytes_moe_dc = int(total_u * 4096 * 1280 * 17 // 32)
    ms_moe_dc = _bench_median_ms(run_moe_dc, warm=2, iters=5)

    # (l) 145 RMSNorms (72 input + 72 post_attn + 1 final)
    def run_norms():
        outs = []
        for i, layer in enumerate(layers):
            outs.append(mx.fast.rms_norm(acts[i]["x"], layer.input_layernorm.weight, layer.input_layernorm.eps))
            outs.append(mx.fast.rms_norm(acts[i]["h_post"], layer.post_attention_layernorm.weight, layer.post_attention_layernorm.eps))
        outs.append(mx.fast.rms_norm(x_last, inner.norm.weight, inner.norm.eps))
        return outs

    ms_norms = _bench_median_ms(run_norms, warm=2, iters=5)

    # (m) lm_head (1.288 GB BF16, cold DRAM)
    bytes_lm = model.lm_head.weight.nbytes
    ms_lm = _bench_median_ms(lambda: gemv_lin(hidden, model.lm_head), warm=2, iters=7)

    comps = [
        ("kda_qkvfg_b (x63)", bytes_kda_qkv, ms_kda_qkv),
        ("kda_o_proj (x63)", bytes_kda_o, ms_kda_o),
        ("kda_pre_seq (x63)", bytes_kda_pre, ms_kda_pre),
        ("kda_gdn_rw (x63)", bytes_kda_gdn_rw, ms_kda_gdn_rw),
        ("mla_full (x9)", bytes_mla, ms_mla),
        ("dense_mlp (x4)", bytes_dense, ms_dense),
        ("rtr_se_sw (x68)", bytes_rtr_se_sw, ms_rtr_se_sw),
        ("rtr_sel_dd (x68)", 0, ms_rtr_sel),
        ("se_down (x68)", bytes_se_dn, ms_se_dn),
        ("moe_gu (x68)", bytes_moe_gu, ms_moe_gu),
        ("moe_dc (x68)", bytes_moe_dc, ms_moe_dc),
        ("rms_norm (x145)", 0, ms_norms),
        ("lm_head (x1)", bytes_lm, ms_lm),
    ]
    sum_ms = sum(m for _, _, m in comps)
    total_bytes = sum(b for _, b, _ in comps)
    bubble_ms = e2e_ms - sum_ms
    eff_gbs = (total_bytes / 1e9) / (e2e_ms / 1e3)
    sum_gbs = (total_bytes / 1e9) / (sum_ms / 1e3)

    print(
        f"\n=== 2. Cold-DRAM Ledger: C={C} L={L} (B={B}, mean_U={mean_u:.2f}) | "
        f"e2e={e2e_ms:.2f}ms ({eff_gbs:.1f} GB/s) | sum={sum_ms:.2f}ms ({sum_gbs:.1f} GB/s) | "
        f"bubble={bubble_ms:+.2f}ms | total={total_bytes/1e9:.2f} GB ===",
        flush=True,
    )
    print(f"  {'Component':22s} | {'GB':>6s} | {'ms':>7s} | {'GB/s':>7s} | {'%e2e':>5s}", flush=True)
    print("  " + "-" * 58, flush=True)
    rows = {}
    for nm, nb, ms in comps:
        gb = nb / 1e9
        gbs = gb / (ms / 1e3) if (nb > 0 and ms > 0) else 0.0
        pct = 100.0 * ms / e2e_ms
        print(f"  {nm:22s} | {gb:6.3f} | {ms:7.2f} | {gbs:7.1f} | {pct:4.1f}%", flush=True)
        rows[nm] = {"gb": round(gb, 4), "ms": round(ms, 3), "gbs": round(gbs, 1), "pct": round(pct, 1)}
    print(
        f"  [alt] kda_gdn_ro (x63) : {bytes_kda_gdn_ro/1e9:6.3f} GB | {ms_kda_gdn_ro:7.2f} ms | "
        f"{(bytes_kda_gdn_ro/1e9)/(ms_kda_gdn_ro/1e3):7.1f} GB/s",
        flush=True,
    )
    print(
        f"  [alt] kda_lazy_cv (x63): {bytes_kda_gdn_rw/1e9:6.3f} GB | {ms_kda_lazy_cv:7.2f} ms | "
        f"{(bytes_kda_gdn_rw/1e9)/(ms_kda_lazy_cv/1e3):7.1f} GB/s",
        flush=True,
    )
    print(
        f"  [alt] kda_fused1L (x63): {(bytes_kda_gdn_rw+bytes_kda_pre)/1e9:6.3f} GB | {ms_kda_fused:7.2f} ms | "
        f"{((bytes_kda_gdn_rw+bytes_kda_pre)/1e9)/(ms_kda_fused/1e3):7.1f} GB/s",
        flush=True,
    )
    return {
        "C": C,
        "L": L,
        "B": B,
        "mean_U": round(mean_u, 2),
        "total_gb": round(total_bytes / 1e9, 3),
        "e2e_ms": round(e2e_ms, 3),
        "e2e_gbs": round(eff_gbs, 1),
        "sum_ms": round(sum_ms, 3),
        "sum_gbs": round(sum_gbs, 1),
        "bubble_ms": round(bubble_ms, 3),
        "kda_gdn_ro_ms": round(ms_kda_gdn_ro, 3),
        "kda_lazy_cv_ms": round(ms_kda_lazy_cv, 3),
        "kda_fused_ms": round(ms_kda_fused, 3),
        "components": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regimes", default="C1_L1,C1_L5,C1_L8,C4_L3")
    ap.add_argument("--skip-raw", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    maybe_set_recommended_wired_limit()
    report = {}
    if not args.skip_raw:
        report["silicon"] = measure_silicon_ceiling()

    model, tok = load(DEFAULT_MODEL)
    mx.eval(model.parameters())
    mtp = BailingMTPLayer.from_target(model, DEFAULT_MODEL, with_q4_head=True)
    report["regimes"] = {}
    for tag in [s.strip() for s in args.regimes.split(",") if s.strip()]:
        c_s, l_s = tag.split("_")
        C, L = int(c_s[1:]), int(l_s[1:])
        report["regimes"][tag] = measure_regime_ledger(model, mtp, tok, C, L)

    if args.out:
        p = Path(args.out).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2))
        print(f"\nWrote {p}", flush=True)


if __name__ == "__main__":
    main()
