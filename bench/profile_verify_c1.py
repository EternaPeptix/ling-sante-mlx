"""Profile C=1 target verification across L=1..16 and sweep bit-exact FP8 / flush configs.

Usage:
  python bench/profile_verify_c1.py [--e2e-only] [--out ~/science/runs/profile_verify_c1.json]
"""
import argparse
import copy
import json
import os
import statistics
import sys
import time
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(_root / "vendor"), str(_root)]

import mlx_lm  # noqa: E402,F401
import mlx.core as mx  # noqa: E402
from mlx_lm import load  # noqa: E402
import mlx_lm.models.bailing_moe_v3.verify_decode as vd  # noqa: E402
from mlx_lm.layers.common import swiglu  # noqa: E402
from mlx_lm.models.bailing_moe_v3.mtp import BailingMTPLayer  # noqa: E402
from mlx_lm.models.bailing_moe_v3.rows.gemv import (  # noqa: E402
    gemv_rows_group,
    gemv_rows_res,
    gemv_swiglu_rows,
)
from mlx_lm.models.bailing_moe_v3.rows.gemv_fp8 import (  # noqa: E402
    _can_coop,
    dequant_fp8_to_bf16,
    gemv_fp8_rows_group,
    gemv_fp8_rows_res,
    gemv_fp8_swiglu_rows,
    gemv_lin,
    gemv_lin_multi,
    gemv_lin_res,
    gemv_lin_swiglu,
)
from mlx_lm.models.bailing_moe_v3.rows.kda import (  # noqa: E402
    kda_gdn_verify_seq_rows,
    kda_pre_seq_rows,
)
from mlx_lm.models.bailing_moe_v3.rows.mla import mla_verify_layer_rows  # noqa: E402
from mlx_lm.models.bailing_moe_v3.rows.moe import (  # noqa: E402
    moe_dedup_rows,
    moe_down_combine_rows,
    moe_gate_up_swiglu_rows,
)
from mlx_lm.models.bailing_moe_v3.rows.router import gate_rows  # noqa: E402
from mlx_lm.models.bailing_moe_v3.spec_engine import _chain_mtp_drafts, prefill_with_hidden  # noqa: E402
from mlx_lm.models.bailing_moe_v3.verify_decode import (  # noqa: E402
    commit_verify_cache,
    fast_verify_forward,
    make_verify_cache,
)
from mlx_lm.utils import maybe_set_recommended_wired_limit  # noqa: E402

from bench._layer_loader import bit_diff_count  # noqa: E402
from bench.exact import DEFAULT_MODEL, PROMPTS, prompt_ids  # noqa: E402


def _flat(x):
    return list(x) if isinstance(x, (list, tuple)) else [x]


def bench_ms(fn, warm: int = 3, iters: int = 7) -> float:
    for _ in range(warm):
        mx.eval(*_flat(fn()))
    mx.synchronize()
    dts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        mx.eval(*_flat(fn()))
        dts.append((time.perf_counter() - t0) * 1e3)
    return round(statistics.median(dts), 3)


def pipe_ms(fn, n_lo: int = 2, n_hi: int = 10, reps: int = 4) -> float:
    mx.eval(*[a for _ in range(2) for a in _flat(fn())])
    mx.synchronize()
    dts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(*[a for _ in range(n_lo) for a in _flat(fn())])
        dt_lo = time.perf_counter() - t0
        t1 = time.perf_counter()
        mx.eval(*[a for _ in range(n_hi) for a in _flat(fn())])
        dt_hi = time.perf_counter() - t1
        dts.append((dt_hi - dt_lo) * 1e3 / (n_hi - n_lo))
    return round(statistics.median(dts), 4)


def _trim_mla_rb(rb: dict, L: int) -> None:
    for mc in rb["mla"]:
        if isinstance(mc, list):
            for ci in mc:
                ci.trim(L)
        else:
            mc.trim(L)


def build_realistic_verify_input(model, mtp, tok, C: int, K: int):
    """Prefill C prompts and draft real K tokens so L = K + 1 routing matches actual decode."""
    L = K + 1
    t_caches, mtp_caches, last_toks = [], [], []
    for _, text in PROMPTS[:C]:
        ids = prompt_ids(tok, text)
        tc, mc, lt = prefill_with_hidden(model, mtp, ids)
        t_caches.append(tc)
        mtp_caches.append(mc)
        last_toks.append(lt)
    vcache = make_verify_cache(model, t_caches)
    logits0, hidden0, rb0 = fast_verify_forward(
        model, mx.array([[lt] for lt in last_toks], dtype=mx.uint32), vcache, return_hidden=True, return_rollback=True
    )
    nxt0 = mx.argmax(logits0[:, 0, :], axis=-1)
    mx.eval(nxt0, hidden0)
    commit_verify_cache(vcache, rb0, 1)

    if K > 0:
        draft_mat, _ = _chain_mtp_drafts(mtp, nxt0, hidden0[:, 0, :], mtp_caches, C=C, K=K, use_q4_head=True)
        inp = mx.concatenate([nxt0[:, None], draft_mat], axis=1)
    else:
        inp = nxt0[:, None]
    mx.eval(inp)
    return vcache, inp


def measure_e2e_ledger(model, mtp, tok) -> dict:
    """Measure end-to-end fast_verify_forward median latency across C=1 L=1..16 and C=4 L=3."""
    regimes = [(1, L) for L in (1, 3, 5, 8, 9, 10, 12, 16)] + [(4, 3)]
    out = {}
    print("\n=== (a) End-to-End fast_verify_forward Ledger (warm=3, iters=7) ===", flush=True)
    for C, L in regimes:
        vcache, inp = build_realistic_verify_input(model, mtp, tok, C=C, K=L - 1)

        def run_one():
            lg, rb = fast_verify_forward(model, inp, vcache, return_hidden=False, return_rollback=True)
            _trim_mla_rb(rb, L)
            return lg

        ms = bench_ms(run_one, warm=3, iters=7)
        key = f"C{C}_L{L}"
        out[key] = ms
        print(f"  {key:8s} (B={C*L:2d}): {ms:6.2f} ms", flush=True)
    return out


def _capture_sample_acts(model, mtp, tok, L: int):
    """Capture realistic activations at KDA layer 4, MLA layer 7, MoE layer 4, and lm_head."""
    vcache, inp = build_realistic_verify_input(model, mtp, tok, C=1, K=L - 1)
    inner = model.model
    x = inner.word_embeddings(inp.reshape(-1))
    samples = {}
    uniq_list = []

    for li, (layer, c) in enumerate(zip(inner.layers, vcache)):
        h_in = mx.fast.rms_norm(x, layer.input_layernorm.weight, layer.input_layernorm.eps)
        if layer.is_linear:
            attn = layer.attention
            q_r, k_r, v_r, f_r, g_r, b_r = gemv_lin_multi(
                h_in, [attn.q_proj, attn.k_proj, attn.v_proj, attn.f_proj, attn.g_proj, attn.b_proj]
            )
            q, k, v, g, beta, h0, h1, h2 = kda_pre_seq_rows(
                q_r,
                k_r,
                v_r,
                f_r,
                b_r,
                c[0],
                c[1],
                c[2],
                attn.q_conv1d.weight,
                attn.k_conv1d.weight,
                attn.v_conv1d.weight,
                attn.A_log,
                attn.dt_bias,
                C=1,
                L=L,
                lower_bound=attn.lower_bound,
                eps=1e-6,
            )
            ov, _ = kda_gdn_verify_seq_rows(q, k, v, g, beta, c[3], g_r, attn.o_norm.weight, C=1, L=L, eps=attn.o_norm.eps)
            h_post = gemv_lin_res(ov, attn.o_proj, x)
            if li == 4:
                samples["kda4"] = {
                    "x": mx.array(x),
                    "h_in": mx.array(h_in),
                    "q_r": mx.array(q_r),
                    "k_r": mx.array(k_r),
                    "v_r": mx.array(v_r),
                    "f_r": mx.array(f_r),
                    "g_r": mx.array(g_r),
                    "b_r": mx.array(b_r),
                    "q": mx.array(q),
                    "k": mx.array(k),
                    "v": mx.array(v),
                    "g": mx.array(g),
                    "beta": mx.array(beta),
                    "c0": mx.array(c[0]),
                    "c1": mx.array(c[1]),
                    "c2": mx.array(c[2]),
                    "c3": mx.array(c[3]),
                    "ov": mx.array(ov),
                }
        else:
            if li == 7:
                samples["mla7"] = {"x": mx.array(x), "h_in": mx.array(h_in), "cache": copy.deepcopy(c)}
            h_post = mla_verify_layer_rows(h_in, x, layer.attention, c, C=1, L=L)
            c.trim(L)

        mlp_in = mx.fast.rms_norm(h_post, layer.post_attention_layernorm.weight, layer.post_attention_layernorm.eps)
        mlp = layer.mlp
        if hasattr(mlp, "switch_mlp"):
            gate = mlp.gate
            inds, wts = gate_rows(
                mlp_in,
                gate.weight,
                gate.expert_bias,
                top_k=gate.top_k,
                n_group=gate.n_group,
                topk_group=gate.topk_group,
                routed_scaling_factor=gate.routed_scaling_factor,
            )
            uniq_list.append(len(set(inds.reshape(-1).tolist())))
            se = mlp.shared_experts
            sh = gemv_lin(gemv_lin_swiglu(mlp_in, se.gate_proj, se.up_proj), se.down_proj)
            sw = mlp.switch_mlp
            dedup = moe_dedup_rows(inds) if L >= 5 else None
            h_mid = moe_gate_up_swiglu_rows(
                mlp_in, sw.gate_proj.weight, sw.gate_proj.scales, sw.up_proj.weight, sw.up_proj.scales, inds, dedup=dedup
            )
            x = moe_down_combine_rows(
                h_mid, sw.down_proj.weight, sw.down_proj.scales, inds, wts, shared=sh, residual=h_post, dedup=dedup
            )
            if li == 4:
                samples["moe4"] = {
                    "h_post": mx.array(h_post),
                    "mlp_in": mx.array(mlp_in),
                    "inds": mx.array(inds),
                    "wts": mx.array(wts),
                    "sh": mx.array(sh),
                    "h_mid": mx.array(h_mid),
                }
        else:
            mid = gemv_lin_swiglu(mlp_in, mlp.gate_proj, mlp.up_proj)
            x = gemv_lin_res(mid, mlp.down_proj, h_post)
        mx.eval(x)

    hidden = mx.fast.rms_norm(x, inner.norm.weight, inner.norm.eps)
    mx.eval(hidden)
    samples["hidden"] = hidden
    samples["mean_uniq_exp"] = round(sum(uniq_list) / max(1, len(uniq_list)), 2)
    return samples


def measure_components(model, mtp, tok) -> tuple[dict, dict]:
    """Component breakdown for C=1 L in (5, 8, 10, 12, 16) + saved samples for FP8 sweep."""
    Ls = (5, 8, 10, 12, 16)
    l4 = model.model.layers[4]
    l7 = model.model.layers[7]
    attn4 = l4.attention
    se4 = l4.mlp.shared_experts
    sw4 = l4.mlp.switch_mlp
    gate4 = l4.mlp.gate

    out = {}
    saved_samples = {}
    print("\n=== (b) Per-Component Breakdown for C=1, L in (5, 8, 10, 12, 16) ===", flush=True)
    for L in Ls:
        s = _capture_sample_acts(model, mtp, tok, L)
        saved_samples[L] = s
        k4 = s["kda4"]
        m7 = s["mla7"]
        e4 = s["moe4"]

        t_kda_multi = pipe_ms(
            lambda: gemv_lin_multi(
                k4["h_in"], [attn4.q_proj, attn4.k_proj, attn4.v_proj, attn4.f_proj, attn4.g_proj, attn4.b_proj]
            )
        )
        t_kda_oproj = pipe_ms(lambda: gemv_lin_res(k4["ov"], attn4.o_proj, k4["x"]))
        t_kda_gemv_1 = t_kda_multi + t_kda_oproj

        t_kda_pre_gdn_1 = pipe_ms(
            lambda: (
                lambda pre: kda_gdn_verify_seq_rows(
                    pre[0], pre[1], pre[2], pre[3], pre[4], k4["c3"], k4["g_r"], attn4.o_norm.weight, C=1, L=L, eps=attn4.o_norm.eps
                )
            )(
                kda_pre_seq_rows(
                    k4["q_r"],
                    k4["k_r"],
                    k4["v_r"],
                    k4["f_r"],
                    k4["b_r"],
                    k4["c0"],
                    k4["c1"],
                    k4["c2"],
                    attn4.q_conv1d.weight,
                    attn4.k_conv1d.weight,
                    attn4.v_conv1d.weight,
                    attn4.A_log,
                    attn4.dt_bias,
                    C=1,
                    L=L,
                    lower_bound=attn4.lower_bound,
                    eps=1e-6,
                )
            )
        )

        def run_mla():
            out_m = mla_verify_layer_rows(m7["h_in"], m7["x"], l7.attention, m7["cache"], C=1, L=L)
            m7["cache"].trim(L)
            return out_m

        t_mla_1 = bench_ms(run_mla, warm=3, iters=7)

        t_router_shared_1 = pipe_ms(
            lambda: (
                gate_rows(
                    e4["mlp_in"],
                    gate4.weight,
                    gate4.expert_bias,
                    top_k=gate4.top_k,
                    n_group=gate4.n_group,
                    topk_group=gate4.topk_group,
                    routed_scaling_factor=gate4.routed_scaling_factor,
                ),
                gemv_lin(gemv_lin_swiglu(e4["mlp_in"], se4.gate_proj, se4.up_proj), se4.down_proj),
            )
        )

        t_moe_1 = pipe_ms(
            lambda: (
                lambda dd: moe_down_combine_rows(
                    moe_gate_up_swiglu_rows(
                        e4["mlp_in"],
                        sw4.gate_proj.weight,
                        sw4.gate_proj.scales,
                        sw4.up_proj.weight,
                        sw4.up_proj.scales,
                        e4["inds"],
                        dedup=dd,
                    ),
                    sw4.down_proj.weight,
                    sw4.down_proj.scales,
                    e4["inds"],
                    e4["wts"],
                    shared=e4["sh"],
                    residual=e4["h_post"],
                    dedup=dd,
                )
            )(moe_dedup_rows(e4["inds"]) if L >= 5 else None)
        )

        t_head = bench_ms(lambda: gemv_lin(s["hidden"], model.lm_head), warm=3, iters=7)

        comp = {
            "KDA_gemv_x63": round(63 * t_kda_gemv_1, 2),
            "KDA_pre_gdn_x63": round(63 * t_kda_pre_gdn_1, 2),
            "MLA_x9": round(9 * t_mla_1, 2),
            "router_shared_x71": round(71 * t_router_shared_1, 2),
            "moe_dedup_gu_dc_x71": round(71 * t_moe_1, 2),
            "lm_head_x1": round(t_head, 2),
            "mean_uniq_exp": s["mean_uniq_exp"],
        }
        comp["sum_ms"] = round(
            comp["KDA_gemv_x63"]
            + comp["KDA_pre_gdn_x63"]
            + comp["MLA_x9"]
            + comp["router_shared_x71"]
            + comp["moe_dedup_gu_dc_x71"]
            + comp["lm_head_x1"],
            2,
        )
        out[f"L{L}"] = comp
        print(
            f"  L={L:2d} (uniq={comp['mean_uniq_exp']:4.1f}): KDA_gemv={comp['KDA_gemv_x63']:5.2f} "
            f"KDA_pre_gdn={comp['KDA_pre_gdn_x63']:5.2f} MLA={comp['MLA_x9']:4.2f} "
            f"rtr+sh={comp['router_shared_x71']:5.2f} MoE={comp['moe_dedup_gu_dc_x71']:5.2f} "
            f"head={comp['lm_head_x1']:4.2f} => sum={comp['sum_ms']:6.2f} ms",
            flush=True,
        )
    return out, saved_samples


def sweep_fp8_cfgs(model, saved_samples: dict) -> dict:
    """Sweep LINGSANTE_FP8_CFG candidates on KDA multi, o_proj, and shared_experts swiglu across B=5,8,9,10,12,16."""
    l4 = model.model.layers[4]
    attn4 = l4.attention
    se4 = l4.mlp.shared_experts

    w_qkvfg = [m.weight_fp8 for m in (attn4.q_proj, attn4.k_proj, attn4.v_proj, attn4.f_proj, attn4.g_proj)]
    s_qkvfg = [m.weight_scale_inv for m in (attn4.q_proj, attn4.k_proj, attn4.v_proj, attn4.f_proj, attn4.g_proj)]
    bf_qkvfg = [dequant_fp8_to_bf16(w, sc) for w, sc in zip(w_qkvfg, s_qkvfg)]
    bf_o = dequant_fp8_to_bf16(attn4.o_proj.weight_fp8, attn4.o_proj.weight_scale_inv)
    bf_sg = dequant_fp8_to_bf16(se4.gate_proj.weight_fp8, se4.gate_proj.weight_scale_inv)
    bf_su = dequant_fp8_to_bf16(se4.up_proj.weight_fp8, se4.up_proj.weight_scale_inv)
    mx.eval(*bf_qkvfg, bf_o, bf_sg, bf_su)

    cands = [(1, 8, 2), (1, 8, 1), (1, 4, 2), (1, 4, 1), (1, 16, 1), (2, 4, 2), (2, 4, 1), (2, 2, 2)]
    Bs = (5, 8, 9, 10, 12, 16)
    out = {}

    print("\n=== (c) FP8 GEMV Config Sweep (exact 0-bit-diff vs dequant BF16 ref) ===", flush=True)
    # Use L=16 slice cleanly for arbitrary B <= 16
    s16 = saved_samples[16]
    h_in_16 = s16["kda4"]["h_in"]
    ov_16 = s16["kda4"]["ov"]
    x_16 = s16["kda4"]["x"]
    mlp_in_16 = s16["moe4"]["mlp_in"]

    for B in Bs:
        x_m = h_in_16[:B]
        ov_b = ov_16[:B]
        res_b = x_16[:B]
        mlp_b = mlp_in_16[:B]

        ref_m = gemv_rows_group([x_m] * 5, bf_qkvfg)
        ref_o = gemv_rows_res(ov_b, bf_o, res_b)
        ref_sw = gemv_swiglu_rows(mlp_b, bf_sg, bf_su)
        mx.eval(*ref_m, ref_o, ref_sw)

        def eval_group(name_op, run_fn, refs, N_check):
            timings = {}
            for bsg, bm_m, tm in cands:
                if N_check % tm != 0 or bsg * bm_m > 32:
                    continue
                cfg = (bsg, (B + bsg - 1) // bsg, bm_m, tm)
                try:
                    got = run_fn(cfg)
                    got_l = got if isinstance(got, list) else [got]
                    mx.eval(*got_l)
                    diffs = sum(bit_diff_count(g, r) for g, r in zip(got_l, refs))
                    if diffs != 0:
                        timings[f"{bsg},{bm_m},{tm}"] = f"DIFF={diffs}"
                        continue
                    ms = pipe_ms(lambda c=cfg: run_fn(c), n_lo=2, n_hi=10, reps=3)
                    timings[f"{bsg},{bm_m},{tm}"] = ms
                except Exception as exc:
                    timings[f"{bsg},{bm_m},{tm}"] = f"ERR:{exc}"
            ok_items = [(k, v) for k, v in timings.items() if isinstance(v, (int, float))]
            best_k, best_v = min(ok_items, key=lambda kv: kv[1])
            return {"best": best_k, "best_ms": best_v, "all": timings}

        r_m = eval_group("kda_multi", lambda c: gemv_fp8_rows_group([x_m] * 5, w_qkvfg, s_qkvfg, cfg=c), ref_m, 8192)
        r_o = eval_group("o_proj", lambda c: gemv_fp8_rows_res(ov_b, attn4.o_proj.weight_fp8, attn4.o_proj.weight_scale_inv, res_b, cfg=c), [ref_o], 4096)
        r_sw = eval_group("shared_sw", lambda c: gemv_fp8_swiglu_rows(mlp_b, se4.gate_proj.weight_fp8, se4.gate_proj.weight_scale_inv, se4.up_proj.weight_fp8, se4.up_proj.weight_scale_inv, cfg=c), [ref_sw], se4.gate_proj.weight_fp8.shape[0])

        out[f"B{B}"] = {"kda_qkvfg": r_m, "kda_oproj": r_o, "shared_swiglu": r_sw}
        print(
            f"  B={B:2d}: qkvfg best={r_m['best']} ({r_m['best_ms']:.4f}ms, x63={63*r_m['best_ms']:.2f}) | "
            f"oproj best={r_o['best']} ({r_o['best_ms']:.4f}ms, x63={63*r_o['best_ms']:.2f}) | "
            f"sh_sw best={r_sw['best']} ({r_sw['best_ms']:.4f}ms, x71={71*r_sw['best_ms']:.2f})",
            flush=True,
        )
        print(f"    all qkvfg={r_m['all']}", flush=True)
        print(f"    all oproj={r_o['all']}", flush=True)
        print(f"    all sh_sw={r_sw['all']}", flush=True)
    return out


def sweep_flush(model, mtp, tok) -> dict:
    """Sweep LINGSANTE_VERIFY_FLUSH in (0, 4, 8, 12, 18, 72) at L in (5, 8, 12)."""
    flushes = (0, 4, 8, 12, 18, 72)
    Ls = (5, 8, 12)
    orig = vd._VERIFY_FLUSH_EVERY
    out = {}
    print("\n=== (d) LINGSANTE_VERIFY_FLUSH Sweep (C=1, L in 5, 8, 12) ===", flush=True)
    try:
        for L in Ls:
            vcache, inp = build_realistic_verify_input(model, mtp, tok, C=1, K=L - 1)
            row = {}
            for fl in flushes:
                vd._VERIFY_FLUSH_EVERY = fl

                def run_one():
                    lg, rb = fast_verify_forward(model, inp, vcache, return_hidden=False, return_rollback=True)
                    _trim_mla_rb(rb, L)
                    return lg

                row[str(fl)] = bench_ms(run_one, warm=2, iters=5)
            out[f"L{L}"] = row
            print(f"  L={L:2d}: {row}", flush=True)
    finally:
        vd._VERIFY_FLUSH_EVERY = orig
    return out


def sweep_coop_and_moe(model, mtp, tok) -> dict:
    """Sweep COOP=1 FP8 GEMV configs (vs COOP=0 default) and moe_down_combine_rows across B=8..16."""
    l4 = model.model.layers[4]
    attn4 = l4.attention
    se4 = l4.mlp.shared_experts
    sw4 = l4.mlp.switch_mlp

    print("\n=== Capturing realistic routed samples at L=8, 10, 12, 16 ===", flush=True)
    moe_samples = {}
    for B in (8, 10, 12, 16):
        moe_samples[B] = _capture_sample_acts(model, mtp, tok, B)
        print(f"  L={B:2d} captured (mean_uniq_exp={moe_samples[B]['mean_uniq_exp']})", flush=True)

    w_qkvfg = [m.weight_fp8 for m in (attn4.q_proj, attn4.k_proj, attn4.v_proj, attn4.f_proj, attn4.g_proj)]
    s_qkvfg = [m.weight_scale_inv for m in (attn4.q_proj, attn4.k_proj, attn4.v_proj, attn4.f_proj, attn4.g_proj)]
    bf_qkvfg = [dequant_fp8_to_bf16(w, sc) for w, sc in zip(w_qkvfg, s_qkvfg)]
    bf_o = dequant_fp8_to_bf16(attn4.o_proj.weight_fp8, attn4.o_proj.weight_scale_inv)
    bf_sg = dequant_fp8_to_bf16(se4.gate_proj.weight_fp8, se4.gate_proj.weight_scale_inv)
    bf_su = dequant_fp8_to_bf16(se4.up_proj.weight_fp8, se4.up_proj.weight_scale_inv)
    mx.eval(*bf_qkvfg, bf_o, bf_sg, bf_su)

    s16 = moe_samples[16]
    h_in_16 = s16["kda4"]["h_in"]
    ov_16 = s16["kda4"]["ov"]
    x_16 = s16["kda4"]["x"]
    mlp_in_16 = s16["moe4"]["mlp_in"]

    cands_lin = [(2, 4, 2), (2, 8, 2), (2, 2, 2), (2, 4, 4), (2, 2, 4), (4, 2, 4), (4, 4, 4)]
    cands_sw = [(2, 4, 1), (2, 8, 1), (2, 4, 2), (2, 2, 2), (2, 2, 1), (4, 2, 2)]
    n_sh = se4.gate_proj.weight_fp8.shape[0]

    fp8_res = {}
    print("\n=== COOP=1 vs COOP=0 FP8 GEMV Sweep (0-bit-diff asserted vs dequant BF16 ref) ===", flush=True)
    for B in (5, 8, 9, 10, 12, 16):
        x_m, ov_b, res_b, mlp_b = h_in_16[:B], ov_16[:B], x_16[:B], mlp_in_16[:B]
        ref_m = gemv_rows_group([x_m] * 5, bf_qkvfg)
        ref_o = gemv_rows_res(ov_b, bf_o, res_b)
        ref_sw = gemv_swiglu_rows(mlp_b, bf_sg, bf_su)
        mx.eval(*ref_m, ref_o, ref_sw)

        def eval_op(run_def, run_cfg, refs, N_check, cands, is_sw, n_lo, n_hi):
            got_def = run_def()
            def_l = got_def if isinstance(got_def, list) else [got_def]
            mx.eval(*def_l)
            diffs_def = sum(bit_diff_count(g, r) for g, r in zip(def_l, refs))
            assert diffs_def == 0, f"Default COOP=0 bit diffs={diffs_def}"
            base_ms = pipe_ms(run_def, n_lo=n_lo, n_hi=n_hi, reps=5)

            coop_timings = {}
            for bsg, bm_m, tm in cands:
                if N_check % tm != 0 or bsg * bm_m > 32 or not _can_coop(bsg, bm_m, tm, is_sw):
                    continue
                cfg = (bsg, (B + bsg - 1) // bsg, bm_m, tm)
                got = run_cfg(cfg, True)
                got_l = got if isinstance(got, list) else [got]
                mx.eval(*got_l)
                diffs = sum(bit_diff_count(g, r) for g, r in zip(got_l, refs))
                assert diffs == 0, f"COOP=1 B={B} cfg={cfg} bit diffs={diffs}"
                ms = pipe_ms(lambda c=cfg: run_cfg(c, True), n_lo=n_lo, n_hi=n_hi, reps=5)
                coop_timings[f"{bsg},{bm_m},{tm}"] = ms
            best_k, best_v = min(coop_timings.items(), key=lambda kv: kv[1])
            return {"default_coop0_ms": base_ms, "best_coop1": best_k, "best_coop1_ms": best_v, "all_coop1": coop_timings}

        r_m = eval_op(
            lambda: gemv_fp8_rows_group([x_m] * 5, w_qkvfg, s_qkvfg, coop=False),
            lambda c, cp: gemv_fp8_rows_group([x_m] * 5, w_qkvfg, s_qkvfg, cfg=c, coop=cp),
            ref_m,
            8192,
            cands_lin,
            False,
            4,
            24,
        )
        r_o = eval_op(
            lambda: gemv_fp8_rows_res(ov_b, attn4.o_proj.weight_fp8, attn4.o_proj.weight_scale_inv, res_b, coop=False),
            lambda c, cp: gemv_fp8_rows_res(ov_b, attn4.o_proj.weight_fp8, attn4.o_proj.weight_scale_inv, res_b, cfg=c, coop=cp),
            [ref_o],
            4096,
            cands_lin,
            False,
            10,
            60,
        )
        r_sw = eval_op(
            lambda: gemv_fp8_swiglu_rows(mlp_b, se4.gate_proj.weight_fp8, se4.gate_proj.weight_scale_inv, se4.up_proj.weight_fp8, se4.up_proj.weight_scale_inv, coop=False),
            lambda c, cp: gemv_fp8_swiglu_rows(mlp_b, se4.gate_proj.weight_fp8, se4.gate_proj.weight_scale_inv, se4.up_proj.weight_fp8, se4.up_proj.weight_scale_inv, cfg=c, coop=cp),
            [ref_sw],
            n_sh,
            cands_sw,
            True,
            10,
            60,
        )
        sav_q = 63 * (r_m["default_coop0_ms"] - r_m["best_coop1_ms"])
        sav_o = 63 * (r_o["default_coop0_ms"] - r_o["best_coop1_ms"])
        sav_s = 71 * (r_sw["default_coop0_ms"] - r_sw["best_coop1_ms"])
        fp8_res[f"B{B}"] = {"kda_qkvfg": r_m, "kda_oproj": r_o, "shared_swiglu": r_sw}
        print(
            f"  B={B:2d}: qkvfg def={r_m['default_coop0_ms']:.4f} vs coop[{r_m['best_coop1']}]={r_m['best_coop1_ms']:.4f} (x63 {sav_q:+.2f}ms) | "
            f"oproj def={r_o['default_coop0_ms']:.4f} vs coop[{r_o['best_coop1']}]={r_o['best_coop1_ms']:.4f} (x63 {sav_o:+.2f}ms) | "
            f"sh_sw def={r_sw['default_coop0_ms']:.4f} vs coop[{r_sw['best_coop1']}]={r_sw['best_coop1_ms']:.4f} (x71 {sav_s:+.2f}ms)",
            flush=True,
        )
        print(f"    coop qkvfg={r_m['all_coop1']}", flush=True)
        print(f"    coop oproj={r_o['all_coop1']}", flush=True)
        print(f"    coop sh_sw={r_sw['all_coop1']}", flush=True)

    moe_res = {}
    print("\n=== moe_down_combine_rows Sweep (B=8, 10, 12, 16; 0-bit-diff asserted) ===", flush=True)
    for B in (8, 10, 12, 16):
        e4 = moe_samples[B]["moe4"]
        dd = moe_dedup_rows(e4["inds"])
        mx.eval(dd)
        ref_dc = moe_down_combine_rows(
            e4["h_mid"], sw4.down_proj.weight, sw4.down_proj.scales, e4["inds"], e4["wts"], shared=e4["sh"], residual=e4["h_post"], dedup=dd
        )
        mx.eval(ref_dc)
        def_ms = pipe_ms(
            lambda: moe_down_combine_rows(
                e4["h_mid"], sw4.down_proj.weight, sw4.down_proj.scales, e4["inds"], e4["wts"], shared=e4["sh"], residual=e4["h_post"], dedup=dd
            ),
            n_lo=4,
            n_hi=24,
            reps=5,
        )
        timings = {}
        for tm in (2, 4, 8):
            for nsg in (2, 4, 8):
                for hdq in (0, 1, 2):
                    tag = f"tm={tm},nsg={nsg},hdq={hdq}"
                    try:
                        got = moe_down_combine_rows(
                            e4["h_mid"],
                            sw4.down_proj.weight,
                            sw4.down_proj.scales,
                            e4["inds"],
                            e4["wts"],
                            shared=e4["sh"],
                            residual=e4["h_post"],
                            dedup=dd,
                            tmax=tm,
                            nsg=nsg,
                            hoist_dq=hdq,
                        )
                        mx.eval(got)
                        diffs = bit_diff_count(got, ref_dc)
                        assert diffs == 0, f"moe_dc B={B} {tag} bit diffs={diffs}"
                        ms = pipe_ms(
                            lambda t=tm, n=nsg, hq=hdq: moe_down_combine_rows(
                                e4["h_mid"],
                                sw4.down_proj.weight,
                                sw4.down_proj.scales,
                                e4["inds"],
                                e4["wts"],
                                shared=e4["sh"],
                                residual=e4["h_post"],
                                dedup=dd,
                                tmax=t,
                                nsg=n,
                                hoist_dq=hq,
                            ),
                            n_lo=4,
                            n_hi=24,
                            reps=5,
                        )
                        timings[tag] = ms
                    except Exception as exc:
                        timings[tag] = f"ERR:{exc}"
        ok = [(k, v) for k, v in timings.items() if isinstance(v, (int, float))]
        best_k, best_v = min(ok, key=lambda kv: kv[1])
        sav = 71 * (def_ms - best_v)
        moe_res[f"B{B}"] = {"default_ms": def_ms, "best": best_k, "best_ms": best_v, "x71_save_ms": round(sav, 2), "all": timings}
        print(f"  B={B:2d}: def={def_ms:.4f}ms | best[{best_k}]={best_v:.4f}ms (x71 save={sav:+.2f}ms)", flush=True)
        print(f"    all={timings}", flush=True)

    return {"fp8_coop_sweep": fp8_res, "moe_dc_sweep": moe_res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--e2e-only", action="store_true")
    ap.add_argument("--sweep-coop", action="store_true")
    ap.add_argument("--out", default=os.path.expanduser("~/science/runs/profile_verify_c1.json"))
    a = ap.parse_args()

    maybe_set_recommended_wired_limit()
    t0 = time.perf_counter()
    model, tok = load(a.model)
    mx.eval(model.parameters())
    mtp = BailingMTPLayer.from_target(model, a.model, with_q4_head=True)
    print(f"Loaded target (fp8={getattr(model, '_lingsante_fp8', False)}) + MTP in {time.perf_counter()-t0:.1f}s ({mx.get_active_memory()/1e9:.1f} GB active)", flush=True)

    if a.sweep_coop:
        payload = sweep_coop_and_moe(model, mtp, tok)
    else:
        payload = {"e2e_ms": measure_e2e_ledger(model, mtp, tok)}
        if not a.e2e_only:
            comps, samples = measure_components(model, mtp, tok)
            payload["components_ms"] = comps
            payload["fp8_sweep"] = sweep_fp8_cfgs(model, samples)
            payload["flush_sweep_ms"] = sweep_flush(model, mtp, tok)

    out_path = Path(a.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=1))
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
