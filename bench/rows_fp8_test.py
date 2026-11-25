"""Bit-exactness verification and pipelined GB/s microbenchmark for FP8 linear/MultiLinear kernels."""
import argparse
import os
import statistics
import sys
import time
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(_root / "vendor"), str(_root)]

import mlx_lm  # noqa: E402,F401
import mlx.core as mx  # noqa: E402
from mlx_lm.layers.common import swiglu  # noqa: E402
from mlx_lm.models.bailing_moe_v3.rows.gemv import (  # noqa: E402
    gemv_rows,
    gemv_rows_multi,
    gemv_rows_res,
    gemv_swiglu_rows,
    router_gemv_rows,
)
from mlx_lm.models.bailing_moe_v3.rows.gemv_fp8 import (  # noqa: E402
    dequant_fp8_multilinear_to_bf16,
    dequant_fp8_to_bf16,
    gemv_fp8_rows,
    gemv_fp8_rows_group,
    gemv_fp8_rows_res,
    gemv_fp8_swiglu_rows,
    gemv_fp8_with_bf16_tail_rows,
    gemv_kda_qkvfgb_merged_rows,
    gemv_lin,
    gemv_lin_multi,
    gemv_lin_res,
    gemv_lin_swiglu,
    gemv_router_se_swiglu_rows,
    pack_fp8_k_major,
    pack_qkvfg_fp8,
)
from mlx_lm.models.bailing_moe_v3.rows.mla import mla_multilinear_fp8_rows, mla_multilinear_lin, mla_multilinear_rows  # noqa: E402

from bench._layer_loader import bit_diff_count, load_head, load_layer  # noqa: E402


def check(name, got, ref):
    d = bit_diff_count(got, ref)
    print(f"  {name:52s} {str(got.shape):18s} {'PASS' if d == 0 else f'FAIL ({d}/{ref.size})'}", flush=True)
    return d == 0


def _flat(x):
    return list(x) if isinstance(x, (list, tuple)) else [x]


def pipe_ms(fn, n_lo=4, n_hi=20, reps=4):
    mx.eval(*[a for _ in range(3) for a in _flat(fn())])
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
    return statistics.median(dts)


def test_weights_and_prefill(l0_bf, l0_fp8, l4_bf, l4_fp8, l7_bf, l7_fp8):
    print("\n=== 1. FP8 Dequant & Prefill (L=32, 128) Bit-Exactness ===", flush=True)
    ok = True
    pairs_2d = [
        ("L0.mlp.gate", l0_bf.mlp.gate_proj, l0_fp8.mlp.gate_proj),
        ("L0.mlp.up", l0_bf.mlp.up_proj, l0_fp8.mlp.up_proj),
        ("L0.mlp.down", l0_bf.mlp.down_proj, l0_fp8.mlp.down_proj),
        ("L4.attn.q", l4_bf.attention.q_proj, l4_fp8.attention.q_proj),
        ("L4.attn.k", l4_bf.attention.k_proj, l4_fp8.attention.k_proj),
        ("L4.attn.v", l4_bf.attention.v_proj, l4_fp8.attention.v_proj),
        ("L4.attn.f", l4_bf.attention.f_proj, l4_fp8.attention.f_proj),
        ("L4.attn.g", l4_bf.attention.g_proj, l4_fp8.attention.g_proj),
        ("L4.attn.o", l4_bf.attention.o_proj, l4_fp8.attention.o_proj),
        ("L4.shared.gate", l4_bf.mlp.shared_experts.gate_proj, l4_fp8.mlp.shared_experts.gate_proj),
        ("L4.shared.up", l4_bf.mlp.shared_experts.up_proj, l4_fp8.mlp.shared_experts.up_proj),
        ("L4.shared.down", l4_bf.mlp.shared_experts.down_proj, l4_fp8.mlp.shared_experts.down_proj),
        ("L7.attn.q_a", l7_bf.attention.q_a_proj, l7_fp8.attention.q_a_proj),
        ("L7.attn.q_b", l7_bf.attention.q_b_proj, l7_fp8.attention.q_b_proj),
        ("L7.attn.kv_a", l7_bf.attention.kv_a_proj_with_mqa, l7_fp8.attention.kv_a_proj_with_mqa),
        ("L7.attn.dense", l7_bf.attention.dense, l7_fp8.attention.dense),
    ]
    for nm, mb, mf in pairs_2d:
        assert hasattr(mf, "weight_fp8") and not hasattr(mb, "weight_fp8"), nm
        ok &= check(f"dequant.{nm}", dequant_fp8_to_bf16(mf.weight_fp8, mf.weight_scale_inv), mb.weight)

    ok &= check(
        "dequant.L7.embed_q",
        dequant_fp8_multilinear_to_bf16(l7_fp8.attention.embed_q.weight_fp8, l7_fp8.attention.embed_q.weight_scale_inv, True),
        l7_bf.attention.embed_q.weight,
    )
    ok &= check(
        "dequant.L7.unembed_out",
        dequant_fp8_multilinear_to_bf16(l7_fp8.attention.unembed_out.weight_fp8, l7_fp8.attention.unembed_out.weight_scale_inv, False),
        l7_bf.attention.unembed_out.weight,
    )

    mx.random.seed(5)
    for T in (32, 128):
        x4 = (mx.random.normal((1, T, 4096)) * 0.25).astype(mx.bfloat16)
        x8 = (mx.random.normal((1, T, 8192)) * 0.25).astype(mx.bfloat16)
        xq = (mx.random.normal((1, 64, T, 128)) * 0.25).astype(mx.bfloat16)
        xk = (mx.random.normal((1, 64, T, 512)) * 0.25).astype(mx.bfloat16)
        mx.eval(x4, x8, xq, xk)
        ok &= check(f"prefill T={T} L4.q_proj", l4_fp8.attention.q_proj(x4), l4_bf.attention.q_proj(x4))
        ok &= check(f"prefill T={T} L4.o_proj", l4_fp8.attention.o_proj(x8), l4_bf.attention.o_proj(x8))
        ok &= check(f"prefill T={T} L7.kv_a", l7_fp8.attention.kv_a_proj_with_mqa(x4), l7_bf.attention.kv_a_proj_with_mqa(x4))
        ok &= check(f"prefill T={T} L7.embed_q(T=True)", l7_fp8.attention.embed_q(xq, transpose=True), l7_bf.attention.embed_q(xq, transpose=True))
        ok &= check(f"prefill T={T} L7.embed_q(T=False)", l7_fp8.attention.embed_q(xk[:, :1], transpose=False), l7_bf.attention.embed_q(xk[:, :1], transpose=False))
        ok &= check(f"prefill T={T} L7.unembed_out", l7_fp8.attention.unembed_out(xk), l7_bf.attention.unembed_out(xk))
    return ok


def test_gemv_exactness(l0_bf, l0_fp8, l4_bf, l4_fp8, l7_bf, l7_fp8, lm_head):
    print("\n=== 2. FP8 GEMV & MultiLinear B in (1,2,4,5,8,12,13,16,20,24) ===", flush=True)
    ok = True
    mx.random.seed(19)
    a4b, a4f = l4_bf.attention, l4_fp8.attention
    a7b, a7f = l7_bf.attention, l7_fp8.attention
    se_b, se_f = l4_bf.mlp.shared_experts, l4_fp8.mlp.shared_experts
    m0b, m0f = l0_bf.mlp, l0_fp8.mlp

    kda_wb = [a4b.q_proj.weight, a4b.k_proj.weight, a4b.v_proj.weight, a4b.f_proj.weight, a4b.g_proj.weight, a4b.b_proj.weight]
    kda_pf = [a4f.q_proj, a4f.k_proj, a4f.v_proj, a4f.f_proj, a4f.g_proj, a4f.b_proj]
    mla_wb = [a7b.q_a_proj.weight, a7b.kv_a_proj_with_mqa.weight, a7b.g_proj.weight]
    mla_pf = [a7f.q_a_proj, a7f.kv_a_proj_with_mqa, a7f.g_proj]

    for B in (1, 2, 4, 5, 8, 12, 13, 16, 20, 24):
        x4 = (mx.random.normal((B, 4096)) * 0.25).astype(mx.bfloat16)
        x8 = (mx.random.normal((B, 8192)) * 0.25).astype(mx.bfloat16)
        xd = (mx.random.normal((B, 10240)) * 0.25).astype(mx.bfloat16)
        xs = (mx.random.normal((B, 1280)) * 0.25).astype(mx.bfloat16)
        qa = (mx.random.normal((B, 1536)) * 0.25).astype(mx.bfloat16)
        qn = (mx.random.normal((B, 64, 1, 128)) * 0.25).astype(mx.bfloat16)
        ao = (mx.random.normal((B, 64, 1, 512)) * 0.25).astype(mx.bfloat16)
        gr = (mx.random.normal((B, 64)) * 0.5).astype(mx.bfloat16)
        mx.eval(x4, x8, xd, xs, qa, qn, ao, gr)

        ref_kda = gemv_rows_multi(x4, kda_wb)
        got_kda = gemv_lin_multi(x4, kda_pf)
        for nm, g, r in zip(("q", "k", "v", "f", "g", "b"), got_kda, ref_kda):
            ok &= check(f"kda.{nm} B={B}", g, r)

        ok &= check(f"kda.o_proj_res B={B}", gemv_lin_res(x8, a4f.o_proj, x4), gemv_rows_res(x8, a4b.o_proj.weight, x4))

        ref_mla = gemv_rows_multi(x4, mla_wb)
        got_mla = gemv_lin_multi(x4, mla_pf)
        for nm, g, r in zip(("q_a", "kv_a", "g"), got_mla, ref_mla):
            ok &= check(f"mla.{nm} B={B}", g, r)

        ok &= check(f"mla.q_b B={B}", gemv_lin(qa, a7f.q_b_proj), gemv_rows(qa, a7b.q_b_proj.weight))
        ok &= check(f"mla.embed_q B={B}", mla_multilinear_lin(qn, a7f.embed_q), mla_multilinear_rows(qn, a7b.embed_q.weight))
        ok &= check(
            f"mla.unembed_gate B={B}",
            mla_multilinear_lin(ao, a7f.unembed_out, g_raw=gr),
            mla_multilinear_rows(ao, a7b.unembed_out.weight, g_raw=gr),
        )
        ok &= check(f"mla.dense_res B={B}", gemv_lin_res(x8, a7f.dense, x4), gemv_rows_res(x8, a7b.dense.weight, x4))

        ok &= check(
            f"shared.swiglu B={B}",
            gemv_lin_swiglu(x4, se_f.gate_proj, se_f.up_proj),
            gemv_swiglu_rows(x4, se_b.gate_proj.weight, se_b.up_proj.weight),
        )
        ok &= check(f"shared.down B={B}", gemv_lin(xs, se_f.down_proj), gemv_rows(xs, se_b.down_proj.weight))

        ok &= check(
            f"dense.swiglu B={B}",
            gemv_lin_swiglu(x4, m0f.gate_proj, m0f.up_proj),
            gemv_swiglu_rows(x4, m0b.gate_proj.weight, m0b.up_proj.weight),
        )
        ok &= check(f"dense.down_res B={B}", gemv_lin_res(xd, m0f.down_proj, x4), gemv_rows_res(xd, m0b.down_proj.weight, x4))

        # Check BF16 lm_head against B=1 MLX ref (verifies BSG=1 + tg_acc fix at B>=13)
        ref_lh = mx.concatenate([lm_head(x4[b : b + 1]) for b in range(B)], axis=0)
        ok &= check(f"lm_head(bf16) B={B}", gemv_lin(x4, lm_head), ref_lh)

        # Check router_gemv_rows (ntg=256) and fused gemv_router_se_swiglu_rows
        ref_rtr = mx.concatenate(
            [x4[b : b + 1].astype(mx.float32) @ l4_bf.mlp.gate.weight.astype(mx.float32).T for b in range(B)], axis=0
        )
        ok &= check(f"router_gemv(ntg256) B={B}", router_gemv_rows(x4, l4_bf.mlp.gate.weight), ref_rtr)
        f_rtr, f_se = gemv_router_se_swiglu_rows(x4, l4_fp8.mlp.gate.weight, se_f.gate_proj, se_f.up_proj)
        ref_se = gemv_swiglu_rows(x4, se_b.gate_proj.weight, se_b.up_proj.weight)
        ok &= check(f"fused.router B={B}", f_rtr, ref_rtr)
        ok &= check(f"fused.se_swiglu B={B}", f_se, ref_se)
    return ok


def test_part_ss_and_repack(l0_bf, l0_fp8, l4_bf, l4_fp8, l7_bf, l7_fp8):
    print("\n=== 2b. T6 part_ss Parity (vs mx.fast.rms_norm) & T3 Repacking Exactness ===", flush=True)
    ok = True
    mx.random.seed(41)
    a4f, a7f = l4_fp8.attention, l7_fp8.attention
    se_f = l4_fp8.mlp.shared_experts
    m0f = l0_fp8.mlp
    w_norm4 = l4_fp8.post_attention_layernorm.weight
    eps4 = l4_fp8.post_attention_layernorm.eps
    w_norm_qa = a7f.q_a_layernorm.weight
    eps_qa = a7f.q_a_layernorm.eps

    # T3 K-major & merged qkvfg packing exactness
    qkv_mods = [a4f.q_proj, a4f.k_proj, a4f.v_proj, a4f.f_proj, a4f.g_proj]
    w_km_o = pack_fp8_k_major(a4f.o_proj.weight_fp8)
    w_qkv_cat, s_qkv_cat = pack_qkvfg_fp8(qkv_mods, k_major=False)
    w_qkv_km, s_qkv_km = pack_qkvfg_fp8(qkv_mods, k_major=True)
    mx.eval(w_km_o, w_qkv_cat, s_qkv_cat, w_qkv_km, s_qkv_km)
    ok &= check("T3.dequant.o_proj_km", dequant_fp8_to_bf16(w_km_o, a4f.o_proj.weight_scale_inv, k_major=True), l4_bf.attention.o_proj.weight)

    for B in (1, 2, 4, 5, 6, 8, 9, 12, 16):
        x4 = (mx.random.normal((B, 4096)) * 0.25).astype(mx.bfloat16)
        x8 = (mx.random.normal((B, 8192)) * 0.25).astype(mx.bfloat16)
        xd = (mx.random.normal((B, 10240)) * 0.25).astype(mx.bfloat16)
        mx.eval(x4, x8, xd)

        # 1) FP8 o_proj_res -> part_ss -> fused router + se_swiglu (N=4096, TM=2 -> P=512 or P=1024)
        h_post, pss = gemv_lin_res(x8, a4f.o_proj, x4, return_part_ss=True)
        h_ref = gemv_lin_res(x8, a4f.o_proj, x4)
        ok &= check(f"T6.o_proj_res.h B={B}", h_post, h_ref)
        mlp_in_ref = mx.fast.rms_norm(h_ref, w_norm4, eps4)
        ref_rtr, ref_se = gemv_router_se_swiglu_rows(mlp_in_ref, l4_fp8.mlp.gate.weight, se_f.gate_proj, se_f.up_proj)
        got_rtr, got_se = gemv_router_se_swiglu_rows(
            h_post, l4_fp8.mlp.gate.weight, se_f.gate_proj, se_f.up_proj, part_ss=pss, norm_w=w_norm4, norm_eps=eps4
        )
        ok &= check(f"T6.post_attn->router B={B}", got_rtr, ref_rtr)
        ok &= check(f"T6.post_attn->se_sw B={B}", got_se, ref_se)

        # 2) FP8 q_a_proj (N=1536) -> part_ss -> q_b_proj
        qa_out, pss_qa = gemv_lin(x4, a7f.q_a_proj, return_part_ss=True)
        qa_norm_ref = mx.fast.rms_norm(qa_out, w_norm_qa, eps_qa)
        ok &= check(
            f"T6.q_a->q_b B={B}",
            gemv_lin(qa_out, a7f.q_b_proj, part_ss=pss_qa, norm_w=w_norm_qa, norm_eps=eps_qa),
            gemv_lin(qa_norm_ref, a7f.q_b_proj),
        )

        # 3) FP8 dense.down_res -> part_ss -> KDA qkvfgb
        x_dn, pss_dn = gemv_lin_res(xd, m0f.down_proj, x4, return_part_ss=True)
        x_dn_norm = mx.fast.rms_norm(x_dn, l4_fp8.input_layernorm.weight, l4_fp8.input_layernorm.eps)
        kda_pf = [a4f.q_proj, a4f.k_proj, a4f.v_proj, a4f.f_proj, a4f.g_proj, a4f.b_proj]
        ref_kda = gemv_lin_multi(x_dn_norm, kda_pf)
        got_kda = gemv_lin_multi(
            x_dn, kda_pf, part_ss=pss_dn, norm_w=l4_fp8.input_layernorm.weight, norm_eps=l4_fp8.input_layernorm.eps
        )
        for nm, g, r in zip(("q", "k", "v", "f", "g", "b"), got_kda, ref_kda):
            ok &= check(f"T6.dn->kda.{nm} B={B}", g, r)

        # 4) BF16 producer return_part_ss (TM=4 at B<=4, TM=2 at B>=5) -> router_gemv_rows
        h_bf, pss_bf = gemv_rows_res(x8, l4_bf.attention.o_proj.weight, x4, return_part_ss=True)
        ok &= check(
            f"T6.bf16->router B={B}",
            router_gemv_rows(h_bf, l4_bf.mlp.gate.weight, part_ss=pss_bf, norm_w=w_norm4, norm_eps=eps4),
            router_gemv_rows(mx.fast.rms_norm(h_bf, w_norm4, eps4), l4_bf.mlp.gate.weight),
        )

        # 5) T3 K-major & merged qkvfg exactness
        ok &= check(
            f"T3.o_proj_km B={B}",
            gemv_fp8_rows_res(x8, w_km_o, a4f.o_proj.weight_scale_inv, x4, k_major=True),
            h_ref,
        )
        ref_qkvb = gemv_lin_multi(x4, kda_pf)
        m_outs = gemv_kda_qkvfgb_merged_rows(x4, w_qkv_cat, s_qkv_cat, a4f.b_proj.weight)
        for nm, g, r in zip(("q", "k", "v", "f", "g", "b"), m_outs, ref_qkvb):
            ok &= check(f"T3.kda_merged.{nm} B={B}", g, r)
        m_pss_outs = gemv_kda_qkvfgb_merged_rows(
            x_dn, w_qkv_cat, s_qkv_cat, a4f.b_proj.weight,
            part_ss=pss_dn, norm_w=l4_fp8.input_layernorm.weight, norm_eps=l4_fp8.input_layernorm.eps
        )
        for nm, g, r in zip(("q", "k", "v", "f", "g", "b"), m_pss_outs, ref_kda):
            ok &= check(f"T3+T6.kda_merged.{nm} B={B}", g, r)
        km_out, km_b = gemv_fp8_with_bf16_tail_rows(x4, [w_qkv_km], [s_qkv_km], a4f.b_proj.weight, k_major=True)
        ref_cat = mx.concatenate(ref_qkvb[:5], axis=-1)
        ok &= check(f"T3.qkv_km B={B}", km_out, ref_cat)
        ok &= check(f"T3.qkv_km_b B={B}", km_b, ref_qkvb[5])
    return ok


def bench_and_tune(l0_bf, l0_fp8, l4_bf, l4_fp8, l7_bf, l7_fp8, lm_head):
    print("\n=== 3. Pipelined Kernel Microbenchmark (BF16 vs FP8, GB/s) & Config Tuning ===", flush=True)
    a4b, a4f = l4_bf.attention, l4_fp8.attention
    m0b, m0f = l0_bf.mlp, l0_fp8.mlp
    se_b, se_f = l4_bf.mlp.shared_experts, l4_fp8.mlp.shared_experts

    w_qkv_bf = [a4b.q_proj.weight, a4b.k_proj.weight, a4b.v_proj.weight, a4b.f_proj.weight, a4b.g_proj.weight]
    w_qkv_f8 = [a4f.q_proj.weight_fp8, a4f.k_proj.weight_fp8, a4f.v_proj.weight_fp8, a4f.f_proj.weight_fp8, a4f.g_proj.weight_fp8]
    s_qkv_f8 = [a4f.q_proj.weight_scale_inv, a4f.k_proj.weight_scale_inv, a4f.v_proj.weight_scale_inv, a4f.f_proj.weight_scale_inv, a4f.g_proj.weight_scale_inv]
    kda_pf = [a4f.q_proj, a4f.k_proj, a4f.v_proj, a4f.f_proj, a4f.g_proj, a4f.b_proj]
    w_qkv_cat, s_qkv_cat = pack_qkvfg_fp8(kda_pf[:5], k_major=False)
    w_qkv_km, s_qkv_km = pack_qkvfg_fp8(kda_pf[:5], k_major=True)
    w_o_km = pack_fp8_k_major(a4f.o_proj.weight_fp8)
    mx.eval(w_qkv_cat, s_qkv_cat, w_qkv_km, s_qkv_km, w_o_km)
    bytes_qkv_f8 = 5 * 8192 * 4096

    for B in (1, 5, 6, 8, 9, 12, 16, 20, 24):
        x4 = (mx.random.normal((B, 4096)) * 0.25).astype(mx.bfloat16)
        x8 = (mx.random.normal((B, 8192)) * 0.25).astype(mx.bfloat16)
        qa = (mx.random.normal((B, 1536)) * 0.25).astype(mx.bfloat16)
        mx.eval(x4, x8, qa)

        t_qkv_f8 = pipe_ms(lambda: gemv_fp8_rows_group([x4] * 5, w_qkv_f8, s_qkv_f8))
        t_qkvb = pipe_ms(lambda: gemv_lin_multi(x4, kda_pf))
        t_qkv_m = pipe_ms(lambda: gemv_kda_qkvfgb_merged_rows(x4, w_qkv_cat, s_qkv_cat, a4f.b_proj.weight))
        gb_s = (bytes_qkv_f8 / 1e9) / (t_qkv_f8 / 1e3)

        t_op_f8 = pipe_ms(lambda: gemv_lin_res(x8, a4f.o_proj, x4))
        t_op_pss = pipe_ms(lambda: gemv_lin_res(x8, a4f.o_proj, x4, return_part_ss=True))
        t_den_f8 = pipe_ms(lambda: gemv_lin_swiglu(x4, m0f.gate_proj, m0f.up_proj))
        t_sh_f8 = pipe_ms(lambda: gemv_lin_swiglu(x4, se_f.gate_proj, se_f.up_proj))
        t_rtr = pipe_ms(lambda: router_gemv_rows(x4, l4_fp8.mlp.gate.weight))
        t_rtr_se = pipe_ms(lambda: gemv_router_se_swiglu_rows(x4, l4_fp8.mlp.gate.weight, se_f.gate_proj, se_f.up_proj))
        t_lh = pipe_ms(lambda: gemv_lin(x4, lm_head))

        print(
            f"  B={B:2d}: qkv5={t_qkv_f8*1000:4.0f}us ({gb_s:5.1f}GB/s, x63={t_qkv_f8*63:4.1f}ms) "
            f"qkvb={t_qkvb*1000:4.0f}us mrg={t_qkv_m*1000:4.0f}us | "
            f"op={t_op_f8*1000:3.0f}us(pss={t_op_pss*1000:3.0f}) | denSw={t_den_f8*1000:3.0f}us | "
            f"rtr+shSw={t_rtr*1000:2.0f}+{t_sh_f8*1000:2.0f}->fus={t_rtr_se*1000:2.0f}us | lm_head={t_lh*1000:4.0f}us",
            flush=True,
        )

    print("  --- Candidate _fp8_cfg tuning on qkv5, den_sw, sh_sw (E4M3_LUT) ---", flush=True)
    for B, cands_qkv, cands_sw in [
        (5, [(1, 8, 2), (1, 4, 2)], [(1, 4, 1), (1, 4, 2), (2, 4, 2)]),
        (8, [(1, 8, 2), (1, 4, 2), (1, 8, 1)], [(1, 4, 1), (1, 8, 1), (2, 4, 2), (2, 2, 2)]),
        (12, [(2, 2, 2), (2, 4, 2), (2, 4, 1), (1, 8, 1)], [(2, 4, 1), (2, 2, 1), (3, 2, 2), (4, 2, 2)]),
        (16, [(2, 2, 2), (2, 4, 2), (2, 4, 1), (2, 2, 1), (4, 2, 2)], [(2, 4, 1), (2, 2, 1), (4, 2, 1), (4, 2, 2)]),
        (20, [(2, 4, 1), (3, 2, 1), (3, 2, 2), (4, 2, 2)], [(4, 2, 1), (3, 2, 1), (5, 2, 2), (4, 2, 2)]),
        (24, [(3, 2, 1), (4, 2, 1), (3, 2, 2), (4, 2, 2)], [(4, 2, 1), (3, 2, 1), (5, 2, 2), (6, 1, 2)]),
    ]:
        x4 = (mx.random.normal((B, 4096)) * 0.25).astype(mx.bfloat16)
        mx.eval(x4)
        q_str = []
        for bsg, bm_m, tm in cands_qkv:
            cfg = (bsg, (B + bsg - 1) // bsg, bm_m, tm)
            t = pipe_ms(lambda cfg=cfg: gemv_fp8_rows_group([x4] * 5, w_qkv_f8, s_qkv_f8, cfg=cfg))
            q_str.append(f"({bsg},{bm_m},{tm})={t*1000:.0f}us")
        s_str = []
        sh_str = []
        for bsg, bm_m, tm in cands_sw:
            cfg = (bsg, (B + bsg - 1) // bsg, bm_m, tm)
            t = pipe_ms(
                lambda cfg=cfg: gemv_fp8_swiglu_rows(
                    x4, m0f.gate_proj.weight_fp8, m0f.gate_proj.weight_scale_inv,
                    m0f.up_proj.weight_fp8, m0f.up_proj.weight_scale_inv, cfg=cfg
                )
            )
            tsh = pipe_ms(
                lambda cfg=cfg: gemv_fp8_swiglu_rows(
                    x4, se_f.gate_proj.weight_fp8, se_f.gate_proj.weight_scale_inv,
                    se_f.up_proj.weight_fp8, se_f.up_proj.weight_scale_inv, cfg=cfg
                )
            )
            s_str.append(f"({bsg},{bm_m},{tm})={t*1000:.0f}us")
            sh_str.append(f"({bsg},{bm_m},{tm})={tsh*1000:.0f}us")
        print(f"    B={B:2d} qkv5 : {' '.join(q_str)}", flush=True)
        print(f"    B={B:2d} denSw: {' '.join(s_str)}", flush=True)
        print(f"    B={B:2d} shSw : {' '.join(sh_str)}", flush=True)


def sweep_b(l0_bf, l0_fp8, l4_bf, l4_fp8, l7_bf, l7_fp8, lm_head):
    print("\n=== Sweep B in (8, 9, 10, 12, 16, 20, 24) for FP8, MLA MultiLinear, and BF16 lm_head ===", flush=True)
    a4b, a4f = l4_bf.attention, l4_fp8.attention
    a7b, a7f = l7_bf.attention, l7_fp8.attention
    se_b, se_f = l4_bf.mlp.shared_experts, l4_fp8.mlp.shared_experts
    m0b, m0f = l0_bf.mlp, l0_fp8.mlp
    w_qkv_bf = [a4b.q_proj.weight, a4b.k_proj.weight, a4b.v_proj.weight, a4b.f_proj.weight, a4b.g_proj.weight]
    w_qkv_f8 = [a4f.q_proj.weight_fp8, a4f.k_proj.weight_fp8, a4f.v_proj.weight_fp8, a4f.f_proj.weight_fp8, a4f.g_proj.weight_fp8]
    s_qkv_f8 = [a4f.q_proj.weight_scale_inv, a4f.k_proj.weight_scale_inv, a4f.v_proj.weight_scale_inv, a4f.f_proj.weight_scale_inv, a4f.g_proj.weight_scale_inv]

    # Synthetic (4096, 4096) FP8 matrix (exact 0-diff ref via dequant_fp8_to_bf16)
    mx.random.seed(99)
    w4096_u8 = mx.random.randint(0, 240, (4096, 4096), dtype=mx.uint8)
    s4096 = mx.full((32, 32), 0.01, dtype=mx.float32)
    w4096_bf = dequant_fp8_to_bf16(w4096_u8, s4096)
    mx.eval(w4096_u8, s4096, w4096_bf)

    ns_cands = [(1, 8, 1), (1, 4, 1), (1, 2, 1), (1, 8, 2), (1, 4, 2), (1, 2, 2), (2, 4, 2), (2, 2, 2), (2, 4, 1), (2, 2, 1)]
    sw_cands = [(1, 4, 1), (1, 8, 1), (2, 4, 1), (2, 2, 1), (2, 4, 2)]
    ml_cands = [(1, 1), (1, 2), (2, 2), (2, 1), (3, 2)]
    lh_cands = [(8, 1, 2), (4, 1, 2), (8, 1, 1), (4, 1, 1), (4, 2, 2), (4, 2, 1), (4, 3, 2)]

    for B in (8, 9, 10, 12, 16, 20, 24):
        x4 = (mx.random.normal((B, 4096)) * 0.25).astype(mx.bfloat16)
        x8 = (mx.random.normal((B, 8192)) * 0.25).astype(mx.bfloat16)
        qn = (mx.random.normal((B, 64, 1, 128)) * 0.25).astype(mx.bfloat16)
        ao = (mx.random.normal((B, 64, 1, 512)) * 0.25).astype(mx.bfloat16)
        gr = (mx.random.normal((B, 64)) * 0.5).astype(mx.bfloat16)
        mx.eval(x4, x8, qn, ao, gr)

        ref_qkv = gemv_rows_multi(x4, w_qkv_bf)
        ref_op = gemv_rows(x8, a4b.o_proj.weight)
        ref_4096 = gemv_rows(x4, w4096_bf)
        ref_sh = gemv_swiglu_rows(x4, se_b.gate_proj.weight, se_b.up_proj.weight)
        ref_eq = mla_multilinear_rows(qn, a7b.embed_q.weight)
        ref_uo = mla_multilinear_rows(ao, a7b.unembed_out.weight, g_raw=gr)
        ref_lh = mx.concatenate([lm_head(x4[b : b + 1]) for b in range(B)], axis=0)
        mx.eval(*ref_qkv, ref_op, ref_4096, ref_sh, ref_eq, ref_uo, ref_lh)

        cands_b = ns_cands + ([(3, 2, 1), (3, 2, 2), (4, 2, 1)] if B >= 20 else [])
        res_qkv, res_op, res_4096 = [], [], []
        for bsg, bm_m, tm in cands_b:
            cfg = (bsg, (B + bsg - 1) // bsg, bm_m, tm)
            gq = gemv_fp8_rows_group([x4] * 5, w_qkv_f8, s_qkv_f8, cfg=cfg)
            go = gemv_fp8_rows(x8, a4f.o_proj.weight_fp8, a4f.o_proj.weight_scale_inv, cfg=cfg)
            g4 = gemv_fp8_rows(x4, w4096_u8, s4096, cfg=cfg)
            assert all(bit_diff_count(a, b) == 0 for a, b in zip(gq, ref_qkv)), f"qkv diff B={B} {cfg}"
            assert bit_diff_count(go, ref_op) == 0, f"op diff B={B} {cfg}"
            assert bit_diff_count(g4, ref_4096) == 0, f"4096 diff B={B} {cfg}"
            res_qkv.append((pipe_ms(lambda c=cfg: gemv_fp8_rows_group([x4] * 5, w_qkv_f8, s_qkv_f8, cfg=c)) * 1000, f"({bsg},{bm_m},{tm})"))
            res_op.append((pipe_ms(lambda c=cfg: gemv_fp8_rows(x8, a4f.o_proj.weight_fp8, a4f.o_proj.weight_scale_inv, cfg=c)) * 1000, f"({bsg},{bm_m},{tm})"))
            res_4096.append((pipe_ms(lambda c=cfg: gemv_fp8_rows(x4, w4096_u8, s4096, cfg=c)) * 1000, f"({bsg},{bm_m},{tm})"))

        res_sh = []
        cands_sw = [
            c for c in (sw_cands + ([(3, 2, 1), (4, 2, 1)] if B >= 20 else []))
            if ((B + c[0] - 1) // c[0]) * c[2] <= 12
        ]
        for bsg, bm_m, tm in cands_sw:
            cfg = (bsg, (B + bsg - 1) // bsg, bm_m, tm)
            gs = gemv_fp8_swiglu_rows(x4, se_f.gate_proj.weight_fp8, se_f.gate_proj.weight_scale_inv, se_f.up_proj.weight_fp8, se_f.up_proj.weight_scale_inv, cfg=cfg)
            assert bit_diff_count(gs, ref_sh) == 0, f"sh diff B={B} {cfg}"
            res_sh.append((pipe_ms(lambda c=cfg: gemv_fp8_swiglu_rows(x4, se_f.gate_proj.weight_fp8, se_f.gate_proj.weight_scale_inv, se_f.up_proj.weight_fp8, se_f.up_proj.weight_scale_inv, cfg=c)) * 1000, f"({bsg},{bm_m},{tm})"))

        res_ml = []
        for bsg, tm in ml_cands:
            cfg = (bsg, (B + bsg - 1) // bsg, tm)
            ge = mla_multilinear_fp8_rows(qn, a7f.embed_q.weight_fp8, a7f.embed_q.weight_scale_inv, scale_by_n=True, cfg=cfg)
            gu = mla_multilinear_fp8_rows(ao, a7f.unembed_out.weight_fp8, a7f.unembed_out.weight_scale_inv, scale_by_n=False, g_raw=gr, cfg=cfg)
            assert bit_diff_count(ge, ref_eq) == 0 and bit_diff_count(gu, ref_uo) == 0, f"ml diff B={B} {cfg}"
            te = pipe_ms(lambda c=cfg: mla_multilinear_fp8_rows(qn, a7f.embed_q.weight_fp8, a7f.embed_q.weight_scale_inv, scale_by_n=True, cfg=c)) * 1000
            tu = pipe_ms(lambda c=cfg: mla_multilinear_fp8_rows(ao, a7f.unembed_out.weight_fp8, a7f.unembed_out.weight_scale_inv, scale_by_n=False, g_raw=gr, cfg=c)) * 1000
            res_ml.append((te + tu, f"({bsg},{tm})={te:.0f}+{tu:.0f}"))

        res_lh = []
        for bm, bsg, tm in lh_cands:
            cfg = (bm, 1, 1, 32, tm, 4, bsg)
            gl = gemv_rows(x4, lm_head.weight, cfg=cfg)
            assert bit_diff_count(gl, ref_lh) == 0, f"lh diff B={B} {cfg}"
            tl = pipe_ms(lambda c=cfg: gemv_rows(x4, lm_head.weight, cfg=c)) * 1000
            res_lh.append((tl, f"(bm={bm},bsg={bsg},tm={tm})={tl:.0f}us"))

        for r in (res_qkv, res_op, res_4096, res_sh, res_ml, res_lh):
            r.sort()
        print(f"  B={B:2d}:", flush=True)
        print(f"    qkv5(8192,4096) : " + " ".join(f"{lb}={t:.0f}us" for t, lb in res_qkv[:4]), flush=True)
        print(f"    op  (4096,8192) : " + " ".join(f"{lb}={t:.0f}us" for t, lb in res_op[:4]), flush=True)
        print(f"    w4k (4096,4096) : " + " ".join(f"{lb}={t:.0f}us" for t, lb in res_4096[:4]), flush=True)
        print(f"    shSw(1280,4096) : " + " ".join(f"{lb}={t:.0f}us" for t, lb in res_sh[:4]), flush=True)
        print(f"    mla_fp8(eq+uo)  : " + " ".join(f"{lb}" for _, lb in res_ml[:4]), flush=True)
        print(f"    lm_head(bf16)   : " + " ".join(f"{lb}" for _, lb in res_lh[:4]), flush=True)


def bench_cold_multilayer():
    """Cold multi-layer (>=16L / >=2.7 GB, wired) DRAM GB/s benchmark across B = 1, 5, 8, 12."""
    info = mx.device_info()
    max_ws = int(info.get("max_recommended_working_set_size", 0))
    if max_ws > 0:
        mx.set_wired_limit(max_ws)
    print(f"\n=== Cold Multi-Layer DRAM Benchmark (wired={max_ws/1e9:.1f} GB) ===", flush=True)

    def _run_cold(fn, total_bytes, nlayers, reps=5):
        out = fn()
        mx.eval(*out) if isinstance(out, (list, tuple)) else mx.eval(out)
        mx.synchronize()
        dts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            out = fn()
            mx.eval(*out) if isinstance(out, (list, tuple)) else mx.eval(out)
            dts.append(time.perf_counter() - t0)
        dt = statistics.median(dts)
        return dt * 1e6 / nlayers, (total_bytes / 1e9) / dt

    mx.random.seed(7)
    NL, NL_SM = 16, 32
    qkv_w = [[mx.random.randint(0, 120, (8192, 4096), dtype=mx.uint8) for _ in range(5)] for _ in range(NL)]
    qkv_s = [[mx.full((64, 32), 0.01, dtype=mx.float32) for _ in range(5)] for _ in range(NL)]
    b_w = [mx.random.normal((64, 4096)).astype(mx.bfloat16) for _ in range(NL)]
    qkv_m_w = [mx.concatenate(qkv_w[i], axis=0) for i in range(NL)]
    qkv_m_s = [mx.concatenate(qkv_s[i], axis=0) for i in range(NL)]
    op_w = [mx.random.randint(0, 120, (4096, 8192), dtype=mx.uint8) for _ in range(NL)]
    op_s = [mx.full((32, 64), 0.01, dtype=mx.float32) for _ in range(NL)]
    dsw_g = [mx.random.randint(0, 120, (10240, 4096), dtype=mx.uint8) for _ in range(NL)]
    dsw_u = [mx.random.randint(0, 120, (10240, 4096), dtype=mx.uint8) for _ in range(NL)]
    dsw_sg = [mx.full((80, 32), 0.01, dtype=mx.float32) for _ in range(NL)]
    dsw_su = [mx.full((80, 32), 0.01, dtype=mx.float32) for _ in range(NL)]
    ddn_w = [mx.random.randint(0, 120, (4096, 10240), dtype=mx.uint8) for _ in range(NL)]
    ddn_s = [mx.full((32, 80), 0.01, dtype=mx.float32) for _ in range(NL)]
    rtr_w = [mx.random.normal((512, 4096)).astype(mx.bfloat16) for _ in range(NL_SM)]
    sesw_g = [mx.random.randint(0, 120, (1280, 4096), dtype=mx.uint8) for _ in range(NL_SM)]
    sesw_u = [mx.random.randint(0, 120, (1280, 4096), dtype=mx.uint8) for _ in range(NL_SM)]
    sesw_sg = [mx.full((10, 32), 0.01, dtype=mx.float32) for _ in range(NL_SM)]
    sesw_su = [mx.full((10, 32), 0.01, dtype=mx.float32) for _ in range(NL_SM)]
    sedn_w = [mx.random.randint(0, 120, (4096, 1280), dtype=mx.uint8) for _ in range(NL_SM)]
    sedn_s = [mx.full((32, 10), 0.01, dtype=mx.float32) for _ in range(NL_SM)]
    lh_w = [mx.random.normal((157184, 4096)).astype(mx.bfloat16) for _ in range(2)]
    mx.eval(
        *[w for lw in qkv_w for w in lw], *[s for ls in qkv_s for s in ls], *b_w,
        *qkv_m_w, *qkv_m_s, *op_w, *op_s,
        *dsw_g, *dsw_u, *dsw_sg, *dsw_su, *ddn_w, *ddn_s,
        *rtr_w, *sesw_g, *sesw_u, *sesw_sg, *sesw_su, *sedn_w, *sedn_s, *lh_w,
    )
    mx.synchronize()

    class _FakeMod:
        def __init__(self, w, s):
            self.weight_fp8 = w
            self.weight_scale_inv = s

    se_g = [_FakeMod(sesw_g[i], sesw_sg[i]) for i in range(NL_SM)]
    se_u = [_FakeMod(sesw_u[i], sesw_su[i]) for i in range(NL_SM)]
    B_QKVB = 5 * 8192 * 4096 + 64 * 4096 * 2
    B_OP = 4096 * 8192
    B_DSW = 2 * 10240 * 4096
    B_DDN = 4096 * 10240
    B_RS = 512 * 4096 * 2 + 2 * 1280 * 4096
    B_SD = 4096 * 1280
    B_LH = 157184 * 4096 * 2

    for B in (1, 5, 8, 12):
        x4 = (mx.random.normal((B, 4096)) * 0.25).astype(mx.bfloat16)
        x8 = (mx.random.normal((B, 8192)) * 0.25).astype(mx.bfloat16)
        xd = (mx.random.normal((B, 10240)) * 0.25).astype(mx.bfloat16)
        xs = (mx.random.normal((B, 1280)) * 0.25).astype(mx.bfloat16)
        mx.eval(x4, x8, xd, xs)
        u_qb, g_qb = _run_cold(lambda: [o for i in range(NL) for o in gemv_fp8_with_bf16_tail_rows(x4, qkv_w[i], qkv_s[i], b_w[i])], NL * B_QKVB, NL)
        u_qm, g_qm = _run_cold(lambda: [o for i in range(NL) for o in gemv_kda_qkvfgb_merged_rows(x4, qkv_m_w[i], qkv_m_s[i], b_w[i])], NL * B_QKVB, NL)
        u_op, g_op = _run_cold(lambda: [gemv_fp8_rows_res(x8, op_w[i], op_s[i], x4) for i in range(NL)], NL * B_OP, NL)
        u_dsw, g_dsw = _run_cold(lambda: [gemv_fp8_swiglu_rows(x4, dsw_g[i], dsw_sg[i], dsw_u[i], dsw_su[i]) for i in range(NL)], NL * B_DSW, NL)
        u_ddn, g_ddn = _run_cold(lambda: [gemv_fp8_rows_res(xd, ddn_w[i], ddn_s[i], x4) for i in range(NL)], NL * B_DDN, NL)
        u_rs, g_rs = _run_cold(lambda: [o for i in range(NL_SM) for o in gemv_router_se_swiglu_rows(x4, rtr_w[i], se_g[i], se_u[i])], NL_SM * B_RS, NL_SM)
        u_sd, g_sd = _run_cold(lambda: [gemv_fp8_rows(xs, sedn_w[i], sedn_s[i]) for i in range(NL_SM)], NL_SM * B_SD, NL_SM)
        u_lh, g_lh = _run_cold(lambda: [gemv_rows(x4, lh_w[i]) for i in range(2)], 2 * B_LH, 2)
        print(
            f"B={B:2d}: qkvfg+b={u_qb:5.1f}us/{g_qb:5.1f}G (x63={u_qb*63/1e3:5.2f}ms) | "
            f"mrg={u_qm:5.1f}us/{g_qm:5.1f}G (x63={u_qm*63/1e3:5.2f}ms) | "
            f"o_proj={u_op:4.1f}us/{g_op:5.1f}G (x63={u_op*63/1e3:4.2f}ms) | "
            f"denSw={u_dsw:5.1f}us/{g_dsw:5.1f}G | denDn={u_ddn:4.1f}us/{g_ddn:5.1f}G | "
            f"rtr+seSw={u_rs:4.1f}us/{g_rs:5.1f}G (x68={u_rs*68/1e3:4.2f}ms) | "
            f"se_dn={u_sd:4.1f}us/{g_sd:5.1f}G (x68={u_sd*68/1e3:4.2f}ms) | "
            f"lm_head={u_lh:5.0f}us/{g_lh:5.1f}G",
            flush=True,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-b", action="store_true")
    ap.add_argument("--skip-exact", action="store_true")
    ap.add_argument("--cold", action="store_true")
    a = ap.parse_args()

    if a.cold:
        bench_cold_multilayer()
        return 0

    l0_bf, _ = load_layer(0, use_fp8=False)
    l4_bf, _ = load_layer(4, use_fp8=False)
    l7_bf, _ = load_layer(7, use_fp8=False)
    l0_fp8, _ = load_layer(0, use_fp8=True)
    l4_fp8, _ = load_layer(4, use_fp8=True)
    l7_fp8, _ = load_layer(7, use_fp8=True)
    _, lm_head, _ = load_head()

    if a.sweep_b:
        sweep_b(l0_bf, l0_fp8, l4_bf, l4_fp8, l7_bf, l7_fp8, lm_head)
        return 0

    ok = True
    if not a.skip_exact:
        ok &= test_weights_and_prefill(l0_bf, l0_fp8, l4_bf, l4_fp8, l7_bf, l7_fp8)
        ok &= test_gemv_exactness(l0_bf, l0_fp8, l4_bf, l4_fp8, l7_bf, l7_fp8, lm_head)
        ok &= test_part_ss_and_repack(l0_bf, l0_fp8, l4_bf, l4_fp8, l7_bf, l7_fp8)
    if ok:
        bench_and_tune(l0_bf, l0_fp8, l4_bf, l4_fp8, l7_bf, l7_fp8, lm_head)
    print("\nSUMMARY:", "ALL PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
