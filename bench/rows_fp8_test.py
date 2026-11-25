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
)
from mlx_lm.models.bailing_moe_v3.rows.gemv_fp8 import (  # noqa: E402
    dequant_fp8_multilinear_to_bf16,
    dequant_fp8_to_bf16,
    gemv_fp8_rows,
    gemv_fp8_rows_group,
    gemv_fp8_swiglu_rows,
    gemv_lin,
    gemv_lin_multi,
    gemv_lin_res,
    gemv_lin_swiglu,
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

        # Check BF16 lm_head against B=1 MLX ref (verifies BSG spill fix at B>=13)
        ref_lh = mx.concatenate([lm_head(x4[b : b + 1]) for b in range(B)], axis=0)
        ok &= check(f"lm_head(bf16) B={B}", gemv_lin(x4, lm_head), ref_lh)
    return ok


def bench_and_tune(l0_bf, l0_fp8, l4_bf, l4_fp8, l7_bf, l7_fp8, lm_head):
    print("\n=== 3. Pipelined Kernel Microbenchmark (BF16 vs FP8, GB/s) & Config Tuning ===", flush=True)
    a4b, a4f = l4_bf.attention, l4_fp8.attention
    m0b, m0f = l0_bf.mlp, l0_fp8.mlp
    se_b, se_f = l4_bf.mlp.shared_experts, l4_fp8.mlp.shared_experts

    w_qkv_bf = [a4b.q_proj.weight, a4b.k_proj.weight, a4b.v_proj.weight, a4b.f_proj.weight, a4b.g_proj.weight]
    w_qkv_f8 = [a4f.q_proj.weight_fp8, a4f.k_proj.weight_fp8, a4f.v_proj.weight_fp8, a4f.f_proj.weight_fp8, a4f.g_proj.weight_fp8]
    s_qkv_f8 = [a4f.q_proj.weight_scale_inv, a4f.k_proj.weight_scale_inv, a4f.v_proj.weight_scale_inv, a4f.f_proj.weight_scale_inv, a4f.g_proj.weight_scale_inv]
    bytes_qkv_f8 = 5 * 8192 * 4096

    for B in (1, 5, 8, 12, 16, 20, 24):
        x4 = (mx.random.normal((B, 4096)) * 0.25).astype(mx.bfloat16)
        x8 = (mx.random.normal((B, 8192)) * 0.25).astype(mx.bfloat16)
        qa = (mx.random.normal((B, 1536)) * 0.25).astype(mx.bfloat16)
        mx.eval(x4, x8, qa)

        t_qkv_bf = pipe_ms(lambda: gemv_rows_multi(x4, w_qkv_bf))
        t_qkv_f8 = pipe_ms(lambda: gemv_fp8_rows_group([x4] * 5, w_qkv_f8, s_qkv_f8))
        gb_s = (bytes_qkv_f8 / 1e9) / (t_qkv_f8 / 1e3)

        t_op_bf = pipe_ms(lambda: gemv_rows_res(x8, a4b.o_proj.weight, x4))
        t_op_f8 = pipe_ms(lambda: gemv_lin_res(x8, a4f.o_proj, x4))

        t_den_bf = pipe_ms(lambda: gemv_swiglu_rows(x4, m0b.gate_proj.weight, m0b.up_proj.weight))
        t_den_f8 = pipe_ms(lambda: gemv_lin_swiglu(x4, m0f.gate_proj, m0f.up_proj))

        t_sh_bf = pipe_ms(lambda: gemv_swiglu_rows(x4, se_b.gate_proj.weight, se_b.up_proj.weight))
        t_sh_f8 = pipe_ms(lambda: gemv_lin_swiglu(x4, se_f.gate_proj, se_f.up_proj))

        t_lh = pipe_ms(lambda: gemv_lin(x4, lm_head))

        print(
            f"  B={B:2d}: qkv5 bf16={t_qkv_bf*1000:5.0f}us -> fp8={t_qkv_f8*1000:5.0f}us ({gb_s:5.1f} GB/s, x63={t_qkv_f8*63:4.1f}ms) | "
            f"o_proj {t_op_bf*1000:4.0f}->{t_op_f8*1000:4.0f}us | den_sw {t_den_bf*1000:4.0f}->{t_den_f8*1000:4.0f}us | "
            f"sh_sw {t_sh_bf*1000:3.0f}->{t_sh_f8*1000:3.0f}us | lm_head={t_lh*1000:4.0f}us",
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-b", action="store_true")
    ap.add_argument("--skip-exact", action="store_true")
    a = ap.parse_args()

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
    if ok:
        bench_and_tune(l0_bf, l0_fp8, l4_bf, l4_fp8, l7_bf, l7_fp8, lm_head)
    print("\nSUMMARY:", "ALL PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
