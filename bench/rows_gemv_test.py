"""Bit-exactness test for rows/gemv.py across all LingSante bf16 linear shapes and B in {1,2,3,4}."""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(_root / "vendor"), str(_root)]

import mlx.core as mx  # noqa: E402
from mlx_lm.layers.common import swiglu  # noqa: E402
from mlx_lm.models.bailing_moe_v3.rows.gemv import (  # noqa: E402
    gemv_rows,
    gemv_rows_multi,
    gemv_rows_res,
    gemv_swiglu_rows,
    router_gemv_rows,
)

from bench._layer_loader import bit_diff_count, load_head, load_layer, load_visits  # noqa: E402


def check_case(name, got, ref):
    diffs = bit_diff_count(got, ref)
    status = "PASS" if diffs == 0 else f"FAIL ({diffs}/{ref.size})"
    print(f"  {name:42s} {str(got.shape):16s} {status}")
    return diffs == 0


def main():
    ok = True
    print("=== Loading L0 (KDA+Dense), L4 (KDA+MoE), L7 (MLA+MoE), head ===")
    l0, _ = load_layer(0)
    l4, _ = load_layer(4)
    l7, _ = load_layer(7)
    norm, lm_head, _ = load_head()

    v0 = load_visits(0, limit=4)
    v4 = load_visits(4, limit=4)
    v7 = load_visits(7, limit=4)
    vh = load_visits("head", limit=4)

    for B in (1, 2, 3, 4):
        print(f"\n--- B={B} ---")
        # 1. L0 dense MLP swiglu + down_res
        x0 = mx.concatenate([v["mlp_in"][0, 0:1] for v in v0[:B]], axis=0)
        res0 = mx.concatenate([v["h_post"][0, 0:1] for v in v0[:B]], axis=0)
        ref_g, ref_u = [], []
        for b in range(B):
            xb = x0[b : b + 1]
            ref_g.append(l0.mlp.gate_proj(xb))
            ref_u.append(l0.mlp.up_proj(xb))
        ref_g_cat = mx.concatenate(ref_g, axis=0)
        ref_u_cat = mx.concatenate(ref_u, axis=0)
        got_g, got_u = gemv_rows_multi(x0, [l0.mlp.gate_proj.weight, l0.mlp.up_proj.weight])
        ok &= check_case("L0.dense.gate_proj", got_g, ref_g_cat)
        ok &= check_case("L0.dense.up_proj", got_u, ref_u_cat)

        ref_sw = mx.concatenate([swiglu(ref_g[b], ref_u[b]) for b in range(B)], axis=0)
        got_sw = gemv_swiglu_rows(x0, l0.mlp.gate_proj.weight, l0.mlp.up_proj.weight)
        ok &= check_case("L0.dense.gemv_swiglu", got_sw, ref_sw)

        ref_dn = mx.concatenate([l0.mlp.down_proj(ref_sw[b : b + 1]) + res0[b : b + 1] for b in range(B)], axis=0)
        got_dn = gemv_rows_res(ref_sw, l0.mlp.down_proj.weight, res0)
        ok &= check_case("L0.dense.down_res", got_dn, ref_dn)

        # 2. L4 KDA q,k,v,f,g,b + o_proj + shared_experts + router
        x4 = mx.concatenate([v["h_in"][0, 0:1] for v in v4[:B]], axis=0)
        a4 = l4.attention
        projs = [a4.q_proj, a4.k_proj, a4.v_proj, a4.f_proj, a4.g_proj, a4.b_proj]
        refs4 = [mx.concatenate([p(x4[b : b + 1]) for b in range(B)], axis=0) for p in projs]
        gots4 = gemv_rows_multi(x4, [p.weight for p in projs])
        for nm, g, r in zip(["q", "k", "v", "f", "g", "b"], gots4, refs4):
            ok &= check_case(f"L4.kda.{nm}_proj", g, r)

        xo = mx.concatenate([v["x"][0, 0:1] for v in v4[:B]], axis=0)  # (B, 4096)
        oi = mx.random.normal((B, 8192)).astype(mx.bfloat16)
        mx.eval(oi)
        ref_o = mx.concatenate([a4.o_proj(oi[b : b + 1]) + xo[b : b + 1] for b in range(B)], axis=0)
        got_o = gemv_rows_res(oi, a4.o_proj.weight, xo)
        ok &= check_case("L4.kda.o_proj_res", got_o, ref_o)

        xm4 = mx.concatenate([v["mlp_in"][0, 0:1] for v in v4[:B]], axis=0)
        se4 = l4.mlp.shared_experts
        ref_se_sw = mx.concatenate([swiglu(se4.gate_proj(xm4[b : b + 1]), se4.up_proj(xm4[b : b + 1])) for b in range(B)], axis=0)
        got_se_sw = gemv_swiglu_rows(xm4, se4.gate_proj.weight, se4.up_proj.weight)
        ok &= check_case("L4.shared.gemv_swiglu", got_se_sw, ref_se_sw)
        ref_se_dn = mx.concatenate([se4.down_proj(ref_se_sw[b : b + 1]) for b in range(B)], axis=0)
        got_se_dn = gemv_rows(ref_se_sw, se4.down_proj.weight)
        ok &= check_case("L4.shared.down_proj", got_se_dn, ref_se_dn)

        ref_rlog = mx.concatenate(
            [xm4[b : b + 1].astype(mx.float32) @ l4.mlp.gate.weight.astype(mx.float32).T for b in range(B)], axis=0
        )
        got_rlog = router_gemv_rows(xm4, l4.mlp.gate.weight)
        ok &= check_case("L4.router_gemv_bf16", got_rlog, ref_rlog)

        # 3. L7 MLA q_a, kv_a, g_proj, q_b, dense
        x7 = mx.concatenate([v["h_in"][0, 0:1] for v in v7[:B]], axis=0)
        a7 = l7.attention
        m_projs = [a7.q_a_proj, a7.kv_a_proj_with_mqa, a7.g_proj]
        refs7 = [mx.concatenate([p(x7[b : b + 1]) for b in range(B)], axis=0) for p in m_projs]
        gots7 = gemv_rows_multi(x7, [p.weight for p in m_projs])
        for nm, g, r in zip(["q_a", "kv_a", "g"], gots7, refs7):
            ok &= check_case(f"L7.mla.{nm}_proj", g, r)
        q_a_in = refs7[0]
        ref_qb = mx.concatenate([a7.q_b_proj(q_a_in[b : b + 1]) for b in range(B)], axis=0)
        got_qb = gemv_rows(q_a_in, a7.q_b_proj.weight)
        ok &= check_case("L7.mla.q_b_proj", got_qb, ref_qb)

        # 4. lm_head
        xh = mx.concatenate([v["hidden"][0, 0:1] for v in vh[:B]], axis=0)
        ref_lh = mx.concatenate([lm_head(xh[b : b + 1]) for b in range(B)], axis=0)
        got_lh = gemv_rows(xh, lm_head.weight)
        ok &= check_case("lm_head", got_lh, ref_lh)

    print("\nSUMMARY:", "ALL PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
