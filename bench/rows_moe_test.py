"""Bit-exactness test for rows/gather.py and rows/moe.py on L4 & L64 real visits and duplicate-expert collisions."""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(_root / "vendor"), str(_root)]

import mlx.core as mx  # noqa: E402
from mlx_lm.layers.common import swiglu  # noqa: E402
from mlx_lm.models.bailing_moe_v3.rows.gemv import gemv_rows, gemv_swiglu_rows  # noqa: E402
from mlx_lm.models.bailing_moe_v3.rows.gather import gather_rows  # noqa: E402
from mlx_lm.models.bailing_moe_v3.rows.moe import (  # noqa: E402
    moe_combine8_sh_res_rows,
    moe_down_combine_rows,
    moe_gate_up_swiglu_rows,
)

from bench._layer_loader import bit_diff_count, load_layer, load_visits  # noqa: E402


def check(name, got, ref):
    d = bit_diff_count(got, ref)
    print(f"  {name:44s} {str(got.shape):16s} {'PASS' if d == 0 else f'FAIL ({d}/{ref.size})'}")
    return d == 0


def stock_routed_b1(layer, xm, inds, wts):
    B = xm.shape[0]
    sw = layer.mlp.switch_mlp
    gates, ups, mids, raw_outs, comb_outs = [], [], [], [], []
    for b in range(B):
        xb = xm[b : b + 1, None, :]  # (1, 1, 4096)
        ib = inds[b : b + 1, None, :]  # (1, 1, 8)
        wb = wts[b : b + 1, None, :]  # (1, 1, 8)
        xb_exp = mx.expand_dims(xb, (-2, -3))  # (1, 1, 1, 1, 4096)
        g = sw.gate_proj(xb_exp, ib)[0, 0, :, 0, :]  # (8, 1280)
        u = sw.up_proj(xb_exp, ib)[0, 0, :, 0, :]  # (8, 1280)
        m = swiglu(g, u)  # (8, 1280)
        dn = sw.down_proj(m[None, None, :, None, :], ib)[0, 0, :, 0, :]  # (8, 4096)
        routed = (dn[None, None, :, :] * wb[..., None]).sum(axis=-2).astype(xm.dtype)[0, 0]  # (4096,)
        gates.append(g[None])
        ups.append(u[None])
        mids.append(m[None])
        raw_outs.append(dn[None])
        comb_outs.append(routed[None])
    return (
        mx.concatenate(gates, axis=0),
        mx.concatenate(ups, axis=0),
        mx.concatenate(mids, axis=0),
        mx.concatenate(raw_outs, axis=0),
        mx.concatenate(comb_outs, axis=0),
    )


def main():
    ok = True
    for L in (4, 64):
        print(f"=== Layer L{L} ===")
        layer, _ = load_layer(L)
        sw = layer.mlp.switch_mlp
        se = layer.mlp.shared_experts
        visits = load_visits(L, limit=8)

        # 1. Real captured visits B = 1..4
        for B in (1, 2, 3, 4):
            xm = mx.concatenate([v["mlp_in"][0, 0:1] for v in visits[:B]], axis=0)
            hp = mx.concatenate([v["h_post"][0, 0:1] for v in visits[:B]], axis=0)
            inds = mx.concatenate([v["r_inds"][0, 0:1] for v in visits[:B]], axis=0).astype(mx.uint32)
            wts = mx.concatenate([v["r_wts"][0, 0:1] for v in visits[:B]], axis=0)
            ref_routed = mx.concatenate([v["routed_out"][0, 0:1] for v in visits[:B]], axis=0)
            ref_out = mx.concatenate([v["out"][0, 0:1] for v in visits[:B]], axis=0)

            ref_g, ref_u, ref_m, ref_dn, ref_comb = stock_routed_b1(layer, xm, inds, wts)
            ok &= check(f"L{L} B={B} stock_vs_visit_routed", ref_comb, ref_routed)

            got_g = gather_rows(xm, sw.gate_proj.weight, sw.gate_proj.scales, None, inds)
            ok &= check(f"L{L} B={B} gather_rows(gate)", got_g, ref_g)

            got_m = moe_gate_up_swiglu_rows(
                xm, sw.gate_proj.weight, sw.gate_proj.scales, sw.up_proj.weight, sw.up_proj.scales, inds
            )
            ok &= check(f"L{L} B={B} moe_gate_up_swiglu", got_m, ref_m)

            got_dn = gather_rows(got_m, sw.down_proj.weight, sw.down_proj.scales, None, inds)
            ok &= check(f"L{L} B={B} gather_rows(down)", got_dn, ref_dn)

            got_routed = moe_down_combine_rows(got_m, sw.down_proj.weight, sw.down_proj.scales, inds, wts)
            ok &= check(f"L{L} B={B} moe_down_combine", got_routed, ref_routed)

            sh = gemv_rows(gemv_swiglu_rows(xm, se.gate_proj.weight, se.up_proj.weight), se.down_proj.weight)
            got_full = moe_down_combine_rows(
                got_m, sw.down_proj.weight, sw.down_proj.scales, inds, wts, shared=sh, residual=hp
            )
            ok &= check(f"L{L} B={B} moe_down_combine+sh+res", got_full, ref_out)
            got_gc = moe_combine8_sh_res_rows(got_dn, wts, shared=sh, residual=hp)
            ok &= check(f"L{L} B={B} gather+combine8+sh+res", got_gc, ref_out)

        # 2. Forced duplicate-expert collision stress (intra-row & cross-row)
        mx.random.seed(123)
        coll_rows = [
            [17, 17, 42, 42, 100, 7, 100, 7],
            [17, 42, 3, 17, 9, 9, 42, 8],
            [7, 17, 17, 17, 511, 0, 42, 100],
            [17, 17, 17, 17, 17, 17, 17, 17],
        ]
        for B in (2, 3, 4):
            xm = mx.random.normal((B, 4096)).astype(mx.bfloat16)
            hp = mx.random.normal((B, 4096)).astype(mx.bfloat16)
            sh = mx.random.normal((B, 4096)).astype(mx.bfloat16)
            inds = mx.array(coll_rows[:B], dtype=mx.uint32)
            wts = mx.softmax(mx.random.normal((B, 8)), axis=-1) * 2.5
            mx.eval(xm, hp, sh, inds, wts)

            _, _, ref_m, _, ref_comb = stock_routed_b1(layer, xm, inds, wts)
            ref_full = (ref_comb + sh) + hp

            got_m = moe_gate_up_swiglu_rows(
                xm, sw.gate_proj.weight, sw.gate_proj.scales, sw.up_proj.weight, sw.up_proj.scales, inds
            )
            ok &= check(f"L{L} B={B} coll gate_up_swiglu", got_m, ref_m)

            got_full = moe_down_combine_rows(
                got_m, sw.down_proj.weight, sw.down_proj.scales, inds, wts, shared=sh, residual=hp
            )
            ok &= check(f"L{L} B={B} coll down_combine+sh+res", got_full, ref_full)

    print("SUMMARY:", "ALL PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
