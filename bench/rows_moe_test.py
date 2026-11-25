"""Bit-exactness test for rows/gather.py and rows/moe.py on L4 & L64 real visits and duplicate-expert collisions."""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(_root / "vendor"), str(_root)]

import mlx.core as mx  # noqa: E402
from mlx_lm.layers.common import swiglu  # noqa: E402
from mlx_lm.models.bailing_moe_v3.rows.gemv import gemv_rows, gemv_swiglu_rows  # noqa: E402
from mlx_lm.models.bailing_moe_v3.rows.gemv_fp8 import gemv_lin, gemv_lin_swiglu  # noqa: E402
from mlx_lm.models.bailing_moe_v3.rows.gather import gather_rows  # noqa: E402
from mlx_lm.models.bailing_moe_v3.rows.moe import (  # noqa: E402
    moe_combine8_sh_res_rows,
    moe_dedup_rows,
    moe_down_combine_packed_rows,
    moe_down_combine_rows,
    moe_gate_up_swiglu_packed_rows,
    moe_gate_up_swiglu_rows,
    pack_moe_dn_weights,
    pack_moe_gu_weights,
    unpack_moe_dn_weights,
    unpack_moe_gu_weights,
)
from mlx_lm.models.bailing_moe_v3.rows.router import gate_rows  # noqa: E402

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
        for B in (1, 2, 3, 4):
            xm = mx.random.normal((B, 4096)).astype(mx.bfloat16)
            hp = mx.random.normal((B, 4096)).astype(mx.bfloat16)
            sh = mx.random.normal((B, 4096)).astype(mx.bfloat16)
            inds = mx.array(coll_rows[:B], dtype=mx.uint32)
            wts = mx.softmax(mx.random.normal((B, 8)), axis=-1) * 2.5
            mx.eval(xm, hp, sh, inds, wts)

            _, _, ref_m, _, ref_comb = stock_routed_b1(layer, xm, inds, wts)
            ref_full = (ref_comb + sh) + hp
            dd_c = moe_dedup_rows(inds) if B >= 2 else None

            got_m = moe_gate_up_swiglu_rows(
                xm, sw.gate_proj.weight, sw.gate_proj.scales, sw.up_proj.weight, sw.up_proj.scales, inds, dedup=dd_c
            )
            ok &= check(f"L{L} B={B} coll gate_up_swiglu", got_m, ref_m)

            got_full = moe_down_combine_rows(
                got_m, sw.down_proj.weight, sw.down_proj.scales, inds, wts, shared=sh, residual=hp, dedup=dd_c
            )
            ok &= check(f"L{L} B={B} coll down_combine+sh+res", got_full, ref_full)

        # 3. High-B adaptive row-split test (B=5..24): realistic router + forced cnt=1,2,3,4,5,8,12,16
        for B in (5, 8, 12, 16, 20, 24):
            base = (mx.random.normal((1, 4096)) * 0.25).astype(mx.bfloat16)
            xm = (base + (mx.random.normal((B, 4096)) * 0.12)).astype(mx.bfloat16)
            hp = (mx.random.normal((B, 4096)) * 0.25).astype(mx.bfloat16)
            sh = (mx.random.normal((B, 4096)) * 0.25).astype(mx.bfloat16)
            gate = layer.mlp.gate
            inds_real, wts = gate_rows(
                xm, gate.weight, gate.expert_bias,
                top_k=gate.top_k, n_group=gate.n_group, topk_group=gate.topk_group,
                routed_scaling_factor=gate.routed_scaling_factor,
            )
            mx.eval(xm, hp, sh, inds_real, wts)
            _, _, ref_m, _, ref_comb = stock_routed_b1(layer, xm, inds_real, wts)
            ref_full = (ref_comb + sh) + hp
            dd = moe_dedup_rows(inds_real)
            got_m = moe_gate_up_swiglu_rows(
                xm, sw.gate_proj.weight, sw.gate_proj.scales, sw.up_proj.weight, sw.up_proj.scales,
                inds_real, dedup=dd, adaptive=1,
            )
            ok &= check(f"L{L} B={B} real adapt gu", got_m, ref_m)
            got_f = moe_down_combine_rows(
                got_m, sw.down_proj.weight, sw.down_proj.scales, inds_real, wts,
                shared=sh, residual=hp, dedup=dd,
            )
            ok &= check(f"L{L} B={B} real adapt dc", got_f, ref_full)

        # Forced exact cnt=1,2,3,4,5,8,12,16 collisions across B=16 rows (all with adaptive=1 dedup & inline)
        xm = (mx.random.normal((16, 4096)) * 0.25).astype(mx.bfloat16)
        hp = (mx.random.normal((16, 4096)) * 0.25).astype(mx.bfloat16)
        sh = (mx.random.normal((16, 4096)) * 0.25).astype(mx.bfloat16)
        wts = mx.softmax(mx.random.normal((16, 8)), axis=-1) * 2.5
        forced_rows = []
        for b in range(16):
            row = [100 + b * 8 + j for j in range(8)]
            if b < 16:
                row[0] = 10   # cnt=16
            if b < 12:
                row[1] = 20   # cnt=12
            if b < 8:
                row[2] = 30   # cnt=8
            if b < 5:
                row[3] = 40   # cnt=5
            if b < 4:
                row[4] = 50   # cnt=4
            if b < 3:
                row[5] = 60   # cnt=3
            if b < 2:
                row[6] = 70   # cnt=2 (row[7] gives cnt=1 experts!)
            forced_rows.append(row)
        inds_f = mx.array(forced_rows, dtype=mx.uint32)
        mx.eval(xm, hp, sh, wts, inds_f)
        _, _, ref_mf, _, ref_combf = stock_routed_b1(layer, xm, inds_f, wts)
        ref_fullf = (ref_combf + sh) + hp
        dd_f = moe_dedup_rows(inds_f)
        for mode, dd_arg in (("dedup", dd_f), ("inline", None)):
            gm = moe_gate_up_swiglu_rows(
                xm, sw.gate_proj.weight, sw.gate_proj.scales, sw.up_proj.weight, sw.up_proj.scales,
                inds_f, dedup=dd_arg, adaptive=1,
            )
            ok &= check(f"L{L} B=16 forced({mode}) adapt gu", gm, ref_mf)
            gf = moe_down_combine_rows(
                gm, sw.down_proj.weight, sw.down_proj.scales, inds_f, wts,
                shared=sh, residual=hp, dedup=dd_arg,
            )
            ok &= check(f"L{L} B=16 forced({mode}) adapt dc", gf, ref_fullf)

        # 4. T3 lossless pack/unpack round-trip + packed kernels & T7 se_down fusion (FP8 shared_experts)
        layer_fp8, _ = load_layer(L, use_fp8=True)
        se_fp8 = layer_fp8.mlp.shared_experts
        sw_fp8 = layer_fp8.mlp.switch_mlp
        gu_pk = pack_moe_gu_weights(
            sw_fp8.gate_proj.weight, sw_fp8.gate_proj.scales,
            sw_fp8.up_proj.weight, sw_fp8.up_proj.scales,
        )
        dn_pk = pack_moe_dn_weights(sw_fp8.down_proj.weight, sw_fp8.down_proj.scales)
        u_wg, u_sg, u_wu, u_su = unpack_moe_gu_weights(gu_pk)
        u_wd, u_sd = unpack_moe_dn_weights(dn_pk)
        ok &= check(f"L{L} T3 unpack wg", u_wg, sw_fp8.gate_proj.weight)
        ok &= check(f"L{L} T3 unpack sg", u_sg, sw_fp8.gate_proj.scales)
        ok &= check(f"L{L} T3 unpack wu", u_wu, sw_fp8.up_proj.weight)
        ok &= check(f"L{L} T3 unpack su", u_su, sw_fp8.up_proj.scales)
        ok &= check(f"L{L} T3 unpack wd", u_wd, sw_fp8.down_proj.weight)
        ok &= check(f"L{L} T3 unpack sd", u_sd, sw_fp8.down_proj.scales)

        for B in (1, 3, 5, 8, 12):
            base = (mx.random.normal((1, 4096)) * 0.25).astype(mx.bfloat16)
            xm = (base + (mx.random.normal((B, 4096)) * 0.14)).astype(mx.bfloat16)
            hp = (mx.random.normal((B, 4096)) * 0.25).astype(mx.bfloat16)
            gate = layer_fp8.mlp.gate
            inds_p, wts_p = gate_rows(
                xm, gate.weight, gate.expert_bias,
                top_k=gate.top_k, n_group=gate.n_group, topk_group=gate.topk_group,
                routed_scaling_factor=gate.routed_scaling_factor,
            )
            dd_p = moe_dedup_rows(inds_p) if B >= 2 else None
            se_mid = gemv_lin_swiglu(xm, se_fp8.gate_proj, se_fp8.up_proj)
            sh_ref = gemv_lin(se_mid, se_fp8.down_proj)
            ref_m = moe_gate_up_swiglu_rows(
                xm, sw_fp8.gate_proj.weight, sw_fp8.gate_proj.scales,
                sw_fp8.up_proj.weight, sw_fp8.up_proj.scales, inds_p, dedup=dd_p,
            )
            ref_dc = moe_down_combine_rows(
                ref_m, sw_fp8.down_proj.weight, sw_fp8.down_proj.scales,
                inds_p, wts_p, shared=sh_ref, residual=hp, dedup=dd_p,
            )
            got_t7 = moe_down_combine_rows(
                ref_m, sw_fp8.down_proj.weight, sw_fp8.down_proj.scales,
                inds_p, wts_p, residual=hp, dedup=dd_p,
                se_mid=se_mid, se_w_fp8=se_fp8.down_proj.weight_fp8, se_sc=se_fp8.down_proj.weight_scale_inv,
            )
            ok &= check(f"L{L} B={B} T7 dc+se_down", got_t7, ref_dc)
            for skb in (False, True):
                got_pk_gu = moe_gate_up_swiglu_packed_rows(xm, gu_pk, inds_p, dedup=dd_p, single_kb=skb)
                ok &= check(f"L{L} B={B} T3 gu_packed(skb={int(skb)})", got_pk_gu, ref_m)
            got_pk_dc = moe_down_combine_packed_rows(
                ref_m, dn_pk, inds_p, wts_p, shared=sh_ref, residual=hp, dedup=dd_p,
            )
            ok &= check(f"L{L} B={B} T3 dc_packed", got_pk_dc, ref_dc)
            got_pk_t7 = moe_down_combine_packed_rows(
                ref_m, dn_pk, inds_p, wts_p, residual=hp, dedup=dd_p,
                se_mid=se_mid, se_w_fp8=se_fp8.down_proj.weight_fp8, se_sc=se_fp8.down_proj.weight_scale_inv,
            )
            ok &= check(f"L{L} B={B} T3+T7 dc_packed+se", got_pk_t7, ref_dc)

    print("SUMMARY:", "ALL PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
