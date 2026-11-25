"""Bit-exactness test for rows/gather.py, rows/router.py, and rows/moe.py (Phase A T1-T7).

Verifies 0 bit diffs on L4 & L64 real visits, tie stress, duplicate-expert collisions,
B=1..24, 2-PSO split, LPT dedup, fused group_router_dedup, T6 part_ss RMSNorm parity,
T7 fused se_down (FP8), and T3 runtime-on-load packed/unpacked MoE weights.
"""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(_root / "vendor"), str(_root)]

import mlx.core as mx  # noqa: E402
from mlx_lm.layers.common import swiglu  # noqa: E402
from mlx_lm.models.bailing_moe_v3.rows.gather import gather_rows  # noqa: E402
from mlx_lm.models.bailing_moe_v3.rows.gemv import gemv_rows, gemv_swiglu_rows, router_gemv_rows  # noqa: E402
from mlx_lm.models.bailing_moe_v3.rows.gemv_fp8 import gemv_lin, gemv_lin_swiglu  # noqa: E402
from mlx_lm.models.bailing_moe_v3.rows.glue import GLUE_HEADER  # noqa: E402
from mlx_lm.models.bailing_moe_v3.rows.moe import (  # noqa: E402
    moe_combine8_sh_res_rows,
    moe_dedup_rows,
    moe_down_combine_packed_rows,
    moe_down_combine_rows,
    moe_gate_up_swiglu_packed_rows,
    moe_gate_up_swiglu_rows,
    moe_gate_up_swiglu_split_rows,
    pack_moe_dn_weights,
    pack_moe_gu_weights,
    unpack_moe_dn_weights,
    unpack_moe_gu_weights,
)
from mlx_lm.models.bailing_moe_v3.rows.router import gate_dedup_rows, gate_rows, group_router_dedup_rows  # noqa: E402

from bench._layer_loader import bit_diff_count, load_layer, load_visits  # noqa: E402

# Minimal kernel verifying T6 part_ss reproduces mx.fast.rms_norm bit-for-bit via exact 32-group butterfly.
_RMS_FROM_PART_SS_SRC = r"""
    const int b = int(threadgroup_position_in_grid.y);
    const uint lid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    threadgroup float gsum[32];
    const device float* ps = part_ss + size_t(b) * (NN / 4);
    for (int g = int(sg); g < 32; g += 8) {
      float acc = ps[g * 32 + int(lane)];
      float s = simd_sum(acc);
      if (lane == 0) gsum[g] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = simd_sum(lane < 32 ? gsum[lane] : 0.0f);
    float inv = metal::precise::rsqrt(tot / float(NN) + 1e-6f);
    const device bfloat16_t* xr = x + size_t(b) * NN;
    device bfloat16_t* yr = y + size_t(b) * NN;
    for (int i = int(lid); i < NN; i += 256) {
      yr[i] = glue_rms_apply<bfloat16_t, bfloat16_t>(static_cast<float>(xr[i]), inv, w[i]);
    }
"""

_RMS_PART_KERN = None


def _rms_from_part_ss(x: mx.array, part_ss: mx.array, w: mx.array) -> mx.array:
    global _RMS_PART_KERN
    if _RMS_PART_KERN is None:
        _RMS_PART_KERN = mx.fast.metal_kernel(
            name="bv3_test_rms_from_part_ss",
            input_names=["x", "part_ss", "w"],
            output_names=["y"],
            header=GLUE_HEADER,
            source=_RMS_FROM_PART_SS_SRC,
            ensure_row_contiguous=True,
        )
    B, N = x.shape
    (y,) = _RMS_PART_KERN(
        inputs=[x, part_ss, w],
        template=[("NN", N)],
        grid=(256, B, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(B, N)],
        output_dtypes=[mx.bfloat16],
    )
    return y


def check(name, got, ref):
    d = bit_diff_count(got, ref)
    print(f"  {name:48s} {str(got.shape):16s} {'PASS' if d == 0 else f'FAIL ({d}/{ref.size})'}", flush=True)
    return d == 0


def check_lpt_dedup(name, inds, dd):
    u_meta, u_pqs, n_hot, _, n_uniq = dd
    mx.eval(u_meta, u_pqs, n_hot, n_uniq)
    nu = int(n_uniq.item())
    nh = int(n_hot.item())
    metas = u_meta[:nu].tolist()
    pqs = u_pqs.tolist()
    flat_inds = inds.reshape(-1).tolist()
    keys = []
    seen_pqs = set()
    ok = True
    hot_cnt = 0
    for m in metas:
        e = m & 0x3FF
        off = (m >> 10) & 0xFF
        cnt = (m >> 18) & 0xFF
        keys.append((-cnt, e))
        if cnt >= 2:
            hot_cnt += 1
        for t in range(cnt):
            p = pqs[off + t]
            if flat_inds[p] != e or p in seen_pqs:
                ok = False
            seen_pqs.add(p)
    if keys != sorted(keys) or hot_cnt != nh or len(seen_pqs) != len(flat_inds):
        ok = False
    print(f"  {name:48s} U={nu:2d} hot={nh:2d}     {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def stock_routed_b1(layer, xm, inds, wts):
    B = xm.shape[0]
    sw = layer.mlp.switch_mlp
    gates, ups, mids, raw_outs, comb_outs = [], [], [], [], []
    for b in range(B):
        xb = xm[b : b + 1, None, :]
        ib = inds[b : b + 1, None, :]
        wb = wts[b : b + 1, None, :]
        xb_exp = mx.expand_dims(xb, (-2, -3))
        g = sw.gate_proj(xb_exp, ib)[0, 0, :, 0, :]
        u = sw.up_proj(xb_exp, ib)[0, 0, :, 0, :]
        m = swiglu(g, u)
        dn = sw.down_proj(m[None, None, :, None, :], ib)[0, 0, :, 0, :]
        routed = (dn[None, None, :, :] * wb[..., None]).sum(axis=-2).astype(xm.dtype)[0, 0]
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
        print(f"=== Layer L{L} ===", flush=True)
        layer, _ = load_layer(L, use_fp8=True)
        sw = layer.mlp.switch_mlp
        se = layer.mlp.shared_experts
        gate = layer.mlp.gate
        visits = load_visits(L, limit=8)

        # 1. Real captured visits B = 1..4 (inline, dedup, split, T6 part_ss, T7 se_down)
        for B in (1, 2, 3, 4):
            xm = mx.concatenate([v["mlp_in"][0, 0:1] for v in visits[:B]], axis=0)
            hp = mx.concatenate([v["h_post"][0, 0:1] for v in visits[:B]], axis=0)
            inds = mx.concatenate([v["r_inds"][0, 0:1] for v in visits[:B]], axis=0).astype(mx.uint32)
            wts = mx.concatenate([v["r_wts"][0, 0:1] for v in visits[:B]], axis=0)
            ref_routed = mx.concatenate([v["routed_out"][0, 0:1] for v in visits[:B]], axis=0)
            ref_out = mx.concatenate([v["out"][0, 0:1] for v in visits[:B]], axis=0)

            ref_g, ref_u, ref_m, ref_dn, ref_comb = stock_routed_b1(layer, xm, inds, wts)
            ok &= check(f"L{L} B={B} stock_vs_visit_routed", ref_comb, ref_routed)

            # Fused router + LPT dedup check
            ri_d, rw_d, dd_r = gate_dedup_rows(
                xm, gate.weight, gate.expert_bias,
                top_k=gate.top_k, n_group=gate.n_group, topk_group=gate.topk_group,
                routed_scaling_factor=gate.routed_scaling_factor,
            )
            ok &= check(f"L{L} B={B} gate_dedup inds", ri_d, inds)
            ok &= check(f"L{L} B={B} gate_dedup wts", rw_d, wts)
            ok &= check_lpt_dedup(f"L{L} B={B} lpt_dedup invariant", ri_d, dd_r)

            got_g = gather_rows(xm, sw.gate_proj.weight, sw.gate_proj.scales, None, inds)
            ok &= check(f"L{L} B={B} gather_rows(gate)", got_g, ref_g)

            got_m = moe_gate_up_swiglu_rows(
                xm, sw.gate_proj.weight, sw.gate_proj.scales, sw.up_proj.weight, sw.up_proj.scales, inds
            )
            ok &= check(f"L{L} B={B} moe_gu inline", got_m, ref_m)

            dd = moe_dedup_rows(inds)
            got_m_dd = moe_gate_up_swiglu_rows(
                xm, sw.gate_proj.weight, sw.gate_proj.scales, sw.up_proj.weight, sw.up_proj.scales, inds, dedup=dd
            )
            ok &= check(f"L{L} B={B} moe_gu dedup", got_m_dd, ref_m)

            got_dn = gather_rows(got_m, sw.down_proj.weight, sw.down_proj.scales, None, inds)
            ok &= check(f"L{L} B={B} gather_rows(down)", got_dn, ref_dn)

            got_routed = moe_down_combine_rows(got_m, sw.down_proj.weight, sw.down_proj.scales, inds, wts)
            ok &= check(f"L{L} B={B} moe_dc inline", got_routed, ref_routed)

            se_mid = gemv_lin_swiglu(xm, se.gate_proj, se.up_proj)
            sh = gemv_lin(se_mid, se.down_proj)
            got_full = moe_down_combine_rows(
                got_m, sw.down_proj.weight, sw.down_proj.scales, inds, wts, shared=sh, residual=hp
            )
            ok &= check(f"L{L} B={B} moe_dc+sh+res inline", got_full, ref_out)

            got_full_dd, pss = moe_down_combine_rows(
                got_m_dd, sw.down_proj.weight, sw.down_proj.scales, inds, wts,
                shared=sh, residual=hp, dedup=dd, return_part_ss=True,
            )
            ok &= check(f"L{L} B={B} moe_dc+sh+res dedup", got_full_dd, ref_out)
            rms_ref = mx.fast.rms_norm(ref_out, layer.input_layernorm.weight, 1e-6)
            rms_got = _rms_from_part_ss(got_full_dd, pss, layer.input_layernorm.weight)
            ok &= check(f"L{L} B={B} T6 part_ss rms_norm parity", rms_got, rms_ref)

            # T7 fused se_down + 2-PSO split
            h_split = moe_gate_up_swiglu_split_rows(
                xm, sw.gate_proj.weight, sw.gate_proj.scales, sw.up_proj.weight, sw.up_proj.scales, dd
            )
            got_se_split, pss2 = moe_down_combine_rows(
                h_split, sw.down_proj.weight, sw.down_proj.scales, inds, wts,
                residual=hp, dedup=dd,
                se_mid=se_mid, se_w_fp8=se.down_proj.weight_fp8, se_sc=se.down_proj.weight_scale_inv,
                return_part_ss=True,
            )
            ok &= check(f"L{L} B={B} 2-PSO split + T7 se_down", got_se_split, ref_out)
            ok &= check(f"L{L} B={B} 2-PSO + T7 + T6 part_ss", _rms_from_part_ss(got_se_split, pss2, layer.input_layernorm.weight), rms_ref)

            got_gc = moe_combine8_sh_res_rows(got_dn, wts, shared=sh, residual=hp)
            ok &= check(f"L{L} B={B} gather+combine8+sh+res", got_gc, ref_out)

        # 2. Forced duplicate-expert collision stress (intra-row & cross-row, including all-same [17]*8!)
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

            dd = moe_dedup_rows(inds)
            ok &= check_lpt_dedup(f"L{L} B={B} coll lpt_dedup", inds, dd)

            got_m = moe_gate_up_swiglu_rows(
                xm, sw.gate_proj.weight, sw.gate_proj.scales, sw.up_proj.weight, sw.up_proj.scales, inds
            )
            ok &= check(f"L{L} B={B} coll gu inline", got_m, ref_m)
            got_m_dd = moe_gate_up_swiglu_rows(
                xm, sw.gate_proj.weight, sw.gate_proj.scales, sw.up_proj.weight, sw.up_proj.scales, inds, dedup=dd
            )
            ok &= check(f"L{L} B={B} coll gu dedup", got_m_dd, ref_m)

            got_full = moe_down_combine_rows(
                got_m, sw.down_proj.weight, sw.down_proj.scales, inds, wts, shared=sh, residual=hp
            )
            ok &= check(f"L{L} B={B} coll dc inline", got_full, ref_full)
            h_sp = moe_gate_up_swiglu_split_rows(
                xm, sw.gate_proj.weight, sw.gate_proj.scales, sw.up_proj.weight, sw.up_proj.scales, dd
            )
            got_full_sp = moe_down_combine_rows(
                h_sp, sw.down_proj.weight, sw.down_proj.scales, inds, wts, shared=sh, residual=hp, dedup=dd
            )
            ok &= check(f"L{L} B={B} coll 2-PSO split+dc", got_full_sp, ref_full)

        # 3. High-B tests (B=5,6,8,9,12,16,20,24): real router + fused router_dedup + 1-PSO + 2-PSO + T7 se_down + T6
        for B in (5, 6, 8, 9, 12, 16, 20, 24):
            base = (mx.random.normal((1, 4096)) * 0.25).astype(mx.bfloat16)
            xm = (base + (mx.random.normal((B, 4096)) * 0.12)).astype(mx.bfloat16)
            hp = (mx.random.normal((B, 4096)) * 0.25).astype(mx.bfloat16)
            se_mid = gemv_lin_swiglu(xm, se.gate_proj, se.up_proj)
            sh = gemv_lin(se_mid, se.down_proj)
            inds_ref, wts_ref = gate_rows(
                xm, gate.weight, gate.expert_bias,
                top_k=gate.top_k, n_group=gate.n_group, topk_group=gate.topk_group,
                routed_scaling_factor=gate.routed_scaling_factor,
            )
            inds_d, wts_d, dd = gate_dedup_rows(
                xm, gate.weight, gate.expert_bias,
                top_k=gate.top_k, n_group=gate.n_group, topk_group=gate.topk_group,
                routed_scaling_factor=gate.routed_scaling_factor,
            )
            mx.eval(xm, hp, se_mid, sh, inds_ref, wts_ref, inds_d, wts_d)
            ok &= check(f"L{L} B={B} gate_dedup inds", inds_d, inds_ref)
            ok &= check(f"L{L} B={B} gate_dedup wts", wts_d, wts_ref)
            ok &= check_lpt_dedup(f"L{L} B={B} lpt_dedup", inds_d, dd)

            _, _, ref_m, _, ref_comb = stock_routed_b1(layer, xm, inds_ref, wts_ref)
            ref_full = (ref_comb + sh) + hp

            got_m_flat = moe_gate_up_swiglu_rows(
                xm, sw.gate_proj.weight, sw.gate_proj.scales, sw.up_proj.weight, sw.up_proj.scales,
                inds_ref, dedup=dd, adaptive=0,
            )
            got_m_lean2 = moe_gate_up_swiglu_rows(
                xm, sw.gate_proj.weight, sw.gate_proj.scales, sw.up_proj.weight, sw.up_proj.scales,
                inds_ref, dedup=dd, adaptive=1,
            )
            ok &= check(f"L{L} B={B} gu flat (mode=0)", got_m_flat, ref_m)
            ok &= check(f"L{L} B={B} gu lean2 (mode=1)", got_m_lean2, ref_m)

            got_f = moe_down_combine_rows(
                got_m_lean2, sw.down_proj.weight, sw.down_proj.scales, inds_ref, wts_ref,
                shared=sh, residual=hp, dedup=dd,
            )
            ok &= check(f"L{L} B={B} dc dedup+worksteal", got_f, ref_full)

            h_sp = moe_gate_up_swiglu_split_rows(
                xm, sw.gate_proj.weight, sw.gate_proj.scales, sw.up_proj.weight, sw.up_proj.scales, dd
            )
            got_sp_se, pss = moe_down_combine_rows(
                h_sp, sw.down_proj.weight, sw.down_proj.scales, inds_ref, wts_ref,
                residual=hp, dedup=dd,
                se_mid=se_mid, se_w_fp8=se.down_proj.weight_fp8, se_sc=se.down_proj.weight_scale_inv,
                return_part_ss=True,
            )
            ok &= check(f"L{L} B={B} 2-PSO + T7 se_down", got_sp_se, ref_full)
            rms_ref = mx.fast.rms_norm(ref_full, layer.input_layernorm.weight, 1e-6)
            ok &= check(f"L{L} B={B} T6 part_ss rms_norm", _rms_from_part_ss(got_sp_se, pss, layer.input_layernorm.weight), rms_ref)

        # 4. Forced exact cnt=1,2,3,4,5,8,12,16 collisions across B=16 (dedup, inline, 2-PSO)
        xm = (mx.random.normal((16, 4096)) * 0.25).astype(mx.bfloat16)
        hp = (mx.random.normal((16, 4096)) * 0.25).astype(mx.bfloat16)
        sh = (mx.random.normal((16, 4096)) * 0.25).astype(mx.bfloat16)
        wts = mx.softmax(mx.random.normal((16, 8)), axis=-1) * 2.5
        forced_rows = []
        for b in range(16):
            row = [100 + b * 8 + j for j in range(8)]
            if b < 16:
                row[0] = 10
            if b < 12:
                row[1] = 20
            if b < 8:
                row[2] = 30
            if b < 5:
                row[3] = 40
            if b < 4:
                row[4] = 50
            if b < 3:
                row[5] = 60
            if b < 2:
                row[6] = 70
            forced_rows.append(row)
        inds_f = mx.array(forced_rows, dtype=mx.uint32)
        mx.eval(xm, hp, sh, wts, inds_f)
        _, _, ref_mf, _, ref_combf = stock_routed_b1(layer, xm, inds_f, wts)
        ref_fullf = (ref_combf + sh) + hp
        dd_f = moe_dedup_rows(inds_f)
        ok &= check_lpt_dedup(f"L{L} B=16 forced lpt_dedup", inds_f, dd_f)
        for mode, dd_arg in (("dedup", dd_f), ("inline", None)):
            gm = moe_gate_up_swiglu_rows(
                xm, sw.gate_proj.weight, sw.gate_proj.scales, sw.up_proj.weight, sw.up_proj.scales,
                inds_f, dedup=dd_arg, adaptive=1,
            )
            ok &= check(f"L{L} B=16 forced({mode}) gu", gm, ref_mf)
            gf = moe_down_combine_rows(
                gm, sw.down_proj.weight, sw.down_proj.scales, inds_f, wts,
                shared=sh, residual=hp, dedup=dd_arg,
            )
            ok &= check(f"L{L} B=16 forced({mode}) dc", gf, ref_fullf)
        h_spf = moe_gate_up_swiglu_split_rows(
            xm, sw.gate_proj.weight, sw.gate_proj.scales, sw.up_proj.weight, sw.up_proj.scales, dd_f
        )
        gf_sp = moe_down_combine_rows(
            h_spf, sw.down_proj.weight, sw.down_proj.scales, inds_f, wts,
            shared=sh, residual=hp, dedup=dd_f,
        )
        ok &= check(f"L{L} B=16 forced 2-PSO split+dc", gf_sp, ref_fullf)

        # 5. T3 Runtime-on-load packed/unpacked bit-exactness (on L4)
        if L == 4:
            print("=== T3 Packed Weight Round-Trip & Kernel Parity (L4) ===", flush=True)
            gu_pk = pack_moe_gu_weights(sw.gate_proj.weight, sw.gate_proj.scales, sw.up_proj.weight, sw.up_proj.scales)
            dn_pk = pack_moe_dn_weights(sw.down_proj.weight, sw.down_proj.scales)
            mx.eval(gu_pk, dn_pk)
            uwg, usg, uwu, usu = unpack_moe_gu_weights(gu_pk)
            uwd, usd = unpack_moe_dn_weights(dn_pk)
            mx.eval(uwg, usg, uwu, usu, uwd, usd)
            ok &= check("T3 unpack wg round-trip", uwg, sw.gate_proj.weight)
            ok &= check("T3 unpack sg round-trip", usg, sw.gate_proj.scales)
            ok &= check("T3 unpack wu round-trip", uwu, sw.up_proj.weight)
            ok &= check("T3 unpack su round-trip", usu, sw.up_proj.scales)
            ok &= check("T3 unpack wd round-trip", uwd, sw.down_proj.weight)
            ok &= check("T3 unpack sd round-trip", usd, sw.down_proj.scales)

            for B in (1, 4, 5, 8, 12, 16):
                base = (mx.random.normal((1, 4096)) * 0.25).astype(mx.bfloat16)
                xm = (base + (mx.random.normal((B, 4096)) * 0.12)).astype(mx.bfloat16)
                hp = (mx.random.normal((B, 4096)) * 0.25).astype(mx.bfloat16)
                se_mid = gemv_lin_swiglu(xm, se.gate_proj, se.up_proj)
                sh = gemv_lin(se_mid, se.down_proj)
                inds_p, wts_p, dd_p = gate_dedup_rows(
                    xm, gate.weight, gate.expert_bias,
                    top_k=gate.top_k, n_group=gate.n_group, topk_group=gate.topk_group,
                    routed_scaling_factor=gate.routed_scaling_factor,
                )
                mx.eval(xm, hp, se_mid, sh, inds_p, wts_p)
                _, _, ref_m, _, ref_comb = stock_routed_b1(layer, xm, inds_p, wts_p)
                ref_full = (ref_comb + sh) + hp
                gm_pk = moe_gate_up_swiglu_packed_rows(xm, gu_pk, dd_p)
                gf_pk, pss_pk = moe_down_combine_packed_rows(
                    gm_pk, dn_pk, dd_p, wts_p, residual=hp,
                    se_mid=se_mid, se_w_fp8=se.down_proj.weight_fp8, se_sc=se.down_proj.weight_scale_inv,
                    return_part_ss=True,
                )
                ok &= check(f"T3 B={B:2d} gu_packed", gm_pk, ref_m)
                ok &= check(f"T3 B={B:2d} dc_packed+se_down", gf_pk, ref_full)
                rms_ref = mx.fast.rms_norm(ref_full, layer.input_layernorm.weight, 1e-6)
                ok &= check(f"T3 B={B:2d} dc_packed+T6 part_ss", _rms_from_part_ss(gf_pk, pss_pk, layer.input_layernorm.weight), rms_ref)

    print("SUMMARY:", "ALL PASS" if ok else "FAIL", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
