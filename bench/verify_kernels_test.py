"""Bit-exactness & speed test for Phase 1 multi-token verification kernels (B = 1..16, C*L <= 16)."""
import sys
import time
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(_root / "vendor"), str(_root)]

import mlx_lm  # noqa: E402,F401
import mlx.core as mx  # noqa: E402
from mlx_lm.layers.common import swiglu  # noqa: E402
from mlx_lm.models.bailing_moe_v3.rows.gemv_fp8 import (  # noqa: E402
    gemv_lin,
    gemv_lin_multi,
    gemv_lin_res,
    gemv_lin_swiglu,
)
from mlx_lm.models.bailing_moe_v3.rows.kda import (  # noqa: E402
    kda_gdn_advance_only,
    kda_gdn_advance_split_only,
    kda_gdn_verify_seq_rows,
    kda_gdn_verify_seq_split_rows,
    kda_pre_seq_rows,
    kda_slice_conv_rows,
)
from mlx_lm.models.bailing_moe_v3.rows.mla import mla_verify_layer_rows, mla_verify_varlen_rows  # noqa: E402
from mlx_lm.models.bailing_moe_v3.rows.moe import (  # noqa: E402
    moe_dedup_rows,
    moe_down_combine_rows,
    moe_gate_up_swiglu_rows,
)
from mlx_lm.models.bailing_moe_v3.rows.norm import rms_norm_rows  # noqa: E402
from mlx_lm.models.bailing_moe_v3.rows.router import gate_rows  # noqa: E402
from mlx_lm.models.cache import ArraysCache, BatchKVCache, KVCache  # noqa: E402

from bench._layer_loader import bit_diff_count, load_head, load_layer, load_visits  # noqa: E402


def check(name, got, ref):
    d = bit_diff_count(got, ref)
    print(f"  {name:52s} {str(got.shape):16s} {'PASS' if d == 0 else f'FAIL ({d}/{ref.size})'}", flush=True)
    return d == 0


def bench_ms(fn, warm=5, iters=20):
    for _ in range(warm):
        mx.eval(fn())
    mx.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


def test_gemv(l0, l4, l7, norm, lm_head):
    print("\n=== 1. GEMV & Router B in (1..16, 20, 24) ===", flush=True)
    ok = True
    mx.random.seed(7)
    for B in (1, 4, 6, 8, 10, 12, 13, 14, 15, 16, 20, 24):
        x = (mx.random.normal((B, 4096)) * 0.25).astype(mx.bfloat16)
        v_dense = (mx.random.normal((B, 10240)) * 0.25).astype(mx.bfloat16)
        v_kda = (mx.random.normal((B, 8192)) * 0.25).astype(mx.bfloat16)
        qa = (mx.random.normal((B, 1536)) * 0.25).astype(mx.bfloat16)
        mx.eval(x, v_dense, v_kda, qa)

        # Dense SwiGLU + Shared SwiGLU + down_res
        ref_sw = mx.concatenate([swiglu(l0.mlp.gate_proj(x[b : b + 1]), l0.mlp.up_proj(x[b : b + 1])) for b in range(B)], axis=0)
        got_sw = gemv_lin_swiglu(x, l0.mlp.gate_proj, l0.mlp.up_proj)
        ok &= check(f"gemv_swiglu(dense) B={B}", got_sw, ref_sw)

        se = l4.mlp.shared_experts
        ref_sh = mx.concatenate([swiglu(se.gate_proj(x[b : b + 1]), se.up_proj(x[b : b + 1])) for b in range(B)], axis=0)
        got_sh = gemv_lin_swiglu(x, se.gate_proj, se.up_proj)
        ok &= check(f"gemv_swiglu(shared) B={B}", got_sh, ref_sh)

        ref_dr = mx.concatenate([l0.mlp.down_proj(v_dense[b : b + 1]) + x[b : b + 1] for b in range(B)], axis=0)
        got_dr = gemv_lin_res(v_dense, l0.mlp.down_proj, x)
        ok &= check(f"gemv_rows_res(dense) B={B}", got_dr, ref_dr)

        # KDA qkvfgb
        a4 = l4.attention
        ps = [a4.q_proj, a4.k_proj, a4.v_proj, a4.f_proj, a4.g_proj, a4.b_proj]
        gots = gemv_lin_multi(x, ps)
        for nm, g, p in zip(("q", "k", "v", "f", "g", "b"), gots, ps):
            ref = mx.concatenate([p(x[b : b + 1]) for b in range(B)], axis=0)
            ok &= check(f"kda.{nm}_proj B={B}", g, ref)

        if B in (1, 4, 8, 12, 14, 15, 16, 20, 24):
            # MLA q_b + router + lm_head
            ref_qb = mx.concatenate([l7.attention.q_b_proj(qa[b : b + 1]) for b in range(B)], axis=0)
            got_qb = gemv_lin(qa, l7.attention.q_b_proj)
            ok &= check(f"mla.q_b_proj B={B}", got_qb, ref_qb)

            ref_inds, ref_wts = [], []
            for b in range(B):
                i, w = l4.mlp.gate(x[b : b + 1])
                ref_inds.append(i.astype(mx.uint32))
                ref_wts.append(w)
            got_inds, got_wts = gate_rows(x, l4.mlp.gate.weight, l4.mlp.gate.expert_bias)
            ok &= check(f"gate_rows.inds B={B}", got_inds, mx.concatenate(ref_inds, axis=0))
            ok &= check(f"gate_rows.wts B={B}", got_wts, mx.concatenate(ref_wts, axis=0))

            # lm_head
            hn = rms_norm_rows(x, norm.weight, norm.eps)
            ref_lh = mx.concatenate([lm_head(hn[b : b + 1]) for b in range(B)], axis=0)
            got_lh = gemv_lin(hn, lm_head)
            ok &= check(f"lm_head B={B}", got_lh, ref_lh)

    print("  --- GEMV qkvfgb & lm_head micro-timing (us) ---", flush=True)
    a4 = l4.attention
    ps = [a4.q_proj, a4.k_proj, a4.v_proj, a4.f_proj, a4.g_proj, a4.b_proj]
    for B in (8, 12, 14, 15, 16, 20, 24):
        x = (mx.random.normal((B, 4096)) * 0.25).astype(mx.bfloat16)
        mx.eval(x)
        t_qkv = bench_ms(lambda: gemv_lin_multi(x, ps))
        t_lh = bench_ms(lambda: gemv_lin(x, lm_head))
        print(f"    B={B:2d}: kda_qkvfgb={t_qkv*1000:.0f}us  lm_head={t_lh*1000:.0f}us", flush=True)
    return ok


def stock_routed_b1(layer, xm, inds, wts):
    B = xm.shape[0]
    sw = layer.mlp.switch_mlp
    mids, comb_outs = [], []
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
        mids.append(m[None])
        comb_outs.append(routed[None])
    return mx.concatenate(mids, axis=0), mx.concatenate(comb_outs, axis=0)


def test_moe(l4, l64):
    print("\n=== 2. MoE B in (1, 4, 8, 12, 16, 20, 24) + Collision Stress ===", flush=True)
    ok = True
    mx.random.seed(11)
    sw = l4.mlp.switch_mlp
    se = l4.mlp.shared_experts
    for B in (1, 4, 8, 12, 16, 20, 24):
        xm = (mx.random.normal((B, 4096)) * 0.25).astype(mx.bfloat16)
        hp = (mx.random.normal((B, 4096)) * 0.25).astype(mx.bfloat16)
        # Heavy overlap across rows & within rows to exercise multi-chunk NPAIR > 32 dedup
        inds = mx.array([[((b * 3 + k) % 24) if (b % 2 == 0) else (17 if k < 3 else (b * 11 + k * 7) % 512) for k in range(8)] for b in range(B)], dtype=mx.uint32)
        wts = mx.softmax(mx.random.normal((B, 8)), axis=-1) * 2.5
        sh = gemv_lin(gemv_lin_swiglu(xm, se.gate_proj, se.up_proj), se.down_proj)
        mx.eval(xm, hp, inds, wts, sh)

        ref_m, ref_comb = stock_routed_b1(l4, xm, inds, wts)
        ref_full = (ref_comb + sh) + hp

        for gu_tm in (1, 2, 4) if B in (8, 16) else (4,):
            got_m = moe_gate_up_swiglu_rows(xm, sw.gate_proj.weight, sw.gate_proj.scales, sw.up_proj.weight, sw.up_proj.scales, inds, tmax=gu_tm)
            ok &= check(f"L4 B={B} gu_sw(tm={gu_tm})", got_m, ref_m)
        dedup_b = moe_dedup_rows(inds)
        got_m_dd = moe_gate_up_swiglu_rows(xm, sw.gate_proj.weight, sw.gate_proj.scales, sw.up_proj.weight, sw.up_proj.scales, inds, dedup=dedup_b)
        ok &= check(f"L4 B={B} gu_sw_dedup", got_m_dd, ref_m)
        for dc_tm, dc_nsg in ((1, 2), (2, 4), (1, 8)) if B in (8, 12, 16, 20) else ((1, 2),):
            got_full = moe_down_combine_rows(ref_m, sw.down_proj.weight, sw.down_proj.scales, inds, wts, shared=sh, residual=hp, tmax=dc_tm, nsg=dc_nsg)
            ok &= check(f"L4 B={B} dn_comb(tm={dc_tm},nsg={dc_nsg})", got_full, ref_full)
        got_dc_dd = moe_down_combine_rows(ref_m, sw.down_proj.weight, sw.down_proj.scales, inds, wts, shared=sh, residual=hp, dedup=dedup_b)
        ok &= check(f"L4 B={B} dn_comb_dedup", got_dc_dd, ref_full)

    # All-same expert edge case at B=24 (NPAIR=192 all identical!)
    xm24 = (mx.random.normal((24, 4096)) * 0.25).astype(mx.bfloat16)
    hp24 = (mx.random.normal((24, 4096)) * 0.25).astype(mx.bfloat16)
    inds24 = mx.full((24, 8), 42, dtype=mx.uint32)
    wts24 = mx.full((24, 8), 0.3125, dtype=mx.float32)
    mx.eval(xm24, hp24, inds24, wts24)
    ref_m24, ref_c24 = stock_routed_b1(l64, xm24, inds24, wts24)
    sw64 = l64.mlp.switch_mlp
    dedup24 = moe_dedup_rows(inds24)
    got_m24 = moe_gate_up_swiglu_rows(xm24, sw64.gate_proj.weight, sw64.gate_proj.scales, sw64.up_proj.weight, sw64.up_proj.scales, inds24, dedup=dedup24)
    got_f24 = moe_down_combine_rows(got_m24, sw64.down_proj.weight, sw64.down_proj.scales, inds24, wts24, residual=hp24, dedup=dedup24)
    ok &= check("L64 B=24 all-dup gu_sw_dedup", got_m24, ref_m24)
    ok &= check("L64 B=24 all-dup dn_comb_dedup", got_f24, ref_c24 + hp24)

    # Speed sweep on L4 real-like 25-unique-expert pattern for B=8, 12, 16, 20
    print("  --- MoE TMAX/NSG micro-timing (us) ---", flush=True)
    for B in (8, 12, 16, 20):
        xm = (mx.random.normal((B, 4096)) * 0.25).astype(mx.bfloat16)
        hp = (mx.random.normal((B, 4096)) * 0.25).astype(mx.bfloat16)
        sh = (mx.random.normal((B, 4096)) * 0.25).astype(mx.bfloat16)
        inds = mx.array([[(k * 7 + b * 3) % 28 for k in range(8)] for b in range(B)], dtype=mx.uint32)
        wts = mx.full((B, 8), 0.3125, dtype=mx.float32)
        mx.eval(xm, hp, sh, inds, wts)
        hm = moe_gate_up_swiglu_rows(xm, sw.gate_proj.weight, sw.gate_proj.scales, sw.up_proj.weight, sw.up_proj.scales, inds)
        mx.eval(hm)
        parts = []
        for tm in (1, 2, 4):
            t_gu = bench_ms(lambda tm=tm: moe_gate_up_swiglu_rows(xm, sw.gate_proj.weight, sw.gate_proj.scales, sw.up_proj.weight, sw.up_proj.scales, inds, tmax=tm))
            parts.append(f"gu_tm{tm}={t_gu*1000:.0f}us")
        for tm in (1, 2):
            for nsg in (2, 4, 8):
                t_dc = bench_ms(lambda tm=tm, nsg=nsg: moe_down_combine_rows(hm, sw.down_proj.weight, sw.down_proj.scales, inds, wts, shared=sh, residual=hp, tmax=tm, nsg=nsg))
                parts.append(f"dc_t{tm}_n{nsg}={t_dc*1000:.0f}us")
        print(f"    B={B:2d}: " + " ".join(parts), flush=True)
    return ok


def stock_kda_seq(layer, x_cl, cs_init):
    """Run C streams independently over L stock B=1 steps, capturing state after every step."""
    C, L, _ = x_cl.shape
    attn = layer.attention
    step_hpost = [[None] * L for _ in range(C)]
    step_states = [[None] * (L + 1) for _ in range(C)]
    for c in range(C):
        ac = ArraysCache(size=4)
        for j in range(4):
            ac[j] = mx.array(cs_init[c][j])
        step_states[c][0] = [mx.array(ac[j]) for j in range(4)]
        for t in range(L):
            xt = x_cl[c : c + 1, t : t + 1, :]
            hin = layer.input_layernorm(xt)
            ao = attn(hin, mask=None, cache=ac)
            step_hpost[c][t] = (xt + ao)[:, 0, :]
            step_states[c][t + 1] = [mx.array(ac[j]) for j in range(4)]
    ref_hpost = mx.concatenate([step_hpost[c][t] for c in range(C) for t in range(L)], axis=0)
    return ref_hpost, step_states


def test_kda(l0, l4):
    print("\n=== 3. KDA Sequential Verify + Replay Rollback ===", flush=True)
    ok = True
    mx.random.seed(23)
    for lname, layer, visits in [("L0", l0, load_visits(0, limit=4)), ("L4", l4, load_visits(4, limit=4))]:
        attn = layer.attention
        cases = [
            (1, 1), (1, 4), (1, 8), (1, 12), (1, 16), (1, 20),
            (2, 4), (4, 3), (4, 4), (4, 5), (4, 6),
        ]
        for C, L in cases:
            x_seq = (mx.random.normal((C, L, 4096)) * 0.25).astype(mx.bfloat16)
            cs_init = [[mx.array(visits[c][f"cs{j}"]) for j in range(4)] for c in range(C)]
            mx.eval(x_seq, *[a for cs in cs_init for a in cs])

            ref_hpost, step_states = stock_kda_seq(layer, x_seq, cs_init)

            # Fast path
            x_flat = x_seq.reshape(C * L, 4096)
            h_in = rms_norm_rows(x_flat, layer.input_layernorm.weight, layer.input_layernorm.eps)
            q_r, k_r, v_r, f_r, g_r, b_r = gemv_lin_multi(
                h_in, [attn.q_proj, attn.k_proj, attn.v_proj, attn.f_proj, attn.g_proj, attn.b_proj]
            )
            cs0 = mx.concatenate([cs_init[c][0] for c in range(C)], axis=0)
            cs1 = mx.concatenate([cs_init[c][1] for c in range(C)], axis=0)
            cs2 = mx.concatenate([cs_init[c][2] for c in range(C)], axis=0)
            cs3 = mx.concatenate([cs_init[c][3] for c in range(C)], axis=0)
            q, k, v, g, beta, h0, h1, h2 = kda_pre_seq_rows(
                q_r, k_r, v_r, f_r, b_r, cs0, cs1, cs2,
                attn.q_conv1d.weight, attn.k_conv1d.weight, attn.v_conv1d.weight,
                attn.A_log, attn.dt_bias, C=C, L=L, lower_bound=attn.lower_bound, eps=1e-6,
            )
            ov, s_end = kda_gdn_verify_seq_rows(q, k, v, g, beta, cs3, g_r, attn.o_norm.weight, C=C, L=L, eps=attn.o_norm.eps)
            got_hpost = gemv_lin_res(ov, attn.o_proj, x_flat)
            ok &= check(f"{lname} C={C} L={L:2d} h_post", got_hpost, ref_hpost)

            ref_s_end = mx.concatenate([step_states[c][L][3] for c in range(C)], axis=0)
            ok &= check(f"{lname} C={C} L={L:2d} s_end", s_end, ref_s_end)

            if C > 1:
                ov_sp, s_end_sp = kda_gdn_verify_seq_split_rows(
                    q, k, v, g, beta, [cs_init[c][3] for c in range(C)], g_r, attn.o_norm.weight, C=C, L=L, eps=attn.o_norm.eps
                )
                ok &= check(f"{lname} C={C} L={L:2d} split ov", ov_sp, ov)
                for c_i in range(C):
                    ok &= check(f"{lname} C={C} L={L:2d} split s{c_i}", s_end_sp[c_i], step_states[c_i][L][3])
                if C == 4 and L in (3, 5):
                    ncs_test = [min(L, x) for x in (1, 2, 1, 2)]
                    for rej_idx in ([0], [1, 3], [0, 2, 3], [0, 1, 2, 3]):
                        outs_rj = kda_gdn_advance_split_only(
                            k,
                            v,
                            g,
                            beta,
                            [cs_init[ci][3] for ci in rej_idx],
                            mx.array(rej_idx, dtype=mx.int32),
                            mx.array([ncs_test[ci] for ci in rej_idx], dtype=mx.int32),
                            L=L,
                        )
                        for ci, o in zip(rej_idx, outs_rj):
                            ok &= check(f"{lname} C=4 L={L} adv_split rej={rej_idx} c{ci}", o, step_states[ci][ncs_test[ci]][3])

            # Check conv + GDN rollback at varied n_consumed
            if C == 1:
                nc_checks = [[nc] for nc in sorted({1, max(1, L // 2), max(1, L - 1), L})]
            else:
                nc_checks = [[min(L, c + 1) for c in range(C)], [L] * C]
            for ncs in nc_checks:
                ns0, ns1, ns2 = kda_slice_conv_rows(h0, h1, h2, ncs)
                ns3 = kda_gdn_advance_only(None, k, v, g, beta, cs3, ncs, C=C, L=L)
                for j, got_ns in enumerate((ns0, ns1, ns2, ns3)):
                    ref_ns = mx.concatenate([step_states[c][ncs[c]][j] for c in range(C)], axis=0)
                    ok &= check(f"{lname} C={C} L={L} nc={ncs} cs{j}", got_ns, ref_ns)
    return ok


def stock_mla_seq(layer, x_cl, kv_prev):
    C, L, _ = x_cl.shape
    attn = layer.attention
    step_hpost = [[None] * L for _ in range(C)]
    final_kvs = []
    for c in range(C):
        kc = KVCache()
        kc.update_and_fetch(mx.array(kv_prev[c][0]), mx.array(kv_prev[c][1]))
        for t in range(L):
            xt = x_cl[c : c + 1, t : t + 1, :]
            hin = layer.input_layernorm(xt)
            ao = attn(hin, mask=None, cache=kc)
            step_hpost[c][t] = (xt + ao)[:, 0, :]
        final_kvs.append(kc.keys_and_values())
    return mx.concatenate([step_hpost[c][t] for c in range(C) for t in range(L)], axis=0), final_kvs


def test_mla(l7, l71):
    print("\n=== 4. MLA Multi-Token Causal Verify ===", flush=True)
    ok = True
    mx.random.seed(37)
    for lname, layer in [("L7", l7), ("L71", l71)]:
        for C, L, lens in [
            (1, 1, [128]),
            (1, 4, [128]),
            (1, 8, [512]),
            (1, 12, [128]),
            (1, 16, [256]),
            (1, 20, [128]),
            (4, 3, [512, 300, 450, 128]),
            (4, 5, [200, 150, 200, 90]),
        ]:
            x_seq = (mx.random.normal((C, L, 4096)) * 0.25).astype(mx.bfloat16)
            kv_prev = [
                (
                    (mx.random.normal((1, 1, s, 512)) * 0.1).astype(mx.bfloat16),
                    (mx.random.normal((1, 1, s, 64)) * 0.1).astype(mx.bfloat16),
                )
                for s in lens
            ]
            mx.eval(x_seq, *[a for pair in kv_prev for a in pair])
            ref_hp, ref_kvs = stock_mla_seq(layer, x_seq, kv_prev)

            # Test both KVCache/list[KVCache] AND BatchKVCache
            x_flat = x_seq.reshape(C * L, 4096)
            hin = rms_norm_rows(x_flat, layer.input_layernorm.weight, layer.input_layernorm.eps)

            kcs = []
            for kp, vp in kv_prev:
                kc = KVCache()
                kc.update_and_fetch(mx.array(kp), mx.array(vp))
                kcs.append(kc)
            cache_arg = kcs[0] if C == 1 else kcs
            got_hp = mla_verify_layer_rows(hin, x_flat, layer.attention, cache_arg, C=C, L=L)
            ok &= check(f"{lname} C={C} L={L:2d} list/kv h_post", got_hp, ref_hp)
            for c_i in range(C):
                kc_i = cache_arg if C == 1 else cache_arg[c_i]
                gk, gv = kc_i.keys_and_values()
                ok &= check(f"{lname} C={C} L={L:2d} s{c_i} k_all", gk, ref_kvs[c_i][0])
                ok &= check(f"{lname} C={C} L={L:2d} s{c_i} v_all", gv, ref_kvs[c_i][1])

            if C > 1:
                bcs = []
                for kp, vp in kv_prev:
                    kc = KVCache()
                    kc.update_and_fetch(mx.array(kp), mx.array(vp))
                    bcs.append(kc)
                bc = BatchKVCache.merge(bcs)
                got_hp_b = mla_verify_layer_rows(hin, x_flat, layer.attention, bc, C=C, L=L)
                ok &= check(f"{lname} C={C} L={L:2d} BatchKV h_post", got_hp_b, ref_hp)

        # Variable-length MTP refresh test
        for Ls, lens in [([3, 1, 2, 1], [100, 64, 150, 80]), ([2, 3, 1, 3], [250, 120, 90, 300])]:
            kv_prev = [
                (
                    (mx.random.normal((1, 1, s, 512)) * 0.1).astype(mx.bfloat16),
                    (mx.random.normal((1, 1, s, 64)) * 0.1).astype(mx.bfloat16),
                )
                for s in lens
            ]
            x_streams = [(mx.random.normal((1, lc, 4096)) * 0.25).astype(mx.bfloat16) for lc in Ls]
            mx.eval(*x_streams, *[a for pair in kv_prev for a in pair])

            ref_parts, ref_kvs = [], []
            for xs, pair in zip(x_streams, kv_prev):
                rh, rk = stock_mla_seq(layer, xs, [pair])
                ref_parts.append(rh)
                ref_kvs.append(rk[0])
            ref_hp = mx.concatenate(ref_parts, axis=0)

            x_flat = mx.concatenate([xs[0] for xs in x_streams], axis=0)
            hin = rms_norm_rows(x_flat, layer.input_layernorm.weight, layer.input_layernorm.eps)
            kcs = []
            for kp, vp in kv_prev:
                kc = KVCache()
                kc.update_and_fetch(mx.array(kp), mx.array(vp))
                kcs.append(kc)
            got_hp = mla_verify_varlen_rows(hin, x_flat, layer.attention, kcs, Ls)
            ok &= check(f"{lname} varlen Ls={Ls} h_post", got_hp, ref_hp)
            for c_i in range(len(Ls)):
                gk, gv = kcs[c_i].keys_and_values()
                ok &= check(f"{lname} varlen Ls={Ls} s{c_i} k", gk, ref_kvs[c_i][0])
                ok &= check(f"{lname} varlen Ls={Ls} s{c_i} v", gv, ref_kvs[c_i][1])
    return ok


def main():
    l0, _ = load_layer(0)
    l4, _ = load_layer(4)
    l64, _ = load_layer(64)
    l7, _ = load_layer(7)
    l71, _ = load_layer(71)
    norm, lm_head, _ = load_head()

    ok = True
    ok &= test_gemv(l0, l4, l7, norm, lm_head)
    ok &= test_moe(l4, l64)
    ok &= test_kda(l0, l4)
    ok &= test_mla(l7, l71)
    print("\nSUMMARY:", "ALL PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
