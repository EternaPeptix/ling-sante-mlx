"""Step 1: Unbiased pipelined B=12/B=16 ledger and instant bit-exact A/B kernel probe."""
import argparse
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
from mlx_lm.models.bailing_moe_v3.mtp import BailingMTPLayer  # noqa: E402
from mlx_lm.models.bailing_moe_v3.rows import gather as gather_mod  # noqa: E402
from mlx_lm.models.bailing_moe_v3.rows import gemv as gemv_mod  # noqa: E402
from mlx_lm.models.bailing_moe_v3.rows.gather import gather_rows  # noqa: E402
from mlx_lm.models.bailing_moe_v3.rows.gemv import (  # noqa: E402
    gemv_rows,
    gemv_rows_multi,
    gemv_rows_res,
    gemv_swiglu_rows,
)
from mlx_lm.models.bailing_moe_v3.rows.kda import (  # noqa: E402
    kda_gdn_verify_seq_rows,
    kda_gdn_verify_seq_split_rows,
    kda_pre_seq_rows,
)
from mlx_lm.models.bailing_moe_v3.rows.mla import (  # noqa: E402
    mla_multilinear_rows,
    mla_prep_kv_rope_rows,
)
from mlx_lm.models.bailing_moe_v3.rows.moe import (  # noqa: E402
    moe_combine8_sh_res_rows,
    moe_down_combine_rows,
    moe_gate_up_swiglu_rows,
)
from mlx_lm.models.bailing_moe_v3.rows.norm import add_rms_norm_rows, rms_norm_rows  # noqa: E402
from mlx_lm.models.bailing_moe_v3.rows.router import gate_rows  # noqa: E402
from mlx_lm.models.bailing_moe_v3.verify_decode import fast_verify_forward  # noqa: E402
from mlx_lm.utils import maybe_set_recommended_wired_limit  # noqa: E402

from bench._layer_loader import bit_diff_count  # noqa: E402
from bench.exact import DEFAULT_BASE, DEFAULT_MODEL, PROMPTS  # noqa: E402
from bench.profile_verify import capture_real_batch  # noqa: E402


def _flat(x):
    if isinstance(x, (list, tuple)):
        out = []
        for v in x:
            out.extend(_flat(v))
        return out
    return [x]


def pipe_ms(fn, n_lo: int = 4, n_hi: int = 24, iters: int = 4) -> float:
    """Measure true pipelined GPU time per call via (T(n_hi) - T(n_lo)) / (n_hi - n_lo)."""
    # Warmup
    mx.eval(*[a for _ in range(2) for a in _flat(fn())])
    mx.synchronize()
    dts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        mx.eval(*[a for _ in range(n_lo) for a in _flat(fn())])
        t_lo = time.perf_counter() - t0
        t1 = time.perf_counter()
        mx.eval(*[a for _ in range(n_hi) for a in _flat(fn())])
        t_hi = time.perf_counter() - t1
        dts.append((t_hi - t_lo) * 1e3 / (n_hi - n_lo))
    return round(statistics.median(dts), 4)


def single_ms(fn, warm: int = 2, iters: int = 5) -> float:
    for _ in range(warm):
        mx.eval(*_flat(fn()))
    mx.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        mx.eval(*_flat(fn()))
        ts.append((time.perf_counter() - t0) * 1e3)
    return round(statistics.median(ts), 4)


def run_probe_for_B(model, mtp, prompts_data, C: int, K: int):
    L = K + 1
    B = C * L
    print(f"\n================ PIPELINED PROBE C={C}, K={K}, L={L}, B={B} ================", flush=True)
    samp = capture_real_batch(model, mtp, prompts_data, C=C, K=K)

    # True e2e verify forward
    inp2 = mx.array([[lt] + samp["draft_lists"][c] for c, lt in enumerate(samp["last_toks"])], dtype=mx.uint32)
    def run_e2e():
        lg, hd = fast_verify_forward(model, inp2, samp["vcache"], return_hidden=True, return_rollback=False)
        for c in samp["vcache"]:
            if isinstance(c, list):
                for ci in c:
                    ci.trim(L)
        return lg, hd
    t_e2e = single_ms(run_e2e, warm=2, iters=5)
    print(f"E2E fast_verify_forward: {t_e2e:.2f} ms", flush=True)

    # 1. Unbiased pipelined ledger
    l0 = model.model.layers[0]
    l4 = model.model.layers[4]
    a4 = l4.attention
    k4 = samp["kda4"]
    qkv_ws = [a4.q_proj.weight, a4.k_proj.weight, a4.v_proj.weight, a4.f_proj.weight, a4.g_proj.weight, a4.b_proj.weight]

    t_rms_fast = pipe_ms(lambda: mx.fast.rms_norm(k4["x"], l4.input_layernorm.weight, l4.input_layernorm.eps))
    t_rms_row = pipe_ms(lambda: rms_norm_rows(k4["x"], l4.input_layernorm.weight, l4.input_layernorm.eps))
    t_qkv = pipe_ms(lambda: gemv_rows_multi(k4["h_in"], qkv_ws))
    t_kpre = pipe_ms(
        lambda: kda_pre_seq_rows(
            k4["q_r"], k4["k_r"], k4["v_r"], k4["f_r"], k4["b_r"],
            k4["cs"][0], k4["cs"][1], k4["cs"][2],
            a4.q_conv1d.weight, a4.k_conv1d.weight, a4.v_conv1d.weight,
            a4.A_log, a4.dt_bias, C=C, L=L, lower_bound=a4.lower_bound, eps=1e-6,
        )
    )
    t_kgdn = pipe_ms(
        lambda: kda_gdn_verify_seq_rows(
            k4["q"], k4["k"], k4["v"], k4["g"], k4["beta"], k4["cs"][3], k4["g_r"],
            a4.o_norm.weight, C=C, L=L, eps=a4.o_norm.eps,
        )
    )
    s_split = [mx.array(k4["cs"][3][c : c + 1]) for c in range(C)]
    mx.eval(*s_split)
    t_kgdn_split = pipe_ms(
        lambda: kda_gdn_verify_seq_split_rows(
            k4["q"], k4["k"], k4["v"], k4["g"], k4["beta"], s_split, k4["g_r"],
            a4.o_norm.weight, C=C, L=L, eps=a4.o_norm.eps,
        )
    )
    t_kop = pipe_ms(lambda: gemv_rows_res(k4["ov"], a4.o_proj.weight, k4["x"]))

    # MLA 7 (non-mutating view over kv_snap + dummy new K,V concatenated)
    l7 = model.model.layers[7]
    a7 = l7.attention
    m7 = samp["mla7"]
    t_m_qa = pipe_ms(lambda: gemv_rows_multi(m7["h_in"], [a7.q_a_proj.weight, a7.kv_a_proj_with_mqa.weight, a7.g_proj.weight]))
    q_a, kv_a_full, g_raw = gemv_rows_multi(m7["h_in"], [a7.q_a_proj.weight, a7.kv_a_proj_with_mqa.weight, a7.g_proj.weight])
    q_a_norm = mx.fast.rms_norm(q_a, a7.q_a_layernorm.weight, a7.q_a_layernorm.eps)
    t_m_qb = pipe_ms(lambda: gemv_rows(q_a_norm, a7.q_b_proj.weight))
    q_full = gemv_rows(q_a_norm, a7.q_b_proj.weight)
    offs_m = mx.array([k_s[0].shape[2] + t for k_s in m7["kv_snap"] for t in range(L)], dtype=mx.int32)
    t_m_prep = pipe_ms(
        lambda: mla_prep_kv_rope_rows(
            q_full, kv_a_full, a7.kv_a_layernorm.weight, offs_m,
            kv_eps=a7.kv_a_layernorm.eps, base=float(a7.rope.base), scale=float(a7.rope.scale),
        )
    )
    q_nope, q_rope, kv_latent, k_rope = mla_prep_kv_rope_rows(
        q_full, kv_a_full, a7.kv_a_layernorm.weight, offs_m,
        kv_eps=a7.kv_a_layernorm.eps, base=float(a7.rope.base), scale=float(a7.rope.scale),
    )
    t_m_eq = pipe_ms(lambda: mla_multilinear_rows(q_nope, a7.embed_q.weight))
    q_nope_e = mla_multilinear_rows(q_nope, a7.embed_q.weight)
    kl_fulls = [mx.concatenate([ks[0], kv_latent[c * L : (c + 1) * L].reshape(1, 1, L, 512)], axis=2) for c, ks in enumerate(m7["kv_snap"])]
    kr_fulls = [mx.concatenate([ks[1], k_rope[c * L : (c + 1) * L].reshape(1, 1, L, 64)], axis=2) for c, ks in enumerate(m7["kv_snap"])]
    mx.eval(q_nope_e, q_rope, q_full, kv_a_full, g_raw, offs_m, *kl_fulls, *kr_fulls)

    def mla_sdpa_pipe():
        outs = []
        for c_i in range(C):
            off_c = m7["kv_snap"][c_i][0].shape[2]
            for t in range(L):
                b = c_i * L + t
                end = off_c + t + 1
                kl_t = kl_fulls[c_i][:, :, :end, :]
                kr_t = kr_fulls[c_i][:, :, :end, :]
                pe = (q_rope[b : b + 1] * a7.scale) @ kr_t.swapaxes(-1, -2)
                sc = (q_nope_e[b : b + 1] * a7.scale) @ kl_t.swapaxes(-1, -2) + pe
                pr = mx.softmax(sc, axis=-1, precise=True)
                outs.append(pr @ kl_t)
        return mx.concatenate(outs, axis=0)

    out_sdpa = mla_sdpa_pipe()
    mx.eval(out_sdpa)
    t_m_sdpa = pipe_ms(mla_sdpa_pipe)
    t_m_ue = pipe_ms(lambda: mla_multilinear_rows(out_sdpa, a7.unembed_out.weight, g_raw=g_raw))
    gated = mla_multilinear_rows(out_sdpa, a7.unembed_out.weight, g_raw=g_raw)
    mx.eval(gated)
    t_m_dn = pipe_ms(lambda: gemv_rows_res(gated, a7.dense.weight, m7["x"]))

    # MoE (68 layers!) + Dense (4 layers!)
    m4 = samp["moe"]["moe4"]
    sw4 = l4.mlp.switch_mlp
    se4 = l4.mlp.shared_experts
    t_gate = pipe_ms(
        lambda: gate_rows(
            m4["mlp_in"], l4.mlp.gate.weight, l4.mlp.gate.expert_bias,
            top_k=l4.mlp.gate.top_k, n_group=l4.mlp.gate.n_group,
            topk_group=l4.mlp.gate.topk_group, routed_scaling_factor=l4.mlp.gate.routed_scaling_factor,
        )
    )
    t_sh = pipe_ms(lambda: gemv_rows(gemv_swiglu_rows(m4["mlp_in"], se4.gate_proj.weight, se4.up_proj.weight), se4.down_proj.weight))
    t_gu = pipe_ms(lambda: moe_gate_up_swiglu_rows(m4["mlp_in"], sw4.gate_proj.weight, sw4.gate_proj.scales, sw4.up_proj.weight, sw4.up_proj.scales, m4["inds"]))
    t_dc = pipe_ms(lambda: moe_down_combine_rows(m4["h_mid"], sw4.down_proj.weight, sw4.down_proj.scales, m4["inds"], m4["wts"], shared=m4["sh"], residual=m4["h_post"]))

    d0 = samp["dense0"]
    ml0 = l0.mlp
    t_d0 = pipe_ms(lambda: gemv_rows_res(gemv_swiglu_rows(d0["mlp_in"], ml0.gate_proj.weight, ml0.up_proj.weight), ml0.down_proj.weight, d0["h_post"]))

    hd = samp["head"]
    t_lh = pipe_ms(lambda: gemv_rows(hd["hidden"], model.lm_head.weight))
    t_lse = pipe_ms(lambda: mx.argmax(hd["logits"] - mx.logsumexp(hd["logits"], axis=-1, keepdims=True), axis=-1))

    ledger = {
        "rms_norm_x145": round(145 * t_rms_fast, 2),
        "rms_rows_x145": round(145 * t_rms_row, 2),
        "kda_qkvfgb_x63": round(63 * t_qkv, 2),
        "kda_pre_x63": round(63 * t_kpre, 2),
        "kda_gdn_merged_x63": round(63 * t_kgdn, 2),
        "kda_gdn_split_x63": round(63 * t_kgdn_split, 2),
        "kda_oproj_x63": round(63 * t_kop, 2),
        "mla_proj_prep_x9": round(9 * (t_m_qa + t_m_qb + t_m_prep + t_m_eq + t_m_ue + t_m_dn), 2),
        "mla_sdpa_x9": round(9 * t_m_sdpa, 2),
        "moe_gate_x68": round(68 * t_gate, 2),
        "moe_shared_x68": round(68 * t_sh, 2),
        "moe_gu_x68": round(68 * t_gu, 2),
        "moe_dc_x68": round(68 * t_dc, 2),
        "dense_x4": round(4 * t_d0, 2),
        "lm_head": t_lh,
        "lse_argmax": t_lse,
    }
    ledger_sum = round(
        ledger["rms_norm_x145"]
        + ledger["kda_qkvfgb_x63"]
        + ledger["kda_pre_x63"]
        + ledger["kda_gdn_split_x63"]
        + ledger["kda_oproj_x63"]
        + ledger["mla_proj_prep_x9"]
        + ledger["mla_sdpa_x9"]
        + ledger["moe_gate_x68"]
        + ledger["moe_shared_x68"]
        + ledger["moe_gu_x68"]
        + ledger["moe_dc_x68"]
        + ledger["dense_x4"]
        + ledger["lm_head"],
        2,
    )
    print(f"  LEDGER SUM = {ledger_sum:.2f}ms vs E2E = {t_e2e:.2f}ms", flush=True)
    for k, v in ledger.items():
        print(f"    {k:22s}: {v:6.2f} ms", flush=True)

    # 2. A/B MoE down_proj: moe_down_combine vs gather_rows(cap=2,4,8) + combine8
    moe_keys = list(samp["moe"].keys())
    moe_ab = {"dc_cur_x68": [], "gather_cap": {2: [], 4: [], 8: []}, "gather_only_cap": {2: [], 4: [], 8: []}, "diffs": {2: 0, 4: 0, 8: 0}}
    old_cap = gather_mod._TMAX_CAP
    try:
        for lk in moe_keys:
            s = samp["moe"][lk]
            sw = model.model.layers[s["layer_idx"]].mlp.switch_mlp
            ref_out = moe_down_combine_rows(s["h_mid"], sw.down_proj.weight, sw.down_proj.scales, s["inds"], s["wts"], shared=s["sh"], residual=s["h_post"])
            mx.eval(ref_out)
            moe_ab["dc_cur_x68"].append(
                pipe_ms(lambda s=s, sw=sw: moe_down_combine_rows(s["h_mid"], sw.down_proj.weight, sw.down_proj.scales, s["inds"], s["wts"], shared=s["sh"], residual=s["h_post"]))
            )
            for cap in (2, 4, 8):
                gather_mod._TMAX_CAP = cap
                dn = gather_rows(s["h_mid"], sw.down_proj.weight, sw.down_proj.scales, None, s["inds"])
                out = moe_combine8_sh_res_rows(dn, s["wts"], shared=s["sh"], residual=s["h_post"])
                mx.eval(out)
                moe_ab["diffs"][cap] += bit_diff_count(out, ref_out)
                moe_ab["gather_only_cap"][cap].append(
                    pipe_ms(lambda s=s, sw=sw: gather_rows(s["h_mid"], sw.down_proj.weight, sw.down_proj.scales, None, s["inds"]))
                )
                moe_ab["gather_cap"][cap].append(
                    pipe_ms(lambda s=s, sw=sw: moe_combine8_sh_res_rows(gather_rows(s["h_mid"], sw.down_proj.weight, sw.down_proj.scales, None, s["inds"]), s["wts"], shared=s["sh"], residual=s["h_post"]))
                )
    finally:
        gather_mod._TMAX_CAP = old_cap

    dc_cur_mean = round(68 * sum(moe_ab["dc_cur_x68"]) / len(moe_keys), 2)
    moe_ab_summary = {"dc_cur_x68": dc_cur_mean}
    for cap in (2, 4, 8):
        g_m = round(68 * sum(moe_ab["gather_only_cap"][cap]) / len(moe_keys), 2)
        gc_m = round(68 * sum(moe_ab["gather_cap"][cap]) / len(moe_keys), 2)
        moe_ab_summary[f"gather_cap{cap}"] = {"gather_x68": g_m, "gather+comb_x68": gc_m, "diffs": moe_ab["diffs"][cap]}
    print("  MoE down_proj A/B (x68 ms):", moe_ab_summary, flush=True)

    # 3. MoE gate_up pipelined (tmax=2,4 x nsg=2,4)
    gu_pipe = {}
    for tm, nsg in ((2, 2), (4, 2), (2, 4), (4, 4)):
        vals = []
        diffs = 0
        for lk in moe_keys:
            s = samp["moe"][lk]
            sw = model.model.layers[s["layer_idx"]].mlp.switch_mlp
            ref = moe_gate_up_swiglu_rows(s["mlp_in"], sw.gate_proj.weight, sw.gate_proj.scales, sw.up_proj.weight, sw.up_proj.scales, s["inds"], tmax=2, nsg=2)
            got = moe_gate_up_swiglu_rows(s["mlp_in"], sw.gate_proj.weight, sw.gate_proj.scales, sw.up_proj.weight, sw.up_proj.scales, s["inds"], tmax=tm, nsg=nsg)
            mx.eval(ref, got)
            diffs += bit_diff_count(got, ref)
            vals.append(pipe_ms(lambda s=s, sw=sw, tm=tm, nsg=nsg: moe_gate_up_swiglu_rows(s["mlp_in"], sw.gate_proj.weight, sw.gate_proj.scales, sw.up_proj.weight, sw.up_proj.scales, s["inds"], tmax=tm, nsg=nsg)))
        gu_pipe[f"tm{tm}_n{nsg}"] = {"x68_ms": round(68 * sum(vals) / len(vals), 2), "diffs": diffs}
    print("  MoE gate_up pipelined (x68 ms):", gu_pipe, flush=True)

    # 4. GEMV sweep at current B (bm, tm) exactness + pipelined speed
    orig_cfg = gemv_mod._cfg
    gemv_sweep = {"non_swiglu": {}, "swiglu": {}}
    ref_qkv = gemv_rows_multi(k4["h_in"], qkv_ws)
    ref_op = gemv_rows_res(k4["ov"], a4.o_proj.weight, k4["x"])
    ref_lh = gemv_rows(hd["hidden"], model.lm_head.weight)
    ref_sh = gemv_swiglu_rows(m4["mlp_in"], se4.gate_proj.weight, se4.up_proj.weight)
    ref_d0 = gemv_swiglu_rows(d0["mlp_in"], ml0.gate_proj.weight, ml0.up_proj.weight)
    mx.eval(*ref_qkv, ref_op, ref_lh, ref_sh, ref_d0)
    try:
        for bm, tm in ((8, 2), (4, 2), (4, 4), (8, 4), (4, 1), (8, 1)):
            def custom_cfg(N, K, B_=1, is_swiglu=False, _bm=bm, _tm=tm):
                if K >= 16 * N:
                    return (1, 8, 1, 32, min(_tm, N), 4)
                return (_bm if N >= 4096 else 4, 1, 1, 32, 1 if N < _tm else _tm, 4)
            gemv_mod._cfg = custom_cfg
            try:
                g_qkv = gemv_rows_multi(k4["h_in"], qkv_ws)
                g_op = gemv_rows_res(k4["ov"], a4.o_proj.weight, k4["x"])
                g_lh = gemv_rows(hd["hidden"], model.lm_head.weight)
                mx.eval(*g_qkv, g_op, g_lh)
                diffs = sum(bit_diff_count(a, b) for a, b in zip(g_qkv, ref_qkv)) + bit_diff_count(g_op, ref_op) + bit_diff_count(g_lh, ref_lh)
                tq = pipe_ms(lambda: gemv_rows_multi(k4["h_in"], qkv_ws))
                to = pipe_ms(lambda: gemv_rows_res(k4["ov"], a4.o_proj.weight, k4["x"]))
                tl = pipe_ms(lambda: gemv_rows(hd["hidden"], model.lm_head.weight))
                gemv_sweep["non_swiglu"][f"bm{bm}_tm{tm}"] = {
                    "qkv_x63": round(63 * tq, 2), "op_x63": round(63 * to, 2), "lm_head": tl,
                    "tot": round(63 * (tq + to) + tl, 2), "diffs": diffs,
                }
            except Exception as exc:
                gemv_sweep["non_swiglu"][f"bm{bm}_tm{tm}"] = {"error": str(exc)}

        for bm, tm in ((4, 1), (8, 1), (4, 2), (8, 2)):
            def custom_sw_cfg(N, K, B_=1, is_swiglu=False, _bm=bm, _tm=tm):
                return (_bm if N >= 4096 else 4, 1, 1, 32, _tm, 4)
            gemv_mod._cfg = custom_sw_cfg
            try:
                g_sh = gemv_swiglu_rows(m4["mlp_in"], se4.gate_proj.weight, se4.up_proj.weight)
                g_d0 = gemv_swiglu_rows(d0["mlp_in"], ml0.gate_proj.weight, ml0.up_proj.weight)
                mx.eval(g_sh, g_d0)
                diffs = bit_diff_count(g_sh, ref_sh) + bit_diff_count(g_d0, ref_d0)
                tsh = pipe_ms(lambda: gemv_swiglu_rows(m4["mlp_in"], se4.gate_proj.weight, se4.up_proj.weight))
                td0 = pipe_ms(lambda: gemv_swiglu_rows(d0["mlp_in"], ml0.gate_proj.weight, ml0.up_proj.weight))
                gemv_sweep["swiglu"][f"bm{bm}_tm{tm}"] = {
                    "sh_x68": round(68 * tsh, 2), "d0_x4": round(4 * td0, 2),
                    "tot": round(68 * tsh + 4 * td0, 2), "diffs": diffs,
                }
            except Exception as exc:
                gemv_sweep["swiglu"][f"bm{bm}_tm{tm}"] = {"error": str(exc)}
    finally:
        gemv_mod._cfg = orig_cfg
    print("  GEMV non_swiglu sweep:", gemv_sweep["non_swiglu"], flush=True)
    print("  GEMV swiglu sweep:", gemv_sweep["swiglu"], flush=True)

    # 5. MTP varlen refresh + L=1 draft pipelined time
    from mlx_lm.models.cache import KVCache
    def clone_mtp_cs(cs):
        out = []
        for kc in cs:
            c2 = KVCache()
            c2.keys = mx.array(kc.keys)
            c2.values = mx.array(kc.values)
            c2.offset = kc.offset
            out.append(c2)
        mx.eval(*[k for c2 in out for k in (c2.keys, c2.values)])
        return out
    mtp_cs = samp["mtp_caches"]
    ncs_var = [min(L, x) for x in (3, 3, 2, 1)]
    toks_v = mx.array([samp["last_toks"][c] for c, nc in enumerate(ncs_var) for _ in range(nc)], dtype=mx.uint32)
    hs_v = mx.concatenate([samp["hidden_cl"][c, :nc, :] for c, nc in enumerate(ncs_var)], axis=0)
    toks_1 = mx.array(samp["last_toks"], dtype=mx.uint32)
    hs_1 = samp["hidden_cl"][:, 0, :]
    mx.eval(toks_v, hs_v, toks_1, hs_1)
    # Use single_ms for mtp since KVCache offset trims cleanly per call
    def run_mtp_varlen_once():
        offs = [mc.offset for mc in mtp_cs]
        out = mtp.step_varlen_rows(toks_v, hs_v, mtp_cs, ncs_var, use_q4_head=True)
        for mc, o in zip(mtp_cs, offs):
            mc.trim(mc.offset - o)
        return out
    def run_mtp_both_once():
        offs = [mc.offset for mc in mtp_cs]
        h_d, dl0 = mtp.step_varlen_rows(toks_v, hs_v, mtp_cs, ncs_var, use_q4_head=True)
        d0_a = mx.argmax(dl0, axis=-1)
        _, dl1 = mtp.step_rows(d0_a, h_d, mtp_cs, C=C, L=1, use_q4_head=True)
        d1_a = mx.argmax(dl1, axis=-1)
        for mc, o in zip(mtp_cs, offs):
            mc.trim(mc.offset - o)
        return d0_a, d1_a
    t_mtp_var = single_ms(run_mtp_varlen_once)
    t_mtp_both = single_ms(run_mtp_both_once)
    print(f"  MTP varlen_1pass={t_mtp_var:.2f}ms  varlen+draft1={t_mtp_both:.2f}ms", flush=True)

    return {
        "C": C, "K": K, "L": L, "B": B,
        "e2e_verify_ms": t_e2e,
        "ledger": ledger,
        "ledger_sum": ledger_sum,
        "moe_ab": moe_ab_summary,
        "gu_pipe": gu_pipe,
        "gemv_sweep": gemv_sweep,
        "mtp_varlen_ms": t_mtp_var,
        "mtp_varlen_plus_draft1_ms": t_mtp_both,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser(DEFAULT_MODEL))
    ap.add_argument("--base", default=os.path.expanduser(DEFAULT_BASE))
    ap.add_argument("--out", default=os.path.expanduser("~/science/runs/profile_b12_pipelined.json"))
    a = ap.parse_args()

    maybe_set_recommended_wired_limit()
    base = json.loads(Path(a.base).read_text())
    prompts_data = {nm: base["prompts"][nm] for nm, _ in PROMPTS}

    model, _ = load(a.model)
    mtp = BailingMTPLayer.from_target(model, a.model, with_q4_head=True)
    mx.eval(model.parameters())

    out = {
        "B12": run_probe_for_B(model, mtp, prompts_data, C=4, K=2),
        "B16": run_probe_for_B(model, mtp, prompts_data, C=4, K=3),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\nSaved {a.out}", flush=True)


if __name__ == "__main__":
    main()
