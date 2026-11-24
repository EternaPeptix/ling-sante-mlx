"""Step 2 profiler: real-routing B=12 (C=4,L=3) & B=16 (C=4,L=4) breakdown + MoE/GEMV sweep."""
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
from mlx_lm.models.bailing_moe_v3.rows import gemv as gemv_mod  # noqa: E402
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
    mla_verify_layer_rows,
)
from mlx_lm.models.bailing_moe_v3.rows.moe import (  # noqa: E402
    moe_down_combine_rows,
    moe_gate_up_swiglu_rows,
)
from mlx_lm.models.bailing_moe_v3.rows.norm import add_rms_norm_rows, rms_norm_rows  # noqa: E402
from mlx_lm.models.bailing_moe_v3.rows.router import gate_rows  # noqa: E402
from mlx_lm.models.bailing_moe_v3.spec_engine import _chain_mtp_drafts, prefill_with_hidden  # noqa: E402
from mlx_lm.models.bailing_moe_v3.verify_decode import (  # noqa: E402
    commit_verify_cache,
    fast_verify_forward,
    make_verify_cache,
)
from mlx_lm.models.cache import KVCache  # noqa: E402
from mlx_lm.utils import maybe_set_recommended_wired_limit  # noqa: E402

from bench._layer_loader import bit_diff_count  # noqa: E402
from bench.exact import DEFAULT_BASE, DEFAULT_MODEL, PROMPTS  # noqa: E402


def bench_ms(fn, warm: int = 3, iters: int = 7) -> float:
    for _ in range(warm):
        res = fn()
        mx.eval(res)
    mx.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        res = fn()
        mx.eval(res)
        ts.append((time.perf_counter() - t0) * 1e3)
    return round(statistics.median(ts), 4)


def capture_real_batch(model, mtp, prompts_data, C: int, K: int):
    """Prefill C prompts, draft K tokens per stream, and capture per-layer activations + real MoE routing."""
    L = K + 1
    B = C * L
    t_caches, mtp_caches, last_toks = [], [], []
    for nm, _ in PROMPTS[:C]:
        tc, mc, lt = prefill_with_hidden(model, mtp, prompts_data[nm]["ids"])
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

    draft_mat, _ = _chain_mtp_drafts(mtp, nxt0, hidden0[:, 0, :], mtp_caches, C=C, K=K, use_q4_head=True)
    inp = mx.concatenate([nxt0[:, None], draft_mat], axis=1)
    mx.eval(inp)

    # Walk model manually for 1 verify pass to capture real inds & layer inputs cleanly
    inner = model.model
    x = inner.word_embeddings(inp.reshape(-1))
    samples = {}
    uniq_counts = []
    for li, (layer, c) in enumerate(zip(inner.layers, vcache)):
        h_in = mx.fast.rms_norm(x, layer.input_layernorm.weight, layer.input_layernorm.eps)
        if layer.is_linear:
            attn = layer.attention
            q_r, k_r, v_r, f_r, g_r, b_r = gemv_rows_multi(
                h_in,
                [
                    attn.q_proj.weight,
                    attn.k_proj.weight,
                    attn.v_proj.weight,
                    attn.f_proj.weight,
                    attn.g_proj.weight,
                    attn.b_proj.weight,
                ],
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
                C=C,
                L=L,
                lower_bound=attn.lower_bound,
                eps=1e-6,
            )
            if len(c.cache) > 4:
                ov, s_end = kda_gdn_verify_seq_split_rows(
                    q, k, v, g, beta, c.cache[3:], g_r, attn.o_norm.weight, C=C, L=L, eps=attn.o_norm.eps
                )
                cs_full = mx.concatenate(c.cache[3:], axis=0)
            else:
                ov, s_end = kda_gdn_verify_seq_rows(
                    q, k, v, g, beta, c[3], g_r, attn.o_norm.weight, C=C, L=L, eps=attn.o_norm.eps
                )
                cs_full = mx.array(c[3])
            h_post = gemv_rows_res(ov, attn.o_proj.weight, x)
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
                    "cs": [mx.array(c[0]), mx.array(c[1]), mx.array(c[2]), cs_full],
                    "ov": mx.array(ov),
                }
        else:
            if li == 7:
                mla7_kv_snap = [ci.keys_and_values() for ci in (c if isinstance(c, list) else [c])]
                samples["mla7"] = {"x": mx.array(x), "h_in": mx.array(h_in), "kv_snap": mla7_kv_snap}
            h_post = mla_verify_layer_rows(h_in, x, layer.attention, c, C=C, L=L)

        mlp_in = mx.fast.rms_norm(
            h_post, layer.post_attention_layernorm.weight, layer.post_attention_layernorm.eps
        )
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
            uniq = len(set(inds.reshape(-1).tolist()))
            uniq_counts.append(uniq)
            se = mlp.shared_experts
            sh = gemv_rows(
                gemv_swiglu_rows(mlp_in, se.gate_proj.weight, se.up_proj.weight),
                se.down_proj.weight,
            )
            sw = mlp.switch_mlp
            h_mid = moe_gate_up_swiglu_rows(
                mlp_in,
                sw.gate_proj.weight,
                sw.gate_proj.scales,
                sw.up_proj.weight,
                sw.up_proj.scales,
                inds,
            )
            x = moe_down_combine_rows(
                h_mid,
                sw.down_proj.weight,
                sw.down_proj.scales,
                inds,
                wts,
                shared=sh,
                residual=h_post,
            )
            if li in (4, 10, 35, 65):
                samples.setdefault("moe", {})[f"moe{li}"] = {
                    "layer_idx": li,
                    "h_post": mx.array(h_post),
                    "mlp_in": mx.array(mlp_in),
                    "inds": mx.array(inds),
                    "wts": mx.array(wts),
                    "sh": mx.array(sh),
                    "h_mid": mx.array(h_mid),
                    "uniq": uniq,
                }
        else:
            mid = gemv_swiglu_rows(mlp_in, mlp.gate_proj.weight, mlp.up_proj.weight)
            x = gemv_rows_res(mid, mlp.down_proj.weight, h_post)
            samples["dense0"] = {"h_post": mx.array(h_post), "mlp_in": mx.array(mlp_in), "mid": mx.array(mid)}
        mx.eval(x)

    hidden = mx.fast.rms_norm(x, inner.norm.weight, inner.norm.eps)
    logits = gemv_rows(hidden, model.lm_head.weight).reshape(C, L, -1)
    mx.eval(logits)
    samples["head"] = {"x": mx.array(x), "hidden": mx.array(hidden), "logits": mx.array(logits)}
    samples["vcache"] = vcache
    samples["mtp_caches"] = mtp_caches
    samples["draft_lists"] = draft_mat.tolist()
    samples["last_toks"] = last_toks
    samples["hidden_cl"] = hidden.reshape(C, L, -1)
    samples["uniq_stats"] = {
        "n_moe_layers": len(uniq_counts),
        "total_pairs": 8 * B,
        "mean": round(sum(uniq_counts) / len(uniq_counts), 2),
        "min": min(uniq_counts),
        "max": max(uniq_counts),
    }
    return samples


def profile_regime(model, mtp, prompts_data, C: int, K: int):
    L = K + 1
    B = C * L
    print(f"\n================ PROFILE C={C}, K={K}, L={L}, B={B} ================", flush=True)
    samp = capture_real_batch(model, mtp, prompts_data, C=C, K=K)
    u = samp["uniq_stats"]
    print(f"Real MoE unique experts/{u['total_pairs']} pairs (over {u['n_moe_layers']} layers): mean={u['mean']} min={u['min']} max={u['max']}", flush=True)

    # Full forward wall-clock
    inp2 = mx.array([[lt] + samp["draft_lists"][c] for c, lt in enumerate(samp["last_toks"])], dtype=mx.uint32)
    def run_e2e():
        lg, hd, rb = fast_verify_forward(model, inp2, samp["vcache"], return_hidden=True, return_rollback=True)
        for mc in rb["mla"]:
            if isinstance(mc, list):
                for ci in mc:
                    ci.trim(L)
            else:
                mc.trim(L)
        return lg
    t_e2e = bench_ms(run_e2e, warm=2, iters=5)

    res = {"C": C, "K": K, "L": L, "B": B, "e2e_verify_ms": t_e2e, "uniq_stats": u, "components_ms": {}, "moe_sweep": {}, "gemv_bm_sweep": {}}

    # 1. RMSNorm comparison (145 calls: 72*2 + 1)
    l4 = model.model.layers[4]
    hp = samp["moe"]["moe4"]["h_post"]
    t_fast_rms = bench_ms(lambda: mx.fast.rms_norm(hp, l4.post_attention_layernorm.weight, l4.post_attention_layernorm.eps))
    t_row_rms = bench_ms(lambda: rms_norm_rows(hp, l4.post_attention_layernorm.weight, l4.post_attention_layernorm.eps))
    ov4 = samp["kda4"]["ov"]
    x4 = samp["kda4"]["x"]
    # Compare o_proj_res + rms_norm vs o_proj(no res) + add_rms_norm_rows
    t_sep_res_norm = bench_ms(
        lambda: mx.fast.rms_norm(
            gemv_rows_res(ov4, l4.attention.o_proj.weight, x4),
            l4.post_attention_layernorm.weight,
            l4.post_attention_layernorm.eps,
        )
    )
    t_fused_add_norm = bench_ms(
        lambda: add_rms_norm_rows(
            gemv_rows(ov4, l4.attention.o_proj.weight),
            x4,
            l4.post_attention_layernorm.weight,
            l4.post_attention_layernorm.eps,
        )
    )
    res["components_ms"]["rms_norm_1call"] = {
        "mx_fast_rms_norm": t_fast_rms,
        "rms_norm_rows": t_row_rms,
        "x145_mx_fast": round(145 * t_fast_rms, 2),
        "x145_row_rms": round(145 * t_row_rms, 2),
        "oproj_res_then_fast_rms": t_sep_res_norm,
        "oproj_then_add_rms_rows": t_fused_add_norm,
    }
    print(
        f"  RMSNorm x1: mx.fast={t_fast_rms*1000:.1f}us (x145={145*t_fast_rms:.2f}ms)  "
        f"rms_rows={t_row_rms*1000:.1f}us (x145={145*t_row_rms:.2f}ms)  "
        f"oproj+norm sep={t_sep_res_norm*1000:.1f}us vs add_rms={t_fused_add_norm*1000:.1f}us",
        flush=True,
    )

    # 2. KDA L4 breakdown (x63)
    k4 = samp["kda4"]
    a4 = l4.attention
    qkv_ws = [a4.q_proj.weight, a4.k_proj.weight, a4.v_proj.weight, a4.f_proj.weight, a4.g_proj.weight, a4.b_proj.weight]
    t_kda_qkv = bench_ms(lambda: gemv_rows_multi(k4["h_in"], qkv_ws))
    t_kda_pre = bench_ms(
        lambda: kda_pre_seq_rows(
            k4["q_r"], k4["k_r"], k4["v_r"], k4["f_r"], k4["b_r"],
            k4["cs"][0], k4["cs"][1], k4["cs"][2],
            a4.q_conv1d.weight, a4.k_conv1d.weight, a4.v_conv1d.weight,
            a4.A_log, a4.dt_bias, C=C, L=L, lower_bound=a4.lower_bound, eps=1e-6,
        )
    )
    t_kda_gdn = bench_ms(
        lambda: kda_gdn_verify_seq_rows(
            k4["q"], k4["k"], k4["v"], k4["g"], k4["beta"], k4["cs"][3], k4["g_r"],
            a4.o_norm.weight, C=C, L=L, eps=a4.o_norm.eps,
        )
    )
    t_kda_oproj = bench_ms(lambda: gemv_rows_res(k4["ov"], a4.o_proj.weight, k4["x"]))
    kda_sum1 = t_kda_qkv + t_kda_pre + t_kda_gdn + t_kda_oproj
    res["components_ms"]["kda"] = {
        "qkvfgb_1": t_kda_qkv,
        "pre_seq_1": t_kda_pre,
        "gdn_seq_1": t_kda_gdn,
        "o_proj_1": t_kda_oproj,
        "total_x63": round(63 * kda_sum1, 2),
    }
    print(
        f"  KDA x63 = {63*kda_sum1:.2f}ms  (1 layer={kda_sum1:.3f}ms: qkvfgb={t_kda_qkv:.3f}, pre={t_kda_pre:.3f}, gdn={t_kda_gdn:.3f}, o_proj={t_kda_oproj:.3f})",
        flush=True,
    )

    # 3. MLA L7 breakdown (x9)
    l7 = model.model.layers[7]
    a7 = l7.attention
    m7 = samp["mla7"]
    t_mla_qa = bench_ms(lambda: gemv_rows_multi(m7["h_in"], [a7.q_a_proj.weight, a7.kv_a_proj_with_mqa.weight, a7.g_proj.weight]))
    q_a, kv_a_full, g_raw = gemv_rows_multi(m7["h_in"], [a7.q_a_proj.weight, a7.kv_a_proj_with_mqa.weight, a7.g_proj.weight])
    q_a_norm = mx.fast.rms_norm(q_a, a7.q_a_layernorm.weight, a7.q_a_layernorm.eps)
    t_mla_qb = bench_ms(lambda: gemv_rows(q_a_norm, a7.q_b_proj.weight))
    q_full = gemv_rows(q_a_norm, a7.q_b_proj.weight)
    offs_m = mx.array([k_s[0].shape[2] + t for k_s in m7["kv_snap"] for t in range(L)], dtype=mx.int32)
    mx.eval(q_full, kv_a_full, g_raw, offs_m)
    t_mla_prep = bench_ms(
        lambda: mla_prep_kv_rope_rows(
            q_full, kv_a_full, a7.kv_a_layernorm.weight, offs_m,
            kv_eps=a7.kv_a_layernorm.eps, base=float(a7.rope.base), scale=float(a7.rope.scale),
        )
    )
    q_nope, q_rope, kv_latent, k_rope = mla_prep_kv_rope_rows(
        q_full, kv_a_full, a7.kv_a_layernorm.weight, offs_m,
        kv_eps=a7.kv_a_layernorm.eps, base=float(a7.rope.base), scale=float(a7.rope.scale),
    )
    t_mla_eq = bench_ms(lambda: mla_multilinear_rows(q_nope, a7.embed_q.weight))
    q_nope_e = mla_multilinear_rows(q_nope, a7.embed_q.weight)
    kl_in = kv_latent.reshape(C, 1, L, 512)
    kr_in = k_rope.reshape(C, 1, L, 64)
    mx.eval(q_nope_e, q_rope, kl_in, kr_in)

    # Full mla_verify_layer_rows using fresh non-mutating clones each call
    def run_mla_sdpa_and_rest():
        outs = []
        for c_i in range(C):
            kc = KVCache()
            kc.keys, kc.values = m7["kv_snap"][c_i]
            kc.offset = kc.keys.shape[2]
            off_c = kc.offset
            kl_c, kr_c = kc.update_and_fetch(kl_in[c_i : c_i + 1], kr_in[c_i : c_i + 1])
            for t in range(L):
                b = c_i * L + t
                end = off_c + t + 1
                kl_t = kl_c[:, :, :end, :]
                kr_t = kr_c[:, :, :end, :]
                pe = (q_rope[b : b + 1] * a7.scale) @ kr_t.swapaxes(-1, -2)
                sc = (q_nope_e[b : b + 1] * a7.scale) @ kl_t.swapaxes(-1, -2) + pe
                pr = mx.softmax(sc, axis=-1, precise=True)
                outs.append(pr @ kl_t)
        return mx.concatenate(outs, axis=0)

    out_sdpa = run_mla_sdpa_and_rest()
    mx.eval(out_sdpa)
    t_mla_sdpa = bench_ms(run_mla_sdpa_and_rest)
    t_mla_ue = bench_ms(lambda: mla_multilinear_rows(out_sdpa, a7.unembed_out.weight, g_raw=g_raw))
    gated = mla_multilinear_rows(out_sdpa, a7.unembed_out.weight, g_raw=g_raw)
    mx.eval(gated)
    t_mla_dn = bench_ms(lambda: gemv_rows_res(gated, a7.dense.weight, m7["x"]))
    mla_sum1 = t_mla_qa + t_mla_qb + t_mla_prep + t_mla_eq + t_mla_sdpa + t_mla_ue + t_mla_dn
    res["components_ms"]["mla"] = {
        "qkv_a_1": t_mla_qa,
        "q_b_1": t_mla_qb,
        "prep_1": t_mla_prep,
        "embed_q_1": t_mla_eq,
        "sdpa_1": t_mla_sdpa,
        "unembed_1": t_mla_ue,
        "dense_1": t_mla_dn,
        "total_x9": round(9 * mla_sum1, 2),
    }
    print(
        f"  MLA x9  = {9*mla_sum1:.2f}ms  (1 layer={mla_sum1:.3f}ms: qa={t_mla_qa:.3f}, qb={t_mla_qb:.3f}, prep={t_mla_prep:.3f}, eq={t_mla_eq:.3f}, sdpa={t_mla_sdpa:.3f}, ue={t_mla_ue:.3f}, dn={t_mla_dn:.3f})",
        flush=True,
    )

    # 4. MoE breakdown + parameter sweep across sampled MoE layers
    moe_keys = list(samp["moe"].keys())
    moe_layer_objs = {}
    for lk in moe_keys:
        li = samp["moe"][lk]["layer_idx"]
        moe_layer_objs[lk] = model.model.layers[li]

    k0 = moe_keys[0]
    m1 = samp["moe"][k0]
    ml1 = moe_layer_objs[k0]
    t_gate = bench_ms(
        lambda: gate_rows(
            m1["mlp_in"], ml1.mlp.gate.weight, ml1.mlp.gate.expert_bias,
            top_k=ml1.mlp.gate.top_k, n_group=ml1.mlp.gate.n_group,
            topk_group=ml1.mlp.gate.topk_group, routed_scaling_factor=ml1.mlp.gate.routed_scaling_factor,
        )
    )
    se1 = ml1.mlp.shared_experts
    t_sh = bench_ms(lambda: gemv_rows(gemv_swiglu_rows(m1["mlp_in"], se1.gate_proj.weight, se1.up_proj.weight), se1.down_proj.weight))

    gu_combos = [(1, 2), (2, 2), (4, 2), (1, 4), (2, 4), (4, 4)]
    dc_combos = [(1, 2), (1, 4), (1, 8), (2, 2), (2, 4), (2, 8), (4, 2), (4, 4), (4, 8)]
    gu_sweep = {f"tm{tm}_n{nsg}": [] for tm, nsg in gu_combos}
    dc_sweep = {f"tm{tm}_n{nsg}": [] for tm, nsg in dc_combos}
    gu_diffs = {f"tm{tm}_n{nsg}": 0 for tm, nsg in gu_combos}
    dc_diffs = {f"tm{tm}_n{nsg}": 0 for tm, nsg in dc_combos}

    for lk in moe_keys:
        s = samp["moe"][lk]
        sw = moe_layer_objs[lk].mlp.switch_mlp
        ref_gu = moe_gate_up_swiglu_rows(
            s["mlp_in"], sw.gate_proj.weight, sw.gate_proj.scales, sw.up_proj.weight, sw.up_proj.scales, s["inds"], tmax=1, nsg=2
        )
        ref_dc = moe_down_combine_rows(
            s["h_mid"], sw.down_proj.weight, sw.down_proj.scales, s["inds"], s["wts"], shared=s["sh"], residual=s["h_post"], tmax=1, nsg=4
        )
        mx.eval(ref_gu, ref_dc)
        for tm, nsg in gu_combos:
            key = f"tm{tm}_n{nsg}"
            got = moe_gate_up_swiglu_rows(
                s["mlp_in"], sw.gate_proj.weight, sw.gate_proj.scales, sw.up_proj.weight, sw.up_proj.scales, s["inds"], tmax=tm, nsg=nsg
            )
            mx.eval(got)
            gu_diffs[key] += bit_diff_count(got, ref_gu)
            gu_sweep[key].append(
                bench_ms(
                    lambda s=s, sw=sw, tm=tm, nsg=nsg: moe_gate_up_swiglu_rows(
                        s["mlp_in"], sw.gate_proj.weight, sw.gate_proj.scales, sw.up_proj.weight, sw.up_proj.scales, s["inds"], tmax=tm, nsg=nsg
                    )
                )
            )
        for tm, nsg in dc_combos:
            key = f"tm{tm}_n{nsg}"
            got = moe_down_combine_rows(
                s["h_mid"], sw.down_proj.weight, sw.down_proj.scales, s["inds"], s["wts"], shared=s["sh"], residual=s["h_post"], tmax=tm, nsg=nsg
            )
            mx.eval(got)
            dc_diffs[key] += bit_diff_count(got, ref_dc)
            dc_sweep[key].append(
                bench_ms(
                    lambda s=s, sw=sw, tm=tm, nsg=nsg: moe_down_combine_rows(
                        s["h_mid"], sw.down_proj.weight, sw.down_proj.scales, s["inds"], s["wts"], shared=s["sh"], residual=s["h_post"], tmax=tm, nsg=nsg
                    )
                )
            )

    gu_avg = {k: {"mean_ms": round(sum(v) / len(v), 4), "diffs": gu_diffs[k]} for k, v in gu_sweep.items()}
    dc_avg = {k: {"mean_ms": round(sum(v) / len(v), 4), "diffs": dc_diffs[k]} for k, v in dc_sweep.items()}
    res["moe_sweep"] = {"gate_up": gu_avg, "down_combine": dc_avg}

    cur_gu = gu_avg["tm2_n2"]["mean_ms"]
    cur_dc = dc_avg["tm1_n4"]["mean_ms"]
    moe_cur1 = t_gate + t_sh + cur_gu + cur_dc
    res["components_ms"]["moe_current"] = {
        "gate_1": t_gate,
        "shared_1": t_sh,
        "gu_cur_1": cur_gu,
        "dc_cur_1": cur_dc,
        "total_x71": round(71 * moe_cur1, 2),
    }
    print(f"  MoE current x71 = {71*moe_cur1:.2f}ms (gate={t_gate:.3f}, sh={t_sh:.3f}, gu={cur_gu:.3f}, dc={cur_dc:.3f})", flush=True)
    print("  MoE gu sweep (x71 ms):", "  ".join(f"{k}:{v['mean_ms']*71:.1f}({v['diffs']})" for k, v in sorted(gu_avg.items(), key=lambda kv: kv[1]["mean_ms"])), flush=True)
    print("  MoE dc sweep (x71 ms):", "  ".join(f"{k}:{v['mean_ms']*71:.1f}({v['diffs']})" for k, v in sorted(dc_avg.items(), key=lambda kv: kv[1]["mean_ms"])), flush=True)

    # 5. Dense L0 + Head + Argmax vs Logsumexp
    d0 = samp["dense0"]
    ml0 = model.model.layers[0].mlp
    t_d0 = bench_ms(lambda: gemv_rows_res(gemv_swiglu_rows(d0["mlp_in"], ml0.gate_proj.weight, ml0.up_proj.weight), ml0.down_proj.weight, d0["h_post"]))
    hd = samp["head"]
    t_lh = bench_ms(lambda: gemv_rows(hd["hidden"], model.lm_head.weight))
    lg_cl = hd["logits"]
    t_lse = bench_ms(lambda: mx.argmax(lg_cl - mx.logsumexp(lg_cl, axis=-1, keepdims=True), axis=-1))
    t_amx = bench_ms(lambda: mx.argmax(lg_cl, axis=-1))
    res["components_ms"]["dense_and_head"] = {
        "dense0_1": t_d0,
        "lm_head": t_lh,
        "logsumexp_argmax": t_lse,
        "argmax_only": t_amx,
    }
    print(f"  Dense0={t_d0:.3f}ms  lm_head={t_lh:.3f}ms  lse+argmax={t_lse:.3f}ms  argmax_only={t_amx:.3f}ms", flush=True)

    # 6. GEMV BM=4 vs BM=8 check on KDA qkvfgb, o_proj, lm_head (exactness + speed)
    ref_qkv = gemv_rows_multi(k4["h_in"], qkv_ws)
    ref_op = gemv_rows_res(k4["ov"], a4.o_proj.weight, k4["x"])
    ref_lh = gemv_rows(hd["hidden"], model.lm_head.weight)
    mx.eval(*ref_qkv, ref_op, ref_lh)
    for bm in (8, 4):
        gemv_mod._GEMV_BM_LARGE = bm
        g_qkv = gemv_rows_multi(k4["h_in"], qkv_ws)
        g_op = gemv_rows_res(k4["ov"], a4.o_proj.weight, k4["x"])
        g_lh = gemv_rows(hd["hidden"], model.lm_head.weight)
        mx.eval(*g_qkv, g_op, g_lh)
        d = sum(bit_diff_count(a, b) for a, b in zip(g_qkv, ref_qkv)) + bit_diff_count(g_op, ref_op) + bit_diff_count(g_lh, ref_lh)
        t_q = bench_ms(lambda: gemv_rows_multi(k4["h_in"], qkv_ws))
        t_o = bench_ms(lambda: gemv_rows_res(k4["ov"], a4.o_proj.weight, k4["x"]))
        t_l = bench_ms(lambda: gemv_rows(hd["hidden"], model.lm_head.weight))
        res["gemv_bm_sweep"][f"bm{bm}"] = {
            "qkv_x63": round(63 * t_q, 2),
            "op_x63": round(63 * t_o, 2),
            "lm_head": t_l,
            "total": round(63 * (t_q + t_o) + t_l, 2),
            "diffs": d,
        }
    gemv_mod._GEMV_BM_LARGE = 8
    print("  GEMV BM sweep:", res["gemv_bm_sweep"], flush=True)

    # 7. Commit + MTP serial by_nc vs 1-pass comparison
    ncs = [min(L, x) for x in [3, 3, 2, 1][:C]] if C > 1 else [3]
    _, _, rb = fast_verify_forward(model, inp2, samp["vcache"], return_hidden=True, return_rollback=True)
    mx.eval(rb["kda"][0][2])
    mx.synchronize()
    t0 = time.perf_counter()
    commit_verify_cache(samp["vcache"], rb, ncs)
    mx.eval([item[0][0] for item in rb["kda"]], [item[0][3] for item in rb["kda"]])
    t_commit = (time.perf_counter() - t0) * 1e3

    mtp_cs = samp["mtp_caches"]
    mtp_arg = mtp_cs[0] if C == 1 else mtp_cs
    toks_l1 = mx.array(samp["last_toks"], dtype=mx.uint32)
    hs_l1 = samp["hidden_cl"][:, 0, :]
    t_mtp_draft1 = bench_ms(
        lambda: (
            (mc_offs := [mc.offset for mc in mtp_cs]),
            out := mtp.step_rows(toks_l1, hs_l1, mtp_arg, C=C, L=1, use_q4_head=True),
            [mc.trim(mc.offset - o) for mc, o in zip(mtp_cs, mc_offs)],
            out[1],
        )[-1]
    )

    if C == 4:
        def run_by_nc_refresh():
            mc_offs = [mc.offset for mc in mtp_cs]
            out_parts = []
            for g_cs, g_l in (([0, 1], min(L, 3)), ([2], min(L, 2)), ([3], 1)):
                tg = mx.array([[samp["last_toks"][c]] * g_l for c in g_cs], dtype=mx.uint32)
                hg = samp["hidden_cl"][: len(g_cs), :g_l, :]
                sub = mtp_cs[g_cs[0]] if len(g_cs) == 1 else [mtp_cs[c] for c in g_cs]
                _, dl = mtp.step_rows(tg, hg, sub, C=len(g_cs), L=g_l, use_q4_head=True)
                out_parts.append(dl)
            for mc, o in zip(mtp_cs, mc_offs):
                mc.trim(mc.offset - o)
            return mx.concatenate(out_parts, axis=0)

        def run_single_pass_approx():
            mc_offs = [mc.offset for mc in mtp_cs]
            tg = mx.array([[samp["last_toks"][c]] * L for c in range(C)], dtype=mx.uint32)
            hg = samp["hidden_cl"][:, :L, :]
            _, dl = mtp.step_rows(tg, hg, mtp_arg, C=C, L=L, use_q4_head=True)
            for mc, o in zip(mtp_cs, mc_offs):
                mc.trim(mc.offset - o)
            return dl

        t_mtp_bync = bench_ms(run_by_nc_refresh)
        t_mtp_1pass = bench_ms(run_single_pass_approx)
    else:
        t_mtp_bync = 0.0
        t_mtp_1pass = 0.0

    res["components_ms"]["commit_and_mtp"] = {
        "commit_ncs": ncs,
        "commit_once_ms": round(t_commit, 3),
        "mtp_bync_refresh": t_mtp_bync,
        "mtp_1pass_L_refresh": t_mtp_1pass,
        "mtp_draft_L1": t_mtp_draft1,
    }
    print(
        f"  e2e_verify={t_e2e:.2f}ms  Commit(ncs={ncs})={t_commit:.2f}ms  MTP by_nc={t_mtp_bync:.2f}ms vs 1pass={t_mtp_1pass:.2f}ms  MTP L=1={t_mtp_draft1:.2f}ms",
        flush=True,
    )
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser(DEFAULT_MODEL))
    ap.add_argument("--base", default=os.path.expanduser(DEFAULT_BASE))
    ap.add_argument("--out", default=os.path.expanduser("~/science/runs/profile_verify_b12.json"))
    a = ap.parse_args()

    maybe_set_recommended_wired_limit()
    base = json.loads(Path(a.base).read_text())
    prompts_data = {nm: base["prompts"][nm] for nm, _ in PROMPTS}

    t0 = time.perf_counter()
    model, _ = load(a.model)
    mtp = BailingMTPLayer.from_target(model, a.model, with_q4_head=True)
    mx.eval(model.parameters())
    print(f"Loaded in {time.perf_counter()-t0:.1f}s ({mx.get_active_memory()/1e9:.1f} GB)", flush=True)

    out = {
        "B12": profile_regime(model, mtp, prompts_data, C=4, K=2),
        "B16": profile_regime(model, mtp, prompts_data, C=4, K=3),
        "B5": profile_regime(model, mtp, prompts_data, C=1, K=4),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\nSaved {a.out}", flush=True)


if __name__ == "__main__":
    main()
