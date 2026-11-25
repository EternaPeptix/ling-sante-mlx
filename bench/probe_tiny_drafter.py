"""Probe inclusionAI/Ling-3.0-tiny Q4 + SuffixMatcher on the 20 internal validation sequences (zero eval leakage)."""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ["LINGSANTE_FAST_DECODE"] = "0"
os.environ["LINGSANTE_FASTDECODE"] = "0"
os.environ["LINGSANTE_FP8"] = "0"

_root = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(_root / "vendor"), str(_root)]

import mlx_lm  # noqa: E402,F401
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
from mlx_lm.models.bailing_moe_v3.suffix import SuffixMatcher  # noqa: E402
from mlx_lm.utils import load_model, maybe_set_recommended_wired_limit  # noqa: E402

# Verified Phase 2B C=1 target verify latencies (ms) with moe_gu adaptive row-split
_VERIFY_ANCHORS = [(1, 39.0), (3, 48.0), (5, 56.6), (8, 71.1), (10, 88.4), (12, 100.7), (16, 120.8)]


def verify_cost_ms(L: float) -> float:
    if L <= _VERIFY_ANCHORS[0][0]:
        return _VERIFY_ANCHORS[0][1]
    if L >= _VERIFY_ANCHORS[-1][0]:
        return _VERIFY_ANCHORS[-1][1]
    for (x0, y0), (x1, y1) in zip(_VERIFY_ANCHORS[:-1], _VERIFY_ANCHORS[1:]):
        if x0 <= L <= x1:
            t = (L - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return _VERIFY_ANCHORS[-1][1]


def load_val_seqs(data_dir: Path) -> list[dict]:
    idx_data = json.loads((data_dir / "index.json").read_text())
    shard_cache: dict[str, dict] = {}
    val_seqs = []
    for entry in idx_data["entries"]:
        if entry["split"] != "val":
            continue
        sh = entry["shard"]
        if sh not in shard_cache:
            shard_cache[sh] = mx.load(str(data_dir / "shards" / sh))
        st = shard_cache[sh]
        k = entry["key"]
        ids_mx = st[f"{k}.ids"]
        t32_ids = st[f"{k}.top32_ids"]
        mx.eval(ids_mx, t32_ids)
        val_seqs.append({
            "idx": entry["idx"],
            "domain": entry["domain"],
            "prompt_len": int(entry["prompt_len"]),
            "total_len": int(entry["total_len"]),
            "ids": ids_mx.tolist(),
            "ids_mx": ids_mx,
            "top32_ids": t32_ids,
        })
    return val_seqs


def sample_anchors(plen: int, T: int, k_max: int, n_blk: int = 5) -> list[int]:
    lo = plen
    hi = T - 1 - k_max
    if hi < lo:
        return []
    span = hi - lo + 1
    out = []
    for m in range(n_blk):
        b0 = lo + int(m * span / n_blk)
        b1 = lo + int((m + 1) * span / n_blk) - 1
        b1 = max(b0, min(hi, b1))
        out.append((b0 + b1) // 2)
    return out


def measure_decode_speed(model, B: int, prefill_len: int = 128, n_steps: int = 64) -> dict:
    cache = model.make_cache()
    pref = mx.full((B, prefill_len), 100, dtype=mx.uint32)
    lg = model(pref, cache=cache)
    tok = mx.argmax(lg[:, -1:, :], axis=-1).astype(mx.uint32)
    mx.eval(tok, *[c.state for c in cache])
    for _ in range(8):
        lg = model(tok, cache=cache)
        tok = mx.argmax(lg, axis=-1).astype(mx.uint32)
        mx.eval(tok)
    mx.synchronize()
    dts = []
    for _ in range(n_steps):
        t0 = time.perf_counter()
        lg = model(tok, cache=cache)
        tok = mx.argmax(lg, axis=-1).astype(mx.uint32)
        mx.eval(tok)
        dts.append((time.perf_counter() - t0) * 1e3)
    med_ms = statistics.median(dts)
    mean_ms = statistics.mean(dts)
    return {
        "B": B,
        "med_ms": round(med_ms, 3),
        "mean_ms": round(mean_ms, 3),
        "tok_s": round(1e3 / med_ms, 2),
    }


def eval_teacher_forced(model, val_seqs: list[dict]) -> dict:
    n_tot, n_med = 0, 0
    hits1_tot, hits1_med = 0, 0
    hits4_tot, hits4_med = 0, 0
    t32_1_tot, t32_1_med = 0, 0
    ks = (4, 7, 11, 15)
    e_sum = {k: 0 for k in ks}
    e_cnt = {k: 0 for k in ks}
    e_med_sum = {k: 0 for k in ks}
    e_med_cnt = {k: 0 for k in ks}

    for seq in val_seqs:
        plen = seq["prompt_len"]
        T = seq["total_len"]
        is_med = seq["domain"] == "med"
        cache = model.make_cache()
        logits = model(seq["ids_mx"][None], cache=cache)[0]  # (T, V)
        pred_sl = logits[plen - 1 : T - 1]  # (N_gen, V)
        top1 = mx.argmax(pred_sl, axis=-1)
        top4 = mx.argpartition(-pred_sl, kth=3, axis=-1)[:, :4]
        tgts = seq["ids_mx"][plen:T]
        t32_top1 = seq["top32_ids"][plen - 1 : T - 1, 0]
        m1 = (top1 == tgts)
        m4 = mx.any(top4 == tgts[:, None], axis=-1)
        mt = (top1 == t32_top1)
        mx.eval(m1, m4, mt)
        m1_l = m1.tolist()
        N = len(m1_l)
        c1 = sum(1 for v in m1_l if v)
        c4 = int(mx.sum(m4).item())
        ct = int(mx.sum(mt).item())
        n_tot += N
        hits1_tot += c1
        hits4_tot += c4
        t32_1_tot += ct
        if is_med:
            n_med += N
            hits1_med += c1
            hits4_med += c4
            t32_1_med += ct
        for k in ks:
            for i in range(0, max(0, N - k + 1)):
                acc = 1
                for j in range(k):
                    if m1_l[i + j]:
                        acc += 1
                    else:
                        break
                e_sum[k] += acc
                e_cnt[k] += 1
                if is_med:
                    e_med_sum[k] += acc
                    e_med_cnt[k] += 1

    return {
        "tokens": n_tot,
        "med_tokens": n_med,
        "p1_all": round(hits1_tot / max(1, n_tot), 4),
        "p1_med": round(hits1_med / max(1, n_med), 4),
        "top4_all": round(hits4_tot / max(1, n_tot), 4),
        "top4_med": round(hits4_med / max(1, n_med), 4),
        "p1_vs_prefill_t32": round(t32_1_tot / max(1, n_tot), 4),
        "p1_med_vs_prefill_t32": round(t32_1_med / max(1, n_med), 4),
        "E_acc_tf": {str(k): round(e_sum[k] / max(1, e_cnt[k]), 3) for k in ks},
        "E_acc_tf_med": {str(k): round(e_med_sum[k] / max(1, e_med_cnt[k]), 3) for k in ks},
    }


def eval_true_ar_and_suffix(model, val_seqs: list[dict], tiny_b1_ms: float, *, k_max: int = 15, n_blk: int = 5) -> dict:
    records = []
    for seq in val_seqs:
        anchors = sample_anchors(seq["prompt_len"], seq["total_len"], k_max, n_blk=n_blk)
        ids = seq["ids"]
        ids_mx = seq["ids_mx"]
        is_med = seq["domain"] == "med"
        for a in anchors:
            # Prefill cleanly up to and including anchor `a`
            cache = model.make_cache()
            lg = model(ids_mx[: a + 1][None], cache=cache)[:, -1, :]
            drafts, probs = [], []
            for j in range(k_max):
                logits_f = lg.astype(mx.float32)
                tok_mx = mx.argmax(logits_f, axis=-1)
                max_l = mx.max(logits_f, axis=-1, keepdims=True)
                p_mx = mx.exp(mx.max(logits_f - max_l, axis=-1)) / mx.sum(mx.exp(logits_f - max_l), axis=-1)
                mx.eval(tok_mx, p_mx)
                d = int(tok_mx.item())
                p = float(p_mx.item())
                drafts.append(d)
                probs.append(p)
                if j + 1 < k_max:
                    lg = model(tok_mx.reshape(1, 1).astype(mx.uint32), cache=cache)[:, 0, :]

            gold = ids[a + 1 : a + 1 + k_max]
            match_len = 0
            for d, g in zip(drafts, gold):
                if d == g:
                    match_len += 1
                else:
                    break

            sm3 = SuffixMatcher(ids[: a + 1], min_gram=3, max_gram=8)
            g3, suf_d3 = sm3.match(7)
            suf3_match = 0
            if suf_d3:
                for sd, gd in zip(suf_d3, gold[:7]):
                    if sd == gd:
                        suf3_match += 1
                    else:
                        break
            sm2 = SuffixMatcher(ids[: a + 1], min_gram=2, max_gram=8)
            g2, suf_d2 = sm2.match(7)
            suf2_match = 0
            if suf_d2:
                for sd, gd in zip(suf_d2, gold[:7]):
                    if sd == gd:
                        suf2_match += 1
                    else:
                        break

            records.append({
                "is_med": is_med,
                "probs": probs,
                "match_len": match_len,
                "pos1_ok": int(drafts[0] == gold[0]),
                "suf3_hit": int(bool(suf_d3)),
                "suf3_acc": 1 + suf3_match if suf_d3 else 0,
                "suf2_hit": int(bool(suf_d2)),
                "suf2_acc": 1 + suf2_match if suf_d2 else 0,
            })

    n = len(records)
    med_recs = [r for r in records if r["is_med"]]
    ks = (3, 5, 7, 11, 15)
    fixed = {}
    for k in ks:
        accs = [min(r["match_len"], k) + 1 for r in records]
        maccs = [min(r["match_len"], k) + 1 for r in med_recs]
        ea = sum(accs) / n
        em = sum(maccs) / len(med_recs)
        cyc = verify_cost_ms(k + 1) + k * tiny_b1_ms
        fixed[str(k)] = {
            "L": k + 1,
            "E_acc": round(ea, 3),
            "E_acc_med": round(em, 3),
            "draft_ms": round(k * tiny_b1_ms, 2),
            "verify_ms": round(verify_cost_ms(k + 1), 2),
            "proj_cyc_ms": round(cyc, 2),
            "proj_tok_s": round(1e3 * ea / cyc, 2),
        }

    thetas = (0.10, 0.20, 0.30, 0.45, 0.60)
    dyn_kmin1 = {}
    dyn_kmin0 = {}
    for th in thetas:
        ls1, accs1, cycs1 = [], [], []
        ls0, accs0, cycs0 = [], [], []
        for r in records:
            cum = 1.0
            k_best = 0
            for j, p in enumerate(r["probs"], start=1):
                cum *= p
                if cum >= th:
                    k_best = j
                else:
                    break
            k1 = max(1, k_best)
            a1 = min(r["match_len"], k1) + 1
            c1 = verify_cost_ms(k1 + 1) + k1 * tiny_b1_ms
            ls1.append(k1 + 1)
            accs1.append(a1)
            cycs1.append(c1)

            k0 = k_best
            a0 = min(r["match_len"], k0) + 1 if k0 > 0 else 1
            c0 = verify_cost_ms(k0 + 1) + (k0 if k0 > 0 else 1) * tiny_b1_ms
            ls0.append(k0 + 1)
            accs0.append(a0)
            cycs0.append(c0)

        dyn_kmin1[str(th)] = {
            "mean_L": round(sum(ls1) / n, 2),
            "E_acc": round(sum(accs1) / n, 3),
            "mean_cyc_ms": round(sum(cycs1) / n, 2),
            "proj_tok_s": round(1e3 * sum(accs1) / sum(cycs1), 2),
        }
        dyn_kmin0[str(th)] = {
            "mean_L": round(sum(ls0) / n, 2),
            "E_acc": round(sum(accs0) / n, 3),
            "mean_cyc_ms": round(sum(cycs0) / n, 2),
            "proj_tok_s": round(1e3 * sum(accs0) / sum(cycs0), 2),
        }

    # Suffix matcher stats + combined with best dyn_kmin1
    suf3_hits = [r for r in records if r["suf3_hit"]]
    suf2_hits = [r for r in records if r["suf2_hit"]]
    best_th = max(dyn_kmin1, key=lambda t: dyn_kmin1[t]["proj_tok_s"])
    comb_accs, comb_cycs = [], []
    th_best = float(best_th)
    for r in records:
        if r["suf3_hit"]:
            comb_accs.append(r["suf3_acc"])
            comb_cycs.append(verify_cost_ms(8))
        else:
            cum = 1.0
            kb = 1
            for j, p in enumerate(r["probs"], start=1):
                cum *= p
                if cum >= th_best:
                    kb = j
                else:
                    break
            comb_accs.append(min(r["match_len"], kb) + 1)
            comb_cycs.append(verify_cost_ms(kb + 1) + kb * tiny_b1_ms)

    return {
        "n_blocks": n,
        "n_med_blocks": len(med_recs),
        "p1_blocks": round(sum(r["pos1_ok"] for r in records) / n, 4),
        "p1_med_blocks": round(sum(r["pos1_ok"] for r in med_recs) / len(med_recs), 4),
        "fixed_K": fixed,
        "dyn_kmin1": dyn_kmin1,
        "dyn_kmin0": dyn_kmin0,
        "suffix": {
            "min3_K7_hit_rate": round(len(suf3_hits) / n, 3),
            "min3_K7_E_acc_on_hit": round(sum(r["suf3_acc"] for r in suf3_hits) / max(1, len(suf3_hits)), 3),
            "min2_K7_hit_rate": round(len(suf2_hits) / n, 3),
            "min2_K7_E_acc_on_hit": round(sum(r["suf2_acc"] for r in suf2_hits) / max(1, len(suf2_hits)), 3),
            "combined_with_best_th": {
                "theta": th_best,
                "E_acc": round(sum(comb_accs) / n, 3),
                "mean_cyc_ms": round(sum(comb_cycs) / n, 2),
                "proj_tok_s": round(1e3 * sum(comb_accs) / sum(comb_cycs), 2),
            },
        },
    }


def test_compile_and_breakdown(model):
    from mlx_lm.layers.attention import scaled_dot_product_attention
    from mlx_lm.layers.recurrent import gated_delta_update, normalize_qk

    model.lm_head = nn.QuantizedLinear.from_linear(model.lm_head, group_size=64, bits=4, mode="affine")
    model.eval()
    mx.eval(model.parameters())
    print(f"Tiny q4 + q4_head ready ({mx.get_active_memory()/1e9:.2f} GB active)", flush=True)

    # 1. Per-component decode breakdown (B=1 after 128-tok prefill)
    cache = model.make_cache()
    pref = mx.full((1, 128), 100, dtype=mx.uint32)
    lg0 = model(pref, cache=cache)
    mx.eval(lg0, *[c.state for c in cache])
    x1 = (mx.random.normal((1, 1, 1536)) * 0.2).astype(mx.bfloat16)
    mx.eval(x1)

    l_dense = model.model.layers[0]
    l_kda = model.model.layers[1]
    l_mla = model.model.layers[3]
    assert not l_mla.is_linear and l_kda.is_linear

    def bench_one(fn, iters=64):
        for _ in range(8):
            o = fn()
            mx.eval(o)
        mx.synchronize()
        dts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            o = fn()
            mx.eval(o)
            dts.append((time.perf_counter() - t0) * 1e3)
        return statistics.median(dts)

    def run_kda_once():
        c = l_kda_tmp_cache
        return l_kda.attention(x1, mask=None, cache=c)

    def run_mla_once():
        c = l_mla_tmp_cache
        o = l_mla.attention(x1, mask=None, cache=c)
        c.trim(1)
        return o

    # Freeze cache copies so repeated KDA calls stay 1-step without growing
    l_kda_tmp_cache = cache[1]
    l_mla_tmp_cache = cache[3]
    ms_kda = bench_one(run_kda_once)
    ms_mla = bench_one(run_mla_once)
    ms_den = bench_one(lambda: l_dense.mlp(x1))
    ms_moe = bench_one(lambda: l_kda.mlp(x1))
    ms_head = bench_one(lambda: model.lm_head(x1))
    sum_comp = 18 * ms_kda + 6 * ms_mla + 1 * ms_den + 23 * ms_moe + ms_head

    eager_sp = measure_decode_speed(model, B=1)
    print("\n=== Ling-3.0-tiny-q4 (B=1, q4_head) Component Breakdown ===", flush=True)
    print(f"  18 x KDA attn : {ms_kda:.3f} ms/call -> {18*ms_kda:.2f} ms", flush=True)
    print(f"   6 x MLA attn : {ms_mla:.3f} ms/call -> { 6*ms_mla:.2f} ms", flush=True)
    print(f"   1 x dense MLP: {ms_den:.3f} ms/call -> { 1*ms_den:.2f} ms", flush=True)
    print(f"  23 x MoE MLP  : {ms_moe:.3f} ms/call -> {23*ms_moe:.2f} ms", flush=True)
    print(f"   1 x q4_head  : {ms_head:.3f} ms/call -> {   ms_head:.2f} ms", flush=True)
    print(f"  Component sum : {sum_comp:.2f} ms | Eager full step: {eager_sp['med_ms']:.3f} ms/tok ({eager_sp['tok_s']:.1f} tok/s)", flush=True)

    # 2. Functional stateless 24-layer step + @mx.compile test
    print("\n=== @mx.compile (Full 24-layer Static-KV Step) Test ===", flush=True)
    MAX_KV = 256
    arange_kv = mx.arange(MAX_KV, dtype=mx.int32)

    def build_init_state():
        c0 = model.make_cache()
        prompt = mx.arange(100, 228, dtype=mx.uint32)[None]
        lg = model(prompt, cache=c0)
        tok0 = mx.argmax(lg[:, -1:, :], axis=-1).astype(mx.uint32)
        mx.eval(tok0, *[c.state for c in c0])
        st = []
        for layer, lc in zip(model.model.layers, c0):
            if layer.is_linear:
                st.extend([mx.array(lc[0]), mx.array(lc[1]), mx.array(lc[2]), mx.array(lc[3])])
            else:
                kbuf = mx.zeros((1, 1, MAX_KV, 512), dtype=lc.keys.dtype)
                rbuf = mx.zeros((1, 1, MAX_KV, 64), dtype=lc.values.dtype)
                kbuf[..., :128, :] = lc.keys[..., :128, :]
                rbuf[..., :128, :] = lc.values[..., :128, :]
                st.extend([kbuf, rbuf])
        mx.eval(*st)
        return c0, tok0, st

    def make_step_fn(use_kernel: bool):
        layers = model.model.layers
        norm_fn = model.model.norm
        emb_fn = model.model.word_embeddings
        head_fn = model.lm_head

        def step_fn(tok_in, pos_arr, *states):
            h = emb_fn(tok_in)
            new_states = []
            si = 0
            idx_eq = (arange_kv == pos_arr)[None, None, :, None]
            valid_mask = (arange_kv <= pos_arr)[None, None, None, :]
            for layer in layers:
                x_norm = layer.input_layernorm(h)
                if layer.is_linear:
                    a = layer.attention
                    c0, c1, c2, s_gdn = states[si], states[si + 1], states[si + 2], states[si + 3]
                    si += 4
                    qi = mx.concatenate([c0, a.q_proj(x_norm)], axis=1)
                    ki = mx.concatenate([c1, a.k_proj(x_norm)], axis=1)
                    vi = mx.concatenate([c2, a.v_proj(x_norm)], axis=1)
                    nc0 = qi[:, -3:, :]
                    nc1 = ki[:, -3:, :]
                    nc2 = vi[:, -3:, :]
                    q = nn.silu(a.q_conv1d(qi)).reshape(1, 1, a.num_heads, a.head_dim)
                    k = nn.silu(a.k_conv1d(ki)).reshape(1, 1, a.num_heads, a.head_dim)
                    v = nn.silu(a.v_conv1d(vi)).reshape(1, 1, a.num_heads, a.head_dim)
                    q, k = normalize_qk(q, k, inv_scale=a.head_dim**-0.5, eps=1e-6)
                    raw_gate = a.f_proj(x_norm).reshape(1, 1, a.num_heads, a.head_dim)
                    out, ns = gated_delta_update(
                        q, k, v, raw_gate, a.b_proj(x_norm).astype(mx.float32),
                        a.A_log.reshape(a.num_heads, 1), a.dt_bias.reshape(a.num_heads, a.head_dim),
                        state=s_gdn, mask=None, use_kernel=use_kernel, lower_bound=a.lower_bound,
                    )
                    out_gate = a.g_proj(x_norm).reshape(1, 1, a.num_heads, a.head_dim)
                    out = a.o_norm(out) * mx.sigmoid(out_gate.astype(mx.float32)).astype(out.dtype)
                    h_post = h + a.o_proj(out.reshape(1, 1, -1))
                    new_states.extend([nc0, nc1, nc2, ns])
                else:
                    a = layer.attention
                    kb, rb = states[si], states[si + 1]
                    si += 2
                    q = a.q_b_proj(a.q_a_layernorm(a.q_a_proj(x_norm)))
                    q = q.reshape(1, 1, a.num_heads, a.qk_head_dim).transpose(0, 2, 1, 3)
                    q_nope, q_rope = mx.split(q, [a.qk_nope_head_dim], axis=-1)
                    comp = a.kv_a_proj_with_mqa(x_norm)
                    comp, k_rope = mx.split(comp, [a.kv_lora_rank], axis=-1)
                    kv_lat = a.kv_a_layernorm(comp)
                    k_rope = k_rope.reshape(1, 1, 1, a.qk_rope_head_dim).transpose(0, 2, 1, 3)
                    q_rope = mx.fast.rope(q_rope, a.qk_rope_head_dim, traditional=True, base=6000000.0, scale=1.0, offset=pos_arr)
                    k_rope = mx.fast.rope(k_rope, a.qk_rope_head_dim, traditional=True, base=6000000.0, scale=1.0, offset=pos_arr)
                    nkb = mx.where(idx_eq, mx.expand_dims(kv_lat, axis=1), kb)
                    nrb = mx.where(idx_eq, k_rope, rb)
                    pe = (q_rope * a.scale) @ nrb.swapaxes(-1, -2)
                    pe = mx.where(valid_mask, pe, mx.array(mx.finfo(pe.dtype).min, pe.dtype))
                    q_nope = a.embed_q(q_nope)
                    out = scaled_dot_product_attention(q_nope, nkb, nkb, cache=None, scale=a.scale, mask=pe)
                    out = a.unembed_out(out).transpose(0, 2, 1, 3)
                    gate = mx.sigmoid(a.g_proj(x_norm).astype(mx.float32)).astype(out.dtype)
                    h_post = h + a.dense((out * gate[..., None]).reshape(1, 1, -1))
                    new_states.extend([nkb, nrb])
                h = h_post + layer.mlp(layer.post_attention_layernorm(h_post))
            tok_out = mx.argmax(head_fn(norm_fn(h)), axis=-1).astype(mx.uint32)
            return (tok_out, *new_states)

        return step_fn

    for uk in (True, False):
        lbl = f"use_kernel={uk}"
        try:
            c_ref, tok_ref, st_cur = build_init_state()
            compiled_step = mx.compile(make_step_fn(uk))
            tok_c = mx.array(tok_ref)
            ref_toks, got_toks = [], []
            for step_i in range(5):
                lg_r = model(tok_ref, cache=c_ref)
                tok_ref = mx.argmax(lg_r, axis=-1).astype(mx.uint32)
                out_t = compiled_step(tok_c, mx.array(128 + step_i, dtype=mx.int32), *st_cur)
                tok_c, st_cur = out_t[0], list(out_t[1:])
                mx.eval(tok_ref, tok_c, *st_cur)
                ref_toks.append(int(tok_ref.item()))
                got_toks.append(int(tok_c.item()))
            diffs = sum(1 for a, b in zip(ref_toks, got_toks) if a != b)
            print(f"  @mx.compile ({lbl}) 5-step verify: diffs={diffs}/5 (ref={ref_toks} got={got_toks})", flush=True)

            # Time 64 steps after 8 warmups (wrapping position mod 128 in [128..255])
            for w_i in range(8):
                out_t = compiled_step(tok_c, mx.array(133 + w_i, dtype=mx.int32), *st_cur)
                tok_c, st_cur = out_t[0], list(out_t[1:])
                mx.eval(tok_c, *st_cur)
            mx.synchronize()
            dts = []
            for s_i in range(64):
                p_arr = mx.array(141 + (s_i % 64), dtype=mx.int32)
                t0 = time.perf_counter()
                out_t = compiled_step(tok_c, p_arr, *st_cur)
                tok_c, st_cur = out_t[0], list(out_t[1:])
                mx.eval(tok_c, *st_cur)
                dts.append((time.perf_counter() - t0) * 1e3)
            med_ms = statistics.median(dts)
            print(f"  @mx.compile ({lbl}) B=1 speed: {med_ms:.3f} ms/tok ({1e3/med_ms:.1f} tok/s)", flush=True)
        except Exception as e:
            print(f"  @mx.compile ({lbl}) FAILED: {type(e).__name__}: {e}", flush=True)


def run_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/science/models/ling-3.0-tiny-q4"))
    ap.add_argument("--data-dir", default=os.path.expanduser("~/science/data/lingsante_dflash_train"))
    ap.add_argument("--out", default=os.path.expanduser("~/science/runs/probe_tiny_drafter.json"))
    ap.add_argument("--test-compile", action="store_true")
    a = ap.parse_args()

    maybe_set_recommended_wired_limit()
    t0 = time.perf_counter()
    model, _ = load_model(Path(a.model))
    mx.eval(model.parameters())
    print(f"Loaded {a.model} in {time.perf_counter()-t0:.1f}s ({mx.get_active_memory()/1e9:.2f} GB active)", flush=True)

    if a.test_compile:
        test_compile_and_breakdown(model)
        return

    main_full(model, a)


def main_full(model, a):
    val_seqs = load_val_seqs(Path(a.data_dir))
    print(f"Loaded {len(val_seqs)} internal val_seqs (med={sum(1 for s in val_seqs if s['domain']=='med')})", flush=True)

    print("\n=== Test 1a: Decode Speed (bf16 lm_head) ===", flush=True)
    sp_bf16_b1 = measure_decode_speed(model, B=1)
    sp_bf16_b2 = measure_decode_speed(model, B=2)
    print(f"  bf16 lm_head: B=1 {sp_bf16_b1['med_ms']:.3f} ms/tok ({sp_bf16_b1['tok_s']:.1f} tok/s) | B=2 {sp_bf16_b2['med_ms']:.3f} ms/step", flush=True)

    print("\n=== Test 2a: Zero-Shot Teacher-Forced Alignment (bf16 lm_head) ===", flush=True)
    tf_bf16 = eval_teacher_forced(model, val_seqs)
    print(f"  bf16 tf: p1={tf_bf16['p1_all']:.4f} (med={tf_bf16['p1_med']:.4f}) top4={tf_bf16['top4_all']:.4f} (med={tf_bf16['top4_med']:.4f}) E_acc_tf={tf_bf16['E_acc_tf']}", flush=True)

    print("\n=== Quantizing tiny model.lm_head -> q4 (gs=64) ===", flush=True)
    model.lm_head = nn.QuantizedLinear.from_linear(model.lm_head, group_size=64, bits=4, mode="affine")
    model.eval()
    mx.eval(model.parameters())
    print(f"Active memory after q4 lm_head: {mx.get_active_memory()/1e9:.2f} GB", flush=True)

    print("\n=== Test 1b: Decode Speed (q4 lm_head) ===", flush=True)
    sp_q4_b1 = measure_decode_speed(model, B=1)
    sp_q4_b2 = measure_decode_speed(model, B=2)
    print(f"  q4   lm_head: B=1 {sp_q4_b1['med_ms']:.3f} ms/tok ({sp_q4_b1['tok_s']:.1f} tok/s) | B=2 {sp_q4_b2['med_ms']:.3f} ms/step", flush=True)

    print("\n=== Test 2b: Zero-Shot Teacher-Forced Alignment (q4 lm_head) ===", flush=True)
    tf_q4 = eval_teacher_forced(model, val_seqs)
    print(f"  q4   tf: p1={tf_q4['p1_all']:.4f} (med={tf_q4['p1_med']:.4f}) top4={tf_q4['top4_all']:.4f} (med={tf_q4['top4_med']:.4f}) E_acc_tf={tf_q4['E_acc_tf']}", flush=True)

    print("\n=== Test 3: True Greedy AR Rollouts + Dynamic Confidence Stopping + SuffixMatcher (100 blocks) ===", flush=True)
    ar_res = eval_true_ar_and_suffix(model, val_seqs, sp_q4_b1["med_ms"])
    for k_s, v in ar_res["fixed_K"].items():
        print(f"  Fixed K={k_s:>2s} (L={v['L']:2d}): E_acc={v['E_acc']:.3f} (med={v['E_acc_med']:.3f}) draft={v['draft_ms']:.2f}ms ver={v['verify_ms']:.2f}ms -> {v['proj_tok_s']:5.2f} tok/s", flush=True)
    for th_s, v in ar_res["dyn_kmin1"].items():
        print(f"  Dyn(k>=1) th={th_s}: mean_L={v['mean_L']:4.2f} E_acc={v['E_acc']:.3f} cyc={v['mean_cyc_ms']:.2f}ms -> {v['proj_tok_s']:5.2f} tok/s", flush=True)
    print("  SuffixMatcher:", json.dumps(ar_res["suffix"]), flush=True)

    out = {
        "model": a.model,
        "active_gb_q4_head": round(mx.get_active_memory() / 1e9, 2),
        "speed": {"bf16_b1": sp_bf16_b1, "bf16_b2": sp_bf16_b2, "q4_b1": sp_q4_b1, "q4_b2": sp_q4_b2},
        "teacher_forced": {"bf16": tf_bf16, "q4": tf_q4},
        "ar_rollout": ar_res,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"\nSaved {a.out}", flush=True)


if __name__ == "__main__":
    run_cli()
