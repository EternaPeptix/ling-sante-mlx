"""Phase 2 speculative decoding benchmark & strict bit-exactness gate (MTP + Suffix, C=1..4)."""
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
from mlx_lm.models.bailing_moe_v3.suffix import SuffixMatcher  # noqa: E402
from mlx_lm.models.bailing_moe_v3.verify_decode import (  # noqa: E402
    commit_verify_cache,
    fast_verify_forward,
    make_verify_cache,
)
from mlx_lm.models.cache import KVCache, make_prompt_cache  # noqa: E402
from mlx_lm.utils import maybe_set_recommended_wired_limit  # noqa: E402

from bench.exact import DEFAULT_BASE, DEFAULT_MODEL, PROMPTS, lp_digest  # noqa: E402
from bench._layer_loader import bit_diff_count  # noqa: E402


def prefill_with_hidden(model, mtp: BailingMTPLayer, ids: list[int], step: int = 512):
    """Prefill `p[:-1]` through target (matching generate_step), populate `mtp_cache` cleanly, return `(t_cache, mtp_cache, last_tok)`."""
    t_cache = make_prompt_cache(model)
    p = mx.array(ids, dtype=mx.uint32)
    h_chunks = []
    pos = 0
    while p.size - pos > 1:
        n = min(step, p.size - 1 - pos)
        h = model.model(p[pos : pos + n][None], cache=t_cache)
        mx.eval(h, [c.state for c in t_cache])
        h_chunks.append(h)
        pos += n
    mtp_cache = KVCache()
    if h_chunks:
        h_all = h_chunks[0] if len(h_chunks) == 1 else mx.concatenate(h_chunks, axis=1)
        tokens_shifted = p[1:][None]
        s_pos = 0
        tot = tokens_shifted.shape[1]
        while s_pos < tot:
            m = min(step, tot - s_pos)
            mtp_out = mtp.prefill(tokens_shifted[:, s_pos : s_pos + m], h_all[:, s_pos : s_pos + m, :], mtp_cache)
            mx.eval(mtp_out, mtp_cache.state)
            s_pos += m
    return t_cache, mtp_cache, int(p[-1].item())


def test_mtp_exactness(model, mtp: BailingMTPLayer, p_entry: dict):
    """Verify `BailingMTPLayer.step_rows(L=1)` and `step_rows(L=4)` match sequential stock 1-token forward bit-for-bit."""
    ids = p_entry["ids"]
    _, mtp_c0, _ = prefill_with_hidden(model, mtp, ids[:-8])
    mtp_c1 = KVCache()
    mtp_c2 = KVCache()
    if mtp_c0.keys is not None:
        for mc in (mtp_c1, mtp_c2):
            mc.keys = mx.array(mtp_c0.keys)
            mc.values = mx.array(mtp_c0.values)
            mc.offset = mtp_c0.offset

    toks = mx.array([ids[-4:]], dtype=mx.uint32)
    h_in = (mx.random.normal((1, 4, 4096)) * 0.2).astype(mx.bfloat16)
    mx.eval(toks, h_in)

    ref_hs, got_hs, ref_ls, got_ls = [], [], [], []
    for t in range(4):
        rh = mtp.prefill(toks[:, t : t + 1], h_in[:, t : t + 1, :], mtp_c0)[:, 0, :]
        rl = mtp._lm_head(rh)
        gh, gl = mtp.step_rows(toks[:, t], h_in[:, t, :], mtp_c1, C=1, L=1, use_q4_head=False)
        ref_hs.append(rh)
        got_hs.append(gh)
        ref_ls.append(rl)
        got_ls.append(gl)
    ref_h = mx.stack(ref_hs, axis=1)
    got_h = mx.stack(got_hs, axis=1)
    ref_l = mx.stack(ref_ls, axis=1)
    got_l = mx.stack(got_ls, axis=1)
    gh4, gl4 = mtp.step_rows(toks[0], h_in[0], mtp_c2, C=1, L=4, use_q4_head=False)
    mx.eval(ref_h, got_h, ref_l, got_l, gh4, gl4)

    dh = bit_diff_count(got_h, ref_h)
    dl = bit_diff_count(got_l, ref_l)
    dh4 = bit_diff_count(gh4, ref_h[:, 3, :])
    dl4 = bit_diff_count(gl4, ref_l[:, 3, :])
    print(f"MTP unit exactness vs L=1 stock: h_diffs={dh} logits_diffs={dl} batch4=(h:{dh4},l:{dl4})", flush=True)
    return dh == 0 and dl == 0 and dh4 == 0 and dl4 == 0


def run_c1_spec(
    model,
    mtp: BailingMTPLayer,
    p_entry: dict,
    *,
    K: int,
    max_tokens: int = 128,
    use_q4_head: bool = True,
    suffix_mode: str | None = None,
    suffix_min_gram: int = 4,
    suffix_extra: int = 0,
    check_exact: bool = True,
    diag_topk: bool = False,
):
    ids = p_entry["ids"]
    gold_toks = p_entry["tokens"]
    gold_digs = p_entry["logprob_sha256_32"]

    t_cache, mtp_cache, last_tok = prefill_with_hidden(model, mtp, ids)
    suffix = SuffixMatcher(ids) if suffix_mode else None

    # Initial L=1 target forward on last prompt token
    logits0, hidden0, rb0 = fast_verify_forward(model, mx.array([[last_tok]], dtype=mx.uint32), t_cache, return_hidden=True, return_rollback=True)
    lp0 = logits0[0, 0] - mx.logsumexp(logits0[0, 0], axis=-1, keepdims=True)
    cur_tok = int(mx.argmax(lp0).item())
    cur_h = hidden0[0, 0:1, :]
    mx.eval(cur_h, lp0)
    commit_verify_cache(t_cache, rb0, 1)

    emitted = [cur_tok]
    tok_diffs = 0
    lp_diffs = 0
    if check_exact:
        if len(gold_toks) > 0 and cur_tok != gold_toks[0]:
            tok_diffs += 1
        if len(gold_digs) > 0 and lp_digest(lp0) != gold_digs[0]:
            lp_diffs += 1
    if suffix:
        suffix.extend([cur_tok])

    # Produce initial K drafts using (cur_tok, cur_h)
    mtp_exact_off = mtp_cache.offset
    d_toks_init = []
    if K > 0:
        d_arr, d_logits = mtp.step_rows(
            mx.array([cur_tok], dtype=mx.uint32), cur_h, mtp_cache, C=1, L=1, use_q4_head=use_q4_head
        )
        mtp_exact_off = mtp_cache.offset
        nxt = mx.argmax(d_logits, axis=-1)
        mx.eval(d_arr, nxt)
        d0 = int(nxt[0].item())
        d_toks_init = [d0]
        h_d = d_arr
        d_prev = d0
        for _ in range(1, K):
            h_d, dl = mtp.step_rows(mx.array([d_prev], dtype=mx.uint32), h_d, mtp_cache, C=1, L=1, use_q4_head=use_q4_head)
            dnxt = mx.argmax(dl, axis=-1)
            mx.eval(h_d, dnxt)
            d_prev = int(dnxt[0].item())
            d_toks_init.append(d_prev)
    drafts = d_toks_init
    last_draft_logits = None

    n_verify_steps = 0
    sum_emitted = 0
    hits = [0] * max(K + suffix_extra, 1)
    trials = [0] * max(K + suffix_extra, 1)
    top2_hits = [0] * K
    top4_hits = [0] * K
    top_trials = [0] * K
    t_draft = 0.0
    t_verify = 0.0
    t_commit = 0.0
    t_sync = 0.0

    mx.reset_peak_memory()
    t_loop0 = time.perf_counter()

    while len(emitted) < max_tokens:
        rem = max_tokens - len(emitted)
        use_drafts = drafts[: max(0, min(len(drafts), rem))]
        L = 1 + len(use_drafts)
        inp = mx.array([[cur_tok] + use_drafts], dtype=mx.uint32)

        tv0 = time.perf_counter()
        logits, hidden, rb = fast_verify_forward(model, inp, t_cache, return_hidden=True, return_rollback=True)
        # Only need logsumexp/digest on first 32 checked steps; argmax is bit-exact with raw logits or logprobs!
        # Keep lp exact for steps < 32 (and always compute argmax on lp or logits; use lp only when check_exact & <32 for speed? No, test with actual logprobs when check_exact!).
        if check_exact and len(emitted) < len(gold_digs):
            lp = logits[0] - mx.logsumexp(logits[0], axis=-1, keepdims=True)
            preds = mx.argmax(lp, axis=-1)
            mx.eval(preds, hidden, lp)
        else:
            preds = mx.argmax(logits[0], axis=-1)
            mx.eval(preds, hidden)
        tv1 = time.perf_counter()
        t_verify += tv1 - tv0

        pred_list = preds.tolist()
        a = 0
        for j, d_tok in enumerate(use_drafts):
            if j < len(trials):
                trials[j] += 1
            if pred_list[j] == d_tok:
                if j < len(hits):
                    hits[j] += 1
                a += 1
            else:
                break

        if diag_topk and last_draft_logits is not None:
            for j in range(min(len(last_draft_logits), len(use_drafts))):
                top_trials[j] += 1
                tgt = pred_list[j]
                dl_j = last_draft_logits[j]
                top4 = mx.argpartition(-dl_j, 3)[:4].tolist()
                if tgt in top4[:4]:
                    top4_hits[j] += 1
                    # Check exact top-2
                    top2 = mx.argpartition(-dl_j, 1)[:2].tolist()
                    if tgt in top2:
                        top2_hits[j] += 1
                if j >= a:
                    break

        n_new = min(a + 1, rem)
        for j in range(n_new):
            step_pos = len(emitted)
            tok = pred_list[j]
            emitted.append(tok)
            if check_exact:
                if step_pos < len(gold_toks) and tok != gold_toks[step_pos]:
                    tok_diffs += 1
                if step_pos < len(gold_digs):
                    lp_j = lp[j]
                    if lp_digest(lp_j) != gold_digs[step_pos]:
                        lp_diffs += 1

        if suffix:
            suffix.extend(pred_list[:n_new])

        tc0 = time.perf_counter()
        commit_verify_cache(t_cache, rb, a + 1)
        t_commit += time.perf_counter() - tc0

        n_verify_steps += 1
        sum_emitted += n_new
        cur_tok = pred_list[a]

        if len(emitted) >= max_tokens:
            break

        # Refresh MTP + generate next draft
        ts0 = time.perf_counter()
        trim_n = mtp_cache.offset - mtp_exact_off
        if trim_n > 0:
            mtp_cache.trim(trim_n)

        k_mtp_goal = K
        sync_toks = mx.array([use_drafts[:a] + [cur_tok]], dtype=mx.uint32)  # (1, a+1)
        sync_h = hidden[:, : a + 1, :]  # (1, a+1, 4096)

        suffix_override = None
        if suffix_mode == "override":
            g_len, s_cand = suffix.match(K + suffix_extra, min_gram=suffix_min_gram)
            if g_len >= suffix_min_gram and s_cand:
                suffix_override = s_cand

        h_d, dl0 = mtp.step_rows(
            sync_toks,
            sync_h,
            mtp_cache,
            C=1,
            L=a + 1,
            use_q4_head=use_q4_head,
            head_only_last=(suffix_override is None),
            return_logits=(suffix_override is None),
        )
        mtp_exact_off = mtp_cache.offset
        t_sync_end = time.perf_counter()
        t_sync += t_sync_end - ts0

        if suffix_override is not None:
            mx.eval(h_d)
            drafts = suffix_override
            last_draft_logits = None
            t_draft += time.perf_counter() - t_sync_end
        else:
            td0 = time.perf_counter()
            if K == 0:
                drafts = []
                last_draft_logits = None
            else:
                nxt0 = mx.argmax(dl0, axis=-1)
                mx.eval(h_d, nxt0)
                if diag_topk:
                    mx.eval(dl0)
                    last_draft_logits = [dl0[0]]
                d_prev = int(nxt0[0].item())
                next_drafts = [d_prev]
                for _ in range(1, k_mtp_goal):
                    h_d, dlk = mtp.step_rows(
                        mx.array([d_prev], dtype=mx.uint32), h_d, mtp_cache, C=1, L=1, use_q4_head=use_q4_head
                    )
                    dnxt = mx.argmax(dlk, axis=-1)
                    mx.eval(h_d, dnxt)
                    if diag_topk:
                        mx.eval(dlk)
                        last_draft_logits.append(dlk[0])
                    d_prev = int(dnxt[0].item())
                    next_drafts.append(d_prev)
                if suffix_mode == "extend" and suffix_extra > 0:
                    _, ext = suffix.match(suffix_extra, min_gram=suffix_min_gram, tail_extra=next_drafts)
                    if ext:
                        next_drafts.extend(ext)
                drafts = next_drafts
            t_draft += time.perf_counter() - td0

    wall_dt = time.perf_counter() - t_loop0
    gen_after_first = len(emitted) - 1
    tps = gen_after_first / wall_dt
    e_acc = sum_emitted / max(n_verify_steps, 1)
    alphas = [round(hits[j] / trials[j], 3) if trials[j] > 0 else 0.0 for j in range(len(trials)) if trials[j] > 0]
    res = {
        "prompt": p_entry.get("name", ""),
        "C": 1,
        "K": K,
        "q4_head": use_q4_head,
        "suffix": suffix_mode or "none",
        "tokens": len(emitted),
        "steps": n_verify_steps,
        "tok_s": round(tps, 2),
        "E_acc": round(e_acc, 3),
        "verify_ms": round(t_verify * 1e3 / max(n_verify_steps, 1), 2),
        "commit_ms": round(t_commit * 1e3 / max(n_verify_steps, 1), 2),
        "sync_ms": round(t_sync * 1e3 / max(n_verify_steps, 1), 2),
        "draft_ms": round(t_draft * 1e3 / max(n_verify_steps, 1), 2),
        "alphas": alphas,
        "tok_diffs": tok_diffs,
        "lp_diffs": lp_diffs,
        "peak_gb": round(mx.get_peak_memory() / 1e9, 2),
    }
    if diag_topk:
        res["top2_recall"] = [round(top2_hits[j] / top_trials[j], 3) if top_trials[j] > 0 else 0.0 for j in range(K)]
        res["top4_recall"] = [round(top4_hits[j] / top_trials[j], 3) if top_trials[j] > 0 else 0.0 for j in range(K)]
    return res


def run_batched_spec(
    model,
    mtp: BailingMTPLayer,
    prompts_data: dict,
    *,
    C: int,
    K: int,
    max_tokens: int = 128,
    use_q4_head: bool = True,
):
    names = [nm for nm, _ in PROMPTS[:C]]
    t_caches, mtp_caches, last_toks = [], [], []
    for nm in names:
        tc, mc, lt = prefill_with_hidden(model, mtp, prompts_data[nm]["ids"])
        t_caches.append(tc)
        mtp_caches.append(mc)
        last_toks.append(lt)

    vcache = make_verify_cache(model, t_caches)

    # Initial L=1 verify step on last prompt tokens
    logits0, hidden0, rb0 = fast_verify_forward(
        model, mx.array([[lt] for lt in last_toks], dtype=mx.uint32), vcache, return_hidden=True, return_rollback=True
    )
    lp0 = logits0[:, 0, :] - mx.logsumexp(logits0[:, 0, :], axis=-1, keepdims=True)
    nxt0 = mx.argmax(lp0, axis=-1)
    mx.eval(nxt0, hidden0, lp0)
    commit_verify_cache(vcache, rb0, 1)

    cur_toks = nxt0.tolist()
    cur_h = hidden0[:, 0, :]  # (C, 4096)
    emitted = [[cur_toks[c]] for c in range(C)]
    tok_diffs = 0
    lp_diffs = 0
    for c, nm in enumerate(names):
        if cur_toks[c] != prompts_data[nm]["tokens"][0]:
            tok_diffs += 1
        if lp_digest(lp0[c]) != prompts_data[nm]["logprob_sha256_32"][0]:
            lp_diffs += 1

    # Initial batched draft
    h_d, dl0 = mtp.step_rows(
        mx.array(cur_toks, dtype=mx.uint32), cur_h, mtp_caches, C=C, L=1, use_q4_head=use_q4_head
    )
    mtp_exact_offs = [mc.offset for mc in mtp_caches]
    dn0 = mx.argmax(dl0, axis=-1)
    mx.eval(h_d, dn0)
    draft_cols = [dn0.tolist()]
    d_prev = dn0
    for _ in range(1, K):
        h_d, dlk = mtp.step_rows(d_prev, h_d, mtp_caches, C=C, L=1, use_q4_head=use_q4_head)
        dnk = mx.argmax(dlk, axis=-1)
        mx.eval(h_d, dnk)
        draft_cols.append(dnk.tolist())
        d_prev = dnk
    drafts = [[draft_cols[k][c] for k in range(K)] for c in range(C)]

    n_steps = 0
    sum_emitted = 0
    t_verify = 0.0
    t_sync_draft = 0.0
    mx.reset_peak_memory()
    t0 = time.perf_counter()

    while min(len(e) for e in emitted) < max_tokens:
        inp = mx.array([[cur_toks[c]] + drafts[c] for c in range(C)], dtype=mx.uint32)  # (C, K+1)
        tv0 = time.perf_counter()
        logits, hidden, rb = fast_verify_forward(model, inp, vcache, return_hidden=True, return_rollback=True)
        need_lp = any(len(emitted[c]) < len(prompts_data[names[c]]["logprob_sha256_32"]) for c in range(C))
        if need_lp:
            lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            preds = mx.argmax(lp, axis=-1)
            mx.eval(preds, hidden, lp)
        else:
            preds = mx.argmax(logits, axis=-1)
            mx.eval(preds, hidden)
        t_verify += time.perf_counter() - tv0

        pred_lists = preds.tolist()
        ncs = []
        next_cur_toks = []
        for c, nm in enumerate(names):
            gtoks = prompts_data[nm]["tokens"]
            gdigs = prompts_data[nm]["logprob_sha256_32"]
            a = 0
            for j in range(K):
                if pred_lists[c][j] == drafts[c][j]:
                    a += 1
                else:
                    break
            rem = max_tokens - len(emitted[c])
            n_take = min(a + 1, rem)
            for j in range(n_take):
                pos = len(emitted[c])
                tok = pred_lists[c][j]
                emitted[c].append(tok)
                if pos < len(gtoks) and tok != gtoks[pos]:
                    tok_diffs += 1
                if pos < len(gdigs) and lp_digest(lp[c, j]) != gdigs[pos]:
                    lp_diffs += 1
            sum_emitted += n_take
            ncs.append(a + 1)
            next_cur_toks.append(pred_lists[c][a])

        commit_verify_cache(vcache, rb, ncs)
        cur_toks = next_cur_toks
        n_steps += 1

        if min(len(e) for e in emitted) >= max_tokens:
            break

        ts0 = time.perf_counter()
        for c in range(C):
            tr = mtp_caches[c].offset - mtp_exact_offs[c]
            if tr > 0:
                mtp_caches[c].trim(tr)

        # Group streams with identical nc so equal-length refreshes run batched.
        h0_rows = [None] * C
        d0_rows = [0] * C
        by_nc: dict[int, list[int]] = {}
        for c, nc in enumerate(ncs):
            by_nc.setdefault(nc, []).append(c)

        for nc, cs in by_nc.items():
            toks_g = mx.array([drafts[c][: nc - 1] + [cur_toks[c]] for c in cs], dtype=mx.uint32)
            hs_g = mx.concatenate([hidden[c : c + 1, :nc, :] for c in cs], axis=0)
            sub_caches = [mtp_caches[c] for c in cs]
            hg, dlg = mtp.step_rows(toks_g, hs_g, sub_caches, C=len(cs), L=nc, use_q4_head=use_q4_head)
            dn = mx.argmax(dlg, axis=-1)
            mx.eval(hg, dn)
            dn_list = dn.tolist()
            for idx_g, c in enumerate(cs):
                mtp_exact_offs[c] = mtp_caches[c].offset
                h0_rows[c] = hg[idx_g : idx_g + 1]
                d0_rows[c] = dn_list[idx_g]

        h_d = mx.concatenate(h0_rows, axis=0)
        d_prev = mx.array(d0_rows, dtype=mx.uint32)
        cols = [d0_rows]
        for _ in range(1, K):
            h_d, dlk = mtp.step_rows(d_prev, h_d, mtp_caches, C=C, L=1, use_q4_head=use_q4_head)
            dnk = mx.argmax(dlk, axis=-1)
            mx.eval(h_d, dnk)
            cols.append(dnk.tolist())
            d_prev = dnk
        drafts = [[cols[k][c] for k in range(K)] for c in range(C)]
        t_sync_draft += time.perf_counter() - ts0

    dt = time.perf_counter() - t0
    agg_toks = sum(len(e) - 1 for e in emitted)
    agg_tps = agg_toks / dt
    e_acc = sum_emitted / (n_steps * C)
    print(
        f"  C={C} K={K} q4={int(use_q4_head)}: agg={agg_tps:6.2f} tok/s ({agg_tps/C:5.2f}/stream)  "
        f"E_acc={e_acc:.3f}  verify={t_verify*1e3/n_steps:.1f}ms  draft={t_sync_draft*1e3/n_steps:.1f}ms  "
        f"exact={tok_diffs==0 and lp_diffs==0} (td={tok_diffs},ld={lp_diffs}) peak={mx.get_peak_memory()/1e9:.1f}GB",
        flush=True,
    )
    return {
        "C": C,
        "K": K,
        "q4_head": use_q4_head,
        "agg_tok_s": round(agg_tps, 2),
        "per_stream_tok_s": round(agg_tps / C, 2),
        "E_acc": round(e_acc, 3),
        "verify_ms": round(t_verify * 1e3 / n_steps, 2),
        "draft_ms": round(t_sync_draft * 1e3 / n_steps, 2),
        "exact_tok": tok_diffs == 0,
        "exact_lp": lp_diffs == 0,
        "peak_gb": round(mx.get_peak_memory() / 1e9, 2),
    }


def summarize_c1(label: str, rows: list[dict]):
    med_tps = round(statistics.median(r["tok_s"] for r in rows), 2)
    mean_acc = round(sum(r["E_acc"] for r in rows) / len(rows), 3)
    td = sum(r["tok_diffs"] for r in rows)
    ld = sum(r["lp_diffs"] for r in rows)
    n_al = max(len(r["alphas"]) for r in rows)
    mean_alphas = [round(sum(r["alphas"][j] for r in rows if j < len(r["alphas"])) / len(rows), 3) for j in range(n_al)]
    v_ms = round(statistics.median(r["verify_ms"] for r in rows), 2)
    sd_ms = round(statistics.median(r["sync_ms"] + r["draft_ms"] for r in rows), 2)
    print(
        f"  {label:28s} med={med_tps:6.2f} tok/s  E_acc={mean_acc:.3f}  "
        f"v={v_ms:5.1f}ms d+s={sd_ms:4.1f}ms  alphas={mean_alphas}  exact={td==0 and ld==0}",
        flush=True,
    )
    extra = {}
    if "top2_recall" in rows[0]:
        K = len(rows[0]["top2_recall"])
        extra["top2_recall"] = [round(sum(r["top2_recall"][j] for r in rows) / len(rows), 3) for j in range(K)]
        extra["top4_recall"] = [round(sum(r["top4_recall"][j] for r in rows) / len(rows), 3) for j in range(K)]
        print(f"    top2_recall={extra['top2_recall']}  top4_recall={extra['top4_recall']}", flush=True)
    return {
        "label": label,
        "med_tok_s": med_tps,
        "mean_E_acc": mean_acc,
        "verify_ms": v_ms,
        "draft_sync_ms": sd_ms,
        "mean_alphas": mean_alphas,
        "exact_ok": td == 0 and ld == 0,
        "per_prompt": rows,
        **extra,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser(DEFAULT_MODEL))
    ap.add_argument("--base", default=os.path.expanduser(DEFAULT_BASE))
    ap.add_argument("--out", default=os.path.expanduser("~/science/runs/spec_phase2.json"))
    ap.add_argument("--tokens", type=int, default=128)
    a = ap.parse_args()

    maybe_set_recommended_wired_limit()
    base = json.loads(Path(a.base).read_text())
    prompts_data = {nm: {**base["prompts"][nm], "name": nm} for nm, _ in PROMPTS}

    t0 = time.perf_counter()
    model, _ = load(a.model)
    mtp = BailingMTPLayer.from_target(model, a.model, with_q4_head=True)
    mx.eval(model.parameters())
    print(f"Loaded target + MTP + q4_head in {time.perf_counter()-t0:.1f}s ({mx.get_active_memory()/1e9:.1f} GB)", flush=True)

    if not test_mtp_exactness(model, mtp, prompts_data["medical_stemi"]):
        return 1

    print("\n=== C=1 MTP Chain & Q4 vs BF16 Draft Head (4 prompts, exact check) ===", flush=True)
    c1_results = []
    # Warmup compile across K values
    _ = run_c1_spec(model, mtp, prompts_data["chemistry_kmno4"], K=4, max_tokens=16, use_q4_head=True, check_exact=False)

    for q4 in (True, False):
        ks = (1, 2, 3, 4, 5, 6, 8) if q4 else (2, 4, 6)
        for K in ks:
            rows = [
                run_c1_spec(
                    model, mtp, prompts_data[nm], K=K, max_tokens=a.tokens, use_q4_head=q4, check_exact=True, diag_topk=(K == 6)
                )
                for nm, _ in PROMPTS
            ]
            c1_results.append(summarize_c1(f"C=1 K={K} {'q4' if q4 else 'bf16'}", rows))

    print("\n=== C=1 MTP + Suffix Hybrid ===", flush=True)
    for smode, sgram, K, sextra in [
        ("override", 4, 5, 3),
        ("override", 3, 5, 5),
        ("extend", 3, 5, 5),
        ("extend", 4, 4, 4),
    ]:
        rows = [
            run_c1_spec(
                model,
                mtp,
                prompts_data[nm],
                K=K,
                max_tokens=a.tokens,
                use_q4_head=True,
                suffix_mode=smode,
                suffix_min_gram=sgram,
                suffix_extra=sextra,
                check_exact=True,
            )
            for nm, _ in PROMPTS
        ]
        c1_results.append(summarize_c1(f"C=1 K={K}+{sextra} {smode}(g>={sgram})", rows))

    # 256-token run on best K=5 & K=6
    print("\n=== C=1 Longer Generation (256 tokens) ===", flush=True)
    long_rows = []
    for K in (4, 5, 6):
        rows = [
            run_c1_spec(model, mtp, prompts_data[nm], K=K, max_tokens=256, use_q4_head=True, check_exact=True)
            for nm, _ in PROMPTS
        ]
        long_rows.append(summarize_c1(f"C=1 256tok K={K} q4", rows))

    print("\n=== Batched C=2..4 + MTP ===", flush=True)
    batched_results = []
    for C, ks in [(2, (1, 2, 3, 4, 6)), (3, (1, 2, 3, 4)), (4, (1, 2, 3))]:
        for K in ks:
            if C * (K + 1) <= 16:
                batched_results.append(run_batched_spec(model, mtp, prompts_data, C=C, K=K, max_tokens=a.tokens, use_q4_head=True))

    out = {"c1": c1_results, "c1_256": long_rows, "batched": batched_results}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\nSaved {a.out}", flush=True)

    all_ok = all(r["exact_ok"] for r in c1_results + long_rows) and all(r["exact_tok"] and r["exact_lp"] for r in batched_results)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
