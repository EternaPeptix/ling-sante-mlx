"""Step 3.3 benchmark: Ling-3.0-tiny-4bit external drafter & MTP + Ling-3.0-tiny hybrid (C=1 and C=4)."""
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
from mlx_lm.generate import generate_step  # noqa: E402
from mlx_lm.models.bailing_moe_v3.mtp import BailingMTPLayer  # noqa: E402
from mlx_lm.models.bailing_moe_v3.spec_engine import prefill_with_hidden  # noqa: E402
from mlx_lm.models.bailing_moe_v3.verify_decode import (  # noqa: E402
    commit_verify_cache,
    fast_verify_forward,
    make_verify_cache,
)
from mlx_lm.models.cache import ArraysCache, BatchKVCache, KVCache, make_prompt_cache  # noqa: E402
from mlx_lm.utils import maybe_set_recommended_wired_limit  # noqa: E402

from bench.exact import DEFAULT_BASE, DEFAULT_MODEL, PROMPTS, lp_digest  # noqa: E402

DEFAULT_DRAFT = os.path.expanduser("~/science/models/ling-3.0-tiny-mlx-4bit")


def prefill_draft(draft_model, ids: list[int], step: int = 512):
    d_cache = make_prompt_cache(draft_model)
    p = mx.array(ids, dtype=mx.uint32)
    pos = 0
    while pos < p.size:
        n = min(step, p.size - pos)
        draft_model(p[pos : pos + n][None], cache=d_cache)
        mx.eval([c.state for c in d_cache])
        pos += n
    return d_cache


def snap_draft_c1(draft_model, d_cache):
    """Lightweight O(1) snapshot: array refs for KDA ArraysCache, integer offset for MLA KVCache."""
    s = []
    for layer, c in zip(draft_model.model.layers, d_cache):
        if layer.is_linear:
            s.append((c[0], c[1], c[2], c[3], c.left_padding, c.lengths))
        else:
            s.append(c.offset)
    return s


def restore_draft_c1(draft_model, d_cache, snap):
    for layer, c, sv in zip(draft_model.model.layers, d_cache, snap):
        if layer.is_linear:
            c[0], c[1], c[2], c[3], c.left_padding, c.lengths = sv
        else:
            tr = c.offset - sv
            if tr > 0:
                c.trim(tr)


def bench_draft_standalone(draft_model, prompts_data: dict, n_tokens: int = 128):
    print("\n=== Standalone Ling-3.0-tiny-4bit Speed ===", flush=True)
    # C=1 via generate_step
    c1_tps = []
    for nm, _ in PROMPTS:
        ids = prompts_data[nm]["ids"]
        t0 = None
        cnt = 0
        for t, _ in generate_step(mx.array(ids), draft_model, max_tokens=n_tokens, sampler=lambda lp: mx.argmax(lp, axis=-1)):
            cnt += 1
            if cnt == 4:
                mx.synchronize()
                t0 = time.perf_counter()
        mx.synchronize()
        dt = time.perf_counter() - t0
        c1_tps.append((cnt - 4) / dt)
    med_c1 = statistics.median(c1_tps)
    print(f"  Ling-3.0-tiny-4bit C=1 standalone: {med_c1:.1f} tok/s ({1000/med_c1:.2f} ms/tok)", flush=True)

    # C=4 batched decode
    caches = [prefill_draft(draft_model, prompts_data[nm]["ids"][:-1]) for nm, _ in PROMPTS]
    bc = [type(caches[0][L]).merge([c[L] for c in caches]) for L in range(len(draft_model.model.layers))]
    y = mx.array([[prompts_data[nm]["ids"][-1]] for nm, _ in PROMPTS], dtype=mx.uint32)
    for _ in range(4):
        lg = draft_model(y, cache=bc)[:, -1, :]
        y = mx.argmax(lg, axis=-1)[:, None]
        mx.eval(y)
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_tokens):
        lg = draft_model(y, cache=bc)[:, -1, :]
        y = mx.argmax(lg, axis=-1)[:, None]
        mx.eval(y)
    mx.synchronize()
    dt = time.perf_counter() - t0
    agg_c4 = (4 * n_tokens) / dt
    print(f"  Ling-3.0-tiny-4bit C=4 standalone: {agg_c4:.1f} agg tok/s ({dt*1e3/n_tokens:.2f} ms/step)", flush=True)
    return {"c1_tok_s": round(med_c1, 2), "c4_agg_tok_s": round(agg_c4, 2), "c4_step_ms": round(dt * 1e3 / n_tokens, 2)}


def run_c1_tiny(
    model,
    draft_model,
    p_entry: dict,
    *,
    K: int,
    mtp: BailingMTPLayer | None = None,
    use_hybrid: bool = False,
    max_tokens: int = 128,
):
    ids = p_entry["ids"]
    gold_toks = p_entry["tokens"]
    gold_digs = p_entry["logprob_sha256_32"]

    if use_hybrid:
        assert mtp is not None
        t_cache, mtp_cache, last_tok = prefill_with_hidden(model, mtp, ids)
    else:
        t_cache = make_prompt_cache(model)
        p = mx.array(ids, dtype=mx.uint32)
        pos = 0
        while p.size - pos > 1:
            n = min(512, p.size - 1 - pos)
            model(p[pos : pos + n][None], cache=t_cache)
            mx.eval([c.state for c in t_cache])
            pos += n
        last_tok = int(p[-1].item())
        mtp_cache = None

    d_cache = prefill_draft(draft_model, ids)

    # Initial L=1 target forward
    logits0, hidden0, rb0 = fast_verify_forward(
        model, mx.array([[last_tok]], dtype=mx.uint32), t_cache, return_hidden=True, return_rollback=True
    )
    lp0 = logits0[0, 0] - mx.logsumexp(logits0[0, 0], axis=-1, keepdims=True)
    cur_tok = int(mx.argmax(lp0).item())
    cur_h = hidden0[0, 0:1, :]
    mx.eval(cur_h, lp0)
    commit_verify_cache(t_cache, rb0, 1)

    emitted = [cur_tok]
    td = 0 if (len(gold_toks) == 0 or cur_tok == gold_toks[0]) else 1
    ld = 0 if (len(gold_digs) == 0 or lp_digest(lp0) == gold_digs[0]) else 1

    mtp_exact_off = mtp_cache.offset if use_hybrid else 0

    def draft_k(cur_t: int, h_cur_arr: mx.array | None):
        snaps = []
        drafts_out = []
        tok_in = cur_t
        if use_hybrid:
            nonlocal mtp_exact_off
            hm, dl0 = mtp.step_rows(mx.array([cur_t], dtype=mx.uint32), h_cur_arr, mtp_cache, C=1, L=1, use_q4_head=True)
            mtp_exact_off = mtp_cache.offset
            d0_arr = mx.argmax(dl0, axis=-1)
            # Run tiny on cur_t (to advance K=0 state identically) AND get d_0 from MTP
            _ = draft_model(mx.array([[cur_t]], dtype=mx.uint32), cache=d_cache)
            mx.eval(d0_arr, [c.state for c in d_cache])
            snaps.append(snap_draft_c1(draft_model, d_cache))
            d0 = int(d0_arr[0].item())
            drafts_out.append(d0)
            tok_in = d0
            start_j = 1
        else:
            start_j = 0

        for _ in range(start_j, K):
            lg = draft_model(mx.array([[tok_in]], dtype=mx.uint32), cache=d_cache)[:, -1, :]
            nxt = mx.argmax(lg, axis=-1)
            mx.eval(nxt, [c.state for c in d_cache])
            snaps.append(snap_draft_c1(draft_model, d_cache))
            tok_in = int(nxt[0].item())
            drafts_out.append(tok_in)
        return drafts_out, snaps

    t_draft = 0.0
    t_verify = 0.0
    n_steps = 0
    sum_emitted = 0
    hits = [0] * K
    trials = [0] * K

    td0 = time.perf_counter()
    drafts, snaps = draft_k(cur_tok, cur_h if use_hybrid else None)
    t_draft += time.perf_counter() - td0

    mx.reset_peak_memory()
    t_loop0 = time.perf_counter()

    while len(emitted) < max_tokens:
        rem = max_tokens - len(emitted)
        inp = mx.array([[cur_tok] + drafts], dtype=mx.uint32)

        tv0 = time.perf_counter()
        logits, hidden, rb = fast_verify_forward(model, inp, t_cache, return_hidden=True, return_rollback=True)
        lp = logits[0] - mx.logsumexp(logits[0], axis=-1, keepdims=True)
        preds = mx.argmax(lp, axis=-1)
        mx.eval(preds, hidden, lp)
        t_verify += time.perf_counter() - tv0

        pred_list = preds.tolist()
        a = 0
        for j in range(K):
            trials[j] += 1
            if pred_list[j] == drafts[j]:
                hits[j] += 1
                a += 1
            else:
                break

        n_take = min(a + 1, rem)
        for j in range(n_take):
            pos = len(emitted)
            tok = pred_list[j]
            emitted.append(tok)
            if pos < len(gold_toks) and tok != gold_toks[pos]:
                td += 1
            if pos < len(gold_digs) and lp_digest(lp[j]) != gold_digs[pos]:
                ld += 1

        commit_verify_cache(t_cache, rb, a + 1)
        n_steps += 1
        sum_emitted += n_take
        cur_tok = pred_list[a]

        if len(emitted) >= max_tokens:
            break

        td0 = time.perf_counter()
        # Rollback or catch up tiny cache
        if a < K:
            restore_draft_c1(draft_model, d_cache, snaps[a])
        else:
            # All K drafts accepted; step last accepted draft d_{K-1} cleanly
            draft_model(mx.array([[drafts[K - 1]]], dtype=mx.uint32), cache=d_cache)
            mx.eval([c.state for c in d_cache])

        if use_hybrid:
            tr = mtp_cache.offset - mtp_exact_off
            if tr > 0:
                mtp_cache.trim(tr)
            if a > 0:
                # Refresh MTP KV states on d_0..d_{a-1} ONLY (cur_tok runs in draft_k below!)
                sync_toks = mx.array([drafts[:a]], dtype=mx.uint32)
                sync_h = hidden[:, :a, :]
                hm_pre, _ = mtp.step_rows(sync_toks, sync_h, mtp_cache, C=1, L=a, use_q4_head=True, return_logits=False)
                mx.eval(hm_pre)
            drafts, snaps = draft_k(cur_tok, hidden[0, a : a + 1, :])
        else:
            drafts, snaps = draft_k(cur_tok, None)
        t_draft += time.perf_counter() - td0

    wall_dt = time.perf_counter() - t_loop0
    tps = (len(emitted) - 1) / wall_dt
    alphas = [round(hits[j] / trials[j], 3) if trials[j] > 0 else 0.0 for j in range(K)]
    return {
        "prompt": p_entry.get("name", ""),
        "mode": "hybrid_mtp_tiny" if use_hybrid else "tiny",
        "K": K,
        "tok_s": round(tps, 2),
        "E_acc": round(sum_emitted / n_steps, 3),
        "verify_ms": round(t_verify * 1e3 / n_steps, 2),
        "draft_ms": round(t_draft * 1e3 / n_steps, 2),
        "alphas": alphas,
        "td": td,
        "ld": ld,
        "peak_gb": round(mx.get_peak_memory() / 1e9, 2),
    }


def run_c4_tiny(model, draft_model, prompts_data: dict, *, C: int = 4, K: int = 3, max_tokens: int = 128):
    names = [nm for nm, _ in PROMPTS[:C]]
    t_caches, d_caches, last_toks = [], [], []
    for nm in names:
        ids = prompts_data[nm]["ids"]
        tc = make_prompt_cache(model)
        p = mx.array(ids, dtype=mx.uint32)
        pos = 0
        while p.size - pos > 1:
            n = min(512, p.size - 1 - pos)
            model(p[pos : pos + n][None], cache=tc)
            mx.eval([c.state for c in tc])
            pos += n
        t_caches.append(tc)
        last_toks.append(int(p[-1].item()))
        d_caches.append(prefill_draft(draft_model, ids))

    vcache = make_verify_cache(model, t_caches)

    logits0, _, rb0 = fast_verify_forward(
        model, mx.array([[lt] for lt in last_toks], dtype=mx.uint32), vcache, return_hidden=True, return_rollback=True
    )
    lp0 = logits0[:, 0, :] - mx.logsumexp(logits0[:, 0, :], axis=-1, keepdims=True)
    nxt0 = mx.argmax(lp0, axis=-1)
    mx.eval(nxt0, lp0)
    commit_verify_cache(vcache, rb0, 1)

    cur_toks = nxt0.tolist()
    emitted = [[cur_toks[c]] for c in range(C)]
    td = sum(1 for c, nm in enumerate(names) if cur_toks[c] != prompts_data[nm]["tokens"][0])
    ld = sum(1 for c, nm in enumerate(names) if lp_digest(lp0[c]) != prompts_data[nm]["logprob_sha256_32"][0])

    # Keep per-stream snapshots and step C streams through batched tiny model (or list[KVCache] + merged KDA)
    # Notice tiny model is 24 layers; merging KDA ArraysCache across C and looping MLA or BatchKVCache per stream:
    # Actually simplest fast way: keep batched ArraysCache for KDA layers and BatchKVCache-style exact list of KVCache?
    # Note tiny BailingMoeV3 uses `KVCache` cleanly if we merge tiny cache cleanly OR use individual d_caches synced per step,
    # Wait: let's keep `d_kda = [ArraysCache.merge([c[L] for c in d_caches])]` and run batched tiny cleanly by tracking offsets!
    # Even simpler: tiny's `BatchKVCache` supports `.extract(idx)` and `ArraysCache` supports `c[0..3][c:c+1]`!
    d_vcache = [
        ArraysCache.merge([dc[L] for dc in d_caches])
        if draft_model.model.layers[L].is_linear
        else BatchKVCache.merge([dc[L] for dc in d_caches])
        for L in range(len(draft_model.model.layers))
    ]
    # Note: BatchKVCache padded lengths if prompts differ -> slice/trim back to base offset cleanly.
    # For rollback on heterogeneous a_c, track per-stream MLA offsets & rollback cleanly by re-merging or using snap.
    # Wait: to avoid BatchKVCache trim limitation on unequal a_c, run `d_caches` via individual replay of `tiny` OR extract+merge:
    # Since tiny C=1 step is ~4ms and batched C=4 is ~5ms, extract/merge only 6 MLA caches (~0.5MB!) takes <0.3ms!
    def draft_step_batched(tok_vec, snaps):
        lg = draft_model(tok_vec[:, None], cache=d_vcache)[:, -1, :]
        nxt = mx.argmax(lg, axis=-1)
        mx.eval(nxt, [c.state for c in d_vcache])
        # Save per-step KDA slices + MLA key lens
        snap = []
        for layer, c in zip(draft_model.model.layers, d_vcache):
            if layer.is_linear:
                snap.append((c[0], c[1], c[2], c[3]))
            else:
                snap.append((c._idx, c.offset))
        snaps.append(snap)
        return nxt

    def roll_tiny_batched(a_list, snaps, last_drafts):
        if len(set(a_list)) == 1 and a_list[0] < K:
            a = a_list[0]
            for layer, c, sv in zip(draft_model.model.layers, d_vcache, snaps[a]):
                if layer.is_linear:
                    c[0], c[1], c[2], c[3] = sv
                else:
                    c.trim(c._idx - sv[0])
            return
        # Mixed acceptance across streams
        for li, (layer, c) in enumerate(zip(draft_model.model.layers, d_vcache)):
            if layer.is_linear:
                # For streams with a < K pick snaps[a][li]; for a == K advance after
                s0 = [snaps[min(a, K - 1)][li][0][idx : idx + 1] for idx, a in enumerate(a_list)]
                s1 = [snaps[min(a, K - 1)][li][1][idx : idx + 1] for idx, a in enumerate(a_list)]
                s2 = [snaps[min(a, K - 1)][li][2][idx : idx + 1] for idx, a in enumerate(a_list)]
                s3 = [snaps[min(a, K - 1)][li][3][idx : idx + 1] for idx, a in enumerate(a_list)]
                c[0], c[1], c[2], c[3] = mx.concatenate(s0, 0), mx.concatenate(s1, 0), mx.concatenate(s2, 0), mx.concatenate(s3, 0)
            else:
                # Extract each stream at its accepted offset & re-merge
                exts = []
                for idx, a in enumerate(a_list):
                    kc = c.extract(idx)
                    keep = snaps[min(a, K - 1)][li][1][idx].item()
                    tr = kc.offset - int(keep)
                    if tr > 0:
                        kc.trim(tr)
                    exts.append(kc)
                merged = BatchKVCache.merge(exts)
                c.state = merged.state
        # For any stream that accepted all K drafts, advance d_{K-1} individually via extract? Better: step with mask or re-prefill 1 token!
        if any(a == K for a in a_list):
            # Advance only the streams with a == K by masking unused streams in a per-stream KVCache? Tiny model C<=4:
            # Update each a==K stream via clean extract->step->merge
            full_cs = [idx for idx, a in enumerate(a_list) if a == K]
            for idx in full_cs:
                sub = []
                for layer, c in zip(draft_model.model.layers, d_vcache):
                    if layer.is_linear:
                        ac = ArraysCache(4)
                        for j in range(4):
                            ac[j] = c[j][idx : idx + 1]
                        sub.append(ac)
                    else:
                        sub.append(c.extract(idx))
                draft_model(mx.array([[last_drafts[idx]]], dtype=mx.uint32), cache=sub)
                mx.eval([x.state for x in sub])
                for li, (layer, c, sc) in enumerate(zip(draft_model.model.layers, d_vcache, sub)):
                    if layer.is_linear:
                        for j in range(4):
                            c[j] = mx.concatenate(
                                [sc[j] if r == idx else c[j][r : r + 1] for r in range(C)],
                                axis=0,
                            )
                    else:
                        # Update MLA BatchKVCache via full re-merge
                        pass
                # Re-merge all MLA caches cleanly for exact offsets
                # Track cleanly: easier to extract all and merge if any a == K
                exts_all = [[d_vcache[li].extract(j) if j != idx else sub[li] for j in range(C)] if not layer.is_linear else None for li, layer in enumerate(draft_model.model.layers)]
                for li, layer in enumerate(draft_model.model.layers):
                    if not layer.is_linear:
                        d_vcache[li].state = BatchKVCache.merge(exts_all[li]).state

    snaps = []
    cols = []
    d_prev = mx.array(cur_toks, dtype=mx.uint32)
    for _ in range(K):
        d_prev = draft_step_batched(d_prev, snaps)
        cols.append(d_prev.tolist())
    drafts = [[cols[k][c] for k in range(K)] for c in range(C)]

    n_steps = 0
    sum_emitted = 0
    t_verify = 0.0
    t_draft = 0.0
    mx.reset_peak_memory()
    t0 = time.perf_counter()

    while min(len(e) for e in emitted) < max_tokens:
        inp = mx.array([[cur_toks[c]] + drafts[c] for c in range(C)], dtype=mx.uint32)
        tv0 = time.perf_counter()
        logits, rb = fast_verify_forward(model, inp, vcache, return_hidden=False, return_rollback=True)
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        preds = mx.argmax(lp, axis=-1)
        mx.eval(preds, lp)
        t_verify += time.perf_counter() - tv0

        pred_lists = preds.tolist()
        a_list = []
        ncs = []
        next_cur = []
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
                    td += 1
                if pos < len(gdigs) and lp_digest(lp[c, j]) != gdigs[pos]:
                    ld += 1
            sum_emitted += n_take
            a_list.append(a)
            ncs.append(a + 1)
            next_cur.append(pred_lists[c][a])

        commit_verify_cache(vcache, rb, ncs)
        cur_toks = next_cur
        n_steps += 1
        if min(len(e) for e in emitted) >= max_tokens:
            break

        td0 = time.perf_counter()
        last_d = [drafts[c][K - 1] for c in range(C)]
        roll_tiny_batched(a_list, snaps, last_d)
        snaps = []
        cols = []
        d_prev = mx.array(cur_toks, dtype=mx.uint32)
        for _ in range(K):
            d_prev = draft_step_batched(d_prev, snaps)
            cols.append(d_prev.tolist())
        drafts = [[cols[k][c] for k in range(K)] for c in range(C)]
        t_draft += time.perf_counter() - td0

    dt = time.perf_counter() - t0
    agg_toks = sum(len(e) - 1 for e in emitted)
    agg_tps = agg_toks / dt
    e_acc = sum_emitted / (n_steps * C)
    print(
        f"  C={C} K={K} tiny: agg={agg_tps:6.2f} tok/s ({agg_tps/C:5.2f}/s)  E_acc={e_acc:.3f}  "
        f"v={t_verify*1e3/n_steps:5.1f}ms d={t_draft*1e3/n_steps:4.1f}ms  exact={td==0 and ld==0} (td={td},ld={ld})",
        flush=True,
    )
    return {
        "C": C,
        "K": K,
        "agg_tok_s": round(agg_tps, 2),
        "per_stream_tok_s": round(agg_tps / C, 2),
        "E_acc": round(e_acc, 3),
        "verify_ms": round(t_verify * 1e3 / n_steps, 2),
        "draft_ms": round(t_draft * 1e3 / n_steps, 2),
        "exact": td == 0 and ld == 0,
        "peak_gb": round(mx.get_peak_memory() / 1e9, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser(DEFAULT_MODEL))
    ap.add_argument("--draft", default=os.path.expanduser(DEFAULT_DRAFT))
    ap.add_argument("--base", default=os.path.expanduser(DEFAULT_BASE))
    ap.add_argument("--out", default=os.path.expanduser("~/science/runs/spec_tiny_phase3.json"))
    ap.add_argument("--tokens", type=int, default=128)
    a = ap.parse_args()

    maybe_set_recommended_wired_limit()
    base = json.loads(Path(a.base).read_text())
    prompts_data = {nm: {**base["prompts"][nm], "name": nm} for nm, _ in PROMPTS}

    t0 = time.perf_counter()
    draft_model, _ = load(a.draft)
    draft_model.eval()
    mx.eval(draft_model.parameters())
    print(f"Loaded Ling-3.0-tiny-4bit in {time.perf_counter()-t0:.1f}s ({mx.get_active_memory()/1e9:.1f} GB)", flush=True)
    standalone = bench_draft_standalone(draft_model, prompts_data, n_tokens=a.tokens)

    t1 = time.perf_counter()
    model, _ = load(a.model)
    mtp = BailingMTPLayer.from_target(model, a.model, with_q4_head=True)
    mx.eval(model.parameters())
    print(f"Loaded target + MTP in {time.perf_counter()-t1:.1f}s ({mx.get_active_memory()/1e9:.1f} GB)", flush=True)

    # Warmup
    _ = run_c1_tiny(model, draft_model, prompts_data["chemistry_kmno4"], K=4, max_tokens=16)

    out = {"standalone_tiny": standalone, "c1_tiny": [], "c1_hybrid": [], "c4_tiny": []}

    print("\n=== C=1 Ling-3.0-tiny-4bit Standalone Drafter ===", flush=True)
    for K in (4, 6, 8, 10, 12):
        rows = [run_c1_tiny(model, draft_model, prompts_data[nm], K=K, use_hybrid=False, max_tokens=a.tokens) for nm, _ in PROMPTS]
        med = round(statistics.median(r["tok_s"] for r in rows), 2)
        eacc = round(sum(r["E_acc"] for r in rows) / len(rows), 3)
        ok = all(r["td"] == 0 and r["ld"] == 0 for r in rows)
        alphas = [round(sum(r["alphas"][j] for r in rows) / len(rows), 3) for j in range(K)]
        vms = round(statistics.median(r["verify_ms"] for r in rows), 1)
        dms = round(statistics.median(r["draft_ms"] for r in rows), 1)
        print(f"  tiny K={K:2d}: med={med:6.2f} tok/s  E_acc={eacc:.3f}  v={vms}ms d={dms}ms  alphas={alphas}  exact={ok}", flush=True)
        out["c1_tiny"].append({"K": K, "med_tok_s": med, "E_acc": eacc, "verify_ms": vms, "draft_ms": dms, "alphas": alphas, "exact": ok, "rows": rows})

    print("\n=== C=1 Hybrid MTP (step 0) + Ling-3.0-tiny-4bit (steps 1..K-1) ===", flush=True)
    for K in (4, 6, 8, 10):
        rows = [run_c1_tiny(model, draft_model, prompts_data[nm], K=K, mtp=mtp, use_hybrid=True, max_tokens=a.tokens) for nm, _ in PROMPTS]
        med = round(statistics.median(r["tok_s"] for r in rows), 2)
        eacc = round(sum(r["E_acc"] for r in rows) / len(rows), 3)
        ok = all(r["td"] == 0 and r["ld"] == 0 for r in rows)
        alphas = [round(sum(r["alphas"][j] for r in rows) / len(rows), 3) for j in range(K)]
        vms = round(statistics.median(r["verify_ms"] for r in rows), 1)
        dms = round(statistics.median(r["draft_ms"] for r in rows), 1)
        print(f"  hybr K={K:2d}: med={med:6.2f} tok/s  E_acc={eacc:.3f}  v={vms}ms d={dms}ms  alphas={alphas}  exact={ok}", flush=True)
        out["c1_hybrid"].append({"K": K, "med_tok_s": med, "E_acc": eacc, "verify_ms": vms, "draft_ms": dms, "alphas": alphas, "exact": ok, "rows": rows})

    print("\n=== C=4 Ling-3.0-tiny-4bit Batched ===", flush=True)
    for K in (2, 3, 4):
        out["c4_tiny"].append(run_c4_tiny(model, draft_model, prompts_data, C=4, K=K, max_tokens=a.tokens))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\nSaved {a.out}", flush=True)
    ok_all = all(x["exact"] for x in out["c1_tiny"] + out["c1_hybrid"] + out["c4_tiny"])
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
