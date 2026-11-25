"""Phase 1 DFlash2 C=1 bit-exact benchmark, oracle UBTree potential, and L=8..24 verify ledger."""
import argparse
import heapq
import json
import math
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
from mlx_lm.models.bailing_moe_v3.dflash2 import DFlash2DraftModel, prefill_target_and_dflash  # noqa: E402
from mlx_lm.models.bailing_moe_v3.rows.gemv_fp8 import (  # noqa: E402
    gemv_lin,
    gemv_lin_multi,
    gemv_lin_res,
    gemv_lin_swiglu,
)
from mlx_lm.models.bailing_moe_v3.rows.kda import (  # noqa: E402
    kda_gdn_verify_seq_rows,
    kda_pre_seq_rows,
)
from mlx_lm.models.bailing_moe_v3.rows.mla import mla_verify_layer_rows  # noqa: E402
from mlx_lm.models.bailing_moe_v3.rows.moe import (  # noqa: E402
    moe_dedup_rows,
    moe_down_combine_rows,
    moe_gate_up_swiglu_rows,
)
from mlx_lm.models.bailing_moe_v3.rows.router import gate_rows  # noqa: E402
from mlx_lm.models.bailing_moe_v3.spec_engine import _logits_to_preds  # noqa: E402
from mlx_lm.models.bailing_moe_v3.verify_decode import commit_verify_cache, fast_verify_forward  # noqa: E402
from mlx_lm.utils import maybe_set_recommended_wired_limit  # noqa: E402

from bench.exact import DEFAULT_BASE, DEFAULT_MODEL, PROMPTS, lp_digest  # noqa: E402

DEFAULT_DFLASH = os.path.expanduser("~/science/models/ling-3.1-flash-dflash2")


def verify_exact(emitted: list[int], lp_arrays: list[mx.array], gold_entry: dict, max_tokens: int):
    td = sum(1 for a, b in zip(emitted[:max_tokens], gold_entry["tokens"][:max_tokens]) if a != b)
    gdigs = gold_entry["logprob_sha256_32"]
    ld = sum(1 for a, b in zip(lp_arrays[: len(gdigs)], gdigs) if lp_digest(a) != b)
    return td, ld


def run_dflash_c1_prompt(
    model,
    dflash: DFlash2DraftModel,
    ids: list[int],
    *,
    block_size: int = 8,
    use_q4_head: bool = True,
    use_selector: bool = True,
    max_tokens: int = 128,
    layer_ids: tuple[int, ...] | None = None,
    measure_phases: bool = False,
):
    lids = dflash.target_layer_ids if layer_ids is None else tuple(layer_ids)
    K = block_size - 1
    t_cache, d_cache, y0, lp0, feats0 = prefill_target_and_dflash(
        model, dflash, ids, step=512, layer_ids=lids
    )
    anchor0 = int(y0.item())
    emitted = [anchor0]
    lp_arrays = [lp0]
    draft_cache_args = d_cache

    if measure_phases:
        # Clean step-timed loop so draft / verify / commit are directly isolated
        cur_anchor = y0
        cur_feats = feats0
        draft_ms_list, verify_ms_list, commit_ms_list = [], [], []
        hits = [0] * K
        trials = [0] * K
        accs = []
        mx.reset_peak_memory()
        t0 = time.perf_counter()
        while len(emitted) < max_tokens:
            td0 = time.perf_counter()
            draft_toks, _, _ = dflash.draft_block(
                cur_anchor,
                cur_feats,
                draft_cache_args,
                block_size=block_size,
                use_q4_head=use_q4_head,
                use_selector=use_selector,
                return_lattice=False,
            )
            mx.eval(draft_toks, *[c.state for c in d_cache])
            draft_ms_list.append((time.perf_counter() - td0) * 1e3)

            inp = mx.concatenate([cur_anchor.reshape(1, 1), draft_toks[None]], axis=1)
            tv0 = time.perf_counter()
            logits, rb = fast_verify_forward(
                model, inp, t_cache, return_hidden=False, return_rollback=True, capture_layers=lids
            )
            preds, lp = _logits_to_preds(logits)
            mx.eval(preds, lp, rb["captured"])
            verify_ms_list.append((time.perf_counter() - tv0) * 1e3)

            pred_row = preds[0].tolist()
            draft_row = draft_toks.tolist()
            a = 0
            for j in range(K):
                trials[j] += 1
                if pred_row[j] == draft_row[j]:
                    hits[j] += 1
                    a += 1
                else:
                    break
            rem = max_tokens - len(emitted)
            n_take = min(a + 1, rem)
            emitted.extend(pred_row[:n_take])
            for j in range(n_take):
                if len(lp_arrays) < 32:
                    lp_arrays.append(lp[0, j])
            accs.append(a + 1)
            nc = a + 1

            tc0 = time.perf_counter()
            commit_verify_cache(t_cache, rb, nc)
            mx.eval(*[x for li, lyr in enumerate(model.model.layers) if lyr.is_linear for x in t_cache[li].cache if x is not None])
            commit_ms_list.append((time.perf_counter() - tc0) * 1e3)

            cur_anchor = preds[0, a]
            cur_feats = rb["captured"][:, :nc, :]

        wall_dt = time.perf_counter() - t0
        gen_toks = len(emitted) - 1
        return {
            "emitted": emitted,
            "lp_arrays": lp_arrays,
            "tok_s": gen_toks / wall_dt,
            "E_acc": sum(accs) / len(accs),
            "steps": len(accs),
            "cycle_ms": statistics.median(
                [d + v + c for d, v, c in zip(draft_ms_list, verify_ms_list, commit_ms_list)]
            ),
            "draft_ms": statistics.median(draft_ms_list),
            "verify_ms": statistics.median(verify_ms_list),
            "commit_ms": statistics.median(commit_ms_list),
            "alphas": [round(hits[j] / trials[j], 3) if trials[j] > 0 else 0.0 for j in range(K)],
            "peak_gb": round(mx.get_peak_memory() / 1e9, 2),
        }

    # Production single-sync async-pipelined loop
    draft0, _, _ = dflash.draft_block(
        y0,
        feats0,
        d_cache,
        block_size=block_size,
        use_q4_head=use_q4_head,
        use_selector=use_selector,
        return_lattice=False,
    )
    mx.async_eval(draft0)
    inp0 = mx.concatenate([y0.reshape(1, 1), draft0[None]], axis=1)
    logits, rb = fast_verify_forward(
        model, inp0, t_cache, return_hidden=False, return_rollback=True, capture_layers=lids
    )
    preds, lp = _logits_to_preds(logits)

    mx.reset_peak_memory()
    t0 = time.perf_counter()
    mx.async_eval(preds, lp, draft0, rb["captured"])

    draft_toks = draft0
    hits = [0] * K
    trials = [0] * K
    accs = []
    n_steps = 0

    while True:
        mx.eval(preds, lp, draft_toks, rb["captured"])
        pred_row = preds[0].tolist()
        draft_row = draft_toks.tolist()

        a = 0
        for j in range(K):
            trials[j] += 1
            if pred_row[j] == draft_row[j]:
                hits[j] += 1
                a += 1
            else:
                break

        rem = max_tokens - len(emitted)
        n_take = min(a + 1, rem)
        pending_toks = pred_row[:n_take]
        base_pos = len(emitted)
        pending_lps = [lp[0, j] for j in range(n_take) if base_pos + j < 32]
        accs.append(n_take)
        n_steps += 1

        if base_pos + n_take >= max_tokens:
            emitted.extend(pending_toks)
            lp_arrays.extend(pending_lps)
            commit_verify_cache(t_cache, rb, n_take)
            mx.synchronize()
            break

        nc = a + 1
        commit_verify_cache(t_cache, rb, nc)
        next_anchor = preds[0, a]
        new_feats = rb["captured"][:, :nc, :]

        draft_next, _, _ = dflash.draft_block(
            next_anchor,
            new_feats,
            d_cache,
            block_size=block_size,
            use_q4_head=use_q4_head,
            use_selector=use_selector,
            return_lattice=False,
        )
        flush_arrs = [draft_next]
        for li, lyr in enumerate(model.model.layers):
            if lyr.is_linear:
                flush_arrs.extend(x for x in t_cache[li].cache if x is not None)
        mx.async_eval(*flush_arrs)

        inp_next = mx.concatenate([next_anchor.reshape(1, 1), draft_next[None]], axis=1)
        logits_next, rb_next = fast_verify_forward(
            model, inp_next, t_cache, return_hidden=False, return_rollback=True, capture_layers=lids
        )
        preds_next, lp_next = _logits_to_preds(logits_next)
        mx.async_eval(preds_next, lp_next, draft_next, rb_next["captured"])

        emitted.extend(pending_toks)
        lp_arrays.extend(pending_lps)
        preds, lp, draft_toks, rb = preds_next, lp_next, draft_next, rb_next

    wall_dt = time.perf_counter() - t0
    gen_toks = len(emitted) - 1
    return {
        "emitted": emitted,
        "lp_arrays": lp_arrays,
        "tok_s": gen_toks / wall_dt,
        "E_acc": sum(accs) / n_steps,
        "steps": n_steps,
        "cycle_ms": (wall_dt * 1e3) / n_steps,
        "alphas": [round(hits[j] / trials[j], 3) if trials[j] > 0 else 0.0 for j in range(K)],
        "peak_gb": round(mx.get_peak_memory() / 1e9, 2),
    }


def _oracle_tree_match_depth(
    cand_ids_list: list[list[int]],
    lattice_list: list[list[list[float]]],
    gold_future: list[int],
    budget_nodes: int,
) -> int:
    """Build greedy top-probability tree of `budget_nodes` (anchor + B-1 candidates) and return accepted drafts."""
    K_pos = len(cand_ids_list)
    if budget_nodes <= 1 or K_pos == 0 or not gold_future:
        return 0
    top_k = len(cand_ids_list[0])
    # Convert row scores along child axis into log-probs
    logp = []
    for pos in range(K_pos):
        pos_rows = []
        n_p = 1 if pos == 0 else top_k
        for p in range(n_p):
            row = lattice_list[pos][p]
            m = max(row)
            lse = m + math.log(sum(math.exp(v - m) for v in row))
            pos_rows.append([v - lse for v in row])
        logp.append(pos_rows)

    # Best-first search over tree nodes (neg_score, depth, slot_idx, token_path)
    # Depth 0 is anchor (1 node in budget). Expand depth=1 children immediately.
    pq = []
    for c in range(top_k):
        heapq.heappush(pq, (-logp[0][0][c], 1, c, (cand_ids_list[0][c],)))

    accepted_paths = []
    chosen = 0
    while pq and chosen < (budget_nodes - 1):
        neg_s, d, slot, path = heapq.heappop(pq)
        accepted_paths.append(path)
        chosen += 1
        if d < K_pos:
            score_so_far = -neg_s
            for c in range(top_k):
                heapq.heappush(
                    pq,
                    (-(score_so_far + logp[d][slot][c]), d + 1, c, path + (cand_ids_list[d][c],)),
                )

    best_match = 0
    max_d = min(K_pos, len(gold_future))
    for path in accepted_paths:
        m = 0
        for j, tok in enumerate(path):
            if j < max_d and tok == gold_future[j]:
                m += 1
            else:
                break
        if m > best_match:
            best_match = m
    return best_match


def run_oracle_ubtree_analysis(model, dflash: DFlash2DraftModel, prompts_data: dict, prompt_names: list[str], *, use_q4_head: bool = True, max_tokens: int = 128):
    """Step along gold trajectory with DFlash2 (B=8) and evaluate oracle UBTree E_acc at B_tree in (8,12,16,20,24,32)."""
    budgets = (8, 12, 16, 20, 24, 32)
    lids = dflash.target_layer_ids
    tot_acc = {b: 0 for b in budgets}
    tot_full7 = {b: 0 for b in budgets}
    n_steps = 0

    for nm in prompt_names:
        entry = prompts_data[nm]
        ids = entry["ids"]
        gold = entry["tokens"]
        t_cache, d_cache, y0, _, feats0 = prefill_target_and_dflash(model, dflash, ids, step=512, layer_ids=lids)
        # Walk strictly along deterministic gold chain using B=8 steps
        cur_anchor = y0
        cur_feats = feats0
        pos = 1  # gold[0] == y0; we check gold[pos:] as future tokens from cur_anchor
        while pos < max_tokens:
            draft_toks, lat_tup, _ = dflash.draft_block(
                cur_anchor,
                cur_feats,
                d_cache,
                block_size=8,
                use_q4_head=use_q4_head,
                use_selector=True,
                return_lattice=True,
            )
            cand_ids, lattice = lat_tup
            mx.eval(draft_toks, cand_ids, lattice, *[c.state for c in d_cache])
            c_list = cand_ids.tolist()
            l_list = lattice.tolist()
            gold_fut = gold[pos : pos + 7]
            if len(gold_fut) == 7:
                for b in budgets:
                    m = _oracle_tree_match_depth(c_list, l_list, gold_fut, b)
                    tot_acc[b] += m + 1
                    if m == 7:
                        tot_full7[b] += 1
                n_steps += 1

            # Advance target strictly on linear chain to get next exact target feats
            inp = mx.concatenate([cur_anchor.reshape(1, 1), draft_toks[None]], axis=1)
            logits, rb = fast_verify_forward(
                model, inp, t_cache, return_hidden=False, return_rollback=True, capture_layers=lids
            )
            preds, _ = _logits_to_preds(logits)
            mx.eval(preds, rb["captured"])
            pred_row = preds[0].tolist()
            draft_row = draft_toks.tolist()
            a = 0
            for j in range(7):
                if pred_row[j] == draft_row[j]:
                    a += 1
                else:
                    break
            nc = min(a + 1, max_tokens - pos)
            commit_verify_cache(t_cache, rb, nc)
            cur_anchor = preds[0, a]
            cur_feats = rb["captured"][:, :nc, :]
            pos += nc

    return {
        "steps": n_steps,
        "chain_e_acc": round(tot_acc[8] / n_steps, 3),
        "budgets": {
            str(b): {
                "oracle_E_acc": round(tot_acc[b] / n_steps, 3),
                "full7_rate": round(tot_full7[b] / n_steps, 3),
            }
            for b in budgets
        },
    }


def _flat(x):
    if isinstance(x, (list, tuple)):
        out = []
        for v in x:
            out.extend(_flat(v))
        return out
    return [x]


def _pipe_ms(fn, n_lo: int = 3, n_hi: int = 15, iters: int = 3) -> float:
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
    return statistics.median(dts)


def profile_c1_verify_range(model, prompts_data: dict, Ls=(8, 12, 16, 20, 24)):
    """Measure C=1 fast_verify_forward latency + pipelined per-component ledger + unique experts U(L)."""
    ids = prompts_data["medical_stemi"]["ids"]
    out = []
    for L in Ls:
        # Fresh prefill + sample real tokens
        from mlx_lm.models.cache import make_prompt_cache

        cache = make_prompt_cache(model)
        p = mx.array(ids, dtype=mx.uint32)
        h_in_s = model(p[:-1][None], cache=cache)
        mx.eval(h_in_s, [c.state for c in cache])
        toks_vec = mx.array([ids[-1]] + prompts_data["medical_stemi"]["tokens"][: L - 1], dtype=mx.uint32)[None]

        # 1. Single-call e2e verify time (trimmed MLA each call, readonly KDA)
        for _ in range(2):
            lg = fast_verify_forward(model, toks_vec, cache, return_hidden=False, return_rollback=False)
            for li, lyr in enumerate(model.model.layers):
                if not lyr.is_linear:
                    cache[li].trim(L)
            mx.eval(lg)
        mx.synchronize()
        e2e_ts = []
        for _ in range(5):
            t0 = time.perf_counter()
            lg = fast_verify_forward(model, toks_vec, cache, return_hidden=False, return_rollback=False)
            mx.eval(lg)
            e2e_ts.append((time.perf_counter() - t0) * 1e3)
            for li, lyr in enumerate(model.model.layers):
                if not lyr.is_linear:
                    cache[li].trim(L)
        e2e_ms = statistics.median(e2e_ts)

        # 2. Compute true unique experts across 68 MoE layers + capture L4/L7 activations
        x = model.model.word_embeddings(toks_vec.reshape(-1))
        uniq_counts = []
        samp = {}
        for li, (layer, lc) in enumerate(zip(model.model.layers, cache)):
            h_in = mx.fast.rms_norm(x, layer.input_layernorm.weight, layer.input_layernorm.eps)
            if li == 4:
                samp["k_in"] = h_in
            if li == 7:
                samp["m_in"] = h_in
                samp["m_x"] = x
                samp["m_c"] = lc
            if layer.is_linear:
                a = layer.attention
                qr, kr, vr, fr, gr, br = gemv_lin_multi(
                    h_in, [a.q_proj, a.k_proj, a.v_proj, a.f_proj, a.g_proj, a.b_proj]
                )
                q, k, v, g, beta, _, _, _ = kda_pre_seq_rows(
                    qr, kr, vr, fr, br, lc[0], lc[1], lc[2],
                    a.q_conv1d.weight, a.k_conv1d.weight, a.v_conv1d.weight, a.A_log, a.dt_bias,
                    C=1, L=L, lower_bound=a.lower_bound, eps=1e-6,
                )
                ov, _ = kda_gdn_verify_seq_rows(q, k, v, g, beta, lc[3], gr, a.o_norm.weight, C=1, L=L, eps=a.o_norm.eps)
                if li == 4:
                    samp.update(dict(a=a, lc=lc, qr=qr, kr=kr, vr=vr, fr=fr, gr=gr, br=br, q=q, k=k, v=v, g=g, beta=beta, ov=ov, x=x))
                h_post = gemv_lin_res(ov, a.o_proj, x)
            else:
                h_post = mla_verify_layer_rows(h_in, x, layer.attention, lc, C=1, L=L)
                lc.trim(L)
            mlp_in = mx.fast.rms_norm(h_post, layer.post_attention_layernorm.weight, layer.post_attention_layernorm.eps)
            mlp = layer.mlp
            if hasattr(mlp, "switch_mlp"):
                gate = mlp.gate
                inds, wts = gate_rows(
                    mlp_in, gate.weight, gate.expert_bias,
                    top_k=gate.top_k, n_group=gate.n_group, topk_group=gate.topk_group,
                    routed_scaling_factor=gate.routed_scaling_factor,
                )
                dd = moe_dedup_rows(inds)
                mx.eval(dd[4])
                uniq_counts.append(int(dd[4].item()))
                se = mlp.shared_experts
                sh = gemv_lin(gemv_lin_swiglu(mlp_in, se.gate_proj, se.up_proj), se.down_proj)
                sw = mlp.switch_mlp
                h_mid = moe_gate_up_swiglu_rows(mlp_in, sw.gate_proj.weight, sw.gate_proj.scales, sw.up_proj.weight, sw.up_proj.scales, inds, dedup=dd)
                if li == 4:
                    samp.update(dict(mlp=mlp, mlp_in=mlp_in, h_post=h_post, inds=inds, wts=wts, dd=dd, sh=sh, h_mid=h_mid))
                x = moe_down_combine_rows(h_mid, sw.down_proj.weight, sw.down_proj.scales, inds, wts, shared=sh, residual=h_post, dedup=dd)
            else:
                if li == 0:
                    samp["d_mlp"] = mlp
                    samp["d_in"] = mlp_in
                    samp["d_post"] = h_post
                x = gemv_lin_res(gemv_lin_swiglu(mlp_in, mlp.gate_proj, mlp.up_proj), mlp.down_proj, h_post)
        mx.eval(x)

        # 3. Pipelined component ledger scaled by layer count
        a4 = samp["a"]
        lc4 = samp["lc"]
        mlp4 = samp["mlp"]
        sw4 = mlp4.switch_mlp
        l7 = model.model.layers[7]
        d0 = samp["d_mlp"]

        t_qkv = _pipe_ms(lambda: gemv_lin_multi(samp["k_in"], [a4.q_proj, a4.k_proj, a4.v_proj, a4.f_proj, a4.g_proj, a4.b_proj])) * 63
        t_kpre = _pipe_ms(lambda: kda_pre_seq_rows(samp["qr"], samp["kr"], samp["vr"], samp["fr"], samp["br"], lc4[0], lc4[1], lc4[2], a4.q_conv1d.weight, a4.k_conv1d.weight, a4.v_conv1d.weight, a4.A_log, a4.dt_bias, C=1, L=L, lower_bound=a4.lower_bound, eps=1e-6)) * 63
        t_gdn = _pipe_ms(lambda: kda_gdn_verify_seq_rows(samp["q"], samp["k"], samp["v"], samp["g"], samp["beta"], lc4[3], samp["gr"], a4.o_norm.weight, C=1, L=L, eps=a4.o_norm.eps)) * 63
        t_op = _pipe_ms(lambda: gemv_lin_res(samp["ov"], a4.o_proj, samp["x"])) * 63

        def _mla():
            o = mla_verify_layer_rows(samp["m_in"], samp["m_x"], l7.attention, samp["m_c"], C=1, L=L)
            samp["m_c"].trim(L)
            return o
        t_mla = _pipe_ms(_mla) * 9

        t_gate = _pipe_ms(lambda: gate_rows(samp["mlp_in"], mlp4.gate.weight, mlp4.gate.expert_bias, top_k=mlp4.gate.top_k, n_group=mlp4.gate.n_group, topk_group=mlp4.gate.topk_group, routed_scaling_factor=mlp4.gate.routed_scaling_factor)) * 68
        t_sh = _pipe_ms(lambda: gemv_lin(gemv_lin_swiglu(samp["mlp_in"], mlp4.shared_experts.gate_proj, mlp4.shared_experts.up_proj), mlp4.shared_experts.down_proj)) * 68
        t_gu = _pipe_ms(lambda: moe_gate_up_swiglu_rows(samp["mlp_in"], sw4.gate_proj.weight, sw4.gate_proj.scales, sw4.up_proj.weight, sw4.up_proj.scales, samp["inds"], dedup=samp["dd"])) * 68
        t_dc = _pipe_ms(lambda: moe_down_combine_rows(samp["h_mid"], sw4.down_proj.weight, sw4.down_proj.scales, samp["inds"], samp["wts"], shared=samp["sh"], residual=samp["h_post"], dedup=samp["dd"])) * 68
        t_den = _pipe_ms(lambda: gemv_lin_res(gemv_lin_swiglu(samp["d_in"], d0.gate_proj, d0.up_proj), d0.down_proj, samp["d_post"])) * 4
        t_head = _pipe_ms(lambda: gemv_lin(x, model.lm_head))

        rec = {
            "L": L,
            "e2e_ms": round(e2e_ms, 2),
            "uniq_exp_mean": round(sum(uniq_counts) / len(uniq_counts), 1),
            "kda_qkvfgb_x63": round(t_qkv, 2),
            "kda_pre_x63": round(t_kpre, 2),
            "kda_gdn_x63": round(t_gdn, 2),
            "kda_oproj_x63": round(t_op, 2),
            "mla_x9": round(t_mla, 2),
            "moe_gate_x68": round(t_gate, 2),
            "moe_shared_x68": round(t_sh, 2),
            "moe_gu_x68": round(t_gu, 2),
            "moe_dc_x68": round(t_dc, 2),
            "dense_x4": round(t_den, 2),
            "lm_head": round(t_head, 2),
            "pipe_sum": round(t_qkv + t_kpre + t_gdn + t_op + t_mla + t_gate + t_sh + t_gu + t_dc + t_den + t_head, 2),
        }
        out.append(rec)
        print(
            f"  Verify C=1 L={L:2d}: e2e={rec['e2e_ms']:6.2f}ms  pipe={rec['pipe_sum']:6.2f}ms  "
            f"U={rec['uniq_exp_mean']:5.1f}  (qkv={rec['kda_qkvfgb_x63']:.1f}, gu={rec['moe_gu_x68']:.1f}, "
            f"dc={rec['moe_dc_x68']:.1f}, gdn={rec['kda_gdn_x63']:.1f}, mla={rec['mla_x9']:.1f}, "
            f"sh={rec['moe_shared_x68']:.1f}, head={rec['lm_head']:.1f})",
            flush=True,
        )
        del cache, samp
        mx.clear_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dflash", default=DEFAULT_DFLASH)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--out", default=os.path.expanduser("~/science/runs/dflash2_phase1.json"))
    ap.add_argument("--ledger-only", action="store_true")
    ap.add_argument("--q4-b8-only", action="store_true")
    ap.add_argument("--eval-ft", action="store_true")
    ap.add_argument("--ledger-ls", default="8,12,16,20,24")
    a = ap.parse_args()

    maybe_set_recommended_wired_limit()
    base = json.loads(Path(a.base).read_text())
    prompt_names = [nm for nm, _ in PROMPTS]
    prompts_data = {nm: base["prompts"][nm] for nm in prompt_names}
    ledger_ls = tuple(int(x) for x in a.ledger_ls.split(",") if x)

    t0 = time.perf_counter()
    model, _ = load(a.model)
    print(f"Loaded target (fp8={getattr(model, '_lingsante_fp8', False)}) in {time.perf_counter()-t0:.1f}s ({mx.get_active_memory()/1e9:.1f} GB)", flush=True)

    out = {"fp8": getattr(model, "_lingsante_fp8", False), "configs": [], "oracle_ubtree": None, "verify_ledger": None}

    if a.ledger_only:
        print(f"\n=== Target C=1 Verification Ledger L={ledger_ls} ===", flush=True)
        out["verify_ledger"] = profile_c1_verify_range(model, prompts_data, Ls=ledger_ls)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1))
        print(f"\nSaved {a.out}", flush=True)
        return

    if a.q4_b8_only:
        dflash4 = DFlash2DraftModel.from_target(model, a.dflash, quant_bits=4, with_q4_head=True)
        layer_ids = dflash4.target_layer_ids
        out["layer_ids"] = list(layer_ids)
        print(f"\n=== DFlash2 q4 + q4_head B=8 Benchmark ({mx.get_active_memory()/1e9:.1f} GB) ===", flush=True)

        def eval_q4(name, *, measure_phases=False):
            rows = []
            for nm in prompt_names:
                r = run_dflash_c1_prompt(
                    model, dflash4, prompts_data[nm]["ids"],
                    block_size=8, use_q4_head=True, use_selector=True,
                    max_tokens=a.tokens, layer_ids=layer_ids, measure_phases=measure_phases,
                )
                td, ld = verify_exact(r["emitted"], r["lp_arrays"], prompts_data[nm], a.tokens)
                assert td == 0 and ld == 0, f"{name}/{nm} exactness failed: td={td}, ld={ld}"
                row = {
                    "prompt": nm, "tok_s": round(r["tok_s"], 2), "E_acc": round(r["E_acc"], 3),
                    "cycle_ms": round(r["cycle_ms"], 2), "alphas": r["alphas"], "td": td, "ld": ld, "peak_gb": r["peak_gb"],
                }
                if measure_phases:
                    row.update(draft_ms=round(r["draft_ms"], 2), verify_ms=round(r["verify_ms"], 2), commit_ms=round(r["commit_ms"], 2))
                rows.append(row)
            rec = {
                "name": name, "block_size": 8, "use_q4_head": True, "use_selector": True, "max_tokens": a.tokens,
                "med_tok_s": round(statistics.median(x["tok_s"] for x in rows), 2),
                "mean_E_acc": round(statistics.mean(x["E_acc"] for x in rows), 3),
                "med_cycle_ms": round(statistics.median(x["cycle_ms"] for x in rows), 2),
                "exact": True, "rows": rows,
            }
            if measure_phases:
                rec["med_draft_ms"] = round(statistics.median(x["draft_ms"] for x in rows), 2)
                rec["med_verify_ms"] = round(statistics.median(x["verify_ms"] for x in rows), 2)
                rec["med_commit_ms"] = round(statistics.median(x["commit_ms"] for x in rows), 2)
                print(
                    f"  [{name:28s}] med={rec['med_tok_s']:6.2f} tok/s  E_acc={rec['mean_E_acc']:.3f}  "
                    f"cyc={rec['med_cycle_ms']:6.2f}ms (draft={rec['med_draft_ms']:.2f} verify={rec['med_verify_ms']:.2f} commit={rec['med_commit_ms']:.2f})  exact=True",
                    flush=True,
                )
            else:
                print(
                    f"  [{name:28s}] med={rec['med_tok_s']:6.2f} tok/s  E_acc={rec['mean_E_acc']:.3f}  "
                    f"cyc={rec['med_cycle_ms']:6.2f}ms  a0={rows[0]['alphas']}  exact=True",
                    flush=True,
                )
            out["configs"].append(rec)

        eval_q4("q4_q4head_sel_B8_async", measure_phases=False)
        eval_q4("q4_q4head_sel_B8_phases", measure_phases=True)
        del dflash4
        mx.clear_cache()
        print(f"\n=== Target C=1 Verification Ledger L={ledger_ls} ===", flush=True)
        out["verify_ledger"] = profile_c1_verify_range(model, prompts_data, Ls=ledger_ls)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1))
        print(f"\nSaved {a.out}", flush=True)
        return

    if a.eval_ft:
        dflash4 = DFlash2DraftModel.from_target(model, a.dflash, quant_bits=4, with_q4_head=True)
        layer_ids = dflash4.target_layer_ids
        out["layer_ids"] = list(layer_ids)
        out["dflash_path"] = a.dflash
        print(f"\n=== Adapted DFlash2 4-bit + q4_head Evaluation ({mx.get_active_memory()/1e9:.1f} GB) ===", flush=True)

        def eval_ft_cfg(name, *, block_size=8, max_tokens=a.tokens, measure_phases=False):
            rows = []
            for nm in prompt_names:
                r = run_dflash_c1_prompt(
                    model, dflash4, prompts_data[nm]["ids"],
                    block_size=block_size, use_q4_head=True, use_selector=True,
                    max_tokens=max_tokens, layer_ids=layer_ids, measure_phases=measure_phases,
                )
                td, ld = verify_exact(r["emitted"], r["lp_arrays"], prompts_data[nm], max_tokens)
                assert td == 0 and ld == 0, f"{name}/{nm} exactness failed: td={td}, ld={ld}"
                row = {
                    "prompt": nm, "tok_s": round(r["tok_s"], 2), "E_acc": round(r["E_acc"], 3),
                    "cycle_ms": round(r["cycle_ms"], 2), "alphas": r["alphas"], "td": td, "ld": ld, "peak_gb": r["peak_gb"],
                }
                if measure_phases:
                    row.update(draft_ms=round(r["draft_ms"], 2), verify_ms=round(r["verify_ms"], 2), commit_ms=round(r["commit_ms"], 2))
                rows.append(row)
            rec = {
                "name": name, "block_size": block_size, "use_q4_head": True, "use_selector": True, "max_tokens": max_tokens,
                "med_tok_s": round(statistics.median(x["tok_s"] for x in rows), 2),
                "mean_E_acc": round(statistics.mean(x["E_acc"] for x in rows), 3),
                "med_cycle_ms": round(statistics.median(x["cycle_ms"] for x in rows), 2),
                "exact": True, "rows": rows,
            }
            if measure_phases:
                rec["med_draft_ms"] = round(statistics.median(x["draft_ms"] for x in rows), 2)
                rec["med_verify_ms"] = round(statistics.median(x["verify_ms"] for x in rows), 2)
                rec["med_commit_ms"] = round(statistics.median(x["commit_ms"] for x in rows), 2)
                print(
                    f"  [{name:28s}] med={rec['med_tok_s']:6.2f} tok/s  E_acc={rec['mean_E_acc']:.3f}  "
                    f"cyc={rec['med_cycle_ms']:6.2f}ms (draft={rec['med_draft_ms']:.2f} verify={rec['med_verify_ms']:.2f} commit={rec['med_commit_ms']:.2f})  exact=True",
                    flush=True,
                )
            else:
                per_p = " ".join(f"{r['prompt'][:5]}={r['tok_s']:.1f}({r['E_acc']:.2f})" for r in rows)
                print(
                    f"  [{name:28s}] med={rec['med_tok_s']:6.2f} tok/s  E_acc={rec['mean_E_acc']:.3f}  "
                    f"cyc={rec['med_cycle_ms']:6.2f}ms  [{per_p}]  exact=True",
                    flush=True,
                )
            out["configs"].append(rec)
            return rec

        best_rec = None
        for bs in (8, 10, 12, 16):
            r = eval_ft_cfg(f"q4_q4head_sel_B{bs}_async", block_size=bs)
            if best_rec is None or r["med_tok_s"] > best_rec["med_tok_s"]:
                best_rec = r
        eval_ft_cfg("q4_q4head_sel_B8_phases", block_size=8, measure_phases=True)
        eval_ft_cfg(f"q4_best_B{best_rec['block_size']}_256tok", block_size=best_rec["block_size"], max_tokens=256)

        print("\n=== Oracle UBTree Potential Analysis (Adapted DFlash2 q4, B_tree = 8..32) ===", flush=True)
        out["oracle_ubtree"] = run_oracle_ubtree_analysis(model, dflash4, prompts_data, prompt_names, use_q4_head=True, max_tokens=128)
        print("  Oracle UBTree:", json.dumps(out["oracle_ubtree"]), flush=True)

        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1))
        print(f"\nSaved {a.out}", flush=True)
        return

    dflash = DFlash2DraftModel.from_target(model, a.dflash, quant_bits=None, with_q4_head=True)
    print(f"Attached DFlash2(bf16) ({mx.get_active_memory()/1e9:.1f} GB)", flush=True)

    # 1. Layer-ID sanity check on prompt 0
    s0 = run_dflash_c1_prompt(
        model, dflash, prompts_data["medical_stemi"]["ids"], block_size=8, use_q4_head=True, use_selector=True, max_tokens=48
    )
    td0, ld0 = verify_exact(s0["emitted"], s0["lp_arrays"], prompts_data["medical_stemi"], 48)
    print(f"Layer sanity (1,18,35,52,70) p0 48tok: alpha={s0['alphas']} E_acc={s0['E_acc']:.3f} tok/s={s0['tok_s']:.2f} td={td0} ld={ld0}", flush=True)
    layer_ids = dflash.target_layer_ids
    alt_sanity = None
    if s0["alphas"][0] < 0.70:
        alt = (0, 17, 34, 51, 69)
        s1 = run_dflash_c1_prompt(
            model, dflash, prompts_data["medical_stemi"]["ids"], block_size=8, use_q4_head=True, use_selector=True, max_tokens=48, layer_ids=alt
        )
        print(f"Alt layer sanity {alt} p0 48tok: alpha={s1['alphas']} E_acc={s1['E_acc']:.3f}", flush=True)
        alt_sanity = {"layer_ids": list(alt), "alphas": s1["alphas"], "E_acc": round(s1["E_acc"], 3)}
        if s1["E_acc"] > s0["E_acc"]:
            layer_ids = alt
    assert td0 == 0 and ld0 == 0, f"Sanity exactness failed td={td0}, ld={ld0}"

    out = {"layer_ids": list(layer_ids), "alt_sanity": alt_sanity, "configs": [], "oracle_ubtree": None, "verify_ledger": None}

    def eval_cfg(name, draft_mod, *, block_size=8, use_q4_head=True, use_selector=True, max_tokens=a.tokens, measure_phases=False):
        rows = []
        for nm in prompt_names:
            r = run_dflash_c1_prompt(
                model, draft_mod, prompts_data[nm]["ids"],
                block_size=block_size, use_q4_head=use_q4_head, use_selector=use_selector,
                max_tokens=max_tokens, layer_ids=layer_ids, measure_phases=measure_phases,
            )
            td, ld = verify_exact(r["emitted"], r["lp_arrays"], prompts_data[nm], max_tokens)
            assert td == 0 and ld == 0, f"{name}/{nm} exactness failed: td={td}, ld={ld}"
            row = {
                "prompt": nm,
                "tok_s": round(r["tok_s"], 2),
                "E_acc": round(r["E_acc"], 3),
                "cycle_ms": round(r["cycle_ms"], 2),
                "alphas": r["alphas"],
                "td": td,
                "ld": ld,
                "peak_gb": r["peak_gb"],
            }
            if measure_phases:
                row.update(draft_ms=round(r["draft_ms"], 2), verify_ms=round(r["verify_ms"], 2), commit_ms=round(r["commit_ms"], 2))
            rows.append(row)
        med_tps = round(statistics.median(x["tok_s"] for x in rows), 2)
        mean_e = round(statistics.mean(x["E_acc"] for x in rows), 3)
        med_cyc = round(statistics.median(x["cycle_ms"] for x in rows), 2)
        rec = {
            "name": name,
            "block_size": block_size,
            "use_q4_head": use_q4_head,
            "use_selector": use_selector,
            "max_tokens": max_tokens,
            "med_tok_s": med_tps,
            "mean_E_acc": mean_e,
            "med_cycle_ms": med_cyc,
            "exact": True,
            "rows": rows,
        }
        if measure_phases:
            rec["med_draft_ms"] = round(statistics.median(x["draft_ms"] for x in rows), 2)
            rec["med_verify_ms"] = round(statistics.median(x["verify_ms"] for x in rows), 2)
            rec["med_commit_ms"] = round(statistics.median(x["commit_ms"] for x in rows), 2)
            print(
                f"  [{name:28s}] med={med_tps:6.2f} tok/s  E_acc={mean_e:.3f}  "
                f"cyc={med_cyc:6.2f}ms (draft={rec['med_draft_ms']:.2f} verify={rec['med_verify_ms']:.2f} commit={rec['med_commit_ms']:.2f})  exact=True",
                flush=True,
            )
        else:
            print(
                f"  [{name:28s}] med={med_tps:6.2f} tok/s  E_acc={mean_e:.3f}  "
                f"cyc={med_cyc:6.2f}ms  a0={rows[0]['alphas']}  exact=True",
                flush=True,
            )
        out["configs"].append(rec)
        return rec

    print("\n=== DFlash2 Zero-Shot C=1 Benchmarks (bf16 draft) ===", flush=True)
    eval_cfg("bf16_lmhead_nosel_B8", dflash, block_size=8, use_q4_head=False, use_selector=False)
    eval_cfg("bf16_lmhead_sel_B8", dflash, block_size=8, use_q4_head=False, use_selector=True)
    eval_cfg("bf16_q4head_sel_B8_async", dflash, block_size=8, use_q4_head=True, use_selector=True)
    eval_cfg("bf16_q4head_sel_B8_phases", dflash, block_size=8, use_q4_head=True, use_selector=True, measure_phases=True)

    print("\n=== Oracle UBTree Potential Analysis (B_tree = 8..32) ===", flush=True)
    out["oracle_ubtree"] = run_oracle_ubtree_analysis(model, dflash, prompts_data, prompt_names, use_q4_head=True, max_tokens=128)
    print("  Oracle UBTree:", json.dumps(out["oracle_ubtree"]), flush=True)

    # Switch to 4-bit DFlash2
    del dflash
    mx.clear_cache()
    dflash4 = DFlash2DraftModel.from_target(model, a.dflash, quant_bits=4, with_q4_head=True)
    print(f"\n=== DFlash2 4-bit + q4_head Benchmarks ({mx.get_active_memory()/1e9:.1f} GB) ===", flush=True)
    best_rec = None
    for bs in (8, 10, 12, 16):
        r = eval_cfg(f"q4_q4head_sel_B{bs}_async", dflash4, block_size=bs, use_q4_head=True, use_selector=True)
        if best_rec is None or r["med_tok_s"] > best_rec["med_tok_s"]:
            best_rec = r
    eval_cfg("q4_q4head_sel_B8_phases", dflash4, block_size=8, use_q4_head=True, use_selector=True, measure_phases=True)
    eval_cfg(f"q4_best_B{best_rec['block_size']}_256tok", dflash4, block_size=best_rec["block_size"], use_q4_head=True, use_selector=True, max_tokens=256)

    del dflash4
    mx.clear_cache()
    print("\n=== Target C=1 Verification Ledger L=(8,12,16,20,24) ===", flush=True)
    out["verify_ledger"] = profile_c1_verify_range(model, prompts_data, Ls=(8, 12, 16, 20, 24))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"\nSaved {a.out}", flush=True)


if __name__ == "__main__":
    main()
