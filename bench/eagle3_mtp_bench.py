"""Held-out bit-exact benchmark for EAGLE-3 v2 MTP on C=1 (`bench/exact.py` prompts).

Usage:
  python bench/eagle3_mtp_bench.py [--out ~/science/runs/eagle3_mtp_exact_bench.json]
"""
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
import mlx.nn as nn  # noqa: E402
from bench.dflash2_bench import verify_exact  # noqa: E402
from bench.exact import DEFAULT_BASE, DEFAULT_MODEL, PROMPTS, prompt_ids  # noqa: E402
from bench.probe_mtp_eagle import (  # noqa: E402
    apply_bridge,
    load_bridge_and_feat_fuse,
    qmtp_step,
)
from mlx_lm import load  # noqa: E402
from mlx_lm.layers.recurrent import create_ssm_mask  # noqa: E402
from mlx_lm.models.bailing_moe_v3.dflash2 import fast_mask_helper  # noqa: E402
from mlx_lm.models.bailing_moe_v3.mtp import (  # noqa: E402
    BailingMTPLayer,
    _build_bpe_non_cjk_shortlist,
    eh_norm_concat_rows,
    logits_argmax_prob,
)
from mlx_lm.models.bailing_moe_v3.spec_engine import _logits_to_preds  # noqa: E402
from mlx_lm.models.bailing_moe_v3.suffix import SuffixMatcher  # noqa: E402
from mlx_lm.models.bailing_moe_v3.verify_decode import (  # noqa: E402
    commit_verify_cache,
    fast_verify_forward,
)
from mlx_lm.models.cache import KVCache, make_prompt_cache  # noqa: E402
from mlx_lm.utils import infer_quant_config, maybe_set_recommended_wired_limit  # noqa: E402

FEAT_LAYERS = (1, 18, 35, 52, 70)
DEFAULT_EAGLE3_DIR = os.path.expanduser("~/science/models/lingsante-mlx-mxfp4-bf16/eagle3-mtp")
DEFAULT_EAGLE3_V2_DIR = os.path.expanduser("~/science/models/lingsante-eagle3-mtp-v2")


def load_eagle3_q4d_mtp(model, stock_mtp: BailingMTPLayer, mtp_dir: Path) -> BailingMTPLayer:
    """Load trained EAGLE-3 MTP + FeedbackBridge (v1/v2) + FeatFuse and quantize dense projections to Q4."""
    args = model.args
    mtp = BailingMTPLayer(args)
    w = mx.load(str(mtp_dir / "model-mtp.safetensors"))

    def pred(p, m):
        if not hasattr(m, "to_quantized") or f"{p}.scales" not in w:
            return False
        return infer_quant_config(p, m, w)

    nn.quantize(mtp, group_size=32, bits=4, mode="mxfp4", class_predicate=pred)
    mtp.load_weights(list(w.items()), strict=True)

    bridge, feat_fuse = load_bridge_and_feat_fuse(
        mtp_dir / "feedback_bridge.safetensors", args.hidden_size, args.rms_norm_eps, n_feat_layers=len(FEAT_LAYERS)
    )

    def qpred_dense(p, m):
        if not hasattr(m, "to_quantized") or isinstance(m, nn.QuantizedLinear):
            return False
        if p == "eh_proj" or p.startswith("mlp.shared_experts."):
            return True
        if p in ("attention.q_a_proj", "attention.q_b_proj", "attention.kv_a_proj_with_mqa", "attention.dense"):
            return True
        return False

    nn.quantize(mtp, group_size=64, bits=4, mode="affine", class_predicate=qpred_dense)
    if bridge is not None:
        nn.quantize(bridge, group_size=64, bits=4, mode="affine")
    if feat_fuse is not None:
        nn.quantize(feat_fuse, group_size=64, bits=4, mode="affine")

    mtp.eval()
    eval_arrs = [mtp.parameters()]
    if bridge is not None:
        bridge.eval()
        eval_arrs.append(bridge.parameters())
    if feat_fuse is not None:
        feat_fuse.eval()
        eval_arrs.append(feat_fuse.parameters())
    mx.eval(*eval_arrs)

    mtp._embed = model.model.word_embeddings
    mtp._lm_head = model.lm_head
    mtp._q4_head = stock_mtp._q4_head
    mtp._bridge = bridge
    mtp._feat_fuse = feat_fuse
    return mtp


def mtp_kv_append(mtp: BailingMTPLayer, token_ids: mx.array, prev_hidden: mx.array, cache: KVCache) -> None:
    """Bit-exact KV-only MTP cache update (skips Q/SDPA/dense/MLP/lm_head)."""
    toks_2d = token_ids.reshape(1, -1)
    L_tok = toks_2d.shape[1]
    prev_3d = prev_hidden.reshape(1, L_tok, -1)
    embeds = mtp._embed(toks_2d)
    e_norm = mx.fast.rms_norm(embeds, mtp.enorm.weight, mtp.enorm.eps)
    h_norm = mx.fast.rms_norm(prev_3d, mtp.hnorm.weight, mtp.hnorm.eps)
    h = mtp.eh_proj(mx.concatenate([e_norm, h_norm], axis=-1))
    x_a = mx.fast.rms_norm(h, mtp.input_layernorm.weight, mtp.input_layernorm.eps)
    attn = mtp.attention
    compressed = attn.kv_a_proj_with_mqa(x_a)
    compressed, k_rope = mx.split(compressed, [attn.kv_lora_rank], axis=-1)
    kv_latent = attn.kv_a_layernorm(compressed)
    k_rope = k_rope.reshape(1, L_tok, 1, attn.qk_rope_head_dim).transpose(0, 2, 1, 3)
    offset = cache.offset if cache is not None else 0
    k_rope = attn.rope(k_rope, offset=offset)
    kv_latent = mx.expand_dims(kv_latent, axis=1)
    cache.update_and_fetch(kv_latent, k_rope)


def stock_mtp_step(mtp: BailingMTPLayer, token_ids: mx.array, prev_hidden: mx.array, cache: KVCache):
    return mtp.step_rows(token_ids, prev_hidden, cache, C=1, L=1, use_q4_head=True)


def verify_mtp_kv_append(mtp: BailingMTPLayer, label: str) -> None:
    """Assert mtp_kv_append matches mtp.prefill KV state bit-for-bit at offset=0 and offset=16."""
    toks = mx.arange(100, 120, dtype=mx.uint32)
    h = (mx.random.normal((1, 20, 4096)) * 0.2).astype(mx.bfloat16)
    mx.eval(toks, h)

    c_ref, c_fast = KVCache(), KVCache()
    mtp.prefill(toks[:16][None], h[:, :16, :], c_ref)
    mtp.prefill(toks[16:][None], h[:, 16:, :], c_ref)
    mtp_kv_append(mtp, toks[:16], h[:, :16, :], c_fast)
    mtp_kv_append(mtp, toks[16:], h[:, 16:, :], c_fast)

    k1, v1, off1 = c_ref.state
    k2, v2, off2 = c_fast.state
    assert off1 == off2 == 20, f"{label} offset mismatch: {off1} vs {off2}"
    k1, v1 = k1[..., :off1, :], v1[..., :off1, :]
    k2, v2 = k2[..., :off2, :], v2[..., :off2, :]
    mx.eval(k1, v1, k2, v2)
    dk = float(mx.max(mx.abs(k1 - k2)).item())
    dv = float(mx.max(mx.abs(v1 - v2)).item())
    assert dk == 0.0 and dv == 0.0, f"{label} mtp_kv_append mismatch: dk={dk}, dv={dv}"
    print(f"Startup self-test ({label} mtp_kv_append vs prefill, 16+4 tok): PASS (dk={dk:.1f}, dv={dv:.1f})", flush=True)


def prefill_target_and_mtp(model, mtp: BailingMTPLayer, ids: list[int], *, step: int = 512):
    """Exact B=1 target prompt prefill + KV-only shifted MTP cache population + first token y0."""
    feat_fuse = getattr(mtp, "_feat_fuse", None)
    lids = FEAT_LAYERS if feat_fuse is not None else None
    cap_set = set(lids) if lids is not None else set()
    inner = model.model
    t_cache = make_prompt_cache(model)
    mtp_cache = KVCache()

    p = mx.array(ids, dtype=mx.uint32)
    h_chunks = []
    pos = 0
    while p.size - pos > 1:
        n = min(step, p.size - 1 - pos)
        chunk = p[pos : pos + n][None]
        if lids is not None:
            h = inner.word_embeddings(chunk)
            attn_mask = (
                fast_mask_helper(inner, h, t_cache[inner.attention_index], attention=True)
                if inner.attention_index is not None
                else None
            )
            lin_mask = (
                create_ssm_mask(h, t_cache[inner.linear_index])
                if inner.linear_index is not None
                else None
            )
            captured = {}
            for li, (layer, lc) in enumerate(zip(inner.layers, t_cache)):
                mask = lin_mask if layer.is_linear else attn_mask
                h = layer(h, mask=mask, cache=lc)
                if li in cap_set:
                    captured[li] = h
            h_final = inner.norm(h)
            feats = mx.concatenate([captured[li] for li in lids], axis=-1)
            h_eff = (h_final.astype(mx.float32) + feat_fuse(feats).astype(mx.float32)).astype(mx.bfloat16)
        else:
            h_eff = inner(chunk, cache=t_cache)
        mx.eval(h_eff, [c.state for c in t_cache])
        h_chunks.append(h_eff)
        pos += n

    if h_chunks:
        h_all = h_chunks[0] if len(h_chunks) == 1 else mx.concatenate(h_chunks, axis=1)
        tokens_shifted = p[1:][None]
        s_pos = 0
        tot = tokens_shifted.shape[1]
        while s_pos < tot:
            m = min(step, tot - s_pos)
            mtp_kv_append(mtp, tokens_shifted[:, s_pos : s_pos + m], h_all[:, s_pos : s_pos + m, :], mtp_cache)
            mx.eval(mtp_cache.state)
            s_pos += m

    last_tok = p[-1:].reshape(1, 1)
    logits0, h0, rb0 = fast_verify_forward(
        model,
        last_tok,
        t_cache,
        return_hidden=True,
        return_rollback=True,
        capture_layers=lids,
    )
    commit_verify_cache(t_cache, rb0, 1)
    lp0 = logits0[0, 0, :] - mx.logsumexp(logits0[0, 0, :], axis=-1, keepdims=True)
    y0 = mx.argmax(lp0, axis=-1)
    h_eff_cur = h0[0]
    if feat_fuse is not None:
        h_eff_cur = (h_eff_cur.astype(mx.float32) + feat_fuse(rb0["captured"][0]).astype(mx.float32)).astype(mx.bfloat16)
    mx.eval(y0, lp0, h_eff_cur, *[c.state for c in t_cache], mtp_cache.state)
    return t_cache, mtp_cache, y0, lp0, h_eff_cur


def _draft_fixed(
    mtp: BailingMTPLayer,
    step_fn,
    bridge,
    cur_anchor: mx.array,
    h_eff_cur: mx.array,
    mtp_cache: KVCache,
    K_eff: int,
) -> mx.array:
    """Enqueue K_eff MTP draft steps lazily with zero host-GPU sync."""
    drafts = []
    h_in = h_eff_cur
    tok_in = cur_anchor
    for k_idx in range(K_eff):
        h_next, lg = step_fn(mtp, tok_in, h_in, mtp_cache)
        tok_in = mx.argmax(lg, axis=-1)[0]
        drafts.append(tok_in)
        if k_idx < K_eff - 1:
            h_in = apply_bridge(bridge, h_next, h_in)
    return mx.stack(drafts)


def _draft_dyn_bucket(
    mtp: BailingMTPLayer,
    step_fn,
    bridge,
    cur_anchor: mx.array,
    h_eff_cur: mx.array,
    mtp_cache: KVCache,
    K_cap: int,
    dyn_th: float,
) -> mx.array:
    """Draft in {2, 4, 6} buckets syncing only cumulative confidence scalar at k=2 and k=4."""
    drafts = []
    h_in = h_eff_cur
    tok_in = cur_anchor
    cum_p_mx = None
    for k_idx in range(K_cap):
        if k_idx > 0:
            h_in = apply_bridge(bridge, h_next, h_in)
        h_next, lg = step_fn(mtp, tok_in, h_in, mtp_cache)
        tok_in = mx.argmax(lg, axis=-1)[0]
        drafts.append(tok_in)
        if k_idx + 1 < K_cap:
            lg_f = lg[0].astype(mx.float32)
            p_k = mx.exp(mx.max(lg_f) - mx.logsumexp(lg_f))
            cum_p_mx = p_k if cum_p_mx is None else (cum_p_mx * p_k)
            if k_idx in (1, 3):
                mx.eval(cum_p_mx)
                if float(cum_p_mx.item()) < dyn_th:
                    break
    return mx.stack(drafts)


def verify_mtp_d1_unit_tests(mtp: BailingMTPLayer) -> None:
    """Verify D1 fused kernels, P0 step_varlen_rows fix (Ls=[1,3], [2,1,3]), refresh_and_step0, and router inds."""
    # 1. logits_argmax_prob vs FP32 reference across C=1..4 & V in (157184, 65536, 49152)
    mx.random.seed(123)
    for C in (1, 2, 4):
        for V in (157184, 65536, 49152):
            lg = (mx.random.normal((C, V)) * 2.5).astype(mx.bfloat16)
            mx.eval(lg)
            arg_k, p1_k = logits_argmax_prob(lg)
            lg_f = lg.astype(mx.float32)
            arg_r = mx.argmax(lg_f, axis=-1).astype(mx.uint32)
            p1_r = mx.exp(mx.max(lg_f, axis=-1) - mx.logsumexp(lg_f, axis=-1))
            mx.eval(arg_k, p1_k, arg_r, p1_r)
            assert arg_k.tolist() == arg_r.tolist(), f"logits_argmax_prob argmax mismatch C={C} V={V}"
            dp = float(mx.max(mx.abs(p1_k - p1_r)).item())
            assert dp < 2e-5, f"logits_argmax_prob p1 diff={dp} >= 2e-5 (C={C} V={V})"
    print("Unit test 1 (logits_argmax_prob vs FP32 ref, C=1..4 V=49K..157K): PASS", flush=True)

    # 2. P0 step_varlen_rows with q4_dense=True for heterogeneous Ls containing lc==1: [1, 3], [2, 1, 3], [1, 1, 1]
    pre_toks = mx.arange(100, 116, dtype=mx.uint32)
    pre_h = (mx.random.normal((1, 16, 4096)) * 0.2).astype(mx.bfloat16)
    mx.eval(pre_toks, pre_h)
    for Ls in ([1, 3], [2, 1, 3], [1, 1, 1]):
        C = len(Ls)
        B = sum(Ls)
        toks_flat = mx.arange(200, 200 + B, dtype=mx.uint32)
        hs_flat = (mx.random.normal((B, 4096)) * 0.2).astype(mx.bfloat16)
        mx.eval(toks_flat, hs_flat)

        c_var = [KVCache() for _ in range(C)]
        c_ref = [KVCache() for _ in range(C)]
        for i in range(C):
            mtp.kv_append(pre_toks[: 8 + i * 2], pre_h[:, : 8 + i * 2, :], c_var[i])
            mtp.kv_append(pre_toks[: 8 + i * 2], pre_h[:, : 8 + i * 2, :], c_ref[i])
            mx.eval(c_var[i].state, c_ref[i].state)

        h_var, lg_var = mtp.step_varlen_rows(toks_flat, hs_flat, c_var, Ls, use_q4_head=True, return_logits=True)
        h_ref_list, lg_ref_list = [], []
        pos = 0
        for i, lc in enumerate(Ls):
            if lc > 1:
                mtp.kv_append(toks_flat[pos : pos + lc - 1], hs_flat[pos : pos + lc - 1], c_ref[i])
            hi, lgi = mtp.step_rows(
                toks_flat[pos + lc - 1 : pos + lc],
                hs_flat[pos + lc - 1 : pos + lc],
                c_ref[i],
                C=1,
                L=1,
                use_q4_head=True,
                return_logits=True,
            )
            h_ref_list.append(hi[0])
            lg_ref_list.append(lgi[0])
            pos += lc
        h_ref = mx.stack(h_ref_list, axis=0)
        lg_ref = mx.stack(lg_ref_list, axis=0)
        mx.eval(h_var, lg_var, h_ref, lg_ref, *[x for c in c_var for x in c.state], *[x for c in c_ref for x in c.state])
        for i, lc in enumerate(Ls):
            assert c_var[i].offset == c_ref[i].offset == 8 + i * 2 + lc
        dh = float(mx.max(mx.abs(h_var.astype(mx.float32) - h_ref.astype(mx.float32))).item())
        a_var = mx.argmax(lg_var, axis=-1).tolist()
        a_ref = mx.argmax(lg_ref, axis=-1).tolist()
        assert dh < 0.15 and a_var == a_ref, f"step_varlen_rows Ls={Ls} mismatch dh={dh}, {a_var} vs {a_ref}"
        assert mtp._last_router_inds is not None and mtp._last_router_inds.shape == (1, 8)
    # Also verify _last_router_inds after C=3 varlen call
    c3 = [KVCache() for _ in range(3)]
    for i in range(3):
        mtp.kv_append(pre_toks[:8], pre_h[:, :8, :], c3[i])
    _, tok_ap, p1_ap = mtp.step_varlen_rows(
        mx.arange(300, 306, dtype=mx.uint32),
        (mx.random.normal((6, 4096)) * 0.2).astype(mx.bfloat16),
        c3,
        [2, 1, 3],
        use_q4_head=True,
        return_argmax_prob=True,
    )
    mx.eval(tok_ap, p1_ap, mtp._last_router_inds)
    assert mtp._last_router_inds is not None and mtp._last_router_inds.shape == (3, 8)
    print("Unit test 2 (P0 step_varlen_rows Ls=[1,3],[2,1,3],[1,1,1] + refresh_and_step0 + _last_router_inds): PASS", flush=True)

    # 3. eh_norm_concat_rows exact bit parity vs mx.fast.rms_norm + concat
    for B in (1, 2, 4, 6):
        e_in = (mx.random.normal((B, 4096)) * 0.25).astype(mx.bfloat16)
        h_in = (mx.random.normal((B, 4096)) * 0.25).astype(mx.bfloat16)
        mx.eval(e_in, h_in)
        got = eh_norm_concat_rows(e_in, h_in, mtp.enorm.weight, mtp.hnorm.weight, mtp.enorm.eps)
        ref = mx.concatenate(
            [
                mx.fast.rms_norm(e_in, mtp.enorm.weight, mtp.enorm.eps),
                mx.fast.rms_norm(h_in, mtp.hnorm.weight, mtp.hnorm.eps),
            ],
            axis=-1,
        )
        mx.eval(got, ref)
        de = float(mx.max(mx.abs(got.astype(mx.float32) - ref.astype(mx.float32))).item())
        assert de == 0.0, f"eh_norm_concat_rows B={B} diff={de}"
    print("Unit test 3 (eh_norm_concat_rows bit-exact vs mx.fast.rms_norm+concat, B=1..6): PASS", flush=True)

    # 4. 64-token sliding-window dynamic shortlist fallback test (preserve caller head mode!)
    prev_sl_ids = mtp._aux.d.get("shortlist_ids")
    prev_rescore_k = int(mtp._aux.d.get("rescore_top_k", 0))
    sl_test = mx.arange(0, 1024, dtype=mx.uint32)
    mtp.configure_head_mode(rescore_top_k=0, shortlist_ids=sl_test)
    assert mtp.reset_vocab_window([10, 20, 30]) is True
    assert mtp.update_vocab_window([5000]) is False  # OOV token -> auto fallback to full q4_head
    assert mtp.update_vocab_window([10] * 63) is False  # OOV still in 64-tok window
    assert mtp.update_vocab_window([10]) is True  # OOV aged out -> shortlist restored
    mtp.configure_head_mode(rescore_top_k=prev_rescore_k, shortlist_ids=prev_sl_ids)
    mtp.reset_vocab_window(())
    print("Unit test 4 (64-token dynamic shortlist sliding-window fallback): PASS", flush=True)


def bench_mtp_step_micro(legacy_mtp: BailingMTPLayer, d1_mtp: BailingMTPLayer, n_steps: int = 64) -> dict:
    """Measure isolated C=1 draft step latency (legacy qmtp_step vs D1 fused step_rows + refresh_and_step0)."""
    dummy_toks = mx.full((1, 128), 100, dtype=mx.uint32)
    dummy_h = (mx.random.normal((1, 128, 4096)) * 0.2).astype(mx.bfloat16)
    mx.eval(dummy_toks, dummy_h)

    def _run(fn):
        for _ in range(8):
            fn()
        mx.synchronize()
        dts = []
        for _ in range(n_steps):
            t0 = time.perf_counter()
            fn()
            dts.append((time.perf_counter() - t0) * 1e3)
        return round(statistics.median(dts), 3)

    # Legacy step + FP32 softmax
    c_leg = KVCache()
    legacy_mtp.kv_append(dummy_toks[0], dummy_h[0], c_leg)
    mx.eval(c_leg.state)
    tok = mx.array([100], dtype=mx.uint32)
    h = (mx.random.normal((1, 4096)) * 0.2).astype(mx.bfloat16)
    mx.eval(tok, h)
    def _leg_step():
        nonlocal tok, h
        hn, lg = qmtp_step(legacy_mtp, tok, h, c_leg)
        h = apply_bridge(legacy_mtp._bridge, hn, h)
        lg_f = lg[0].astype(mx.float32)
        p = mx.exp(mx.max(lg_f) - mx.logsumexp(lg_f))
        tok = mx.argmax(lg, axis=-1).astype(mx.uint32)
        mx.eval(tok, h, p)
        c_leg.trim(1)

    # D1 fused step_rows(return_argmax_prob=True)
    c_d1 = KVCache()
    d1_mtp.kv_append(dummy_toks[0], dummy_h[0], c_d1)
    mx.eval(c_d1.state)
    tok2 = mx.array([100], dtype=mx.uint32)
    h2 = (mx.random.normal((1, 4096)) * 0.2).astype(mx.bfloat16)
    mx.eval(tok2, h2)
    def _d1_step():
        nonlocal tok2, h2
        hn, tok2, p = d1_mtp.step_rows(tok2, h2, c_d1, C=1, L=1, use_q4_head=True, return_argmax_prob=True)
        h2 = d1_mtp.apply_feedback_bridge(hn, h2)
        mx.eval(tok2, h2, p)
        c_d1.trim(1)

    # Legacy refresh(2 tok) + step0 vs D1 refresh_and_step0(2 tok)
    acc_t = mx.array([101, 102], dtype=mx.uint32)
    acc_h = (mx.random.normal((2, 4096)) * 0.2).astype(mx.bfloat16)
    mx.eval(acc_t, acc_h)
    def _leg_ref_step0():
        mtp_kv_append(legacy_mtp, acc_t, acc_h, c_leg)
        hn, lg = qmtp_step(legacy_mtp, tok, h, c_leg)
        lg_f = lg[0].astype(mx.float32)
        p = mx.exp(mx.max(lg_f) - mx.logsumexp(lg_f))
        tk = mx.argmax(lg, axis=-1).astype(mx.uint32)
        mx.eval(hn, tk, p)
        c_leg.trim(3)

    def _d1_ref_step0():
        hn, tk, p = d1_mtp.refresh_and_step0(
            acc_t, acc_h, tok2, h2, c_d1, [2], use_q4_head=True, return_argmax_prob=True
        )
        mx.eval(hn, tk, p)
        c_d1.trim(3)

    return {
        "legacy_step_ms": _run(_leg_step),
        "d1_step_ms": _run(_d1_step),
        "legacy_ref2_step0_ms": _run(_leg_ref_step0),
        "d1_ref2_step0_ms": _run(_d1_ref_step0),
    }


def _draft_dyn_step_d1(
    mtp: BailingMTPLayer,
    cur_anchor: mx.array,
    h_eff_cur: mx.array,
    mtp_cache: KVCache,
    K_cap: int,
    dyn_th: float,
    p1_min: float = 0.0,
    *,
    acc_toks: mx.array | None = None,
    acc_hs: mx.array | None = None,
) -> mx.array:
    """D1 fused dyn_step using refresh_and_step0 + fused step_rows(return_argmax_prob=True)."""
    drafts = []
    h_in = h_eff_cur
    tok_in = cur_anchor.reshape(1)
    cum_p_mx = None
    for k_idx in range(K_cap):
        if k_idx == 0 and acc_toks is not None and acc_hs is not None and acc_toks.size > 0:
            h_next, tok_out, p_k = mtp.refresh_and_step0(
                acc_toks, acc_hs, tok_in, h_in, mtp_cache, [int(acc_toks.size)],
                use_q4_head=True, return_argmax_prob=True,
            )
        else:
            if k_idx > 0:
                h_in = mtp.apply_feedback_bridge(h_next, h_in)
            h_next, tok_out, p_k = mtp.step_rows(
                tok_in, h_in, mtp_cache, C=1, L=1, use_q4_head=True, return_argmax_prob=True
            )
        tok_in = tok_out
        drafts.append(tok_out[0])
        if k_idx + 1 < K_cap:
            cum_p_mx = p_k[0] if cum_p_mx is None else (cum_p_mx * p_k[0])
            mx.eval(cum_p_mx)
            p_val = float(cum_p_mx.item())
            if (k_idx == 0 and p_val < p1_min) or p_val < dyn_th:
                break
    return mx.stack(drafts)


def _draft_fixed_d1(
    mtp: BailingMTPLayer,
    cur_anchor: mx.array,
    h_eff_cur: mx.array,
    mtp_cache: KVCache,
    K_eff: int,
    *,
    acc_toks: mx.array | None = None,
    acc_hs: mx.array | None = None,
) -> mx.array:
    """D1 fused fixed-K lazy drafting with refresh_and_step0 + head_argmax_prob."""
    drafts = []
    h_in = h_eff_cur
    tok_in = cur_anchor.reshape(1)
    for k_idx in range(K_eff):
        if k_idx == 0 and acc_toks is not None and acc_hs is not None and acc_toks.size > 0:
            h_next, tok_out, _ = mtp.refresh_and_step0(
                acc_toks, acc_hs, tok_in, h_in, mtp_cache, [int(acc_toks.size)],
                use_q4_head=True, return_argmax_prob=True,
            )
        else:
            if k_idx > 0:
                h_in = mtp.apply_feedback_bridge(h_next, h_in)
            h_next, tok_out, _ = mtp.step_rows(
                tok_in, h_in, mtp_cache, C=1, L=1, use_q4_head=True, return_argmax_prob=True
            )
        tok_in = tok_out
        drafts.append(tok_out[0])
    return mx.stack(drafts)


def _draft_dyn_step(
    mtp: BailingMTPLayer,
    step_fn,
    bridge,
    cur_anchor: mx.array,
    h_eff_cur: mx.array,
    mtp_cache: KVCache,
    K_cap: int,
    dyn_th: float,
    p1_min: float = 0.0,
) -> mx.array:
    """Draft K in 1..K_cap with per-step cumulative confidence early stop (and k=1 p1_min gate)."""
    drafts = []
    h_in = h_eff_cur
    tok_in = cur_anchor
    cum_p_mx = None
    for k_idx in range(K_cap):
        if k_idx > 0:
            h_in = apply_bridge(bridge, h_next, h_in)
        h_next, lg = step_fn(mtp, tok_in, h_in, mtp_cache)
        tok_in = mx.argmax(lg, axis=-1)[0]
        drafts.append(tok_in)
        if k_idx + 1 < K_cap:
            lg_f = lg[0].astype(mx.float32)
            p_k = mx.exp(mx.max(lg_f) - mx.logsumexp(lg_f))
            cum_p_mx = p_k if cum_p_mx is None else (cum_p_mx * p_k)
            mx.eval(cum_p_mx)
            p_val = float(cum_p_mx.item())
            if (k_idx == 0 and p_val < p1_min) or p_val < dyn_th:
                break
    return mx.stack(drafts)


def run_mtp_c1_prompt(
    model,
    mtp: BailingMTPLayer,
    ids: list[int],
    max_tokens: int,
    *,
    q4_dense: bool,
    use_d1: bool = False,
    draft_mode: str = "fixed",
    fixed_K: int = 4,
    k_cap: int = 7,
    dyn_th: float = 0.25,
    p1_min: float = 0.0,
    use_suffix: bool = False,
    suffix_K: int = 6,
    profile_phases: bool = False,
) -> dict:
    """Run exact C=1 speculative decode on one prompt and return tokens, logprobs, and stats."""
    step_fn = qmtp_step if q4_dense else stock_mtp_step
    bridge = getattr(mtp, "_bridge", None)
    feat_fuse = getattr(mtp, "_feat_fuse", None)
    lids = FEAT_LAYERS if feat_fuse is not None else None
    prompt_len = len(ids)

    t_cache, mtp_cache, y0, lp0, h_eff_cur = prefill_target_and_mtp(model, mtp, ids)
    anchor0 = int(y0.item())
    emitted = [anchor0]
    lp_arrays = [lp0]
    if use_d1:
        mtp.reset_vocab_window(ids + [anchor0])
    matcher = SuffixMatcher(ids + [anchor0], min_gram=4, max_gram=8) if use_suffix else None

    cur_anchor = y0
    pending_acc_toks = None
    pending_acc_hs = None
    max_pos = suffix_K if use_suffix else (fixed_K if draft_mode == "fixed" else k_cap)
    hits = [0] * max_pos
    trials = [0] * max_pos
    accs, Ls = [], []
    n_cycles = 0
    suffix_hits = 0
    draft_ms, verify_ms, commit_ms, refresh_ms = [], [], [], []

    mx.reset_peak_memory()
    t0 = time.perf_counter()

    while len(emitted) < max_tokens:
        rem = max_tokens - len(emitted)
        if rem == 1:
            logits, rb = fast_verify_forward(
                model,
                cur_anchor.reshape(1, 1),
                t_cache,
                return_hidden=False,
                return_rollback=True,
                capture_layers=None,
            )
            preds, lp = _logits_to_preds(logits)
            commit_verify_cache(t_cache, rb, 1)
            mx.eval(preds, lp)
            emitted.append(int(preds[0, 0].item()))
            if len(lp_arrays) < 32:
                lp_arrays.append(lp[0, 0])
            accs.append(1)
            Ls.append(1)
            break

        n_cycles += 1
        base_off = mtp_cache.offset + (int(pending_acc_toks.size) if pending_acc_toks is not None else 0)
        mtp_ran_k1 = True

        td0 = time.perf_counter() if profile_phases else 0.0
        if matcher is not None:
            gl, stoks = matcher.match(min(suffix_K, rem - 1), min_gram=4)
            if gl >= 4 and len(stoks) > 0:
                draft_toks = mx.array(stoks, dtype=mx.uint32)
                mtp_ran_k1 = False
                suffix_hits += 1
        if mtp_ran_k1:
            if use_d1 and draft_mode == "dyn_step":
                draft_toks = _draft_dyn_step_d1(
                    mtp, cur_anchor, h_eff_cur, mtp_cache, min(k_cap, rem - 1), dyn_th, p1_min,
                    acc_toks=pending_acc_toks, acc_hs=pending_acc_hs,
                )
            elif use_d1 and draft_mode == "fixed":
                draft_toks = _draft_fixed_d1(
                    mtp, cur_anchor, h_eff_cur, mtp_cache, min(fixed_K, rem - 1),
                    acc_toks=pending_acc_toks, acc_hs=pending_acc_hs,
                )
            elif draft_mode == "fixed":
                draft_toks = _draft_fixed(
                    mtp, step_fn, bridge, cur_anchor, h_eff_cur, mtp_cache, min(fixed_K, rem - 1)
                )
            elif draft_mode == "dyn_bucket":
                draft_toks = _draft_dyn_bucket(
                    mtp, step_fn, bridge, cur_anchor, h_eff_cur, mtp_cache, min(k_cap, rem - 1), dyn_th
                )
            elif draft_mode == "dyn_step":
                draft_toks = _draft_dyn_step(
                    mtp, step_fn, bridge, cur_anchor, h_eff_cur, mtp_cache, min(k_cap, rem - 1), dyn_th, p1_min
                )
            else:
                raise ValueError(f"Unknown draft_mode={draft_mode}")
            pending_acc_toks = None
            pending_acc_hs = None
        K_eff = int(draft_toks.shape[0])
        if profile_phases:
            mx.eval(draft_toks, mtp_cache.state)
            draft_ms.append((time.perf_counter() - td0) * 1e3)

        inp = mx.concatenate([cur_anchor.reshape(1, 1), draft_toks[None]], axis=1)
        tv0 = time.perf_counter() if profile_phases else 0.0
        logits, hidden, rb = fast_verify_forward(
            model,
            inp,
            t_cache,
            return_hidden=True,
            return_rollback=True,
            capture_layers=lids,
        )
        preds, lp = _logits_to_preds(logits)
        to_eval = [preds, lp, draft_toks, hidden]
        if lids is not None:
            to_eval.append(rb["captured"])
        mx.eval(*to_eval)
        if profile_phases:
            verify_ms.append((time.perf_counter() - tv0) * 1e3)

        d_host = draft_toks.tolist()
        p_host = preds[0].tolist()
        n_match = 0
        for d_id, p_id in zip(d_host, p_host[:-1]):
            if d_id == p_id:
                n_match += 1
            else:
                break
        n_consumed = 1 + n_match

        for j in range(K_eff):
            if j < max_pos:
                trials[j] += 1
                if j < n_match:
                    hits[j] += 1
        accs.append(n_consumed)
        Ls.append(1 + K_eff)

        step_toks = d_host[:n_match] + [p_host[n_match]]
        emitted.extend(step_toks)
        if use_d1:
            mtp.update_vocab_window(step_toks)
        for j in range(n_consumed):
            if len(lp_arrays) < 32:
                lp_arrays.append(lp[0, j])
        if matcher is not None:
            matcher.extend(step_toks)

        if len(emitted) >= max_tokens:
            if profile_phases:
                tc0 = time.perf_counter()
                commit_verify_cache(t_cache, rb, n_consumed)
                mx.eval([c.state for c in t_cache])
                commit_ms.append((time.perf_counter() - tc0) * 1e3)
            break

        tc0 = time.perf_counter() if profile_phases else 0.0
        commit_verify_cache(t_cache, rb, n_consumed)
        if profile_phases:
            mx.eval([c.state for c in t_cache])
            commit_ms.append((time.perf_counter() - tc0) * 1e3)

        tr0 = time.perf_counter() if profile_phases else 0.0
        h_eff_seq = hidden[0, :n_consumed]
        if feat_fuse is not None:
            h_eff_seq = (
                h_eff_seq.astype(mx.float32) + feat_fuse(rb["captured"][0, :n_consumed]).astype(mx.float32)
            ).astype(mx.bfloat16)
        if mtp_ran_k1:
            trim_n = mtp_cache.offset - (base_off + 1)
            if trim_n > 0:
                mtp_cache.trim(trim_n)
            if n_match > 0:
                if use_d1 and not use_suffix:
                    # Fuse accepted KV append directly into next cycle's step-0 via refresh_and_step0
                    pending_acc_toks = inp[0, 1 : 1 + n_match]
                    pending_acc_hs = h_eff_seq[:n_match]
                else:
                    mtp_kv_append(mtp, inp[0, 1 : 1 + n_match], h_eff_seq[:n_match], mtp_cache)
        else:
            hs_sync = h_eff_cur if n_match == 0 else mx.concatenate([h_eff_cur, h_eff_seq[:n_match]], axis=0)
            mtp_kv_append(mtp, inp[0, :n_consumed], hs_sync, mtp_cache)

        expected_off = prompt_len - 2 + len(emitted)
        actual_off = mtp_cache.offset + (int(pending_acc_toks.size) if pending_acc_toks is not None else 0)
        assert actual_off == expected_off, f"mtp_cache.offset={actual_off} != expected {expected_off}"
        cur_anchor = preds[0, n_match]
        h_eff_cur = h_eff_seq[n_match : n_match + 1]
        if profile_phases:
            if pending_acc_hs is not None:
                mx.eval(cur_anchor, h_eff_cur, pending_acc_hs)
            else:
                mx.eval(cur_anchor, h_eff_cur, mtp_cache.state)
            refresh_ms.append((time.perf_counter() - tr0) * 1e3)
        else:
            if pending_acc_hs is None:
                mx.async_eval(h_eff_cur, mtp_cache.state)

    dt = time.perf_counter() - t0
    gen_n = len(emitted) - 1
    return {
        "emitted": emitted,
        "lp_arrays": lp_arrays,
        "tok_s": round(gen_n / dt, 3),
        "decode_s": round(dt, 4),
        "gen_tokens": len(emitted),
        "n_cycles": len(accs),
        "cyc_ms": round(1e3 * dt / max(1, len(accs)), 2),
        "E_acc": round(sum(accs) / max(1, len(accs)), 3),
        "mean_L": round(sum(Ls) / max(1, len(Ls)), 2),
        "hits": hits,
        "trials": trials,
        "suffix_hits": suffix_hits,
        "suffix_cycles": n_cycles,
        "draft_ms": round(statistics.mean(draft_ms), 2) if draft_ms else None,
        "verify_ms": round(statistics.mean(verify_ms), 2) if verify_ms else None,
        "commit_ms": round(statistics.mean(commit_ms), 2) if commit_ms else None,
        "refresh_ms": round(statistics.mean(refresh_ms), 2) if refresh_ms else None,
    }


def eval_config(
    model,
    mtp: BailingMTPLayer,
    tok,
    gold: dict,
    cfg: dict,
    *,
    max_tokens: int,
    profile_phases: bool = False,
) -> dict:
    """Run one config across all 4 held-out prompts and assert bit-exact parity."""
    per_prompt = {}
    tot_tok, tot_dt, tot_cyc = 0, 0.0, 0
    tot_acc_num, tot_l_num = 0, 0
    k_cap = int(cfg.get("k_cap", 7))
    max_pos = 6 if cfg["use_suffix"] else (cfg["fixed_K"] if cfg["draft_mode"] == "fixed" else k_cap)
    tot_hits = [0] * max_pos
    tot_trials = [0] * max_pos
    tot_sfx_hits, tot_sfx_cyc = 0, 0
    p_draft, p_verify, p_commit, p_refresh = [], [], [], []
    tps_list = []

    for name, text in PROMPTS:
        ids = prompt_ids(tok, text)
        res = run_mtp_c1_prompt(
            model,
            mtp,
            ids,
            max_tokens,
            q4_dense=cfg["q4_dense"],
            use_d1=cfg.get("use_d1", False),
            draft_mode=cfg["draft_mode"],
            fixed_K=cfg["fixed_K"],
            k_cap=k_cap,
            dyn_th=cfg["dyn_th"],
            p1_min=cfg.get("p1_min", 0.0),
            use_suffix=cfg["use_suffix"],
            suffix_K=6,
            profile_phases=profile_phases,
        )
        g_entry = gold["prompts"][name]
        if max_tokens == 128:
            assert len(res["emitted"]) == len(g_entry["tokens"]), f"{name}: len {len(res['emitted'])} != {len(g_entry['tokens'])}"
        td, ld = verify_exact(res["emitted"], res["lp_arrays"], g_entry, min(128, max_tokens))
        assert td == 0 and ld == 0, f"{cfg['name']} {name} ({max_tokens} tok) EXACT FAIL: td={td}, ld={ld}"

        tps_list.append(res["tok_s"])
        tot_tok += res["gen_tokens"] - 1
        tot_dt += res["decode_s"]
        tot_cyc += res["n_cycles"]
        tot_acc_num += res["gen_tokens"] - 1  # y0 was emitted pre-loop; sum(accs) == tot_tok + n_cycles_last
        tot_l_num += round(res["mean_L"] * res["n_cycles"])
        for j in range(max_pos):
            tot_hits[j] += res["hits"][j]
            tot_trials[j] += res["trials"][j]
        tot_sfx_hits += res["suffix_hits"]
        tot_sfx_cyc += res["suffix_cycles"]
        if profile_phases and res["draft_ms"] is not None:
            p_draft.append(res["draft_ms"])
            p_verify.append(res["verify_ms"])
            p_commit.append(res["commit_ms"])
            p_refresh.append(res["refresh_ms"])

        per_prompt[name] = {
            k: v
            for k, v in res.items()
            if k not in ("emitted", "lp_arrays")
        }
        per_prompt[name]["td"] = td
        per_prompt[name]["ld"] = ld

    alphas = [round(tot_hits[j] / tot_trials[j], 3) if tot_trials[j] > 0 else 0.0 for j in range(max_pos)]
    med_tps = round(statistics.median(tps_list), 2)
    agg_e_acc = round(sum(p["E_acc"] * p["n_cycles"] for p in per_prompt.values()) / max(1, tot_cyc), 3)
    agg_mean_l = round(sum(p["mean_L"] * p["n_cycles"] for p in per_prompt.values()) / max(1, tot_cyc), 2)
    agg_cyc = round(1e3 * tot_dt / max(1, tot_cyc), 2)
    sfx_rate = round(tot_sfx_hits / max(1, tot_sfx_cyc), 3) if cfg["use_suffix"] else 0.0

    out = {
        "name": cfg["name"],
        "max_tokens": max_tokens,
        "tok_s_median": med_tps,
        "tok_s_weighted": round(tot_tok / max(1e-9, tot_dt), 2),
        "E_acc": agg_e_acc,
        "mean_L": agg_mean_l,
        "cyc_ms": agg_cyc,
        "alphas": alphas,
        "suffix_hit_rate": sfx_rate,
        "td": 0,
        "ld": 0,
        "prompts": per_prompt,
    }
    if profile_phases and p_draft:
        out["phase_ms"] = {
            "draft": round(statistics.mean(p_draft), 2),
            "verify": round(statistics.mean(p_verify), 2),
            "commit": round(statistics.mean(p_commit), 2),
            "refresh": round(statistics.mean(p_refresh), 2),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--mtp-dir", "--eagle3-dir", dest="mtp_dir", default=DEFAULT_EAGLE3_DIR)
    ap.add_argument("--v2-dir", default=DEFAULT_EAGLE3_V2_DIR)
    ap.add_argument("--also-v2", action="store_true")
    ap.add_argument("--ab-d1", action="store_true", help="Run full D1 before/after + Q8/BF16 + rescore vs shortlist A/B suite")
    ap.add_argument("--dyn-th", type=float, default=0.55)
    ap.add_argument("--p1-min", type=float, default=0.0)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--out", default=os.path.expanduser("~/science/runs/eagle3_mtp_d1_bench.json"))
    a = ap.parse_args()

    maybe_set_recommended_wired_limit()
    t0 = time.perf_counter()
    model, tok = load(a.model)
    mx.eval(model.parameters())
    gold = json.loads(Path(a.base).expanduser().read_text())
    e3_dir = Path(a.mtp_dir).expanduser()

    if a.ab_d1:
        # Load legacy Q4-adapter MTP (baseline) and D1 MTPs sharing q4_head to save RAM
        legacy_mtp = load_eagle3_q4d_mtp(
            model, BailingMTPLayer.from_target(model, a.model, with_q4_head=True), e3_dir
        )
        d1_bf16 = BailingMTPLayer.from_eagle3(
            model, e3_dir, q4_head=legacy_mtp._q4_head, q4_dense=True, adapter_bits=16, rescore_top_k=0, shortlist_size=0
        )
        d1_q8 = BailingMTPLayer.from_eagle3(
            model, e3_dir, q4_head=legacy_mtp._q4_head, q4_dense=True, adapter_bits=8, rescore_top_k=0, shortlist_size=0
        )
        d1_q4 = BailingMTPLayer.from_eagle3(
            model, e3_dir, q4_head=legacy_mtp._q4_head, q4_dense=True, adapter_bits=4, rescore_top_k=0, shortlist_size=0
        )
        print(
            f"Loaded FP8 target + legacy/D1 EAGLE-3 MTPs in {time.perf_counter()-t0:.1f}s "
            f"({mx.get_active_memory()/1e9:.1f} GB active)",
            flush=True,
        )

        verify_mtp_kv_append(legacy_mtp, "legacy_q4d")
        verify_mtp_kv_append(d1_bf16, "d1_bf16")
        verify_mtp_d1_unit_tests(d1_bf16)

        micro = bench_mtp_step_micro(legacy_mtp, d1_q8)
        sl_48k = _build_bpe_non_cjk_shortlist(Path(a.model).expanduser(), 49152, model.args.vocab_size)
        sl_64k = _build_bpe_non_cjk_shortlist(Path(a.model).expanduser(), 65536, model.args.vocab_size)
        sl_80k = _build_bpe_non_cjk_shortlist(Path(a.model).expanduser(), 81920, model.args.vocab_size)
        d1_q8.configure_head_mode(rescore_top_k=0, shortlist_ids=sl_64k)
        d1_q8.reset_vocab_window([10])
        micro_sl64 = bench_mtp_step_micro(legacy_mtp, d1_q8)
        d1_q8.configure_head_mode(rescore_top_k=0, shortlist_ids=None)
        micro["d1_sl64k_step_ms"] = micro_sl64["d1_step_ms"]
        micro["d1_sl64k_ref2_step0_ms"] = micro_sl64["d1_ref2_step0_ms"]
        print(f"\n=== Isolated C=1 MTP Microbench (ms/step) ===\n  {micro}", flush=True)

        # Warmup L=2..8 shapes
        warm_ids = prompt_ids(tok, PROMPTS[0][1])
        for k_w in range(1, 8):
            run_mtp_c1_prompt(model, d1_bf16, warm_ids, 16, q4_dense=True, use_d1=True, draft_mode="fixed", fixed_K=k_w)
        run_mtp_c1_prompt(model, legacy_mtp, warm_ids, 16, q4_dense=True, use_d1=False, draft_mode="fixed", fixed_K=4)
        print("Warmup complete (L=2..8 shapes compiled).", flush=True)

        ab_variants = [
            ("legacy_q4ad_157k", legacy_mtp, False, 0, None),
            ("d1_q8ad_157k", d1_q8, True, 0, None),
            ("d1_q8ad_sl64k", d1_q8, True, 0, sl_64k),
            ("d1_q8ad_sl80k", d1_q8, True, 0, sl_80k),
            ("d1_q4ad_sl64k", d1_q4, True, 0, sl_64k),
            ("d1_q4ad_sl80k", d1_q4, True, 0, sl_80k),
            ("d1_q8ad_157k_res32", d1_q8, True, 32, None),
            ("d1_q8ad_sl64k_res32", d1_q8, True, 32, sl_64k),
        ]

        ab_results = []
        print(f"\n=== D1 Held-Out 4-Prompt Bit-Exact A/B (128 tok, dyn_step th={a.dyn_th:.2f}) ===", flush=True)
        for name, mtp_m, use_d1_f, res_k, sl_ids in ab_variants:
            if use_d1_f:
                mtp_m.configure_head_mode(rescore_top_k=res_k, shortlist_ids=sl_ids)
                # 1-prompt tiny warmup for new head kernel shape
                run_mtp_c1_prompt(model, mtp_m, warm_ids, 16, q4_dense=True, use_d1=True, draft_mode="dyn_step", k_cap=7, dyn_th=a.dyn_th)
            cfg = {
                "name": name, "mtp": mtp_m, "q4_dense": True, "use_d1": use_d1_f,
                "draft_mode": "dyn_step", "fixed_K": 7, "k_cap": 7, "dyn_th": a.dyn_th, "p1_min": a.p1_min, "use_suffix": False,
            }
            r = eval_config(model, mtp_m, tok, gold, cfg, max_tokens=128, profile_phases=False)
            rp = eval_config(model, mtp_m, tok, gold, cfg, max_tokens=128, profile_phases=True)
            r["phase_ms"] = rp.get("phase_ms")
            ab_results.append(r)
            print(
                f"  {r['name']:24s} med={r['tok_s_median']:6.2f} (w={r['tok_s_weighted']:6.2f}) | "
                f"E_acc={r['E_acc']:.3f} L={r['mean_L']:.2f} cyc={r['cyc_ms']:5.2f}ms | "
                f"phase={r['phase_ms']} td={r['td']} ld={r['ld']}",
                flush=True,
            )

        best_r = max(ab_results, key=lambda x: (x["tok_s_median"], x["tok_s_weighted"]))
        print(f"\nWINNER: {best_r['name']} med={best_r['tok_s_median']:.2f} w={best_r['tok_s_weighted']:.2f} E_acc={best_r['E_acc']:.3f} phase={best_r['phase_ms']}", flush=True)
        payload = {"model": a.model, "mtp_dir": str(e3_dir), "micro": micro, "ab_results": ab_results, "winner": best_r["name"]}
        out_path = Path(a.out).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=1))
        print(f"Saved {out_path}", flush=True)
        return

    stock_mtp = BailingMTPLayer.from_target(model, a.model, with_q4_head=True)
    e3_mtp = BailingMTPLayer.from_eagle3(model, e3_dir, q4_head=stock_mtp._q4_head, q4_dense=True)
    e2_mtp = load_eagle3_q4d_mtp(model, stock_mtp, Path(a.v2_dir).expanduser()) if a.also_v2 else None
    print(
        f"Loaded FP8 target + stock MTP + D1 EAGLE-3 Q4D MTP ({a.mtp_dir}) in {time.perf_counter()-t0:.1f}s "
        f"({mx.get_active_memory()/1e9:.1f} GB active)",
        flush=True,
    )

    verify_mtp_kv_append(stock_mtp, "stock_mtp")
    verify_mtp_kv_append(e3_mtp, "e3_d1_q4d")
    verify_mtp_d1_unit_tests(e3_mtp)
    if e2_mtp is not None:
        verify_mtp_kv_append(e2_mtp, "e3v2_q4d")

    # Warmup compile for L=2..8 verification shapes and drafters
    warm_ids = prompt_ids(tok, PROMPTS[0][1])
    for mtp_w, q4d_w, d1_w, k_w in (
        (stock_mtp, False, False, 2),
        (e3_mtp, True, True, 1),
        (e3_mtp, True, True, 2),
        (e3_mtp, True, True, 3),
        (e3_mtp, True, True, 4),
        (e3_mtp, True, True, 5),
        (e3_mtp, True, True, 6),
        (e3_mtp, True, True, 7),
    ):
        run_mtp_c1_prompt(model, mtp_w, warm_ids, 16, q4_dense=q4d_w, use_d1=d1_w, draft_mode="fixed", fixed_K=k_w)
    if e2_mtp is not None:
        run_mtp_c1_prompt(model, e2_mtp, warm_ids, 16, q4_dense=True, draft_mode="fixed", fixed_K=4)
    print("Warmup complete (L=2..8 shapes compiled).", flush=True)

    configs = [
        {"name": "stock_mtp_K2", "mtp": stock_mtp, "q4_dense": False, "use_d1": False, "draft_mode": "fixed", "fixed_K": 2, "k_cap": 2, "dyn_th": 0.0, "p1_min": 0.0, "use_suffix": False},
    ]
    if e2_mtp is not None:
        configs.extend([
            {"name": "e3v2_q4d_dyn_040", "mtp": e2_mtp, "q4_dense": True, "use_d1": False, "draft_mode": "dyn_bucket", "fixed_K": 6, "k_cap": 6, "dyn_th": 0.40, "p1_min": 0.0, "use_suffix": False},
            {"name": "e3v2_q4d_dyn_step", "mtp": e2_mtp, "q4_dense": True, "use_d1": False, "draft_mode": "dyn_step", "fixed_K": 7, "k_cap": 7, "dyn_th": a.dyn_th, "p1_min": a.p1_min, "use_suffix": False},
        ])
    configs.extend([
        {"name": "e3v4_d1_K2", "mtp": e3_mtp, "q4_dense": True, "use_d1": True, "draft_mode": "fixed", "fixed_K": 2, "k_cap": 2, "dyn_th": 0.0, "p1_min": 0.0, "use_suffix": False},
        {"name": "e3v4_d1_K4", "mtp": e3_mtp, "q4_dense": True, "use_d1": True, "draft_mode": "fixed", "fixed_K": 4, "k_cap": 4, "dyn_th": 0.0, "p1_min": 0.0, "use_suffix": False},
        {"name": "e3v4_d1_dyn_step", "mtp": e3_mtp, "q4_dense": True, "use_d1": True, "draft_mode": "dyn_step", "fixed_K": 7, "k_cap": 7, "dyn_th": a.dyn_th, "p1_min": a.p1_min, "use_suffix": False},
    ])

    results_128 = []
    print(f"\n=== Held-out 4-prompt Bit-Exact Benchmark (128 tokens, dyn_th={a.dyn_th:.2f}, p1_min={a.p1_min:.2f}) ===", flush=True)
    for cfg in configs:
        r = eval_config(model, cfg["mtp"], tok, gold, cfg, max_tokens=128)
        results_128.append(r)
        print(
            f"  {r['name']:22s} med={r['tok_s_median']:6.2f} tok/s (w={r['tok_s_weighted']:6.2f}) | "
            f"E_acc={r['E_acc']:.3f} L={r['mean_L']:.2f} cyc={r['cyc_ms']:5.2f}ms | "
            f"sfx={r['suffix_hit_rate']:.3f} alphas={r['alphas']} td={r['td']} ld={r['ld']}",
            flush=True,
        )

    best_idx = max(range(len(results_128)), key=lambda i: results_128[i]["tok_s_median"])
    best_cfg = configs[best_idx]

    run_256 = []
    print("\n=== Long-Horizon Bit-Exact Verification (256 tokens) ===", flush=True)
    r256 = eval_config(model, best_cfg["mtp"], tok, gold, best_cfg, max_tokens=256)
    run_256.append(r256)
    print(
        f"  {r256['name']:22s} @256 med={r256['tok_s_median']:6.2f} tok/s (w={r256['tok_s_weighted']:6.2f}) | "
        f"E_acc={r256['E_acc']:.3f} L={r256['mean_L']:.2f} cyc={r256['cyc_ms']:5.2f}ms | "
        f"td={r256['td']} ld={r256['ld']}",
        flush=True,
    )

    print("\n=== Phase-Timing Decomposition on Winning Config (128 tokens, profiled) ===", flush=True)
    phase_prof = eval_config(model, best_cfg["mtp"], tok, gold, best_cfg, max_tokens=128, profile_phases=True)
    print(f"  {phase_prof['name']} phase_ms={phase_prof.get('phase_ms')} | unprofiled={results_128[best_idx]['cyc_ms']}ms", flush=True)

    payload = {
        "model": a.model,
        "mtp_dir": str(Path(a.mtp_dir).expanduser()),
        "dyn_th": a.dyn_th,
        "p1_min": a.p1_min,
        "results_128": results_128,
        "results_256": run_256,
        "winning_phase_profile": phase_prof,
    }
    out_path = Path(a.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=1))
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
