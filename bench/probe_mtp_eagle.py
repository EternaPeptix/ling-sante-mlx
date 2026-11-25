"""Probe Layer-72 MTP (quantization, dynamic stopping, Beam-W tree) & EAGLE-3 no-drift ceiling on 20 val_seqs."""
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
import mlx.nn as nn  # noqa: E402
from mlx_lm.layers.common import swiglu  # noqa: E402
from mlx_lm.models.bailing_moe_v3.config import ModelArgs  # noqa: E402
from mlx_lm.models.bailing_moe_v3.mtp import BailingMTPLayer  # noqa: E402
from mlx_lm.models.bailing_moe_v3.rows.moe import (  # noqa: E402
    moe_down_combine_rows,
    moe_gate_up_swiglu_rows,
)
from mlx_lm.models.bailing_moe_v3.rows.router import gate_rows  # noqa: E402
from mlx_lm.models.cache import KVCache, make_prompt_cache  # noqa: E402
from mlx_lm.utils import infer_quant_config, load_model, maybe_set_recommended_wired_limit  # noqa: E402

# Verified Phase 2B C=1 target verify e2e latencies (ms) from task-2762
_VERIFY_ANCHORS = [(1, 39.1), (3, 49.0), (5, 57.2), (8, 70.3), (9, 86.3), (10, 90.0), (12, 99.6), (16, 121.0)]


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
        feats_mx = st[f"{k}.feats"]
        mx.eval(ids_mx, feats_mx)
        val_seqs.append({
            "key": k,
            "idx": entry["idx"],
            "domain": entry["domain"],
            "prompt_len": int(entry["prompt_len"]),
            "total_len": int(entry["total_len"]),
            "ids": ids_mx.tolist(),
            "ids_mx": ids_mx,
            "feats": feats_mx,
        })
    return val_seqs


def sample_anchors(plen: int, T: int, k_max: int = 15, n_blk: int = 5) -> list[int]:
    lo = max(2, plen)
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


def capture_val_hfinal(model_dir: Path, val_seqs: list[dict], out_path: Path):
    if out_path.exists():
        print(f"Found existing {out_path}; skipping Phase 1 target capture.", flush=True)
        return
    t0 = time.perf_counter()
    model, _ = load_model(model_dir)
    mx.eval(model.parameters())
    print(f"Loaded target (fp8={getattr(model, '_lingsante_fp8', False)}) in {time.perf_counter()-t0:.1f}s ({mx.get_active_memory()/1e9:.1f} GB)", flush=True)

    tensors = {}
    tot_pos, tot_match = 0, 0
    for seq in val_seqs:
        ids_mx = seq["ids_mx"]
        ids = seq["ids"]
        plen = seq["prompt_len"]
        T = seq["total_len"]
        cache = make_prompt_cache(model)
        h_chunks = []
        pos = 0
        while pos < T:
            n = min(512, T - pos)
            hc = model.model(ids_mx[pos : pos + n][None], cache=cache)
            mx.eval(hc, [c.state for c in cache])
            h_chunks.append(hc[0].astype(mx.bfloat16))
            pos += n
        hf = mx.concatenate(h_chunks, axis=0) if len(h_chunks) > 1 else h_chunks[0]
        lg = model.lm_head(hf[None])[0]
        preds = mx.argmax(lg[plen - 1 : T - 1], axis=-1)
        mx.eval(hf, preds)
        p_l = preds.tolist()
        gold = ids[plen:T]
        hits = sum(1 for a, b in zip(p_l, gold) if a == b)
        tot_pos += len(gold)
        tot_match += hits
        tensors[seq["key"]] = hf
        del cache
        mx.clear_cache()

    rate = tot_match / max(1, tot_pos)
    print(f"Captured {len(tensors)} val h_final tensors: top1 match={tot_match}/{tot_pos} ({rate:.4f})", flush=True)
    assert rate >= 0.95, f"h_final alignment failed: {rate:.4f} < 0.95"
    mx.save_safetensors(str(out_path), tensors)
    print(f"Saved {out_path} ({out_path.stat().st_size/1e6:.1f} MB)", flush=True)
    del model
    mx.clear_cache()


class FeedbackBridge(nn.Module):
    """Residual feedback adapter mapping MTP hidden states onto target h_final."""

    def __init__(self, hidden_size: int = 4096, rank: int = 1024, eps: float = 1e-6):
        super().__init__()
        self._is_v2 = False
        self.norm = nn.RMSNorm(hidden_size, eps=eps)
        self.up = nn.Linear(hidden_size, rank, bias=False)
        self.down = nn.Linear(rank, hidden_size, bias=False)

    def __call__(self, h: mx.array) -> mx.array:
        return h + self.down(nn.silu(self.up(self.norm(h))))


class FeedbackBridgeV2(nn.Module):
    """Dual-input SwiGLU residual bridge combining MTP h_out with lookahead h_in."""

    def __init__(self, hidden_size: int = 4096, rank: int = 1024, eps: float = 1e-6):
        super().__init__()
        self._is_v2 = True
        self.norm_out = nn.RMSNorm(hidden_size, eps=eps)
        self.norm_in = nn.RMSNorm(hidden_size, eps=eps)
        self.gate = nn.Linear(hidden_size * 2, rank, bias=False)
        self.up = nn.Linear(hidden_size * 2, rank, bias=False)
        self.down = nn.Linear(rank, hidden_size, bias=False)

    def __call__(self, h_out: mx.array, h_in: mx.array) -> mx.array:
        xo = h_out.astype(mx.float32)
        xi = h_in.astype(mx.float32)
        cat = mx.concatenate([self.norm_out(xo), self.norm_in(xi)], axis=-1).astype(h_out.dtype)
        return h_out + self.down(nn.silu(self.gate(cat)) * self.up(cat))


def apply_bridge(bridge, h_out: mx.array, h_in: mx.array) -> mx.array:
    """Apply FeedbackBridge or dual-input FeedbackBridgeV2 uniformly."""
    if bridge is None:
        return h_out
    if getattr(bridge, "_is_v2", False):
        return bridge(h_out, h_in)
    return bridge(h_out)


class FeatFuse(nn.Module):
    """Residual projection from concatenated multi-layer target features onto hidden_size."""

    def __init__(self, in_dim: int = 20480, rank: int = 256, out_dim: int = 4096, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.RMSNorm(in_dim, eps=eps)
        self.up = nn.Linear(in_dim, rank, bias=False)
        self.down = nn.Linear(rank, out_dim, bias=False)

    def __call__(self, feats: mx.array) -> mx.array:
        return self.down(nn.silu(self.up(self.norm(feats))))


def load_bridge_and_feat_fuse(fb_path: Path, hidden_size: int, eps: float, n_feat_layers: int = 5):
    """Auto-detect & load FeedbackBridge (v1 or v2) and optional FeatFuse from safetensors."""
    bridge, feat_fuse = None, None
    if not fb_path.exists():
        return bridge, feat_fuse
    fb_w = mx.load(str(fb_path))
    fb_items = [(k.removeprefix("feedback_bridge."), v.astype(mx.bfloat16)) for k, v in fb_w.items() if k.startswith("feedback_bridge.")]
    ff_items = [(k.removeprefix("feat_fuse."), v.astype(mx.bfloat16)) for k, v in fb_w.items() if k.startswith("feat_fuse.")]
    if fb_items:
        if "feedback_bridge.gate.weight" in fb_w:
            rank = int(fb_w["feedback_bridge.gate.weight"].shape[0])
            bridge = FeedbackBridgeV2(hidden_size, rank=rank, eps=eps)
        else:
            rank = int(fb_w["feedback_bridge.up.weight"].shape[0])
            bridge = FeedbackBridge(hidden_size, rank=rank, eps=eps)
        bridge.load_weights(fb_items, strict=True)
    if ff_items:
        ff_rank = int(fb_w["feat_fuse.up.weight"].shape[0])
        feat_fuse = FeatFuse(hidden_size * n_feat_layers, rank=ff_rank, out_dim=hidden_size, eps=eps)
        feat_fuse.load_weights(ff_items, strict=True)
    return bridge, feat_fuse


def sim_dyn_step_k(probs: list[float], dyn_th: float, p1_min: float = 0.0, k_cap: int = 7) -> int:
    """Exact offline simulator matching _draft_dyn_step stopping semantics for K in 1..k_cap."""
    cum = 1.0
    kb = 1
    for k_idx in range(min(k_cap, len(probs))):
        kb = k_idx + 1
        if k_idx + 1 < k_cap:
            p_k = probs[k_idx]
            cum *= p_k
            if (k_idx == 0 and p_k < p1_min) or cum < dyn_th:
                break
    return kb


def load_standalone_mtp(model_dir: Path, mtp_dir: Path, data_dir: Path, *, q4_dense: bool = False):
    cfg = json.loads((model_dir / "config.json").read_text())
    args = ModelArgs.from_dict(cfg)
    mtp = BailingMTPLayer(args)
    w = mx.load(str(mtp_dir / "model-mtp.safetensors"))

    def pred(p, m):
        if not hasattr(m, "to_quantized") or f"{p}.scales" not in w:
            return False
        return infer_quant_config(p, m, w)

    nn.quantize(mtp, group_size=32, bits=4, mode="mxfp4", class_predicate=pred)
    mtp.load_weights(list(w.items()), strict=True)

    sio = mx.load(str(data_dir / "shared_io.safetensors"))
    emb = nn.Embedding(args.vocab_size, args.hidden_size)
    emb.weight = sio["word_embeddings.weight"]
    head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
    head.weight = sio["lm_head.weight"]
    q4h = nn.QuantizedLinear.from_linear(head, group_size=64, bits=4, mode="affine")

    bridge, feat_fuse = load_bridge_and_feat_fuse(mtp_dir / "feedback_bridge.safetensors", args.hidden_size, args.rms_norm_eps)

    if q4_dense:
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
    emb.eval()
    q4h.eval()
    eval_arrs = [mtp.parameters(), emb.parameters(), head.parameters(), q4h.parameters()]
    if bridge is not None:
        bridge.eval()
        eval_arrs.append(bridge.parameters())
    if feat_fuse is not None:
        feat_fuse.eval()
        eval_arrs.append(feat_fuse.parameters())
    mx.eval(*eval_arrs)
    mtp._embed = emb
    mtp._lm_head = head
    mtp._q4_head = q4h
    mtp._bridge = bridge
    mtp._feat_fuse = feat_fuse
    return mtp


def qmtp_step(qmtp: BailingMTPLayer, token_ids: mx.array, prev_hidden: mx.array, cache: KVCache):
    """Forward for dense-Q4 MTP using QuantizedLinear modules + MXFP4 MoE row kernels."""
    toks = token_ids.reshape(-1)
    prev_f = prev_hidden.reshape(toks.size, -1)
    embeds = qmtp._embed(toks)
    e_norm = mx.fast.rms_norm(embeds, qmtp.enorm.weight, qmtp.enorm.eps)
    h_norm = mx.fast.rms_norm(prev_f, qmtp.hnorm.weight, qmtp.hnorm.eps)
    h = qmtp.eh_proj(mx.concatenate([e_norm, h_norm], axis=-1))[:, None, :]
    h_post = h + qmtp.attention(
        mx.fast.rms_norm(h, qmtp.input_layernorm.weight, qmtp.input_layernorm.eps),
        mask=None,
        cache=cache,
    )
    h_post_2d = h_post[:, 0, :]
    mlp_in = mx.fast.rms_norm(h_post_2d, qmtp.post_attention_layernorm.weight, qmtp.post_attention_layernorm.eps)
    gate = qmtp.mlp.gate
    inds, wts = gate_rows(
        mlp_in,
        gate.weight,
        gate.expert_bias,
        top_k=gate.top_k,
        n_group=gate.n_group,
        topk_group=gate.topk_group,
        routed_scaling_factor=gate.routed_scaling_factor,
    )
    se = qmtp.mlp.shared_experts
    sh = se.down_proj(swiglu(se.gate_proj(mlp_in), se.up_proj(mlp_in)))
    sw = qmtp.mlp.switch_mlp
    h_mid = moe_gate_up_swiglu_rows(mlp_in, sw.gate_proj.weight, sw.gate_proj.scales, sw.up_proj.weight, sw.up_proj.scales, inds)
    h_mlp = moe_down_combine_rows(h_mid, sw.down_proj.weight, sw.down_proj.scales, inds, wts, shared=sh, residual=h_post_2d)
    h_out = mx.fast.rms_norm(h_mlp, qmtp.final_layernorm.weight, qmtp.final_layernorm.eps)
    return h_out, qmtp._q4_head(h_out)


def clone_kvcache(mc: KVCache) -> KVCache:
    cp = KVCache()
    if mc.keys is not None and mc.offset > 0:
        cp.keys = mx.contiguous(mc.keys[..., : mc.offset, :])
        cp.values = mx.contiguous(mc.values[..., : mc.offset, :])
        cp.offset = mc.offset
    return cp


def measure_mtp_latency(mtp: BailingMTPLayer, qmtp: BailingMTPLayer, prefill_len: int = 128, n_steps: int = 64) -> dict:
    res = {"stock": {}, "q4_dense": {}}
    dummy_toks = mx.full((1, prefill_len), 100, dtype=mx.uint32)
    dummy_h = (mx.random.normal((1, prefill_len, 4096)) * 0.2).astype(mx.bfloat16)
    mx.eval(dummy_toks, dummy_h)

    for B in (1, 2, 4):
        # Stock MTP via step_rows
        caches = [KVCache() for _ in range(B)]
        for c in caches:
            o = mtp.prefill(dummy_toks, dummy_h, c)
            mx.eval(o, c.state)
        c_arg = caches[0] if B == 1 else caches
        tok = mx.full((B,), 100, dtype=mx.uint32)
        h = (mx.random.normal((B, 4096)) * 0.2).astype(mx.bfloat16)
        mx.eval(tok, h)
        for _ in range(8):
            h_in_b = h
            h, lg = mtp.step_rows(tok, h_in_b, c_arg, C=B, L=1, use_q4_head=True)
            h = apply_bridge(mtp._bridge, h, h_in_b)
            tok = mx.argmax(lg, axis=-1).astype(mx.uint32)
            mx.eval(tok, h)
        mx.synchronize()
        dts = []
        for _ in range(n_steps):
            t0 = time.perf_counter()
            h_in_b = h
            h, lg = mtp.step_rows(tok, h_in_b, c_arg, C=B, L=1, use_q4_head=True)
            h = apply_bridge(mtp._bridge, h, h_in_b)
            tok = mx.argmax(lg, axis=-1).astype(mx.uint32)
            mx.eval(tok, h)
            dts.append((time.perf_counter() - t0) * 1e3)
        res["stock"][f"B={B}"] = round(statistics.median(dts), 3)

        # Q4-dense MTP
        qcache = KVCache()
        o = qmtp.prefill(mx.repeat(dummy_toks, B, axis=0), mx.repeat(dummy_h, B, axis=0), qcache)
        mx.eval(o, qcache.state)
        tok = mx.full((B,), 100, dtype=mx.uint32)
        h = (mx.random.normal((B, 4096)) * 0.2).astype(mx.bfloat16)
        mx.eval(tok, h)
        for _ in range(8):
            h_in_b = h
            h, lg = qmtp_step(qmtp, tok, h_in_b, qcache)
            h = apply_bridge(qmtp._bridge, h, h_in_b)
            tok = mx.argmax(lg, axis=-1).astype(mx.uint32)
            mx.eval(tok, h)
        mx.synchronize()
        dts_q = []
        for _ in range(n_steps):
            t0 = time.perf_counter()
            h_in_b = h
            h, lg = qmtp_step(qmtp, tok, h_in_b, qcache)
            h = apply_bridge(qmtp._bridge, h, h_in_b)
            tok = mx.argmax(lg, axis=-1).astype(mx.uint32)
            mx.eval(tok, h)
            dts_q.append((time.perf_counter() - t0) * 1e3)
        res["q4_dense"][f"B={B}"] = round(statistics.median(dts_q), 3)
    return res


def eval_tf_alignment(mtp: BailingMTPLayer, val_seqs: list[dict], hfinals: dict) -> tuple[dict, dict]:
    n_tot, n_med = 0, 0
    h1_tot, h2_tot, h4_tot = 0, 0, 0
    h1_med, h2_med, h4_med = 0, 0, 0
    ks = (2, 3, 4, 6, 8, 12, 15)
    e_sum = {k: 0 for k in ks}
    e_cnt = {k: 0 for k in ks}
    e_med_sum = {k: 0 for k in ks}
    e_med_cnt = {k: 0 for k in ks}
    tf_seq_info = {}

    for seq in val_seqs:
        T = seq["total_len"]
        plen = seq["prompt_len"]
        is_med = seq["domain"] == "med"
        hf = hfinals[seq["key"]]
        if getattr(mtp, "_feat_fuse", None) is not None:
            hf = (hf.astype(mx.float32) + mtp._feat_fuse(seq["feats"].astype(mx.bfloat16)).astype(mx.float32)).astype(mx.bfloat16)
        ids_mx = seq["ids_mx"]
        h_out = mtp.prefill(ids_mx[1:T][None], hf[: T - 1][None], KVCache())[0]  # (T-1, 4096)
        # h_out[t] pairs (ids[t+1], hf[t]) and predicts ids[t+2]; predict all t in [0..T-3]
        lg_all = mtp._mtp_head(h_out[: T - 2], use_q4_head=True).astype(mx.float32)  # predicts ids[2..T-1]
        lse = mx.logsumexp(lg_all, axis=-1, keepdims=True)
        idx4 = mx.argpartition(-lg_all, kth=3, axis=-1)[:, :4]
        val4 = mx.take_along_axis(lg_all - lse, idx4, axis=-1)
        ord4 = mx.argsort(-val4, axis=-1)
        t4_ids = mx.take_along_axis(idx4, ord4, axis=-1)
        t4_lps = mx.take_along_axis(val4, ord4, axis=-1)
        mx.eval(t4_ids, t4_lps)
        t4_ids_l = t4_ids.tolist()
        t4_lps_l = t4_lps.tolist()
        tf_seq_info[seq["key"]] = (t4_ids_l, t4_lps_l)

        # Compare generated slice: t from plen-1 .. T-3 -> ids[plen+1 .. T-1]
        sl_start = plen - 1
        tgts = seq["ids"][plen + 1 : T]
        hits1_flags = []
        for idx_i, tgt in enumerate(tgts):
            row4 = t4_ids_l[sl_start + idx_i]
            hit1 = row4[0] == tgt
            hits1_flags.append(hit1)
            n_tot += 1
            if hit1:
                h1_tot += 1
            if tgt in row4[:2]:
                h2_tot += 1
            if tgt in row4:
                h4_tot += 1
            if is_med:
                n_med += 1
                if hit1:
                    h1_med += 1
                if tgt in row4[:2]:
                    h2_med += 1
                if tgt in row4:
                    h4_med += 1

        N = len(hits1_flags)
        for k in ks:
            for i in range(0, max(0, N - k + 1)):
                acc = 1
                for j in range(k):
                    if hits1_flags[i + j]:
                        acc += 1
                    else:
                        break
                e_sum[k] += acc
                e_cnt[k] += 1
                if is_med:
                    e_med_sum[k] += acc
                    e_med_cnt[k] += 1

    metrics = {
        "tokens": n_tot,
        "med_tokens": n_med,
        "p1_tf": round(h1_tot / max(1, n_tot), 4),
        "p1_tf_med": round(h1_med / max(1, n_med), 4),
        "top2_tf": round(h2_tot / max(1, n_tot), 4),
        "top2_tf_med": round(h2_med / max(1, n_med), 4),
        "top4_tf": round(h4_tot / max(1, n_tot), 4),
        "top4_tf_med": round(h4_med / max(1, n_med), 4),
        "E_acc_nodrift": {str(k): round(e_sum[k] / max(1, e_cnt[k]), 3) for k in ks},
        "E_acc_nodrift_med": {str(k): round(e_med_sum[k] / max(1, e_med_cnt[k]), 3) for k in ks},
    }
    return metrics, tf_seq_info


def _beam_tree_for_anchor(mtp: BailingMTPLayer, base_mc: KVCache, tok_a: mx.array, h_am1: mx.array, gold: list[int], width: int, max_depth: int = 6):
    """Expand width-`W` beam tree up to `max_depth` and score budgets B_tree in (4,6,8,10,12,16)."""
    mc0 = clone_kvcache(base_mc)
    h1, lg1 = mtp.step_rows(tok_a, h_am1, mc0, C=1, L=1, use_q4_head=True)
    h1 = apply_bridge(mtp._bridge, h1, h_am1)
    lp1 = (lg1.astype(mx.float32) - mx.logsumexp(lg1.astype(mx.float32), axis=-1, keepdims=True)).squeeze(0)
    idx_w = mx.argpartition(-lp1, kth=width - 1)[:width]
    val_w = lp1[idx_w]
    ord_w = mx.argsort(-val_w)
    top_toks = idx_w[ord_w]
    top_lps = val_w[ord_w]
    mx.eval(h1, top_toks, top_lps, mc0.state)

    nodes = []  # (cum_lp, depth, path_tuple, h_arr, mc_cache)
    frontier = []
    for tok, lp in zip(top_toks.tolist(), top_lps.tolist()):
        nd = (float(lp), 1, (int(tok),), h1[0:1], mc0)
        nodes.append((float(lp), 1, (int(tok),)))
        frontier.append(nd)

    for d in range(2, max_depth + 1):
        frontier.sort(key=lambda x: -x[0])
        active = frontier[:width]
        next_frontier = []
        # Step each active beam node individually (single-layer KVCache each)
        for cum_lp, _, path, h_u, mc_u in active:
            mc_c = clone_kvcache(mc_u)
            tok_in = mx.array([path[-1]], dtype=mx.uint32)
            h_nxt, lg_nxt = mtp.step_rows(tok_in, h_u, mc_c, C=1, L=1, use_q4_head=True)
            h_nxt = apply_bridge(mtp._bridge, h_nxt, h_u)
            lp_n = (lg_nxt.astype(mx.float32) - mx.logsumexp(lg_nxt.astype(mx.float32), axis=-1, keepdims=True)).squeeze(0)
            iw = mx.argpartition(-lp_n, kth=width - 1)[:width]
            vw = lp_n[iw]
            ow = mx.argsort(-vw)
            tt = iw[ow]
            tl = vw[ow]
            mx.eval(h_nxt, tt, tl, mc_c.state)
            for tok, lp in zip(tt.tolist(), tl.tolist()):
                clp = cum_lp + float(lp)
                npath = path + (int(tok),)
                nodes.append((clp, d, npath))
                next_frontier.append((clp, d, npath, h_nxt[0:1], mc_c))
        frontier = next_frontier

    # Sort all nodes (depth >= 1) by (-cum_lp, depth); monotonic cum_lp ensures prefix closure
    nodes.sort(key=lambda x: (-x[0], x[1]))
    res = {}
    for B_tree in (4, 6, 8, 10, 12, 16):
        chosen = nodes[: B_tree - 1]
        best_acc = 0
        max_d = 1
        for _, d, path in chosen:
            if d > max_d:
                max_d = d
            m = 0
            for j, tok in enumerate(path):
                if j < len(gold) and tok == gold[j]:
                    m += 1
                else:
                    break
            if m > best_acc:
                best_acc = m
        res[B_tree] = (1 + best_acc, max_d)
    return res


def _eagle3_tree_bounds(t4_ids_l: list[list[int]], t4_lps_l: list[list[float]], a: int, gold: list[int], p1_logp: float):
    """Simulate (1) realistic greedy best-first tree with synthetic p1 continuation on misses & (2) optimistic top-4 node-count bound."""
    # a predicts ids[a+1], which lives at index a-1 of t4_ids_l (since t4[t] predicts ids[t+2])
    budgets = (4, 6, 8, 10, 12, 16)
    max_pos = min(15, len(gold))
    # (1) Realistic best-first tree (heap of (-cum_lp, depth, is_gold, path_gold_ok, wrong_flag))
    pq = []
    pos0 = a - 1
    row0_ids = t4_ids_l[pos0]
    row0_lps = t4_lps_l[pos0]
    for c in range(4):
        is_g = row0_ids[c] == gold[0]
        heapq.heappush(pq, (-row0_lps[c], 1, is_g, 1 if is_g else 0))

    nodes_taken = []
    while pq and len(nodes_taken) < 15:
        neg_lp, d, is_g, g_len = heapq.heappop(pq)
        nodes_taken.append((d, g_len))
        cum_lp = -neg_lp
        if d >= max_pos:
            continue
        if is_g:
            pos_d = pos0 + d
            if pos_d < len(t4_ids_l):
                r_ids = t4_ids_l[pos_d]
                r_lps = t4_lps_l[pos_d]
                for c in range(4):
                    cg = r_ids[c] == gold[d]
                    heapq.heappush(pq, (-(cum_lp + r_lps[c]), d + 1, cg, d + 1 if cg else g_len))
        else:
            # Synthetic continuation along wrong branch (top child decays at p1_logp, second at log(1-exp(p1_logp)))
            heapq.heappush(pq, (-(cum_lp + p1_logp), d + 1, False, g_len))

    real_res = {}
    for B_tree in budgets:
        sub = nodes_taken[: B_tree - 1]
        best_g = max((g for _, g in sub), default=0)
        max_d = max((d for d, _ in sub), default=1)
        real_res[B_tree] = (1 + best_g, max_d)

    # (2) Optimistic top-4 bound: to reach depth m along gold, count rank+1 nodes per step (assuming misses spend rank children)
    opt_res = {}
    for B_tree in budgets:
        used = 0
        best_m = 0
        for d in range(max_pos):
            row = t4_ids_l[pos0 + d]
            if gold[d] in row:
                rank = row.index(gold[d])
                cost = 1 + rank  # gold + higher-prob siblings at depth d
                if used + cost <= B_tree - 1:
                    used += cost
                    best_m = d + 1
                else:
                    break
            else:
                break
        opt_res[B_tree] = (1 + best_m, max(1, best_m))
    return real_res, opt_res


def _step_any(mtp_mod: BailingMTPLayer, tok: mx.array, h_in: mx.array, cache: KVCache, *, q4_dense: bool):
    h_out, lg = qmtp_step(mtp_mod, tok, h_in, cache) if q4_dense else mtp_mod.step_rows(tok, h_in, cache, C=1, L=1, use_q4_head=True)
    return apply_bridge(mtp_mod._bridge, h_out, h_in), lg


def _unroll_anchor(mtp_mod: BailingMTPLayer, mc_base: KVCache, tok_start: mx.array, h_start: mx.array, gold8: list[int], is_med: bool, *, q4_dense: bool) -> dict:
    mc_u = clone_kvcache(mc_base)
    h_p = h_start
    tok_p = tok_start
    drafts, probs = [], []
    for _ in range(8):
        h_p, lg = _step_any(mtp_mod, tok_p, h_p, mc_u, q4_dense=q4_dense)
        lg_f = lg.astype(mx.float32)
        tok_p = mx.argmax(lg_f, axis=-1).astype(mx.uint32)
        ml = mx.max(lg_f, axis=-1, keepdims=True)
        pmx = mx.exp(mx.max(lg_f - ml, axis=-1)) / mx.sum(mx.exp(lg_f - ml), axis=-1)
        mx.eval(tok_p, h_p, pmx)
        drafts.append(int(tok_p.item()))
        probs.append(float(pmx.item()))
    pos_ok = [d_tok == g_tok for d_tok, g_tok in zip(drafts, gold8)]
    mlen = 0
    for fl in pos_ok:
        if fl:
            mlen += 1
        else:
            break
    return {"is_med": is_med, "probs": probs, "match_len": mlen, "pos_ok": pos_ok}


def _score_dyn_grid(recs: list[dict], step_ms: float) -> tuple[dict, dict, dict]:
    """Sweep uniform th in (0.25..0.60) and 2-param (p1_min, th) for K in 1..7."""
    n = len(recs)
    uniform_res = {}
    for th in (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
        ls, accs, cycs = [], [], []
        for r in recs:
            kb = sim_dyn_step_k(r["probs"], th, p1_min=0.0, k_cap=7)
            ls.append(kb + 1)
            accs.append(min(r["match_len"], kb) + 1)
            cycs.append(verify_cost_ms(kb + 1) + kb * step_ms)
        uniform_res[f"{th:.2f}"] = {
            "p1_min": 0.0,
            "th": th,
            "mean_L": round(sum(ls) / n, 2),
            "E_acc": round(sum(accs) / n, 3),
            "mean_cyc_ms": round(sum(cycs) / n, 2),
            "proj_tok_s": round(1e3 * sum(accs) / sum(cycs), 2),
        }

    two_param_res = {}
    for p1_min in (0.35, 0.45, 0.55):
        for th in (0.35, 0.40, 0.45, 0.50, 0.55):
            ls, accs, cycs = [], [], []
            for r in recs:
                kb = sim_dyn_step_k(r["probs"], th, p1_min=p1_min, k_cap=7)
                ls.append(kb + 1)
                accs.append(min(r["match_len"], kb) + 1)
                cycs.append(verify_cost_ms(kb + 1) + kb * step_ms)
            key = f"p1={p1_min:.2f},th={th:.2f}"
            two_param_res[key] = {
                "p1_min": p1_min,
                "th": th,
                "mean_L": round(sum(ls) / n, 2),
                "E_acc": round(sum(accs) / n, 3),
                "mean_cyc_ms": round(sum(cycs) / n, 2),
                "proj_tok_s": round(1e3 * sum(accs) / sum(cycs), 2),
            }

    all_cands = list(uniform_res.values()) + list(two_param_res.values())
    best = max(all_cands, key=lambda d: (d["proj_tok_s"], d["E_acc"], -d["mean_L"]))
    return uniform_res, two_param_res, best


def run_probes(mtp: BailingMTPLayer, qmtp: BailingMTPLayer, val_seqs: list[dict], hfinals: dict) -> dict:
    print("\n=== Probe 2.1: MTP Single/Batched Latency (ms/step) ===", flush=True)
    lat = measure_mtp_latency(mtp, qmtp)
    print(f"  Stock MTP (q4_head):  B=1 {lat['stock']['B=1']:.3f}ms | B=2 {lat['stock']['B=2']:.3f}ms | B=4 {lat['stock']['B=4']:.3f}ms", flush=True)
    print(f"  Q4-Dense MTP:         B=1 {lat['q4_dense']['B=1']:.3f}ms | B=2 {lat['q4_dense']['B=2']:.3f}ms | B=4 {lat['q4_dense']['B=4']:.3f}ms", flush=True)

    print("\n=== Probe 2.2a: Teacher-Forced No-Drift Alignment (all 20 val_seqs) ===", flush=True)
    tf_stock, tf_seq_info = eval_tf_alignment(mtp, val_seqs, hfinals)
    print(
        f"  Stock MTP TF: p1={tf_stock['p1_tf']:.4f} (med={tf_stock['p1_tf_med']:.4f}) "
        f"top2={tf_stock['top2_tf']:.4f} top4={tf_stock['top4_tf']:.4f} E_acc_nodrift={tf_stock['E_acc_nodrift']}",
        flush=True,
    )
    assert tf_stock["p1_tf"] >= 0.80, f"Unexpected p1_tf={tf_stock['p1_tf']} < 0.80; check MTP shift!"

    tf_q4, _ = eval_tf_alignment(qmtp, val_seqs, hfinals)
    print(f"  Q4-Dense TF:  p1={tf_q4['p1_tf']:.4f} (med={tf_q4['p1_tf_med']:.4f}) top4={tf_q4['top4_tf']:.4f}", flush=True)

    print("\n=== Probes 2.2b / 2.3 / 2.4: Unrolled MTP (Stock & Q4-Dense), Dynamic Stopping, & Trees (100 anchors) ===", flush=True)
    b1_ms = lat["stock"]["B=1"]
    b2_ms = lat["stock"]["B=2"]
    b4_ms = lat["stock"]["B=4"]
    qb1_ms = lat["q4_dense"]["B=1"]
    qb2_ms = lat["q4_dense"]["B=2"]
    qb4_ms = lat["q4_dense"]["B=4"]
    p1_logp = math.log(max(1e-6, tf_stock["p1_tf"]))

    unrolled_recs = []
    q_unrolled_recs = []
    tree_w2_accs = {b: [] for b in (4, 6, 8, 10, 12, 16)}
    tree_w2_ds = {b: [] for b in (4, 6, 8, 10, 12, 16)}
    tree_w3_accs = {b: [] for b in (4, 6, 8, 10, 12, 16)}
    tree_w3_ds = {b: [] for b in (4, 6, 8, 10, 12, 16)}
    e3_real_accs = {b: [] for b in (4, 6, 8, 10, 12, 16)}
    e3_real_ds = {b: [] for b in (4, 6, 8, 10, 12, 16)}
    e3_opt_accs = {b: [] for b in (4, 6, 8, 10, 12, 16)}
    e3_opt_ds = {b: [] for b in (4, 6, 8, 10, 12, 16)}

    for seq in val_seqs:
        anchors = sample_anchors(seq["prompt_len"], seq["total_len"], k_max=15, n_blk=5)
        hf_s = hfinals[seq["key"]]
        hf_q = hf_s
        if getattr(mtp, "_feat_fuse", None) is not None:
            hf_s = (hf_s.astype(mx.float32) + mtp._feat_fuse(seq["feats"].astype(mx.bfloat16)).astype(mx.float32)).astype(mx.bfloat16)
        if getattr(qmtp, "_feat_fuse", None) is not None:
            hf_q = (hf_q.astype(mx.float32) + qmtp._feat_fuse(seq["feats"].astype(mx.bfloat16)).astype(mx.float32)).astype(mx.bfloat16)
        mx.eval(hf_s, hf_q)
        ids_mx = seq["ids_mx"]
        ids = seq["ids"]
        is_med = seq["domain"] == "med"
        t4_ids_l, t4_lps_l = tf_seq_info[seq["key"]]

        for a in anchors:
            mc_base = KVCache()
            mc_qbase = KVCache()
            if a > 1:
                o = mtp.prefill(ids_mx[1:a][None], hf_s[: a - 1][None], mc_base)
                oq = qmtp.prefill(ids_mx[1:a][None], hf_q[: a - 1][None], mc_qbase)
                mx.eval(o, mc_base.state, oq, mc_qbase.state)

            gold = ids[a + 1 : a + 1 + 15]
            unrolled_recs.append(_unroll_anchor(mtp, mc_base, ids_mx[a : a + 1], hf_s[a - 1 : a], gold[:8], is_med, q4_dense=False))
            q_unrolled_recs.append(_unroll_anchor(qmtp, mc_qbase, ids_mx[a : a + 1], hf_q[a - 1 : a], gold[:8], is_med, q4_dense=True))

            # Beam trees W=2 and W=3
            t2 = _beam_tree_for_anchor(mtp, mc_base, ids_mx[a : a + 1], hf_s[a - 1 : a], gold, width=2, max_depth=6)
            t3 = _beam_tree_for_anchor(mtp, mc_base, ids_mx[a : a + 1], hf_s[a - 1 : a], gold, width=3, max_depth=6)
            er, eo = _eagle3_tree_bounds(t4_ids_l, t4_lps_l, a, gold, p1_logp)
            for b in (4, 6, 8, 10, 12, 16):
                tree_w2_accs[b].append(t2[b][0])
                tree_w2_ds[b].append(t2[b][1])
                tree_w3_accs[b].append(t3[b][0])
                tree_w3_ds[b].append(t3[b][1])
                e3_real_accs[b].append(er[b][0])
                e3_real_ds[b].append(er[b][1])
                e3_opt_accs[b].append(eo[b][0])
                e3_opt_ds[b].append(eo[b][1])

    n_blk = len(unrolled_recs)
    med_recs = [r for r in unrolled_recs if r["is_med"]]
    alphas = [round(sum(1 for r in unrolled_recs if r["pos_ok"][j]) / n_blk, 3) for j in range(8)]
    alphas_med = [round(sum(1 for r in med_recs if r["pos_ok"][j]) / len(med_recs), 3) for j in range(8)]
    q_alphas = [round(sum(1 for r in q_unrolled_recs if r["pos_ok"][j]) / n_blk, 3) for j in range(8)]
    print(f"  Unrolled MTP alphas (1..8):     {alphas}", flush=True)
    print(f"  Unrolled Q4D alphas (1..8):     {q_alphas}", flush=True)
    print(f"  Unrolled MTP alphas_med (1..8): {alphas_med}", flush=True)

    unrolled_K = {}
    q_unrolled_K = {}
    for k in range(1, 9):
        ea = sum(min(r["match_len"], k) + 1 for r in unrolled_recs) / n_blk
        em = sum(min(r["match_len"], k) + 1 for r in med_recs) / len(med_recs)
        qea = sum(min(r["match_len"], k) + 1 for r in q_unrolled_recs) / n_blk
        cyc = verify_cost_ms(k + 1) + k * b1_ms
        qcyc = verify_cost_ms(k + 1) + k * qb1_ms
        tps = 1e3 * ea / cyc
        qtps = 1e3 * qea / qcyc
        unrolled_K[str(k)] = {"L": k + 1, "E_acc": round(ea, 3), "E_acc_med": round(em, 3), "cyc_ms": round(cyc, 2), "proj_tok_s": round(tps, 2)}
        q_unrolled_K[str(k)] = {"L": k + 1, "E_acc": round(qea, 3), "cyc_ms": round(qcyc, 2), "proj_tok_s": round(qtps, 2)}
        print(
            f"  Unrolled K={k}: L={k+1:2d} E_acc={ea:.3f} (med={em:.3f}) -> {tps:5.2f} tok/s | "
            f"Q4D E_acc={qea:.3f} cyc={qcyc:6.2f}ms -> {qtps:5.2f} tok/s",
            flush=True,
        )

    dyn_uniform, dyn_two_param, best_q4d_dyn = _score_dyn_grid(q_unrolled_recs, qb1_ms)
    print("\n  Q4-Dense Per-Step DynStop (K in 1..7, L in 2..8) Sweep:", flush=True)
    for k_th, d in dyn_uniform.items():
        print(
            f"    th={k_th}: mean_L={d['mean_L']:4.2f} E_acc={d['E_acc']:.3f} cyc={d['mean_cyc_ms']:5.2f}ms -> {d['proj_tok_s']:5.2f} q4d tok/s",
            flush=True,
        )
    print("  Q4-Dense Two-Param DynStop (p1_min, th):", flush=True)
    for key_2p, d in dyn_two_param.items():
        print(
            f"    {key_2p:18s}: mean_L={d['mean_L']:4.2f} E_acc={d['E_acc']:.3f} cyc={d['mean_cyc_ms']:5.2f}ms -> {d['proj_tok_s']:5.2f} q4d tok/s",
            flush=True,
        )
    print(
        f"  BEST Q4D DynStop (val_seqs): p1_min={best_q4d_dyn['p1_min']:.2f}, th={best_q4d_dyn['th']:.2f}, K_cap=7 -> "
        f"mean_L={best_q4d_dyn['mean_L']:.2f}, E_acc={best_q4d_dyn['E_acc']:.3f}, {best_q4d_dyn['proj_tok_s']:.2f} q4d tok/s",
        flush=True,
    )

    trees = {"beam_w2": {}, "beam_w3": {}, "eagle3_real": {}, "eagle3_opt": {}}
    print("\n  Tree Projections:", flush=True)
    for b in (4, 6, 8, 10, 12, 16):
        ea2 = sum(tree_w2_accs[b]) / n_blk
        d2 = sum(tree_w2_ds[b]) / n_blk
        c2 = verify_cost_ms(b) + d2 * b2_ms
        t2 = 1e3 * ea2 / c2
        qt2 = 1e3 * ea2 / (verify_cost_ms(b) + d2 * qb2_ms)

        ea3 = sum(tree_w3_accs[b]) / n_blk
        d3 = sum(tree_w3_ds[b]) / n_blk
        c3 = verify_cost_ms(b) + d3 * b4_ms
        t3 = 1e3 * ea3 / c3
        qt3 = 1e3 * ea3 / (verify_cost_ms(b) + d3 * qb4_ms)

        eer = sum(e3_real_accs[b]) / n_blk
        der = sum(e3_real_ds[b]) / n_blk
        cer = verify_cost_ms(b) + der * b2_ms
        ter = 1e3 * eer / cer

        eeo = sum(e3_opt_accs[b]) / n_blk
        deo = sum(e3_opt_ds[b]) / n_blk
        ceo = verify_cost_ms(b) + deo * b2_ms
        teo = 1e3 * eeo / ceo

        trees["beam_w2"][str(b)] = {"E_acc": round(ea2, 3), "mean_d": round(d2, 2), "cyc_ms": round(c2, 2), "proj_tok_s": round(t2, 2), "q4d_proj_tok_s": round(qt2, 2)}
        trees["beam_w3"][str(b)] = {"E_acc": round(ea3, 3), "mean_d": round(d3, 2), "cyc_ms": round(c3, 2), "proj_tok_s": round(t3, 2), "q4d_proj_tok_s": round(qt3, 2)}
        trees["eagle3_real"][str(b)] = {"E_acc": round(eer, 3), "mean_d": round(der, 2), "cyc_ms": round(cer, 2), "proj_tok_s": round(ter, 2)}
        trees["eagle3_opt"][str(b)] = {"E_acc": round(eeo, 3), "mean_d": round(deo, 2), "cyc_ms": round(ceo, 2), "proj_tok_s": round(teo, 2)}
        print(
            f"    B_tree={b:2d}: W2={ea2:.3f} ({t2:5.2f} / q4d={qt2:5.2f} tok/s) | W3={ea3:.3f} ({t3:5.2f} / q4d={qt3:5.2f}) | "
            f"E3_real={eer:.3f} ({ter:5.2f}) | E3_opt={eeo:.3f} ({teo:5.2f})",
            flush=True,
        )

    w2_b8_qtps = trees["beam_w2"]["8"]["q4d_proj_tok_s"]
    tree_delta = round(w2_b8_qtps - best_q4d_dyn["proj_tok_s"], 2)
    print(
        f"\n  Conditional Tree Gate (val_seqs): Beam-W2 B_tree=8 ({w2_b8_qtps:.2f}) vs Best Q4D DynStop ({best_q4d_dyn['proj_tok_s']:.2f}) -> delta = {tree_delta:+.2f} tok/s",
        flush=True,
    )

    return {
        "latency_ms": lat,
        "teacher_forced": {"stock": tf_stock, "q4_dense": tf_q4},
        "unrolled": {
            "n_blocks": n_blk,
            "alphas": alphas,
            "q4d_alphas": q_alphas,
            "alphas_med": alphas_med,
            "fixed_K": unrolled_K,
            "q4d_fixed_K": q_unrolled_K,
            "dyn_stop_uniform": dyn_uniform,
            "dyn_stop_two_param": dyn_two_param,
            "best_q4d_dyn": best_q4d_dyn,
            "tree_w2_b8_vs_dyn_delta": tree_delta,
        },
        "trees": trees,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/science/models/lingsante-mlx-mxfp4-bf16"))
    ap.add_argument("--mtp-dir", default="")
    ap.add_argument("--data-dir", default=os.path.expanduser("~/science/data/lingsante_dflash_train"))
    ap.add_argument("--hfinal", default=os.path.expanduser("~/science/data/lingsante_dflash_train/val_hfinal.safetensors"))
    ap.add_argument("--out", default=os.path.expanduser("~/science/runs/probe_mtp_eagle.json"))
    a = ap.parse_args()

    maybe_set_recommended_wired_limit()
    model_dir = Path(a.model).expanduser()
    mtp_dir = Path(a.mtp_dir).expanduser() if a.mtp_dir else (model_dir / "mtp")
    data_dir = Path(a.data_dir).expanduser()
    hfinal_path = Path(a.hfinal).expanduser()

    val_seqs = load_val_seqs(data_dir)
    print(f"Loaded {len(val_seqs)} internal val_seqs (med={sum(1 for s in val_seqs if s['domain']=='med')})", flush=True)

    capture_val_hfinal(model_dir, val_seqs, hfinal_path)
    hfinals = mx.load(str(hfinal_path))
    mx.eval(*hfinals.values())

    mtp = load_standalone_mtp(model_dir, mtp_dir, data_dir, q4_dense=False)
    qmtp = load_standalone_mtp(model_dir, mtp_dir, data_dir, q4_dense=True)
    print(
        f"Loaded MTP from {mtp_dir} + Q4-Dense MTP (bridge={mtp._bridge is not None}, "
        f"feat_fuse={mtp._feat_fuse is not None}, {mx.get_active_memory()/1e9:.2f} GB active)",
        flush=True,
    )

    out = run_probes(mtp, qmtp, val_seqs, hfinals)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"\nSaved {a.out}", flush=True)


if __name__ == "__main__":
    main()
