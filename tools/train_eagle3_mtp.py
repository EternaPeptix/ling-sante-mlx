"""Train EAGLE-3 multi-step unrolled Layer-72 MTP drafter on 512S1 (target unloaded, ~10 GB RAM)."""
import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(_root / "vendor"), str(_root)]

import mlx_lm  # noqa: E402,F401
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import mlx.optimizers as optim  # noqa: E402
from mlx.utils import tree_flatten, tree_map, tree_unflatten  # noqa: E402
from mlx_lm.models.bailing_moe_v3.config import ModelArgs  # noqa: E402
from mlx_lm.models.bailing_moe_v3.mtp import BailingMTPLayer  # noqa: E402
from mlx_lm.models.cache import KVCache  # noqa: E402
from mlx_lm.utils import infer_quant_config, maybe_set_recommended_wired_limit  # noqa: E402

_VERIFY_ANCHORS = [(1, 39.1), (3, 49.0), (5, 57.2), (8, 70.3), (9, 86.3), (10, 90.0), (12, 99.6), (16, 121.0)]
_NORM_PREFIXES = (
    "enorm.",
    "hnorm.",
    "input_layernorm.",
    "post_attention_layernorm.",
    "final_layernorm.",
    "attention.q_a_layernorm.",
    "attention.kv_a_layernorm.",
)
_DEFAULT_DEPTH_WEIGHTS = "2.0,1.3,1.0,0.85,0.70,0.55"


def verify_cost_ms(L: float) -> float:
    if L <= _VERIFY_ANCHORS[0][0]:
        return _VERIFY_ANCHORS[0][1]
    if L >= _VERIFY_ANCHORS[-1][0]:
        return _VERIFY_ANCHORS[-1][1]
    for (x0, y0), (x1, y1) in zip(_VERIFY_ANCHORS[:-1], _VERIFY_ANCHORS[1:]):
        if x0 <= L <= x1:
            return y0 + ((L - x0) / (x1 - x0)) * (y1 - y0)
    return _VERIFY_ANCHORS[-1][1]


class LoRALinear(nn.Module):
    """Zero-init low-rank adapter around a frozen float32 base Linear weight."""

    def __init__(self, linear: nn.Linear, r: int = 32, alpha: float = 32.0):
        super().__init__()
        out_dim, in_dim = linear.weight.shape
        self.weight = linear.weight.astype(mx.float32)
        self.r = int(r)
        self.scale = float(alpha) / float(r)
        self.lora_A = mx.random.normal((r, in_dim), dtype=mx.float32) * (1.0 / math.sqrt(in_dim))
        self.lora_B = mx.zeros((out_dim, r), dtype=mx.float32)
        self.freeze(keys=["weight"], recurse=False)

    def __call__(self, x: mx.array) -> mx.array:
        xf = x.astype(mx.float32)
        return (xf @ self.weight.T) + ((xf @ self.lora_A.T) @ self.lora_B.T) * self.scale

    def merged_weight_bf16(self) -> mx.array:
        return (self.weight + self.scale * (self.lora_B @ self.lora_A)).astype(mx.bfloat16)

    def to_linear(self) -> nn.Linear:
        out_dim, in_dim = self.weight.shape
        lin = nn.Linear(in_dim, out_dim, bias=False)
        lin.weight = self.merged_weight_bf16()
        return lin


class FeedbackBridge(nn.Module):
    """Zero-init residual MLP mapping MTP post-norm hidden states onto target h_final manifold."""

    def __init__(self, hidden_size: int = 4096, rank: int = 1024, eps: float = 1e-6):
        super().__init__()
        self._is_v2 = False
        self.norm = nn.RMSNorm(hidden_size, eps=eps)
        self.up = nn.Linear(hidden_size, rank, bias=False)
        self.down = nn.Linear(rank, hidden_size, bias=False)
        self.down.weight = mx.zeros((hidden_size, rank), dtype=mx.float32)

    def __call__(self, h: mx.array) -> mx.array:
        xf = h.astype(mx.float32)
        return xf + self.down(nn.silu(self.up(self.norm(xf))))


class FeedbackBridgeV2(nn.Module):
    """Zero-init dual-input SwiGLU residual bridge combining MTP h_out with lookahead h_in."""

    def __init__(self, hidden_size: int = 4096, rank: int = 1024, eps: float = 1e-6):
        super().__init__()
        self._is_v2 = True
        self.norm_out = nn.RMSNorm(hidden_size, eps=eps)
        self.norm_in = nn.RMSNorm(hidden_size, eps=eps)
        self.gate = nn.Linear(hidden_size * 2, rank, bias=False)
        self.up = nn.Linear(hidden_size * 2, rank, bias=False)
        self.down = nn.Linear(rank, hidden_size, bias=False)
        self.down.weight = mx.zeros((hidden_size, rank), dtype=mx.float32)

    def __call__(self, h_out: mx.array, h_in: mx.array) -> mx.array:
        xo = h_out.astype(mx.float32)
        xi = h_in.astype(mx.float32)
        cat = mx.concatenate([self.norm_out(xo), self.norm_in(xi)], axis=-1)
        return xo + self.down(nn.silu(self.gate(cat)) * self.up(cat))


def _apply_bridge(bridge, h_out: mx.array, h_in: mx.array) -> mx.array:
    if bridge is None:
        return h_out
    if getattr(bridge, "_is_v2", False):
        return bridge(h_out, h_in)
    return bridge(h_out)


class FeatFuse(nn.Module):
    """Zero-init residual projection from concatenated multi-layer target features onto hidden_size."""

    def __init__(self, in_dim: int = 20480, rank: int = 256, out_dim: int = 4096, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.RMSNorm(in_dim, eps=eps)
        self.up = nn.Linear(in_dim, rank, bias=False)
        self.down = nn.Linear(rank, out_dim, bias=False)
        self.down.weight = mx.zeros((out_dim, rank), dtype=mx.float32)

    def __call__(self, feats: mx.array) -> mx.array:
        xf = feats.astype(mx.float32)
        return self.down(nn.silu(self.up(self.norm(xf))))


class EAGLE3MTPWrapper(nn.Module):
    """Trainable wrapper combining Layer-72 MTP (LoRA + norms, frozen MXFP4 experts), FeedbackBridge, and optional FeatFuse."""

    def __init__(self, mtp: BailingMTPLayer, bridge: FeedbackBridge | FeedbackBridgeV2, feat_fuse: FeatFuse | None = None):
        super().__init__()
        self.mtp = mtp
        self.feedback_bridge = bridge
        self.feat_fuse = feat_fuse


def build_mtp_float32(model_dir: Path, raw_weights: dict) -> tuple[BailingMTPLayer, ModelArgs]:
    cfg = json.loads((model_dir / "config.json").read_text())
    args = ModelArgs.from_dict(cfg)
    mtp = BailingMTPLayer(args)

    def pred(p, m):
        if not hasattr(m, "to_quantized") or f"{p}.scales" not in raw_weights:
            return False
        return infer_quant_config(p, m, raw_weights)

    nn.quantize(mtp, group_size=32, bits=4, mode="mxfp4", class_predicate=pred)
    mtp.load_weights(list(raw_weights.items()), strict=True)

    # Cast non-quantized float arrays to float32 while keeping frozen MXFP4/embed_q/unembed_out/gate exact
    keep_prefixes = ("mlp.switch_mlp.", "mlp.gate.", "attention.embed_q.", "attention.unembed_out.")
    new_params = []
    for k, v in tree_flatten(mtp.parameters()):
        if not any(k.startswith(p) for p in keep_prefixes) and v.dtype in (mx.bfloat16, mx.float16):
            new_params.append((k, v.astype(mx.float32)))
        else:
            new_params.append((k, v))
    mtp.update(tree_unflatten(new_params))
    return mtp, args


def wrap_mtp_trainable(
    mtp: BailingMTPLayer,
    args: ModelArgs,
    r: int = 32,
    alpha: float = 32.0,
    use_feat_fuse: bool = False,
    bridge_v2: bool = False,
    bridge_rank: int = 1024,
) -> EAGLE3MTPWrapper:
    mtp.eh_proj = LoRALinear(mtp.eh_proj, r=r, alpha=alpha)
    sa = mtp.attention
    for attr in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "dense", "g_proj"):
        setattr(sa, attr, LoRALinear(getattr(sa, attr), r=r, alpha=alpha))
    se = mtp.mlp.shared_experts
    for attr in ("gate_proj", "up_proj", "down_proj"):
        setattr(se, attr, LoRALinear(getattr(se, attr), r=r, alpha=alpha))

    if bridge_v2:
        bridge = FeedbackBridgeV2(args.hidden_size, rank=bridge_rank, eps=args.rms_norm_eps)
    else:
        bridge = FeedbackBridge(args.hidden_size, rank=bridge_rank, eps=args.rms_norm_eps)
    bridge.update(tree_map(lambda x: x.astype(mx.float32), bridge.parameters()))
    feat_fuse = None
    if use_feat_fuse:
        feat_fuse = FeatFuse(args.hidden_size * 5, rank=256, out_dim=args.hidden_size, eps=args.rms_norm_eps)
        feat_fuse.update(tree_map(lambda x: x.astype(mx.float32), feat_fuse.parameters()))
    wrapper = EAGLE3MTPWrapper(mtp, bridge, feat_fuse=feat_fuse)
    freeze_for_eagle3(wrapper)
    return wrapper


def freeze_for_eagle3(wrapper: EAGLE3MTPWrapper):
    wrapper.freeze(recurse=True)
    mtp = wrapper.mtp
    for _, mod in mtp.named_modules():
        if isinstance(mod, LoRALinear):
            mod.unfreeze(keys=["lora_A", "lora_B"], recurse=False)
    for norm_mod in (
        mtp.enorm,
        mtp.hnorm,
        mtp.input_layernorm,
        mtp.post_attention_layernorm,
        mtp.final_layernorm,
        mtp.attention.q_a_layernorm,
        mtp.attention.kv_a_layernorm,
    ):
        norm_mod.unfreeze(recurse=True)
    wrapper.feedback_bridge.unfreeze(recurse=True)
    if wrapper.feat_fuse is not None:
        wrapper.feat_fuse.unfreeze(recurse=True)


def save_eagle3_checkpoint(wrapper: EAGLE3MTPWrapper, raw_weights: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    mtp = wrapper.mtp
    mods = dict(mtp.named_modules())
    flat = dict(tree_flatten(mtp.parameters()))
    handled_prefixes = set()
    mtp_out = {}
    for name, mod in mods.items():
        if isinstance(mod, LoRALinear):
            mtp_out[f"{name}.weight"] = mod.merged_weight_bf16()
            handled_prefixes.add(name + ".")
    for k, v in flat.items():
        if any(k.startswith(p) for p in handled_prefixes):
            continue
        if k in raw_weights:
            mtp_out[k] = v.astype(raw_weights[k].dtype)
        else:
            mtp_out[k] = v.astype(mx.bfloat16) if v.dtype == mx.float32 else v

    fb_out = {
        f"feedback_bridge.{k}": v.astype(mx.bfloat16)
        for k, v in tree_flatten(wrapper.feedback_bridge.parameters())
    }
    if wrapper.feat_fuse is not None:
        for k, v in tree_flatten(wrapper.feat_fuse.parameters()):
            fb_out[f"feat_fuse.{k}"] = v.astype(mx.bfloat16)
    mx.eval(*mtp_out.values(), *fb_out.values())
    mx.save_safetensors(str(out_dir / "model-mtp.safetensors"), mtp_out)
    mx.save_safetensors(str(out_dir / "feedback_bridge.safetensors"), fb_out)


def eagle3_unroll_forward(
    wrapper: EAGLE3MTPWrapper,
    embed_w_bf16: mx.array,
    ids_mx: mx.array,
    h_final: mx.array,
    *,
    feats: mx.array | None = None,
    K: int = 6,
) -> tuple[list[mx.array], list[mx.array]]:
    """Exact parallel K-step EAGLE-3 unroll matching sequential KVCache multi-step semantics."""
    mtp = wrapper.mtp
    bridge = wrapper.feedback_bridge
    attn = mtp.attention
    se = mtp.mlp.shared_experts
    sw = mtp.mlp.switch_mlp
    gate_mod = mtp.mlp.gate

    T = int(ids_mx.shape[0])
    L = T - 1
    assert T > K + 1

    # Shared upper-triangular causal mask for context length L
    idx_row = mx.arange(L, dtype=mx.int32)[:, None]
    idx_col = mx.arange(L, dtype=mx.int32)[None, :]
    full_causal_mask = mx.where(idx_col > idx_row, mx.array(-1e9, dtype=mx.float32), mx.array(0.0, dtype=mx.float32))

    h_in_base = h_final[:L].astype(mx.float32)
    if wrapper.feat_fuse is not None:
        assert feats is not None
        h_in_base = h_in_base + wrapper.feat_fuse(feats[:L])
    h_fb_prev = h_in_base[None]
    kl_1 = None
    kr_1 = None
    anc_hist: list[tuple[mx.array, mx.array]] = []
    h_outs: list[mx.array] = []
    h_fbs: list[mx.array] = []

    for k in range(1, K + 1):
        L_k = L - (k - 1)
        toks_k = ids_mx[k:T][None]  # (1, L_k)
        h_in_k = h_fb_prev if k == 1 else h_fb_prev[:, :-1, :]  # (1, L_k, 4096)

        e_n = mtp.enorm(embed_w_bf16[toks_k].astype(mx.float32))
        h_n = mtp.hnorm(h_in_k)
        h = mtp.eh_proj(mx.concatenate([e_n, h_n], axis=-1))

        # Absorbed MLA attention
        x_a = mtp.input_layernorm(h)
        q = attn.q_b_proj(attn.q_a_layernorm(attn.q_a_proj(x_a)))
        q = q.reshape(1, L_k, attn.num_heads, attn.qk_head_dim).transpose(0, 2, 1, 3)
        q_nope, q_rope = mx.split(q, [attn.qk_nope_head_dim], axis=-1)

        comp = attn.kv_a_proj_with_mqa(x_a)
        comp, k_rope = mx.split(comp, [attn.kv_lora_rank], axis=-1)
        kv_lat = attn.kv_a_layernorm(comp)[:, None, :, :]  # (1, 1, L_k, 512)
        k_rope = k_rope.reshape(1, L_k, 1, attn.qk_rope_head_dim).transpose(0, 2, 1, 3)

        q_rope = attn.rope(q_rope, offset=k - 1)
        k_rope = attn.rope(k_rope, offset=k - 1)

        q_abs = attn.embed_q(q_nope.astype(mx.bfloat16)).astype(mx.float32)  # (1, 64, L_k, 512)
        if k == 1:
            kl_1 = kv_lat
            kr_1 = k_rope
            sc = (q_abs * attn.scale) @ kl_1.swapaxes(-1, -2) + (q_rope * attn.scale) @ kr_1.swapaxes(-1, -2)
            probs = mx.softmax(sc + full_causal_mask[None, None, :, :], axis=-1)
            out_lat = probs @ kl_1
        else:
            anc_hist.append((kv_lat, k_rope))
            sc_ctx = (q_abs * attn.scale) @ kl_1.swapaxes(-1, -2) + (q_rope * attn.scale) @ kr_1.swapaxes(-1, -2)
            sc_ctx = sc_ctx + full_causal_mask[:L_k, :][None, None, :, :]
            anc_sliced = [(kl_m[:, :, :L_k, :], kr_m[:, :, :L_k, :]) for kl_m, kr_m in anc_hist]
            sc_ms = [
                (mx.sum(q_abs * kl_s, axis=-1, keepdims=True) + mx.sum(q_rope * kr_s, axis=-1, keepdims=True)) * attn.scale
                for kl_s, kr_s in anc_sliced
            ]
            sc_all = mx.concatenate([sc_ctx, *sc_ms], axis=-1)
            probs = mx.softmax(sc_all, axis=-1)
            out_lat = probs[..., :L] @ kl_1
            for idx_m, (kl_s, _) in enumerate(anc_sliced):
                out_lat = out_lat + probs[..., L + idx_m : L + idx_m + 1] * kl_s

        out_v = attn.unembed_out(out_lat.astype(mx.bfloat16)).astype(mx.float32).transpose(0, 2, 1, 3)
        gate_g = mx.sigmoid(attn.g_proj(x_a))
        h_post = h + attn.dense((out_v * gate_g[..., None]).reshape(1, L_k, -1))

        # MLP: frozen MXFP4 switch_mlp + LoRA shared_experts
        mlp_in = mtp.post_attention_layernorm(h_post)
        mlp_bf = mlp_in.astype(mx.bfloat16)
        inds, scores = gate_mod(mlp_bf)
        routed = (sw(mlp_bf, inds).astype(mx.float32) * scores[..., None]).sum(axis=-2)
        sh = se.down_proj(nn.silu(se.gate_proj(mlp_in)) * se.up_proj(mlp_in))
        h_out_k = mtp.final_layernorm(h_post + routed + sh)
        h_fb_k = _apply_bridge(bridge, h_out_k, h_in_k)

        h_outs.append(h_out_k)
        h_fbs.append(h_fb_k)
        h_fb_prev = h_fb_k

    return h_outs, h_fbs


def sequential_step(wrapper: EAGLE3MTPWrapper, embed_w_bf16: mx.array, tok: mx.array, prev_h: mx.array, cache: KVCache, q4_head):
    """Single-token autoregressive step for both Startup Test 1 equivalence and unrolled validation."""
    mtp = wrapper.mtp
    bridge = wrapper.feedback_bridge
    e_n = mtp.enorm(embed_w_bf16[tok.reshape(1, 1)].astype(mx.float32))
    h_n = mtp.hnorm(prev_h.reshape(1, 1, -1).astype(mx.float32))
    h = mtp.eh_proj(mx.concatenate([e_n, h_n], axis=-1))

    h_a = mtp.attention(mtp.input_layernorm(h), mask=None, cache=cache)
    h_post = h + h_a
    mlp_in = mtp.post_attention_layernorm(h_post)
    mlp_bf = mlp_in.astype(mx.bfloat16)
    inds, scores = mtp.mlp.gate(mlp_bf)
    routed = (mtp.mlp.switch_mlp(mlp_bf, inds).astype(mx.float32) * scores[..., None]).sum(axis=-2)
    se = mtp.mlp.shared_experts
    sh = se.down_proj(nn.silu(se.gate_proj(mlp_in)) * se.up_proj(mlp_in))
    h_out = mtp.final_layernorm(h_post + routed + sh)
    h_fb = _apply_bridge(bridge, h_out, prev_h.reshape(1, 1, -1))
    lg = q4_head(h_out[:, 0, :].astype(mx.bfloat16)).astype(mx.float32)
    return h_out[:, 0, :], h_fb[:, 0, :], lg


def _sequential_prefill(wrapper: EAGLE3MTPWrapper, embed_w_bf16: mx.array, toks_shifted: mx.array, hs: mx.array, cache: KVCache) -> mx.array:
    mtp = wrapper.mtp
    e_n = mtp.enorm(embed_w_bf16[toks_shifted].astype(mx.float32))
    h_n = mtp.hnorm(hs.astype(mx.float32))
    h = mtp.eh_proj(mx.concatenate([e_n, h_n], axis=-1))
    from mlx_lm.layers.attention import create_attention_mask
    mask = create_attention_mask(h, cache, return_array=True)
    h_post = h + mtp.attention(mtp.input_layernorm(h), mask=mask, cache=cache)
    mlp_in = mtp.post_attention_layernorm(h_post)
    mlp_bf = mlp_in.astype(mx.bfloat16)
    inds, scores = mtp.mlp.gate(mlp_bf)
    routed = (mtp.mlp.switch_mlp(mlp_bf, inds).astype(mx.float32) * scores[..., None]).sum(axis=-2)
    se = mtp.mlp.shared_experts
    sh = se.down_proj(nn.silu(se.gate_proj(mlp_in)) * se.up_proj(mlp_in))
    return mtp.final_layernorm(h_post + routed + sh)


def loss_on_sequence(
    wrapper: EAGLE3MTPWrapper,
    embed_w_bf16: mx.array,
    head_t_bf16: mx.array,
    seq: dict,
    anchor_init: dict,
    anchor_elems: int,
    *,
    K: int = 6,
    feat_weight: float = 0.0,
    depth_weights: tuple[float, ...] = (2.0, 1.3, 1.0, 0.85, 0.70, 0.55),
):
    ids_mx = seq["ids_mx"]
    hf = seq["hf"]
    feats = seq.get("feats")
    plen = seq["prompt_len"]
    T = seq["total_len"]
    t32_ids = seq["top32_ids"]
    t32_lgs = seq["top32_logits"]

    h_outs, h_fbs = eagle3_unroll_forward(wrapper, embed_w_bf16, ids_mx, hf, feats=feats, K=K)

    loss_acc = mx.array(0.0, dtype=mx.float32)
    kl1_v = mx.array(0.0, dtype=mx.float32)
    feat1_v = mx.array(0.0, dtype=mx.float32)
    w_sum = 0.0
    for idx_k in range(K):
        k = idx_k + 1
        sl_end = T - 1 - k
        if sl_end <= plen - 1:
            continue
        wk = depth_weights[idx_k]
        w_sum += wk
        h_out_g = h_outs[idx_k][0, plen - 1 : sl_end]
        h_fb_g = h_fbs[idx_k][0, plen - 1 : sl_end]
        tgt_ids = ids_mx[plen + k : T]
        b_t32_ids = t32_ids[plen + k - 1 : T - 1]
        b_t32_probs = mx.softmax(t32_lgs[plen + k - 1 : T - 1].astype(mx.float32), axis=-1)
        hf_tgt = hf[plen + k - 1 : T - 1].astype(mx.float32)

        logits_k = (h_out_g.astype(mx.bfloat16) @ head_t_bf16).astype(mx.float32)
        lse = mx.logsumexp(logits_k, axis=-1, keepdims=True)
        lp = logits_k - lse
        ce_k = -mx.mean(mx.take_along_axis(lp, tgt_ids[:, None], axis=-1).squeeze(-1))
        stu_t32_lp = mx.take_along_axis(lp, b_t32_ids, axis=-1)
        log_pt = mx.log(mx.maximum(b_t32_probs, 1e-12))
        kl_k = mx.mean(mx.sum(b_t32_probs * (log_pt - stu_t32_lp), axis=-1))

        if feat_weight > 0.0:
            cos_num = mx.sum(h_fb_g * hf_tgt, axis=-1)
            cos_den = mx.sqrt(mx.sum(h_fb_g * h_fb_g, axis=-1) + 1e-8) * mx.sqrt(mx.sum(hf_tgt * hf_tgt, axis=-1) + 1e-8)
            cos_mean = mx.mean(cos_num / cos_den)
            l1_normed = 0.01 * mx.mean(mx.abs(h_fb_g - hf_tgt))
            feat_k = (1.0 - cos_mean) + l1_normed
        else:
            feat_k = mx.array(0.0, dtype=mx.float32)

        loss_acc = loss_acc + wk * (0.8 * kl_k + 0.2 * ce_k + feat_weight * feat_k)
        if k == 1:
            kl1_v = kl_k
            feat1_v = feat_k

    cur_map = dict(tree_flatten(wrapper.trainable_parameters()))
    anc_sq = mx.array(0.0, dtype=mx.float32)
    for ak, w0 in anchor_init.items():
        d = cur_map[ak] - w0
        anc_sq = anc_sq + mx.sum(d * d)
    anc_pen = 0.01 * (anc_sq / max(1, anchor_elems))

    tot = (loss_acc / max(1.0, w_sum)) + anc_pen
    return tot, (kl1_v, feat1_v, anc_pen)


def _check_k6_par_vs_seq(
    wrapper: EAGLE3MTPWrapper,
    embed_w_bf16: mx.array,
    head_w_bf16: mx.array,
    val0_seq: dict,
    q4h,
) -> float:
    """Verify K=6 parallel eagle3_unroll_forward matches sequential KVCache step-for-step."""
    ids_mx = val0_seq["ids_mx"]
    hf = val0_seq["hf"]
    feats = val0_seq.get("feats")
    plen = val0_seq["prompt_len"]
    a = plen + 20

    h_outs, _ = eagle3_unroll_forward(
        wrapper,
        embed_w_bf16,
        ids_mx[: a + 8],
        hf[: a + 8],
        feats=feats[: a + 8] if feats is not None else None,
        K=6,
    )
    hf_eff = hf[: a + 8].astype(mx.float32)
    if wrapper.feat_fuse is not None:
        assert feats is not None
        hf_eff = hf_eff + wrapper.feat_fuse(feats[: a + 8])
        mx.eval(hf_eff)
    mc = KVCache()
    if a > 1:
        _ = _sequential_prefill(wrapper, embed_w_bf16, ids_mx[1:a][None], hf_eff[: a - 1][None], mc)
        mx.eval(mc.state)

    hp = hf_eff[a - 1 : a]
    max_hdiff = 0.0
    for idx_k in range(6):
        k = idx_k + 1
        tok_k = ids_mx[a + k - 1 : a + k]
        ho_seq, hp, _ = sequential_step(wrapper, embed_w_bf16, tok_k, hp, mc, q4h)
        ho_par = h_outs[idx_k][0, a - 1 : a]
        lg_seq = (ho_seq.astype(mx.bfloat16) @ head_w_bf16.T).astype(mx.float32)
        lg_par = (ho_par.astype(mx.bfloat16) @ head_w_bf16.T).astype(mx.float32)
        p_seq, p_par = mx.argmax(lg_seq, axis=-1), mx.argmax(lg_par, axis=-1)
        mx.eval(ho_seq, ho_par, p_seq, p_par)
        d_k = float(mx.max(mx.abs(ho_seq - ho_par)).item())
        max_hdiff = max(max_hdiff, d_k)
        assert d_k < 5e-2, f"Parallel K={k} hidden diff={d_k} >= 5e-2"
        assert int(p_seq.item()) == int(p_par.item()), f"Parallel K={k} argmax mismatch {int(p_seq.item())} vs {int(p_par.item())}"
    return max_hdiff


def run_startup_tests(
    base_mtp: BailingMTPLayer,
    wrapper: EAGLE3MTPWrapper,
    raw_weights: dict,
    embed_w_bf16: mx.array,
    head_w_bf16: mx.array,
    val0_seq: dict,
    out_dir: Path,
    feat_weight: float = 0.0,
    depth_weights: tuple[float, ...] = (2.0, 1.3, 1.0, 0.85, 0.70, 0.55),
):
    """Verify K=6 parallel vs sequential equivalence (zero & non-zero bridge), Ep-0 identity, reload, and VJP grads."""
    ids_mx = val0_seq["ids_mx"]
    hf = val0_seq["hf"]
    feats = val0_seq.get("feats")
    q4h = nn.QuantizedLinear.from_linear(
        wrapper.mtp._lm_head if wrapper.mtp._lm_head is not None else _make_head(head_w_bf16),
        group_size=64,
        bits=4,
        mode="affine",
    )
    mx.eval(q4h.parameters())

    max_hdiff = _check_k6_par_vs_seq(wrapper, embed_w_bf16, head_w_bf16, val0_seq, q4h)
    print(f"Startup test 1 (Parallel K=6 vs Sequential KVCache max_diff={max_hdiff:.2e}, argmax 6/6): PASS", flush=True)

    # Test 1b: Non-zero FeedbackBridge equivalence (catches h_out / h_in indexing bugs in bridge_v2)
    orig_down = mx.array(wrapper.feedback_bridge.down.weight)
    mx.eval(orig_down)
    wrapper.feedback_bridge.down.weight = mx.full(orig_down.shape, 1e-3, dtype=mx.float32)
    mx.eval(wrapper.feedback_bridge.down.weight)
    max_hdiff_nz = _check_k6_par_vs_seq(wrapper, embed_w_bf16, head_w_bf16, val0_seq, q4h)
    wrapper.feedback_bridge.down.weight = orig_down
    mx.eval(wrapper.feedback_bridge.down.weight)
    print(f"Startup test 1b (Non-zero FeedbackBridge K=6 max_diff={max_hdiff_nz:.2e}, argmax 6/6): PASS", flush=True)

    # Test 2: Zero-init LoRA + FeedbackBridge (+ FeatFuse) Epoch-0 identity vs unwrapped base_mtp
    sub_ids, sub_hf = ids_mx[:40], hf[:40]
    sub_feats = feats[:40] if feats is not None else None
    h_wrap_0, h_fb_0 = eagle3_unroll_forward(wrapper, embed_w_bf16, sub_ids, sub_hf, feats=sub_feats, K=2)
    base_wrap = EAGLE3MTPWrapper(base_mtp, FeedbackBridge(wrapper.mtp.enorm.weight.shape[0], rank=1024))
    h_base_0, _ = eagle3_unroll_forward(base_wrap, embed_w_bf16, sub_ids, sub_hf, K=2)
    mx.eval(h_wrap_0[0], h_fb_0[0], h_base_0[0])
    d_id = max(
        float(mx.max(mx.abs(h_wrap_0[0] - h_base_0[0])).item()),
        float(mx.max(mx.abs(h_fb_0[0] - h_wrap_0[0])).item()),
    )
    assert d_id == 0.0, f"Test 2 Epoch-0 identity diff={d_id}"
    print(f"Startup test 2 (Zero-init LoRA + FeedbackBridge Ep0 diff={d_id:.2e}): PASS", flush=True)

    # Test 3: Save merged model-mtp.safetensors + feedback_bridge.safetensors & strict=True reload
    tmp_dir = out_dir.parent / "_tmp_eagle3_test"
    save_eagle3_checkpoint(wrapper, raw_weights, tmp_dir)
    rel_w = mx.load(str(tmp_dir / "model-mtp.safetensors"))
    rel_mtp, rel_args = build_mtp_float32(Path(os.path.expanduser("~/science/models/lingsante-mlx-mxfp4-bf16")), rel_w)
    rel_fb_w = mx.load(str(tmp_dir / "feedback_bridge.safetensors"))
    fb_items = [(k.removeprefix("feedback_bridge."), v.astype(mx.float32)) for k, v in rel_fb_w.items() if k.startswith("feedback_bridge.")]
    if "feedback_bridge.gate.weight" in rel_fb_w:
        rel_rank = int(rel_fb_w["feedback_bridge.gate.weight"].shape[0])
        rel_fb = FeedbackBridgeV2(rel_args.hidden_size, rank=rel_rank, eps=rel_args.rms_norm_eps)
    else:
        rel_rank = int(rel_fb_w["feedback_bridge.up.weight"].shape[0])
        rel_fb = FeedbackBridge(rel_args.hidden_size, rank=rel_rank, eps=rel_args.rms_norm_eps)
    rel_fb.load_weights(fb_items, strict=True)
    rel_ff = None
    ff_items = [(k.removeprefix("feat_fuse."), v.astype(mx.float32)) for k, v in rel_fb_w.items() if k.startswith("feat_fuse.")]
    if wrapper.feat_fuse is not None:
        assert len(ff_items) > 0, "Missing feat_fuse.* weights in saved feedback_bridge.safetensors"
        rel_ff = FeatFuse(rel_args.hidden_size * 5, rank=256, out_dim=rel_args.hidden_size, eps=rel_args.rms_norm_eps)
        rel_ff.load_weights(ff_items, strict=True)
    rel_wrap = EAGLE3MTPWrapper(rel_mtp, rel_fb, feat_fuse=rel_ff)
    h_rel, _ = eagle3_unroll_forward(rel_wrap, embed_w_bf16, sub_ids, sub_hf, feats=sub_feats, K=2)
    mx.eval(h_rel[0])
    d_rel = float(mx.max(mx.abs(h_rel[0] - h_wrap_0[0])).item())
    assert d_rel < 5e-2, f"Test 3 merged reload diff={d_rel}"
    for p in tmp_dir.iterdir():
        p.unlink()
    tmp_dir.rmdir()
    print(f"Startup test 3 (Merged strict=True reload diff={d_rel:.2e}): PASS", flush=True)
    del rel_mtp, rel_fb, rel_ff, rel_wrap, base_wrap

    # Test 4: Trainable param isolation + K=6 VJP grad check on 32-token slice
    trainable = tree_flatten(wrapper.trainable_parameters())
    n_train = sum(v.size for _, v in trainable)
    for tk, _ in trainable:
        assert not any(bad in tk for bad in ("switch_mlp", "embed_q", "unembed_out", "mlp.gate.")), f"Frozen key trainable: {tk}"
        ok_key = (
            tk.endswith(".lora_A")
            or tk.endswith(".lora_B")
            or tk.startswith("feedback_bridge.")
            or tk.startswith("feat_fuse.")
            or any(tk.startswith(f"mtp.{np}") for np in _NORM_PREFIXES)
        )
        assert ok_key, f"Unexpected trainable key: {tk}"

    fake_seq = {
        "ids_mx": ids_mx[:40],
        "hf": hf[:40],
        "feats": feats[:40] if feats is not None else None,
        "prompt_len": 12,
        "total_len": 40,
        "top32_ids": val0_seq["top32_ids"][:40],
        "top32_logits": val0_seq["top32_logits"][:40],
    }
    anc_init = {
        k: mx.array(v)
        for k, v in trainable
        if any(k.startswith(f"mtp.{np}") for np in _NORM_PREFIXES)
    }
    anc_elems = max(1, sum(v.size for v in anc_init.values()))
    mx.eval(*anc_init.values())

    loss_val, grads = nn.value_and_grad(
        wrapper,
        lambda: loss_on_sequence(
            wrapper,
            embed_w_bf16,
            head_w_bf16.T,
            fake_seq,
            anc_init,
            anc_elems,
            K=len(depth_weights),
            feat_weight=feat_weight,
            depth_weights=depth_weights,
        ),
    )()
    mx.eval(loss_val, grads)
    lv = float(loss_val[0].item())
    g_flat = [(k, g) for k, g in tree_flatten(grads) if g is not None]
    g_norm = math.sqrt(sum(float(mx.sum(g * g).item()) for _, g in g_flat))
    bridge_gn = math.sqrt(sum(float(mx.sum(g * g).item()) for k, g in g_flat if k.startswith("feedback_bridge.")))
    ff_gn = math.sqrt(sum(float(mx.sum(g * g).item()) for k, g in g_flat if k.startswith("feat_fuse.")))
    lora_gn = math.sqrt(sum(float(mx.sum(g * g).item()) for k, g in g_flat if ".lora_" in k))
    assert math.isfinite(lv) and math.isfinite(g_norm) and g_norm > 1e-6 and bridge_gn > 1e-8 and lora_gn > 1e-6
    if wrapper.feat_fuse is not None:
        assert ff_gn > 1e-8, f"feat_fuse grad vanished: {ff_gn}"
    print(
        f"Startup test 4 (Trainable={n_train/1e6:.2f}M, K=6 VJP loss={lv:.4f}, gnorm={g_norm:.3e}, bridge_gn={bridge_gn:.3e}, ff_gn={ff_gn:.3e}, lora_gn={lora_gn:.3e}): PASS",
        flush=True,
    )


def _make_head(head_w_bf16: mx.array) -> nn.Linear:
    h = nn.Linear(head_w_bf16.shape[1], head_w_bf16.shape[0], bias=False)
    h.weight = head_w_bf16
    return h


def _load_dataset_dir(d: Path, *, need_feats: bool, load_splits: set[str]):
    idx_data = json.loads((d / "index.json").read_text())
    has_inline_hf = bool(idx_data.get("has_inline_hfinal", False))
    tr_hf = None
    va_hf = None
    if not has_inline_hf:
        if "train" in load_splits and (d / "train_hfinal.safetensors").exists():
            tr_hf = mx.load(str(d / "train_hfinal.safetensors"))
            mx.eval(*tr_hf.values())
        if "val" in load_splits and (d / "val_hfinal.safetensors").exists():
            va_hf = mx.load(str(d / "val_hfinal.safetensors"))
            mx.eval(*va_hf.values())

    shard_cache: dict[str, dict] = {}
    train_seqs, val_seqs = [], []
    for entry in idx_data["entries"]:
        split = entry["split"]
        if split not in load_splits:
            continue
        sh = entry["shard"]
        if sh not in shard_cache:
            shard_cache[sh] = mx.load(str(d / "shards" / sh))
        st = shard_cache[sh]
        k = entry["key"]
        ids_mx = st[f"{k}.ids"]
        t32_ids = st[f"{k}.top32_ids"]
        t32_lgs = st[f"{k}.top32_logits"]
        feats_mx = st[f"{k}.feats"] if need_feats else None
        if f"{k}.h_final" in st:
            hf_mx = st[f"{k}.h_final"]
        else:
            hf_mx = (tr_hf if split == "train" else va_hf)[k]
        to_ev = [ids_mx, t32_ids, t32_lgs, hf_mx]
        if feats_mx is not None:
            to_ev.append(feats_mx)
        mx.eval(*to_ev)
        rec = {
            "key": k,
            "idx": entry["idx"],
            "domain": entry["domain"],
            "split": split,
            "prompt_len": int(entry["prompt_len"]),
            "total_len": int(entry["total_len"]),
            "ids": ids_mx.tolist(),
            "ids_mx": ids_mx,
            "hf": hf_mx,
            "top32_ids": t32_ids,
            "top32_logits": t32_lgs,
        }
        if feats_mx is not None:
            rec["feats"] = feats_mx
        if split == "val":
            val_seqs.append(rec)
        else:
            train_seqs.append(rec)
    del shard_cache
    mx.clear_cache()
    return train_seqs, val_seqs


def load_full_dataset(data_dir: Path, need_feats: bool = False, extra_data_dirs: list[Path] | None = None):
    sio = mx.load(str(data_dir / "shared_io.safetensors"))
    embed_w_bf16 = sio["word_embeddings.weight"]
    head_w_bf16 = sio["lm_head.weight"]
    mx.eval(embed_w_bf16, head_w_bf16)

    train_seqs, val_seqs = _load_dataset_dir(data_dir, need_feats=need_feats, load_splits={"train", "val"})
    for ed in extra_data_dirs or []:
        ex_tr, _ = _load_dataset_dir(ed, need_feats=need_feats, load_splits={"train"})
        train_seqs.extend(ex_tr)
    print(f"Loaded dataset: {len(train_seqs)} train, {len(val_seqs)} val ({mx.get_active_memory()/1e9:.1f} GB active)", flush=True)
    return train_seqs, val_seqs, embed_w_bf16, head_w_bf16


def _sample_val_anchors(plen: int, T: int, k_max: int = 15, n_blk: int = 5) -> list[int]:
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


def _sim_dyn_step_k(probs: list[float], dyn_th: float, p1_min: float = 0.0, k_cap: int = 7) -> int:
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


def eval_unrolled_val(wrapper: EAGLE3MTPWrapper, val_seqs: list[dict], embed_w_bf16: mx.array, q4_head, step_ms: float = 1.38) -> dict:
    recs = []
    cos1_list = []
    for seq in val_seqs:
        anchors = _sample_val_anchors(seq["prompt_len"], seq["total_len"], k_max=15, n_blk=5)
        hf = seq["hf"]
        T_seq = seq["total_len"]
        hf_eff = hf[:T_seq].astype(mx.float32)
        if wrapper.feat_fuse is not None:
            hf_eff = hf_eff + wrapper.feat_fuse(seq["feats"][:T_seq])
            mx.eval(hf_eff)
        ids_mx = seq["ids_mx"]
        ids = seq["ids"]
        is_med = seq["domain"] == "med"
        for a in anchors:
            mc = KVCache()
            if a > 1:
                _ = _sequential_prefill(wrapper, embed_w_bf16, ids_mx[1:a][None], hf_eff[: a - 1][None], mc)
                mx.eval(mc.state)
            hp = hf_eff[a - 1 : a]
            tok = ids_mx[a : a + 1]
            drafts, probs = [], []
            for j in range(8):
                _, h_fb, lg = sequential_step(wrapper, embed_w_bf16, tok, hp, mc, q4_head)
                lg_f = lg.astype(mx.float32)
                tok = mx.argmax(lg_f, axis=-1).astype(mx.uint32)
                ml = mx.max(lg_f, axis=-1, keepdims=True)
                p_mx = mx.exp(mx.max(lg_f - ml, axis=-1)) / mx.sum(mx.exp(lg_f - ml), axis=-1)
                if j == 0:
                    hf_t = hf[a : a + 1].astype(mx.float32)
                    c1 = mx.sum(h_fb * hf_t) / (mx.linalg.norm(h_fb) * mx.linalg.norm(hf_t) + 1e-8)
                    mx.eval(tok, h_fb, p_mx, c1)
                    cos1_list.append(float(c1.item()))
                else:
                    mx.eval(tok, h_fb, p_mx)
                hp = h_fb
                drafts.append(int(tok.item()))
                probs.append(float(p_mx.item()))
            gold = ids[a + 1 : a + 9]
            pos_ok = [d == g for d, g in zip(drafts, gold)]
            mlen = 0
            for ok in pos_ok:
                if ok:
                    mlen += 1
                else:
                    break
            recs.append({"is_med": is_med, "probs": probs, "match_len": mlen, "pos_ok": pos_ok})

    n = len(recs)
    med = [r for r in recs if r["is_med"]]
    alphas = [round(sum(1 for r in recs if r["pos_ok"][j]) / n, 3) for j in range(8)]
    e_k = {k: round(sum(min(r["match_len"], k) + 1 for r in recs) / n, 3) for k in (2, 4, 6, 8)}
    e_med = {k: round(sum(min(r["match_len"], k) + 1 for r in med) / len(med), 3) for k in (2, 4, 6, 8)}

    best_dyn_tps = 0.0
    best_th = 0.45
    for th in (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
        accs, cycs = [], []
        for r in recs:
            kb = _sim_dyn_step_k(r["probs"], th, p1_min=0.0, k_cap=7)
            accs.append(min(r["match_len"], kb) + 1)
            cycs.append(verify_cost_ms(kb + 1) + kb * step_ms)
        tps = 1e3 * sum(accs) / sum(cycs)
        if tps > best_dyn_tps:
            best_dyn_tps = tps
            best_th = th

    return {
        "p1_acc": alphas[0],
        "alphas": alphas,
        "E_acc": e_k,
        "E_acc_med": e_med,
        "cos1": round(statistics.mean(cos1_list), 4),
        "dyn_best_tok_s": round(best_dyn_tps, 2),
        "dyn_best_th": best_th,
    }


def _cosine_lr(step: int, total_steps: int, max_lr: float, min_lr: float, warmup_frac: float = 0.10) -> float:
    warm = max(1, int(total_steps * warmup_frac))
    if step < warm:
        return max_lr * (step + 1) / warm
    t = (step - warm) / max(1, total_steps - warm)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * min(1.0, t)))


def _add_grads(acc, g):
    if acc is None:
        return g
    return tree_map(lambda a, b: a + b if (a is not None and b is not None) else (a if b is None else b), acc, g)


def _clip_and_scale_grads(acc, scale: float, max_norm: float):
    scaled = tree_map(lambda g: g * scale if g is not None else None, acc)
    sq_sum = mx.array(0.0, dtype=mx.float32)
    for _, g in tree_flatten(scaled):
        if g is not None:
            sq_sum = sq_sum + mx.sum(g * g)
    norm = mx.sqrt(sq_sum)
    clip_coef = mx.minimum(mx.array(1.0, dtype=mx.float32), max_norm / (norm + 1e-6))
    return tree_map(lambda g: g * clip_coef if g is not None else None, scaled), norm


def _parse_depth_weights(spec: str, K: int) -> tuple[float, ...]:
    vals = tuple(float(v.strip()) for v in spec.split(",") if v.strip())
    assert len(vals) == K and all(v > 0.0 for v in vals), f"Invalid depth_weights={spec!r} for K={K}"
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-model", default=os.path.expanduser("~/science/models/lingsante-mlx-mxfp4-bf16"))
    ap.add_argument("--data-dir", default=os.path.expanduser("~/science/data/lingsante_dflash_train"))
    ap.add_argument("--extra-data-dirs", nargs="*", default=[])
    ap.add_argument("--out-dir", default=os.path.expanduser("~/science/models/lingsante-eagle3-mtp"))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=float, default=32.0)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-lr", type=float, default=2e-4)
    ap.add_argument("--min-lr", type=float, default=1e-5)
    ap.add_argument("--feat-weight", type=float, default=0.0)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--use-feat-fuse", action="store_true")
    ap.add_argument("--bridge-v2", action="store_true")
    ap.add_argument("--bridge-rank", type=int, default=1024)
    ap.add_argument("--depth-weights", default=_DEFAULT_DEPTH_WEIGHTS)
    ap.add_argument("--select-by", choices=["dyn", "e8"], default="dyn")
    ap.add_argument("--unit-test-only", action="store_true")
    a = ap.parse_args()

    depth_weights = _parse_depth_weights(a.depth_weights, a.K)
    maybe_set_recommended_wired_limit()
    src_dir = Path(a.src_model).expanduser()
    data_dir = Path(a.data_dir).expanduser()
    extra_dirs = [Path(d).expanduser() for d in a.extra_data_dirs]
    out_dir = Path(a.out_dir).expanduser()

    train_seqs, val_seqs, embed_w_bf16, head_w_bf16 = load_full_dataset(
        data_dir, need_feats=a.use_feat_fuse, extra_data_dirs=extra_dirs
    )
    raw_weights = mx.load(str(src_dir / "mtp" / "model-mtp.safetensors"))

    base_mtp, args = build_mtp_float32(src_dir, raw_weights)
    train_mtp, _ = build_mtp_float32(src_dir, raw_weights)
    wrapper = wrap_mtp_trainable(
        train_mtp,
        args,
        r=a.lora_r,
        alpha=a.lora_alpha,
        use_feat_fuse=a.use_feat_fuse,
        bridge_v2=a.bridge_v2,
        bridge_rank=a.bridge_rank,
    )
    mx.eval(base_mtp.parameters(), wrapper.parameters())

    run_startup_tests(
        base_mtp,
        wrapper,
        raw_weights,
        embed_w_bf16,
        head_w_bf16,
        val_seqs[0],
        out_dir,
        feat_weight=a.feat_weight,
        depth_weights=depth_weights,
    )
    del base_mtp
    mx.clear_cache()
    if a.unit_test_only:
        return 0

    head_t_bf16 = head_w_bf16.T
    q4_head = nn.QuantizedLinear.from_linear(_make_head(head_w_bf16), group_size=64, bits=4, mode="affine")
    mx.eval(q4_head.parameters())

    anchor_init = {
        k: mx.array(v)
        for k, v in tree_flatten(wrapper.trainable_parameters())
        if any(k.startswith(f"mtp.{np}") for np in _NORM_PREFIXES)
    }
    anchor_elems = max(1, sum(v.size for v in anchor_init.values()))
    mx.eval(*anchor_init.values())

    def _sel_score(vr_d: dict) -> tuple[float, float]:
        if a.select_by == "dyn":
            return (float(vr_d["dyn_best_tok_s"]), float(vr_d["E_acc"][8]))
        return (float(vr_d["E_acc"][8]), float(vr_d["dyn_best_tok_s"]))

    vr0 = eval_unrolled_val(wrapper, val_seqs, embed_w_bf16, q4_head)
    print(
        f"  Val Ep  0: p1={vr0['p1_acc']:.3f} E2/4/6/8={vr0['E_acc'][2]:.3f}/{vr0['E_acc'][4]:.3f}/{vr0['E_acc'][6]:.3f}/{vr0['E_acc'][8]:.3f} "
        f"(med={vr0['E_acc_med'][4]:.3f}/{vr0['E_acc_med'][8]:.3f}) cos1={vr0['cos1']:.4f} dyn={vr0['dyn_best_tok_s']:.2f} (th={vr0['dyn_best_th']:.2f}) alphas={vr0['alphas']}",
        flush=True,
    )
    ep0_k4 = vr0["E_acc"][4]
    best_score = _sel_score(vr0)
    best_vr = vr0
    best_state = [(k, mx.array(v)) for k, v in tree_flatten(wrapper.trainable_parameters())]
    mx.eval(*[v for _, v in best_state])

    def _loss_fn(seq):
        return loss_on_sequence(
            wrapper,
            embed_w_bf16,
            head_t_bf16,
            seq,
            anchor_init,
            anchor_elems,
            K=a.K,
            feat_weight=a.feat_weight,
            depth_weights=depth_weights,
        )

    vg = nn.value_and_grad(wrapper, _loss_fn)
    opt = optim.AdamW(learning_rate=a.max_lr, weight_decay=a.weight_decay)
    steps_per_ep = math.ceil(len(train_seqs) / a.grad_accum)
    total_opt_steps = a.epochs * steps_per_ep
    opt_step = 0
    rng = random.Random(42)

    for ep in range(1, a.epochs + 1):
        t_ep0 = time.perf_counter()
        order = list(range(len(train_seqs)))
        rng.shuffle(order)
        ep_loss, ep_kl1, ep_feat1, ep_anc, ep_n = 0.0, 0.0, 0.0, 0.0, 0
        grad_acc = None
        acc_cnt = 0

        for idx_i, s_idx in enumerate(order):
            (loss, (kl1, f1, anc_v)), grads = vg(train_seqs[s_idx])
            grad_acc = _add_grads(grad_acc, grads)
            acc_cnt += 1
            mx.eval(loss, kl1, f1, anc_v, grad_acc)
            lv = float(loss.item())
            assert math.isfinite(lv), f"NaN/Inf loss at ep={ep} idx={s_idx}"
            ep_loss += lv
            ep_kl1 += float(kl1.item())
            ep_feat1 += float(f1.item())
            ep_anc += float(anc_v.item())
            ep_n += 1

            if acc_cnt >= a.grad_accum or (idx_i == len(order) - 1 and acc_cnt > 0):
                opt.learning_rate = _cosine_lr(opt_step, total_opt_steps, a.max_lr, a.min_lr, warmup_frac=0.10)
                clipped, _ = _clip_and_scale_grads(grad_acc, 1.0 / acc_cnt, 1.0)
                opt.update(wrapper, clipped)
                mx.eval(wrapper.trainable_parameters(), opt.state)
                grad_acc = None
                acc_cnt = 0
                opt_step += 1

        vr = eval_unrolled_val(wrapper, val_seqs, embed_w_bf16, q4_head)
        tag = ""
        cur_score = _sel_score(vr)
        if cur_score > best_score:
            best_score = cur_score
            best_vr = vr
            best_state = [(k, mx.array(v)) for k, v in tree_flatten(wrapper.trainable_parameters())]
            mx.eval(*[v for _, v in best_state])
            save_eagle3_checkpoint(wrapper, raw_weights, out_dir)
            tag = " [BEST*]"
        print(
            f"  Ep {ep:2d}/{a.epochs} ({time.perf_counter()-t_ep0:.1f}s) L={ep_loss/ep_n:.4f} kl1={ep_kl1/ep_n:.4f} || "
            f"p1={vr['p1_acc']:.3f} E2/4/6/8={vr['E_acc'][2]:.3f}/{vr['E_acc'][4]:.3f}/{vr['E_acc'][6]:.3f}/{vr['E_acc'][8]:.3f} "
            f"(med={vr['E_acc_med'][4]:.3f}/{vr['E_acc_med'][8]:.3f}) cos1={vr['cos1']:.4f} dyn={vr['dyn_best_tok_s']:.2f} (th={vr['dyn_best_th']:.2f}) "
            f"alphas={vr['alphas']}{tag}",
            flush=True,
        )

        if ep == 3 and vr["E_acc"][4] < ep0_k4 + 0.20:
            print(
                f"EARLY STOP AT EPOCH 3: val E_acc(K=4)={vr['E_acc'][4]:.3f} < {ep0_k4 + 0.20:.3f} (Ep0 + 0.20); handing back per plan.",
                flush=True,
            )
            return 2

    wrapper.update(tree_unflatten(best_state))
    save_eagle3_checkpoint(wrapper, raw_weights, out_dir)
    print(
        f"Saved best EAGLE-3 MTP checkpoint to {out_dir} "
        f"(select={a.select_by}, best_dyn={best_vr['dyn_best_tok_s']:.2f}, E_acc[K=8]={best_vr['E_acc'][8]:.3f}, p1={best_vr['p1_acc']:.3f})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
