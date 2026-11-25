"""Train LingSante-DFlash2 (LoRA r=32 + anchored norms/convs + Stage 2 CandidateSelector).

All runs stay on 512S1 with the 300 GB target unloaded (~15 GB RAM). Trainable weights run in
float32 for clean AdamW convergence and are merged & exported in bfloat16 compatible with
DFlash2DraftModel.from_target(strict=True).
"""
import argparse
import json
import math
import os
import random
import shutil
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
from mlx_lm.models.bailing_moe_v3.dflash2 import DFlash2DraftModel  # noqa: E402
from mlx_lm.utils import maybe_set_recommended_wired_limit  # noqa: E402


class LoRALinear(nn.Module):
    """Low-rank adapter around a frozen float32 base Linear weight."""

    def __init__(self, linear: nn.Linear, r: int = 32, alpha: float = 32.0):
        super().__init__()
        out_dim, in_dim = linear.weight.shape
        self.weight = linear.weight
        self.r = int(r)
        self.scale = float(alpha) / float(r)
        self.lora_A = mx.random.normal((r, in_dim), dtype=mx.float32) * (1.0 / math.sqrt(in_dim))
        self.lora_B = mx.zeros((out_dim, r), dtype=mx.float32)
        self.freeze(keys=["weight"], recurse=False)

    def __call__(self, x: mx.array) -> mx.array:
        return (x @ self.weight.T) + ((x @ self.lora_A.T) @ self.lora_B.T) * self.scale

    def to_linear(self) -> nn.Linear:
        out_dim, in_dim = self.weight.shape
        lin = nn.Linear(in_dim, out_dim, bias=False)
        lin.weight = (self.weight + self.scale * (self.lora_B @ self.lora_A)).astype(self.weight.dtype)
        return lin


def wrap_lora(model: DFlash2DraftModel, r: int = 32, alpha: float = 32.0):
    model.fc = LoRALinear(model.fc, r=r, alpha=alpha)
    for lyr in model.layers:
        sa = lyr.self_attn
        sa.q_proj = LoRALinear(sa.q_proj, r=r, alpha=alpha)
        sa.k_proj = LoRALinear(sa.k_proj, r=r, alpha=alpha)
        sa.v_proj = LoRALinear(sa.v_proj, r=r, alpha=alpha)
        sa.o_proj = LoRALinear(sa.o_proj, r=r, alpha=alpha)
        mlp = lyr.mlp
        mlp.gate_proj = LoRALinear(mlp.gate_proj, r=r, alpha=alpha)
        mlp.up_proj = LoRALinear(mlp.up_proj, r=r, alpha=alpha)
        mlp.down_proj = LoRALinear(mlp.down_proj, r=r, alpha=alpha)


def merge_lora_to_dflash2(model: DFlash2DraftModel):
    if isinstance(model.fc, LoRALinear):
        model.fc = model.fc.to_linear()
    for lyr in model.layers:
        sa = lyr.self_attn
        for attr in ("q_proj", "k_proj", "v_proj", "o_proj"):
            m = getattr(sa, attr)
            if isinstance(m, LoRALinear):
                setattr(sa, attr, m.to_linear())
        mlp = lyr.mlp
        for attr in ("gate_proj", "up_proj", "down_proj"):
            m = getattr(mlp, attr)
            if isinstance(m, LoRALinear):
                setattr(mlp, attr, m.to_linear())


def merged_bf16_state_dict(model: DFlash2DraftModel) -> dict[str, mx.array]:
    mods = dict(model.named_modules())
    flat = dict(tree_flatten(model.parameters()))
    out = {}
    handled_prefixes = set()
    for name, mod in mods.items():
        if isinstance(mod, LoRALinear):
            w_merged = mod.weight + mod.scale * (mod.lora_B @ mod.lora_A)
            out[f"{name}.weight"] = w_merged.astype(mx.bfloat16)
            handled_prefixes.add(name + ".")
    for k, v in flat.items():
        if any(k.startswith(p) for p in handled_prefixes):
            continue
        out[k] = v.astype(mx.bfloat16)
    mx.eval(*out.values())
    return out


def save_dflash2_bf16(model: DFlash2DraftModel, src_dir: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    params_bf16 = merged_bf16_state_dict(model)
    mx.save_safetensors(str(dst_dir / "model.safetensors"), params_bf16)
    shutil.copy2(src_dir / "config.json", dst_dir / "config.json")


def apply_rope_pos(
    x: mx.array, pos_ids: mx.array, head_dim: int = 128, base: float = 6000000.0
) -> mx.array:
    """Exact traditional=False RoPE supporting arbitrary non-contiguous `pos_ids`."""
    half = head_dim // 2
    inv_freq = mx.exp(
        -mx.arange(0, head_dim, 2, dtype=mx.float32) * (math.log(float(base)) / head_dim)
    )
    pos = pos_ids.astype(mx.float32)
    if pos.ndim == 1:
        freqs = pos[:, None] * inv_freq[None, :]
        cos = mx.cos(freqs)[None, None, :, :]
        sin = mx.sin(freqs)[None, None, :, :]
    else:
        freqs = pos[:, :, None] * inv_freq[None, None, :]
        cos = mx.cos(freqs)[:, None, :, :]
        sin = mx.sin(freqs)[:, None, :, :]
    xf = x.astype(mx.float32)
    x1 = xf[..., :half]
    x2 = xf[..., half:]
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return mx.concatenate([y1, y2], axis=-1).astype(x.dtype)


def _layer_attn_forward(
    sa,
    hn: mx.array,
    k_ctx: mx.array,
    v_ctx: mx.array,
    noise_pos: mx.array,
    attn_mask: mx.array,
    *,
    use_sdpa: bool,
) -> mx.array:
    B_dim, M, _ = hn.shape
    q = sa.q_norm(sa.q_proj(hn).reshape(B_dim, M, sa.num_heads, sa.head_dim)).transpose(0, 2, 1, 3)
    q = apply_rope_pos(q, noise_pos, sa.head_dim, sa.rope.base)

    k_noise = sa.k_norm(sa.k_proj(hn).reshape(B_dim, M, sa.num_kv_heads, sa.head_dim)).transpose(0, 2, 1, 3)
    k_noise = apply_rope_pos(k_noise, noise_pos, sa.head_dim, sa.rope.base)
    v_noise = sa.v_proj(hn).reshape(B_dim, M, sa.num_kv_heads, sa.head_dim).transpose(0, 2, 1, 3)

    k_all = mx.concatenate([k_ctx, k_noise], axis=2)
    v_all = mx.concatenate([v_ctx, v_noise], axis=2)
    if use_sdpa:
        out = mx.fast.scaled_dot_product_attention(q, k_all, v_all, scale=sa.scale, mask=attn_mask)
    else:
        reps = sa.num_heads // sa.num_kv_heads
        k_exp = mx.repeat(k_all, reps, axis=1)
        v_exp = mx.repeat(v_all, reps, axis=1)
        scores = (q * sa.scale) @ k_exp.swapaxes(-1, -2) + attn_mask
        probs = mx.softmax(scores, axis=-1)
        out = probs @ v_exp
    return sa.o_proj(out.transpose(0, 2, 1, 3).reshape(B_dim, M, -1))


def multiblock_forward(
    model: DFlash2DraftModel,
    embed_w_bf16: mx.array,
    feats: mx.array,
    noise_toks: mx.array,
    noise_pos: mx.array,
    attn_mask: mx.array,
    *,
    block_size: int,
    use_sdpa: bool = True,
) -> mx.array:
    """Shared-context multi-block forward matching DFlash2 + KVCache semantics identically."""
    T = int(feats.shape[0])
    ctx_pos = mx.arange(T, dtype=mx.int32)
    th = model.project_features(feats.astype(mx.float32)[None])  # (1, T, 4096)
    h = embed_w_bf16[noise_toks].astype(mx.float32)[None]  # (1, M, 4096)

    for layer in model.layers:
        sa = layer.self_attn
        k_ctx = sa.k_norm(sa.k_proj(th).reshape(1, T, sa.num_kv_heads, sa.head_dim)).transpose(0, 2, 1, 3)
        k_ctx = apply_rope_pos(k_ctx, ctx_pos, sa.head_dim, sa.rope.base)
        v_ctx = sa.v_proj(th).reshape(1, T, sa.num_kv_heads, sa.head_dim).transpose(0, 2, 1, 3)

        residual = h
        hn = layer.input_layernorm(h)
        hn, attn_k1 = layer.attention_conv.prepare(hn, block_size=block_size)
        hn = _layer_attn_forward(sa, hn, k_ctx, v_ctx, noise_pos, attn_mask, use_sdpa=use_sdpa)
        hn = layer.attention_conv.finish(hn, attn_k1, block_size=block_size)
        h = residual + hn

        residual = h
        hn = layer.post_attention_layernorm(h)
        hn, mlp_k1 = layer.mlp_conv.prepare(hn, block_size=block_size)
        hn = layer.mlp(hn)
        hn = layer.mlp_conv.finish(hn, mlp_k1, block_size=block_size)
        h = residual + hn

    return model.norm(h)


def build_block_batch(seq: dict, anchors: list[int], block_size: int, mask_token_id: int):
    T = seq["total_len"]
    ids = seq["ids"]
    t32_ids = seq["top32_ids"]
    t32_lgs = seq["top32_logits"]
    N_blk = len(anchors)
    M = N_blk * block_size

    nt_list = []
    np_list = []
    tgt_list = []
    prev_idx_list = []
    mask_rows = []
    for m, a in enumerate(anchors):
        nt_list.append(int(ids[a]))
        nt_list.extend([mask_token_id] * (block_size - 1))
        for j in range(block_size):
            np_list.append(a + j)
        for j in range(1, block_size):
            tgt_list.append(int(ids[a + j]))
            prev_idx_list.append(a + j - 1)
        ctx_row = [0.0] * a + [-1e9] * (T - a)
        noi_row = [-1e9] * (m * block_size) + [0.0] * block_size + [-1e9] * ((N_blk - m - 1) * block_size)
        mask_rows.extend([ctx_row + noi_row] * block_size)

    noise_toks = mx.array(nt_list, dtype=mx.uint32)
    noise_pos = mx.array(np_list, dtype=mx.int32)
    attn_mask = mx.array(mask_rows, dtype=mx.float32).reshape(1, 1, M, T + M)
    tgt_ids = mx.array(tgt_list, dtype=mx.uint32)
    prev_idx = mx.array(prev_idx_list, dtype=mx.int32)
    prev_ids = seq["ids_mx"][prev_idx]
    b_t32_ids = t32_ids[prev_idx]
    b_t32_probs = mx.softmax(t32_lgs[prev_idx].astype(mx.float32), axis=-1)
    return noise_toks, noise_pos, attn_mask, tgt_ids, prev_ids, b_t32_ids, b_t32_probs


def _freeze_to_backbone(model: DFlash2DraftModel):
    model.unfreeze(recurse=True)
    model.candidate_selector.freeze(recurse=True)
    for _, mod in model.named_modules():
        if isinstance(mod, LoRALinear):
            mod.freeze(keys=["weight"], recurse=False)


def _freeze_to_selector(model: DFlash2DraftModel):
    model.freeze(recurse=True)
    model.candidate_selector.unfreeze(recurse=True)


def stage1_bounded_loss(
    logits: mx.array,
    tgt_ids: mx.array,
    b_t32_ids: mx.array,
    b_t32_probs: mx.array,
    N_a: int,
    K: int,
):
    """Differentiable expected acceptance length + stop-grad frontier log1p(CE) and top-32 TV."""
    lse1 = mx.logsumexp(logits, axis=-1, keepdims=True)
    lse2 = mx.logsumexp(logits * 0.5, axis=-1, keepdims=True)
    z_tgt = mx.take_along_axis(logits, tgt_ids[:, None], axis=-1).squeeze(-1)
    ce_i = lse1.squeeze(-1) - z_tgt
    q1 = mx.exp(-ce_i).reshape(N_a, K)
    q2 = mx.exp(z_tgt * 0.5 - lse2.squeeze(-1)).reshape(N_a, K)
    p_stu_32 = mx.exp(mx.take_along_axis(logits, b_t32_ids, axis=-1) - lse1)
    tv_32 = (1.0 - mx.sum(mx.minimum(p_stu_32, b_t32_probs), axis=-1)).reshape(N_a, K)

    s1 = mx.ones((N_a,), dtype=mx.float32)
    s2 = mx.ones((N_a,), dtype=mx.float32)
    e1 = mx.zeros((N_a,), dtype=mx.float32)
    e2 = mx.zeros((N_a,), dtype=mx.float32)
    for j in range(K):
        s1 = s1 * q1[:, j]
        s2 = s2 * q2[:, j]
        e1 = e1 + s1
        e2 = e2 + s2
    e_acc_1 = mx.mean(e1)
    e_acc_2 = mx.mean(e2)

    match = mx.stop_gradient((mx.argmax(logits, axis=-1) == tgt_ids).astype(mx.float32).reshape(N_a, K))
    pw_cols = []
    pw = mx.ones((N_a,), dtype=mx.float32)
    for j in range(K):
        pw_cols.append(pw)
        pw = pw * (0.15 + 0.85 * match[:, j])
    pref_w = mx.stop_gradient(mx.stack(pw_cols, axis=-1))
    pw_norm = mx.maximum(mx.sum(pref_w), 1.0)
    front_ce = mx.sum(pref_w * mx.log1p(mx.maximum(ce_i.reshape(N_a, K), 0.0))) / pw_norm
    front_tv = mx.sum(pref_w * tv_32) / pw_norm

    tot = -e_acc_1 - 0.5 * e_acc_2 + 0.40 * front_ce + 0.40 * front_tv
    return tot, (e_acc_1, front_ce, front_tv)


def run_startup_tests(
    base_model: DFlash2DraftModel,
    lora_model: DFlash2DraftModel,
    cfg: dict,
    embed_w_bf16: mx.array,
    head_w_bf16: mx.array,
):
    """Verify RoPE, single-block KV equiv, LoRA Epoch-0 identity, merged bf16 reload, and frozen weight grads."""
    mx.random.seed(7)
    x = mx.random.normal((1, 8, 12, 128), dtype=mx.float32)
    ref_rope = base_model.layers[0].self_attn.rope
    for off in (0, 17, 250):
        got = apply_rope_pos(x, mx.arange(off, off + 12, dtype=mx.int32), 128, ref_rope.base)
        ref = ref_rope(x, offset=off)
        mx.eval(got, ref)
        assert float(mx.max(mx.abs(got - ref)).item()) < 1e-4
    print("Startup test 1 (apply_rope_pos vs nn.RoPE): PASS", flush=True)

    T, a0, B = 64, 40, 8
    feats = (mx.random.normal((T, len(base_model.target_layer_ids) * 4096)) * 0.25).astype(mx.float32)
    ids = mx.random.randint(10, 5000, (T + B,), dtype=mx.uint32)
    fake_seq = {
        "total_len": T,
        "ids": ids[:T].tolist(),
        "ids_mx": ids[:T],
        "top32_ids": mx.zeros((T, 32), dtype=mx.uint32),
        "top32_logits": mx.zeros((T, 32), dtype=mx.float16),
    }
    nt, np_, am, _, _, _, _ = build_block_batch(fake_seq, [a0], B, base_model.mask_token_id)

    base_model._refs_box.d["embed"] = lambda t: embed_w_bf16[t].astype(mx.float32)
    caches = base_model.make_cache()
    base_model.prefill_ctx(feats[:a0][None], caches)
    mx.eval(*[c.state for c in caches])
    ref_h = base_model(nt[None], None, caches)

    got_base = multiblock_forward(base_model, embed_w_bf16, feats, nt, np_, am, block_size=B, use_sdpa=True)
    got_lora0 = multiblock_forward(lora_model, embed_w_bf16, feats, nt, np_, am, block_size=B, use_sdpa=True)
    mx.eval(ref_h, got_base, got_lora0)
    d_kv = float(mx.max(mx.abs(got_base - ref_h)).item())
    d_lora = float(mx.max(mx.abs(got_lora0 - got_base)).item())
    assert d_kv < 1e-2, f"Single-block KV equiv diff={d_kv}"
    assert d_lora < 1e-5, f"LoRA Epoch-0 mismatch diff={d_lora}"
    print(f"Startup test 2 (KV equiv={d_kv:.2e}, LoRA Ep0 diff={d_lora:.2e}): PASS", flush=True)

    # Merged bf16 state compatibility check
    bf_sd = merged_bf16_state_dict(lora_model)
    m_rel = DFlash2DraftModel(cfg)
    m_rel.load_weights(list(bf_sd.items()), strict=True)
    m_rel.update(tree_map(lambda x: x.astype(mx.float32), m_rel.parameters()))
    got_rel = multiblock_forward(m_rel, embed_w_bf16, feats, nt, np_, am, block_size=B, use_sdpa=True)
    mx.eval(got_rel)
    d_rel = float(mx.max(mx.abs(got_rel - got_base)).item())
    assert d_rel < 5e-2, f"Merged bf16 reload diff={d_rel}"
    print(f"Startup test 3 (Merged bf16 strict=True reload diff={d_rel:.2e}): PASS", flush=True)
    del m_rel, bf_sd

    # Verify LoRA grad isolation + SDPA VJP vs manual GQA
    _freeze_to_backbone(lora_model)
    trainable = tree_flatten(lora_model.trainable_parameters())
    n_trainable = sum(v.size for _, v in trainable)
    for k, _ in trainable:
        assert "candidate_selector" not in k, f"Selector unfrozen: {k}"
        is_lora_base = k == "fc.weight" or any(
            k.endswith(f".{p}.weight")
            for p in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
        )
        assert not is_lora_base, f"Base linear weight unfrozen: {k}"

    feats_s = feats[:24]
    fake_s = {
        "total_len": 24,
        "ids": ids[:24].tolist(),
        "ids_mx": ids[:24],
        "top32_ids": mx.zeros((24, 32), dtype=mx.uint32),
        "top32_logits": mx.zeros((24, 32), dtype=mx.float16),
    }
    nt_s, np_s, am_s, _, _, _, _ = build_block_batch(fake_s, [12, 16], 4, lora_model.mask_token_id)

    def _loss(sdpa_flag):
        h = multiblock_forward(lora_model, embed_w_bf16, feats_s, nt_s, np_s, am_s, block_size=4, use_sdpa=sdpa_flag)
        return mx.mean(h * h)

    _, g_man = nn.value_and_grad(lora_model, lambda: _loss(False))()
    _, g_sdpa = nn.value_and_grad(lora_model, lambda: _loss(True))()
    mx.eval(g_man, g_sdpa)
    flat_m = dict(tree_flatten(g_man))
    flat_s = dict(tree_flatten(g_sdpa))
    max_gdiff = max(
        float(mx.max(mx.abs(flat_m[k] - flat_s[k])).item())
        for k in flat_m
        if k in flat_s and flat_s[k] is not None
    )
    assert max_gdiff < 1e-2
    print(f"Startup test 4 (LoRA trainable={n_trainable/1e6:.2f}M, SDPA VJP grad_diff={max_gdiff:.2e}): PASS", flush=True)

    # Verify Stage 1 bounded objective VJP end-to-end through frozen lm_head
    head_t_bf16 = head_w_bf16.T
    nt_5, np_5, am_5, tgt_5, _, b_ids_5, b_probs_5 = build_block_batch(fake_s, [12, 16], 4, lora_model.mask_token_id)

    def _loss_bnd():
        h = multiblock_forward(lora_model, embed_w_bf16, feats_s, nt_5, np_5, am_5, block_size=4, use_sdpa=True)
        hp = h.reshape(2, 4, -1)[:, 1:, :].reshape(6, -1)
        lg = (hp.astype(mx.bfloat16) @ head_t_bf16).astype(mx.float32)
        tot, _ = stage1_bounded_loss(lg, tgt_5, b_ids_5, b_probs_5, 2, 3)
        return tot

    l5, g5 = nn.value_and_grad(lora_model, _loss_bnd)()
    mx.eval(l5, g5)
    l5_val = float(l5.item())
    g5_sq = sum(float(mx.sum(g * g).item()) for _, g in tree_flatten(g5) if g is not None)
    g5_norm = math.sqrt(g5_sq)
    assert math.isfinite(l5_val) and math.isfinite(g5_norm) and g5_norm > 1e-6, f"Bad VJP l5={l5_val} gn={g5_norm}"
    print(f"Startup test 5 (Bounded loss VJP finite loss={l5_val:.4f}, grad_norm={g5_norm:.4e}): PASS", flush=True)
    return True


def load_dataset(data_dir: Path):
    idx_data = json.loads((data_dir / "index.json").read_text())
    sio = mx.load(str(data_dir / "shared_io.safetensors"))
    embed_w_bf16 = sio["word_embeddings.weight"]
    head_w_bf16 = sio["lm_head.weight"]
    mx.eval(embed_w_bf16, head_w_bf16)

    shard_cache: dict[str, dict] = {}
    train_seqs, val_seqs = [], []
    for entry in idx_data["entries"]:
        sh = entry["shard"]
        if sh not in shard_cache:
            shard_cache[sh] = mx.load(str(data_dir / "shards" / sh))
        st = shard_cache[sh]
        k = entry["key"]
        ids_mx = st[f"{k}.ids"]
        feats = st[f"{k}.feats"]
        t32_ids = st[f"{k}.top32_ids"]
        t32_lgs = st[f"{k}.top32_logits"]
        mx.eval(ids_mx, feats, t32_ids, t32_lgs)
        rec = {
            "idx": entry["idx"],
            "domain": entry["domain"],
            "split": entry["split"],
            "prompt_len": int(entry["prompt_len"]),
            "total_len": int(entry["total_len"]),
            "ids": ids_mx.tolist(),
            "ids_mx": ids_mx,
            "feats": feats,
            "top32_ids": t32_ids,
            "top32_logits": t32_lgs,
        }
        if entry["split"] == "val":
            val_seqs.append(rec)
        else:
            train_seqs.append(rec)
    del shard_cache
    mx.clear_cache()
    n_med_v = sum(1 for s in val_seqs if s["domain"] == "med")
    print(f"Loaded dataset: {len(train_seqs)} train, {len(val_seqs)} val (med={n_med_v}) ({mx.get_active_memory()/1e9:.1f} GB active)", flush=True)
    return train_seqs, val_seqs, embed_w_bf16, head_w_bf16


def _sample_anchors(plen: int, T: int, block_size: int, n_blk: int, rng: random.Random | None) -> list[int]:
    lo = plen
    hi = T - block_size
    if hi < lo:
        return []
    span = hi - lo + 1
    out = []
    for m in range(n_blk):
        b0 = lo + int(m * span / n_blk)
        b1 = lo + int((m + 1) * span / n_blk) - 1
        b1 = max(b0, min(hi, b1))
        out.append((b0 + b1) // 2 if rng is None else rng.randint(b0, b1))
    return out


def eval_validation(
    model: DFlash2DraftModel,
    val_seqs: list[dict],
    embed_w_bf16: mx.array,
    head_w_bf16: mx.array,
    *,
    block_sizes=(8, 10, 12, 16),
    n_blk: int = 20,
    use_sdpa: bool = True,
    with_selector: bool = False,
):
    results = {}
    head_t_bf16 = head_w_bf16.T
    for B in block_sizes:
        K = B - 1
        hits_all, blocks_all, sum_am_all, sum_sel_all, sum_lmax_all = [0] * K, 0, 0, 0, 0.0
        hits_med, blocks_med, sum_am_med, sum_sel_med = [0] * K, 0, 0, 0
        for seq in val_seqs:
            is_med = seq["domain"] == "med"
            anchors = _sample_anchors(seq["prompt_len"], seq["total_len"], B, n_blk, None)
            if not anchors:
                continue
            nt, np_, am, tgt_ids, _, _, _ = build_block_batch(seq, anchors, B, model.mask_token_id)
            h_out = multiblock_forward(model, embed_w_bf16, seq["feats"], nt, np_, am, block_size=B, use_sdpa=use_sdpa)
            h_pred = h_out.reshape(len(anchors), B, -1)[:, 1:, :]
            logits = (h_pred.astype(mx.bfloat16) @ head_t_bf16).astype(mx.float32)
            preds_am = mx.argmax(logits, axis=-1)
            lmax_sum = mx.sum(mx.max(logits, axis=-1))
            t_mat = tgt_ids.reshape(len(anchors), K)
            mx.eval(preds_am, t_mat, lmax_sum)
            sum_lmax_all += float(lmax_sum.item())
            pm = preds_am.tolist()
            tm = t_mat.tolist()
            for row_p, row_t in zip(pm, tm):
                blocks_all += 1
                if is_med:
                    blocks_med += 1
                acc = 1
                ok_prefix = True
                for j in range(K):
                    if row_p[j] == row_t[j]:
                        hits_all[j] += 1
                        if is_med:
                            hits_med[j] += 1
                        if ok_prefix:
                            acc += 1
                    else:
                        ok_prefix = False
                sum_am_all += acc
                if is_med:
                    sum_am_med += acc

            if with_selector:
                top_k = model.selector_top_k
                idx = mx.argpartition(-logits, kth=top_k - 1, axis=-1)[..., :top_k]
                vals = mx.take_along_axis(logits, idx, axis=-1)
                ord_i = mx.argsort(-vals, axis=-1)
                cand_ids = mx.take_along_axis(idx, ord_i, axis=-1)
                u_topk = mx.take_along_axis(vals, ord_i, axis=-1)
                anchor_toks = mx.array([seq["ids"][a] for a in anchors], dtype=mx.uint32)
                preds_sel = model.candidate_selector.greedy_path(
                    candidate_ids=cand_ids,
                    unary_logits=u_topk,
                    hidden_states=h_pred,
                    anchor_token_id=anchor_toks,
                )
                mx.eval(preds_sel)
                for row_s, row_t in zip(preds_sel.tolist(), tm):
                    acc = 1
                    for j in range(K):
                        if row_s[j] == row_t[j]:
                            acc += 1
                        else:
                            break
                    sum_sel_all += acc
                    if is_med:
                        sum_sel_med += acc

        rec = {
            "E_acc": round(sum_am_all / max(1, blocks_all), 3),
            "pos_acc": [round(h / max(1, blocks_all), 3) for h in hits_all],
            "med_E_acc": round(sum_am_med / max(1, blocks_med), 3),
            "med_pos_acc": [round(h / max(1, blocks_med), 3) for h in hits_med],
            "logit_max_mean": round(sum_lmax_all / max(1, blocks_all * K), 2),
        }
        if with_selector:
            rec["E_acc_sel"] = round(sum_sel_all / max(1, blocks_all), 3)
            rec["med_E_acc_sel"] = round(sum_sel_med / max(1, blocks_med), 3)
        results[B] = rec
    return results


def _pos_weights(K: int, tau: float = 4.0) -> mx.array:
    w = mx.exp(-mx.arange(K, dtype=mx.float32) / tau)
    return w / mx.mean(w)


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
    flat = tree_flatten(scaled)
    sq_sum = mx.array(0.0, dtype=mx.float32)
    for _, g in flat:
        if g is not None:
            sq_sum = sq_sum + mx.sum(g * g)
    norm = mx.sqrt(sq_sum)
    clip_coef = mx.minimum(mx.array(1.0, dtype=mx.float32), max_norm / (norm + 1e-6))
    return tree_map(lambda g: g * clip_coef if g is not None else None, scaled), norm


def _fmt_val_row(vr: dict) -> str:
    all_s = " ".join(f"B{B}={vr[B]['E_acc']:.3f}(p1={vr[B]['pos_acc'][0]:.3f})" for B in (8, 10, 12, 16))
    med_s = " ".join(f"B{B}={vr[B]['med_E_acc']:.3f}(p1={vr[B]['med_pos_acc'][0]:.3f})" for B in (8, 12))
    return f"ALL[{all_s}] MED[{med_s}] lmax={vr[12]['logit_max_mean']:.2f}"


def train_stage1_lora(
    model: DFlash2DraftModel,
    train_seqs: list[dict],
    val_seqs: list[dict],
    embed_w_bf16: mx.array,
    head_w_bf16: mx.array,
    src_dir: Path,
    dst_dir: Path,
    *,
    epochs: int,
    n_blk: int,
    grad_accum: int,
    max_lr: float,
    min_lr: float,
    use_sdpa: bool,
):
    print("\n=== Stage 1: LoRA (r=32, a=32) + Anchored Norms/Convs (50% B=8, 50% B=12, bounded E_acc_soft + frontier) ===", flush=True)
    _freeze_to_backbone(model)
    head_t_bf16 = head_w_bf16.T
    rng = random.Random(42)

    # Snapshot initial non-LoRA trainable weights (norms + convs) for L2 anchor-to-init regularization
    anchor_init = {
        k: mx.array(v)
        for k, v in tree_flatten(model.trainable_parameters())
        if not k.endswith(".lora_A") and not k.endswith(".lora_B")
    }
    mx.eval(*anchor_init.values())
    anchor_elems = max(1, sum(v.size for v in anchor_init.values()))

    val0 = eval_validation(model, val_seqs, embed_w_bf16, head_w_bf16, use_sdpa=use_sdpa, with_selector=False)
    print(f"  Val Ep  0: {_fmt_val_row(val0)}", flush=True)

    best_score = val0[12]["E_acc"]
    best_train_state = [(k, mx.array(v)) for k, v in tree_flatten(model.trainable_parameters())]
    mx.eval(*[v for _, v in best_train_state])

    steps_per_epoch = math.ceil(len(train_seqs) / grad_accum)
    total_opt_steps = epochs * steps_per_epoch
    opt = optim.AdamW(learning_rate=max_lr, weight_decay=0.01)

    def loss_seq(seq_feats, nt, np_, am, tgt_ids, b_t32_ids, b_t32_probs, B):
        N_a = int(nt.shape[0]) // B
        K = B - 1
        h_out = multiblock_forward(model, embed_w_bf16, seq_feats, nt, np_, am, block_size=B, use_sdpa=use_sdpa)
        h_pred = h_out.reshape(N_a, B, -1)[:, 1:, :].reshape(N_a * K, -1)
        logits = (h_pred.astype(mx.bfloat16) @ head_t_bf16).astype(mx.float32)
        obj_loss, (e1_v, fce_v, ftv_v) = stage1_bounded_loss(logits, tgt_ids, b_t32_ids, b_t32_probs, N_a, K)

        cur_map = dict(tree_flatten(model.trainable_parameters()))
        anc_sq = mx.array(0.0, dtype=mx.float32)
        for k, w0 in anchor_init.items():
            d = cur_map[k] - w0
            anc_sq = anc_sq + mx.sum(d * d)
        anc_pen = 0.01 * (anc_sq / anchor_elems)

        tot = obj_loss + anc_pen
        return tot, (e1_v, fce_v, ftv_v, anc_pen)

    vg = nn.value_and_grad(model, loss_seq)

    opt_step = 0
    t_start = time.perf_counter()
    improved = False
    for ep in range(1, epochs + 1):
        ep_t0 = time.perf_counter()
        order = list(range(len(train_seqs)))
        rng.shuffle(order)
        ep_e1, ep_fce, ep_ftv, ep_anc, ep_n = 0.0, 0.0, 0.0, 0.0, 0
        grad_acc = None
        acc_cnt = 0

        for idx_i, s_idx in enumerate(order):
            seq = train_seqs[s_idx]
            B = 8 if rng.random() < 0.50 else 12
            anchors = _sample_anchors(seq["prompt_len"], seq["total_len"], B, n_blk, rng)
            if not anchors:
                continue
            nt, np_, am, tgt_ids, _, b_t32_ids, b_t32_probs = build_block_batch(seq, anchors, B, model.mask_token_id)
            (loss, (e1_v, fce_v, ftv_v, anc_v)), grads = vg(seq["feats"], nt, np_, am, tgt_ids, b_t32_ids, b_t32_probs, B)
            grad_acc = _add_grads(grad_acc, grads)
            acc_cnt += 1
            mx.eval(loss, e1_v, fce_v, ftv_v, anc_v, grad_acc)
            ep_e1 += float(e1_v.item())
            ep_fce += float(fce_v.item())
            ep_ftv += float(ftv_v.item())
            ep_anc += float(anc_v.item())
            ep_n += 1

            is_last = idx_i == len(order) - 1
            if acc_cnt >= grad_accum or (is_last and acc_cnt > 0):
                opt.learning_rate = _cosine_lr(opt_step, total_opt_steps, max_lr, min_lr, warmup_frac=0.10)
                clipped, _ = _clip_and_scale_grads(grad_acc, 1.0 / acc_cnt, 1.0)
                opt.update(model, clipped)
                mx.eval(model.trainable_parameters(), opt.state)
                grad_acc = None
                acc_cnt = 0
                opt_step += 1

        ep_dt = time.perf_counter() - ep_t0
        vr = eval_validation(model, val_seqs, embed_w_bf16, head_w_bf16, use_sdpa=use_sdpa, with_selector=False)
        tag = ""
        if vr[12]["E_acc"] > best_score:
            best_score = vr[12]["E_acc"]
            improved = True
            best_train_state = [(k, mx.array(v)) for k, v in tree_flatten(model.trainable_parameters())]
            mx.eval(*[v for _, v in best_train_state])
            save_dflash2_bf16(model, src_dir, dst_dir)
            tag = " [BEST*]"
        print(
            f"  Ep {ep:2d}/{epochs} ({ep_dt:.1f}s) e1={ep_e1/ep_n:.3f} fce={ep_fce/ep_n:.4f} ftv={ep_ftv/ep_n:.4f} anc={ep_anc/ep_n:.2e} || {_fmt_val_row(vr)}{tag}",
            flush=True,
        )

        # Re-plan check after Epoch 4 per plan
        if ep == 4 and not improved:
            print(
                f"EARLY STOP AT EPOCH 4: val B=12 E_acc never exceeded Ep0 ({val0[12]['E_acc']:.3f}); handing back per plan.",
                flush=True,
            )
            return best_score, False

    model.update(tree_unflatten(best_train_state))
    mx.eval(model.parameters())
    print(f"Stage 1 finished in {time.perf_counter()-t_start:.1f}s; best val B=12 E_acc={best_score:.3f} (Ep0={val0[12]['E_acc']:.3f})", flush=True)
    return best_score, improved


def train_stage2_selector(
    model: DFlash2DraftModel,
    train_seqs: list[dict],
    val_seqs: list[dict],
    embed_w_bf16: mx.array,
    head_w_bf16: mx.array,
    src_dir: Path,
    dst_dir: Path,
    *,
    epochs: int,
    n_blk: int,
    use_sdpa: bool,
):
    print("\n=== Stage 2: Fine-Tuning CandidateSelector (merged backbone frozen, B=12) ===", flush=True)
    val_pre = eval_validation(model, val_seqs, embed_w_bf16, head_w_bf16, use_sdpa=use_sdpa, with_selector=True)
    print(
        "  Val Pre-Sel: "
        + " | ".join(
            f"B={B}: am={val_pre[B]['E_acc']:.3f} sel={val_pre[B]['E_acc_sel']:.3f} med_sel={val_pre[B]['med_E_acc_sel']:.3f}"
            for B in (8, 10, 12, 16)
        ),
        flush=True,
    )

    if epochs <= 0:
        save_dflash2_bf16(model, src_dir, dst_dir)
        return

    B = 12
    K = B - 1
    top_k = model.selector_top_k
    head_t_bf16 = head_w_bf16.T
    cached_train = []
    t_c0 = time.perf_counter()
    for s_idx, seq in enumerate(train_seqs):
        anchors = _sample_anchors(seq["prompt_len"], seq["total_len"], B, n_blk, random.Random(1000 + s_idx))
        if not anchors:
            continue
        nt, np_, am, tgt_ids, prev_ids, _, _ = build_block_batch(seq, anchors, B, model.mask_token_id)
        h_out = multiblock_forward(model, embed_w_bf16, seq["feats"], nt, np_, am, block_size=B, use_sdpa=use_sdpa)
        h_pred = h_out.reshape(len(anchors), B, -1)[:, 1:, :].reshape(len(anchors) * K, -1)
        logits = (h_pred.astype(mx.bfloat16) @ head_t_bf16).astype(mx.float32)
        idx = mx.argpartition(-logits, kth=top_k - 1, axis=-1)[..., :top_k]
        vals = mx.take_along_axis(logits, idx, axis=-1)
        ord_i = mx.argsort(-vals, axis=-1)
        cand_ids = mx.take_along_axis(idx, ord_i, axis=-1)
        u_topk = mx.take_along_axis(vals, ord_i, axis=-1)

        matches = cand_ids == tgt_ids[:, None]
        valid = mx.any(matches, axis=-1)
        slot_tgt = mx.argmax(matches.astype(mx.int32), axis=-1)
        mx.eval(h_pred, cand_ids, u_topk, prev_ids, valid, slot_tgt)
        cached_train.append((h_pred, cand_ids, u_topk, prev_ids, valid.astype(mx.float32), slot_tgt))
    print(f"  Cached {len(cached_train)} train selector tensors in {time.perf_counter()-t_c0:.1f}s", flush=True)

    _freeze_to_selector(model)
    best_sel_score = val_pre[12]["E_acc_sel"]
    best_sel_weights = [(k, mx.array(v)) for k, v in tree_flatten(model.candidate_selector.parameters())]
    mx.eval(*[v for _, v in best_sel_weights])

    cs = model.candidate_selector
    opt = optim.AdamW(learning_rate=1e-4, weight_decay=0.0)
    total_steps = epochs * math.ceil(len(cached_train) / 8)
    step = 0
    rng = random.Random(77)

    def sel_loss(h_p, c_ids, u_lgs, p_ids, v_mask, s_tgt):
        pred_v = cs.predecessor_codebook[p_ids]
        succ_v = cs.successor_codebook[c_ids]
        ctx = pred_v * cs.hidden_projection(h_p)
        trans = mx.sum(ctx[:, None, :] * succ_v, axis=-1)
        scores = u_lgs + trans
        ce = nn.losses.cross_entropy(scores, s_tgt, reduction="none") * v_mask
        return mx.sum(ce) / mx.maximum(mx.sum(v_mask), 1.0)

    vg_sel = nn.value_and_grad(model, sel_loss)
    for ep in range(1, epochs + 1):
        order = list(range(len(cached_train)))
        rng.shuffle(order)
        ep_l, ep_n = 0.0, 0
        grad_acc = None
        acc_cnt = 0
        for i, ci in enumerate(order):
            loss, grads = vg_sel(*cached_train[ci])
            grad_acc = _add_grads(grad_acc, grads)
            acc_cnt += 1
            mx.eval(loss, grad_acc)
            ep_l += float(loss.item())
            ep_n += 1
            if acc_cnt >= 8 or (i == len(order) - 1 and acc_cnt > 0):
                opt.learning_rate = _cosine_lr(step, total_steps, 1e-4, 1e-5, warmup_frac=0.10)
                clipped, _ = _clip_and_scale_grads(grad_acc, 1.0 / acc_cnt, 1.0)
                opt.update(model, clipped)
                mx.eval(model.candidate_selector.parameters(), opt.state)
                grad_acc = None
                acc_cnt = 0
                step += 1

        vp = eval_validation(model, val_seqs, embed_w_bf16, head_w_bf16, use_sdpa=use_sdpa, with_selector=True)
        tag = ""
        if vp[12]["E_acc_sel"] > best_sel_score:
            best_sel_score = vp[12]["E_acc_sel"]
            best_sel_weights = [(k, mx.array(v)) for k, v in tree_flatten(model.candidate_selector.parameters())]
            mx.eval(*[v for _, v in best_sel_weights])
            tag = " [BEST*]"
        print(
            f"  Sel Ep {ep}/{epochs} loss={ep_l/ep_n:.4f} || VAL "
            + " | ".join(f"B={B}: sel={vp[B]['E_acc_sel']:.3f}" for B in (8, 10, 12, 16))
            + tag,
            flush=True,
        )

    model.candidate_selector.update(tree_unflatten(best_sel_weights))
    mx.eval(model.parameters())
    save_dflash2_bf16(model, src_dir, dst_dir)
    print(f"Saved final LingSante-DFlash2 (bfloat16) to {dst_dir} (best val B=12 sel={best_sel_score:.3f})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dflash", default=os.path.expanduser("~/science/models/ling-3.1-flash-dflash2"))
    ap.add_argument("--data-dir", default=os.path.expanduser("~/science/data/lingsante_dflash_train"))
    ap.add_argument("--out-dir", default=os.path.expanduser("~/science/models/lingsante-dflash2"))
    ap.add_argument("--stage1-epochs", type=int, default=8)
    ap.add_argument("--stage2-epochs", type=int, default=5)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=float, default=32.0)
    ap.add_argument("--n-blk", type=int, default=24)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-lr", type=float, default=3e-5)
    ap.add_argument("--min-lr", type=float, default=3e-6)
    ap.add_argument("--unit-test-only", action="store_true")
    a = ap.parse_args()

    maybe_set_recommended_wired_limit()
    src_dir = Path(a.src_dflash).expanduser()
    data_dir = Path(a.data_dir).expanduser()
    out_dir = Path(a.out_dir).expanduser()

    cfg = json.loads((src_dir / "config.json").read_text())
    base_model = DFlash2DraftModel(cfg)
    base_model.load_weights(list(mx.load(str(src_dir / "model.safetensors")).items()), strict=True)
    base_model.update(tree_map(lambda x: x.astype(mx.float32), base_model.parameters()))
    mx.eval(base_model.parameters())

    lora_model = DFlash2DraftModel(cfg)
    lora_model.load_weights(list(mx.load(str(src_dir / "model.safetensors")).items()), strict=True)
    lora_model.update(tree_map(lambda x: x.astype(mx.float32), lora_model.parameters()))
    wrap_lora(lora_model, r=a.lora_r, alpha=a.lora_alpha)
    mx.eval(lora_model.parameters())

    sio = mx.load(str(data_dir / "shared_io.safetensors"))
    embed_w_bf16 = sio["word_embeddings.weight"]
    head_w_bf16 = sio["lm_head.weight"]
    mx.eval(embed_w_bf16, head_w_bf16)

    use_sdpa = run_startup_tests(base_model, lora_model, cfg, embed_w_bf16, head_w_bf16)
    del base_model, head_w_bf16, sio
    mx.clear_cache()
    if a.unit_test_only:
        return 0

    train_seqs, val_seqs, embed_w_bf16, head_w_bf16 = load_dataset(data_dir)
    if a.stage1_epochs > 0:
        _, improved = train_stage1_lora(
            lora_model, train_seqs, val_seqs, embed_w_bf16, head_w_bf16, src_dir, out_dir,
            epochs=a.stage1_epochs, n_blk=a.n_blk, grad_accum=a.grad_accum,
            max_lr=a.max_lr, min_lr=a.min_lr, use_sdpa=use_sdpa,
        )
        if not improved:
            return 2

    merge_lora_to_dflash2(lora_model)
    mx.eval(lora_model.parameters())

    train_stage2_selector(
        lora_model, train_seqs, val_seqs, embed_w_bf16, head_w_bf16, src_dir, out_dir,
        epochs=a.stage2_epochs, n_blk=a.n_blk, use_sdpa=use_sdpa,
    )

    verify_m = DFlash2DraftModel(cfg)
    verify_m.load_weights(list(mx.load(str(out_dir / "model.safetensors")).items()), strict=True)
    mx.eval(verify_m.parameters())
    print(f"Verified strict=True reload from {out_dir}: PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
