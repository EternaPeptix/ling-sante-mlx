"""DFlash2 parallel block-diffusion drafter (+ UBTree lattice) for LingSante / Ling-3.1-flash."""
import json
from pathlib import Path
from typing import Any, Sequence

import mlx.core as mx
from mlx import nn

from mlx_lm.layers.recurrent import create_ssm_mask
from mlx_lm.models.cache import KVCache, make_prompt_cache

from .verify_decode import commit_verify_cache, fast_verify_forward


class DFlashGroupedConv(nn.Module):
    """Grouped dynamic causal depthwise convolution within a proposal block."""

    def __init__(
        self,
        hidden_size: int = 4096,
        taps: int = 2,
        group_size: int = 16,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.taps = int(taps)
        self.group_size = int(group_size)
        self.num_groups = self.hidden_size // self.group_size

        base_kernel = mx.zeros((2, self.taps, self.hidden_size), dtype=mx.bfloat16)
        base_kernel[:, 0, :] = mx.array(1.0, dtype=mx.bfloat16)
        self.base_kernel = base_kernel
        self.kernel_projection = nn.Linear(
            self.hidden_size, 2 * self.taps * self.num_groups, bias=False
        )

    def _convolve(
        self,
        hidden_states: mx.array,
        delta: mx.array,
        *,
        side: int,
        block_size: int | None = None,
    ) -> mx.array:
        B, L, H = hidden_states.shape
        bs = L if block_size is None else int(block_size)
        n_blk = L // bs
        blocks = hidden_states.reshape(B, n_blk, bs, self.num_groups, self.group_size)
        dyn = delta.reshape(B, n_blk, bs, self.taps, self.num_groups)[..., None]
        base = self.base_kernel[side].reshape(1, 1, 1, self.taps, self.num_groups, self.group_size)
        coeffs = base + dyn.astype(base.dtype)
        out = coeffs[:, :, :, 0] * blocks
        if self.taps == 2:
            if bs >= 2:
                z = mx.zeros((B, n_blk, 1, self.num_groups, self.group_size), dtype=blocks.dtype)
                shifted = mx.concatenate([z, blocks[:, :, : bs - 1]], axis=2)
                out = out + coeffs[:, :, :, 1] * shifted
        else:
            for tap in range(1, self.taps):
                if tap >= bs:
                    break
                z = mx.zeros((B, n_blk, tap, self.num_groups, self.group_size), dtype=blocks.dtype)
                shifted = mx.concatenate([z, blocks[:, :, : bs - tap]], axis=2)
                out = out + coeffs[:, :, :, tap] * shifted
        return out.reshape(B, L, H)

    def prepare(self, hidden_states: mx.array, *, block_size: int | None = None):
        B, L, _ = hidden_states.shape
        coeffs = self.kernel_projection(hidden_states).reshape(
            B, L, 2, self.taps, self.num_groups
        )
        prepared = self._convolve(hidden_states, coeffs[:, :, 0, :, :], side=0, block_size=block_size)
        return prepared, coeffs[:, :, 1, :, :]

    def finish(self, hidden_states: mx.array, delta1: mx.array, *, block_size: int | None = None) -> mx.array:
        return self._convolve(hidden_states, delta1, side=1, block_size=block_size)


class Qwen3DFlashAttention(nn.Module):
    """Non-causal context+noise Qwen3 attention matching DFlash safetensors."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.hidden_size = int(cfg["hidden_size"])
        self.num_heads = int(cfg["num_attention_heads"])
        self.num_kv_heads = int(cfg["num_key_value_heads"])
        self.head_dim = int(cfg.get("head_dim", self.hidden_size // self.num_heads))
        self.scale = self.head_dim ** -0.5
        eps = float(cfg.get("rms_norm_eps", 1e-6))
        rope_theta = float(
            (cfg.get("rope_parameters") or {}).get("rope_theta", cfg.get("rope_theta", 6000000.0))
        )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=eps)
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=rope_theta)

    def prefill_ctx(self, target_hidden: mx.array, cache: KVCache) -> None:
        B, L_ctx, _ = target_hidden.shape
        if L_ctx <= 0:
            return
        past = cache.offset
        k = self.k_norm(self.k_proj(target_hidden).reshape(B, L_ctx, self.num_kv_heads, self.head_dim)).transpose(0, 2, 1, 3)
        k = self.rope(k, offset=past)
        v = self.v_proj(target_hidden).reshape(B, L_ctx, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        cache.update_and_fetch(k, v)

    def __call__(
        self,
        hidden_states: mx.array,
        target_hidden: mx.array | None,
        cache: KVCache,
    ) -> mx.array:
        B, L_q, _ = hidden_states.shape
        L_ctx = int(target_hidden.shape[1]) if target_hidden is not None else 0
        past = cache.offset

        q = self.q_norm(self.q_proj(hidden_states).reshape(B, L_q, self.num_heads, self.head_dim)).transpose(0, 2, 1, 3)
        q = self.rope(q, offset=past + L_ctx)

        if L_ctx > 0:
            k_ctx = self.k_norm(self.k_proj(target_hidden).reshape(B, L_ctx, self.num_kv_heads, self.head_dim)).transpose(0, 2, 1, 3)
            k_ctx = self.rope(k_ctx, offset=past)
            v_ctx = self.v_proj(target_hidden).reshape(B, L_ctx, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
            k_cached, v_cached = cache.update_and_fetch(k_ctx, v_ctx)
        elif not cache.empty():
            k_cached, v_cached = cache.keys_and_values()
        else:
            k_cached, v_cached = None, None

        k_noise = self.k_norm(self.k_proj(hidden_states).reshape(B, L_q, self.num_kv_heads, self.head_dim)).transpose(0, 2, 1, 3)
        k_noise = self.rope(k_noise, offset=past + L_ctx)
        v_noise = self.v_proj(hidden_states).reshape(B, L_q, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        if k_cached is not None:
            k_all = mx.concatenate([k_cached, k_noise], axis=2)
            v_all = mx.concatenate([v_cached, v_noise], axis=2)
        else:
            k_all, v_all = k_noise, v_noise

        out = mx.fast.scaled_dot_product_attention(q, k_all, v_all, scale=self.scale, mask=None)
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(B, L_q, -1))


class DFlashMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3DFlash2DecoderLayer(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        H = int(cfg["hidden_size"])
        I = int(cfg["intermediate_size"])
        eps = float(cfg.get("rms_norm_eps", 1e-6))
        dc = cfg.get("dflash_config") or {}
        taps = int(dc.get("conv_kernel_size", 2))
        group_size = int(dc.get("conv_group_size", 16))

        self.input_layernorm = nn.RMSNorm(H, eps=eps)
        self.attention_conv = DFlashGroupedConv(H, taps=taps, group_size=group_size)
        self.self_attn = Qwen3DFlashAttention(cfg)
        self.post_attention_layernorm = nn.RMSNorm(H, eps=eps)
        self.mlp = DFlashMLP(H, I)
        self.mlp_conv = DFlashGroupedConv(H, taps=taps, group_size=group_size)

    def __call__(self, hidden_states: mx.array, target_hidden: mx.array | None, cache: KVCache) -> mx.array:
        residual = hidden_states
        h = self.input_layernorm(hidden_states)
        h, attn_k1 = self.attention_conv.prepare(h)
        h = self.self_attn(h, target_hidden, cache)
        h = self.attention_conv.finish(h, attn_k1)
        hidden_states = residual + h

        residual = hidden_states
        h = self.post_attention_layernorm(hidden_states)
        h, mlp_k1 = self.mlp_conv.prepare(h)
        h = self.mlp(h)
        h = self.mlp_conv.finish(h, mlp_k1)
        return residual + h


_COMPILED_SEL_CACHE: dict[int, Any] = {}


def _get_compiled_selector(top_k: int):
    fn = _COMPILED_SEL_CACHE.get(top_k)
    if fn is None:
        @mx.compile
        def _sel(logits, h_out, anchor_tok, pred_cb, succ_cb, hp_w):
            idx = mx.argpartition(-logits, kth=top_k - 1, axis=-1)[..., :top_k]
            vals = mx.take_along_axis(logits, idx, axis=-1)
            order = mx.argsort(-vals, axis=-1)
            cand_ids = mx.take_along_axis(idx, order, axis=-1)
            u_f32 = mx.take_along_axis(vals, order, axis=-1).astype(mx.float32)

            hp = (h_out @ hp_w.T).astype(mx.float32)
            succ = succ_cb[cand_ids].astype(mx.float32)
            pred_id = anchor_tok.reshape(1)
            cols = []
            K_pos = logits.shape[1]
            for pos in range(K_pos):
                pred_vec = pred_cb[pred_id].astype(mx.float32)
                ctx = pred_vec * hp[:, pos, :]
                trans = mx.sum(ctx[:, None, :] * succ[:, pos, :, :], axis=-1)
                scores = u_f32[:, pos, :] + trans
                sel = mx.argmax(scores, axis=-1, keepdims=True)
                pred_id = mx.take_along_axis(cand_ids[:, pos, :], sel, axis=-1).squeeze(-1)
                cols.append(pred_id)
            return mx.stack(cols, axis=1).squeeze(0)

        fn = _COMPILED_SEL_CACHE[top_k] = _sel
    return fn


class CandidateSelector(nn.Module):
    """Low-rank bigram transition selector for DFlash2."""

    def __init__(self, hidden_size: int, vocab_size: int, state_rank: int, top_k: int):
        super().__init__()
        self.top_k = int(top_k)
        self.state_rank = int(state_rank)
        self.predecessor_codebook = mx.zeros((vocab_size, state_rank), dtype=mx.bfloat16)
        self.successor_codebook = mx.zeros((vocab_size, state_rank), dtype=mx.bfloat16)
        self.hidden_projection = nn.Linear(hidden_size, state_rank, bias=False)

    def greedy_path(
        self,
        *,
        candidate_ids: mx.array,
        unary_logits: mx.array,
        hidden_states: mx.array,
        anchor_token_id: mx.array,
    ) -> mx.array:
        """Select greedy path along the `(B=1, K_pos, top_k)` candidate slots."""
        B, K_pos, _ = candidate_ids.shape
        hp = self.hidden_projection(hidden_states).astype(mx.float32)  # (B, K_pos, R)
        succ = self.successor_codebook[candidate_ids].astype(mx.float32)  # (B, K_pos, top_k, R)
        u_f32 = unary_logits.astype(mx.float32)

        pred_id = anchor_token_id
        cols = []
        for pos in range(K_pos):
            pred_vec = self.predecessor_codebook[pred_id].astype(mx.float32)  # (B, R)
            ctx = pred_vec * hp[:, pos, :]  # (B, R)
            trans = mx.sum(ctx[:, None, :] * succ[:, pos, :, :], axis=-1)  # (B, top_k)
            scores = u_f32[:, pos, :] + trans
            sel = mx.argmax(scores, axis=-1, keepdims=True)
            pred_id = mx.take_along_axis(candidate_ids[:, pos, :], sel, axis=-1).squeeze(-1)
            cols.append(pred_id)
        return mx.stack(cols, axis=1)

    def build_lattice(
        self,
        *,
        candidate_ids: mx.array,
        unary_logits: mx.array,
        hidden_states: mx.array,
        anchor_token_id: mx.array,
    ) -> mx.array:
        """Build full `(B, K_pos, top_k, top_k)` transition score tensor."""
        B, K_pos, top_k = candidate_ids.shape
        anchor_rep = mx.broadcast_to(anchor_token_id[:, None, None], (B, 1, top_k))
        pred_ids = mx.concatenate([anchor_rep, candidate_ids[:, :-1, :]], axis=1)
        pred = self.predecessor_codebook[pred_ids].astype(mx.float32)  # (B, K_pos, top_k, R)
        succ = self.successor_codebook[candidate_ids].astype(mx.float32)  # (B, K_pos, top_k, R)
        hp = self.hidden_projection(hidden_states).astype(mx.float32)[:, :, None, :]
        ctx = pred * hp  # (B, K_pos, top_k, R)
        # (B, K_pos, P, R) @ (B, K_pos, R, C) -> (B, K_pos, P, C)
        trans = ctx @ succ.transpose(0, 1, 3, 2)
        return unary_logits.astype(mx.float32)[:, :, None, :] + trans


class _RefBox:
    __slots__ = ("d",)

    def __init__(self):
        self.d: dict[str, Any] = {}


class DFlash2DraftModel(nn.Module):
    """DFlash2 parallel block-diffusion drafter with GroupedConv and CandidateSelector."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        dc = cfg.get("dflash_config") or {}
        H = int(cfg["hidden_size"])
        eps = float(cfg.get("rms_norm_eps", 1e-6))

        self.block_size = int(cfg.get("block_size", dc.get("block_size", 8)))
        self.mask_token_id = int(dc.get("mask_token_id", 156894))
        self.target_layer_ids = tuple(int(x) for x in dc.get("target_layer_ids", [1, 18, 35, 52, 70]))
        self.selector_rank = int(dc.get("selector_rank", cfg.get("markov_rank", 256)))
        self.selector_top_k = int(dc.get("selector_top_k", 16))
        self.output_multiplier = float(dc.get("output_multiplier", 1.0))
        self.softcap = dc.get("final_logit_softcapping", None)

        self.fc = nn.Linear(len(self.target_layer_ids) * H, H, bias=False)
        self.hidden_norm = nn.RMSNorm(H, eps=eps)
        self.layers = [Qwen3DFlash2DecoderLayer(cfg) for _ in range(int(cfg["num_hidden_layers"]))]
        self.norm = nn.RMSNorm(H, eps=eps)
        self.candidate_selector = CandidateSelector(
            H, int(cfg["vocab_size"]), self.selector_rank, self.selector_top_k
        )

        # Shared target embeddings/heads held in plain container outside nn.Module tree
        self._refs_box = _RefBox()

    @classmethod
    def from_target(
        cls,
        target_model,
        model_dir: str | Path,
        *,
        quant_bits: int | None = None,
        with_q4_head: bool = True,
    ) -> "DFlash2DraftModel":
        d = Path(model_dir)
        cfg = json.loads((d / "config.json").read_text())
        w = mx.load(str(d / "model.safetensors"))
        m = cls(cfg)
        m.load_weights(list(w.items()), strict=True)
        if quant_bits == 4:
            def _qpred(p, mod):
                if not hasattr(mod, "to_quantized"):
                    return False
                return p == "fc" or ".self_attn." in p or ".mlp." in p

            nn.quantize(m, group_size=64, bits=4, mode="affine", class_predicate=_qpred)
        m.eval()
        mx.eval(m.parameters())

        m._refs_box.d["embed"] = target_model.model.word_embeddings
        m._refs_box.d["lm_head"] = target_model.lm_head
        if with_q4_head:
            holder = getattr(target_model, "_lingsante_mtp_holder", None)
            if holder is not None and getattr(holder.mtp, "_q4_head", None) is not None:
                m._refs_box.d["q4_head"] = holder.mtp._q4_head
            else:
                q4h = nn.QuantizedLinear.from_linear(
                    target_model.lm_head, group_size=64, bits=4, mode="affine"
                )
                q4h.eval()
                mx.eval(q4h.parameters())
                m._refs_box.d["q4_head"] = q4h
        return m

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.layers]

    def project_features(self, raw_target_hidden: mx.array) -> mx.array:
        return self.hidden_norm(self.fc(raw_target_hidden))

    def prefill_ctx(self, raw_target_hidden: mx.array, caches: list[KVCache]) -> None:
        if raw_target_hidden.shape[1] <= 0:
            return
        th = self.project_features(raw_target_hidden)
        for layer, c in zip(self.layers, caches):
            layer.self_attn.prefill_ctx(th, c)

    def _head(self, hidden: mx.array, *, use_q4_head: bool) -> mx.array:
        head = self._refs_box.d.get("q4_head") if use_q4_head else None
        if head is None:
            head = self._refs_box.d["lm_head"]
        logits = head(hidden)
        if self.output_multiplier != 1.0 or self.softcap is not None:
            logits = logits.astype(mx.float32) * self.output_multiplier
            if self.softcap is not None:
                sc = float(self.softcap)
                logits = mx.tanh(logits / sc) * sc
        return logits

    def __call__(
        self,
        noise_tokens: mx.array,
        raw_target_hidden: mx.array | None,
        caches: list[KVCache],
    ) -> mx.array:
        h = self._refs_box.d["embed"](noise_tokens)
        th = self.project_features(raw_target_hidden) if (raw_target_hidden is not None and raw_target_hidden.shape[1] > 0) else None
        for layer, c in zip(self.layers, caches):
            h = layer(h, th, c)
        return self.norm(h)

    def draft_block(
        self,
        anchor_tok: mx.array,
        raw_target_hidden: mx.array | None,
        caches: list[KVCache],
        *,
        block_size: int | None = None,
        use_q4_head: bool = True,
        use_selector: bool = True,
        return_lattice: bool = False,
    ):
        """Draft `K = block_size - 1` tokens in ONE parallel block pass from `(1,) anchor_tok`."""
        bs = self.block_size if block_size is None else int(block_size)
        K_pos = bs - 1
        masks = mx.full((1, K_pos), self.mask_token_id, dtype=mx.uint32)
        inp = mx.concatenate([anchor_tok.reshape(1, 1).astype(mx.uint32), masks], axis=1)
        h_out = self(inp, raw_target_hidden, caches)[:, 1:, :]  # (1, K_pos, 4096)
        logits = self._head(h_out, use_q4_head=use_q4_head)  # (1, K_pos, V)

        if not use_selector and not return_lattice:
            return mx.argmax(logits, axis=-1).squeeze(0), None, None

        top_k = self.selector_top_k
        if use_selector and not return_lattice:
            cs = self.candidate_selector
            draft_toks = _get_compiled_selector(top_k)(
                logits,
                h_out,
                anchor_tok,
                cs.predecessor_codebook,
                cs.successor_codebook,
                cs.hidden_projection.weight,
            )
            return draft_toks, None, None

        idx = mx.argpartition(-logits, kth=top_k - 1, axis=-1)[..., :top_k]
        vals = mx.take_along_axis(logits, idx, axis=-1)
        order = mx.argsort(-vals, axis=-1)
        cand_ids = mx.take_along_axis(idx, order, axis=-1)
        unary_topk = mx.take_along_axis(vals, order, axis=-1)

        if use_selector:
            draft_toks = self.candidate_selector.greedy_path(
                candidate_ids=cand_ids,
                unary_logits=unary_topk,
                hidden_states=h_out,
                anchor_token_id=anchor_tok.reshape(1),
            ).squeeze(0)
        else:
            draft_toks = mx.argmax(logits, axis=-1).squeeze(0)

        lattice = None
        if return_lattice:
            lattice = self.candidate_selector.build_lattice(
                candidate_ids=cand_ids,
                unary_logits=unary_topk,
                hidden_states=h_out,
                anchor_token_id=anchor_tok.reshape(1),
            )
            return draft_toks, (cand_ids.squeeze(0), lattice.squeeze(0)), h_out
        return draft_toks, None, None


def prefill_target_and_dflash(
    target_model,
    dflash: DFlash2DraftModel,
    ids: Sequence[int],
    *,
    step: int = 512,
    layer_ids: tuple[int, ...] | None = None,
):
    """Exact `B=1` target prefill on `ids[:-1]` + exact DFlash2 context KV population."""
    lids = dflash.target_layer_ids if layer_ids is None else tuple(layer_ids)
    cap_set = set(lids)
    inner = target_model.model
    t_cache = make_prompt_cache(target_model)
    d_cache = dflash.make_cache()

    p = mx.array(ids, dtype=mx.uint32)
    pos = 0
    while p.size - pos > 1:
        n = min(step, p.size - 1 - pos)
        chunk = p[pos : pos + n][None]
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
        feats = mx.concatenate([captured[li] for li in lids], axis=-1)
        dflash.prefill_ctx(feats, d_cache)
        mx.eval(h, feats, [c.state for c in t_cache], [c.state for c in d_cache])
        pos += n

    # Last prompt token via fast_verify_forward capturing target features + anchor prediction
    last_tok = p[-1:].reshape(1, 1)
    logits0, rb0 = fast_verify_forward(
        target_model,
        last_tok,
        t_cache,
        return_hidden=False,
        return_rollback=True,
        capture_layers=lids,
    )
    commit_verify_cache(t_cache, rb0, 1)
    lp0 = logits0[0, 0, :] - mx.logsumexp(logits0[0, 0, :], keepdims=True)
    y0 = mx.argmax(lp0, axis=-1)
    feats0 = rb0["captured"]  # (1, 1, 20480)
    mx.eval(y0, lp0, feats0, *[c.state for c in t_cache])
    return t_cache, d_cache, y0, lp0, feats0


def fast_mask_helper(inner, h, cache, attention=True):
    from mlx_lm.layers.attention import create_attention_mask

    return create_attention_mask(h, cache, return_array=True)
