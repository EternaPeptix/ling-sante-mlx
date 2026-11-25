"""Native built-in MTP (layer 72) & EAGLE-3 Q4-Dense drafter for LingSante-3.1-flash-MXFP4."""
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx import nn

from mlx_lm.layers.attention import create_attention_mask
from mlx_lm.layers.common import swiglu
from mlx_lm.models.bailing_moe_v3.config import ModelArgs
from mlx_lm.models.bailing_moe_v3.layers import BailingMLA, BailingSparseMoE
from mlx_lm.models.cache import KVCache
from mlx_lm.utils import infer_quant_config

from .rows.gemv import gemv_rows, gemv_swiglu_rows
from .rows.mla import (
    _mla_append_causal_slice,
    mla_layer_rows,
    mla_multilinear_lin,
    mla_prep_kv_rope_rows,
    mla_verify_layer_rows,
    mla_verify_varlen_rows,
)
from .rows.moe import moe_dedup_rows, moe_down_combine_rows, moe_gate_up_swiglu_rows
from .rows.router import gate_rows

FEAT_LAYERS = (1, 18, 35, 52, 70)


class FeedbackBridge(nn.Module):
    """Residual feedback adapter mapping MTP hidden states onto target h_final manifold."""

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


class FeatFuse(nn.Module):
    """Residual projection from concatenated multi-layer target features onto hidden_size."""

    def __init__(self, in_dim: int = 20480, rank: int = 256, out_dim: int = 4096, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.RMSNorm(in_dim, eps=eps)
        self.up = nn.Linear(in_dim, rank, bias=False)
        self.down = nn.Linear(rank, out_dim, bias=False)

    def __call__(self, feats: mx.array) -> mx.array:
        return self.down(nn.silu(self.up(self.norm(feats))))


def apply_bridge(bridge, h_out: mx.array, h_in: mx.array) -> mx.array:
    """Apply FeedbackBridge or dual-input FeedbackBridgeV2 uniformly."""
    if bridge is None:
        return h_out
    if getattr(bridge, "_is_v2", False):
        return bridge(h_out, h_in)
    return bridge(h_out)


def load_bridge_and_feat_fuse(fb_path: Path, hidden_size: int, eps: float, n_feat_layers: int = 5):
    """Auto-detect and load FeedbackBridge (v1 or v2) and optional FeatFuse from safetensors."""
    bridge, feat_fuse = None, None
    if not fb_path.exists():
        return bridge, feat_fuse
    fb_w = mx.load(str(fb_path))
    fb_items = [
        (k.removeprefix("feedback_bridge."), v.astype(mx.bfloat16))
        for k, v in fb_w.items()
        if k.startswith("feedback_bridge.")
    ]
    ff_items = [
        (k.removeprefix("feat_fuse."), v.astype(mx.bfloat16))
        for k, v in fb_w.items()
        if k.startswith("feat_fuse.")
    ]
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


class BailingMTPLayer(nn.Module):
    """Layer-72 Multi-Token Prediction / EAGLE-3 drafter sharing embeddings & head with target model."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.rms_eps = args.rms_norm_eps
        self.enorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.hnorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.eh_proj = nn.Linear(args.hidden_size * 2, args.hidden_size, bias=False)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.attention = BailingMLA(args)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.mlp = BailingSparseMoE(args)
        self.final_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        # Attached after construction (not registered in nn.Module weights dict)
        self._embed = None
        self._lm_head = None
        self._q4_head = None
        self._bridge = None
        self._feat_fuse = None
        self._feat_layers = None
        self._is_q4_dense = False
        self._eagle3_cfg = None

    @classmethod
    def _load_base_mtp(cls, target_model, mtp_file: Path, *, with_q4_head: bool) -> "BailingMTPLayer":
        if not mtp_file.exists():
            raise FileNotFoundError(f"MTP weights not found: {mtp_file}")
        mtp = cls(target_model.args)
        w = mx.load(str(mtp_file))

        def pred(p, m):
            if not hasattr(m, "to_quantized") or f"{p}.scales" not in w:
                return False
            return infer_quant_config(p, m, w)

        nn.quantize(mtp, group_size=32, bits=4, mode="mxfp4", class_predicate=pred)
        mtp.load_weights(list(w.items()), strict=True)
        mtp._embed = target_model.model.word_embeddings
        mtp._lm_head = target_model.lm_head
        if with_q4_head:
            mtp._q4_head = nn.QuantizedLinear.from_linear(
                target_model.lm_head, group_size=64, bits=4, mode="affine"
            )
            mtp._q4_head.eval()
            mx.eval(mtp._q4_head.parameters())
        return mtp

    @classmethod
    def from_target(
        cls,
        target_model,
        model_dir: str | Path,
        *,
        with_q4_head: bool = True,
    ) -> "BailingMTPLayer":
        mtp = cls._load_base_mtp(
            target_model, Path(model_dir) / "mtp" / "model-mtp.safetensors", with_q4_head=with_q4_head
        )
        mtp.eval()
        mx.eval(mtp.parameters())
        return mtp

    @classmethod
    def from_eagle3(
        cls,
        target_model,
        eagle3_dir: str | Path,
        *,
        with_q4_head: bool = True,
        q4_dense: bool = True,
    ) -> "BailingMTPLayer":
        """Load trained EAGLE-3 checkpoint (MTP + FeedbackBridge + FeatFuse) and optional Q4-Dense quantization."""
        edir = Path(eagle3_dir)
        mtp = cls._load_base_mtp(target_model, edir / "model-mtp.safetensors", with_q4_head=with_q4_head)
        cfg = None
        cfg_path = edir / "eagle3_config.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text())
        feat_layers = tuple(cfg.get("feat_layers", FEAT_LAYERS)) if cfg else FEAT_LAYERS

        args = target_model.args
        bridge, feat_fuse = load_bridge_and_feat_fuse(
            edir / "feedback_bridge.safetensors",
            args.hidden_size,
            args.rms_norm_eps,
            n_feat_layers=len(feat_layers),
        )

        if q4_dense:
            def qpred_dense(p, m):
                if not hasattr(m, "to_quantized") or isinstance(m, nn.QuantizedLinear):
                    return False
                if p == "eh_proj" or p.startswith("mlp.shared_experts."):
                    return True
                if p in (
                    "attention.q_a_proj",
                    "attention.q_b_proj",
                    "attention.kv_a_proj_with_mqa",
                    "attention.dense",
                ):
                    return True
                return False

            nn.quantize(mtp, group_size=64, bits=4, mode="affine", class_predicate=qpred_dense)
            if bridge is not None:
                nn.quantize(bridge, group_size=64, bits=4, mode="affine")
            if feat_fuse is not None:
                nn.quantize(feat_fuse, group_size=64, bits=4, mode="affine")
            mtp._is_q4_dense = True

        mtp.eval()
        eval_arrs = [mtp.parameters()]
        if bridge is not None:
            bridge.eval()
            eval_arrs.append(bridge.parameters())
        if feat_fuse is not None:
            feat_fuse.eval()
            eval_arrs.append(feat_fuse.parameters())
        mx.eval(*eval_arrs)

        mtp._bridge = bridge
        mtp._feat_fuse = feat_fuse
        mtp._feat_layers = feat_layers if feat_fuse is not None else None
        mtp._eagle3_cfg = cfg
        return mtp

    def apply_feat_fuse(self, h_final: mx.array, captured: mx.array | None) -> mx.array:
        if self._feat_fuse is None or captured is None:
            return h_final
        return (
            h_final.astype(mx.float32) + self._feat_fuse(captured.astype(mx.bfloat16)).astype(mx.float32)
        ).astype(mx.bfloat16)

    def apply_feedback_bridge(self, h_out: mx.array, h_in: mx.array) -> mx.array:
        return apply_bridge(self._bridge, h_out, h_in)

    def make_cache(self, C: int = 1) -> KVCache | list[KVCache]:
        return KVCache() if C == 1 else [KVCache() for _ in range(C)]

    def _mlp_block(self, residual: mx.array) -> mx.array:
        mlp_in = self.post_attention_layernorm(residual)
        return residual + self.mlp(mlp_in)

    def _mlp_rows(self, h_post: mx.array) -> mx.array:
        mlp_in = mx.fast.rms_norm(
            h_post, self.post_attention_layernorm.weight, self.post_attention_layernorm.eps
        )
        gate = self.mlp.gate
        inds, wts = gate_rows(
            mlp_in,
            gate.weight,
            gate.expert_bias,
            top_k=gate.top_k,
            n_group=gate.n_group,
            topk_group=gate.topk_group,
            routed_scaling_factor=gate.routed_scaling_factor,
        )
        se = self.mlp.shared_experts
        if isinstance(se.gate_proj, nn.QuantizedLinear):
            sh = se.down_proj(swiglu(se.gate_proj(mlp_in), se.up_proj(mlp_in)))
        else:
            sh = gemv_rows(
                gemv_swiglu_rows(mlp_in, se.gate_proj.weight, se.up_proj.weight),
                se.down_proj.weight,
            )
        sw = self.mlp.switch_mlp
        dedup = moe_dedup_rows(inds) if h_post.shape[0] >= 5 else None
        h_mid = moe_gate_up_swiglu_rows(
            mlp_in,
            sw.gate_proj.weight,
            sw.gate_proj.scales,
            sw.up_proj.weight,
            sw.up_proj.scales,
            inds,
            dedup=dedup,
        )
        return moe_down_combine_rows(
            h_mid,
            sw.down_proj.weight,
            sw.down_proj.scales,
            inds,
            wts,
            shared=sh,
            residual=h_post,
            dedup=dedup,
        )

    def _kv_project(self, toks_2d: mx.array, prev_3d: mx.array) -> tuple[mx.array, mx.array]:
        embeds = self._embed(toks_2d)
        e_norm = mx.fast.rms_norm(embeds, self.enorm.weight, self.enorm.eps)
        h_norm = mx.fast.rms_norm(prev_3d, self.hnorm.weight, self.hnorm.eps)
        h = self.eh_proj(mx.concatenate([e_norm, h_norm], axis=-1))
        x_a = mx.fast.rms_norm(h, self.input_layernorm.weight, self.input_layernorm.eps)
        attn = self.attention
        L_tok = int(toks_2d.shape[1])
        compressed = attn.kv_a_proj_with_mqa(x_a)
        compressed, k_rope = mx.split(compressed, [attn.kv_lora_rank], axis=-1)
        kv_latent = attn.kv_a_layernorm(compressed)
        k_rope = k_rope.reshape(1, L_tok, 1, attn.qk_rope_head_dim).transpose(0, 2, 1, 3)
        return kv_latent, k_rope

    def kv_append(self, token_ids: mx.array, prev_hidden: mx.array, cache: KVCache) -> None:
        """Bit-exact KV-only MTP cache update (skips Q/SDPA/dense/MLP/lm_head)."""
        toks_2d = token_ids.reshape(1, -1)
        L_tok = int(toks_2d.shape[1])
        if L_tok <= 0:
            return
        prev_3d = prev_hidden.reshape(1, L_tok, -1)
        kv_latent, k_rope = self._kv_project(toks_2d, prev_3d)
        k_rope = self.attention.rope(k_rope, offset=cache.offset)
        cache.update_and_fetch(mx.expand_dims(kv_latent, axis=1), k_rope)

    def kv_append_varlen(
        self, token_ids_flat: mx.array, prev_hidden_flat: mx.array, caches: list[KVCache], Ls: list[int]
    ) -> None:
        """Single-pass batched KV-only MTP cache update for variable stream lengths."""
        B = sum(Ls)
        if B <= 0:
            return
        assert token_ids_flat.size == B and len(caches) == len(Ls)
        kv_latent, k_rope = self._kv_project(
            token_ids_flat.reshape(1, B), prev_hidden_flat.reshape(1, B, -1)
        )
        attn = self.attention
        pos = 0
        for c_cache, lc in zip(caches, Ls):
            if lc <= 0:
                continue
            kr_s = attn.rope(k_rope[:, :, pos : pos + lc, :], offset=c_cache.offset)
            kl_s = kv_latent[:, pos : pos + lc, :]
            c_cache.update_and_fetch(mx.expand_dims(kl_s, axis=1), kr_s)
            pos += lc

    def prefill(self, tokens_shifted: mx.array, hidden_states: mx.array, cache: KVCache) -> mx.array:
        """Populate `cache` during prompt prefill (`tokens_shifted` is `(1, L)`, `hidden_states` is `(1, L, H)`)."""
        embeds = self._embed(tokens_shifted)
        e_norm = mx.fast.rms_norm(embeds, self.enorm.weight, self.enorm.eps)
        h_norm = mx.fast.rms_norm(hidden_states, self.hnorm.weight, self.hnorm.eps)
        h = self.eh_proj(mx.concatenate([e_norm, h_norm], axis=-1))
        mask = create_attention_mask(h, cache, return_array=True)
        h = h + self.attention(self.input_layernorm(h), mask=mask, cache=cache)
        h = self._mlp_block(h)
        return self.final_layernorm(h)

    def _mla_q4_step(self, h_in: mx.array, res: mx.array, cache: Any, *, C: int) -> mx.array:
        """Single-token (L=1) Q4-Dense MLA step for C=1 or C>1 over unpadded KVCache list."""
        attn = self.attention
        if C == 1 and not isinstance(cache, list):
            return res + attn(h_in[:, None, :], mask=None, cache=cache)[:, 0, :]

        q_a = attn.q_a_proj(h_in)
        q_a_norm = mx.fast.rms_norm(q_a, attn.q_a_layernorm.weight, attn.q_a_layernorm.eps)
        q_full = attn.q_b_proj(q_a_norm)
        kv_a_full = attn.kv_a_proj_with_mqa(h_in)
        g_raw = gemv_rows(h_in, attn.g_proj.weight)
        caches = cache if isinstance(cache, list) else [cache]
        assert len(caches) == C
        offs = mx.array([int(ci.offset) for ci in caches], dtype=mx.int32)
        q_nope, q_rope, kv_latent, k_rope = mla_prep_kv_rope_rows(
            q_full,
            kv_a_full,
            attn.kv_a_layernorm.weight,
            offs,
            kv_eps=attn.kv_a_layernorm.eps,
            base=float(getattr(attn.rope, "base", 6000000.0)),
            scale=float(getattr(attn.rope, "scale", 1.0)),
        )
        q_nope = mla_multilinear_lin(q_nope, attn.embed_q)
        scale = attn.scale
        outs = []
        for c in range(C):
            kl_c, kr_c = caches[c].update_and_fetch(kv_latent[c : c + 1], k_rope[c : c + 1])
            _mla_append_causal_slice(outs, kl_c, kr_c, q_nope, q_rope, c, 0, kl_c.shape[2] - 1, 1, scale)
        out = mx.concatenate(outs, axis=0)
        gated = mla_multilinear_lin(out, attn.unembed_out, g_raw=g_raw)
        return res + attn.dense(gated)

    def _mtp_embed_and_mla_in(self, token_ids: mx.array, prev_hidden: mx.array, B: int):
        toks_flat = token_ids.reshape(-1)
        prev_flat = prev_hidden.reshape(B, -1)
        embeds = self._embed(toks_flat)
        e_norm = mx.fast.rms_norm(embeds, self.enorm.weight, self.enorm.eps)
        h_norm = mx.fast.rms_norm(prev_flat, self.hnorm.weight, self.hnorm.eps)
        eh_in = mx.concatenate([e_norm, h_norm], axis=-1)
        h = self.eh_proj(eh_in) if self._is_q4_dense else gemv_rows(eh_in, self.eh_proj.weight)
        h_in = mx.fast.rms_norm(h, self.input_layernorm.weight, self.input_layernorm.eps)
        return h, h_in

    def _mtp_head(self, head_in: mx.array, *, use_q4_head: bool):
        if use_q4_head and self._q4_head is not None:
            return self._q4_head(head_in)
        return gemv_rows(head_in, self._lm_head.weight)

    def step_rows(
        self,
        token_ids: mx.array,
        prev_hidden: mx.array,
        cache: Any,
        *,
        C: int = 1,
        L: int = 1,
        use_q4_head: bool = True,
        head_only_last: bool = True,
        return_logits: bool = True,
    ):
        """Fast row-kernel / Q4-Dense MTP step supporting `(C,)` single-step (`L=1`) and `(C*L,)` refresh (`C*L <= 24`).

        `token_ids`: `(C*L,)` or `(C, L)` uint32/int32.
        `prev_hidden`: `(C*L, 4096)` or `(C, L, 4096)` bf16.
        Returns `(next_hidden_last (C, 4096), logits (C, V) or (C, L, V) or None)`.
        """
        B = C * L
        assert 1 <= B <= 24

        if self._is_q4_dense and L > 1:
            toks_2d = token_ids.reshape(C, L)
            prev_3d = prev_hidden.reshape(C, L, -1)
            caches = cache if isinstance(cache, list) else [cache]
            if not return_logits:
                self.kv_append_varlen(toks_2d.reshape(-1), prev_3d.reshape(B, -1), caches, [L] * C)
                return prev_3d[:, -1, :], None
            if head_only_last:
                self.kv_append_varlen(
                    toks_2d[:, :-1].reshape(-1),
                    prev_3d[:, :-1, :].reshape(C * (L - 1), -1),
                    caches,
                    [L - 1] * C,
                )
                return self.step_rows(
                    toks_2d[:, -1],
                    prev_3d[:, -1, :],
                    cache,
                    C=C,
                    L=1,
                    use_q4_head=use_q4_head,
                    return_logits=True,
                )
            h_list, l_list = [], []
            for t in range(L):
                ht, lt = self.step_rows(
                    toks_2d[:, t],
                    prev_3d[:, t, :],
                    cache,
                    C=C,
                    L=1,
                    use_q4_head=use_q4_head,
                    return_logits=True,
                )
                h_list.append(ht)
                l_list.append(lt)
            return h_list[-1], mx.stack(l_list, axis=1)

        h, h_in = self._mtp_embed_and_mla_in(token_ids, prev_hidden, B)

        if self._is_q4_dense:
            h_post = self._mla_q4_step(h_in, h, cache, C=C)
        elif L == 1 and not isinstance(cache, list):
            h_post = mla_layer_rows(h_in, h, self.attention, cache)
        else:
            h_post = mla_verify_layer_rows(h_in, h, self.attention, cache, C=C, L=L)

        h_out = mx.fast.rms_norm(self._mlp_rows(h_post), self.final_layernorm.weight, self.final_layernorm.eps)
        last_hidden = h_out.reshape(C, L, -1)[:, -1, :]

        if not return_logits:
            return last_hidden, None

        head_in = last_hidden if (head_only_last or L == 1) else h_out
        logits = self._mtp_head(head_in, use_q4_head=use_q4_head)
        if not head_only_last and L > 1:
            logits = logits.reshape(C, L, -1)
        return last_hidden, logits

    def step_varlen_rows(
        self,
        token_ids_flat: mx.array,
        prev_hidden_flat: mx.array,
        caches: list[KVCache],
        Ls: list[int],
        *,
        use_q4_head: bool = True,
        return_logits: bool = True,
    ):
        """Single-pass variable-length MTP refresh for heterogeneous stream accept counts (`sum(Ls) <= 24`)."""
        B = sum(Ls)
        assert 1 <= B <= 24 and token_ids_flat.size == B and prev_hidden_flat.reshape(B, -1).shape[0] == B

        if self._is_q4_dense:
            prev_2d = prev_hidden_flat.reshape(B, -1)
            tok_flat = token_ids_flat.reshape(-1)
            if not return_logits:
                self.kv_append_varlen(tok_flat, prev_2d, caches, Ls)
                return None, None
            pre_toks, pre_hs, pre_ls = [], [], []
            last_toks, last_hs = [], []
            pos = 0
            for lc in Ls:
                if lc > 1:
                    pre_toks.append(tok_flat[pos : pos + lc - 1])
                    pre_hs.append(prev_2d[pos : pos + lc - 1])
                    pre_ls.append(lc - 1)
                last_toks.append(tok_flat[pos + lc - 1])
                last_hs.append(prev_2d[pos + lc - 1])
                pos += lc
            if pre_toks:
                self.kv_append_varlen(
                    mx.concatenate(pre_toks, axis=0),
                    mx.concatenate(pre_hs, axis=0),
                    caches,
                    pre_ls,
                )
            return self.step_rows(
                mx.stack(last_toks),
                mx.stack(last_hs, axis=0),
                caches,
                C=len(Ls),
                L=1,
                use_q4_head=use_q4_head,
                return_logits=True,
            )

        h, h_in = self._mtp_embed_and_mla_in(token_ids_flat, prev_hidden_flat, B)
        h_post = mla_verify_varlen_rows(h_in, h, self.attention, caches, Ls)
        h_out = mx.fast.rms_norm(self._mlp_rows(h_post), self.final_layernorm.weight, self.final_layernorm.eps)

        end_rows = []
        pos = 0
        for lc in Ls:
            pos += lc
            end_rows.append(pos - 1)
        last_hidden = h_out[mx.array(end_rows, dtype=mx.int32)]

        if not return_logits:
            return last_hidden, None
        logits = self._mtp_head(last_hidden, use_q4_head=use_q4_head)
        return last_hidden, logits
