"""Native built-in MTP (layer 72) drafter for LingSante-3.1-flash-MXFP4."""
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx import nn

from mlx_lm.layers.attention import create_attention_mask
from mlx_lm.models.bailing_moe_v3.config import ModelArgs
from mlx_lm.models.bailing_moe_v3.layers import BailingMLA, BailingSparseMoE
from mlx_lm.models.cache import KVCache
from mlx_lm.utils import infer_quant_config

from .rows.gemv import gemv_rows, gemv_rows_res, gemv_swiglu_rows
from .rows.mla import mla_layer_rows, mla_verify_layer_rows, mla_verify_varlen_rows
from .rows.moe import moe_dedup_rows, moe_down_combine_rows, moe_gate_up_swiglu_rows
from .rows.router import gate_rows


class BailingMTPLayer(nn.Module):
    """Layer-72 Multi-Token Prediction drafter sharing embeddings & head with target model."""

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

    @classmethod
    def from_target(
        cls,
        target_model,
        model_dir: str | Path,
        *,
        with_q4_head: bool = True,
    ) -> "BailingMTPLayer":
        mtp_file = Path(model_dir) / "mtp" / "model-mtp.safetensors"
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
        mtp.eval()
        mx.eval(mtp.parameters())

        mtp._embed = target_model.model.word_embeddings
        mtp._lm_head = target_model.lm_head
        if with_q4_head:
            mtp._q4_head = nn.QuantizedLinear.from_linear(
                target_model.lm_head, group_size=64, bits=4, mode="affine"
            )
            mtp._q4_head.eval()
            mx.eval(mtp._q4_head.parameters())
        return mtp

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

    def _mtp_embed_and_mla_in(self, token_ids: mx.array, prev_hidden: mx.array, B: int):
        toks_flat = token_ids.reshape(-1)
        prev_flat = prev_hidden.reshape(B, -1)
        embeds = self._embed(toks_flat)
        e_norm = mx.fast.rms_norm(embeds, self.enorm.weight, self.enorm.eps)
        h_norm = mx.fast.rms_norm(prev_flat, self.hnorm.weight, self.hnorm.eps)
        h = gemv_rows(mx.concatenate([e_norm, h_norm], axis=-1), self.eh_proj.weight)
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
        """Fast row-kernel MTP step supporting `(C,)` single-step (`L=1`) and `(C*L,)` refresh (`C*L <= 24`).

        `token_ids`: `(C*L,)` or `(C, L)` uint32/int32.
        `prev_hidden`: `(C*L, 4096)` or `(C, L, 4096)` bf16.
        Returns `(next_hidden_last (C, 4096), logits (C, V) or (C, L, V) or None)`.
        """
        B = C * L
        assert 1 <= B <= 24
        h, h_in = self._mtp_embed_and_mla_in(token_ids, prev_hidden, B)

        if L == 1 and not isinstance(cache, list):
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
