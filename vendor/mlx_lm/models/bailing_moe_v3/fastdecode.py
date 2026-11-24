"""Bit-exact fast single-token decode path (1 <= B <= 4) for LingSante BailingMoeV3."""
import os

import mlx.core as mx

from mlx_lm.models.cache import ArraysCache, BatchKVCache, KVCache

from .rows.gemv import gemv_rows, gemv_rows_multi, gemv_rows_res, gemv_swiglu_rows
from .rows.kda import kda_gdn_norm_gate_rows, kda_pre_rows
from .rows.mla import mla_layer_rows
from .rows.moe import moe_down_combine_rows, moe_gate_up_swiglu_rows
from .rows.router import gate_rows


def can_fast_decode(model, inputs, cache) -> bool:
    if os.environ.get("LINGSANTE_FAST_DECODE", "1") in ("0", ""):
        return False
    args = getattr(model, "args", None)
    if (
        args is None
        or args.hidden_size != 4096
        or args.num_attention_heads != 64
        or args.num_experts != 512
        or args.moe_intermediate_size != 1280
    ):
        return False
    if model.training or model.lm_head is None or inputs.ndim != 2 or inputs.shape[1] != 1:
        return False
    B = inputs.shape[0]
    if not (1 <= B <= 4):
        return False
    layers = model.model.layers
    if cache is None or len(cache) != len(layers):
        return False
    for layer, c in zip(layers, cache):
        if layer.is_linear:
            if not isinstance(c, ArraysCache):
                return False
            if c.lengths is not None:
                return False
            lp = getattr(c, "left_padding", None)
            if lp is not None:
                lp_id = id(lp)
                cached = getattr(c, "_lingsante_lp_ok", None)
                if cached is not None and cached[0] == lp_id:
                    if not cached[1]:
                        return False
                else:
                    ok = not any(v > 0 for v in lp.tolist())
                    c._lingsante_lp_ok = (lp_id, ok)
                    if not ok:
                        return False
            if c[0] is None or c[1] is None or c[2] is None or c[3] is None:
                return False
        else:
            if not isinstance(c, (KVCache, BatchKVCache)):
                return False
            if c.keys is None or c.values is None:
                return False
            if getattr(c, "_right_padding", None) is not None:
                return False
    return True


def fast_decode_forward(model, inputs, cache):
    """Run all 72 decoder layers + final norm + lm_head for (B, 1) with 1 <= B <= 4."""
    inner = model.model
    x = inner.word_embeddings(inputs)[:, 0, :]

    for layer, c in zip(inner.layers, cache):
        h_in = mx.fast.rms_norm(x, layer.input_layernorm.weight, layer.input_layernorm.eps)
        if layer.is_linear:
            attn = layer.attention
            q_r, k_r, v_r, f_r, g_r, b_r = gemv_rows_multi(
                h_in,
                [
                    attn.q_proj.weight,
                    attn.k_proj.weight,
                    attn.v_proj.weight,
                    attn.f_proj.weight,
                    attn.g_proj.weight,
                    attn.b_proj.weight,
                ],
            )
            q, k, v, g, beta, ns0, ns1, ns2 = kda_pre_rows(
                q_r,
                k_r,
                v_r,
                f_r,
                b_r,
                c[0],
                c[1],
                c[2],
                attn.q_conv1d.weight,
                attn.k_conv1d.weight,
                attn.v_conv1d.weight,
                attn.A_log,
                attn.dt_bias,
                lower_bound=attn.lower_bound,
                eps=1e-6,
            )
            ov, ns3 = kda_gdn_norm_gate_rows(
                q,
                k,
                v,
                g,
                beta,
                c[3],
                g_r,
                attn.o_norm.weight,
                eps=attn.o_norm.eps,
                fused=True,
            )
            c[0] = ns0
            c[1] = ns1
            c[2] = ns2
            c[3] = ns3
            c.advance(1)
            h_post = gemv_rows_res(ov, attn.o_proj.weight, x)
        else:
            h_post = mla_layer_rows(h_in, x, layer.attention, c)

        mlp_in = mx.fast.rms_norm(
            h_post,
            layer.post_attention_layernorm.weight,
            layer.post_attention_layernorm.eps,
        )
        mlp = layer.mlp
        if hasattr(mlp, "switch_mlp"):
            gate = mlp.gate
            inds, wts = gate_rows(
                mlp_in,
                gate.weight,
                gate.expert_bias,
                top_k=gate.top_k,
                n_group=gate.n_group,
                topk_group=gate.topk_group,
                routed_scaling_factor=gate.routed_scaling_factor,
            )
            se = mlp.shared_experts
            sh = gemv_rows(
                gemv_swiglu_rows(mlp_in, se.gate_proj.weight, se.up_proj.weight),
                se.down_proj.weight,
            )
            sw = mlp.switch_mlp
            h_mid = moe_gate_up_swiglu_rows(
                mlp_in,
                sw.gate_proj.weight,
                sw.gate_proj.scales,
                sw.up_proj.weight,
                sw.up_proj.scales,
                inds,
            )
            x = moe_down_combine_rows(
                h_mid,
                sw.down_proj.weight,
                sw.down_proj.scales,
                inds,
                wts,
                shared=sh,
                residual=h_post,
            )
        else:
            mid = gemv_swiglu_rows(mlp_in, mlp.gate_proj.weight, mlp.up_proj.weight)
            x = gemv_rows_res(mid, mlp.down_proj.weight, h_post)

    x = mx.fast.rms_norm(x, inner.norm.weight, inner.norm.eps)
    return gemv_rows(x, model.lm_head.weight)[:, None, :]
