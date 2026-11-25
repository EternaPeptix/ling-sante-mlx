"""Bit-exact fast single-token decode path (1 <= B <= 4) for LingSante BailingMoeV3."""
import os

import mlx.core as mx

from mlx_lm.models.cache import ArraysCache, BatchKVCache, KVCache

from .rows.gemv_fp8 import (
    gemv_kda_qkvfgb_merged_rows,
    gemv_lin,
    gemv_lin_multi,
    gemv_lin_res,
    gemv_lin_swiglu,
    gemv_router_se_swiglu_rows,
)
from .rows.kda import kda_gdn_norm_gate_rows, kda_pre_rows
from .rows.mla import mla_layer_rows
from .rows.moe import moe_down_combine_rows, moe_gate_up_swiglu_rows
from .rows.router import group_router_dedup_rows, group_router_rows

_FAST_FLUSH_EVERY = int(os.environ.get("LINGSANTE_FAST_FLUSH", "4"))


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
    layers = inner.layers
    n_layers = len(layers)
    x = inner.word_embeddings(inputs)[:, 0, :]
    x_pss = None
    use_kda_in_pss = os.environ.get("LINGSANTE_PARTSS_KDA_IN", "0") not in ("0", "")

    for li, (layer, c) in enumerate(zip(layers, cache)):
        ln_in = layer.input_layernorm
        is_dense_mlp = not hasattr(layer.mlp, "switch_mlp")
        if layer.is_linear:
            attn = layer.attention
            use_pss_in = (x_pss is not None) and use_kda_in_pss
            if hasattr(attn, "_w_qkvfg"):
                if use_pss_in:
                    q_r, k_r, v_r, f_r, g_r, b_r = gemv_kda_qkvfgb_merged_rows(
                        x, attn._w_qkvfg, attn._s_qkvfg, attn.b_proj.weight,
                        part_ss=x_pss, norm_w=ln_in.weight, norm_eps=ln_in.eps,
                    )
                else:
                    h_in = mx.fast.rms_norm(x, ln_in.weight, ln_in.eps)
                    q_r, k_r, v_r, f_r, g_r, b_r = gemv_kda_qkvfgb_merged_rows(
                        h_in, attn._w_qkvfg, attn._s_qkvfg, attn.b_proj.weight
                    )
            else:
                qkv_mods = [attn.q_proj, attn.k_proj, attn.v_proj, attn.f_proj, attn.g_proj, attn.b_proj]
                if use_pss_in:
                    q_r, k_r, v_r, f_r, g_r, b_r = gemv_lin_multi(
                        x, qkv_mods, part_ss=x_pss, norm_w=ln_in.weight, norm_eps=ln_in.eps
                    )
                else:
                    h_in = mx.fast.rms_norm(x, ln_in.weight, ln_in.eps)
                    q_r, k_r, v_r, f_r, g_r, b_r = gemv_lin_multi(h_in, qkv_mods)
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
            if is_dense_mlp:
                h_post, pss_o = gemv_lin_res(ov, attn.o_proj, x, return_part_ss=True)
            else:
                h_post = gemv_lin_res(ov, attn.o_proj, x)
                pss_o = None
        else:
            if x_pss is not None:
                res_mla = mla_layer_rows(
                    x, x, layer.attention, c,
                    part_ss=x_pss, norm_w=ln_in.weight, norm_eps=ln_in.eps,
                    return_part_ss=is_dense_mlp,
                )
            else:
                h_in = mx.fast.rms_norm(x, ln_in.weight, ln_in.eps)
                res_mla = mla_layer_rows(h_in, x, layer.attention, c, return_part_ss=is_dense_mlp)
            if is_dense_mlp:
                h_post, pss_o = res_mla
            else:
                h_post, pss_o = res_mla, None

        ln_post = layer.post_attention_layernorm
        mlp = layer.mlp
        need_next_pss = (li + 1 < n_layers) and (not layers[li + 1].is_linear or use_kda_in_pss)
        if hasattr(mlp, "switch_mlp"):
            mlp_in = mx.fast.rms_norm(h_post, ln_post.weight, ln_post.eps)
            gate = mlp.gate
            se = mlp.shared_experts
            logits_r, se_mid = gemv_router_se_swiglu_rows(mlp_in, gate.weight, se.gate_proj, se.up_proj)
            if x.shape[0] >= 2:
                inds, wts, dedup = group_router_dedup_rows(
                    logits_r,
                    gate.expert_bias,
                    top_k=gate.top_k,
                    n_group=gate.n_group,
                    topk_group=gate.topk_group,
                    routed_scaling_factor=gate.routed_scaling_factor,
                )
            else:
                inds, wts = group_router_rows(
                    logits_r,
                    gate.expert_bias,
                    top_k=gate.top_k,
                    n_group=gate.n_group,
                    topk_group=gate.topk_group,
                    routed_scaling_factor=gate.routed_scaling_factor,
                )
                dedup = None
            sw = mlp.switch_mlp
            h_mid = moe_gate_up_swiglu_rows(
                mlp_in,
                sw.gate_proj.weight,
                sw.gate_proj.scales,
                sw.up_proj.weight,
                sw.up_proj.scales,
                inds,
                dedup=dedup,
            )
            sh = gemv_lin(se_mid, se.down_proj)
            x = moe_down_combine_rows(
                h_mid,
                sw.down_proj.weight,
                sw.down_proj.scales,
                inds,
                wts,
                shared=sh,
                residual=h_post,
                dedup=dedup,
            )
            x_pss = None
        else:
            if pss_o is not None:
                mid = gemv_lin_swiglu(
                    h_post, mlp.gate_proj, mlp.up_proj,
                    part_ss=pss_o, norm_w=ln_post.weight, norm_eps=ln_post.eps,
                )
            else:
                mlp_in = mx.fast.rms_norm(h_post, ln_post.weight, ln_post.eps)
                mid = gemv_lin_swiglu(mlp_in, mlp.gate_proj, mlp.up_proj)
            if need_next_pss:
                x, x_pss = gemv_lin_res(mid, mlp.down_proj, h_post, return_part_ss=True)
            else:
                x = gemv_lin_res(mid, mlp.down_proj, h_post)
                x_pss = None

        if _FAST_FLUSH_EVERY > 0 and (li + 1) % _FAST_FLUSH_EVERY == 0:
            mx.async_eval(x)

    x = mx.fast.rms_norm(x, inner.norm.weight, inner.norm.eps)
    return gemv_lin(x, model.lm_head)[:, None, :]
