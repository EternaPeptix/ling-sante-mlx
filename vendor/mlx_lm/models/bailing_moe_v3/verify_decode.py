"""Bit-exact multi-token target verification (uniform & varlen) and lazy/eager KDA+MLA commit for LingSante."""
import os
from typing import Sequence

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
from .rows.kda import (
    kda_conv_commit_rows,
    kda_fused_verify_rows,
    kda_gdn_advance_only,
    kda_gdn_advance_split_only,
    kda_gdn_commit_and_verify_seq_rows,
    kda_gdn_commit_and_verify_seq_split_rows,
    kda_gdn_commit_and_verify_varlen_rows,
    kda_gdn_commit_and_verify_varlen_split_rows,
    kda_gdn_verify_seq_rows,
    kda_gdn_verify_seq_split_rows,
    kda_gdn_verify_varlen_rows,
    kda_gdn_verify_varlen_split_rows,
    kda_pre_seq_rows,
    kda_pre_varlen_rows,
)
from .rows.mla import mla_verify_layer_rows, mla_verify_varlen_rows
from .rows.moe import (
    moe_down_combine_packed_rows,
    moe_down_combine_rows,
    moe_gate_up_swiglu_packed_rows,
    moe_gate_up_swiglu_rows,
)
from .rows.router import group_router_dedup_rows, group_router_rows

_VERIFY_FLUSH_EVERY = int(os.environ.get("LINGSANTE_VERIFY_FLUSH", "1"))
_USE_LAZY_KDA = os.environ.get("LINGSANTE_LAZY_KDA", "1") not in ("0", "")
_USE_FUSED_KDA = os.environ.get("LINGSANTE_FUSED_KDA", "1") not in ("0", "")
_USE_PARTSS_KDA_IN = os.environ.get("LINGSANTE_PARTSS_KDA_IN", "0") not in ("0", "")
_USE_T7_SE_MAX_B = int(os.environ.get("LINGSANTE_T7_SE_MAX_B", "0"))
_KDA_CV_NSG = int(os.environ.get("LINGSANTE_KDA_CV_NSG", "1"))


def make_verify_cache(model, per_stream_caches: list[list]):
    """Build verification cache from per-stream `B=1` caches.

    When `C > 1`:
      - KDA layers use `ArraysCache(3 + C)` merging `conv[0..2]` while storing per-stream
        GDN state references in slots `3..3+C-1` so accepted streams need zero state copy.
      - MLA layers keep unpadded `list[KVCache]` so per-stream speculative trim is O(1).
    """
    C = len(per_stream_caches)
    if C == 1:
        return per_stream_caches[0]
    layers = model.model.layers
    out = []
    for li, layer in enumerate(layers):
        cs = [sc[li] for sc in per_stream_caches]
        if layer.is_linear:
            if any(
                c.lengths is not None or c.left_padding is not None or any(c[j] is None for j in range(4))
                for c in cs
            ):
                out.append(ArraysCache.merge(cs))
            else:
                ac = ArraysCache(3 + C)
                for e in range(3):
                    ac[e] = mx.concatenate([c[e] for c in cs], axis=0)
                for i in range(C):
                    ac[3 + i] = cs[i][3]
                out.append(ac)
        else:
            out.append(cs)
    return out


def materialize_lazy_kda(cache) -> None:
    """Eagerly materialize any pending lazy KDA GDN commits attached to `ArraysCache` entries."""
    if not cache:
        return
    for c in cache:
        if not isinstance(c, ArraysCache):
            continue
        lazy = getattr(c, "_lazy_gdn", None)
        if lazy is None:
            continue
        c._lazy_gdn = None
        s0, k, v, g, beta, off_arr, nc_arr, C_prev, L_prev, nc_list = lazy
        if isinstance(s0, list):
            sids = mx.arange(C_prev, dtype=mx.int32)
            outs = kda_gdn_advance_split_only(
                k,
                v,
                g,
                beta,
                s0,
                sids,
                nc_arr,
                L=0 if off_arr is not None else L_prev,
                stream_off=off_arr,
            )
            for ci, o in enumerate(outs):
                c[3 + ci] = o
        else:
            c[3] = kda_gdn_advance_only(
                None,
                k,
                v,
                g,
                beta,
                s0,
                nc_arr,
                C=C_prev,
                L=0 if off_arr is not None else L_prev,
                stream_off=off_arr,
            )


def can_fast_verify(model, inputs: mx.array, cache) -> bool:
    if os.environ.get("LINGSANTE_FAST_VERIFY", "1") in ("0", ""):
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
    if model.training or model.lm_head is None or inputs.ndim != 2:
        return False
    C, L = inputs.shape
    if not (1 <= C <= 4 and 1 <= L <= 24 and 1 <= C * L <= 24):
        return False
    layers = model.model.layers
    if cache is None or len(cache) != len(layers):
        return False
    for layer, c in zip(layers, cache):
        if layer.is_linear:
            if not isinstance(c, ArraysCache) or c.lengths is not None:
                return False
            if c[0] is None or c[1] is None or c[2] is None:
                return False
            n_st = len(c.cache)
            if n_st == 4:
                if c[3] is None:
                    return False
            elif C > 1 and n_st == 3 + C:
                if any(s is None for s in c.cache[3:]):
                    return False
            else:
                return False
        else:
            if isinstance(c, list):
                if len(c) != C or any(not isinstance(ci, KVCache) or ci.keys is None for ci in c):
                    return False
            elif isinstance(c, BatchKVCache):
                if c.keys is None or c.values is None or getattr(c, "_right_padding", None) is not None:
                    return False
            elif isinstance(c, KVCache):
                if C != 1 or c.keys is None or c.values is None:
                    return False
            else:
                return False
    return True


def _fast_verify_impl(
    model,
    inputs_flat: mx.array,
    cache,
    *,
    C: int,
    L: int | None,
    Ls: list[int] | None,
    return_hidden: bool = True,
    return_rollback: bool = True,
    moe_dc_tmax: int | None = None,
    moe_dc_nsg: int | None = None,
    capture_layers: tuple[int, ...] | None = None,
    lazy_kda: bool = False,
):
    varlen = Ls is not None
    if varlen:
        assert 1 <= C <= 4 and len(Ls) == C and all(lc >= 1 for lc in Ls)
        B = sum(Ls)
        assert 1 <= B <= 24 and inputs_flat.size == B
        row_starts = []
        pos = 0
        for lc in Ls:
            row_starts.append(pos)
            pos += lc
        stream_off = mx.array(row_starts, dtype=mx.int32)
        stream_L = mx.array(Ls, dtype=mx.int32)
        stream_id = mx.array([c_i for c_i, lc in enumerate(Ls) for _ in range(lc)], dtype=mx.int32)
        step_in_stream = mx.array([t for lc in Ls for t in range(lc)], dtype=mx.int32)
    else:
        assert L is not None and 1 <= C <= 4 and 1 <= L <= 24 and 1 <= C * L <= 24
        B = C * L
        row_starts = None
        stream_off = None
        stream_L = None
        stream_id = None
        step_in_stream = None

    use_lazy_write = bool(lazy_kda and _USE_LAZY_KDA)
    inner = model.model
    layers = inner.layers
    n_layers = len(layers)
    x = inner.word_embeddings(inputs_flat.reshape(-1))  # (B, 4096)
    x_pss = None

    kda_records = []
    mla_caches = []
    cap_set = set(capture_layers) if capture_layers else None
    captured = {}

    mla_offs_key = None
    mla_offs_arr = None

    for li, (layer, c) in enumerate(zip(layers, cache)):
        ln_in = layer.input_layernorm
        if layer.is_linear:
            attn = layer.attention
            use_pss_in = (x_pss is not None) and _USE_PARTSS_KDA_IN
            if hasattr(attn, "_w_qkvfg"):
                if use_pss_in:
                    q_r, k_r, v_r, f_r, g_r, b_r = gemv_kda_qkvfgb_merged_rows(
                        x,
                        attn._w_qkvfg,
                        attn._s_qkvfg,
                        attn.b_proj.weight,
                        part_ss=x_pss,
                        norm_w=ln_in.weight,
                        norm_eps=ln_in.eps,
                    )
                else:
                    h_in = mx.fast.rms_norm(x, ln_in.weight, ln_in.eps)
                    q_r, k_r, v_r, f_r, g_r, b_r = gemv_kda_qkvfgb_merged_rows(
                        h_in, attn._w_qkvfg, attn._s_qkvfg, attn.b_proj.weight
                    )
            else:
                qkv_mods = [
                    attn.q_proj,
                    attn.k_proj,
                    attn.v_proj,
                    attn.f_proj,
                    attn.g_proj,
                    attn.b_proj,
                ]
                if use_pss_in:
                    q_r, k_r, v_r, f_r, g_r, b_r = gemv_lin_multi(
                        x, qkv_mods, part_ss=x_pss, norm_w=ln_in.weight, norm_eps=ln_in.eps
                    )
                else:
                    h_in = mx.fast.rms_norm(x, ln_in.weight, ln_in.eps)
                    q_r, k_r, v_r, f_r, g_r, b_r = gemv_lin_multi(h_in, qkv_mods)

            cs0, cs1, cs2 = c[0], c[1], c[2]
            lazy_prev = getattr(c, "_lazy_gdn", None)
            is_split = len(c.cache) > 4
            max_l = max(Ls) if varlen else L
            use_mega = _USE_FUSED_KDA and (1 <= max_l <= 16) and not (is_split and lazy_prev is not None)

            if use_mega:
                if lazy_prev is not None:
                    p_s0, p_k, p_v, p_g, p_beta, p_off, p_nc, p_C, p_L, _ = lazy_prev
                    if p_C == C and (isinstance(p_s0, list) == is_split):
                        c._lazy_gdn = None
                        p_off_arg = p_off if p_off is not None else mx.array([ci * p_L for ci in range(C)], dtype=mx.int32)
                        emit_kvgb = bool(return_rollback)
                        ov, s_comm, s_end, k, v, g, beta = kda_fused_verify_rows(
                            q_r, k_r, v_r, f_r, g_r, b_r, cs0, cs1, cs2,
                            attn.q_conv1d.weight, attn.k_conv1d.weight, attn.v_conv1d.weight,
                            attn.A_log, attn.dt_bias, p_s0, attn.o_norm.weight,
                            C=C, L=L, Ls=Ls, lower_bound=attn.lower_bound, eps=1e-6, o_eps=attn.o_norm.eps,
                            write_end=False, emit_kvgb=emit_kvgb,
                            prev_k=p_k, prev_v=p_v, prev_g=p_g, prev_beta=p_beta, prev_off=p_off_arg, prev_nc=p_nc,
                            stream_off=stream_off, stream_L=stream_L,
                        )
                        if is_split:
                            for ci in range(C):
                                c[3 + ci] = s_comm[ci]
                        else:
                            c[3] = s_comm
                        s0 = s_comm
                    else:
                        materialize_lazy_kda([c])
                        lazy_prev = None

                if lazy_prev is None:
                    write_end = not use_lazy_write
                    s0 = list(c.cache[3:]) if is_split else c[3]
                    emit_kvgb = bool(return_rollback and (use_lazy_write or max_l > 1))
                    ov, _, s_end, k, v, g, beta = kda_fused_verify_rows(
                        q_r, k_r, v_r, f_r, g_r, b_r, cs0, cs1, cs2,
                        attn.q_conv1d.weight, attn.k_conv1d.weight, attn.v_conv1d.weight,
                        attn.A_log, attn.dt_bias, s0, attn.o_norm.weight,
                        C=C, L=L, Ls=Ls, lower_bound=attn.lower_bound, eps=1e-6, o_eps=attn.o_norm.eps,
                        write_end=write_end, emit_kvgb=emit_kvgb,
                        stream_off=stream_off, stream_L=stream_L,
                    )
            else:
                if varlen:
                    q, k, v, g, beta = kda_pre_varlen_rows(
                        q_r,
                        k_r,
                        v_r,
                        f_r,
                        b_r,
                        cs0,
                        cs1,
                        cs2,
                        attn.q_conv1d.weight,
                        attn.k_conv1d.weight,
                        attn.v_conv1d.weight,
                        attn.A_log,
                        attn.dt_bias,
                        Ls,
                        lower_bound=attn.lower_bound,
                        eps=1e-6,
                        stream_id=stream_id,
                        step_in_stream=step_in_stream,
                    )
                else:
                    q, k, v, g, beta = kda_pre_seq_rows(
                        q_r,
                        k_r,
                        v_r,
                        f_r,
                        b_r,
                        cs0,
                        cs1,
                        cs2,
                        attn.q_conv1d.weight,
                        attn.k_conv1d.weight,
                        attn.v_conv1d.weight,
                        attn.A_log,
                        attn.dt_bias,
                        C=C,
                        L=L,
                        lower_bound=attn.lower_bound,
                        eps=1e-6,
                        emit_hist=False,
                    )

                if lazy_prev is not None:
                    p_s0, p_k, p_v, p_g, p_beta, p_off, p_nc, p_C, p_L, _ = lazy_prev
                    if p_C == C and (isinstance(p_s0, list) == is_split):
                        c._lazy_gdn = None
                        p_off_arg = p_off if p_off is not None else mx.array([ci * p_L for ci in range(C)], dtype=mx.int32)
                        if varlen:
                            if is_split:
                                ov, s_comm = kda_gdn_commit_and_verify_varlen_split_rows(
                                    q, k, v, g, beta, p_s0, p_k, p_v, p_g, p_beta, p_off_arg, p_nc,
                                    g_r, attn.o_norm.weight, Ls, eps=attn.o_norm.eps, nsg=_KDA_CV_NSG, write_end=False,
                                    stream_off=stream_off, stream_L=stream_L,
                                )
                                for ci in range(C):
                                    c[3 + ci] = s_comm[ci]
                            else:
                                ov, s_comm = kda_gdn_commit_and_verify_varlen_rows(
                                    q, k, v, g, beta, p_s0, p_k, p_v, p_g, p_beta, p_off_arg, p_nc,
                                    g_r, attn.o_norm.weight, Ls, eps=attn.o_norm.eps, nsg=_KDA_CV_NSG, write_end=False,
                                    stream_off=stream_off, stream_L=stream_L,
                                )
                                c[3] = s_comm
                        else:
                            if is_split:
                                ov, s_comm = kda_gdn_commit_and_verify_seq_split_rows(
                                    q, k, v, g, beta, p_s0, p_k, p_v, p_g, p_beta, p_off_arg, p_nc,
                                    g_r, attn.o_norm.weight, C=C, L=L, eps=attn.o_norm.eps, nsg=_KDA_CV_NSG, write_end=False,
                                )
                                for ci in range(C):
                                    c[3 + ci] = s_comm[ci]
                            else:
                                ov, s_comm = kda_gdn_commit_and_verify_seq_rows(
                                    q, k, v, g, beta, p_s0, p_k, p_v, p_g, p_beta, p_off_arg, p_nc,
                                    g_r, attn.o_norm.weight, C=C, L=L, eps=attn.o_norm.eps, nsg=_KDA_CV_NSG, write_end=False,
                                )
                                c[3] = s_comm
                        s0 = s_comm
                        s_end = None
                    else:
                        materialize_lazy_kda([c])
                        lazy_prev = None

                if lazy_prev is None:
                    write_end = not use_lazy_write
                    if is_split:
                        s0 = list(c.cache[3:])
                        if varlen:
                            res_gdn = kda_gdn_verify_varlen_split_rows(
                                q, k, v, g, beta, s0, g_r, attn.o_norm.weight, Ls,
                                eps=attn.o_norm.eps, write_end=write_end,
                                stream_off=stream_off, stream_L=stream_L,
                            )
                        else:
                            res_gdn = kda_gdn_verify_seq_split_rows(
                                q, k, v, g, beta, s0, g_r, attn.o_norm.weight,
                                C=C, L=L, eps=attn.o_norm.eps, write_end=write_end,
                            )
                    else:
                        s0 = c[3]
                        if varlen:
                            res_gdn = kda_gdn_verify_varlen_rows(
                                q, k, v, g, beta, s0, g_r, attn.o_norm.weight, Ls,
                                eps=attn.o_norm.eps, write_end=write_end,
                                stream_off=stream_off, stream_L=stream_L,
                            )
                        else:
                            res_gdn = kda_gdn_verify_seq_rows(
                                q, k, v, g, beta, s0, g_r, attn.o_norm.weight,
                                C=C, L=L, eps=attn.o_norm.eps, write_end=write_end,
                            )
                    if write_end:
                        ov, s_end = res_gdn
                    else:
                        ov, s_end = res_gdn, None

            if return_rollback:
                kda_records.append((c, s0, s_end, q_r, k_r, v_r, k, v, g, beta, cs0, cs1, cs2))

            is_dense_mlp = not hasattr(layer.mlp, "switch_mlp")
            if is_dense_mlp:
                h_post, pss_o = gemv_lin_res(ov, attn.o_proj, x, return_part_ss=True)
            else:
                h_post = gemv_lin_res(ov, attn.o_proj, x)
                pss_o = None
        else:
            # Share MLA RoPE offsets array across all 9 MLA layers when per-stream offsets match
            if isinstance(c, list):
                key_o = tuple(int(ci.offset) for ci in c)
                if key_o != mla_offs_key:
                    mla_offs_key = key_o
                    if varlen:
                        mla_offs_arr = mx.array(
                            [key_o[ci] + t for ci in range(C) for t in range(Ls[ci])], dtype=mx.int32
                        )
                    else:
                        mla_offs_arr = mx.array(
                            [o + t for o in key_o for t in range(L)], dtype=mx.int32
                        )
            elif not hasattr(c, "left_padding"):
                key_o = (int(c.offset),)
                if key_o != mla_offs_key:
                    mla_offs_key = key_o
                    n_tok = Ls[0] if varlen else L
                    mla_offs_arr = mx.arange(key_o[0], key_o[0] + n_tok, dtype=mx.int32)
            else:
                mla_offs_arr = None

            if x_pss is not None:
                h_in_mla = x
                mla_pss_kw = {"part_ss": x_pss, "norm_w": ln_in.weight, "norm_eps": ln_in.eps}
            else:
                h_in_mla = mx.fast.rms_norm(x, ln_in.weight, ln_in.eps)
                mla_pss_kw = {}

            is_dense_mlp = not hasattr(layer.mlp, "switch_mlp")
            if varlen:
                c_list = c if isinstance(c, list) else [c]
                res_mla = mla_verify_varlen_rows(
                    h_in_mla, x, layer.attention, c_list, Ls,
                    offsets=mla_offs_arr, return_part_ss=is_dense_mlp, **mla_pss_kw,
                )
            else:
                res_mla = mla_verify_layer_rows(
                    h_in_mla, x, layer.attention, c, C=C, L=L,
                    offsets=mla_offs_arr, return_part_ss=is_dense_mlp, **mla_pss_kw,
                )
            if is_dense_mlp:
                h_post, pss_o = res_mla
            else:
                h_post, pss_o = res_mla, None
            if return_rollback:
                mla_caches.append(c)

        ln_post = layer.post_attention_layernorm
        mlp = layer.mlp
        need_next_pss = (li + 1 < n_layers) and (not layers[li + 1].is_linear or _USE_PARTSS_KDA_IN)

        if hasattr(mlp, "switch_mlp"):
            mlp_in = mx.fast.rms_norm(h_post, ln_post.weight, ln_post.eps)
            gate = mlp.gate
            se = mlp.shared_experts
            logits_r, se_mid = gemv_router_se_swiglu_rows(mlp_in, gate.weight, se.gate_proj, se.up_proj)
            if B >= 2:
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
            use_packed = hasattr(sw, "_gu_packed")
            if use_packed:
                h_mid = moe_gate_up_swiglu_packed_rows(mlp_in, sw._gu_packed, inds, dedup=dedup)
            else:
                h_mid = moe_gate_up_swiglu_rows(
                    mlp_in,
                    sw.gate_proj.weight,
                    sw.gate_proj.scales,
                    sw.up_proj.weight,
                    sw.up_proj.scales,
                    inds,
                    dedup=dedup,
                )
            use_t7 = (B <= _USE_T7_SE_MAX_B) and hasattr(se.down_proj, "weight_fp8")
            if use_t7:
                if use_packed:
                    x = moe_down_combine_packed_rows(
                        h_mid,
                        sw._dn_packed,
                        inds,
                        wts,
                        residual=h_post,
                        tmax=moe_dc_tmax,
                        nsg=moe_dc_nsg,
                        dedup=dedup,
                        se_mid=se_mid,
                        se_w_fp8=se.down_proj.weight_fp8,
                        se_sc=se.down_proj.weight_scale_inv,
                    )
                else:
                    x = moe_down_combine_rows(
                        h_mid,
                        sw.down_proj.weight,
                        sw.down_proj.scales,
                        inds,
                        wts,
                        residual=h_post,
                        tmax=moe_dc_tmax,
                        nsg=moe_dc_nsg,
                        dedup=dedup,
                        se_mid=se_mid,
                        se_w_fp8=se.down_proj.weight_fp8,
                        se_sc=se.down_proj.weight_scale_inv,
                    )
            else:
                sh = gemv_lin(se_mid, se.down_proj)
                if use_packed:
                    x = moe_down_combine_packed_rows(
                        h_mid,
                        sw._dn_packed,
                        inds,
                        wts,
                        shared=sh,
                        residual=h_post,
                        tmax=moe_dc_tmax,
                        nsg=moe_dc_nsg,
                        dedup=dedup,
                    )
                else:
                    x = moe_down_combine_rows(
                        h_mid,
                        sw.down_proj.weight,
                        sw.down_proj.scales,
                        inds,
                        wts,
                        shared=sh,
                        residual=h_post,
                        tmax=moe_dc_tmax,
                        nsg=moe_dc_nsg,
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

        if cap_set is not None and li in cap_set:
            captured[li] = x

        if _VERIFY_FLUSH_EVERY > 0 and (li + 1) % _VERIFY_FLUSH_EVERY == 0:
            mx.async_eval(x)

    hidden = mx.fast.rms_norm(x, inner.norm.weight, inner.norm.eps)
    logits = gemv_lin(hidden, model.lm_head)
    if not varlen:
        logits = logits.reshape(C, L, -1)
        hidden_out = hidden.reshape(C, L, -1)
    else:
        hidden_out = hidden

    if return_rollback:
        rb = {
            "C": C,
            "L": L,
            "Ls": list(Ls) if varlen else None,
            "row_starts": row_starts,
            "stream_off": stream_off,
            "kda": kda_records,
            "mla": mla_caches,
        }
        if capture_layers:
            cap_cat = mx.concatenate([captured[li] for li in capture_layers], axis=-1)
            rb["captured"] = cap_cat if varlen else cap_cat.reshape(C, L, -1)
    else:
        rb = None

    if return_hidden and return_rollback:
        return logits, hidden_out, rb
    if return_hidden:
        return logits, hidden_out
    if return_rollback:
        return logits, rb
    return logits


def fast_verify_forward(
    model,
    inputs: mx.array,
    cache,
    *,
    return_hidden: bool = True,
    return_rollback: bool = True,
    moe_gu_tmax: int | None = None,
    moe_dc_tmax: int | None = None,
    moe_dc_nsg: int | None = None,
    capture_layers: tuple[int, ...] | None = None,
    lazy_kda: bool = False,
):
    """Run bit-exact multi-token verification forward for `(C, L)` with `C * L <= 24`."""
    del moe_gu_tmax
    C, L = inputs.shape
    return _fast_verify_impl(
        model,
        inputs.reshape(-1),
        cache,
        C=C,
        L=L,
        Ls=None,
        return_hidden=return_hidden,
        return_rollback=return_rollback,
        moe_dc_tmax=moe_dc_tmax,
        moe_dc_nsg=moe_dc_nsg,
        capture_layers=capture_layers,
        lazy_kda=lazy_kda,
    )


def fast_verify_forward_varlen(
    model,
    inputs_flat: mx.array,
    cache,
    Ls: list[int],
    *,
    return_hidden: bool = True,
    return_rollback: bool = True,
    moe_dc_tmax: int | None = None,
    moe_dc_nsg: int | None = None,
    capture_layers: tuple[int, ...] | None = None,
    lazy_kda: bool = False,
):
    """Run bit-exact heterogeneous variable-length verification forward for `Ls` (`sum(Ls) <= 24`)."""
    return _fast_verify_impl(
        model,
        inputs_flat.reshape(-1),
        cache,
        C=len(Ls),
        L=None,
        Ls=list(Ls),
        return_hidden=return_hidden,
        return_rollback=return_rollback,
        moe_dc_tmax=moe_dc_tmax,
        moe_dc_nsg=moe_dc_nsg,
        capture_layers=capture_layers,
        lazy_kda=lazy_kda,
    )


def commit_verify_cache(
    cache,
    rollback_ctx: dict,
    n_consumed: int | Sequence[int],
    *,
    lazy: bool = False,
):
    """Commit accepted/consumed prefix lengths `n_consumed` (`1 <= nc_i <= L_i`) to KDA + MLA caches."""
    del cache  # cache references live directly in rollback_ctx
    C = rollback_ctx["C"]
    Ls = rollback_ctx.get("Ls")
    varlen = Ls is not None
    L = rollback_ctx.get("L")
    if isinstance(n_consumed, int):
        nc_list = [n_consumed] * C
    else:
        nc_list = [int(v) for v in n_consumed]
        assert len(nc_list) == C
    if varlen:
        for nc, lc in zip(nc_list, Ls):
            assert 1 <= nc <= lc
        all_full = all(nc == lc for nc, lc in zip(nc_list, Ls))
        stream_off = rollback_ctx.get("stream_off")
        if stream_off is None:
            row_starts = rollback_ctx["row_starts"]
            stream_off = mx.array(row_starts, dtype=mx.int32)
    else:
        for nc in nc_list:
            assert 1 <= nc <= L
        all_full = all(nc == L for nc in nc_list)
        stream_off = None

    same_nc = len(set(nc_list)) == 1
    nc_arr = mx.array(nc_list, dtype=mx.int32)
    use_lazy = bool(lazy and _USE_LAZY_KDA)

    if varlen:
        rej_idx = [ci for ci, (nc, lc) in enumerate(zip(nc_list, Ls)) if nc < lc]
    else:
        rej_idx = [ci for ci, nc in enumerate(nc_list) if nc < L]
    rej_sids = mx.array(rej_idx, dtype=mx.int32) if rej_idx else None
    rej_ncs = mx.array([nc_list[ci] for ci in rej_idx], dtype=mx.int32) if rej_idx else None
    all_sids = mx.arange(C, dtype=mx.int32) if C > 1 else None

    for c, s0, s_end, q_r, k_r, v_r, k, v, g, beta, cs0, cs1, cs2 in rollback_ctx["kda"]:
        # Always commit tiny conv states immediately (0 hist0..2 allocations)
        c[0], c[1], c[2] = kda_conv_commit_rows(
            cs0,
            cs1,
            cs2,
            q_r,
            k_r,
            v_r,
            nc_arr,
            L=None if varlen else L,
            stream_off=stream_off,
        )

        # GDN state: defer into next verify cycle when lazy=True, else commit immediately
        if use_lazy and s_end is None:
            c._lazy_gdn = (s0, k, v, g, beta, stream_off, nc_arr, C, L, nc_list)
        elif use_lazy and not all_full:
            c._lazy_gdn = (s0, k, v, g, beta, stream_off, nc_arr, C, L, nc_list)
        else:
            c._lazy_gdn = None
            if isinstance(s0, list):
                if s_end is not None:
                    for ci, nc in enumerate(nc_list):
                        if nc == (Ls[ci] if varlen else L):
                            c[3 + ci] = s_end[ci]
                    if rej_idx:
                        outs_rej = kda_gdn_advance_split_only(
                            k,
                            v,
                            g,
                            beta,
                            [s0[ci] for ci in rej_idx],
                            rej_sids,
                            rej_ncs,
                            L=0 if varlen else L,
                            stream_off=stream_off,
                        )
                        for ci, o in zip(rej_idx, outs_rej):
                            c[3 + ci] = o
                else:
                    outs_all = kda_gdn_advance_split_only(
                        k,
                        v,
                        g,
                        beta,
                        s0,
                        all_sids,
                        nc_arr,
                        L=0 if varlen else L,
                        stream_off=stream_off,
                    )
                    for ci, o in enumerate(outs_all):
                        c[3 + ci] = o
            elif all_full and s_end is not None:
                c[3] = s_end
            else:
                c[3] = kda_gdn_advance_only(
                    None,
                    k,
                    v,
                    g,
                    beta,
                    s0,
                    nc_arr,
                    C=C,
                    L=0 if varlen else L,
                    stream_off=stream_off,
                )

        if c.lengths is not None or c.left_padding is not None:
            if same_nc:
                c.advance(nc_list[0])
            else:
                if c.lengths is not None:
                    c.lengths = c.lengths - nc_arr
                if c.left_padding is not None:
                    c.left_padding = c.left_padding - nc_arr

    for mc in rollback_ctx["mla"]:
        if isinstance(mc, list):
            for ci_idx, (ci, nc) in enumerate(zip(mc, nc_list)):
                lc = Ls[ci_idx] if varlen else L
                if nc < lc:
                    ci.trim(lc - nc)
        else:
            assert same_nc and not varlen, "BatchKVCache requires list[KVCache] for heterogeneous rollback"
            if nc_list[0] < L:
                mc.trim(L - nc_list[0])


def commit_verify_cache_varlen(
    cache,
    rollback_ctx: dict,
    n_consumed: int | Sequence[int],
    *,
    lazy: bool = False,
):
    """Commit heterogeneous variable-length verification prefixes."""
    commit_verify_cache(cache, rollback_ctx, n_consumed, lazy=lazy)
