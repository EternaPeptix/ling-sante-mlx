"""Bit-exact multi-token target verification and cache rollback for LingSante (1 <= C <= 4, C*L <= 24)."""
import os
from typing import Sequence

import mlx.core as mx

from mlx_lm.models.cache import ArraysCache, BatchKVCache, KVCache

from .rows.gemv import gemv_rows, gemv_rows_multi, gemv_rows_res, gemv_swiglu_rows
from .rows.kda import (
    kda_gdn_advance_only,
    kda_gdn_advance_split_only,
    kda_gdn_verify_seq_rows,
    kda_gdn_verify_seq_split_rows,
    kda_pre_seq_rows,
    kda_slice_conv_rows,
)
from .rows.mla import mla_verify_layer_rows
from .rows.moe import moe_dedup_rows, moe_down_combine_rows, moe_gate_up_swiglu_rows
from .rows.router import gate_rows

_VERIFY_FLUSH_EVERY = int(os.environ.get("LINGSANTE_VERIFY_FLUSH", "4"))


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
            if any(c.lengths is not None or c.left_padding is not None or any(c[j] is None for j in range(4)) for c in cs):
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
):
    """Run bit-exact multi-token verification forward for `(C, L)` with `C * L <= 24`.

    Does NOT mutate KDA cache states until `commit_verify_cache(...)` is called; MLA
    KV caches are updated in-place and trimmed back in `commit_verify_cache(...)`.
    """
    C, L = inputs.shape
    assert 1 <= C <= 4 and 1 <= L <= 24 and 1 <= C * L <= 24
    inner = model.model
    x = inner.word_embeddings(inputs.reshape(-1))  # (C*L, 4096)

    kda_records = []
    mla_caches = []
    use_dedup = C * L >= 5

    for li, (layer, c) in enumerate(zip(inner.layers, cache)):
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
            q, k, v, g, beta, h0, h1, h2 = kda_pre_seq_rows(
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
                C=C,
                L=L,
                lower_bound=attn.lower_bound,
                eps=1e-6,
            )
            if len(c.cache) > 4:
                s0 = c.cache[3:]
                ov, s_end = kda_gdn_verify_seq_split_rows(
                    q, k, v, g, beta, s0, g_r, attn.o_norm.weight, C=C, L=L, eps=attn.o_norm.eps
                )
            else:
                s0 = c[3]
                ov, s_end = kda_gdn_verify_seq_rows(
                    q, k, v, g, beta, s0, g_r, attn.o_norm.weight, C=C, L=L, eps=attn.o_norm.eps
                )
            if return_rollback:
                kda_records.append((c, s0, s_end, k, v, g, beta, h0, h1, h2))
            h_post = gemv_rows_res(ov, attn.o_proj.weight, x)
        else:
            h_post = mla_verify_layer_rows(h_in, x, layer.attention, c, C=C, L=L)
            if return_rollback:
                mla_caches.append(c)

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
            dedup = moe_dedup_rows(inds) if use_dedup else None
            h_mid = moe_gate_up_swiglu_rows(
                mlp_in,
                sw.gate_proj.weight,
                sw.gate_proj.scales,
                sw.up_proj.weight,
                sw.up_proj.scales,
                inds,
                tmax=moe_gu_tmax,
                dedup=dedup,
            )
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
        else:
            mid = gemv_swiglu_rows(mlp_in, mlp.gate_proj.weight, mlp.up_proj.weight)
            x = gemv_rows_res(mid, mlp.down_proj.weight, h_post)

        if _VERIFY_FLUSH_EVERY > 0 and (li + 1) % _VERIFY_FLUSH_EVERY == 0:
            mx.async_eval(x)

    hidden = mx.fast.rms_norm(x, inner.norm.weight, inner.norm.eps)
    logits = gemv_rows(hidden, model.lm_head.weight).reshape(C, L, -1)

    rb = {"C": C, "L": L, "kda": kda_records, "mla": mla_caches} if return_rollback else None
    if return_hidden and return_rollback:
        return logits, hidden.reshape(C, L, -1), rb
    if return_hidden:
        return logits, hidden.reshape(C, L, -1)
    if return_rollback:
        return logits, rb
    return logits


def commit_verify_cache(cache, rollback_ctx: dict, n_consumed: int | Sequence[int]):
    """Commit accepted/consumed prefix lengths `n_consumed` (`1 <= n <= L`) to KDA + MLA caches."""
    del cache  # cache references live directly in rollback_ctx
    C = rollback_ctx["C"]
    L = rollback_ctx["L"]
    if isinstance(n_consumed, int):
        nc_list = [n_consumed] * C
    else:
        nc_list = [int(v) for v in n_consumed]
        assert len(nc_list) == C
    for nc in nc_list:
        assert 1 <= nc <= L

    all_full = all(nc == L for nc in nc_list)
    same_nc = len(set(nc_list)) == 1
    nc_arr = None if all_full else mx.array(nc_list, dtype=mx.int32)
    same_nc_arr = mx.array([nc_list[0]] * C, dtype=mx.int32) if (C > 1 and same_nc) else None
    rej_idx = [ci for ci, nc in enumerate(nc_list) if nc < L]
    rej_sids = mx.array(rej_idx, dtype=mx.int32) if rej_idx else None
    rej_ncs = mx.array([nc_list[ci] for ci in rej_idx], dtype=mx.int32) if rej_idx else None

    for c, s0, s_end, k, v, g, beta, h0, h1, h2 in rollback_ctx["kda"]:
        if C == 1 or same_nc:
            nc0 = nc_list[0]
            if C == 1:
                c[0] = h0[:, nc0 : nc0 + 3, :]
                c[1] = h1[:, nc0 : nc0 + 3, :]
                c[2] = h2[:, nc0 : nc0 + 3, :]
            else:
                # Kernels produce contiguous (C, 3, 8192) across batch rows in 1 launch.
                c[0], c[1], c[2] = kda_slice_conv_rows(h0, h1, h2, same_nc_arr)
        else:
            c[0], c[1], c[2] = kda_slice_conv_rows(h0, h1, h2, nc_arr)

        if isinstance(s0, list):
            for ci, nc in enumerate(nc_list):
                if nc == L:
                    c[3 + ci] = s_end[ci]
            if rej_idx:
                outs_rej = kda_gdn_advance_split_only(
                    k, v, g, beta, [s0[ci] for ci in rej_idx], rej_sids, rej_ncs, L=L
                )
                for ci, o in zip(rej_idx, outs_rej):
                    c[3 + ci] = o
        elif all_full:
            c[3] = s_end
        else:
            c[3] = kda_gdn_advance_only(None, k, v, g, beta, s0, nc_arr, C=C, L=L)

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
            for ci, nc in zip(mc, nc_list):
                if nc < L:
                    ci.trim(L - nc)
        else:
            assert same_nc, "BatchKVCache requires list[KVCache] for heterogeneous rollback"
            if nc_list[0] < L:
                mc.trim(L - nc_list[0])
