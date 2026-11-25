"""Single-sync GPU-chained speculative decoding engine for LingSante (BailingMoeV3 + EAGLE-3 / MTP, C=1..4)."""
import os
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import mlx.core as mx

from mlx_lm.layers.attention import create_attention_mask
from mlx_lm.layers.recurrent import create_ssm_mask
from mlx_lm.models.cache import ArraysCache, BatchKVCache, KVCache, make_prompt_cache

from . import verify_decode as _vd
from .mtp import BailingMTPLayer
from .suffix import SuffixMatcher
from .verify_decode import can_fast_verify, commit_verify_cache, fast_verify_forward, make_verify_cache


def _has_varlen_verify() -> bool:
    """True when verify_decode exposes heterogeneous varlen verification (`fast_verify_forward_varlen`)."""
    if os.environ.get("LINGSANTE_VARLEN_SPEC", "1") in ("0", ""):
        return False
    return hasattr(_vd, "fast_verify_forward_varlen")


def _varlen_bmax() -> int:
    return max(4, min(24, int(os.environ.get("LINGSANTE_VARLEN_BMAX", "8"))))


def _use_mtp_suffix() -> bool:
    return os.environ.get("LINGSANTE_MTP_SUFFIX", "1") not in ("0", "")


def _suffix_min_gram() -> int:
    return max(3, int(os.environ.get("LINGSANTE_SUFFIX_MIN_GRAM", "5")))


class _MTPBox:
    """Plain Python container so attaching MTP does not register it as an nn.Module child."""

    __slots__ = ("mtp",)

    def __init__(self, mtp: BailingMTPLayer):
        self.mtp = mtp


def attach_lingsante_mtp(model, model_path: str | Path) -> bool:
    """Attach EAGLE-3 Q4-Dense MTP (default if present) or native Layer-72 MTP drafter."""
    if os.environ.get("LINGSANTE_MTP", "1") in ("0", ""):
        return False
    if "bailing_moe_v3" not in type(model).__module__:
        return False
    args = getattr(model, "args", None)
    if (
        args is None
        or args.hidden_size != 4096
        or args.num_attention_heads != 64
        or args.num_experts != 512
        or args.moe_intermediate_size != 1280
        or getattr(model, "lm_head", None) is None
    ):
        return False

    if os.environ.get("LINGSANTE_EAGLE3", "1") not in ("0", ""):
        env_dir = os.environ.get("LINGSANTE_EAGLE3_DIR", "").strip()
        e3_dir = Path(env_dir).expanduser() if env_dir else (Path(model_path) / "eagle3-mtp")
        if (e3_dir / "model-mtp.safetensors").exists():
            mtp = BailingMTPLayer.from_eagle3(model, e3_dir, with_q4_head=True, q4_dense=True)
            model._lingsante_mtp_holder = _MTPBox(mtp)
            return True

    mtp_file = Path(model_path) / "mtp" / "model-mtp.safetensors"
    if not mtp_file.exists():
        return False
    mtp = BailingMTPLayer.from_target(model, model_path, with_q4_head=True)
    model._lingsante_mtp_holder = _MTPBox(mtp)
    return True


def get_lingsante_mtp(model) -> BailingMTPLayer | None:
    """Return attached MTP drafter when all fast-path flags allow speculative decode."""
    if (
        os.environ.get("LINGSANTE_MTP", "1") in ("0", "")
        or os.environ.get("LINGSANTE_FAST_DECODE", "1") in ("0", "")
        or os.environ.get("LINGSANTE_FAST_VERIFY", "1") in ("0", "")
    ):
        return None
    holder = getattr(model, "_lingsante_mtp_holder", None)
    return holder.mtp if holder is not None else None


def mtp_k_for_batch(C: int, mtp: BailingMTPLayer | None = None) -> int:
    """Return default fixed K per stream respecting the B<=8 FP8 GEMV cliff when varlen is absent."""
    cfg = getattr(mtp, "_eagle3_cfg", None) or {}
    if C <= 1:
        return max(1, int(os.environ.get("LINGSANTE_MTP_K1", str(cfg.get("fixed_K1", 4)))))
    if "LINGSANTE_MTP_KB" in os.environ:
        return max(0, int(os.environ["LINGSANTE_MTP_KB"]))
    # Dodge B>=9 FP8 bsg=2 cliff: C=2 -> K=2 (B=6) or dyn_step K<=3 (B<=8); C=3,4 -> K=1 (B=6,8).
    bmax = _varlen_bmax()
    if C >= 3:
        return max(1, (bmax // C) - 1)
    return max(1, min(int(cfg.get("fixed_KB", 2)), (bmax // C) - 1))


def mtp_c1_policy(mtp: BailingMTPLayer) -> tuple[str, int, float, float]:
    """Return `(draft_mode, k_or_cap, dyn_th, p1_min)` for C=1 speculative drafting."""
    cfg = getattr(mtp, "_eagle3_cfg", None) or {}
    use_dyn = os.environ.get("LINGSANTE_MTP_DYN", "1") not in ("0", "")
    if (
        use_dyn
        and "LINGSANTE_MTP_K1" not in os.environ
        and cfg.get("draft_mode") == "dyn_step"
    ):
        k_cap = max(1, int(cfg.get("k_cap", 7)))
        dyn_th = float(os.environ.get("LINGSANTE_DYN_TH", cfg.get("dyn_th", 0.55)))
        p1_min = float(os.environ.get("LINGSANTE_P1_MIN", cfg.get("p1_min", 0.0)))
        return "dyn_step", k_cap, dyn_th, p1_min
    return "fixed", mtp_k_for_batch(1, mtp), 0.0, 0.0


def _c2_use_dyn(mtp: BailingMTPLayer) -> bool:
    if os.environ.get("LINGSANTE_C2_DYN", "0") in ("0", ""):
        return False
    if "LINGSANTE_MTP_KB" in os.environ:
        return False
    cfg = getattr(mtp, "_eagle3_cfg", None) or {}
    return cfg.get("draft_mode") == "dyn_step"


def mtp_target_prefill_hidden(
    model, mtp: BailingMTPLayer, chunk_2d: mx.array, cache, *, return_raw: bool = False
):
    """Run target backbone for prefill and return fused MTP input hidden (and optional raw h_final)."""
    lids = getattr(mtp, "_feat_layers", None)
    if getattr(mtp, "_feat_fuse", None) is None or not lids:
        h_raw = model.model(chunk_2d, cache=cache)
        return (h_raw, h_raw) if return_raw else h_raw

    inner = model.model
    cap_set = set(lids)
    h = inner.word_embeddings(chunk_2d)
    attention_mask = (
        create_attention_mask(h, cache[inner.attention_index], return_array=True)
        if inner.attention_index is not None
        else None
    )
    linear_mask = (
        create_ssm_mask(h, cache[inner.linear_index])
        if inner.linear_index is not None
        else None
    )
    captured = {}
    for li, (layer, lc) in enumerate(zip(inner.layers, cache)):
        mask = linear_mask if layer.is_linear else attention_mask
        h = layer(h, mask=mask, cache=lc)
        if li in cap_set:
            captured[li] = h
    h_final = inner.norm(h)
    feats = mx.concatenate([captured[li] for li in lids], axis=-1)
    h_eff = mtp.apply_feat_fuse(h_final, feats)
    return (h_final, h_eff) if return_raw else h_eff


def is_greedy_sampler(sampler) -> bool:
    """Return True if `sampler` behaves deterministically like argmax on logprobs."""
    if sampler is None:
        return True
    flag = getattr(sampler, "_is_greedy", None)
    if flag is not None:
        return bool(flag)
    saved_rng = mx.array(mx.random.state[0])
    mx.eval(saved_rng)
    try:
        probe = mx.full((4, 32), -1e-4, dtype=mx.float32)
        idx = mx.arange(4)
        probe = probe.at[idx, idx].add(1e-4)
        a = sampler(probe)
        b = sampler(probe)
        mx.eval(a, b, mx.random.state[0])
        same_rng = bool(mx.array_equal(mx.random.state[0], saved_rng).item())
        ok = (
            same_rng
            and a.ndim == 1
            and a.shape[0] == 4
            and a.tolist() == [0, 1, 2, 3]
            and b.tolist() == [0, 1, 2, 3]
        )
    except Exception:
        ok = False
    if not bool(mx.array_equal(mx.random.state[0], saved_rng).item()):
        try:
            mx.random.state[0][:] = saved_rng
            mx.eval(mx.random.state[0])
        except Exception:
            pass
    try:
        sampler._is_greedy = ok
    except Exception:
        pass
    return ok


def extract_sampling_spec(sampler) -> tuple[float, list[Callable]] | None:
    """Extract deterministic `(temp, filter_methods)` from `make_sampler(...)` for exact rejection spec.

    Returns `None` for greedy samplers, stochastic XTC samplers, or unrecognized custom samplers.
    Leaves greedy `temp == 0` bit-exact paths 100% untouched.
    """
    if os.environ.get("LINGSANTE_SAMPLED_SPEC", "1") in ("0", ""):
        return None
    if sampler is None or is_greedy_sampler(sampler):
        return None
    cached = getattr(sampler, "_lingsante_sampling_spec", _UNSET_SPEC)
    if cached is not _UNSET_SPEC:
        return cached
    spec = None
    code = getattr(sampler, "__code__", None)
    closure = getattr(sampler, "__closure__", None)
    if code is not None and closure is not None and code.co_name == "sampler":
        env = dict(zip(code.co_freevars, (c.cell_contents for c in closure)))
        temp = env.get("temp")
        methods = env.get("sampling_methods")
        if isinstance(temp, (int, float)) and temp > 0 and isinstance(methods, list):
            ok = True
            for m in methods:
                mc = getattr(m, "__code__", None)
                if mc is None or "apply_xtc" in mc.co_names:
                    ok = False
                    break
            if ok:
                spec = (float(temp), list(methods))
    try:
        sampler._lingsante_sampling_spec = spec
    except Exception:
        pass
    return spec


_UNSET_SPEC = object()


def _logprobs_to_probs(logprobs: mx.array, spec: tuple[float, list[Callable]]) -> mx.array:
    """Apply `make_sampler`'s exact deterministic filter chain + temperature softmax in FP32."""
    temp, methods = spec
    lp = logprobs
    for m in methods:
        lp = m(lp)
    return mx.softmax(lp.astype(mx.float32) * (1.0 / temp), axis=-1)


def _sample_from_probs(probs: mx.array) -> mx.array:
    """Exact categorical draw from non-negative FP32 probability vector (zero mass where `probs <= 0`)."""
    log_p = mx.where(
        probs > 0,
        mx.log(mx.maximum(probs, 1e-38)),
        mx.array(-float("inf"), dtype=mx.float32),
    )
    return mx.random.categorical(log_p).astype(mx.uint32)


def _rejection_verify_stream(
    tgt_lp_rows: mx.array,
    draft_toks: list[int],
    draft_probs: list[mx.array],
    spec: tuple[float, list[Callable]],
) -> tuple[int, int]:
    """Exact distribution-preserving Leviathan/Chen rejection sampling for one `temp > 0` stream.

    Given normalized target logprobs `(K+1, V)`, drafted tokens `d_0..d_{K-1}` drawn from
    `draft_probs` `q_0..q_{K-1}`, returns `(n_accepted, next_token_id)` whose marginal distribution
    at every step is mathematically identical to sampling directly from `p = _logprobs_to_probs(lp, spec)`.
    """
    K = len(draft_toks)
    if K == 0:
        p0 = _logprobs_to_probs(tgt_lp_rows[0], spec)
        y = _sample_from_probs(p0)
        mx.eval(y)
        return 0, int(y.item())

    p_mat = _logprobs_to_probs(tgt_lp_rows[: K + 1], spec)  # (K+1, V) f32
    q_mat = mx.stack(draft_probs, axis=0)  # (K, V) f32
    rows = mx.arange(K, dtype=mx.int32)
    d_idx = mx.array(draft_toks, dtype=mx.int32)
    p_d = p_mat[rows, d_idx]
    q_d = q_mat[rows, d_idx]
    u = mx.random.uniform(shape=(K,), dtype=mx.float32)
    mx.eval(p_d, q_d, u)

    p_vals = p_d.tolist()
    q_vals = q_d.tolist()
    u_vals = u.tolist()
    a = 0
    for j in range(K):
        # Strictly reject whenever target prob is 0; otherwise accept with exact prob min(1, p / q)
        if p_vals[j] > 0.0 and u_vals[j] * q_vals[j] < p_vals[j]:
            a += 1
        else:
            break

    if a < K:
        res = mx.maximum(p_mat[a] - q_mat[a], 0.0)
        res_sum = mx.sum(res)
        mx.eval(res_sum)
        if float(res_sum.item()) > 1e-7:
            nxt_tok = _sample_from_probs(res)
        else:
            nxt_tok = _sample_from_probs(p_mat[a])
        mx.eval(nxt_tok)
        return a, int(nxt_tok.item())

    nxt_tok = _sample_from_probs(p_mat[K])
    mx.eval(nxt_tok)
    return K, int(nxt_tok.item())


def mtp_prefill_chunk(
    mtp: BailingMTPLayer,
    tokens_2d: mx.array,
    hidden_3d: mx.array,
    mtp_state: tuple[KVCache, mx.array | None] | None,
    *,
    step: int = 512,
) -> tuple[KVCache, mx.array]:
    """Feed target prefill chunk `(1, M)` tokens & `(1, M, H)` hidden into shifted `KVCache`."""
    mc, pending_h = mtp_state if mtp_state is not None else (KVCache(), None)
    M = int(tokens_2d.shape[1])
    if M <= 0:
        return mc, pending_h
    if pending_h is not None:
        m_toks = tokens_2d
        m_hs = pending_h if M == 1 else mx.concatenate([pending_h, hidden_3d[:, :-1, :]], axis=1)
    elif M > 1:
        m_toks = tokens_2d[:, 1:]
        m_hs = hidden_3d[:, :-1, :]
    else:
        m_toks = None
        m_hs = None

    if m_toks is not None:
        tot = int(m_toks.shape[1])
        pos = 0
        while pos < tot:
            m = min(step, tot - pos)
            mtp.kv_append(m_toks[:, pos : pos + m], m_hs[:, pos : pos + m, :], mc)
            mx.eval(mc.state)
            pos += m

    new_pending_h = hidden_3d[:, -1:, :]
    mx.eval(new_pending_h)
    return mc, new_pending_h


def mtp_vcache_from_standard(model, prompt_cache: list, C: int):
    """Convert standard `(C,)` batched `prompt_cache` into split verification cache."""
    per_stream = [
        [
            c.extract(i)
            if (
                C > 1
                or isinstance(c, BatchKVCache)
                or (isinstance(c, ArraysCache) and (c.lengths is not None or c.left_padding is not None))
            )
            else c
            for c in prompt_cache
        ]
        for i in range(C)
    ]
    return make_verify_cache(model, per_stream)


def mtp_extract_from_vcache(
    model,
    vcache: list,
    idx: int,
    C: int,
    *,
    mtp_state: tuple[KVCache, mx.array] | None = None,
    copy_kv: bool = True,
) -> list:
    """Extract a single-stream standard `prompt_cache` from `vcache`."""
    out = []
    for layer, c in zip(model.model.layers, vcache):
        if layer.is_linear:
            ac = ArraysCache(4)
            ac[0] = c[0][idx : idx + 1]
            ac[1] = c[1][idx : idx + 1]
            ac[2] = c[2][idx : idx + 1]
            ac[3] = c[3 + idx] if len(c.cache) > 4 else c[3][idx : idx + 1]
            out.append(ac)
        else:
            kv = c if C == 1 else c[idx]
            if copy_kv:
                kv_copy = KVCache()
                if kv.keys is not None and kv.offset > 0:
                    kv_copy.keys = mx.contiguous(kv.keys[..., : kv.offset, :])
                    kv_copy.values = mx.contiguous(kv.values[..., : kv.offset, :])
                    kv_copy.offset = kv.offset
                out.append(kv_copy)
            else:
                out.append(kv)
    if mtp_state is not None and out:
        mc, pending_h = mtp_state
        mc_copy = KVCache()
        if mc.keys is not None and mc.offset > 0:
            mc_copy.keys = mx.contiguous(mc.keys[..., : mc.offset, :])
            mc_copy.values = mx.contiguous(mc.values[..., : mc.offset, :])
            mc_copy.offset = mc.offset
        out[-1]._lingsante_mtp_state = (mc_copy, pending_h)
    return out


def _snapshot_stream_vcache(model, vcache: list, idx: int, C: int) -> list:
    """Zero-copy snapshot of one stream's KDA slices & MLA KVCache reference prior to contraction."""
    snap = []
    for layer, c in zip(model.model.layers, vcache):
        if layer.is_linear:
            ac = ArraysCache(4)
            ac[0] = c[0][idx : idx + 1]
            ac[1] = c[1][idx : idx + 1]
            ac[2] = c[2][idx : idx + 1]
            ac[3] = c[3 + idx] if len(c.cache) > 4 else c[3][idx : idx + 1]
            snap.append(ac)
        else:
            snap.append(c if C == 1 else c[idx])
    return snap


def _finalize_snap_cache(snap: list, mtp_state: tuple[KVCache, mx.array] | None) -> list:
    """Materialize contiguous MLA & MTP KV copies for a finished stream snapshot."""
    out = []
    for item in snap:
        if isinstance(item, ArraysCache):
            out.append(item)
        else:
            kv_copy = KVCache()
            if item.keys is not None and item.offset > 0:
                kv_copy.keys = mx.contiguous(item.keys[..., : item.offset, :])
                kv_copy.values = mx.contiguous(item.values[..., : item.offset, :])
                kv_copy.offset = item.offset
            out.append(kv_copy)
    if mtp_state is not None and out:
        mc, pending_h = mtp_state
        mc_copy = KVCache()
        if mc.keys is not None and mc.offset > 0:
            mc_copy.keys = mx.contiguous(mc.keys[..., : mc.offset, :])
            mc_copy.values = mx.contiguous(mc.values[..., : mc.offset, :])
            mc_copy.offset = mc.offset
        out[-1]._lingsante_mtp_state = (mc_copy, pending_h)
    return out


def mtp_vcache_to_standard(model, vcache: list, C: int) -> list:
    """Convert split `vcache` back to standard `[ArraysCache(4) | BatchKVCache]`."""
    if C <= 0:
        return []
    per = [mtp_extract_from_vcache(model, vcache, i, C, copy_kv=False) for i in range(C)]
    n_layers = len(per[0])
    return [type(per[0][li]).merge([per[i][li] for i in range(C)]) for li in range(n_layers)]


def mtp_extend_vcache(model, vc_a: list, C_a: int, vc_b: list, C_b: int) -> list:
    """Zero-copy merge of two verification caches `vc_a` (C_a) and `vc_b` (C_b)."""
    C_tot = C_a + C_b
    out = []
    for layer, ca, cb in zip(model.model.layers, vc_a, vc_b):
        if layer.is_linear:
            ac = ArraysCache(4 if C_tot == 1 else 3 + C_tot)
            for j in range(3):
                ac[j] = mx.concatenate([ca[j], cb[j]], axis=0)
            st_a = [ca[3]] if len(ca.cache) == 4 else list(ca.cache[3 : 3 + C_a])
            st_b = [cb[3]] if len(cb.cache) == 4 else list(cb.cache[3 : 3 + C_b])
            if C_tot == 1:
                ac[3] = (st_a + st_b)[0]
            else:
                for ni, s in enumerate(st_a + st_b):
                    ac[3 + ni] = s
            out.append(ac)
        else:
            la = [ca] if C_a == 1 else list(ca)
            lb = [cb] if C_b == 1 else list(cb)
            la.extend(lb)
            out.append(la[0] if C_tot == 1 else la)
    return out


def mtp_vcache_nbytes(vcache: list) -> int:
    total = 0
    for c in vcache:
        if isinstance(c, list):
            total += sum(ci.nbytes for ci in c)
        else:
            total += c.nbytes
    return total


def mtp_cancel_inflight(inflight: dict, mtp_caches: list[KVCache]) -> None:
    """Roll back an unconsumed speculative verify cycle (uniform or varlen) with zero KDA mutation."""
    rb = inflight["rb"]
    Ls = rb.get("Ls")
    if Ls is not None:
        for mc in rb["mla"]:
            if isinstance(mc, list):
                for ci, lc in zip(mc, Ls):
                    ci.trim(lc)
            else:
                mc.trim(Ls[0])
    else:
        L = rb["L"]
        for mc in rb["mla"]:
            if isinstance(mc, list):
                for ci in mc:
                    ci.trim(L)
            else:
                mc.trim(L)
    for mc, base_off in zip(mtp_caches, inflight["mtp_base_offs"]):
        tr = mc.offset - base_off
        if tr > 0:
            mc.trim(tr)


def _commit_verify_any(vcache: list, rb: dict, ncs: int | Sequence[int]) -> None:
    """Dispatch commit to varlen commit helper if `rb` is varlen, else standard `commit_verify_cache`."""
    if rb.get("Ls") is not None and hasattr(_vd, "commit_verify_cache_varlen"):
        _vd.commit_verify_cache_varlen(vcache, rb, ncs)
    else:
        commit_verify_cache(vcache, rb, ncs)


def _verify_eval_targets(
    preds: mx.array, hidden: mx.array, draft_mat: mx.array | None, lp: mx.array, rb: dict
) -> list:
    arrs = [preds, hidden, lp]
    if draft_mat is not None:
        arrs.append(draft_mat)
    cap = rb.get("captured")
    if cap is not None:
        arrs.append(cap)
    return arrs


def _apply_feat_fuse_consumed(
    mtp: BailingMTPLayer,
    hidden: mx.array,
    captured: mx.array | None,
    ncs: list[int],
    *,
    varlen_starts: list[int] | None = None,
) -> list[mx.array]:
    """Apply `FeatFuse` ONLY to consumed positions `[:nc_s]` per stream (zero compute on rejected rows)."""
    C = len(ncs)
    has_ff = getattr(mtp, "_feat_fuse", None) is not None and captured is not None
    if varlen_starts is None:
        if C == 1:
            nc0 = ncs[0]
            h0 = hidden[0, :nc0, :]
            if not has_ff:
                return [h0]
            return [mtp.apply_feat_fuse(h0, captured[0, :nc0, :])]
        h_slices = [hidden[s, : ncs[s], :] for s in range(C)]
        if not has_ff:
            return h_slices
        c_slices = [captured[s, : ncs[s], :] for s in range(C)]
        fused = mtp.apply_feat_fuse(
            mx.concatenate(h_slices, axis=0),
            mx.concatenate(c_slices, axis=0),
        )
    else:
        h_slices = [hidden[st : st + nc, :] for st, nc in zip(varlen_starts, ncs)]
        if not has_ff:
            return h_slices
        c_slices = [captured[st : st + nc, :] for st, nc in zip(varlen_starts, ncs)]
        fused = mtp.apply_feat_fuse(
            mx.concatenate(h_slices, axis=0),
            mx.concatenate(c_slices, axis=0),
        )
    out = []
    pos = 0
    for nc in ncs:
        out.append(fused[pos : pos + nc, :])
        pos += nc
    return out


def _refresh_mtp_accepted_list(
    mtp: BailingMTPLayer,
    mtp_caches: list[KVCache],
    after0_offs: list[int],
    draft_lists: list[list[int]],
    h_eff_list: list[mx.array],
    ncs: list[int],
) -> None:
    """Trim MTP caches to `after0_offs` and append only committed draft tokens `[0..nc-2]`."""
    for mc, aoff in zip(mtp_caches, after0_offs):
        tr = mc.offset - aoff
        if tr > 0:
            mc.trim(tr)
    need = [(ci, nc - 1) for ci, nc in enumerate(ncs) if nc > 1]
    if not need:
        return
    if len(need) == 1:
        ci, Lc = need[0]
        toks = mx.array(draft_lists[ci][:Lc], dtype=mx.uint32)
        mtp.kv_append(toks, h_eff_list[ci][:Lc, :], mtp_caches[ci])
        return
    toks_flat = mx.array(
        [tok for ci, Lc in need for tok in draft_lists[ci][:Lc]],
        dtype=mx.uint32,
    )
    hs_flat = mx.concatenate([h_eff_list[ci][:Lc, :] for ci, Lc in need], axis=0)
    mtp.kv_append_varlen(
        toks_flat,
        hs_flat,
        [mtp_caches[ci] for ci, _ in need],
        [Lc for _, Lc in need],
    )


def _refresh_mtp_accepted(
    mtp: BailingMTPLayer,
    mtp_caches: list[KVCache],
    after0_offs: list[int],
    draft_lists: list[list[int]],
    hidden: mx.array,
    ncs: list[int],
    *,
    use_q4_head: bool = True,
) -> None:
    """Backward-compatible wrapper around `_refresh_mtp_accepted_list`."""
    del use_q4_head
    h_list = [hidden[ci, : ncs[ci], :] for ci in range(len(ncs))]
    _refresh_mtp_accepted_list(mtp, mtp_caches, after0_offs, draft_lists, h_list, ncs)


# ---------------------------------------------------------------------------
# A2: Calibrated renewal-reward + MoE expert-budget stopping & admission helpers
# ---------------------------------------------------------------------------


def _dt_non_moe_ms(B_next: int) -> float:
    """Non-MoE marginal verify cost (ms) moving from B_next-1 -> B_next (OPT_LOG.md:207 empirical)."""
    if B_next <= 3:
        return 0.76
    if B_next <= 5:
        return 0.55  # use_dedup=True at B>=5 offsets tm=4->2 switch
    if B_next <= 8:
        return 1.80
    if B_next == 9:
        return 16.0  # FP8 bsg=2 cliff (if unpatched)
    return 2.20


def _est_delta_uniq_exp(
    B_next: int,
    du72: int | None,
    is_repeat_tok: bool,
) -> float:
    """Estimate marginal 68-layer unique expert count for adding token at batch size B_next."""
    if du72 is not None:
        # Calibrated affine map from Layer-72 router new-expert count (0..8) to 68-layer mean Delta_U
        base = 1.0 + 0.55 * float(du72)
        if is_repeat_tok:
            base *= 0.72
        return base
    if is_repeat_tok:
        return 1.4
    # Empirical U(1)=8.0, U(5)=23.0, U(8)=29.3 -> diminishing marginal Delta_U(B)
    return max(1.8, 4.1 - 0.35 * max(0, B_next - 2))


def _update_router_mask(mtp: BailingMTPLayer, row_idx: int, seen_mask: int) -> tuple[int | None, int]:
    """Read Layer-72 router indices exposed on `mtp._last_router_inds` (0 extra FLOPs/DRAM)."""
    r_inds = getattr(mtp, "_last_router_inds", None)
    if r_inds is None:
        return None, seen_mask
    row = r_inds[row_idx].tolist()
    bits = 0
    for e in row:
        bits |= 1 << int(e)
    du72 = (bits & ~seen_mask).bit_count()
    return du72, (seen_mask | bits)


def _step_top1_and_prob(dl: mx.array | tuple[mx.array, mx.array]) -> tuple[mx.array, mx.array]:
    """Compute `(top1_id (C,) u32, top1_prob (C,) f32)` from draft logits `(C, V)` or pre-fused tuple."""
    if isinstance(dl, tuple):
        return dl[0], dl[1]
    top_id = mx.argmax(dl, axis=-1).astype(mx.uint32)
    dl_f = dl.astype(mx.float32)
    probs = mx.exp(mx.max(dl_f, axis=-1) - mx.logsumexp(dl_f, axis=-1))
    return top_id, probs


def _has_d1_mtp(mtp: BailingMTPLayer) -> bool:
    """True when `mtp` includes D1 fused `refresh_and_step0` + `return_argmax_prob` + vocab window."""
    return hasattr(mtp, "refresh_and_step0") and hasattr(mtp, "reset_vocab_window")


def _mtp_step_top1(
    mtp: BailingMTPLayer,
    toks: mx.array,
    hs: mx.array,
    cache_arg,
    *,
    C: int,
    use_q4_head: bool,
) -> tuple[mx.array, mx.array, mx.array]:
    """Run one MTP draft step and return `(h_out, top1_id (C,) u32, top1_prob (C,) f32)`."""
    if _has_d1_mtp(mtp):
        return mtp.step_rows(
            toks, hs, cache_arg, C=C, L=1, use_q4_head=use_q4_head, return_argmax_prob=True
        )
    h_out, dl = mtp.step_rows(toks, hs, cache_arg, C=C, L=1, use_q4_head=use_q4_head)
    top_id, probs = _step_top1_and_prob(dl)
    return h_out, top_id, probs


def _a2_params(dyn_th: float, p1_min: float) -> tuple[float, float, float, float, float, bool]:
    """Return `(r_star, t_draft, e_p_next, p0_drop, skip_sync_th, use_expert_rr)`."""
    use_rr = os.environ.get("LINGSANTE_DYN_RR", "1") not in ("0", "")
    r_star = float(os.environ.get("LINGSANTE_DYN_RSTAR", "0.053"))
    t_draft = float(os.environ.get("LINGSANTE_DYN_TDRAFT", "1.40"))
    e_p_next = float(os.environ.get("LINGSANTE_DYN_EPNEXT", "0.52"))
    default_p0_drop = p1_min if p1_min > 0.0 else 0.12
    p0_drop = float(os.environ.get("LINGSANTE_P0_DROP", str(default_p0_drop)))
    skip_sync_th = float(os.environ.get("LINGSANTE_SKIP_SYNC_TH", "2.0"))
    return r_star, t_draft, e_p_next, p0_drop, skip_sync_th, use_rr


def _thresholds_for_step(
    B_next: int,
    du72: int | None,
    is_rep: bool,
    *,
    dyn_th: float,
    r_star: float,
    t_draft: float,
    e_p_next: float,
    use_rr: bool,
) -> tuple[float, float]:
    """Return Bellman renewal-reward `(tau_verify, tau_draft)` at verify batch size `B_next`.

    - Post-draft admission (t_draft sunk; accounts for Q4-head flattening & prefix correlation):
      pop trailing d_k iff cum_p < tau_verify (~0.06).
    - Continue-drafting: anchored at calibrated `dyn_th` (0.55) when router indices are absent;
      when Layer-72 router `du72` is exposed (opt/draft D1), modulates around `dyn_th` by
      `+0.025 * (du72 - 5.0)` (clamped to `[0.42, 0.65]`) so expert-reusing drafts go deeper
      and expert-heavy drafts stop earlier.
    """
    pop_env = os.environ.get("LINGSANTE_POP_TH")
    tau_ver = float(pop_env) if pop_env is not None else 0.06
    if not use_rr or "LINGSANTE_DYN_TH" in os.environ:
        return tau_ver, dyn_th
    if du72 is not None:
        tau_dr = max(0.42, min(0.65, dyn_th + 0.025 * (float(du72) - 5.0)))
        if is_rep:
            tau_dr = max(0.40, tau_dr - 0.05)
        return tau_ver, tau_dr
    if is_rep:
        return tau_ver, max(0.45, dyn_th - 0.06)
    return tau_ver, dyn_th


# ---------------------------------------------------------------------------
# Single-stream `generate_step`
# ---------------------------------------------------------------------------


def mtp_generate_step(
    prompt: mx.array,
    model,
    mtp: BailingMTPLayer,
    stream: Any,
    *,
    max_tokens: int = 256,
    prompt_cache: list | None = None,
    prefill_step_size: int = 2048,
    prompt_progress_callback: Callable[[int, int], None] | None = None,
    sampler: Any = None,
):
    """Bit-exact greedy & distribution-preserving `temp > 0` `generate_step` via LingSante MTP / EAGLE-3."""
    if len(prompt) == 0:
        raise ValueError("Prompt must be non-empty.")
    if prompt_cache is None:
        prompt_cache = make_prompt_cache(model)
    prompt_progress_callback = prompt_progress_callback or (lambda *_: None)
    if max_tokens == 0:
        return
    sampling_spec = extract_sampling_spec(sampler) if sampler is not None else None

    with mx.stream(stream):
        prompt_ids_list = prompt.tolist() if (_use_mtp_suffix() and sampling_spec is None) else None
        total_prompt_tokens = len(prompt)
        prompt_processed_tokens = 0
        prompt_progress_callback(0, total_prompt_tokens)

        mtp_state = getattr(prompt_cache[-1], "_lingsante_mtp_state", None) if prompt_cache else None
        if prompt_cache and hasattr(prompt_cache[-1], "_lingsante_mtp_state"):
            delattr(prompt_cache[-1], "_lingsante_mtp_state")

        lids = getattr(mtp, "_feat_layers", None)
        while total_prompt_tokens - prompt_processed_tokens > 1:
            remaining = (total_prompt_tokens - prompt_processed_tokens) - 1
            n_to_process = min(prefill_step_size, remaining)
            processed = prompt[:n_to_process]
            h = mtp_target_prefill_hidden(model, mtp, processed[None], prompt_cache)
            mx.eval(h, [c.state for c in prompt_cache])
            mtp_state = mtp_prefill_chunk(mtp, processed[None], h, mtp_state)
            prompt_processed_tokens += n_to_process
            prompt_progress_callback(prompt_processed_tokens, total_prompt_tokens)
            prompt = prompt[n_to_process:]
            if n_to_process >= 512:
                mx.clear_cache()

        last_tok = prompt
        if can_fast_verify(model, last_tok[None], prompt_cache):
            logits0, hidden0, rb0 = fast_verify_forward(
                model,
                last_tok[None],
                prompt_cache,
                return_hidden=True,
                return_rollback=True,
                capture_layers=lids,
            )
            commit_verify_cache(prompt_cache, rb0, 1)
            hidden0 = mtp.apply_feat_fuse(hidden0, rb0.get("captured"))
        else:
            h_raw, hidden0 = mtp_target_prefill_hidden(
                model, mtp, last_tok[None], prompt_cache, return_raw=True
            )
            logits0 = model.lm_head(h_raw)

        y0, lp0_2d = _logits_to_preds(logits0[:, -1, :])
        lp0 = lp0_2d.squeeze(0)
        if sampling_spec is not None:
            y0 = sampler(lp0_2d).astype(mx.uint32)

        mc, pending_h = mtp_state if mtp_state is not None else (KVCache(), None)
        eval_init = [y0, hidden0, lp0]
        if pending_h is not None:
            mtp.kv_append(last_tok, pending_h[:, 0, :], mc)
            eval_init.extend(mc.state)
        mx.eval(*eval_init)

        prompt_progress_callback(total_prompt_tokens, total_prompt_tokens)
        y0_int = int(y0.item())
        if _has_d1_mtp(mtp):
            tail_ids = (prompt_ids_list[-64:] if prompt_ids_list is not None else []) + [y0_int]
            mtp.reset_vocab_window(tail_ids)
        suffix_matcher = (
            SuffixMatcher(prompt_ids_list + [y0_int], min_gram=_suffix_min_gram(), max_gram=8)
            if prompt_ids_list is not None
            else None
        )

        n = 0
        yield y0_int, lp0
        n += 1
        if n == max_tokens:
            if prompt_cache:
                prompt_cache[-1]._lingsante_mtp_state = (mc, hidden0[:, -1:, :])
            return

        cur_y = y0
        cur_h = hidden0[:, -1, :]
        K_lim = None if max_tokens < 0 else max(0, max_tokens - n - 1)
        draft_q: list[mx.array] | None = None
        if sampling_spec is not None:
            draft_mat, mtp_exact_offs, q_per_stream = _chain_mtp_drafts_mixed(
                mtp,
                cur_y,
                cur_h,
                [mc],
                sampling_specs=[sampling_spec],
                C=1,
                K_limit=K_lim,
                use_q4_head=True,
            )
            draft_q = q_per_stream[0]
        else:
            draft_mat, mtp_exact_offs = _chain_mtp_drafts(
                mtp,
                cur_y,
                cur_h,
                [mc],
                C=1,
                K_limit=K_lim,
                use_q4_head=True,
                suffix_matcher=suffix_matcher,
                anchor_tok_int=y0_int,
            )
        K = int(draft_mat.shape[1])
        if K > 0:
            mx.async_eval(draft_mat)
            inp = mx.concatenate([cur_y[:, None], draft_mat], axis=1)
        else:
            inp = cur_y[:, None]
        logits, hidden, rb = fast_verify_forward(
            model, inp, prompt_cache, return_hidden=True, return_rollback=True, capture_layers=lids
        )
        preds, lp = _logits_to_preds(logits)
        ev_t = _verify_eval_targets(preds, hidden, draft_mat if K > 0 else None, lp, rb)
        if draft_q:
            ev_t.extend(draft_q)
        mx.async_eval(*ev_t)

        inflight = {"rb": rb, "mtp_base_offs": [mtp_exact_offs[0] - 1]}
        cur_burst_rb = None
        cur_burst_nc_committed = 0
        cur_burst_emitted = 0
        cur_burst_after0 = mtp_exact_offs[0]
        cur_burst_drafts = None
        cur_burst_h_eff = None

        try:
            while True:
                ev_w = _verify_eval_targets(preds, hidden, draft_mat if K > 0 else None, lp, rb)
                if draft_q:
                    ev_w.extend(draft_q)
                mx.eval(*ev_w)
                inflight = None
                draft_row = draft_mat[0].tolist() if K > 0 else []

                if sampling_spec is not None:
                    a, next_tok_val = _rejection_verify_stream(
                        lp[0], draft_row, draft_q or [], sampling_spec
                    )
                    pred_row = draft_row[:a] + [next_tok_val]
                else:
                    pred_row = preds[0].tolist()
                    a = 0
                    for j in range(K):
                        if pred_row[j] == draft_row[j]:
                            a += 1
                        else:
                            break

                n_take = a + 1
                if max_tokens > 0:
                    n_take = min(n_take, max_tokens - n)

                # Slice FeatFuse strictly to consumed positions [:nc]
                nc_need = n_take if n_take <= a else (a + 1)
                h_eff_0 = _apply_feat_fuse_consumed(mtp, hidden, rb.get("captured"), [nc_need])[0]

                more_needed = max_tokens < 0 or (n + n_take) < max_tokens
                if not more_needed:
                    # Final burst: yield [0..n_take-2], commit once, yield last token
                    cur_burst_rb = rb
                    cur_burst_nc_committed = 0
                    cur_burst_emitted = 0
                    cur_burst_after0 = mtp_exact_offs[0]
                    cur_burst_drafts = draft_row
                    cur_burst_h_eff = h_eff_0
                    for j in range(n_take - 1):
                        cur_burst_emitted = j + 1
                        n += 1
                        yield pred_row[j], lp[0, j]
                        if n % 256 == 0:
                            mx.clear_cache()
                    last_j = n_take - 1
                    commit_verify_cache(prompt_cache, rb, nc_need)
                    cur_burst_rb = None
                    _refresh_mtp_accepted_list(
                        mtp, [mc], [mtp_exact_offs[0]], [draft_row], [h_eff_0], [nc_need]
                    )
                    last_h = h_eff_0[last_j : last_j + 1, :][None]
                    if prompt_cache:
                        prompt_cache[-1]._lingsante_mtp_state = (mc, last_h)
                    mx.eval(last_h, mc.state)
                    n += 1
                    yield pred_row[last_j], lp[0, last_j]
                    return

                # Overlap Cycle t+1 BEFORE yielding burst tokens [0..n_take-1]!
                nc = a + 1
                next_cur = pred_row[a]
                if _has_d1_mtp(mtp):
                    mtp.update_vocab_window(pred_row[:nc])
                if suffix_matcher is not None:
                    tl = getattr(suffix_matcher, "_last_tail_len", 0)
                    if tl > 0:
                        suffix_matcher.observe_outcome(max(0, a - 1), tl)
                        suffix_matcher._last_tail_len = 0
                    suffix_matcher.extend(pred_row[:nc])

                commit_verify_cache(prompt_cache, rb, nc)
                K_lim_next = None if max_tokens < 0 else max(0, max_tokens - (n + n_take) - 1)
                draft_q_next: list[mx.array] | None = None
                if sampling_spec is not None:
                    draft_mat_next, mtp_exact_offs_next, q_per_stream_next = (
                        _sync_and_chain_mtp_mixed(
                            mtp,
                            [h_eff_0],
                            [draft_row],
                            [next_cur],
                            [nc],
                            [mc],
                            [mtp_exact_offs[0]],
                            sampling_specs=[sampling_spec],
                            C=1,
                            K_limit=K_lim_next,
                            use_q4_head=True,
                        )
                    )
                    draft_q_next = q_per_stream_next[0]
                else:
                    draft_mat_next, mtp_exact_offs_next = _sync_and_chain_mtp_list(
                        mtp,
                        [h_eff_0],
                        [draft_row],
                        [next_cur],
                        [nc],
                        [mc],
                        [mtp_exact_offs[0]],
                        C=1,
                        K_limit=K_lim_next,
                        use_q4_head=True,
                        suffix_matcher=suffix_matcher,
                    )
                K_next = int(draft_mat_next.shape[1])
                if K_next > 0:
                    mx.async_eval(draft_mat_next)
                cur_arr = mx.array([next_cur], dtype=mx.uint32)
                inp_next = (
                    mx.concatenate([cur_arr[:, None], draft_mat_next], axis=1)
                    if K_next > 0
                    else cur_arr[:, None]
                )
                logits_next, hidden_next, rb_next = fast_verify_forward(
                    model, inp_next, prompt_cache, return_hidden=True, return_rollback=True, capture_layers=lids
                )
                preds_next, lp_next = _logits_to_preds(logits_next)
                ev_n = _verify_eval_targets(
                    preds_next, hidden_next, draft_mat_next if K_next > 0 else None, lp_next, rb_next
                )
                if draft_q_next:
                    ev_n.extend(draft_q_next)
                mx.async_eval(*ev_n)

                inflight = {
                    "rb": rb_next,
                    "mtp_base_offs": [mtp_exact_offs_next[0] - 1],
                    "mtp_after0_offs": list(mtp_exact_offs_next),
                    "hidden_next": hidden_next,
                }
                cur_burst_rb = rb
                cur_burst_nc_committed = nc
                cur_burst_emitted = 0
                cur_burst_after0 = mtp_exact_offs[0]
                cur_burst_drafts = draft_row
                cur_burst_h_eff = h_eff_0

                # Yield all tokens of Cycle t while GPU executes Cycle t+1 asynchronously
                for j in range(n_take - 1):
                    cur_burst_emitted = j + 1
                    n += 1
                    yield pred_row[j], lp[0, j]
                    if n % 256 == 0:
                        mx.clear_cache()

                cur_burst_rb = None
                n += 1
                yield next_cur, lp[0, a]
                if n % 256 == 0:
                    mx.clear_cache()

                preds, hidden, draft_mat, lp, rb, K, mtp_exact_offs, draft_q = (
                    preds_next,
                    hidden_next,
                    draft_mat_next,
                    lp_next,
                    rb_next,
                    K_next,
                    mtp_exact_offs_next,
                    draft_q_next,
                )
        finally:
            if cur_burst_rb is not None:
                # Caller broke early inside a burst after cur_burst_emitted tokens.
                nc_fin = cur_burst_emitted + 1
                if inflight is not None:
                    # Cancel overlapped Cycle t+1 MLA updates & MTP cache first.
                    mtp_cancel_inflight(inflight, [mc])
                    inflight = None
                    if cur_burst_nc_committed > nc_fin:
                        delta_trim = cur_burst_nc_committed - nc_fin
                        for mla_c in cur_burst_rb["mla"]:
                            mla_c.trim(delta_trim)
                    # Re-commit KDA only to exact nc_fin from untouched s0/h0..2 in cur_burst_rb.
                    kda_only_rb = {
                        "C": cur_burst_rb["C"],
                        "L": cur_burst_rb["L"],
                        "kda": cur_burst_rb["kda"],
                        "mla": [],
                    }
                    commit_verify_cache(prompt_cache, kda_only_rb, nc_fin)
                else:
                    commit_verify_cache(prompt_cache, cur_burst_rb, nc_fin)
                _refresh_mtp_accepted_list(
                    mtp,
                    [mc],
                    [cur_burst_after0],
                    [cur_burst_drafts],
                    [cur_burst_h_eff],
                    [nc_fin],
                )
                if prompt_cache:
                    prompt_cache[-1]._lingsante_mtp_state = (
                        mc,
                        cur_burst_h_eff[nc_fin - 1 : nc_fin, :][None],
                    )
                mx.synchronize(stream)
            elif inflight is not None:
                if "hidden_next" in inflight:
                    # Caller broke on last token of burst (next_cur, which was inp_next[0]): commit nc=1.
                    rb_i = inflight["rb"]
                    cap_i = rb_i.get("captured")
                    mx.eval(inflight["hidden_next"], cap_i) if cap_i is not None else mx.eval(inflight["hidden_next"])
                    h_next_eff = _apply_feat_fuse_consumed(mtp, inflight["hidden_next"], cap_i, [1])[0]
                    commit_verify_cache(prompt_cache, rb_i, 1)
                    a0 = inflight["mtp_after0_offs"][0]
                    if mc.offset > a0:
                        mc.trim(mc.offset - a0)
                    if prompt_cache:
                        prompt_cache[-1]._lingsante_mtp_state = (mc, h_next_eff[0:1, :][None])
                else:
                    mtp_cancel_inflight(inflight, [mc])
                mx.synchronize(stream)


# ---------------------------------------------------------------------------
# Continuous-batching `LingSanteSpecBatch` (C=1..4, uniform & varlen)
# ---------------------------------------------------------------------------


class LingSanteSpecBatch:
    """Native MTP / EAGLE-3 speculative decoding backend for `BatchGenerator`/`GenerationBatch` (`1 <= C <= 4`)."""

    def __init__(
        self,
        model,
        mtp: BailingMTPLayer,
        vcache: list,
        mtp_caches: list[KVCache],
        mtp_exact_offs: list[int],
        cur_toks: list[int],
        cur_lps: list[mx.array],
        cur_hidden: mx.array,
        inflight: dict | None = None,
        *,
        spec_eligible: list[bool] | None = None,
        sampling_specs: list[tuple[float, list[Callable]] | None] | None = None,
        suffix_matchers: list[SuffixMatcher | None] | None = None,
        pending_joins: list[dict] | None = None,
    ):
        self.model = model
        self.mtp = mtp
        self.vcache = vcache
        self.mtp_caches = mtp_caches
        self.mtp_exact_offs = mtp_exact_offs
        self.cur_toks = cur_toks
        self.cur_lps = cur_lps
        self.cur_hidden = cur_hidden  # (C, 4096)
        self.inflight = inflight
        C = len(cur_toks)
        self.spec_eligible = list(spec_eligible) if spec_eligible is not None else [True] * C
        self.sampling_specs = (
            list(sampling_specs) if sampling_specs is not None else [None] * C
        )
        self.suffix_matchers = (
            list(suffix_matchers) if suffix_matchers is not None else [None] * C
        )
        self.pending_joins = list(pending_joins) if pending_joins is not None else []

    @staticmethod
    def _stream_modes(
        uids: Sequence[int],
        samplers: Sequence[Any] | None,
        fallback_sampler: Any,
        logits_processors: Sequence[Any] | None,
    ) -> tuple[list[bool], list[tuple[float, list[Callable]] | None]]:
        """Return `(spec_eligible, sampling_specs)` per stream.

        - Greedy (no logits_processors): `(True, None)` -> bit-exact argmax spec.
        - `temp > 0` `make_sampler` (no XTC, no logits_processors): `(True, (temp, methods))`
          -> exact distribution-preserving rejection-sampled spec.
        - Logits processors / XTC / custom sampler: `(False, None)` -> exact `K_i=0` fallback.
        """
        n = len(uids)
        eff_s = [fallback_sampler if s is None else s for s in (samplers or [None] * n)]
        eff_lp = list(logits_processors) if logits_processors else [None] * n
        eligs: list[bool] = []
        specs: list[tuple[float, list[Callable]] | None] = []
        for s, lp in zip(eff_s, eff_lp):
            if bool(lp):
                eligs.append(False)
                specs.append(None)
            elif is_greedy_sampler(s):
                eligs.append(True)
                specs.append(None)
            else:
                sp = extract_sampling_spec(s)
                if sp is not None:
                    eligs.append(True)
                    specs.append(sp)
                else:
                    eligs.append(False)
                    specs.append(None)
        return eligs, specs

    @staticmethod
    def _stream_eligibility(
        uids: Sequence[int],
        samplers: Sequence[Any] | None,
        fallback_sampler: Any,
        logits_processors: Sequence[Any] | None,
    ) -> list[bool]:
        elig, _ = LingSanteSpecBatch._stream_modes(
            uids, samplers, fallback_sampler, logits_processors
        )
        return elig

    @staticmethod
    def can_use(
        model,
        uids: Sequence[int],
        samplers: Sequence[Any] | None,
        fallback_sampler: Any,
        logits_processors: Sequence[Any] | None,
    ) -> bool:
        if not (1 <= len(uids) <= 4):
            return False
        if get_lingsante_mtp(model) is None:
            return False
        elig = LingSanteSpecBatch._stream_eligibility(
            uids, samplers, fallback_sampler, logits_processors
        )
        if _has_varlen_verify():
            # Mixed greedy + non-greedy coexistence: keep spec alive whenever >= 1 stream is greedy!
            return any(elig)
        return all(elig)

    def can_absorb_nongreedy(self, other_gen_batch) -> bool:
        """True when varlen verify is available and we can run joining non-greedy streams at K_i=0."""
        if not _has_varlen_verify():
            return False
        if not any(self.spec_eligible):
            return False
        if len(self.cur_toks) + len(self.pending_joins) + len(other_gen_batch.uids) > 4:
            return False
        if not other_gen_batch.prompt_cache:
            return False
        return True

    def absorb_nongreedy(self, other_gen_batch) -> None:
        """Absorb a standard non-greedy `GenerationBatch` into varlen spec mode with `K_i = 0 (L_i = 1)`."""
        self.cancel_inflight()
        if self.pending_joins:
            self._flush_pending_joins_standalone()
        C_b = len(other_gen_batch.uids)
        # Ensure other_gen_batch's initial _step() outputs are evaluated
        mx.eval(other_gen_batch._next_tokens, other_gen_batch._next_logprobs)
        vcache_b = mtp_vcache_from_standard(self.model, other_gen_batch.prompt_cache, C_b)
        C_a = len(self.cur_toks)
        self.vcache = mtp_extend_vcache(self.model, self.vcache, C_a, vcache_b, C_b)
        self.mtp_caches.extend(KVCache() for _ in range(C_b))
        self.mtp_exact_offs.extend(0 for _ in range(C_b))
        self.cur_toks.extend(other_gen_batch._next_tokens.tolist())
        self.cur_lps.extend(list(other_gen_batch._next_logprobs))
        dummy_h = mx.zeros((C_b, self.cur_hidden.shape[-1]), dtype=self.cur_hidden.dtype)
        self.cur_hidden = mx.concatenate([self.cur_hidden, dummy_h], axis=0)
        self.spec_eligible.extend(False for _ in range(C_b))
        self.sampling_specs.extend(None for _ in range(C_b))
        self.suffix_matchers.extend(None for _ in range(C_b))

    def can_piggyback_join(self, split_pb, last_inputs: list[list[int]]) -> bool:
        """True when joining stream(s) with 1 final prompt token can fuse into next varlen verify pass."""
        if not _has_varlen_verify():
            return False
        if self.inflight is not None:
            return False
        if not split_pb.uids or any(len(t) != 1 for t in last_inputs):
            return False
        if len(self.cur_toks) + len(self.pending_joins) + len(split_pb.uids) > 4:
            return False
        if not split_pb.prompt_cache or split_pb.prompt_cache[0].empty():
            return False
        C_j = len(split_pb.uids)
        probe_inp = mx.array([[t[0]] for t in last_inputs], dtype=mx.uint32)
        v_j = mtp_vcache_from_standard(self.model, split_pb.prompt_cache, C_j)
        return can_fast_verify(self.model, probe_inp, v_j)

    def piggyback_join(self, gen_batch, split_pb, last_inputs: list[list[int]]) -> None:
        """Queue newly prefilled stream(s) to verify their `L_new = 1` final token in next varlen pass."""
        C_j = len(split_pb.uids)
        v_j = mtp_vcache_from_standard(self.model, split_pb.prompt_cache, C_j)
        elig_j, specs_j = self._stream_modes(
            split_pb.uids, split_pb.samplers, split_pb.fallback_sampler, split_pb.logits_processors
        )
        for i in range(C_j):
            v_i = (
                v_j
                if C_j == 1
                else mtp_extract_from_vcache(self.model, v_j, i, C_j, copy_kv=False)
            )
            st = split_pb._mtp_states[i] if i < len(split_pb._mtp_states) else None
            mc, pending_h = st if st is not None else (KVCache(), None)
            tok_i = int(last_inputs[i][0])
            self.pending_joins.append(
                {
                    "uid": split_pb.uids[i],
                    "last_tok": tok_i,
                    "vcache": v_i,
                    "mc": mc,
                    "pending_h": pending_h,
                    "tokens": list(split_pb.tokens[i]) + [tok_i],
                    "sampler": split_pb.samplers[i] if i < len(split_pb.samplers) else None,
                    "logits_processor": (
                        split_pb.logits_processors[i]
                        if i < len(split_pb.logits_processors)
                        else ()
                    ),
                    "max_tokens": split_pb.max_tokens[i],
                    "stop_sequence": split_pb.stop_sequences[i],
                    "spec_eligible": elig_j[i],
                    "sampling_spec": specs_j[i],
                }
            )
            # Register metadata on gen_batch so len(gen_batch), uids, extract_cache, remove see it!
            gen_batch.uids.append(split_pb.uids[i])
            gen_batch.tokens.append(self.pending_joins[-1]["tokens"])
            gen_batch.samplers.append(self.pending_joins[-1]["sampler"])
            gen_batch.logits_processors.append(self.pending_joins[-1]["logits_processor"])
            gen_batch.max_tokens.append(self.pending_joins[-1]["max_tokens"])
            gen_batch.stop_sequences.append(self.pending_joins[-1]["stop_sequence"])
            from mlx_lm.models.cache import TokenBuffer

            gen_batch._token_context.append(TokenBuffer(self.pending_joins[-1]["tokens"]))
            gen_batch._num_tokens.append(0)
            gen_batch._matchers.append(self.pending_joins[-1]["stop_sequence"].matcher())

    def _flush_pending_joins_standalone(self) -> None:
        """Fallback materialization of pending_joins if filter/extract/to_standard happens pre-verify."""
        if not self.pending_joins:
            return
        pjs = self.pending_joins
        self.pending_joins = []
        C_j = len(pjs)
        lids = getattr(self.mtp, "_feat_layers", None)
        vc_j = pjs[0]["vcache"]
        for i in range(1, C_j):
            vc_j = mtp_extend_vcache(self.model, vc_j, i, pjs[i]["vcache"], 1)
        inp_j = mx.array([[pj["last_tok"]] for pj in pjs], dtype=mx.uint32)
        logits0, hidden0, rb0 = fast_verify_forward(
            self.model, inp_j, vc_j, return_hidden=True, return_rollback=True, capture_layers=lids
        )
        commit_verify_cache(vc_j, rb0, 1)
        hidden0 = self.mtp.apply_feat_fuse(hidden0, rb0.get("captured"))
        nxt0, lp0 = _logits_to_preds(logits0[:, 0, :])
        nxt_rows = []
        for i, pj in enumerate(pjs):
            sp = pj.get("sampling_spec")
            if sp is not None:
                nxt_rows.append(_sample_from_probs(_logprobs_to_probs(lp0[i], sp)).reshape(1))
            elif pj["spec_eligible"]:
                nxt_rows.append(nxt0[i : i + 1])
            elif pj["sampler"] is not None:
                nxt_rows.append(pj["sampler"](lp0[i : i + 1]).reshape(1).astype(mx.uint32))
            else:
                nxt_rows.append(nxt0[i : i + 1])
        nxt0 = mx.concatenate(nxt_rows, axis=0)
        eval_arrs = [nxt0, hidden0, lp0]
        for i, pj in enumerate(pjs):
            if pj["pending_h"] is not None:
                self.mtp.kv_append(inp_j[i : i + 1, 0], pj["pending_h"][:, 0, :], pj["mc"])
                eval_arrs.extend(pj["mc"].state)
        mx.eval(*eval_arrs)
        C_a = len(self.cur_toks)
        self.vcache = mtp_extend_vcache(self.model, self.vcache, C_a, vc_j, C_j)
        self.mtp_caches.extend(pj["mc"] for pj in pjs)
        self.mtp_exact_offs.extend(pj["mc"].offset for pj in pjs)
        self.cur_toks.extend(nxt0.tolist())
        self.cur_lps.extend(list(lp0))
        self.cur_hidden = mx.concatenate([self.cur_hidden, hidden0[:, 0, :]], axis=0)
        self.spec_eligible.extend(pj["spec_eligible"] for pj in pjs)
        self.sampling_specs.extend(pj.get("sampling_spec") for pj in pjs)
        self.suffix_matchers.extend(
            SuffixMatcher(pj["tokens"] + [int(t)], min_gram=_suffix_min_gram(), max_gram=8)
            if (_use_mtp_suffix() and pj["spec_eligible"] and pj.get("sampling_spec") is None)
            else None
            for pj, t in zip(pjs, nxt0.tolist())
        )

    @classmethod
    def init_from_prompt(
        cls,
        model,
        mtp: BailingMTPLayer,
        inputs: mx.array,
        prompt_cache: list,
        mtp_states: list | None = None,
        *,
        samplers: Sequence[Any] | None = None,
        fallback_sampler: Any = None,
        logits_processors: Sequence[Any] | None = None,
        token_contexts: Sequence[Any] | None = None,
    ) -> "LingSanteSpecBatch":
        C = int(inputs.shape[0])
        lids = getattr(mtp, "_feat_layers", None)
        elig, specs = cls._stream_modes(
            range(C), samplers, fallback_sampler, logits_processors
        )
        if prompt_cache[0].empty():
            h_raw, hidden0 = mtp_target_prefill_hidden(
                model, mtp, inputs[:, None], prompt_cache, return_raw=True
            )
            logits0 = model.lm_head(h_raw)
            vcache = mtp_vcache_from_standard(model, prompt_cache, C)
        else:
            vcache = mtp_vcache_from_standard(model, prompt_cache, C)
            if can_fast_verify(model, inputs[:, None], vcache):
                logits0, hidden0, rb0 = fast_verify_forward(
                    model,
                    inputs[:, None],
                    vcache,
                    return_hidden=True,
                    return_rollback=True,
                    capture_layers=lids,
                )
                commit_verify_cache(vcache, rb0, 1)
                hidden0 = mtp.apply_feat_fuse(hidden0, rb0.get("captured"))
            else:
                std_c = mtp_vcache_to_standard(model, vcache, C)
                h_raw, hidden0 = mtp_target_prefill_hidden(
                    model, mtp, inputs[:, None], std_c, return_raw=True
                )
                logits0 = model.lm_head(h_raw)
                vcache = mtp_vcache_from_standard(model, std_c, C)

        # Sample initial tokens (greedy for greedy streams, exact sampler/lp for sampled/non-greedy)
        lg0 = logits0[:, 0, :]
        if all(elig) and all(sp is None for sp in specs):
            nxt0, lp0 = _logits_to_preds(lg0)
        else:
            nxt_list, lp_list = [], []
            eff_s = [fallback_sampler if s is None else s for s in (samplers or [None] * C)]
            eff_lp = list(logits_processors) if logits_processors else [None] * C
            for i in range(C):
                row_lg = lg0[i : i + 1]
                procs = eff_lp[i] or ()
                if procs and token_contexts is not None:
                    ctx_toks = token_contexts[i].update_and_fetch(inputs[i : i + 1])
                    for p in procs:
                        row_lg = p(ctx_toks, row_lg)
                row_lp = row_lg - mx.logsumexp(row_lg, axis=-1, keepdims=True)
                tok_i = (
                    mx.argmax(row_lp, axis=-1)
                    if (elig[i] and specs[i] is None)
                    else eff_s[i](row_lp)
                )
                nxt_list.append(tok_i.reshape(1))
                lp_list.append(row_lp.squeeze(0))
            nxt0 = mx.concatenate(nxt_list, axis=0)
            lp0 = mx.stack(lp_list, axis=0)

        # S1: Overlap mtp.kv_append(last_tok, pending_h) with the single mx.eval(nxt0, hidden0, lp0)!
        mtp_caches = []
        eval_arrs = [nxt0, hidden0, lp0]
        for i in range(C):
            st = mtp_states[i] if (mtp_states is not None and i < len(mtp_states)) else None
            mc, pending_h = st if st is not None else (KVCache(), None)
            if pending_h is not None and elig[i]:
                mtp.kv_append(inputs[i : i + 1], pending_h[:, 0, :], mc)
                eval_arrs.extend(mc.state)
            mtp_caches.append(mc)
        mx.eval(*eval_arrs)
        mtp_exact_offs = [mc.offset for mc in mtp_caches]

        cur_toks = nxt0.tolist()
        if _has_d1_mtp(mtp) and token_contexts is not None:
            tail = []
            for i in range(C):
                if hasattr(token_contexts[i], "tokens"):
                    tail.extend(token_contexts[i].tokens.tolist()[-32:])
                tail.append(int(cur_toks[i]))
            mtp.reset_vocab_window(tail)
        suffix_matchers = [None] * C
        if _use_mtp_suffix() and token_contexts is not None:
            for i in range(C):
                if elig[i] and specs[i] is None:
                    hist = token_contexts[i].tokens.tolist() if hasattr(token_contexts[i], "tokens") else []
                    suffix_matchers[i] = SuffixMatcher(
                        hist + [int(inputs[i].item()), int(cur_toks[i])],
                        min_gram=_suffix_min_gram(),
                        max_gram=8,
                    )

        return cls(
            model=model,
            mtp=mtp,
            vcache=vcache,
            mtp_caches=mtp_caches,
            mtp_exact_offs=mtp_exact_offs,
            cur_toks=cur_toks,
            cur_lps=list(lp0),
            cur_hidden=hidden0[:, 0, :],
            inflight=None,
            spec_eligible=elig,
            sampling_specs=specs,
            suffix_matchers=suffix_matchers,
        )

    def cancel_inflight(self) -> None:
        if self.inflight is not None:
            inf = self.inflight
            mtp_cancel_inflight(inf, self.mtp_caches)
            joining = inf.get("joining")
            if joining:
                C_act = inf["C_act"]
                _contract_verify_cache(self.model, self.vcache, list(range(C_act)))
                for pj in joining:
                    if pj["pending_h"] is not None and pj["spec_eligible"] and pj["mc"].offset > 0:
                        pj["mc"].trim(1)
                self.pending_joins = list(joining) + self.pending_joins
            self.inflight = None

    def extract_cache(self, idx: int) -> list:
        self.cancel_inflight()
        if self.pending_joins:
            self._flush_pending_joins_standalone()
        C = len(self.cur_toks)
        pending_h = self.cur_hidden[idx : idx + 1].reshape(1, 1, -1)
        mtp_st = (self.mtp_caches[idx], pending_h) if self.spec_eligible[idx] else None
        return mtp_extract_from_vcache(
            self.model,
            self.vcache,
            idx,
            C,
            mtp_state=mtp_st,
            copy_kv=True,
        )

    def to_standard_cache(self) -> list:
        self.cancel_inflight()
        if self.pending_joins:
            self._flush_pending_joins_standalone()
        return mtp_vcache_to_standard(self.model, self.vcache, len(self.cur_toks))

    def nbytes(self) -> int:
        tot = mtp_vcache_nbytes(self.vcache) + sum(mc.nbytes for mc in self.mtp_caches)
        for pj in self.pending_joins:
            tot += mtp_vcache_nbytes(pj["vcache"]) + pj["mc"].nbytes
        return tot

    def filter(self, keep: list[int]) -> None:
        self.cancel_inflight()
        if self.pending_joins:
            self._flush_pending_joins_standalone()
        if not keep:
            self.vcache.clear()
            self.mtp_caches.clear()
            self.mtp_exact_offs.clear()
            self.cur_toks.clear()
            self.cur_lps.clear()
            self.spec_eligible.clear()
            self.sampling_specs.clear()
            self.suffix_matchers.clear()
            return
        C = len(self.cur_toks)
        if len(keep) < C:
            _contract_verify_cache(self.model, self.vcache, keep)
            keep_arr = mx.array(keep, dtype=mx.int32)
            self.mtp_caches = [self.mtp_caches[i] for i in keep]
            self.mtp_exact_offs = [self.mtp_exact_offs[i] for i in keep]
            self.cur_toks = [self.cur_toks[i] for i in keep]
            self.cur_lps = [self.cur_lps[i] for i in keep]
            self.cur_hidden = self.cur_hidden[keep_arr]
            self.spec_eligible = [self.spec_eligible[i] for i in keep]
            self.sampling_specs = [self.sampling_specs[i] for i in keep]
            self.suffix_matchers = [self.suffix_matchers[i] for i in keep]

    def extend(self, other: "LingSanteSpecBatch") -> None:
        self.cancel_inflight()
        other.cancel_inflight()
        if self.pending_joins:
            self._flush_pending_joins_standalone()
        if other.pending_joins:
            other._flush_pending_joins_standalone()
        C_a = len(self.cur_toks)
        C_b = len(other.cur_toks)
        self.vcache = mtp_extend_vcache(self.model, self.vcache, C_a, other.vcache, C_b)
        self.mtp_caches.extend(other.mtp_caches)
        self.mtp_exact_offs.extend(other.mtp_exact_offs)
        self.cur_toks.extend(other.cur_toks)
        self.cur_lps.extend(other.cur_lps)
        self.cur_hidden = mx.concatenate([self.cur_hidden, other.cur_hidden], axis=0)
        self.spec_eligible.extend(other.spec_eligible)
        self.sampling_specs.extend(other.sampling_specs)
        self.suffix_matchers.extend(other.suffix_matchers)

    def _use_varlen_now(self, C_act: int, has_joins: bool) -> bool:
        if not _has_varlen_verify():
            return False
        return has_joins or C_act > 1 or (C_act == 1 and not self.spec_eligible[0])

    def _launch_cycle(self, gen_batch) -> None:
        C_act = len(self.cur_toks)
        pjs = self.pending_joins
        lids = getattr(self.mtp, "_feat_layers", None)

        if self._use_varlen_now(C_act, bool(pjs)):
            self._launch_cycle_varlen(gen_batch)
            return

        max_rem = max(gen_batch.max_tokens[i] - gen_batch._num_tokens[i] for i in range(C_act))
        K_limit = max(0, max_rem - 1)
        cur_arr = mx.array(self.cur_toks, dtype=mx.uint32)
        has_sampled = any(sp is not None for sp in self.sampling_specs)
        draft_q_lists: list[list[mx.array] | None] | None = None
        if has_sampled:
            draft_mat, after0_offs, draft_q_lists = _chain_mtp_drafts_mixed(
                self.mtp,
                cur_arr,
                self.cur_hidden,
                self.mtp_caches,
                sampling_specs=self.sampling_specs,
                C=C_act,
                K_limit=K_limit,
                use_q4_head=True,
            )
        else:
            draft_mat, after0_offs = _chain_mtp_drafts(
                self.mtp,
                cur_arr,
                self.cur_hidden,
                self.mtp_caches,
                C=C_act,
                K_limit=K_limit,
                use_q4_head=True,
                suffix_matcher=self.suffix_matchers[0] if C_act == 1 else None,
                anchor_tok_int=self.cur_toks[0] if C_act == 1 else None,
            )
        K = int(draft_mat.shape[1])
        if K > 0:
            mx.async_eval(draft_mat)
            inp = mx.concatenate([cur_arr[:, None], draft_mat], axis=1)
        else:
            inp = cur_arr[:, None]
        logits, hidden, rb = fast_verify_forward(
            self.model, inp, self.vcache, return_hidden=True, return_rollback=True, capture_layers=lids
        )
        preds, lp = _logits_to_preds(logits)
        ev_t = _verify_eval_targets(preds, hidden, draft_mat if K > 0 else None, lp, rb)
        if draft_q_lists is not None:
            for ql in draft_q_lists:
                if ql:
                    ev_t.extend(ql)
        mx.async_eval(*ev_t)
        self.mtp_exact_offs = after0_offs
        self.inflight = {
            "varlen": False,
            "K": K,
            "preds": preds,
            "hidden": hidden,
            "draft_mat": draft_mat,
            "draft_q_lists": draft_q_lists,
            "lp": lp,
            "rb": rb,
            "mtp_base_offs": [o - 1 for o in after0_offs],
        }

    def _launch_cycle_varlen(
        self,
        gen_batch,
        *,
        h_eff_prev: list[mx.array] | None = None,
        drafts_prev: list[list[int]] | None = None,
        ncs_prev: list[int] | None = None,
        after0_prev: list[int] | None = None,
    ) -> None:
        """Launch heterogeneous varlen verify cycle (`sum(Ls) <= B_max`) + optional `L_new=1` joins."""
        C_act = len(self.cur_toks)
        lids = getattr(self.mtp, "_feat_layers", None)
        pjs = self.pending_joins
        C_join = len(pjs)
        self.pending_joins = []

        # Refresh MTP KV for accepted tokens of previous cycle if called directly from step_next
        if h_eff_prev is not None and drafts_prev is not None and ncs_prev is not None and after0_prev is not None:
            elig_caches = [self.mtp_caches[i] for i in range(C_act) if self.spec_eligible[i]]
            elig_a0 = [after0_prev[i] for i in range(C_act) if self.spec_eligible[i]]
            elig_d = [drafts_prev[i] for i in range(C_act) if self.spec_eligible[i]]
            elig_h = [h_eff_prev[i] for i in range(C_act) if self.spec_eligible[i]]
            elig_nc = [ncs_prev[i] for i in range(C_act) if self.spec_eligible[i]]
            if elig_caches:
                _refresh_mtp_accepted_list(self.mtp, elig_caches, elig_a0, elig_d, elig_h, elig_nc)

        # Merge any pending_joins into vcache now (zero-copy split KDA + list MLA)
        if C_join > 0:
            vc_j = pjs[0]["vcache"]
            for i in range(1, C_join):
                vc_j = mtp_extend_vcache(self.model, vc_j, i, pjs[i]["vcache"], 1)
            self.vcache = mtp_extend_vcache(self.model, self.vcache, C_act, vc_j, C_join)

        b_max = max(C_act + C_join, _varlen_bmax())
        k_limits = [
            max(0, gen_batch.max_tokens[i] - gen_batch._num_tokens[i] - 1)
            if self.spec_eligible[i]
            else 0
            for i in range(C_act)
        ]

        draft_lists, after0_offs, mtp_base_offs, draft_q_lists = _chain_mtp_drafts_varlen(
            self.mtp,
            self.cur_toks,
            self.cur_hidden,
            self.mtp_caches,
            spec_eligible=self.spec_eligible,
            sampling_specs=self.sampling_specs,
            k_limits=k_limits,
            b_budget=b_max - C_join,
            use_q4_head=True,
            suffix_matchers=self.suffix_matchers,
        )

        # Pack flat inputs: active streams [cur_tok, *drafts] followed by pending_joins [last_tok]
        Ls = [1 + len(d) for d in draft_lists] + [1] * C_join
        row_starts = []
        flat_toks = []
        pos = 0
        for i in range(C_act):
            row_starts.append(pos)
            flat_toks.append(self.cur_toks[i])
            flat_toks.extend(draft_lists[i])
            pos += Ls[i]
        for pj in pjs:
            row_starts.append(pos)
            flat_toks.append(pj["last_tok"])
            pos += 1

        inp_flat = mx.array(flat_toks, dtype=mx.uint32)
        logits, hidden, rb = _vd.fast_verify_forward_varlen(
            self.model,
            inp_flat,
            self.vcache,
            Ls=Ls,
            return_hidden=True,
            return_rollback=True,
            capture_layers=lids,
        )
        # Ensure rb carries Ls and row_starts for commit/cancel
        if "Ls" not in rb:
            rb["Ls"] = Ls
        if "row_starts" not in rb:
            rb["row_starts"] = row_starts

        # Enqueue pending_joins MTP kv_append (overlapped with target verify!)
        join_evals = []
        for pj in pjs:
            if pj["pending_h"] is not None and pj["spec_eligible"]:
                self.mtp.kv_append(
                    mx.array([pj["last_tok"]], dtype=mx.uint32),
                    pj["pending_h"][:, 0, :],
                    pj["mc"],
                )
                join_evals.extend(pj["mc"].state)

        preds, lp = _logits_to_preds(logits)
        eval_t = _verify_eval_targets(preds, hidden, None, lp, rb)
        eval_t.extend(join_evals)
        for ql in draft_q_lists:
            if ql:
                eval_t.extend(ql)
        mx.async_eval(*eval_t)

        self.mtp_exact_offs = after0_offs
        self.inflight = {
            "varlen": True,
            "Ls": Ls,
            "row_starts": row_starts,
            "draft_lists": draft_lists,
            "draft_q_lists": draft_q_lists,
            "preds": preds,
            "hidden": hidden,
            "lp": lp,
            "rb": rb,
            "mtp_base_offs": mtp_base_offs,
            "joining": pjs,
            "C_act": C_act,
        }

    def _sample_nongreedy_row(self, gen_batch, s: int, row_lp: mx.array, cur_tok: int) -> tuple[int, mx.array]:
        """Exact bit-identical non-greedy sampling & logits_processors on a single `(V,)` logprobs row."""
        procs = gen_batch.logits_processors[s] if s < len(gen_batch.logits_processors) else ()
        row_2d = row_lp[None, :]
        if procs:
            ctx_toks = gen_batch._token_context[s].update_and_fetch(
                mx.array([cur_tok], dtype=mx.uint32)
            )
            for p in procs:
                row_2d = p(ctx_toks, row_2d)
            row_2d = row_2d - mx.logsumexp(row_2d, axis=-1, keepdims=True)
        sampler = (
            gen_batch.samplers[s]
            if (gen_batch.samplers and gen_batch.samplers[s] is not None)
            else gen_batch.fallback_sampler
        )
        tok = sampler(row_2d)
        mx.eval(tok, row_2d)
        return int(tok.reshape(-1)[0].item()), row_2d.squeeze(0)

    def step_next(self, gen_batch, *, prefetch: bool = True) -> list:
        C = len(self.cur_toks)
        if C == 0:
            if self.pending_joins:
                self._flush_pending_joins_standalone()
                C = len(self.cur_toks)
            if C == 0:
                return []

        lids = getattr(self.mtp, "_feat_layers", None)
        max_rem = max(gen_batch.max_tokens[i] - gen_batch._num_tokens[i] for i in range(C))
        if self.inflight is None and not self.pending_joins and max_rem <= 1:
            # Every stream has 1 token left and its prediction + logprobs are already ready.
            fin_lps = list(self.cur_lps)
            cur_arr = mx.array(self.cur_toks, dtype=mx.uint32)
            elig_idx = [i for i in range(C) if self.spec_eligible[i]]
            if elig_idx:
                self.mtp.kv_append_varlen(
                    mx.array([self.cur_toks[i] for i in elig_idx], dtype=mx.uint32),
                    mx.stack([self.cur_hidden[i] for i in elig_idx], axis=0),
                    [self.mtp_caches[i] for i in elig_idx],
                    [1] * len(elig_idx),
                )
            _, hidden_fin, rb_fin = fast_verify_forward(
                self.model,
                cur_arr[:, None],
                self.vcache,
                return_hidden=True,
                return_rollback=True,
                capture_layers=lids,
            )
            hidden_fin = self.mtp.apply_feat_fuse(hidden_fin, rb_fin.get("captured"))
            mx.eval(hidden_fin, *[x for i in elig_idx for x in self.mtp_caches[i].state])
            commit_verify_cache(self.vcache, rb_fin, 1)
            responses = []
            for i in range(C):
                tok = self.cur_toks[i]
                gen_batch.tokens[i].append(tok)
                gen_batch._num_tokens[i] += 1
                fr = "stop" if gen_batch._matchers[i].advance(tok) else "length"
                mtp_st = (
                    (self.mtp_caches[i], hidden_fin[i : i + 1, 0:1, :])
                    if self.spec_eligible[i]
                    else None
                )
                pc = mtp_extract_from_vcache(
                    self.model,
                    self.vcache,
                    i,
                    C,
                    mtp_state=mtp_st,
                    copy_kv=True,
                )
                responses.append(
                    gen_batch.Response(
                        uid=gen_batch.uids[i],
                        token=tok,
                        logprobs=fin_lps[i],
                        finish_reason=fr,
                        prompt_cache=pc,
                        all_tokens=list(gen_batch.tokens[i]),
                    )
                )
            gen_batch._clear_all()
            return responses

        if self.inflight is None:
            self._launch_cycle(gen_batch)

        inf = self.inflight
        is_varlen = bool(inf.get("varlen", False))
        preds, hidden, lp, rb = inf["preds"], inf["hidden"], inf["lp"], inf["rb"]
        draft_mat = inf.get("draft_mat")
        mx.eval(*_verify_eval_targets(preds, hidden, draft_mat if (draft_mat is not None and draft_mat.shape[1] > 0) else None, lp, rb))
        self.inflight = None

        if is_varlen:
            return self._finish_varlen_step(gen_batch, inf, prefetch=prefetch)

        K = inf["K"]
        draft_q_lists = inf.get("draft_q_lists")
        pred_lists = preds.tolist() if draft_q_lists is None else None
        draft_lists = draft_mat.tolist() if K > 0 else [[] for _ in range(C)]

        ncs: list[int] = []
        keep: list[int] = []
        fin_indices: list[int] = []
        fin_reasons: dict[int, str] = {}
        stream_toks: list[list[int]] = []
        next_cur_toks: list[int] = []
        next_cur_lps: list[mx.array] = []

        for s in range(C):
            sp_s = self.sampling_specs[s]
            if sp_s is not None:
                ql_s = draft_q_lists[s] if (draft_q_lists is not None and draft_q_lists[s]) else []
                a, nxt_tok_s = _rejection_verify_stream(lp[s], draft_lists[s], ql_s, sp_s)
            else:
                if pred_lists is None:
                    pred_lists = preds.tolist()
                a = 0
                for j in range(K):
                    if pred_lists[s][j] == draft_lists[s][j]:
                        a += 1
                    else:
                        break
                nxt_tok_s = pred_lists[s][a]
            cand_toks = [self.cur_toks[s]] + draft_lists[s][:a]
            emitted_s = []
            fr = None
            for tok in cand_toks:
                emitted_s.append(tok)
                gen_batch._num_tokens[s] += 1
                hit_len = gen_batch._num_tokens[s] >= gen_batch.max_tokens[s]
                hit_stop = gen_batch._matchers[s].advance(tok)
                if hit_stop:
                    fr = "stop"
                    break
                if hit_len:
                    fr = "length"
                    break
            k_emit = len(emitted_s)
            gen_batch.tokens[s].extend(emitted_s)
            stream_toks.append(emitted_s)
            ncs.append(k_emit)
            if fr is not None:
                fin_indices.append(s)
                fin_reasons[s] = fr
            else:
                keep.append(s)
                next_cur_toks.append(nxt_tok_s)
                next_cur_lps.append(lp[s, a])
                if C == 1 and self.suffix_matchers[s] is not None:
                    tl = getattr(self.suffix_matchers[s], "_last_tail_len", 0)
                    if tl > 0:
                        self.suffix_matchers[s].observe_outcome(max(0, a - 1), tl)
                        self.suffix_matchers[s]._last_tail_len = 0
                    self.suffix_matchers[s].extend(draft_lists[s][: max(0, k_emit - 1)] + [nxt_tok_s])

        # Slice FeatFuse strictly to consumed [:nc_s] rows!
        h_eff_list = _apply_feat_fuse_consumed(self.mtp, hidden, rb.get("captured"), ncs)

        old_uids = list(gen_batch.uids)
        old_cur_lps = list(self.cur_lps)
        after0_offs = list(self.mtp_exact_offs)

        # Commit verified prefix lengths to target KDA + MLA caches
        commit_verify_cache(self.vcache, rb, ncs)

        # S1: Snapshot finished streams' refs BEFORE contracting vcache, deferring mx.contiguous/eval
        # until AFTER the next async cycle is launched for surviving streams!
        fin_snaps: dict[int, tuple] = {}
        fin_tokens: dict[int, list[int]] = {}
        if fin_indices:
            for s in fin_indices:
                fin_snaps[s] = (
                    _snapshot_stream_vcache(self.model, self.vcache, s, C),
                    self.mtp_caches[s],
                    after0_offs[s],
                    draft_lists[s],
                    h_eff_list[s],
                    ncs[s],
                )
                fin_tokens[s] = list(gen_batch.tokens[s])

        if not keep:
            fin_caches = {}
            eval_fin = []
            for s in fin_indices:
                snap_vc, mc_s, aoff_s, d_s, he_s, nc_s = fin_snaps[s]
                tr = mc_s.offset - aoff_s
                if tr > 0:
                    mc_s.trim(tr)
                if nc_s > 1:
                    self.mtp.kv_append(mx.array(d_s[: nc_s - 1], dtype=mx.uint32), he_s[: nc_s - 1, :], mc_s)
                pending_h = he_s[nc_s - 1 : nc_s, :][None]
                eval_fin.extend(mc_s.state)
                eval_fin.append(pending_h)
                fin_caches[s] = _finalize_snap_cache(snap_vc, (mc_s, pending_h))
            if eval_fin:
                mx.eval(*eval_fin)
            responses = []
            for s in range(C):
                em = stream_toks[s]
                last_k = len(em) - 1
                fr = fin_reasons[s]
                for k_i, tok in enumerate(em):
                    tok_lp = old_cur_lps[s] if k_i == 0 else lp[s, k_i - 1]
                    is_last = k_i == last_k
                    responses.append(
                        gen_batch.Response(
                            uid=old_uids[s],
                            token=tok,
                            logprobs=tok_lp,
                            finish_reason=fr if is_last else None,
                            prompt_cache=fin_caches[s] if is_last else None,
                            all_tokens=fin_tokens[s] if is_last else None,
                        )
                    )
            gen_batch._clear_all()
            return responses

        # Contract active batch if one or more streams finished
        h_eff_kept = h_eff_list
        drafts_kept = draft_lists
        ncs_kept = ncs
        after0_kept = after0_offs
        if len(keep) < C:
            _contract_verify_cache(self.model, self.vcache, keep)
            for s in keep:
                tr = self.mtp_caches[s].offset - after0_offs[s]
                if tr > 0:
                    self.mtp_caches[s].trim(tr)
            self.mtp_caches = [self.mtp_caches[s] for s in keep]
            self.spec_eligible = [self.spec_eligible[s] for s in keep]
            self.sampling_specs = [self.sampling_specs[s] for s in keep]
            self.suffix_matchers = [self.suffix_matchers[s] for s in keep]
            after0_kept = [after0_offs[s] for s in keep]
            drafts_kept = [draft_lists[s] for s in keep]
            ncs_kept = [ncs[s] for s in keep]
            h_eff_kept = [h_eff_list[s] for s in keep]
            gen_batch._filter_metadata(keep)

        C_act = len(keep)
        new_cur_hidden = mx.stack([h_eff_kept[i][ncs_kept[i] - 1, :] for i in range(C_act)], axis=0)
        self.cur_toks = next_cur_toks
        self.cur_lps = next_cur_lps
        self.cur_hidden = new_cur_hidden

        max_rem_next = max(gen_batch.max_tokens[i] - gen_batch._num_tokens[i] for i in range(C_act))
        if prefetch and max_rem_next > 1:
            if self._use_varlen_now(C_act, bool(self.pending_joins)):
                self._launch_cycle_varlen(
                    gen_batch,
                    h_eff_prev=h_eff_kept,
                    drafts_prev=drafts_kept,
                    ncs_prev=ncs_kept,
                    after0_prev=after0_kept,
                )
            else:
                K_lim = max(0, max_rem_next - 1)
                has_sampled = any(sp is not None for sp in self.sampling_specs)
                draft_q_next: list[list[mx.array] | None] | None = None
                if has_sampled:
                    draft_mat_next, new_after0_offs, draft_q_next = _sync_and_chain_mtp_mixed(
                        self.mtp,
                        h_eff_kept,
                        drafts_kept,
                        next_cur_toks,
                        ncs_kept,
                        self.mtp_caches,
                        after0_kept,
                        sampling_specs=self.sampling_specs,
                        C=C_act,
                        K_limit=K_lim,
                        use_q4_head=True,
                    )
                else:
                    draft_mat_next, new_after0_offs = _sync_and_chain_mtp_list(
                        self.mtp,
                        h_eff_kept,
                        drafts_kept,
                        next_cur_toks,
                        ncs_kept,
                        self.mtp_caches,
                        after0_kept,
                        C=C_act,
                        K_limit=K_lim,
                        use_q4_head=True,
                        suffix_matcher=self.suffix_matchers[0] if C_act == 1 else None,
                    )
                K_next = int(draft_mat_next.shape[1])
                if K_next > 0:
                    mx.async_eval(draft_mat_next)
                cur_arr = mx.array(next_cur_toks, dtype=mx.uint32)
                inp_next = (
                    mx.concatenate([cur_arr[:, None], draft_mat_next], axis=1)
                    if K_next > 0
                    else cur_arr[:, None]
                )
                logits_next, hidden_next, rb_next = fast_verify_forward(
                    self.model, inp_next, self.vcache, return_hidden=True, return_rollback=True, capture_layers=lids
                )
                preds_next, lp_next = _logits_to_preds(logits_next)
                ev_n = _verify_eval_targets(
                    preds_next, hidden_next, draft_mat_next if K_next > 0 else None, lp_next, rb_next
                )
                if draft_q_next is not None:
                    for ql in draft_q_next:
                        if ql:
                            ev_n.extend(ql)
                mx.async_eval(*ev_n)
                self.mtp_exact_offs = new_after0_offs
                self.inflight = {
                    "varlen": False,
                    "K": K_next,
                    "preds": preds_next,
                    "hidden": hidden_next,
                    "draft_mat": draft_mat_next,
                    "draft_q_lists": draft_q_next,
                    "lp": lp_next,
                    "rb": rb_next,
                    "mtp_base_offs": [o - 1 for o in new_after0_offs],
                }
        else:
            _refresh_mtp_accepted_list(
                self.mtp, self.mtp_caches, after0_kept, drafts_kept, h_eff_kept, ncs_kept
            )
            self.mtp_exact_offs = [mc.offset for mc in self.mtp_caches]
            self.inflight = None
            flush_arrs = [self.cur_hidden]
            for mc in self.mtp_caches:
                flush_arrs.extend(mc.state)
            mx.async_eval(*flush_arrs)

        # S1: Deferred finish cache materialization (overlapped with GPU verify of surviving streams!)
        fin_caches: dict[int, list] = {}
        if fin_indices:
            eval_fin = []
            for s in fin_indices:
                snap_vc, mc_s, aoff_s, d_s, he_s, nc_s = fin_snaps[s]
                tr = mc_s.offset - aoff_s
                if tr > 0:
                    mc_s.trim(tr)
                if nc_s > 1:
                    self.mtp.kv_append(mx.array(d_s[: nc_s - 1], dtype=mx.uint32), he_s[: nc_s - 1, :], mc_s)
                pending_h = he_s[nc_s - 1 : nc_s, :][None]
                eval_fin.extend(mc_s.state)
                eval_fin.append(pending_h)
                fin_caches[s] = _finalize_snap_cache(snap_vc, (mc_s, pending_h))
            mx.eval(*eval_fin)

        # Keep suffix_matchers & D1 vocab window fresh during GPU verify overlap (0 ms critical path)
        if _has_d1_mtp(self.mtp):
            self.mtp.update_vocab_window(next_cur_toks)
        if C > 1:
            for i, s in enumerate(keep):
                sm = self.suffix_matchers[i]
                if sm is not None:
                    sm.extend(draft_lists[s][: max(0, ncs[s] - 1)] + [next_cur_toks[i]])

        responses = []
        for s in range(C):
            em = stream_toks[s]
            last_k = len(em) - 1
            fr = fin_reasons.get(s)
            for k_i, tok in enumerate(em):
                tok_lp = old_cur_lps[s] if k_i == 0 else lp[s, k_i - 1]
                is_last = k_i == last_k and fr is not None
                responses.append(
                    gen_batch.Response(
                        uid=old_uids[s],
                        token=tok,
                        logprobs=tok_lp,
                        finish_reason=fr if is_last else None,
                        prompt_cache=fin_caches[s] if is_last else None,
                        all_tokens=fin_tokens[s] if is_last else None,
                    )
                )
        return responses

    def _finish_varlen_step(self, gen_batch, inf: dict, *, prefetch: bool) -> list:
        """Complete a heterogeneous varlen verify cycle (with optional mixed non-greedy & L_new=1 joins)."""
        C_act = inf["C_act"]
        pjs = inf["joining"]
        C_join = len(pjs)
        C_tot = C_act + C_join
        Ls = inf["Ls"]
        row_starts = inf["row_starts"]
        draft_lists = inf["draft_lists"]
        draft_q_lists = inf.get("draft_q_lists")
        preds_flat = inf["preds"].tolist()
        hidden = inf["hidden"]
        lp_flat = inf["lp"]
        rb = inf["rb"]

        ncs: list[int] = []
        keep_all: list[int] = []
        fin_indices: list[int] = []
        fin_reasons: dict[int, str] = {}
        stream_toks: list[list[int]] = []
        stream_lps: list[list[mx.array]] = []
        next_cur_toks: list[int] = []
        next_cur_lps: list[mx.array] = []

        # 1. Process active streams 0..C_act-1
        for s in range(C_act):
            st = row_starts[s]
            d_s = draft_lists[s]
            K_s = len(d_s)
            sp_s = self.sampling_specs[s]
            sampled_nxt_s = None
            if self.spec_eligible[s] and sp_s is not None:
                ql_s = draft_q_lists[s] if (draft_q_lists is not None and draft_q_lists[s]) else []
                a, sampled_nxt_s = _rejection_verify_stream(
                    lp_flat[st : st + K_s + 1], d_s, ql_s, sp_s
                )
                cand_toks = [self.cur_toks[s]] + d_s[:a]
                cand_lps = [self.cur_lps[s]] + [lp_flat[st + j] for j in range(a)]
            elif self.spec_eligible[s]:
                a = 0
                for j in range(K_s):
                    if preds_flat[st + j] == d_s[j]:
                        a += 1
                    else:
                        break
                cand_toks = [self.cur_toks[s]] + d_s[:a]
                cand_lps = [self.cur_lps[s]] + [lp_flat[st + j] for j in range(a)]
            else:
                a = 0
                cand_toks = [self.cur_toks[s]]
                cand_lps = [self.cur_lps[s]]

            emitted_s = []
            emitted_lps = []
            fr = None
            for tok, tlp in zip(cand_toks, cand_lps):
                emitted_s.append(tok)
                emitted_lps.append(tlp)
                gen_batch._num_tokens[s] += 1
                hit_len = gen_batch._num_tokens[s] >= gen_batch.max_tokens[s]
                hit_stop = gen_batch._matchers[s].advance(tok)
                if hit_stop:
                    fr = "stop"
                    break
                if hit_len:
                    fr = "length"
                    break
            k_emit = len(emitted_s)
            gen_batch.tokens[s].extend(emitted_s)
            if self.suffix_matchers[s] is not None:
                tl = getattr(self.suffix_matchers[s], "_last_tail_len", 0)
                if tl > 0:
                    self.suffix_matchers[s].observe_outcome(max(0, a - 1), tl)
                    self.suffix_matchers[s]._last_tail_len = 0
                self.suffix_matchers[s].extend(d_s[: max(0, k_emit - 1)])
            stream_toks.append(emitted_s)
            stream_lps.append(emitted_lps)
            ncs.append(k_emit)
            if fr is not None:
                fin_indices.append(s)
                fin_reasons[s] = fr
            else:
                keep_all.append(s)
                if self.spec_eligible[s] and sp_s is not None:
                    nxt_t = int(sampled_nxt_s)
                    nxt_lp = lp_flat[st + a]
                elif self.spec_eligible[s]:
                    nxt_t = int(preds_flat[st + a])
                    nxt_lp = lp_flat[st + a]
                    if self.suffix_matchers[s] is not None:
                        self.suffix_matchers[s].extend([nxt_t])
                else:
                    nxt_t, nxt_lp = self._sample_nongreedy_row(
                        gen_batch, s, lp_flat[st], self.cur_toks[s]
                    )
                next_cur_toks.append(nxt_t)
                next_cur_lps.append(nxt_lp)

        # 2. Process newly joined L_new=1 streams C_act..C_tot-1 (prompt token verified -> ready as cur_toks!)
        for ji, pj in enumerate(pjs):
            s = C_act + ji
            st = row_starts[s]
            ncs.append(1)
            keep_all.append(s)
            pj_sp = pj.get("sampling_spec")
            if pj["spec_eligible"] and pj_sp is None:
                nxt_t = int(preds_flat[st])
                nxt_lp = lp_flat[st]
            elif pj["spec_eligible"] and pj_sp is not None:
                y_j = _sample_from_probs(_logprobs_to_probs(lp_flat[st], pj_sp))
                mx.eval(y_j)
                nxt_t = int(y_j.item())
                nxt_lp = lp_flat[st]
            else:
                nxt_t, nxt_lp = self._sample_nongreedy_row(gen_batch, s, lp_flat[st], pj["last_tok"])
            next_cur_toks.append(nxt_t)
            next_cur_lps.append(nxt_lp)

        # Slice FeatFuse to consumed positions only across all C_tot streams
        h_eff_all = _apply_feat_fuse_consumed(
            self.mtp, hidden, rb.get("captured"), ncs, varlen_starts=row_starts
        )
        old_uids = list(gen_batch.uids[:C_act])
        after0_all = list(self.mtp_exact_offs) + [pj["mc"].offset for pj in pjs]
        drafts_all = list(draft_lists) + [[] for _ in range(C_join)]
        caches_all = list(self.mtp_caches) + [pj["mc"] for pj in pjs]
        elig_all = list(self.spec_eligible) + [pj["spec_eligible"] for pj in pjs]
        specs_all = list(self.sampling_specs) + [pj.get("sampling_spec") for pj in pjs]
        sfx_all = list(self.suffix_matchers) + [
            SuffixMatcher(pj["tokens"] + [next_cur_toks[len(keep_all) - C_join + ji]], min_gram=_suffix_min_gram(), max_gram=8)
            if (_use_mtp_suffix() and pj["spec_eligible"] and pj.get("sampling_spec") is None)
            else None
            for ji, pj in enumerate(pjs)
        ]

        _commit_verify_any(self.vcache, rb, ncs)

        fin_snaps: dict[int, tuple] = {}
        fin_tokens: dict[int, list[int]] = {}
        if fin_indices:
            for s in fin_indices:
                fin_snaps[s] = (
                    _snapshot_stream_vcache(self.model, self.vcache, s, C_tot),
                    caches_all[s],
                    after0_all[s],
                    drafts_all[s],
                    h_eff_all[s],
                    ncs[s],
                    elig_all[s],
                )
                fin_tokens[s] = list(gen_batch.tokens[s])

        if not keep_all:
            fin_caches = {}
            eval_fin = []
            for s in fin_indices:
                snap_vc, mc_s, aoff_s, d_s, he_s, nc_s, el_s = fin_snaps[s]
                mtp_st = None
                if el_s:
                    tr = mc_s.offset - aoff_s
                    if tr > 0:
                        mc_s.trim(tr)
                    if nc_s > 1:
                        self.mtp.kv_append(mx.array(d_s[: nc_s - 1], dtype=mx.uint32), he_s[: nc_s - 1, :], mc_s)
                    pending_h = he_s[nc_s - 1 : nc_s, :][None]
                    eval_fin.extend(mc_s.state)
                    eval_fin.append(pending_h)
                    mtp_st = (mc_s, pending_h)
                fin_caches[s] = _finalize_snap_cache(snap_vc, mtp_st)
            if eval_fin:
                mx.eval(*eval_fin)
            responses = []
            for s in range(C_act):
                em, elp = stream_toks[s], stream_lps[s]
                last_k = len(em) - 1
                fr = fin_reasons[s]
                for k_i, (tok, tok_lp) in enumerate(zip(em, elp)):
                    is_last = k_i == last_k
                    responses.append(
                        gen_batch.Response(
                            uid=old_uids[s],
                            token=tok,
                            logprobs=tok_lp,
                            finish_reason=fr if is_last else None,
                            prompt_cache=fin_caches[s] if is_last else None,
                            all_tokens=fin_tokens[s] if is_last else None,
                        )
                    )
            gen_batch._clear_all()
            return responses

        if len(keep_all) < C_tot:
            _contract_verify_cache(self.model, self.vcache, keep_all)
            gen_batch._filter_metadata(keep_all)

        self.mtp_caches = [caches_all[s] for s in keep_all]
        self.spec_eligible = [elig_all[s] for s in keep_all]
        self.sampling_specs = [specs_all[s] for s in keep_all]
        self.suffix_matchers = [sfx_all[s] for s in keep_all]
        after0_kept = [after0_all[s] for s in keep_all]
        drafts_kept = [drafts_all[s] for s in keep_all]
        ncs_kept = [ncs[s] for s in keep_all]
        h_eff_kept = [h_eff_all[s] for s in keep_all]

        C_new = len(keep_all)
        self.cur_toks = next_cur_toks
        self.cur_lps = next_cur_lps
        self.cur_hidden = mx.stack([h_eff_kept[i][ncs_kept[i] - 1, :] for i in range(C_new)], axis=0)

        max_rem_next = max(gen_batch.max_tokens[i] - gen_batch._num_tokens[i] for i in range(C_new))
        if prefetch and max_rem_next > 1 and any(self.spec_eligible):
            if self._use_varlen_now(C_new, bool(self.pending_joins)):
                self._launch_cycle_varlen(
                    gen_batch,
                    h_eff_prev=h_eff_kept,
                    drafts_prev=drafts_kept,
                    ncs_prev=ncs_kept,
                    after0_prev=after0_kept,
                )
            else:
                # Shrunk to C=1 (greedy or sampled)!
                K_lim = max(0, max_rem_next - 1)
                draft_q_next = None
                if self.sampling_specs[0] is not None:
                    draft_mat_next, new_a0, draft_q_next = _sync_and_chain_mtp_mixed(
                        self.mtp,
                        h_eff_kept,
                        drafts_kept,
                        self.cur_toks,
                        ncs_kept,
                        self.mtp_caches,
                        after0_kept,
                        sampling_specs=self.sampling_specs,
                        C=1,
                        K_limit=K_lim,
                        use_q4_head=True,
                    )
                else:
                    draft_mat_next, new_a0 = _sync_and_chain_mtp_list(
                        self.mtp,
                        h_eff_kept,
                        drafts_kept,
                        self.cur_toks,
                        ncs_kept,
                        self.mtp_caches,
                        after0_kept,
                        C=1,
                        K_limit=K_lim,
                        use_q4_head=True,
                        suffix_matcher=self.suffix_matchers[0],
                    )
                K_next = int(draft_mat_next.shape[1])
                if K_next > 0:
                    mx.async_eval(draft_mat_next)
                cur_arr = mx.array(self.cur_toks, dtype=mx.uint32)
                inp_next = (
                    mx.concatenate([cur_arr[:, None], draft_mat_next], axis=1)
                    if K_next > 0
                    else cur_arr[:, None]
                )
                logits_next, hidden_next, rb_next = fast_verify_forward(
                    self.model, inp_next, self.vcache, return_hidden=True, return_rollback=True,
                    capture_layers=getattr(self.mtp, "_feat_layers", None),
                )
                preds_next, lp_next = _logits_to_preds(logits_next)
                ev_n = _verify_eval_targets(
                    preds_next, hidden_next, draft_mat_next if K_next > 0 else None, lp_next, rb_next
                )
                if draft_q_next is not None:
                    for ql in draft_q_next:
                        if ql:
                            ev_n.extend(ql)
                mx.async_eval(*ev_n)
                self.mtp_exact_offs = new_a0
                self.inflight = {
                    "varlen": False,
                    "K": K_next,
                    "preds": preds_next,
                    "hidden": hidden_next,
                    "draft_mat": draft_mat_next,
                    "draft_q_lists": draft_q_next,
                    "lp": lp_next,
                    "rb": rb_next,
                    "mtp_base_offs": [o - 1 for o in new_a0],
                }
        else:
            elig_idx = [i for i in range(C_new) if self.spec_eligible[i]]
            if elig_idx:
                _refresh_mtp_accepted_list(
                    self.mtp,
                    [self.mtp_caches[i] for i in elig_idx],
                    [after0_kept[i] for i in elig_idx],
                    [drafts_kept[i] for i in elig_idx],
                    [h_eff_kept[i] for i in elig_idx],
                    [ncs_kept[i] for i in elig_idx],
                )
            self.mtp_exact_offs = [mc.offset for mc in self.mtp_caches]
            self.inflight = None
            flush_arrs = [self.cur_hidden]
            for i in elig_idx:
                flush_arrs.extend(self.mtp_caches[i].state)
            mx.async_eval(*flush_arrs)

        # Deferred finish cache materialization
        fin_caches = {}
        if fin_indices:
            eval_fin = []
            for s in fin_indices:
                snap_vc, mc_s, aoff_s, d_s, he_s, nc_s, el_s = fin_snaps[s]
                mtp_st = None
                if el_s:
                    tr = mc_s.offset - aoff_s
                    if tr > 0:
                        mc_s.trim(tr)
                    if nc_s > 1:
                        self.mtp.kv_append(mx.array(d_s[: nc_s - 1], dtype=mx.uint32), he_s[: nc_s - 1, :], mc_s)
                    pending_h = he_s[nc_s - 1 : nc_s, :][None]
                    eval_fin.extend(mc_s.state)
                    eval_fin.append(pending_h)
                    mtp_st = (mc_s, pending_h)
                fin_caches[s] = _finalize_snap_cache(snap_vc, mtp_st)
            if eval_fin:
                mx.eval(*eval_fin)

        responses = []
        for s in range(C_act):
            em, elp = stream_toks[s], stream_lps[s]
            last_k = len(em) - 1
            fr = fin_reasons.get(s)
            for k_i, (tok, tok_lp) in enumerate(zip(em, elp)):
                is_last = k_i == last_k and fr is not None
                responses.append(
                    gen_batch.Response(
                        uid=old_uids[s],
                        token=tok,
                        logprobs=tok_lp,
                        finish_reason=fr if is_last else None,
                        prompt_cache=fin_caches[s] if is_last else None,
                        all_tokens=fin_tokens[s] if is_last else None,
                    )
                )
        return responses


# ---------------------------------------------------------------------------
# Prefill & Drafting Helpers (A2 + A1 Varlen Water-Filling)
# ---------------------------------------------------------------------------


def prefill_with_hidden(model, mtp: BailingMTPLayer, ids: Sequence[int], step: int = 512):
    """Exact `B=1` prefill of `ids[:-1]` through target + shifted MTP cache prefill."""
    t_cache = make_prompt_cache(model)
    p = mx.array(ids, dtype=mx.uint32)
    h_chunks = []
    pos = 0
    while p.size - pos > 1:
        n = min(step, p.size - 1 - pos)
        h = mtp_target_prefill_hidden(model, mtp, p[pos : pos + n][None], t_cache)
        mx.eval(h, [c.state for c in t_cache])
        h_chunks.append(h)
        pos += n
    mtp_cache = KVCache()
    if h_chunks:
        h_all = h_chunks[0] if len(h_chunks) == 1 else mx.concatenate(h_chunks, axis=1)
        tokens_shifted = p[1:][None]
        s_pos = 0
        tot = tokens_shifted.shape[1]
        while s_pos < tot:
            m = min(step, tot - s_pos)
            mtp.kv_append(tokens_shifted[:, s_pos : s_pos + m], h_all[:, s_pos : s_pos + m, :], mtp_cache)
            mx.eval(mtp_cache.state)
            s_pos += m
    return t_cache, mtp_cache, int(p[-1].item())


def _resolve_draft_policy(mtp: BailingMTPLayer, C: int, K: int | None, K_limit: int | None):
    if K is not None:
        draft_mode, k_cap, dyn_th, p1_min = "fixed", K, 0.0, 0.0
    elif C <= 1:
        draft_mode, k_cap, dyn_th, p1_min = mtp_c1_policy(mtp)
    elif C == 2 and _c2_use_dyn(mtp):
        _, _, dyn_th, p1_min = mtp_c1_policy(mtp)
        draft_mode = "dyn_c2"
        k_cap = max(1, (_varlen_bmax() // 2) - 1)  # K in {0, 1, 2, 3} -> B in {2, 4, 6, 8} <= 8
    else:
        draft_mode, k_cap, dyn_th, p1_min = "fixed", mtp_k_for_batch(C, mtp), 0.0, 0.0
    if K_limit is not None:
        k_cap = min(k_cap, max(0, int(K_limit)))
    return draft_mode, max(0, int(k_cap)), dyn_th, p1_min


def _draft_steps_from_logits(
    mtp: BailingMTPLayer,
    cache_arg,
    *,
    C: int,
    K_cap: int,
    dl0: mx.array,
    h0_in: mx.array,
    h0_out: mx.array,
    draft_mode: str,
    dyn_th: float,
    p1_min: float,
    use_q4_head: bool,
    suffix_matcher: SuffixMatcher | None = None,
    anchor_tok_int: int | None = None,
) -> mx.array:
    if K_cap <= 0:
        return mx.zeros((C, 0), dtype=mx.uint32)

    if draft_mode == "dyn_step" and C == 1:
        r_star, t_draft, e_p_next, p0_drop, skip_sync_th, use_rr = _a2_params(dyn_th, p1_min)
        d0_arr, p0_arr = _step_top1_and_prob(dl0)
        r_inds0 = getattr(mtp, "_last_router_inds", None)
        need_d0_host = suffix_matcher is not None or anchor_tok_int is not None
        eval0 = [p0_arr, d0_arr] if need_d0_host else [p0_arr]
        if r_inds0 is not None:
            eval0.append(r_inds0)
        mx.eval(*eval0)

        p0 = float(p0_arr[0].item())
        d0_int = int(d0_arr[0].item()) if need_d0_host else -1
        seen_toks = {anchor_tok_int} if anchor_tok_int is not None else set()
        seen_mask = 0
        du72_0, seen_mask = _update_router_mask(mtp, 0, seen_mask)

        # A2 K=0 drop gate: skip verifying d0 when p0 is below marginal verify break-even!
        _, tau_dr0 = _thresholds_for_step(
            2, du72_0, d0_int in seen_toks, dyn_th=dyn_th, r_star=r_star, t_draft=t_draft, e_p_next=e_p_next, use_rr=use_rr
        )
        if p0 < p0_drop:
            return mx.zeros((1, 0), dtype=mx.uint32)

        # MTP-confirmed suffix extension (online acceptance-guarded min_gram, first_token == d0 -> 0 extra draft steps!)
        if suffix_matcher is not None and K_cap >= 2:
            eff_mg = suffix_matcher.effective_min_gram(_suffix_min_gram())
            gl, stoks = suffix_matcher.match(K_cap, min_gram=eff_mg, first_token=d0_int)
            if gl >= eff_mg and len(stoks) >= 2:
                suffix_matcher._last_tail_len = len(stoks) - 1
                return mx.array([stoks], dtype=mx.uint32)
            suffix_matcher._last_tail_len = 0

        cols = [d0_arr]
        if d0_int >= 0:
            seen_toks.add(d0_int)
        if K_cap <= 1 or p0 < tau_dr0:
            return mx.stack(cols, axis=1)

        cum_q = p0
        h_in = h0_in
        h_d = h0_out
        d_prev = d0_arr
        k_idx = 1

        # Confidence-gated skip-sync: when p0 >= skip_sync_th and K_cap >= 3, enqueue k=1 & k=2 together!
        if p0 >= skip_sync_th and K_cap >= 3:
            h_in_1 = mtp.apply_feedback_bridge(h_d, h_in)
            h_d1, dl1 = mtp.step_rows(d_prev, h_in_1, cache_arg, C=1, L=1, use_q4_head=use_q4_head)
            d1_arr, p1_arr = _step_top1_and_prob(dl1)
            r_inds1 = getattr(mtp, "_last_router_inds", None)

            h_in_2 = mtp.apply_feedback_bridge(h_d1, h_in_1)
            h_d2, dl2 = mtp.step_rows(d1_arr, h_in_2, cache_arg, C=1, L=1, use_q4_head=use_q4_head)
            d2_arr, p2_arr = _step_top1_and_prob(dl2)
            r_inds2 = getattr(mtp, "_last_router_inds", None)

            ev_pair = [p1_arr, p2_arr]
            if r_inds1 is not None:
                ev_pair.append(r_inds1)
            if r_inds2 is not None:
                ev_pair.append(r_inds2)
            mx.eval(*ev_pair)

            p1_v = float(p1_arr[0].item())
            du72_1 = None
            if r_inds1 is not None:
                b1 = 0
                for e in r_inds1[0].tolist():
                    b1 |= 1 << int(e)
                du72_1 = (b1 & ~seen_mask).bit_count()
                seen_mask |= b1
            cum_q1 = cum_q * p1_v
            tau_v1, tau_d1 = _thresholds_for_step(
                3, du72_1, False, dyn_th=dyn_th, r_star=r_star, t_draft=t_draft, e_p_next=e_p_next, use_rr=use_rr
            )
            if cum_q1 < tau_v1:
                return mx.stack(cols, axis=1)
            cols.append(d1_arr)
            cum_q = cum_q1
            if cum_q < tau_d1:
                return mx.stack(cols, axis=1)

            p2_v = float(p2_arr[0].item())
            du72_2 = None
            if r_inds2 is not None:
                b2 = 0
                for e in r_inds2[0].tolist():
                    b2 |= 1 << int(e)
                du72_2 = (b2 & ~seen_mask).bit_count()
                seen_mask |= b2
            cum_q2 = cum_q * p2_v
            tau_v2, tau_d2 = _thresholds_for_step(
                4, du72_2, False, dyn_th=dyn_th, r_star=r_star, t_draft=t_draft, e_p_next=e_p_next, use_rr=use_rr
            )
            if cum_q2 < tau_v2:
                return mx.stack(cols, axis=1)
            cols.append(d2_arr)
            cum_q = cum_q2
            h_in = h_in_2
            h_d = h_d2
            d_prev = d2_arr
            k_idx = 3
            if cum_q < tau_d2 or k_idx >= K_cap:
                return mx.stack(cols, axis=1)

        while k_idx < K_cap:
            h_in_next = mtp.apply_feedback_bridge(h_d, h_in)
            # Skip final-step host sync when cum_q is already high (avoids 1 pipeline drain at K_cap)
            if k_idx + 1 == K_cap and cum_q >= 0.45:
                h_d, d_last, _ = _mtp_step_top1(
                    mtp, d_prev, h_in_next, cache_arg, C=1, use_q4_head=use_q4_head
                )
                cols.append(d_last)
                break
            h_d, d_prev, pk_arr = _mtp_step_top1(
                mtp, d_prev, h_in_next, cache_arg, C=1, use_q4_head=use_q4_head
            )
            h_in = h_in_next
            r_indsk = getattr(mtp, "_last_router_inds", None)
            ev_k = [pk_arr] if r_indsk is None else [pk_arr, r_indsk]
            mx.eval(*ev_k)

            pk_v = float(pk_arr[0].item())
            du72_k, seen_mask = _update_router_mask(mtp, 0, seen_mask)
            cum_q = cum_q * pk_v
            B_next = len(cols) + 2
            tau_vk, tau_dk = _thresholds_for_step(
                B_next, du72_k, False, dyn_th=dyn_th, r_star=r_star, t_draft=t_draft, e_p_next=e_p_next, use_rr=use_rr
            )
            # P1 Trailing-Draft Pop Fix: drop d_prev if it fails marginal verify break-even!
            if cum_q < tau_vk:
                break
            cols.append(d_prev)
            k_idx += 1
            if k_idx >= K_cap or cum_q < tau_dk:
                break
        return mx.stack(cols, axis=1)

    if draft_mode == "dyn_c2" and C == 2:
        # Uniform dyn_step for C=2 (K in 0..3 -> B in 2,4,6,8 <= 8)
        _, _, _, p0_drop, _, _ = _a2_params(dyn_th, p1_min)
        d_prev, p0_arr = _step_top1_and_prob(dl0)
        mx.eval(p0_arr)
        p0_vals = p0_arr.tolist()
        mean_p0 = 0.5 * (p0_vals[0] + p0_vals[1])
        if mean_p0 < p0_drop:
            return mx.zeros((2, 0), dtype=mx.uint32)
        cols = [d_prev]
        if K_cap <= 1 or mean_p0 < 0.45:
            return mx.stack(cols, axis=1)
        cum_q = [p0_vals[0], p0_vals[1]]
        h_in = h0_in
        h_d = h0_out
        for k_idx in range(1, K_cap):
            h_in_next = mtp.apply_feedback_bridge(h_d, h_in)
            h_d, dlk = mtp.step_rows(d_prev, h_in_next, cache_arg, C=2, L=1, use_q4_head=use_q4_head)
            h_in = h_in_next
            d_prev, pk_arr = _step_top1_and_prob(dlk)
            mx.eval(pk_arr)
            pk_vals = pk_arr.tolist()
            cum_q = [cum_q[0] * pk_vals[0], cum_q[1] * pk_vals[1]]
            mean_q = 0.5 * (cum_q[0] + cum_q[1])
            if mean_q < 0.24:
                break
            cols.append(d_prev)
            if k_idx + 1 >= K_cap or mean_q < 0.52:
                break
        return mx.stack(cols, axis=1)

    # Fixed lazy GPU chaining (zero host-GPU sync!)
    d_prev = dl0[0] if isinstance(dl0, tuple) else mx.argmax(dl0, axis=-1).astype(mx.uint32)
    cols = [d_prev]
    h_in = h0_in
    h_d = h0_out
    for _ in range(1, K_cap):
        h_in_next = mtp.apply_feedback_bridge(h_d, h_in)
        if _has_d1_mtp(mtp):
            h_d, d_prev, _ = mtp.step_rows(
                d_prev, h_in_next, cache_arg, C=C, L=1, use_q4_head=use_q4_head, return_argmax_prob=True
            )
        else:
            h_d, dlk = mtp.step_rows(d_prev, h_in_next, cache_arg, C=C, L=1, use_q4_head=use_q4_head)
            d_prev = mx.argmax(dlk, axis=-1).astype(mx.uint32)
        h_in = h_in_next
        cols.append(d_prev)
    return mx.stack(cols, axis=1)


def _chain_mtp_drafts_varlen(
    mtp: BailingMTPLayer,
    cur_toks: list[int],
    cur_hidden: mx.array,
    mtp_caches: list[KVCache],
    *,
    spec_eligible: list[bool],
    sampling_specs: list[tuple[float, list[Callable]] | None] | None = None,
    k_limits: list[int],
    b_budget: int,
    use_q4_head: bool,
    suffix_matchers: list[SuffixMatcher | None] | None = None,
) -> tuple[list[list[int]], list[int], list[int], list[list[mx.array] | None]]:
    """Heterogeneous Varlen `K_i` Water-Filling Scheduler (`sum(1 + K_i) <= b_budget`).

    Allocates the `K_budget = b_budget - C` draft slots across spec-eligible streams (greedy
    and `temp > 0` sampled) by marginal utility, stepping only active subsets in batched
    `mtp.step_rows`. For sampled streams (`sampling_specs[s] is not None`), admission is decided
    strictly BEFORE drawing `d_k ~ q_k` (never popping a drawn token) to guarantee 100% exact
    target sampling distribution.
    """
    C = len(cur_toks)
    draft_lists: list[list[int]] = [[] for _ in range(C)]
    specs = list(sampling_specs) if sampling_specs is not None else [None] * C
    draft_q_lists: list[list[mx.array] | None] = [[] if specs[i] is not None else None for i in range(C)]
    base_offs = [mc.offset for mc in mtp_caches]
    after0_offs = list(base_offs)

    elig_ids = [i for i in range(C) if spec_eligible[i] and k_limits[i] > 0]
    k_budget = max(0, b_budget - C)
    if not elig_ids or k_budget <= 0:
        return draft_lists, after0_offs, base_offs, draft_q_lists

    _, _, dyn_th, p1_min = mtp_c1_policy(mtp)
    r_star, t_draft, e_p_next, p0_drop, _, use_rr = _a2_params(dyn_th, p1_min)
    r_eff = r_star * (1.0 + 0.12 * (C - 1))
    any_sampled = any(specs[i] is not None for i in elig_ids)

    # Round 0: batch all spec-eligible streams in ONE mtp.step_rows call
    e_toks = mx.array([cur_toks[i] for i in elig_ids], dtype=mx.uint32)
    e_hs = cur_hidden[mx.array(elig_ids, dtype=mx.int32)]
    e_caches = [mtp_caches[i] for i in elig_ids]
    c_arg0 = e_caches[0] if len(elig_ids) == 1 else e_caches
    if any_sampled:
        h_d0, dl0 = mtp.step_rows(e_toks, e_hs, c_arg0, C=len(elig_ids), L=1, use_q4_head=use_q4_head)
        dl0_lp = dl0 - mx.logsumexp(dl0, axis=-1, keepdims=True)
        d0_list_arr = []
        p0_list_arr = []
        q0_rows: dict[int, mx.array] = {}
        for idx, s in enumerate(elig_ids):
            sp = specs[s]
            if sp is not None:
                q_row = _logprobs_to_probs(dl0_lp[idx], sp)
                q0_rows[idx] = q_row
                d0_list_arr.append(_sample_from_probs(q_row).reshape(1))
                # Deterministic score from q_row only (pre-draw), never conditioned on drawn token!
                p0_list_arr.append(mx.maximum(mx.max(q_row), mx.array(0.60, dtype=mx.float32)).reshape(1))
            else:
                tid, tprob = _step_top1_and_prob(dl0[idx : idx + 1])
                d0_list_arr.append(tid)
                p0_list_arr.append(tprob)
        d0_arr = mx.concatenate(d0_list_arr, axis=0)
        p0_arr = mx.concatenate(p0_list_arr, axis=0)
    else:
        q0_rows = {}
        h_d0, d0_arr, p0_arr = _mtp_step_top1(
            mtp, e_toks, e_hs, c_arg0, C=len(elig_ids), use_q4_head=use_q4_head
        )
    for i in elig_ids:
        after0_offs[i] = mtp_caches[i].offset

    r_inds0 = getattr(mtp, "_last_router_inds", None)
    ev0 = [d0_arr, p0_arr]
    if r_inds0 is not None:
        ev0.append(r_inds0)
    mx.eval(*ev0)

    d0_vals = d0_arr.tolist()
    p0_vals = p0_arr.tolist()

    seen_mask = 0
    seen_toks = set(cur_toks)
    slots_used = 0
    order0 = sorted(range(len(elig_ids)), key=lambda idx: -p0_vals[idx])
    active_state: dict[int, tuple[int, float, mx.array, mx.array]] = {}

    for idx in order0:
        if slots_used >= k_budget:
            break
        s = elig_ids[idx]
        p0 = float(p0_vals[idx])
        d0 = int(d0_vals[idx])
        du72 = None
        if r_inds0 is not None:
            du72, seen_mask = _update_router_mask(mtp, idx, seen_mask)
        B_next = C + slots_used + 1
        _, tau_d = _thresholds_for_step(
            B_next, du72, d0 in seen_toks, dyn_th=dyn_th, r_star=r_eff, t_draft=t_draft, e_p_next=e_p_next, use_rr=use_rr
        )
        if specs[s] is None and p0 < p0_drop:
            continue

        # MTP-confirmed suffix shortcut for greedy streams only
        if specs[s] is None and suffix_matchers is not None and suffix_matchers[s] is not None and k_limits[s] >= 2:
            rem_slots = min(k_limits[s], k_budget - slots_used)
            if rem_slots >= 2:
                eff_mg = suffix_matchers[s].effective_min_gram(_suffix_min_gram())
                gl, stoks = suffix_matchers[s].match(
                    rem_slots, min_gram=eff_mg, first_token=d0
                )
                if gl >= eff_mg and len(stoks) >= 2:
                    suffix_matchers[s]._last_tail_len = len(stoks) - 1
                    draft_lists[s].extend(stoks)
                    seen_toks.update(stoks)
                    slots_used += len(stoks)
                    continue
                suffix_matchers[s]._last_tail_len = 0

        draft_lists[s].append(d0)
        if specs[s] is not None:
            draft_q_lists[s].append(q0_rows[idx])
        seen_toks.add(d0)
        slots_used += 1
        if len(draft_lists[s]) < k_limits[s] and (specs[s] is not None or p0 >= tau_d):
            active_state[s] = (d0, p0, e_hs[idx : idx + 1], h_d0[idx : idx + 1])

    # Subsequent rounds: step only streams still wanting drafts while slots remain
    max_rounds = min(6, k_budget)
    for _ in range(1, max_rounds):
        if slots_used >= k_budget or not active_state:
            break
        rem_slots = k_budget - slots_used
        ranked_s = sorted(active_state.keys(), key=lambda s: -active_state[s][1])[:rem_slots]
        n_step = len(ranked_s)
        tok_in = mx.array([active_state[s][0] for s in ranked_s], dtype=mx.uint32)
        h_in_cat = mx.concatenate([active_state[s][2] for s in ranked_s], axis=0)
        h_d_cat = mx.concatenate([active_state[s][3] for s in ranked_s], axis=0)
        h_br = mtp.apply_feedback_bridge(h_d_cat, h_in_cat)
        sub_caches = [mtp_caches[s] for s in ranked_s]
        c_arg = sub_caches[0] if n_step == 1 else sub_caches
        step_has_sampled = any(specs[s] is not None for s in ranked_s)
        qn_rows: dict[int, mx.array] = {}
        if step_has_sampled:
            h_dn, dln = mtp.step_rows(tok_in, h_br, c_arg, C=n_step, L=1, use_q4_head=use_q4_head)
            dln_lp = dln - mx.logsumexp(dln, axis=-1, keepdims=True)
            dn_list_arr = []
            pn_list_arr = []
            for idx, s in enumerate(ranked_s):
                sp = specs[s]
                if sp is not None:
                    q_row = _logprobs_to_probs(dln_lp[idx], sp)
                    qn_rows[idx] = q_row
                    dn_list_arr.append(_sample_from_probs(q_row).reshape(1))
                    pn_list_arr.append(mx.maximum(mx.max(q_row), mx.array(0.55, dtype=mx.float32)).reshape(1))
                else:
                    tid, tprob = _step_top1_and_prob(dln[idx : idx + 1])
                    dn_list_arr.append(tid)
                    pn_list_arr.append(tprob)
            dn_arr = mx.concatenate(dn_list_arr, axis=0)
            pn_arr = mx.concatenate(pn_list_arr, axis=0)
        else:
            h_dn, dn_arr, pn_arr = _mtp_step_top1(
                mtp, tok_in, h_br, c_arg, C=n_step, use_q4_head=use_q4_head
            )
        r_indsn = getattr(mtp, "_last_router_inds", None)
        evn = [dn_arr, pn_arr]
        if r_indsn is not None:
            evn.append(r_indsn)
        mx.eval(*evn)

        dn_vals = dn_arr.tolist()
        pn_vals = pn_arr.tolist()
        next_active: dict[int, tuple[int, float, mx.array, mx.array]] = {}

        cands = []
        for idx, s in enumerate(ranked_s):
            cq = active_state[s][1] * float(pn_vals[idx])
            cands.append((cq, idx, s, int(dn_vals[idx])))
        cands.sort(key=lambda x: -x[0])

        for cq, idx, s, dn in cands:
            if slots_used >= k_budget:
                break
            du72 = None
            if r_indsn is not None:
                du72, seen_mask = _update_router_mask(mtp, idx, seen_mask)
            B_next = C + slots_used + 1
            tau_v, tau_d = _thresholds_for_step(
                B_next, du72, dn in seen_toks, dyn_th=dyn_th, r_star=r_eff, t_draft=t_draft, e_p_next=e_p_next, use_rr=use_rr
            )
            # Never pop a sampled token after drawing (preserves 100% exact rejection sampling!)
            if specs[s] is None and cq < tau_v:
                continue
            draft_lists[s].append(dn)
            if specs[s] is not None:
                draft_q_lists[s].append(qn_rows[idx])
            seen_toks.add(dn)
            slots_used += 1
            if len(draft_lists[s]) < k_limits[s] and (specs[s] is not None or cq >= tau_d):
                next_active[s] = (dn, cq, h_br[idx : idx + 1], h_dn[idx : idx + 1])

        active_state = next_active

    return draft_lists, after0_offs, base_offs, draft_q_lists


def _sampled_k_for_batch(C: int, mtp: BailingMTPLayer, K_limit: int | None) -> int:
    if C <= 1:
        k_def = int(os.environ.get("LINGSANTE_SAMPLED_K1", "4"))
    else:
        k_def = mtp_k_for_batch(C, mtp)
    if K_limit is not None:
        k_def = min(k_def, max(0, int(K_limit)))
    return max(0, k_def)


def _chain_mtp_drafts_mixed(
    mtp: BailingMTPLayer,
    first_toks: mx.array,
    first_hidden: mx.array,
    mtp_caches: list[KVCache],
    *,
    sampling_specs: list[tuple[float, list[Callable]] | None],
    C: int,
    K_limit: int | None = None,
    use_q4_head: bool = True,
) -> tuple[mx.array, list[int], list[list[mx.array] | None]]:
    """Zero-sync lazy GPU draft chaining for pure `temp > 0` or mixed greedy + `temp > 0` batches."""
    K_cap = _sampled_k_for_batch(C, mtp, K_limit)
    cache_arg = mtp_caches[0] if C == 1 else mtp_caches
    h_d, dl0 = mtp.step_rows(first_toks, first_hidden, cache_arg, C=C, L=1, use_q4_head=use_q4_head)
    exact_offs = [mc.offset for mc in mtp_caches]
    q_lists: list[list[mx.array] | None] = [[] if sp is not None else None for sp in sampling_specs]
    if K_cap <= 0:
        return mx.zeros((C, 0), dtype=mx.uint32), exact_offs, q_lists

    cols = []
    h_in = first_hidden
    dl_cur = dl0
    for k in range(K_cap):
        dl_lp = dl_cur - mx.logsumexp(dl_cur, axis=-1, keepdims=True)
        step_toks = []
        for s in range(C):
            sp = sampling_specs[s]
            if sp is not None:
                qk = _logprobs_to_probs(dl_lp[s], sp)
                q_lists[s].append(qk)
                step_toks.append(_sample_from_probs(qk).reshape(1))
            else:
                step_toks.append(mx.argmax(dl_cur[s : s + 1], axis=-1).astype(mx.uint32))
        d_prev = mx.concatenate(step_toks, axis=0)
        cols.append(d_prev)
        if k + 1 < K_cap:
            h_in_next = mtp.apply_feedback_bridge(h_d, h_in)
            h_d, dl_cur = mtp.step_rows(d_prev, h_in_next, cache_arg, C=C, L=1, use_q4_head=use_q4_head)
            h_in = h_in_next

    return mx.stack(cols, axis=1), exact_offs, q_lists


def _sync_and_chain_mtp_mixed(
    mtp: BailingMTPLayer,
    h_eff_list: list[mx.array],
    draft_lists: list[list[int]],
    cur_toks: list[int],
    ncs: list[int],
    mtp_caches: list[KVCache],
    mtp_exact_offs: list[int],
    *,
    sampling_specs: list[tuple[float, list[Callable]] | None],
    C: int,
    K_limit: int | None = None,
    use_q4_head: bool = True,
) -> tuple[mx.array, list[int], list[list[mx.array] | None]]:
    """Refresh accepted MTP tokens and run zero-sync lazy mixed greedy/`temp > 0` draft chaining."""
    _refresh_mtp_accepted_list(mtp, mtp_caches, mtp_exact_offs, draft_lists, h_eff_list, ncs)
    cur_arr = mx.array(cur_toks, dtype=mx.uint32)
    cur_hs = mx.stack([h_eff_list[c][ncs[c] - 1, :] for c in range(C)], axis=0)
    return _chain_mtp_drafts_mixed(
        mtp,
        cur_arr,
        cur_hs,
        mtp_caches,
        sampling_specs=sampling_specs,
        C=C,
        K_limit=K_limit,
        use_q4_head=use_q4_head,
    )


def _chain_mtp_drafts(
    mtp: BailingMTPLayer,
    first_toks: mx.array,
    first_hidden: mx.array,
    mtp_caches: list[KVCache],
    *,
    C: int,
    K: int | None = None,
    K_limit: int | None = None,
    use_q4_head: bool,
    suffix_matcher: SuffixMatcher | None = None,
    anchor_tok_int: int | None = None,
) -> tuple[mx.array, list[int]]:
    """Build MTP draft steps on GPU (`dyn_step` at C=1/2 or fixed lazy chaining at C>=3)."""
    draft_mode, k_cap, dyn_th, p1_min = _resolve_draft_policy(mtp, C, K, K_limit)
    cache_arg = mtp_caches[0] if C == 1 else mtp_caches
    if _has_d1_mtp(mtp):
        h_d, d0_id, p0_f = mtp.step_rows(
            first_toks, first_hidden, cache_arg, C=C, L=1, use_q4_head=use_q4_head, return_argmax_prob=True
        )
        dl0 = (d0_id, p0_f)
    else:
        h_d, dl0 = mtp.step_rows(first_toks, first_hidden, cache_arg, C=C, L=1, use_q4_head=use_q4_head)
    exact_offs = [mc.offset for mc in mtp_caches]
    draft_mat = _draft_steps_from_logits(
        mtp,
        cache_arg,
        C=C,
        K_cap=k_cap,
        dl0=dl0,
        h0_in=first_hidden,
        h0_out=h_d,
        draft_mode=draft_mode,
        dyn_th=dyn_th,
        p1_min=p1_min,
        use_q4_head=use_q4_head,
        suffix_matcher=suffix_matcher,
        anchor_tok_int=anchor_tok_int,
    )
    return draft_mat, exact_offs


def _sync_and_chain_mtp_list(
    mtp: BailingMTPLayer,
    h_eff_list: list[mx.array],
    draft_lists: list[list[int]],
    cur_toks: list[int],
    ncs: list[int],
    mtp_caches: list[KVCache],
    mtp_exact_offs: list[int],
    *,
    C: int,
    K: int | None = None,
    K_limit: int | None = None,
    use_q4_head: bool,
    suffix_matcher: SuffixMatcher | None = None,
) -> tuple[mx.array, list[int]]:
    """Trim unaccepted MTP KV entries, KV-append accepted drafts `[0..nc-2]`, step `cur_toks`, and chain drafts."""
    draft_mode, k_cap, dyn_th, p1_min = _resolve_draft_policy(mtp, C, K, K_limit)
    cur_arr = mx.array(cur_toks, dtype=mx.uint32)
    cur_hs = mx.stack([h_eff_list[c][ncs[c] - 1, :] for c in range(C)], axis=0)
    cache_arg = mtp_caches[0] if C == 1 else mtp_caches

    if _has_d1_mtp(mtp):
        for mc, aoff in zip(mtp_caches, mtp_exact_offs):
            tr = mc.offset - aoff
            if tr > 0:
                mc.trim(tr)
        ls_acc = [nc - 1 for nc in ncs]
        if any(lc > 0 for lc in ls_acc):
            acc_toks = mx.array(
                [tok for ci, lc in enumerate(ls_acc) if lc > 0 for tok in draft_lists[ci][:lc]],
                dtype=mx.uint32,
            )
            acc_hs = mx.concatenate(
                [h_eff_list[ci][:lc, :] for ci, lc in enumerate(ls_acc) if lc > 0], axis=0
            )
        else:
            acc_toks = mx.zeros((0,), dtype=mx.uint32)
            acc_hs = mx.zeros((0, cur_hs.shape[-1]), dtype=cur_hs.dtype)
        h_d, d0_id, p0_f = mtp.refresh_and_step0(
            acc_toks,
            acc_hs,
            cur_arr,
            cur_hs,
            cache_arg,
            ls_acc,
            use_q4_head=use_q4_head,
            return_argmax_prob=True,
        )
        dl0 = (d0_id, p0_f)
    else:
        _refresh_mtp_accepted_list(mtp, mtp_caches, mtp_exact_offs, draft_lists, h_eff_list, ncs)
        h_d, dl0 = mtp.step_rows(cur_arr, cur_hs, cache_arg, C=C, L=1, use_q4_head=use_q4_head)

    new_exact_offs = [mc.offset for mc in mtp_caches]
    draft_mat = _draft_steps_from_logits(
        mtp,
        cache_arg,
        C=C,
        K_cap=k_cap,
        dl0=dl0,
        h0_in=cur_hs,
        h0_out=h_d,
        draft_mode=draft_mode,
        dyn_th=dyn_th,
        p1_min=p1_min,
        use_q4_head=use_q4_head,
        suffix_matcher=suffix_matcher,
        anchor_tok_int=cur_toks[0] if C == 1 else None,
    )
    return draft_mat, new_exact_offs


def _sync_and_chain_mtp(
    mtp: BailingMTPLayer,
    hidden: mx.array,
    draft_lists: list[list[int]],
    cur_toks: list[int],
    ncs: list[int],
    mtp_caches: list[KVCache],
    mtp_exact_offs: list[int],
    *,
    C: int,
    K: int | None = None,
    K_limit: int | None = None,
    use_q4_head: bool,
) -> tuple[mx.array, list[int]]:
    """Backward-compatible wrapper around `_sync_and_chain_mtp_list`."""
    h_list = [hidden[c, : ncs[c], :] for c in range(C)]
    return _sync_and_chain_mtp_list(
        mtp,
        h_list,
        draft_lists,
        cur_toks,
        ncs,
        mtp_caches,
        mtp_exact_offs,
        C=C,
        K=K,
        K_limit=K_limit,
        use_q4_head=use_q4_head,
    )


def _logits_to_preds(logits: mx.array):
    lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    return mx.argmax(lp, axis=-1), lp


def _contract_verify_cache(model, vcache: list, keep: list[int]):
    """Shrink active slots in `vcache` to indices in `keep` after one or more streams finish."""
    C_new = len(keep)
    keep_arr = mx.array(keep, dtype=mx.int32)
    for li, layer in enumerate(model.model.layers):
        c = vcache[li]
        if layer.is_linear:
            ac = ArraysCache(4 if C_new == 1 else 3 + C_new)
            for j in range(3):
                ac[j] = c[j][keep_arr]
            if C_new == 1:
                s0 = keep[0]
                ac[3] = c[3 + s0] if len(c.cache) > 4 else c[3][keep_arr]
            else:
                if len(c.cache) > 4:
                    for ni, oi in enumerate(keep):
                        ac[3 + ni] = c[3 + oi]
                else:
                    for ni, oi in enumerate(keep):
                        ac[3 + ni] = c[3][oi : oi + 1]
            vcache[li] = ac
        else:
            vcache[li] = c[keep[0]] if C_new == 1 else [c[s] for s in keep]


def run_mtp_spec_greedy(
    model,
    mtp: BailingMTPLayer,
    prompts_ids: Sequence[Sequence[int]],
    *,
    K: int | None = None,
    max_tokens: int = 128,
    use_q4_head: bool = True,
    prefill_step: int = 512,
    capture_lp_steps: int = 32,
):
    """Single-sync async-pipelined greedy speculative decode for `1 <= C <= 4` streams with contraction."""
    C = len(prompts_ids)
    assert 1 <= C <= 4
    lids = getattr(mtp, "_feat_layers", None)

    t_caches, mtp_caches, last_toks = [], [], []
    for ids in prompts_ids:
        tc, mc, lt = prefill_with_hidden(model, mtp, ids, step=prefill_step)
        t_caches.append(tc)
        mtp_caches.append(mc)
        last_toks.append(lt)

    vcache = make_verify_cache(model, t_caches)

    # Initial L=1 target forward on last prompt token
    logits0, hidden0, rb0 = fast_verify_forward(
        model,
        mx.array([[lt] for lt in last_toks], dtype=mx.uint32),
        vcache,
        return_hidden=True,
        return_rollback=True,
        capture_layers=lids,
    )
    hidden0 = mtp.apply_feat_fuse(hidden0, rb0.get("captured"))
    nxt0, lp0 = _logits_to_preds(logits0[:, 0, :])
    mx.eval(nxt0, hidden0, lp0)
    commit_verify_cache(vcache, rb0, 1)

    cur_toks = nxt0.tolist()
    emitted = [[cur_toks[c]] for c in range(C)]
    lp_arrays: list[list[mx.array]] = [[lp0[c]] if capture_lp_steps > 0 else [] for c in range(C)]
    suffix_matchers = [
        SuffixMatcher(list(prompts_ids[c]) + [cur_toks[c]], min_gram=_suffix_min_gram(), max_gram=8)
        if _use_mtp_suffix()
        else None
        for c in range(C)
    ]

    # Build + launch cycle 1 asynchronously (two-part overlap: submit draft before building 72-layer verify)
    draft_mat, mtp_exact_offs = _chain_mtp_drafts(
        mtp,
        nxt0,
        hidden0[:, 0, :],
        mtp_caches,
        C=C,
        K=K,
        use_q4_head=use_q4_head,
        suffix_matcher=suffix_matchers[0] if C == 1 else None,
        anchor_tok_int=cur_toks[0] if C == 1 else None,
    )
    K_cur = int(draft_mat.shape[1])
    if K_cur > 0:
        mx.async_eval(draft_mat)
    inp = mx.concatenate([nxt0[:, None], draft_mat], axis=1) if K_cur > 0 else nxt0[:, None]
    logits, hidden, rb = fast_verify_forward(
        model, inp, vcache, return_hidden=True, return_rollback=True, capture_layers=lids
    )
    preds, lp = _logits_to_preds(logits)

    mx.reset_peak_memory()
    t_start = time.perf_counter()
    mx.async_eval(*_verify_eval_targets(preds, hidden, draft_mat if K_cur > 0 else None, lp, rb))

    C_act = C
    active_orig = list(range(C))
    n_verify_steps = 0
    active_stream_steps = 0
    sum_emitted = 0
    steady_emitted = 0
    steady_wall_s = None
    max_pos = K if K is not None else 7
    hits = [0] * max_pos
    trials = [0] * max_pos

    while True:
        mx.eval(*_verify_eval_targets(preds, hidden, draft_mat if K_cur > 0 else None, lp, rb))
        pred_lists = preds.tolist()
        draft_lists = draft_mat.tolist() if K_cur > 0 else [[] for _ in range(C_act)]

        ncs = []
        next_cur_toks = []
        pending_tokens = []
        pending_lps = []
        keep = []
        for s in range(C_act):
            orig = active_orig[s]
            a = 0
            for j in range(K_cur):
                if j < max_pos:
                    trials[j] += 1
                if pred_lists[s][j] == draft_lists[s][j]:
                    if j < max_pos:
                        hits[j] += 1
                    a += 1
                else:
                    break
            rem = max_tokens - len(emitted[orig])
            n_take = min(a + 1, rem)
            pending_tokens.append(pred_lists[s][:n_take])
            base_pos = len(emitted[orig])
            pending_lps.append(
                [lp[s, j] for j in range(n_take) if base_pos + j < capture_lp_steps]
            )
            sum_emitted += n_take
            if C_act == C:
                steady_emitted += n_take
            ncs.append(a + 1)
            next_cur_toks.append(pred_lists[s][a])
            if suffix_matchers[orig] is not None:
                suffix_matchers[orig].extend(pred_lists[s][: n_take])
            if base_pos + n_take < max_tokens:
                keep.append(s)

        n_verify_steps += 1
        active_stream_steps += C_act
        if C_act == C and len(keep) < C and steady_wall_s is None and keep:
            steady_wall_s = time.perf_counter() - t_start

        if not keep:
            for s, orig in enumerate(active_orig):
                emitted[orig].extend(pending_tokens[s])
                lp_arrays[orig].extend(pending_lps[s])
            commit_verify_cache(vcache, rb, ncs)
            mx.synchronize()
            if steady_wall_s is None:
                steady_wall_s = time.perf_counter() - t_start
            break

        # Slice FeatFuse strictly to consumed rows [:nc]
        h_eff_list = _apply_feat_fuse_consumed(mtp, hidden, rb.get("captured"), ncs)

        # Commit current active batch, then contract finished streams before MTP/next verify
        commit_verify_cache(vcache, rb, ncs)
        old_active_orig = active_orig
        if len(keep) < C_act:
            _contract_verify_cache(model, vcache, keep)
            for s in range(C_act):
                tr = mtp_caches[s].offset - mtp_exact_offs[s]
                if tr > 0:
                    mtp_caches[s].trim(tr)
            mtp_caches = [mtp_caches[s] for s in keep]
            mtp_exact_offs = [mtp_exact_offs[s] for s in keep]
            draft_lists = [draft_lists[s] for s in keep]
            next_cur_toks = [next_cur_toks[s] for s in keep]
            ncs = [ncs[s] for s in keep]
            h_eff_list = [h_eff_list[s] for s in keep]
            active_orig = [active_orig[s] for s in keep]
            C_act = len(keep)

        draft_mat_next, mtp_exact_offs = _sync_and_chain_mtp_list(
            mtp,
            h_eff_list,
            draft_lists,
            next_cur_toks,
            ncs,
            mtp_caches,
            mtp_exact_offs,
            C=C_act,
            K=K,
            use_q4_head=use_q4_head,
            suffix_matcher=suffix_matchers[active_orig[0]] if C_act == 1 else None,
        )
        K_next = int(draft_mat_next.shape[1])
        if K_next > 0:
            mx.async_eval(draft_mat_next)

        cur_arr = mx.array(next_cur_toks, dtype=mx.uint32)
        inp_next = (
            mx.concatenate([cur_arr[:, None], draft_mat_next], axis=1)
            if K_next > 0
            else cur_arr[:, None]
        )
        logits_next, hidden_next, rb_next = fast_verify_forward(
            model, inp_next, vcache, return_hidden=True, return_rollback=True, capture_layers=lids
        )
        preds_next, lp_next = _logits_to_preds(logits_next)
        mx.async_eval(
            *_verify_eval_targets(
                preds_next, hidden_next, draft_mat_next if K_next > 0 else None, lp_next, rb_next
            )
        )

        # CPU work overlapped with GPU execution of cycle s+1
        for s, orig in enumerate(old_active_orig):
            emitted[orig].extend(pending_tokens[s])
            lp_arrays[orig].extend(pending_lps[s])

        preds, hidden, draft_mat, lp, rb, K_cur = (
            preds_next,
            hidden_next,
            draft_mat_next,
            lp_next,
            rb_next,
            K_next,
        )

    wall_dt = time.perf_counter() - t_start
    gen_toks = sum(len(e) - 1 for e in emitted)
    steady_tps = steady_emitted / steady_wall_s if steady_wall_s and steady_wall_s > 0 else gen_toks / wall_dt
    return {
        "C": C,
        "K": K,
        "q4_head": use_q4_head,
        "emitted": emitted,
        "lp_arrays": lp_arrays,
        "steps": n_verify_steps,
        "gen_toks": gen_toks,
        "wall_s": wall_dt,
        "agg_tok_s": gen_toks / wall_dt,
        "steady_tok_s": steady_tps,
        "per_stream_tok_s": (gen_toks / wall_dt) / C,
        "E_acc": sum_emitted / active_stream_steps,
        "alphas": [round(hits[j] / trials[j], 3) if trials[j] > 0 else 0.0 for j in range(max_pos)],
        "peak_gb": round(mx.get_peak_memory() / 1e9, 2),
    }
