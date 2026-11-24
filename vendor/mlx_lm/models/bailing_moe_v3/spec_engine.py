"""Single-sync GPU-chained speculative decoding engine for LingSante (BailingMoeV3 + MTP, C=1..4)."""
import copy
import os
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import mlx.core as mx

from mlx_lm.models.cache import ArraysCache, BatchKVCache, KVCache, make_prompt_cache

from .mtp import BailingMTPLayer
from .verify_decode import can_fast_verify, commit_verify_cache, fast_verify_forward, make_verify_cache


class _MTPBox:
    """Plain Python container so attaching MTP does not register it as an nn.Module child."""

    __slots__ = ("mtp",)

    def __init__(self, mtp: BailingMTPLayer):
        self.mtp = mtp


def attach_lingsante_mtp(model, model_path: str | Path) -> bool:
    """Attach native Layer-72 MTP drafter onto a matching LingSante model if available."""
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


def mtp_k_for_batch(C: int) -> int:
    if C <= 1:
        return max(1, int(os.environ.get("LINGSANTE_MTP_K1", "4")))
    return max(1, int(os.environ.get("LINGSANTE_MTP_KB", "2")))


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
            mtp_out = mtp.prefill(m_toks[:, pos : pos + m], m_hs[:, pos : pos + m, :], mc)
            mx.eval(mtp_out, mc.state)
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


def mtp_vcache_to_standard(model, vcache: list, C: int) -> list:
    """Convert split `vcache` back to standard `[ArraysCache(4) | BatchKVCache/KVCache]`."""
    per = [mtp_extract_from_vcache(model, vcache, i, C, copy_kv=False) for i in range(C)]
    if C == 1:
        return per[0]
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
    """Roll back an unconsumed speculative verify cycle with zero KDA mutation."""
    L = inflight["rb"]["L"]
    for mc in inflight["rb"]["mla"]:
        if isinstance(mc, list):
            for ci in mc:
                ci.trim(L)
        else:
            mc.trim(L)
    for mc, base_off in zip(mtp_caches, inflight["mtp_base_offs"]):
        tr = mc.offset - base_off
        if tr > 0:
            mc.trim(tr)


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
        mtp.step_rows(
            toks,
            hidden[ci, :Lc, :],
            mtp_caches[ci],
            C=1,
            L=Lc,
            use_q4_head=use_q4_head,
            return_logits=False,
        )
        return
    toks_flat = mx.array(
        [tok for ci, Lc in need for tok in draft_lists[ci][:Lc]],
        dtype=mx.uint32,
    )
    hs_flat = mx.concatenate([hidden[ci, :Lc, :] for ci, Lc in need], axis=0)
    mtp.step_varlen_rows(
        toks_flat,
        hs_flat,
        [mtp_caches[ci] for ci, _ in need],
        [Lc for _, Lc in need],
        use_q4_head=use_q4_head,
        return_logits=False,
    )


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
):
    """Bit-exact greedy `generate_step` generator accelerated by LingSante native MTP."""
    if len(prompt) == 0:
        raise ValueError("Prompt must be non-empty.")
    if prompt_cache is None:
        prompt_cache = make_prompt_cache(model)
    prompt_progress_callback = prompt_progress_callback or (lambda *_: None)
    if max_tokens == 0:
        return

    with mx.stream(stream):
        total_prompt_tokens = len(prompt)
        prompt_processed_tokens = 0
        prompt_progress_callback(0, total_prompt_tokens)

        mtp_state = getattr(prompt_cache[-1], "_lingsante_mtp_state", None) if prompt_cache else None
        if prompt_cache and hasattr(prompt_cache[-1], "_lingsante_mtp_state"):
            delattr(prompt_cache[-1], "_lingsante_mtp_state")

        while total_prompt_tokens - prompt_processed_tokens > 1:
            remaining = (total_prompt_tokens - prompt_processed_tokens) - 1
            n_to_process = min(prefill_step_size, remaining)
            processed = prompt[:n_to_process]
            h = model.model(processed[None], cache=prompt_cache)
            mx.eval(h, [c.state for c in prompt_cache])
            mtp_state = mtp_prefill_chunk(mtp, processed[None], h, mtp_state)
            prompt_processed_tokens += n_to_process
            prompt_progress_callback(prompt_processed_tokens, total_prompt_tokens)
            prompt = prompt[n_to_process:]
            mx.clear_cache()

        last_tok = prompt
        if can_fast_verify(model, last_tok[None], prompt_cache):
            logits0, hidden0, rb0 = fast_verify_forward(
                model, last_tok[None], prompt_cache, return_hidden=True, return_rollback=True
            )
            commit_verify_cache(prompt_cache, rb0, 1)
        else:
            hidden0 = model.model(last_tok[None], cache=prompt_cache)
            logits0 = model.lm_head(hidden0)

        y0, lp0_2d = _logits_to_preds(logits0[:, -1, :])
        lp0 = lp0_2d.squeeze(0)
        mx.eval(y0, hidden0, lp0)

        mc, pending_h = mtp_state if mtp_state is not None else (KVCache(), None)
        if pending_h is not None:
            mtp.step_rows(
                last_tok,
                pending_h[:, 0, :],
                mc,
                C=1,
                L=1,
                use_q4_head=True,
                return_logits=False,
            )
            mx.eval(mc.state)

        prompt_progress_callback(total_prompt_tokens, total_prompt_tokens)
        n = 0
        yield y0.item(), lp0
        n += 1
        if n == max_tokens:
            if prompt_cache:
                prompt_cache[-1]._lingsante_mtp_state = (mc, hidden0[:, -1:, :])
            return

        K_base = mtp_k_for_batch(1)
        cur_y = y0
        cur_h = hidden0[:, -1, :]
        K = K_base if max_tokens < 0 else min(K_base, max_tokens - n)
        draft_mat, mtp_exact_offs = _chain_mtp_drafts(
            mtp, cur_y, cur_h, [mc], C=1, K=K, use_q4_head=True
        )
        mx.async_eval(draft_mat)
        inp = mx.concatenate([cur_y[:, None], draft_mat], axis=1)
        logits, hidden, rb = fast_verify_forward(
            model, inp, prompt_cache, return_hidden=True, return_rollback=True
        )
        preds, lp = _logits_to_preds(logits)
        mx.async_eval(preds, hidden, draft_mat, lp)

        inflight = {"rb": rb, "mtp_base_offs": [mtp_exact_offs[0] - 1]}
        cur_burst_rb = None
        cur_burst_emitted = 0
        cur_burst_after0 = mtp_exact_offs[0]
        cur_burst_drafts = None
        cur_burst_hidden = None

        try:
            while True:
                mx.eval(preds, hidden, draft_mat, lp)
                inflight = None
                pred_row = preds[0].tolist()
                draft_row = draft_mat[0].tolist()

                a = 0
                for j in range(K):
                    if pred_row[j] == draft_row[j]:
                        a += 1
                    else:
                        break

                n_take = a + 1
                if max_tokens > 0:
                    n_take = min(n_take, max_tokens - n)

                cur_burst_rb = rb
                cur_burst_emitted = 0
                cur_burst_after0 = mtp_exact_offs[0]
                cur_burst_drafts = draft_row
                cur_burst_hidden = hidden

                # Yield any already-verified draft tokens [0 .. n_take-2]
                for j in range(n_take - 1):
                    cur_burst_emitted = j + 1
                    n += 1
                    yield pred_row[j], lp[0, j]
                    if n % 256 == 0:
                        mx.clear_cache()

                # Last token of this burst (either accepted limit or correction pred_row[a])
                last_j = n_take - 1
                more_needed = max_tokens < 0 or (n + 1) < max_tokens
                if not more_needed:
                    commit_verify_cache(prompt_cache, rb, n_take if n_take <= a else (a + 1))
                    cur_burst_rb = None
                    _refresh_mtp_accepted(
                        mtp,
                        [mc],
                        [mtp_exact_offs[0]],
                        [draft_row],
                        hidden,
                        [n_take if n_take <= a else (a + 1)],
                        use_q4_head=True,
                    )
                    last_h = hidden[:, last_j : last_j + 1, :]
                    if prompt_cache:
                        prompt_cache[-1]._lingsante_mtp_state = (mc, last_h)
                    mx.eval(last_h, mc.state)
                    n += 1
                    yield pred_row[last_j], lp[0, last_j]
                    return

                # Commit + launch next cycle asynchronously before yielding pred_row[a]
                nc = a + 1
                commit_verify_cache(prompt_cache, rb, nc)
                cur_burst_rb = None
                next_cur = pred_row[a]
                K_next = K_base if max_tokens < 0 else min(K_base, max_tokens - (n + 1))
                draft_mat_next, mtp_exact_offs = _sync_and_chain_mtp(
                    mtp,
                    hidden,
                    [draft_row],
                    [next_cur],
                    [nc],
                    [mc],
                    [mtp_exact_offs[0]],
                    C=1,
                    K=K_next,
                    use_q4_head=True,
                )
                flush_arrs = [draft_mat_next]
                for li, layer in enumerate(model.model.layers):
                    if layer.is_linear:
                        flush_arrs.extend(x for x in prompt_cache[li].cache if x is not None)
                mx.async_eval(*flush_arrs)

                cur_arr = mx.array([next_cur], dtype=mx.uint32)
                inp_next = mx.concatenate([cur_arr[:, None], draft_mat_next], axis=1)
                logits_next, hidden_next, rb_next = fast_verify_forward(
                    model, inp_next, prompt_cache, return_hidden=True, return_rollback=True
                )
                preds_next, lp_next = _logits_to_preds(logits_next)
                mx.async_eval(preds_next, hidden_next, draft_mat_next, lp_next)

                inflight = {
                    "rb": rb_next,
                    "mtp_base_offs": [mtp_exact_offs[0] - 1],
                    "mtp_after0_offs": list(mtp_exact_offs),
                    "hidden_next": hidden_next,
                }
                n += 1
                yield next_cur, lp[0, a]
                if n % 256 == 0:
                    mx.clear_cache()

                preds, hidden, draft_mat, lp, rb, K = (
                    preds_next,
                    hidden_next,
                    draft_mat_next,
                    lp_next,
                    rb_next,
                    K_next,
                )
        finally:
            if inflight is not None:
                if cur_burst_rb is None and "hidden_next" in inflight:
                    # Caller broke on last token of burst (next_cur, which was inp_next[0]): commit nc=1.
                    mx.eval(inflight["hidden_next"])
                    commit_verify_cache(prompt_cache, inflight["rb"], 1)
                    a0 = inflight["mtp_after0_offs"][0]
                    if mc.offset > a0:
                        mc.trim(mc.offset - a0)
                    if prompt_cache:
                        prompt_cache[-1]._lingsante_mtp_state = (mc, inflight["hidden_next"][:, 0:1, :])
                else:
                    mtp_cancel_inflight(inflight, [mc])
                mx.synchronize(stream)
            if cur_burst_rb is not None:
                # Generator exited early inside a burst after cur_burst_emitted tokens.
                nc_fin = cur_burst_emitted + 1
                commit_verify_cache(prompt_cache, cur_burst_rb, nc_fin)
                _refresh_mtp_accepted(
                    mtp,
                    [mc],
                    [cur_burst_after0],
                    [cur_burst_drafts],
                    cur_burst_hidden,
                    [nc_fin],
                    use_q4_head=True,
                )
                if prompt_cache:
                    prompt_cache[-1]._lingsante_mtp_state = (
                        mc,
                        cur_burst_hidden[:, nc_fin - 1 : nc_fin, :],
                    )
                mx.synchronize(stream)


class LingSanteSpecBatch:
    """Native MTP speculative decoding backend for `BatchGenerator`/`GenerationBatch` (`1 <= C <= 4`)."""

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
        if logits_processors and any(logits_processors):
            return False
        eff = [fallback_sampler if s is None else s for s in (samplers or [None] * len(uids))]
        return all(is_greedy_sampler(s) for s in eff)

    @classmethod
    def init_from_prompt(
        cls,
        model,
        mtp: BailingMTPLayer,
        inputs: mx.array,
        prompt_cache: list,
        mtp_states: list | None = None,
    ) -> "LingSanteSpecBatch":
        C = int(inputs.shape[0])
        if prompt_cache[0].empty():
            hidden0 = model.model(inputs[:, None], cache=prompt_cache)
            logits0 = model.lm_head(hidden0)
            vcache = mtp_vcache_from_standard(model, prompt_cache, C)
        else:
            vcache = mtp_vcache_from_standard(model, prompt_cache, C)
            if can_fast_verify(model, inputs[:, None], vcache):
                logits0, hidden0, rb0 = fast_verify_forward(
                    model, inputs[:, None], vcache, return_hidden=True, return_rollback=True
                )
                commit_verify_cache(vcache, rb0, 1)
            else:
                std_c = mtp_vcache_to_standard(model, vcache, C)
                hidden0 = model.model(inputs[:, None], cache=std_c)
                logits0 = model.lm_head(hidden0)
                vcache = mtp_vcache_from_standard(model, std_c, C)

        nxt0, lp0 = _logits_to_preds(logits0[:, 0, :])
        mx.eval(nxt0, hidden0, lp0)

        mtp_caches = []
        mtp_exact_offs = []
        for i in range(C):
            st = mtp_states[i] if (mtp_states is not None and i < len(mtp_states)) else None
            mc, pending_h = st if st is not None else (KVCache(), None)
            if pending_h is not None:
                mtp.step_rows(
                    inputs[i : i + 1],
                    pending_h[:, 0, :],
                    mc,
                    C=1,
                    L=1,
                    use_q4_head=True,
                    return_logits=False,
                )
                mx.eval(mc.state)
            mtp_caches.append(mc)
            mtp_exact_offs.append(mc.offset)

        return cls(
            model=model,
            mtp=mtp,
            vcache=vcache,
            mtp_caches=mtp_caches,
            mtp_exact_offs=mtp_exact_offs,
            cur_toks=nxt0.tolist(),
            cur_lps=list(lp0),
            cur_hidden=hidden0[:, 0, :],
            inflight=None,
        )

    def cancel_inflight(self) -> None:
        if self.inflight is not None:
            mtp_cancel_inflight(self.inflight, self.mtp_caches)
            self.inflight = None

    def extract_cache(self, idx: int) -> list:
        self.cancel_inflight()
        C = len(self.cur_toks)
        pending_h = self.cur_hidden[idx : idx + 1].reshape(1, 1, -1)
        return mtp_extract_from_vcache(
            self.model,
            self.vcache,
            idx,
            C,
            mtp_state=(self.mtp_caches[idx], pending_h),
            copy_kv=True,
        )

    def to_standard_cache(self) -> list:
        self.cancel_inflight()
        return mtp_vcache_to_standard(self.model, self.vcache, len(self.cur_toks))

    def nbytes(self) -> int:
        return mtp_vcache_nbytes(self.vcache) + sum(mc.nbytes for mc in self.mtp_caches)

    def filter(self, keep: list[int]) -> None:
        self.cancel_inflight()
        if not keep:
            self.vcache.clear()
            self.mtp_caches.clear()
            self.mtp_exact_offs.clear()
            self.cur_toks.clear()
            self.cur_lps.clear()
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

    def extend(self, other: "LingSanteSpecBatch") -> None:
        self.cancel_inflight()
        other.cancel_inflight()
        C_a = len(self.cur_toks)
        C_b = len(other.cur_toks)
        self.vcache = mtp_extend_vcache(self.model, self.vcache, C_a, other.vcache, C_b)
        self.mtp_caches.extend(other.mtp_caches)
        self.mtp_exact_offs.extend(other.mtp_exact_offs)
        self.cur_toks.extend(other.cur_toks)
        self.cur_lps.extend(other.cur_lps)
        self.cur_hidden = mx.concatenate([self.cur_hidden, other.cur_hidden], axis=0)

    def _launch_cycle(self, K: int) -> None:
        C = len(self.cur_toks)
        cur_arr = mx.array(self.cur_toks, dtype=mx.uint32)
        draft_mat, after0_offs = _chain_mtp_drafts(
            self.mtp,
            cur_arr,
            self.cur_hidden,
            self.mtp_caches,
            C=C,
            K=K,
            use_q4_head=True,
        )
        mx.async_eval(draft_mat)
        inp = mx.concatenate([cur_arr[:, None], draft_mat], axis=1)
        logits, hidden, rb = fast_verify_forward(
            self.model, inp, self.vcache, return_hidden=True, return_rollback=True
        )
        preds, lp = _logits_to_preds(logits)
        mx.async_eval(preds, hidden, draft_mat, lp)
        self.mtp_exact_offs = after0_offs
        self.inflight = {
            "K": K,
            "preds": preds,
            "hidden": hidden,
            "draft_mat": draft_mat,
            "lp": lp,
            "rb": rb,
            "mtp_base_offs": [o - 1 for o in after0_offs],
        }

    def step_next(self, gen_batch, *, prefetch: bool = True) -> list:
        C = len(self.cur_toks)
        if C == 0:
            return []

        max_rem = max(gen_batch.max_tokens[i] - gen_batch._num_tokens[i] for i in range(C))
        if self.inflight is None and max_rem <= 1:
            # Every stream has 1 token left and its prediction + logprobs are already ready.
            fin_lps = list(self.cur_lps)
            cur_arr = mx.array(self.cur_toks, dtype=mx.uint32)
            cache_arg = self.mtp_caches[0] if C == 1 else self.mtp_caches
            self.mtp.step_rows(
                cur_arr,
                self.cur_hidden,
                cache_arg,
                C=C,
                L=1,
                use_q4_head=True,
                return_logits=False,
            )
            _, hidden_fin, rb_fin = fast_verify_forward(
                self.model, cur_arr[:, None], self.vcache, return_hidden=True, return_rollback=True
            )
            mx.eval(hidden_fin, *[x for mc in self.mtp_caches for x in mc.state])
            commit_verify_cache(self.vcache, rb_fin, 1)
            responses = []
            for i in range(C):
                tok = self.cur_toks[i]
                gen_batch.tokens[i].append(tok)
                gen_batch._num_tokens[i] += 1
                fr = "stop" if gen_batch._matchers[i].advance(tok) else "length"
                pc = mtp_extract_from_vcache(
                    self.model,
                    self.vcache,
                    i,
                    C,
                    mtp_state=(self.mtp_caches[i], hidden_fin[i : i + 1, 0:1, :]),
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
            K = min(mtp_k_for_batch(C), max(1, max_rem - 1))
            self._launch_cycle(K)

        inf = self.inflight
        K = inf["K"]
        preds, hidden, draft_mat, lp, rb = (
            inf["preds"],
            inf["hidden"],
            inf["draft_mat"],
            inf["lp"],
            inf["rb"],
        )
        mx.eval(preds, hidden, draft_mat, lp)
        self.inflight = None

        pred_lists = preds.tolist()
        draft_lists = draft_mat.tolist()

        ncs: list[int] = []
        keep: list[int] = []
        fin_indices: list[int] = []
        fin_reasons: dict[int, str] = {}
        stream_toks: list[list[int]] = []
        next_cur_toks: list[int] = []
        next_cur_lps: list[mx.array] = []
        next_cur_hidden_idx: list[int] = []

        for s in range(C):
            a = 0
            for j in range(K):
                if pred_lists[s][j] == draft_lists[s][j]:
                    a += 1
                else:
                    break
            cand_toks = [self.cur_toks[s]] + draft_lists[s][:a]
            emitted_s = []
            fr = None
            for k_i, tok in enumerate(cand_toks):
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
                next_cur_toks.append(pred_lists[s][a])
                next_cur_lps.append(lp[s, a])
                next_cur_hidden_idx.append(a)

        old_uids = list(gen_batch.uids)
        old_cur_lps = list(self.cur_lps)
        after0_offs = list(self.mtp_exact_offs)

        # Commit verified prefix lengths to target KDA + MLA caches
        commit_verify_cache(self.vcache, rb, ncs)

        # Extract finished stream caches BEFORE shrinking vcache
        fin_caches: dict[int, list] = {}
        fin_tokens: dict[int, list[int]] = {}
        if fin_indices:
            for s in fin_indices:
                mc_s = self.mtp_caches[s]
                tr = mc_s.offset - after0_offs[s]
                if tr > 0:
                    mc_s.trim(tr)
                nc_s = ncs[s]
                if nc_s > 1:
                    mtp_d = mx.array(draft_lists[s][: nc_s - 1], dtype=mx.uint32)
                    self.mtp.step_rows(
                        mtp_d,
                        hidden[s, : nc_s - 1, :],
                        mc_s,
                        C=1,
                        L=nc_s - 1,
                        use_q4_head=True,
                        return_logits=False,
                    )
                pending_h = hidden[s : s + 1, nc_s - 1 : nc_s, :]
                mx.eval(mc_s.state, pending_h)
                fin_caches[s] = mtp_extract_from_vcache(
                    self.model,
                    self.vcache,
                    s,
                    C,
                    mtp_state=(mc_s, pending_h),
                    copy_kv=True,
                )
                fin_tokens[s] = list(gen_batch.tokens[s])

        if not keep:
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
        hidden_kept = hidden
        drafts_kept = draft_lists
        ncs_kept = ncs
        after0_kept = after0_offs
        if len(keep) < C:
            _contract_verify_cache(self.model, self.vcache, keep)
            for s in range(C):
                if s not in fin_caches:
                    tr = self.mtp_caches[s].offset - after0_offs[s]
                    if tr > 0:
                        self.mtp_caches[s].trim(tr)
            keep_arr = mx.array(keep, dtype=mx.int32)
            self.mtp_caches = [self.mtp_caches[s] for s in keep]
            after0_kept = [after0_offs[s] for s in keep]
            drafts_kept = [draft_lists[s] for s in keep]
            ncs_kept = [ncs[s] for s in keep]
            hidden_kept = hidden[keep_arr]
            gen_batch._filter_metadata(keep)

        C_act = len(keep)
        cur_a = ncs_kept[0] - 1 if len(set(ncs_kept)) == 1 else None
        if cur_a is not None:
            new_cur_hidden = hidden_kept[:, cur_a, :]
        else:
            new_cur_hidden = mx.stack(
                [hidden_kept[i, nc - 1, :] for i, nc in enumerate(ncs_kept)], axis=0
            )
        self.cur_toks = next_cur_toks
        self.cur_lps = next_cur_lps
        self.cur_hidden = new_cur_hidden

        max_rem_next = max(
            gen_batch.max_tokens[i] - gen_batch._num_tokens[i] for i in range(C_act)
        )
        if prefetch and max_rem_next > 1:
            K_next = min(mtp_k_for_batch(C_act), max_rem_next - 1)
            draft_mat_next, new_after0_offs = _sync_and_chain_mtp(
                self.mtp,
                hidden_kept,
                drafts_kept,
                next_cur_toks,
                ncs_kept,
                self.mtp_caches,
                after0_kept,
                C=C_act,
                K=K_next,
                use_q4_head=True,
            )
            flush_arrs = [draft_mat_next]
            for li, layer in enumerate(self.model.model.layers):
                if layer.is_linear:
                    flush_arrs.extend(x for x in self.vcache[li].cache if x is not None)
            mx.async_eval(*flush_arrs)

            cur_arr = mx.array(next_cur_toks, dtype=mx.uint32)
            inp_next = mx.concatenate([cur_arr[:, None], draft_mat_next], axis=1)
            logits_next, hidden_next, rb_next = fast_verify_forward(
                self.model, inp_next, self.vcache, return_hidden=True, return_rollback=True
            )
            preds_next, lp_next = _logits_to_preds(logits_next)
            mx.async_eval(preds_next, hidden_next, draft_mat_next, lp_next)
            self.mtp_exact_offs = new_after0_offs
            self.inflight = {
                "K": K_next,
                "preds": preds_next,
                "hidden": hidden_next,
                "draft_mat": draft_mat_next,
                "lp": lp_next,
                "rb": rb_next,
                "mtp_base_offs": [o - 1 for o in new_after0_offs],
            }
        else:
            _refresh_mtp_accepted(
                self.mtp,
                self.mtp_caches,
                after0_kept,
                drafts_kept,
                hidden_kept,
                ncs_kept,
                use_q4_head=True,
            )
            self.mtp_exact_offs = [mc.offset for mc in self.mtp_caches]
            self.inflight = None
            flush_arrs = [self.cur_hidden]
            for mc in self.mtp_caches:
                flush_arrs.extend(mc.state)
            for li, layer in enumerate(self.model.model.layers):
                if layer.is_linear:
                    flush_arrs.extend(x for x in self.vcache[li].cache if x is not None)
            mx.async_eval(*flush_arrs)

        # Build responses on CPU while GPU executes next async verify cycle
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


def prefill_with_hidden(model, mtp: BailingMTPLayer, ids: Sequence[int], step: int = 512):
    """Exact `B=1` prefill of `ids[:-1]` through target + shifted MTP cache prefill.

    Returns `(target_cache, mtp_cache, last_prompt_token_id)`.
    """
    t_cache = make_prompt_cache(model)
    p = mx.array(ids, dtype=mx.uint32)
    h_chunks = []
    pos = 0
    while p.size - pos > 1:
        n = min(step, p.size - 1 - pos)
        h = model.model(p[pos : pos + n][None], cache=t_cache)
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
            mtp_out = mtp.prefill(tokens_shifted[:, s_pos : s_pos + m], h_all[:, s_pos : s_pos + m, :], mtp_cache)
            mx.eval(mtp_out, mtp_cache.state)
            s_pos += m
    return t_cache, mtp_cache, int(p[-1].item())


def _chain_mtp_drafts(
    mtp: BailingMTPLayer,
    first_toks: mx.array,
    first_hidden: mx.array,
    mtp_caches: list[KVCache],
    *,
    C: int,
    K: int,
    use_q4_head: bool,
) -> tuple[mx.array, list[int]]:
    """Build all `K` MTP draft steps lazily on GPU (`0` intermediate evals)."""
    cache_arg = mtp_caches[0] if C == 1 else mtp_caches
    h_d, dl0 = mtp.step_rows(first_toks, first_hidden, cache_arg, C=C, L=1, use_q4_head=use_q4_head)
    exact_offs = [mc.offset for mc in mtp_caches]
    d_prev = mx.argmax(dl0, axis=-1)
    cols = [d_prev]
    for _ in range(1, K):
        h_d, dlk = mtp.step_rows(d_prev, h_d, cache_arg, C=C, L=1, use_q4_head=use_q4_head)
        d_prev = mx.argmax(dlk, axis=-1)
        cols.append(d_prev)
    return mx.stack(cols, axis=1), exact_offs


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
    K: int,
    use_q4_head: bool,
) -> tuple[mx.array, list[int]]:
    """Trim unaccepted MTP KV entries, refresh true target hidden states grouped by `nc`, and chain remaining `K-1` drafts lazily."""
    for c in range(C):
        tr = mtp_caches[c].offset - mtp_exact_offs[c]
        if tr > 0:
            mtp_caches[c].trim(tr)

    if len(set(ncs)) == 1:
        nc = ncs[0]
        toks_g = mx.array([draft_lists[c][: nc - 1] + [cur_toks[c]] for c in range(C)], dtype=mx.uint32)
        hs_g = hidden[:, :nc, :]
        cache_arg = mtp_caches[0] if C == 1 else mtp_caches
        h_d, dl0 = mtp.step_rows(toks_g, hs_g, cache_arg, C=C, L=nc, use_q4_head=use_q4_head)
        new_exact_offs = [mc.offset for mc in mtp_caches]
        d_prev = mx.argmax(dl0, axis=-1)
    else:
        toks_flat = mx.array(
            [tok for c, nc in enumerate(ncs) for tok in (draft_lists[c][: nc - 1] + [cur_toks[c]])],
            dtype=mx.uint32,
        )
        hs_flat = mx.concatenate([hidden[c, :nc, :] for c, nc in enumerate(ncs)], axis=0)
        h_d, dl0 = mtp.step_varlen_rows(toks_flat, hs_flat, mtp_caches, ncs, use_q4_head=use_q4_head)
        new_exact_offs = [mc.offset for mc in mtp_caches]
        d_prev = mx.argmax(dl0, axis=-1)

    cols = [d_prev]
    cache_arg = mtp_caches[0] if C == 1 else mtp_caches
    for _ in range(1, K):
        h_d, dlk = mtp.step_rows(d_prev, h_d, cache_arg, C=C, L=1, use_q4_head=use_q4_head)
        d_prev = mx.argmax(dlk, axis=-1)
        cols.append(d_prev)
    return mx.stack(cols, axis=1), new_exact_offs


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
    K: int,
    max_tokens: int = 128,
    use_q4_head: bool = True,
    prefill_step: int = 512,
    capture_lp_steps: int = 32,
):
    """Single-sync async-pipelined greedy speculative decode for `1 <= C <= 4` streams with contraction."""
    C = len(prompts_ids)
    assert 1 <= C <= 4 and K >= 1 and C * (K + 1) <= 24

    t_caches, mtp_caches, last_toks = [], [], []
    for ids in prompts_ids:
        tc, mc, lt = prefill_with_hidden(model, mtp, ids, step=prefill_step)
        t_caches.append(tc)
        mtp_caches.append(mc)
        last_toks.append(lt)

    vcache = make_verify_cache(model, t_caches)

    # Initial L=1 target forward on last prompt token
    logits0, hidden0, rb0 = fast_verify_forward(
        model, mx.array([[lt] for lt in last_toks], dtype=mx.uint32), vcache, return_hidden=True, return_rollback=True
    )
    nxt0, lp0 = _logits_to_preds(logits0[:, 0, :])
    mx.eval(nxt0, hidden0, lp0)
    commit_verify_cache(vcache, rb0, 1)

    cur_toks = nxt0.tolist()
    emitted = [[cur_toks[c]] for c in range(C)]
    lp_arrays: list[list[mx.array]] = [[lp0[c]] if capture_lp_steps > 0 else [] for c in range(C)]

    # Build + launch cycle 1 (initial MTP draft + first multi-token verify) asynchronously
    draft_mat, mtp_exact_offs = _chain_mtp_drafts(
        mtp, nxt0, hidden0[:, 0, :], mtp_caches, C=C, K=K, use_q4_head=use_q4_head
    )
    inp = mx.concatenate([nxt0[:, None], draft_mat], axis=1)
    logits, hidden, rb = fast_verify_forward(model, inp, vcache, return_hidden=True, return_rollback=True)
    preds, lp = _logits_to_preds(logits)

    mx.reset_peak_memory()
    t_start = time.perf_counter()
    mx.async_eval(preds, hidden, draft_mat, lp)

    C_act = C
    active_orig = list(range(C))
    n_verify_steps = 0
    active_stream_steps = 0
    sum_emitted = 0
    steady_emitted = 0
    steady_wall_s = None
    hits = [0] * K
    trials = [0] * K

    while True:
        mx.eval(preds, hidden, draft_mat, lp)
        pred_lists = preds.tolist()
        draft_lists = draft_mat.tolist()

        ncs = []
        next_cur_toks = []
        pending_tokens = []
        pending_lps = []
        keep = []
        for s in range(C_act):
            orig = active_orig[s]
            a = 0
            for j in range(K):
                trials[j] += 1
                if pred_lists[s][j] == draft_lists[s][j]:
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

        # Commit current active batch, then contract finished streams before MTP/next verify
        commit_verify_cache(vcache, rb, ncs)
        old_active_orig = active_orig
        if len(keep) < C_act:
            _contract_verify_cache(model, vcache, keep)
            for s in range(C_act):
                tr = mtp_caches[s].offset - mtp_exact_offs[s]
                if tr > 0:
                    mtp_caches[s].trim(tr)
            keep_arr = mx.array(keep, dtype=mx.int32)
            mtp_caches = [mtp_caches[s] for s in keep]
            mtp_exact_offs = [mtp_exact_offs[s] for s in keep]
            draft_lists = [draft_lists[s] for s in keep]
            next_cur_toks = [next_cur_toks[s] for s in keep]
            ncs = [ncs[s] for s in keep]
            hidden = hidden[keep_arr]
            active_orig = [active_orig[s] for s in keep]
            C_act = len(keep)

        draft_mat_next, mtp_exact_offs = _sync_and_chain_mtp(
            mtp,
            hidden,
            draft_lists,
            next_cur_toks,
            ncs,
            mtp_caches,
            mtp_exact_offs,
            C=C_act,
            K=K,
            use_q4_head=use_q4_head,
        )
        flush_arrs = [draft_mat_next]
        for li, layer in enumerate(model.model.layers):
            if layer.is_linear:
                flush_arrs.extend(x for x in vcache[li].cache if x is not None)
        mx.async_eval(*flush_arrs)

        cur_arr = mx.array(next_cur_toks, dtype=mx.uint32)
        inp_next = mx.concatenate([cur_arr[:, None], draft_mat_next], axis=1)
        logits_next, hidden_next, rb_next = fast_verify_forward(
            model, inp_next, vcache, return_hidden=True, return_rollback=True
        )
        preds_next, lp_next = _logits_to_preds(logits_next)
        mx.async_eval(preds_next, hidden_next, draft_mat_next, lp_next)

        # CPU work overlapped with GPU execution of cycle s+1
        for s, orig in enumerate(old_active_orig):
            emitted[orig].extend(pending_tokens[s])
            lp_arrays[orig].extend(pending_lps[s])

        preds, hidden, draft_mat, lp, rb = preds_next, hidden_next, draft_mat_next, lp_next, rb_next

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
        "alphas": [round(hits[j] / trials[j], 3) if trials[j] > 0 else 0.0 for j in range(K)],
        "peak_gb": round(mx.get_peak_memory() / 1e9, 2),
    }
