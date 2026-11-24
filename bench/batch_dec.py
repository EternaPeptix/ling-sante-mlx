"""Batched decode benchmark & bit-exact gate for C=1..4 concurrent streams.

Tests two modes against the single-stream B=1 reference (runs/baseline.json):
  --mode model     : independent B=1 prefill per stream (identical to generate_step), merge caches,
                     then step C streams through model(tokens, cache=...) (or fast decoder).
  --mode batchgen  : end-to-end mlx_lm.generate.BatchGenerator with C concurrent prompts.

Reports median over --reps runs: aggregate tok/s, per-stream tok/s, ms/step, peak GB, and bit-exactness.
"""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(_root / "vendor"), str(_root)]

import mlx_lm  # noqa: E402,F401  # before mlx.core so env buffer hooks apply if present
import mlx.core as mx  # noqa: E402
from mlx_lm import load  # noqa: E402
from mlx_lm.generate import BatchGenerator, generate_step  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache  # noqa: E402
from mlx_lm.utils import maybe_set_recommended_wired_limit  # noqa: E402

from bench.exact import DEFAULT_BASE, DEFAULT_MODEL, PROMPTS, lp_digest, prompt_ids  # noqa: E402


def prefill_stream(model, ids: list[int], step: int = 2048):
    cache = make_prompt_cache(model)
    p = mx.array(ids)
    while p.size > 1:
        n = min(step, p.size - 1)
        model(p[:n][None], cache=cache)
        mx.eval([c.state for c in cache])
        p = p[n:]
    return cache, int(p[0].item())


def merge_caches(per_stream_caches: list[list]):
    n_layers = len(per_stream_caches[0])
    return [type(per_stream_caches[0][L]).merge([c[L] for c in per_stream_caches]) for L in range(n_layers)]


def load_or_build_ref(model, tok, base_path: str, n_tokens: int, log_steps: int = 32):
    if base_path and Path(base_path).exists():
        data = json.loads(Path(base_path).read_text())
        if data.get("n_tokens", 0) >= n_tokens:
            return [
                (
                    data["prompts"][name]["ids"],
                    data["prompts"][name]["tokens"][:n_tokens],
                    data["prompts"][name]["logprob_sha256_32"][:log_steps],
                )
                for name, _ in PROMPTS
            ]
    greedy = lambda lp: mx.argmax(lp, axis=-1)
    refs = []
    for _, text in PROMPTS:
        ids = prompt_ids(tok, text)
        toks, digs = [], []
        for s_i, (t, lp) in enumerate(generate_step(mx.array(ids), model, max_tokens=n_tokens, sampler=greedy)):
            toks.append(int(t))
            if s_i < log_steps:
                digs.append(lp_digest(lp))
        refs.append((ids, toks, digs))
    return refs


def run_model_batch(model, refs, C: int, n_tokens: int, warm: int, ctx: int = 0):
    stream_ids = []
    for i in range(C):
        ids = refs[i][0]
        if ctx > 0:
            filler = [(17 * j * j + 3 * j + 7 * i) % 40000 + 1000 for j in range(ctx)]
            ids = filler + ids
        stream_ids.append(ids)

    per_caches = []
    last_toks = []
    for ids in stream_ids:
        c, last = prefill_stream(model, ids)
        per_caches.append(c)
        last_toks.append(last)

    cache = per_caches[0] if C == 1 else merge_caches(per_caches)
    y = mx.array(last_toks, dtype=mx.uint32)

    def step_fn(toks_vec):
        logits = model(toks_vec[:, None], cache=cache)[:, -1, :]
        # Per-row logsumexp or batched: match Batch._step directly
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        nxt = mx.argmax(lp, axis=-1)
        return nxt, lp

    mx.reset_peak_memory()
    tok_hist = []
    lp_digests = [[] for _ in range(C)]

    nxt, lp = step_fn(y)
    mx.async_eval(nxt, lp)
    t0 = None
    for s_i in range(n_tokens):
        nxt2 = lp2 = None
        if s_i + 1 < n_tokens:
            nxt2, lp2 = step_fn(nxt)
            mx.async_eval(nxt2, lp2)
        mx.eval(nxt, lp)
        tok_hist.append(nxt.tolist())
        if s_i < 32:
            for i in range(C):
                lp_digests[i].append(lp_digest(lp[i]))
        if s_i + 1 == warm:
            t0 = time.perf_counter()
        nxt, lp = nxt2, lp2

    dt = time.perf_counter() - t0
    peak_gb = mx.get_peak_memory() / 1e9
    streams = [[row[i] for row in tok_hist] for i in range(C)]

    exact_tok = exact_lp = None
    first_tok_diff = {}
    first_lp_diff = {}
    if ctx == 0:
        for i in range(C):
            ref_toks = refs[i][1][:n_tokens]
            ref_digs = refs[i][2][: min(32, n_tokens)]
            if streams[i] != ref_toks:
                first_tok_diff[i] = next(j for j, (a, b) in enumerate(zip(streams[i], ref_toks)) if a != b)
            if lp_digests[i][: len(ref_digs)] != ref_digs:
                first_lp_diff[i] = next(j for j, (a, b) in enumerate(zip(lp_digests[i], ref_digs)) if a != b)
        exact_tok = not first_tok_diff
        exact_lp = not first_lp_diff

    agg_tps = C * (n_tokens - warm) / dt
    ms_step = 1e3 * dt / (n_tokens - warm)
    del cache, per_caches
    mx.clear_cache()
    return {
        "agg_tok_s": agg_tps,
        "stream_tok_s": agg_tps / C,
        "ms_step": ms_step,
        "peak_gb": peak_gb,
        "exact_tok": exact_tok,
        "exact_lp32": exact_lp,
        "first_tok_diff": first_tok_diff,
        "first_lp_diff": first_lp_diff,
    }


def run_batchgen(model, tok, refs, C: int, n_tokens: int):
    greedy = lambda lp: mx.argmax(lp, axis=-1)
    gen = BatchGenerator(
        model,
        max_tokens=n_tokens,
        sampler=greedy,
        completion_batch_size=C,
        prefill_batch_size=1,
        stop_tokens=[],
    )
    prompts = [refs[i][0] for i in range(C)]
    uids = gen.insert(prompts, max_tokens=[n_tokens] * C)
    uid_to_idx = {u: i for i, u in enumerate(uids)}
    streams = [[] for _ in range(C)]
    lp_digs = [[] for _ in range(C)]

    mx.reset_peak_memory()
    t_start = time.perf_counter()
    first_decode_t = None
    last_steady_t = None
    total_gen = 0
    steady_gen = 0
    steady_steps = 0
    while True:
        p_resps, resps = gen.next()
        if not p_resps and not resps:
            break
        if not resps:
            continue
        for r in resps:
            idx = uid_to_idx[r.uid]
            if len(lp_digs[idx]) < 32:
                lp_digs[idx].append(lp_digest(r.logprobs))
            streams[idx].append(int(r.token))
            total_gen += 1
        if first_decode_t is None:
            if len(resps) == C:
                first_decode_t = time.perf_counter()
        elif len(resps) == C:
            steady_gen += C
            steady_steps += 1
            last_steady_t = time.perf_counter()
    dt_total = time.perf_counter() - t_start
    dt_dec = (last_steady_t - first_decode_t) if (first_decode_t and last_steady_t) else dt_total
    peak_gb = mx.get_peak_memory() / 1e9

    first_tok_diff = {}
    first_lp_diff = {}
    for i in range(C):
        ref_toks = refs[i][1][:n_tokens]
        ref_digs = refs[i][2][: min(32, n_tokens)]
        if streams[i] != ref_toks:
            first_tok_diff[i] = next(
                (j for j, (a, b) in enumerate(zip(streams[i], ref_toks)) if a != b),
                min(len(streams[i]), len(ref_toks)),
            )
        if lp_digs[i][: len(ref_digs)] != ref_digs:
            first_lp_diff[i] = next(
                (j for j, (a, b) in enumerate(zip(lp_digs[i], ref_digs)) if a != b),
                min(len(lp_digs[i]), len(ref_digs)),
            )
    gen.close()
    mx.clear_cache()
    return {
        "agg_tok_s": steady_gen / dt_dec if dt_dec > 0 and steady_gen > 0 else 0.0,
        "e2e_tok_s": total_gen / dt_total,
        "ms_step": 1e3 * dt_dec / max(1, steady_steps),
        "peak_gb": peak_gb,
        "exact_tok": not first_tok_diff,
        "exact_lp32": not first_lp_diff,
        "first_tok_diff": first_tok_diff,
        "first_lp_diff": first_lp_diff,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--C", default="1,2,3,4")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--warm", type=int, default=4)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--mode", default="model", choices=["model", "batchgen", "both"])
    ap.add_argument("--ctx", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    t0 = time.perf_counter()
    maybe_set_recommended_wired_limit()
    model, tok = load(a.model)
    mx.eval(model.parameters())
    print(f"loaded in {time.perf_counter()-t0:.1f}s, active {mx.get_active_memory()/1e9:.1f} GB", flush=True)
    refs = load_or_build_ref(model, tok, a.base, a.tokens)
    Cs = [int(x) for x in a.C.split(",")]
    modes = ["model", "batchgen"] if a.mode == "both" else [a.mode]
    out_data = {
        "flags": {k: v for k, v in os.environ.items() if k.startswith(("LINGSANTE_", "MLX_"))},
        "ctx": a.ctx,
        "tokens": a.tokens,
        "reps": a.reps,
        "runs": {},
    }

    all_ok = True
    for mode in modes:
        for C in Cs:
            runs = []
            for r in range(a.reps):
                res = run_model_batch(model, refs, C, a.tokens, a.warm, ctx=a.ctx) if mode == "model" else run_batchgen(model, tok, refs, C, a.tokens)
                runs.append(res)
            agg_med = statistics.median(x["agg_tok_s"] for x in runs)
            ms_med = statistics.median(x["ms_step"] for x in runs)
            peak_max = max(x["peak_gb"] for x in runs)
            ex_tok = all(x["exact_tok"] in (True, None) for x in runs) and (runs[0]["exact_tok"] is not None)
            ex_lp = all(x["exact_lp32"] in (True, None) for x in runs) and (runs[0]["exact_lp32"] is not None)
            if a.ctx == 0:
                all_ok &= ex_tok and ex_lp
            print(
                f"[{mode:8s}] C={C} agg={agg_med:6.2f} tok/s ({agg_med/C:5.2f}/stream, {ms_med:6.2f} ms/step) "
                f"peak={peak_max:.1f}GB exact_tok={runs[0]['exact_tok']} exact_lp32={runs[0]['exact_lp32']} "
                f"diff_tok={runs[0]['first_tok_diff']} diff_lp={runs[0]['first_lp_diff']}",
                flush=True,
            )
            out_data["runs"][f"{mode}/C{C}"] = {
                "agg_tok_s_med": round(agg_med, 3),
                "agg_tok_s_all": [round(x["agg_tok_s"], 3) for x in runs],
                "ms_step_med": round(ms_med, 3),
                "peak_gb": round(peak_max, 2),
                "exact_tok": runs[0]["exact_tok"],
                "exact_lp32": runs[0]["exact_lp32"],
                "first_tok_diff": runs[0]["first_tok_diff"],
                "first_lp_diff": runs[0]["first_lp_diff"],
            }
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out_data, indent=1))
    if a.ctx == 0 and not all_ok:
        print("NOTE: exactness check had diffs (expected for unoptimized stock B>1)", flush=True)


if __name__ == "__main__":
    main()
