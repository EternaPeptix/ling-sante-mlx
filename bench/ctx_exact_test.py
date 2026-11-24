"""Long-context (--ctx 2048) bit-exactness check: fastdecode C=1 and C=4 vs stock B=1 (0 diffs)."""
import argparse
import os
import sys
import time
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(_root / "vendor"), str(_root)]

import mlx_lm  # noqa: E402,F401
import mlx.core as mx  # noqa: E402
from mlx_lm import load  # noqa: E402
from mlx_lm.generate import generate_step  # noqa: E402
from mlx_lm.utils import maybe_set_recommended_wired_limit  # noqa: E402

from bench.batch_dec import merge_caches, prefill_stream  # noqa: E402
from bench.exact import DEFAULT_MODEL, PROMPTS, lp_digest, prompt_ids  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--tokens", type=int, default=32)
    a = ap.parse_args()

    t0 = time.perf_counter()
    maybe_set_recommended_wired_limit()
    model, tok = load(DEFAULT_MODEL)
    mx.eval(model.parameters())
    print(f"loaded in {time.perf_counter()-t0:.1f}s", flush=True)

    greedy = lambda lp: mx.argmax(lp, axis=-1)
    stream_ids = []
    for i, (_, text) in enumerate(PROMPTS):
        filler = [(17 * j * j + 3 * j + 7 * i) % 40000 + 1000 for j in range(a.ctx)]
        stream_ids.append(filler + prompt_ids(tok, text))

    # 1. Stock B=1 reference with LINGSANTE_FAST_DECODE=0
    os.environ["LINGSANTE_FAST_DECODE"] = "0"
    refs = []
    for i, ids in enumerate(stream_ids):
        toks, digs = [], []
        for s_i, (t, lp) in enumerate(generate_step(mx.array(ids), model, max_tokens=a.tokens, sampler=greedy)):
            toks.append(int(t))
            digs.append(lp_digest(lp))
        refs.append((toks, digs))
        print(f"  stock B=1 stream {i} len={len(ids)}: tok0..2={toks[:3]} d0={digs[0]}", flush=True)

    # 2. Fast decode C=1 and C=4 with LINGSANTE_FAST_DECODE=1
    os.environ["LINGSANTE_FAST_DECODE"] = "1"
    ok = True
    for C in (1, 4):
        per_c, last_t = [], []
        for i in range(C):
            c, lt = prefill_stream(model, stream_ids[i])
            per_c.append(c)
            last_t.append(lt)
        cache = per_c[0] if C == 1 else merge_caches(per_c)
        y = mx.array(last_t, dtype=mx.uint32)
        toks_out = [[] for _ in range(C)]
        digs_out = [[] for _ in range(C)]
        t_start = time.perf_counter()
        for step_i in range(a.tokens):
            lg = model(y[:, None], cache=cache)[:, -1, :]
            lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
            nxt = mx.argmax(lp, axis=-1)
            mx.eval(nxt, lp)
            n_list = nxt.tolist()
            for i in range(C):
                toks_out[i].append(n_list[i])
                digs_out[i].append(lp_digest(lp[i]))
            y = nxt
        dt = time.perf_counter() - t_start
        tok_exact = all(toks_out[i] == refs[i][0] for i in range(C))
        lp_exact = all(digs_out[i] == refs[i][1] for i in range(C))
        ok &= tok_exact and lp_exact
        print(
            f"[ctx={a.ctx}] fastdecode C={C}: {C*a.tokens/dt:.2f} agg tok/s "
            f"exact_tok={tok_exact} exact_lp={lp_exact}",
            flush=True,
        )
        del cache, per_c
        mx.clear_cache()

    print("SUMMARY:", "ALL PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
