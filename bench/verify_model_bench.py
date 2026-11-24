"""Full-model adversarial rollback bit-exactness test and T_verify(C, L) benchmark."""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(_root / "vendor"), str(_root)]

import mlx_lm  # noqa: E402,F401
import mlx.core as mx  # noqa: E402
from mlx_lm import load  # noqa: E402
from mlx_lm.models.bailing_moe_v3.verify_decode import (  # noqa: E402
    commit_verify_cache,
    fast_verify_forward,
    make_verify_cache,
)
from mlx_lm.models.cache import make_prompt_cache  # noqa: E402
from mlx_lm.utils import maybe_set_recommended_wired_limit  # noqa: E402

from bench.exact import DEFAULT_BASE, DEFAULT_MODEL, PROMPTS, lp_digest  # noqa: E402


def prefill_stream(model, ids: list[int], step: int = 2048):
    cache = make_prompt_cache(model)
    p = mx.array(ids, dtype=mx.uint32)
    while p.size > 1:
        n = min(step, p.size - 1)
        model(p[:n][None], cache=cache)
        mx.eval([c.state for c in cache])
        p = p[n:]
    return cache, int(p[0].item())


def test_adversarial_c1(model, p_entry: dict, L: int, max_tok: int = 64):
    ids = p_entry["ids"]
    gold_toks = p_entry["tokens"][:max_tok]
    gold_digs = p_entry["logprob_sha256_32"]

    cache, cur = prefill_stream(model, ids)
    pos = 0
    step_idx = 0
    pattern = [0, 1, L - 1, max(1, (L - 1) // 2), max(0, L - 2)]
    tok_diffs = 0
    lp_diffs = 0

    while pos < len(gold_toks):
        a = pattern[step_idx % len(pattern)]
        a = min(a, L - 1, len(gold_toks) - pos - 1)
        draft = [cur]
        for j in range(L - 1):
            if j < a:
                draft.append(gold_toks[pos + j])
            elif j == a and pos + j < len(gold_toks):
                draft.append((gold_toks[pos + j] + 137) % 157184)
            else:
                draft.append((1000 + j * 17) % 157184)

        logits, _, rb = fast_verify_forward(model, mx.array([draft], dtype=mx.uint32), cache, return_hidden=True, return_rollback=True)
        lp = logits[0] - mx.logsumexp(logits[0], axis=-1, keepdims=True)
        preds = mx.argmax(lp, axis=-1)
        mx.eval(preds, lp)

        pred_list = preds.tolist()
        for j in range(a + 1):
            if pos + j >= len(gold_toks):
                break
            gt = gold_toks[pos + j]
            if pred_list[j] != gt:
                tok_diffs += 1
                if tok_diffs <= 5:
                    print(f"  TOK DIFF L={L} pos={pos+j}: got={pred_list[j]} gold={gt}", flush=True)
            if pos + j < len(gold_digs) and lp_digest(lp[j]) != gold_digs[pos + j]:
                lp_diffs += 1
                if lp_diffs <= 5:
                    print(f"  LP DIFF L={L} pos={pos+j}: got={lp_digest(lp[j])} gold={gold_digs[pos+j]}", flush=True)

        commit_verify_cache(cache, rb, a + 1)
        cur = pred_list[a]
        pos += a + 1
        step_idx += 1

    print(f"  adversarial C=1 L={L:2d}: tok_diffs={tok_diffs} lp_diffs={lp_diffs} steps={step_idx}", flush=True)
    return tok_diffs == 0 and lp_diffs == 0


def test_adversarial_c4(model, prompts_data: dict, C: int = 4, L: int = 4, max_tok: int = 48):
    names = [name for name, _ in PROMPTS[:C]]
    caches, curs = [], []
    for nm in names:
        c, last = prefill_stream(model, prompts_data[nm]["ids"])
        caches.append(c)
        curs.append(last)
    vcache = make_verify_cache(model, caches)
    pos = [0] * C
    step_idx = 0
    tok_diffs = 0
    lp_diffs = 0

    while min(pos) < max_tok:
        batch_rows = []
        ncs = []
        for c in range(C):
            rem = max_tok - pos[c]
            a = min((c + step_idx) % L, L - 1, max(0, rem - 1))
            ncs.append(a + 1)
            gtoks = prompts_data[names[c]]["tokens"]
            row = [curs[c]]
            for j in range(L - 1):
                if j < a:
                    row.append(gtoks[pos[c] + j])
                elif j == a and pos[c] + j < len(gtoks):
                    row.append((gtoks[pos[c] + j] + 91) % 157184)
                else:
                    row.append((2000 + j * 13) % 157184)
            batch_rows.append(row)

        logits, _, rb = fast_verify_forward(model, mx.array(batch_rows, dtype=mx.uint32), vcache, return_hidden=True, return_rollback=True)
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        preds = mx.argmax(lp, axis=-1)
        mx.eval(preds, lp)
        pred_lists = preds.tolist()

        for c in range(C):
            gtoks = prompts_data[names[c]]["tokens"]
            gdigs = prompts_data[names[c]]["logprob_sha256_32"]
            for j in range(ncs[c]):
                p = pos[c] + j
                if p >= max_tok:
                    continue
                if pred_lists[c][j] != gtoks[p]:
                    tok_diffs += 1
                    print(f"  C={C} c={c} pos={p} TOK DIFF got={pred_lists[c][j]} gold={gtoks[p]}")
                if p < len(gdigs) and lp_digest(lp[c, j]) != gdigs[p]:
                    lp_diffs += 1
                    print(f"  C={C} c={c} pos={p} LP DIFF")
            curs[c] = pred_lists[c][ncs[c] - 1]
            pos[c] += ncs[c]

        commit_verify_cache(vcache, rb, ncs)
        mx.eval([c.state for c in vcache if hasattr(c, "state")])
        step_idx += 1

    print(f"  adversarial C={C} L={L:2d}: tok_diffs={tok_diffs} lp_diffs={lp_diffs} steps={step_idx}", flush=True)
    return tok_diffs == 0 and lp_diffs == 0


def bench_verify(model, prompts_data: dict, C: int, L: int, reps: int = 5):
    names = [name for name, _ in PROMPTS[:C]]
    seqs = []
    for nm in names:
        ids = prompts_data[nm]["ids"]
        gtoks = prompts_data[nm]["tokens"]
        seqs.append((ids, gtoks))

    caches = []
    chunks = []
    for c in range(C):
        sc, last = prefill_stream(model, seqs[c][0])
        caches.append(sc)
        chunks.append([last] + seqs[c][1][: L - 1])
    vcache = make_verify_cache(model, caches)
    x_in = mx.array(chunks, dtype=mx.uint32)

    # Snapshot initial KDA state/conv + MLA offset to replay identically for clean repeated timing
    kda_snap = []
    mla_offs = []
    for layer, c in zip(model.model.layers, vcache):
        if layer.is_linear:
            kda_snap.append((c, [mx.array(a) for a in c.cache]))
        else:
            if isinstance(c, list):
                mla_offs.append([ci.offset for ci in c])
            else:
                mla_offs.append(c.offset)

    def restore():
        for c, arrs in kda_snap:
            for j, a in enumerate(arrs):
                c[j] = a
        mi = 0
        for layer, c in zip(model.model.layers, vcache):
            if not layer.is_linear:
                o = mla_offs[mi]
                if isinstance(c, list):
                    for ci, oi in zip(c, o):
                        ci.offset = oi
                else:
                    c.offset = o
                mi += 1

    # Warmup (both full commit and partial rollback)
    for _ in range(2):
        restore()
        logits, hidden, rb = fast_verify_forward(model, x_in, vcache, return_hidden=True, return_rollback=True)
        toks = mx.argmax(logits, axis=-1)
        mx.eval(toks, hidden)
        if L > 1:
            commit_verify_cache(vcache, rb, max(1, L // 2))
            mx.eval([s for c, _ in kda_snap for s in c.cache[3:]])

    mx.reset_peak_memory()
    t_verify, t_full_commit, t_part_commit = [], [], []
    for r in range(reps):
        restore()
        mx.synchronize()
        t0 = time.perf_counter()
        logits, hidden, rb = fast_verify_forward(model, x_in, vcache, return_hidden=True, return_rollback=True)
        toks = mx.argmax(logits, axis=-1)
        mx.eval(toks, hidden)
        t1 = time.perf_counter()
        commit_verify_cache(vcache, rb, L)
        mx.eval([s for c, _ in kda_snap for s in c.cache[3:]])
        t2 = time.perf_counter()
        t_verify.append((t1 - t0) * 1e3)
        t_full_commit.append((t2 - t1) * 1e3)

        if L > 1:
            restore()
            logits, hidden, rb = fast_verify_forward(model, x_in, vcache, return_hidden=True, return_rollback=True)
            toks = mx.argmax(logits, axis=-1)
            mx.eval(toks, hidden)
            t3 = time.perf_counter()
            commit_verify_cache(vcache, rb, max(1, L // 2))
            mx.eval([s for c, _ in kda_snap for s in c.cache[3:]])
            t4 = time.perf_counter()
            t_part_commit.append((t4 - t3) * 1e3)

    vm = statistics.median(t_verify)
    fcm = statistics.median(t_full_commit)
    pcm = statistics.median(t_part_commit) if t_part_commit else 0.0
    peak_gb = mx.get_peak_memory() / 1e9
    print(
        f"  C={C} L={L:2d} (B={C*L:2d}): verify={vm:6.2f} ms  commit_full={fcm:5.2f} ms  rollback_half={pcm:5.2f} ms  "
        f"ceil_tok_s={C*L/((vm+fcm)/1e3):6.2f}  peak={peak_gb:.1f} GB",
        flush=True,
    )
    return {
        "C": C,
        "L": L,
        "B": C * L,
        "verify_ms": round(vm, 3),
        "commit_full_ms": round(fcm, 3),
        "rollback_half_ms": round(pcm, 3),
        "ceiling_tok_s": round(C * L / ((vm + fcm) / 1e3), 2),
        "peak_gb": round(peak_gb, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser(DEFAULT_MODEL))
    ap.add_argument("--base", default=os.path.expanduser(DEFAULT_BASE))
    ap.add_argument("--out", default=os.path.expanduser("~/science/runs/verify_bench_v2.json"))
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--skip-exact", action="store_true")
    a = ap.parse_args()

    maybe_set_recommended_wired_limit()
    base = json.loads(Path(a.base).read_text())
    prompts_data = base["prompts"]

    t0 = time.perf_counter()
    model, _ = load(a.model)
    mx.eval(model.parameters())
    print(f"Loaded model in {time.perf_counter()-t0:.1f}s ({mx.get_active_memory()/1e9:.1f} GB)", flush=True)

    exact_ok = True
    if not a.skip_exact:
        print("\n=== Adversarial Rollback Bit-Exactness vs baseline.json ===", flush=True)
        for L in (4, 8, 12, 16):
            exact_ok &= test_adversarial_c1(model, prompts_data["medical_stemi"], L, max_tok=64)
        exact_ok &= test_adversarial_c4(model, prompts_data, C=4, L=4, max_tok=48)
        exact_ok &= test_adversarial_c4(model, prompts_data, C=4, L=5, max_tok=48)
        if not exact_ok:
            print("EXACTNESS FAILED", flush=True)
            return 1

    print("\n=== T_verify(C, L) & Rollback Benchmark ===", flush=True)
    results = []
    for L in (1, 2, 4, 6, 8, 10, 12, 14, 15, 16, 20):
        results.append(bench_verify(model, prompts_data, C=1, L=L, reps=a.reps))
    for C in (2, 3, 4):
        for L in (2, 3, 4, 5, 6):
            if C * L <= 24 and not (C == 2 and L in (5,)):
                results.append(bench_verify(model, prompts_data, C=C, L=L, reps=a.reps))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"exact_ok": exact_ok, "results": results}, indent=2))
    print(f"Saved {a.out}", flush=True)
    return 0 if exact_ok else 1


if __name__ == "__main__":
    sys.exit(main())
