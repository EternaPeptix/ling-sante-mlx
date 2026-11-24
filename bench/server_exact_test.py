"""Dynamic concurrency BatchGenerator bit-exactness test (1 -> 4 -> 2 -> 3 -> 1) against baseline.json."""
import json
import sys
import time
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(_root / "vendor"), str(_root)]

import mlx_lm  # noqa: E402,F401
import mlx.core as mx  # noqa: E402
from mlx_lm import load  # noqa: E402
from mlx_lm.generate import BatchGenerator  # noqa: E402
from mlx_lm.utils import maybe_set_recommended_wired_limit  # noqa: E402

from bench.exact import DEFAULT_BASE, DEFAULT_MODEL, PROMPTS, lp_digest, prompt_ids  # noqa: E402


def main():
    base = json.loads(Path(DEFAULT_BASE).read_text())
    refs = [(base["prompts"][nm]["ids"], base["prompts"][nm]["tokens"], base["prompts"][nm]["logprob_sha256_32"]) for nm, _ in PROMPTS]

    t0 = time.perf_counter()
    maybe_set_recommended_wired_limit()
    model, _ = load(DEFAULT_MODEL)
    mx.eval(model.parameters())
    print(f"loaded in {time.perf_counter()-t0:.1f}s", flush=True)

    greedy = lambda lp: mx.argmax(lp, axis=-1)
    gen = BatchGenerator(
        model,
        max_tokens=128,
        sampler=greedy,
        completion_batch_size=4,
        prefill_batch_size=1,
        stop_tokens=[],
    )

    records = {}  # uid -> (ref_idx, max_toks, toks, digs)

    def add_req(ref_idx: int, max_t: int):
        [uid] = gen.insert([refs[ref_idx][0]], max_tokens=[max_t])
        records[uid] = [ref_idx, max_t, [], []]
        return uid

    def step_until(pred):
        while not pred():
            p_resps, resps = gen.next()
            if not p_resps and not resps:
                break
            for r in resps:
                rec = records[r.uid]
                if len(rec[3]) < 32:
                    rec[3].append(lp_digest(r.logprobs))
                rec[2].append(int(r.token))

    # Stage 1: C=1
    u0 = add_req(0, 128)
    step_until(lambda: len(records[u0][2]) >= 10)
    print(f"After stage 1 (C=1): P0 len={len(records[u0][2])}", flush=True)

    # Stage 2: 1 -> 4
    u1 = add_req(1, 20)
    u2 = add_req(2, 20)
    u3 = add_req(3, 60)
    step_until(lambda: len(records[u1][2]) >= 20 and len(records[u2][2]) >= 20)
    print(f"After stage 2 (1->4->2): P0={len(records[u0][2])}, P1={len(records[u1][2])}, P2={len(records[u2][2])}, P3={len(records[u3][2])}", flush=True)

    # Stage 3: stay at C=2 for 5 steps, then 2 -> 3 -> 1
    target_p0 = len(records[u0][2]) + 5
    step_until(lambda: len(records[u0][2]) >= target_p0)
    u1b = add_req(1, 15)
    print(f"Stage 3 (2->3): inserted P1 again at P0={len(records[u0][2])}", flush=True)
    step_until(lambda: len(records[u0][2]) >= 128 and len(records[u3][2]) >= 60 and len(records[u1b][2]) >= 15)

    gen.close()

    ok = True
    for uid, (ridx, mt, toks, digs) in records.items():
        ref_t = refs[ridx][1][:mt]
        ref_d = refs[ridx][2][: min(32, mt)]
        tok_ok = (toks == ref_t)
        lp_ok = (digs[: len(ref_d)] == ref_d)
        ok &= tok_ok and lp_ok
        print(
            f"  uid={uid} (prompt {ridx}, {len(toks)}/{mt} tok): exact_tok={tok_ok} exact_lp={lp_ok}"
        )
        if not tok_ok or not lp_ok:
            dt = next((j for j, (a, b) in enumerate(zip(toks, ref_t)) if a != b), None)
            dl = next((j for j, (a, b) in enumerate(zip(digs, ref_d)) if a != b), None)
            print(f"    first_tok_diff={dt} first_lp_diff={dl}")

    print("SUMMARY:", "ALL PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
