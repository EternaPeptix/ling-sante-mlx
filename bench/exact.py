"""Exactness reference capture & gate for LingSante C<=4 decode.

Usage:
  python bench/exact.py --capture [--model DIR] [--out FILE] [--tokens 128]
  python bench/exact.py --check CAND.json [--base BASE.json]
"""
import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(_root / "vendor"), str(_root)]

import mlx.core as mx  # noqa: E402
from mlx_lm import load  # noqa: E402
from mlx_lm.generate import generate_step  # noqa: E402
from mlx_lm.utils import maybe_set_recommended_wired_limit  # noqa: E402

DEFAULT_MODEL = os.path.expanduser("~/science/models/lingsante-mlx-mxfp4-bf16")
DEFAULT_BASE = os.path.expanduser("~/science/runs/baseline.json")

# 4 prompts of distinct token lengths (~49 to ~128 tok) so MLA KV lengths differ across streams.
PROMPTS = [
    (
        "medical_stemi",
        "A 58-year-old man presents with crushing substernal chest pain radiating to the left arm, "
        "diaphoresis, and ST elevation in leads II, III and aVF. What is the most likely diagnosis, "
        "which coronary artery is probably involved, and what are the first three management steps?",
    ),
    (
        "chemistry_kmno4",
        "Balance the reaction of potassium permanganate with hydrochloric acid that produces chlorine gas, "
        "and state the oxidation state changes of manganese and chlorine.",
    ),
    (
        "biology_lac",
        "Describe how the lac operon of E. coli responds when lactose is present and glucose is absent, "
        "naming the roles of the repressor, cAMP-CAP and allolactose.",
    ),
    (
        "medical_renal_pharm",
        "A 72-year-old woman with type 2 diabetes mellitus, stage 3b chronic kidney disease (eGFR 36 mL/min/1.73m2), "
        "heart failure with reduced ejection fraction (EF 30%), and hypertension currently takes metformin 1000 mg twice daily, "
        "lisinopril 20 mg daily, and furosemide 40 mg daily. She presents for routine follow-up with mild ankle edema, "
        "serum potassium 4.8 mEq/L, and HbA1c 7.9%. Which medication adjustments are indicated based on current guidelines, "
        "what monitoring is required after each change, and why should NSAIDs be strictly avoided?",
    ),
]


def prompt_ids(tok, text: str) -> list[int]:
    msgs = [{"role": "user", "content": text}]
    s = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    return tok.encode(s, add_special_tokens=False)


def lp_digest(lp: mx.array) -> str:
    # lp is 1D array of logprobs; hash raw underlying bits (uint16 for bf16, uint32 for f32)
    v = lp.view(mx.uint16) if lp.dtype == mx.bfloat16 else lp.view(mx.uint32)
    return hashlib.sha256(memoryview(v)).hexdigest()[:16]


def run_capture(model_path: str, out_path: str, n_tokens: int, warm: int = 4, log_steps: int = 32):
    t0 = time.perf_counter()
    maybe_set_recommended_wired_limit()
    model, tok = load(model_path)
    mx.eval(model.parameters())
    print(f"loaded in {time.perf_counter()-t0:.1f}s, active {mx.get_active_memory()/1e9:.1f} GB", flush=True)

    greedy = lambda lp: mx.argmax(lp, axis=-1)
    prompts_out = {}
    tps_list = []

    for name, text in PROMPTS:
        ids = prompt_ids(tok, text)
        toks, digests = [], []
        t_start = None
        for step_i, (t, lp) in enumerate(generate_step(mx.array(ids), model, max_tokens=n_tokens, sampler=greedy)):
            toks.append(int(t))
            if step_i < log_steps:
                digests.append(lp_digest(lp))
            if len(toks) == warm:
                t_start = time.perf_counter()
        dt = time.perf_counter() - t_start
        tps = (len(toks) - warm) / dt
        tps_list.append(tps)
        prompts_out[name] = {
            "prompt_len": len(ids),
            "ids": ids,
            "tokens": toks,
            "logprob_sha256_32": digests,
            "tok_s": round(tps, 3),
        }
        print(f"  {name:20s} prompt={len(ids):3d} tok  gen={len(toks)} tok  {tps:.2f} tok/s  d0={digests[0]}", flush=True)

    data = {
        "model": model_path,
        "n_tokens": n_tokens,
        "log_steps": log_steps,
        "tok_s_median": round(statistics.median(tps_list), 3),
        "prompts": prompts_out,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(data, indent=1))
    print(f"saved baseline to {out_path} (median {data['tok_s_median']:.2f} tok/s)")


def run_check(cand_path: str, base_path: str) -> int:
    b = json.loads(Path(base_path).read_text())
    c = json.loads(Path(cand_path).read_text())
    ok = True
    for name, bv in b["prompts"].items():
        cv = c.get("prompts", {}).get(name)
        if cv is None:
            print(f"MISSING {name}")
            ok = False
            continue
        tok_ok = bv["tokens"] == cv["tokens"]
        lp_ok = bv["logprob_sha256_32"] == cv.get("logprob_sha256_32")
        first_tok = next((i for i, (x, y) in enumerate(zip(bv["tokens"], cv["tokens"])) if x != y), None)
        first_lp = next(
            (i for i, (x, y) in enumerate(zip(bv["logprob_sha256_32"], cv.get("logprob_sha256_32", []))) if x != y),
            None,
        )
        ok &= tok_ok and lp_ok
        print(f"{name:20s} tokens={'EXACT' if tok_ok else f'DIFF@{first_tok}'} logprobs32={'EXACT' if lp_ok else f'DIFF@{first_lp}'}")
    print("GATE", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", action="store_true")
    ap.add_argument("--check", default="")
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default=DEFAULT_BASE)
    ap.add_argument("--tokens", type=int, default=128)
    a = ap.parse_args()
    if a.capture:
        run_capture(a.model, a.out, a.tokens)
    elif a.check:
        sys.exit(run_check(a.check, a.base))
    else:
        ap.print_help()
        sys.exit(2)


if __name__ == "__main__":
    main()
