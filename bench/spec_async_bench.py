"""Step 3.2 benchmark: single-sync async MTP speculative decoding across C=1..4, B<=24."""
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
from mlx_lm.models.bailing_moe_v3.mtp import BailingMTPLayer  # noqa: E402
from mlx_lm.models.bailing_moe_v3.spec_engine import get_lingsante_mtp, run_mtp_spec_greedy  # noqa: E402
from mlx_lm.utils import maybe_set_recommended_wired_limit  # noqa: E402

from bench.exact import DEFAULT_BASE, DEFAULT_MODEL, PROMPTS, lp_digest  # noqa: E402


def verify_exact(res: dict, gold_entries: list[dict], max_tokens: int):
    td, ld = 0, 0
    for c, entry in enumerate(gold_entries):
        emitted = res["emitted"][c][:max_tokens]
        gtoks = entry["tokens"][:max_tokens]
        for i, (g, r) in enumerate(zip(emitted, gtoks)):
            if g != r:
                td += 1
        gdigs = entry["logprob_sha256_32"]
        for i, lpa in enumerate(res["lp_arrays"][c][: len(gdigs)]):
            if lp_digest(lpa) != gdigs[i]:
                ld += 1
    return td, ld


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser(DEFAULT_MODEL))
    ap.add_argument("--base", default=os.path.expanduser(DEFAULT_BASE))
    ap.add_argument("--out", default=os.path.expanduser("~/science/runs/spec_async_phase3.json"))
    ap.add_argument("--tokens", type=int, default=128)
    a = ap.parse_args()

    maybe_set_recommended_wired_limit()
    base = json.loads(Path(a.base).read_text())
    prompts_data = {nm: base["prompts"][nm] for nm, _ in PROMPTS}
    prompt_names = [nm for nm, _ in PROMPTS]

    t0 = time.perf_counter()
    model, _ = load(a.model)
    mtp = get_lingsante_mtp(model) or BailingMTPLayer.from_target(model, a.model, with_q4_head=True)
    mx.eval(model.parameters())
    print(f"Loaded target + MTP + q4_head in {time.perf_counter()-t0:.1f}s ({mx.get_active_memory()/1e9:.1f} GB)", flush=True)

    # Warmup compile
    _ = run_mtp_spec_greedy(model, mtp, [prompts_data["chemistry_kmno4"]["ids"]], K=3, max_tokens=16, use_q4_head=True)

    out = {"c1": [], "c1_256": [], "batched": []}

    print("\n=== Async Single-Sync C=1 MTP (4 prompts, exact vs baseline.json) ===", flush=True)
    for K in (2, 3, 4, 5):
        rows = []
        for nm in prompt_names:
            r = run_mtp_spec_greedy(model, mtp, [prompts_data[nm]["ids"]], K=K, max_tokens=a.tokens, use_q4_head=True)
            td, ld = verify_exact(r, [prompts_data[nm]], a.tokens)
            rows.append({
                "prompt": nm,
                "tok_s": round(r["agg_tok_s"], 2),
                "E_acc": round(r["E_acc"], 3),
                "alphas": r["alphas"],
                "step_ms": round(r["wall_s"] * 1e3 / r["steps"], 2),
                "td": td,
                "ld": ld,
                "peak_gb": r["peak_gb"],
            })
        med = round(statistics.median(x["tok_s"] for x in rows), 2)
        eacc = round(sum(x["E_acc"] for x in rows) / len(rows), 3)
        ms = round(statistics.median(x["step_ms"] for x in rows), 2)
        ok = all(x["td"] == 0 and x["ld"] == 0 for x in rows)
        print(f"  C=1 K={K}: med={med:6.2f} tok/s  E_acc={eacc:.3f}  cycle={ms:6.2f}ms  exact={ok}", flush=True)
        out["c1"].append({"C": 1, "K": K, "med_tok_s": med, "E_acc": eacc, "cycle_ms": ms, "exact": ok, "rows": rows})

    print("\n=== Async Single-Sync C=1 Longer Generation (256 tokens) ===", flush=True)
    for K in (3, 4):
        rows = []
        for nm in prompt_names:
            r = run_mtp_spec_greedy(model, mtp, [prompts_data[nm]["ids"]], K=K, max_tokens=256, use_q4_head=True)
            td, ld = verify_exact(r, [prompts_data[nm]], 128)
            rows.append({"prompt": nm, "tok_s": round(r["agg_tok_s"], 2), "E_acc": round(r["E_acc"], 3), "td": td, "ld": ld})
        med = round(statistics.median(x["tok_s"] for x in rows), 2)
        eacc = round(sum(x["E_acc"] for x in rows) / len(rows), 3)
        ok = all(x["td"] == 0 and x["ld"] == 0 for x in rows)
        print(f"  C=1 256tok K={K}: med={med:6.2f} tok/s  E_acc={eacc:.3f}  exact={ok}", flush=True)
        out["c1_256"].append({"C": 1, "K": K, "med_tok_s": med, "E_acc": eacc, "exact": ok, "rows": rows})

    print("\n=== Async Single-Sync Batched C=2..4 + MTP ===", flush=True)
    for C in (2, 3, 4):
        for K in (2, 3, 4):
            if C * (K + 1) > 20:
                continue
            pids = [prompts_data[nm]["ids"] for nm in prompt_names[:C]]
            golds = [prompts_data[nm] for nm in prompt_names[:C]]
            # Run 1 warmup on short tokens if shape might be fresh, then timed 128-tok run
            _ = run_mtp_spec_greedy(model, mtp, pids, K=K, max_tokens=16, use_q4_head=True, capture_lp_steps=0)
            r = run_mtp_spec_greedy(model, mtp, pids, K=K, max_tokens=a.tokens, use_q4_head=True)
            td, ld = verify_exact(r, golds, a.tokens)
            rec = {
                "C": C,
                "K": K,
                "B": C * (K + 1),
                "agg_tok_s": round(r["agg_tok_s"], 2),
                "steady_tok_s": round(r.get("steady_tok_s", r["agg_tok_s"]), 2),
                "per_stream_tok_s": round(r["per_stream_tok_s"], 2),
                "E_acc": round(r["E_acc"], 3),
                "cycle_ms": round(r["wall_s"] * 1e3 / r["steps"], 2),
                "alphas": r["alphas"],
                "td": td,
                "ld": ld,
                "exact": td == 0 and ld == 0,
                "peak_gb": r["peak_gb"],
            }
            print(
                f"  C={C} K={K} (B={rec['B']:2d}): agg={rec['agg_tok_s']:6.2f} tok/s steady={rec['steady_tok_s']:6.2f} ({rec['per_stream_tok_s']:5.2f}/s)  "
                f"E_acc={rec['E_acc']:.3f}  cycle={rec['cycle_ms']:6.2f}ms  exact={rec['exact']} (td={td},ld={ld}) peak={rec['peak_gb']}GB",
                flush=True,
            )
            out["batched"].append(rec)

    # Also test C=4 K=2 at 256 tokens (verifying exactness on first 128 tokens / 32 digests)
    pids4 = [prompts_data[nm]["ids"] for nm in prompt_names[:4]]
    golds4 = [prompts_data[nm] for nm in prompt_names[:4]]
    r256 = run_mtp_spec_greedy(model, mtp, pids4, K=2, max_tokens=256, use_q4_head=True)
    td256, ld256 = verify_exact(r256, golds4, 128)
    rec256 = {
        "C": 4,
        "K": 2,
        "tokens": 256,
        "B": 12,
        "agg_tok_s": round(r256["agg_tok_s"], 2),
        "steady_tok_s": round(r256.get("steady_tok_s", r256["agg_tok_s"]), 2),
        "per_stream_tok_s": round(r256["per_stream_tok_s"], 2),
        "E_acc": round(r256["E_acc"], 3),
        "cycle_ms": round(r256["wall_s"] * 1e3 / r256["steps"], 2),
        "td": td256,
        "ld": ld256,
        "exact": td256 == 0 and ld256 == 0,
        "peak_gb": r256["peak_gb"],
    }
    print(
        f"  C=4 K=2 (256tok): agg={rec256['agg_tok_s']:6.2f} tok/s steady={rec256['steady_tok_s']:6.2f}  "
        f"E_acc={rec256['E_acc']:.3f}  cycle={rec256['cycle_ms']:6.2f}ms  exact={rec256['exact']} (td={td256},ld={ld256})",
        flush=True,
    )
    out["c4_256"] = rec256

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\nSaved {a.out}", flush=True)
    all_ok = all(x["exact"] for x in out["c1"] + out["c1_256"] + out["batched"] + [out["c4_256"]])
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
