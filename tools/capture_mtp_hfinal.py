"""Capture target h_final (inner.norm(h)) for the 200 disjoint train sequences in lingsante_dflash_train."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(_root / "vendor"), str(_root)]

import mlx_lm  # noqa: E402,F401
import mlx.core as mx  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache  # noqa: E402
from mlx_lm.utils import load_model, maybe_set_recommended_wired_limit  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/science/models/lingsante-mlx-mxfp4-bf16"))
    ap.add_argument("--data-dir", default=os.path.expanduser("~/science/data/lingsante_dflash_train"))
    ap.add_argument("--out", default=os.path.expanduser("~/science/data/lingsante_dflash_train/train_hfinal.safetensors"))
    ap.add_argument("--step", type=int, default=512)
    a = ap.parse_args()

    data_dir = Path(a.data_dir).expanduser()
    out_path = Path(a.out).expanduser()
    idx_data = json.loads((data_dir / "index.json").read_text())
    train_entries = [e for e in idx_data["entries"] if e["split"] == "train"]
    req_keys = {e["key"] for e in train_entries}

    if out_path.exists():
        existing = mx.load(str(out_path))
        if req_keys.issubset(existing.keys()):
            print(f"Found complete {out_path} ({len(req_keys)} train keys); skipping capture.", flush=True)
            return 0

    shard_cache: dict[str, dict] = {}
    seqs = []
    for e in train_entries:
        sh = e["shard"]
        if sh not in shard_cache:
            shard_cache[sh] = mx.load(str(data_dir / "shards" / sh))
        ids_mx = shard_cache[sh][f"{e['key']}.ids"]
        mx.eval(ids_mx)
        seqs.append({
            "key": e["key"],
            "prompt_len": int(e["prompt_len"]),
            "total_len": int(e["total_len"]),
            "ids": ids_mx.tolist(),
            "ids_mx": ids_mx,
        })
    del shard_cache
    mx.clear_cache()

    maybe_set_recommended_wired_limit()
    t0 = time.perf_counter()
    model, _ = load_model(Path(a.model).expanduser())
    mx.eval(model.parameters())
    print(f"Loaded target (fp8={getattr(model, '_lingsante_fp8', False)}) in {time.perf_counter()-t0:.1f}s ({mx.get_active_memory()/1e9:.1f} GB)", flush=True)

    tensors = {}
    tot_pos, tot_match = 0, 0
    t_run = time.perf_counter()
    for i, seq in enumerate(seqs):
        ids_mx = seq["ids_mx"]
        ids = seq["ids"]
        plen = seq["prompt_len"]
        T = seq["total_len"]
        cache = make_prompt_cache(model)
        h_chunks = []
        pos = 0
        while pos < T:
            n = min(a.step, T - pos)
            hc = model.model(ids_mx[pos : pos + n][None], cache=cache)
            mx.eval(hc, [c.state for c in cache])
            h_chunks.append(hc[0].astype(mx.bfloat16))
            pos += n
        hf = mx.concatenate(h_chunks, axis=0) if len(h_chunks) > 1 else h_chunks[0]
        lg = model.lm_head(hf[None])[0]
        preds = mx.argmax(lg[plen - 1 : T - 1], axis=-1)
        mx.eval(hf, preds)
        gold = ids[plen:T]
        hits = sum(1 for p_t, g_t in zip(preds.tolist(), gold) if p_t == g_t)
        tot_pos += len(gold)
        tot_match += hits
        tensors[seq["key"]] = hf
        del cache
        if (i + 1) % 25 == 0 or (i + 1) == len(seqs):
            mx.clear_cache()
            print(f"  Captured {i+1}/{len(seqs)} ({time.perf_counter()-t_run:.1f}s, cum_top1={tot_match}/{tot_pos}={tot_match/max(1,tot_pos):.4f})", flush=True)

    rate = tot_match / max(1, tot_pos)
    print(f"Global train h_final alignment: {tot_match}/{tot_pos} ({rate:.4f})", flush=True)
    assert rate >= 0.95, f"Train h_final alignment failed: {rate:.4f} < 0.95"
    mx.save_safetensors(str(out_path), tensors)
    print(f"Saved {out_path} ({out_path.stat().st_size/1e6:.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
