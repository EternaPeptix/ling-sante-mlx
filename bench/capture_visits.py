"""Capture real single-stream layer visits & head snapshots for unit exactness testing.

Runs 4 prompts x 3 decode steps of the stock B=1 model, hooking representative layers [0, 4, 7, 64, 71]
plus final_norm/lm_head and saving exact inputs, caches, and outputs to ~/science/runs/visits/.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(_root / "vendor"), str(_root)]

import mlx.core as mx  # noqa: E402
from mlx_lm import load  # noqa: E402
from mlx_lm.models.bailing_moe_v3.layers import BailingDecoderLayer  # noqa: E402
from mlx_lm.models.bailing_moe_v3.model import Model as BailingCausalModel  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache  # noqa: E402
from mlx_lm.utils import maybe_set_recommended_wired_limit  # noqa: E402

from bench.exact import DEFAULT_MODEL, PROMPTS, prompt_ids  # noqa: E402

LAYERS = [0, 4, 7, 64, 71]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default=os.path.expanduser("~/science/runs/visits"))
    ap.add_argument("--steps", type=int, default=3)
    a = ap.parse_args()

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    maybe_set_recommended_wired_limit()
    model, tok = load(a.model)
    mx.eval(model.parameters())
    print(f"loaded in {time.perf_counter()-t0:.1f}s", flush=True)

    layers = model.model.layers
    id_to_li = {id(layers[L]): L for L in LAYERS}
    captured = {L: [] for L in LAYERS}
    head_caps = []
    rec_flag = [False]
    cur_meta = [0, 0]

    orig_layer_call = BailingDecoderLayer.__call__
    orig_model_call = BailingCausalModel.__call__

    def hooked_layer_call(self, x, mask=None, cache=None):
        li = id_to_li.get(id(self))
        if not rec_flag[0] or li is None:
            return orig_layer_call(self, x, mask=mask, cache=cache)

        snap = {"x": mx.array(x)}
        if mask is not None:
            snap["mask"] = mx.array(mask)
        if self.is_linear:
            for j in range(4):
                if cache[j] is not None:
                    snap[f"cs{j}"] = mx.array(cache[j])
        else:
            k_prev, v_prev = cache.keys, cache.values
            snap["kv_offset"] = mx.array([cache.offset], dtype=mx.int32)
            if k_prev is not None and cache.offset > 0:
                snap["k_prev"] = mx.array(k_prev[..., : cache.offset, :])
                snap["v_prev"] = mx.array(v_prev[..., : cache.offset, :])

        h_in = self.input_layernorm(x)
        attn_out = self.attention(h_in, mask, cache)
        h_post = x + attn_out
        mlp_in = self.post_attention_layernorm(h_post)
        if li >= model.args.first_k_dense_replace:
            r_logits = mlp_in.astype(mx.float32) @ self.mlp.gate.weight.astype(mx.float32).T
            r_inds, r_wts = self.mlp.gate(mlp_in)
            routed_raw = self.mlp.switch_mlp(mlp_in, r_inds)
            routed_out = (routed_raw * r_wts[..., None]).sum(axis=-2).astype(mlp_in.dtype)
            shared_out = self.mlp.shared_experts(mlp_in)
            mlp_out = routed_out + shared_out
            snap["r_logits"] = r_logits
            snap["r_inds"] = r_inds
            snap["r_wts"] = r_wts
            snap["routed_out"] = routed_out
            snap["shared_out"] = shared_out
        else:
            mlp_out = self.mlp(mlp_in)
        out = h_post + mlp_out
        snap["h_in"] = h_in
        snap["attn_out"] = attn_out
        snap["h_post"] = h_post
        snap["mlp_in"] = mlp_in
        snap["mlp_out"] = mlp_out
        snap["out"] = out
        if self.is_linear:
            for j in range(4):
                snap[f"ns{j}"] = mx.array(cache[j])
        else:
            snap["k_new"] = mx.array(cache.keys[..., : cache.offset, :])
            snap["v_new"] = mx.array(cache.values[..., : cache.offset, :])
        mx.eval(list(snap.values()))
        captured[li].append((cur_meta[0], cur_meta[1], snap))
        return out

    def hooked_model_call(self, inputs, cache=None):
        if not rec_flag[0] or id(self) != id(model):
            return orig_model_call(self, inputs, cache=cache)
        hidden = self.model(inputs, cache)
        logits = self.lm_head(hidden)
        snap = {"hidden": mx.array(hidden), "logits": mx.array(logits)}
        mx.eval(list(snap.values()))
        head_caps.append((cur_meta[0], cur_meta[1], snap))
        return logits

    BailingDecoderLayer.__call__ = hooked_layer_call
    BailingCausalModel.__call__ = hooked_model_call
    try:
        for pi, (pname, text) in enumerate(PROMPTS):
            ids = prompt_ids(tok, text)
            cache = make_prompt_cache(model)
            p = mx.array(ids)
            while p.size > 1:
                n = min(2048, p.size - 1)
                model(p[:n][None], cache=cache)
                mx.eval([c.state for c in cache])
                p = p[n:]
            y = p[None]
            rec_flag[0] = True
            for si in range(a.steps):
                cur_meta[0], cur_meta[1] = pi, si
                lg = model(y, cache=cache)[:, -1, :]
                lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
                y = mx.argmax(lp, axis=-1)[None]
                mx.eval(y)
            rec_flag[0] = False
            print(f"captured prompt {pi} ({pname}), {a.steps} steps", flush=True)
    finally:
        BailingDecoderLayer.__call__ = orig_layer_call
        BailingCausalModel.__call__ = orig_model_call

    index = []
    for L in LAYERS:
        for pi, si, snap in captured[L]:
            fn = f"L{L:02d}_p{pi}_s{si}.safetensors"
            mx.save_safetensors(str(out_dir / fn), snap)
            index.append({"layer": L, "prompt": pi, "step": si, "file": fn, "linear": bool(layers[L].is_linear)})
    for pi, si, snap in head_caps:
        fn = f"head_p{pi}_s{si}.safetensors"
        mx.save_safetensors(str(out_dir / fn), snap)
        index.append({"layer": "head", "prompt": pi, "step": si, "file": fn})

    (out_dir / "index.json").write_text(json.dumps({"layers": LAYERS, "steps": a.steps, "entries": index}, indent=1))
    print(f"saved {len(index)} visit files to {out_dir}")


if __name__ == "__main__":
    main()
