"""Per-layer check: MLX (converted weights, bf16) vs independent fp32 torch reference (source weights).

usage: test_layers.py SRC DST L[,L...] [T=48] [--text FILE]
For every layer prints: attention-block error, MLP-block error, whole-layer error for
  (a) one-shot prefill of T tokens, (b) prefill T-8 then 8 single-token decode steps (cache path),
and for MoE layers the top-k expert agreement with the reference router.
"""
import sys

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch

_here = __file__.rsplit("/", 1)[0]
sys.path[:0] = [_here, _here + "/../vendor"]
import ref_layer as R  # noqa: E402
from mlx_lm.layers.attention import create_attention_mask  # noqa: E402
from mlx_lm.models.bailing_moe_v3 import Model, ModelArgs  # noqa: E402
from mlx_lm.utils import infer_quant_config  # noqa: E402

src_dir, dst_dir = sys.argv[1], sys.argv[2]
layers = [int(x) for x in sys.argv[3].split(",")]
T = int(sys.argv[4]) if len(sys.argv) > 4 and not sys.argv[4].startswith("--") else 48
text_file = sys.argv[sys.argv.index("--text") + 1] if "--text" in sys.argv else None

src = R.Src(src_dir)
src.round_bf16 = "--refbf16" in sys.argv
cfg = src.cfg
args = ModelArgs.from_dict(cfg)
model = Model(args)
w = {}
for L in layers:
    w.update(mx.load(f"{dst_dir}/model-layer-{L:03d}.safetensors"))
w = model.sanitize(w)


def pred(p, m):
    if not hasattr(m, "to_quantized") or f"{p}.scales" not in w:
        return False
    return infer_quant_config(p, m, w)


nn.quantize(model, group_size=32, bits=4, mode="mxfp4", class_predicate=pred)
model.load_weights(list(w.items()), strict=False)
FP32 = "--fp32" in sys.argv
if FP32:
    model.set_dtype(mx.float32)
mx.eval([model.model.layers[L].parameters() for L in layers])

if text_file:
    from mlx_lm.utils import load_tokenizer

    tok = load_tokenizer(__import__("pathlib").Path(dst_dir))
    ids = tok.encode(open(text_file).read())[:T]
    T = len(ids)
else:
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(2, cfg["vocab_size"], (T,), generator=g).tolist()
emb = src.t("model.word_embeddings.weight")[torch.tensor(ids)]


def err(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    d = np.abs(a - b)
    return d.max() / (np.abs(b).max() + 1e-30), np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30)


def fmt(e):
    return f"maxabs/maxref {e[0]:.2e}  relL2 {e[1]:.2e}"


with torch.no_grad():
    for L in layers:
        x = emb if L == 0 else emb.roll(L, 0) * 1.0  # distinct inputs per layer, same real-embedding statistics
        ref, parts = R.decoder_layer(x, src, L, return_parts=True)
        layer = model.model.layers[L]
        xm = mx.array(x.numpy()).astype(mx.float32 if FP32 else mx.bfloat16)[None]
        # (a) one-shot prefill
        cache = model.make_cache()[L]
        mask = None if layer.is_linear else create_attention_mask(xm, cache, return_array=True)
        out = layer(xm, mask=mask, cache=cache)
        # sub-blocks fed with the *reference* inputs so errors do not compound
        h_in = mx.array(R.rms(x, src.t(f"model.layers.{L}.input_layernorm.weight")).numpy()).astype(mx.float32 if FP32 else mx.bfloat16)[None]
        c2 = model.make_cache()[L]
        m2 = None if layer.is_linear else create_attention_mask(xm, c2, return_array=True)
        a_out = layer.attention(h_in, m2, c2)
        h_ref = mx.array(parts["h"].numpy()).astype(mx.float32 if FP32 else mx.bfloat16)[None]
        mlp_in = layer.post_attention_layernorm(h_ref)
        m_out = layer.mlp(mlp_in)
        mx.eval(out, a_out, m_out)
        print(f"layer {L} ({parts['kind']}, {'moe' if parts['top'] is not None else 'dense'})")
        print("  attn block  :", fmt(err(np.array(a_out.astype(mx.float32))[0], parts["attn"].numpy())))
        print("  mlp block   :", fmt(err(np.array(m_out.astype(mx.float32))[0], parts["mlp"].numpy())))
        print("  full layer  :", fmt(err(np.array(out.astype(mx.float32))[0], ref.numpy())))
        if parts["top"] is not None:
            idx, _ = layer.mlp.gate(mlp_in)
            idx = np.array(idx)[0]
            agree = np.mean([set(idx[t]) == set(parts["top"][t].tolist()) for t in range(T)])
            print(f"  router top-k set agreement: {agree:.3f}")
        # (b) cache path: prefill T-8, then 8 single steps
        cache = model.make_cache()[L]
        k0 = T - 8
        mask = None if layer.is_linear else create_attention_mask(xm[:, :k0], cache, return_array=True)
        outs = [layer(xm[:, :k0], mask=mask, cache=cache)]
        for t in range(k0, T):
            xt = xm[:, t : t + 1]
            mk = None if layer.is_linear else create_attention_mask(xt, cache, return_array=True)
            outs.append(layer(xt, mask=mk, cache=cache))
        o2 = mx.concatenate(outs, axis=1)
        mx.eval(o2)
        print("  cache path  :", fmt(err(np.array(o2.astype(mx.float32))[0], ref.numpy())), "(prefill T-8 + 8 decode steps)")
print("done")
