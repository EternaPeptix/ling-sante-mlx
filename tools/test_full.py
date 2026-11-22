"""Whole-model logits: mlx_lm.load_model(converted dir) vs the fp32 reference chain (toy checkpoints, or the real
model for a few tokens). usage: test_full.py SRC DST [T=24] [--fp32]"""
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
sys.path[:0] = [str(_here), str(_here.parent / "vendor")]
import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import ref_layer as R  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache  # noqa: E402
from mlx_lm.utils import load_model  # noqa: E402

src = R.Src(sys.argv[1])
src.round_bf16 = "--fp32" in sys.argv
T = int(sys.argv[3]) if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else 24
model, cfg = load_model(Path(sys.argv[2]))
if "--fp32" in sys.argv:
    model.set_dtype(mx.float32)
g = torch.Generator().manual_seed(3)
ids = torch.randint(2, src.cfg["vocab_size"], (T,), generator=g)
with torch.no_grad():
    x = src.t("model.word_embeddings.weight")[ids]
    for li in range(src.cfg["num_hidden_layers"]):
        x = R.decoder_layer(x, src, li)
    ref = R.rms(x, src.t("model.norm.weight")) @ src.t("lm_head.weight").T
out = model(mx.array(ids.numpy())[None]).astype(mx.float32)
mx.eval(out)
a, b = np.array(out)[0], ref.numpy()
print("logits relL2", np.linalg.norm(a - b) / np.linalg.norm(b), "argmax agreement", float(np.mean(a.argmax(-1) == b.argmax(-1))))
cache = make_prompt_cache(model)
o1 = model(mx.array(ids.numpy())[None, : T - 4], cache=cache)
o2 = [model(mx.array(ids.numpy())[None, t : t + 1], cache=cache) for t in range(T - 4, T)]
cc = np.array(mx.concatenate([o1] + o2, axis=1).astype(mx.float32))[0]
print("cached-path logits relL2 vs reference", np.linalg.norm(cc - b) / np.linalg.norm(b))
