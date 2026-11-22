"""Exit 0 iff every tensor of the given layers (comma list) is fully present in the source shards. Prints what is missing."""
import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent)]
from convert import Source, layer_ready  # noqa: E402

src_dir = Path(sys.argv[1])
layers = [int(x) for x in sys.argv[2].split(",")]
cfg = json.loads((src_dir / "config.json").read_text())
src = Source(src_dir)
ok = True
for li in layers:
    r = layer_ready(src, li, cfg)
    pre = f"model.layers.{li}."
    names = [k for k in src.index if k.startswith(pre)]
    miss = sum(1 for k in names if not src.available(k))
    print(f"layer {li}: tensors seen {len(names)} missing/partial {miss} ready {r}")
    ok &= r
print("misc embeddings:", src.available("model.word_embeddings.weight"), "lm_head:", src.available("lm_head.weight"))
sys.exit(0 if ok else 1)
