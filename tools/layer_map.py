"""Summarise which source shards hold each layer, from the worker-*.json weight maps (read-only).

usage: layer_map.py <src_dir>
"""
import collections
import json
import re
import sys
from pathlib import Path

src = Path(sys.argv[1])
wm = {}
for w in sorted(src.glob("worker-*.json")):
    wm.update(json.loads(w.read_text())["weight_map"])
print("tensors", len(wm))
layers = collections.defaultdict(lambda: collections.defaultdict(int))
other = collections.defaultdict(int)
for k, f in wm.items():
    m = re.match(r"model\.layers\.(\d+)\.(.*)", k)
    if m:
        kind = "experts" if ".experts." in k else "dense"
        layers[int(m.group(1))][(kind, f)] += 1
    else:
        other[(k, f)] += 1
for k, f in sorted(other):
    print("OTHER", k, f)


def done(f):
    p = src / f
    return p.exists()


for li in sorted(layers):
    d = collections.defaultdict(set)
    for (kind, f), n in layers[li].items():
        d[kind].add(f)
    if li in (0, 1, 4, 7, 8, 15, 71, 72):
        print(li, {k: sorted(v) for k, v in d.items()})
