"""Print safetensors headers (dtype/shape) for given files. Read-only; never loads tensor data.

usage: st_header.py [-n N] [-g SUBSTR] file...
"""
import json
import struct
import sys


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), n + 8


if __name__ == "__main__":
    args = sys.argv[1:]
    limit, grep = 40, None
    while args and args[0] in ("-n", "-g"):
        if args[0] == "-n":
            limit = int(args[1])
        else:
            grep = args[1]
        args = args[2:]
    for p in args:
        h, off = header(p)
        meta = h.pop("__metadata__", None)
        print("FILE", p, "tensors", len(h), "meta", meta)
        items = [(k, v) for k, v in h.items() if grep is None or grep in k]
        for k, v in items[:limit]:
            print(" ", k, v["dtype"], v["shape"])
