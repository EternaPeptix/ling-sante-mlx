"""Queued microbenchmark for rows/router.py vs stock BailingGate on L4 & L64."""
import statistics as st
import sys
import time
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(_root / "vendor"), str(_root)]

import mlx.core as mx  # noqa: E402
from mlx_lm.models.bailing_moe_v3.rows.router import gate_rows, group_router_rows  # noqa: E402

from bench._layer_loader import load_layer  # noqa: E402


def bench_queued(fn, reps=60, batch=32, warm=8):
    for _ in range(warm):
        mx.eval(fn())
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        outs = [fn() for _ in range(batch)]
        mx.eval(outs)
        ts.append((time.perf_counter() - t0) / batch)
    return 1e6 * st.median(ts)


def main():
    for L in (4, 64):
        layer, _ = load_layer(L)
        gate = layer.mlp.gate
        print(f"=== L{L} BailingGate (w {gate.weight.shape} {gate.weight.dtype}) ===")
        for B in (1, 2, 3, 4):
            x = mx.random.normal((B, 4096)).astype(mx.bfloat16)
            x3 = x[:, None, :]
            logits = mx.random.normal((B, 512)).astype(mx.float32)
            mx.eval(x, x3, logits)

            t_ref = bench_queued(lambda: gate(x3))
            t_row = bench_queued(
                lambda: gate_rows(
                    x, gate.weight, gate.expert_bias,
                    top_k=gate.top_k, n_group=gate.n_group, topk_group=gate.topk_group,
                    routed_scaling_factor=gate.routed_scaling_factor,
                )
            )
            t_sel = bench_queued(
                lambda: group_router_rows(
                    logits, gate.expert_bias,
                    top_k=gate.top_k, n_group=gate.n_group, topk_group=gate.topk_group,
                    routed_scaling_factor=gate.routed_scaling_factor,
                )
            )
            print(f"  B={B}: stock={t_ref:6.1f} us  gate_rows={t_row:6.1f} us ({t_ref/t_row:.2f}x)  select_only={t_sel:5.1f} us")


if __name__ == "__main__":
    main()
