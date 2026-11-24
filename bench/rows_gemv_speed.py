"""Queued microbenchmark for rows/gemv.py and rows/router.py vs stock MLX on real L4, L0, L7, head weights."""
import statistics as st
import sys
import time
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(_root / "vendor"), str(_root)]

import mlx.core as mx  # noqa: E402
from mlx_lm.layers.common import swiglu  # noqa: E402
from mlx_lm.models.bailing_moe_v3.rows.gemv import (  # noqa: E402
    gemv_rows,
    gemv_rows_multi,
    gemv_rows_res,
    gemv_swiglu_rows,
)
from mlx_lm.models.bailing_moe_v3.rows.router import gate_rows  # noqa: E402

from bench._layer_loader import load_head, load_layer  # noqa: E402


def bench_queued(fn, reps=40, batch=16, warm=5):
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
    l0, _ = load_layer(0)
    l4, _ = load_layer(4)
    _, lm_head, _ = load_head()

    a4 = l4.attention
    se4 = l4.mlp.shared_experts
    g4 = l4.mlp.gate

    for B in (1, 2, 3, 4):
        x = mx.random.normal((B, 4096)).astype(mx.bfloat16)
        x3 = x[:, None, :]
        x8 = mx.random.normal((B, 8192)).astype(mx.bfloat16)
        x8_3 = x8[:, None, :]
        mx.eval(x, x3, x8, x8_3)

        # 1. KDA qkvfgb
        projs = [a4.q_proj, a4.k_proj, a4.v_proj, a4.f_proj, a4.g_proj, a4.b_proj]
        ws = [p.weight for p in projs]
        t_ref = bench_queued(lambda: [p(x3) for p in projs])
        t_row = bench_queued(lambda: gemv_rows_multi(x, ws))
        print(f"B={B} KDA qkvfgb   : ref={t_ref:6.1f} us  rows={t_row:6.1f} us  speedup={t_ref/t_row:.2f}x")

        # 2. KDA o_proj + res
        t_ref = bench_queued(lambda: a4.o_proj(x8_3) + x3)
        t_row = bench_queued(lambda: gemv_rows_res(x8, a4.o_proj.weight, x))
        print(f"B={B} KDA o_proj+r : ref={t_ref:6.1f} us  rows={t_row:6.1f} us  speedup={t_ref/t_row:.2f}x")

        # 3. Shared expert gate+up+swiglu + down
        t_ref = bench_queued(lambda: se4.down_proj(swiglu(se4.gate_proj(x3), se4.up_proj(x3))))
        t_row = bench_queued(lambda: gemv_rows(gemv_swiglu_rows(x, se4.gate_proj.weight, se4.up_proj.weight), se4.down_proj.weight))
        print(f"B={B} Shared exp   : ref={t_ref:6.1f} us  rows={t_row:6.1f} us  speedup={t_ref/t_row:.2f}x")

        # 4. Dense MLP (L0)
        t_ref = bench_queued(lambda: l0.mlp.down_proj(swiglu(l0.mlp.gate_proj(x3), l0.mlp.up_proj(x3))) + x3)
        t_row = bench_queued(lambda: gemv_rows_res(gemv_swiglu_rows(x, l0.mlp.gate_proj.weight, l0.mlp.up_proj.weight), l0.mlp.down_proj.weight, x))
        print(f"B={B} Dense MLP    : ref={t_ref:6.1f} us  rows={t_row:6.1f} us  speedup={t_ref/t_row:.2f}x")

        # 5. Router
        t_ref = bench_queued(lambda: g4(x3))
        t_row = bench_queued(lambda: gate_rows(x, g4.weight, g4.expert_bias))
        print(f"B={B} Router       : ref={t_ref:6.1f} us  rows={t_row:6.1f} us  speedup={t_ref/t_row:.2f}x")

        # 6. lm_head
        t_ref = bench_queued(lambda: lm_head(x3), reps=20, batch=8)
        t_row = bench_queued(lambda: gemv_rows(x, lm_head.weight), reps=20, batch=8)
        print(f"B={B} lm_head      : ref={t_ref:6.1f} us  rows={t_row:6.1f} us  speedup={t_ref/t_row:.2f}x")
        print()


if __name__ == "__main__":
    main()
