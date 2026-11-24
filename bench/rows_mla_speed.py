"""Queued microbenchmark for Wave 4 MLA rows vs stock BailingMLA at S=82 and S=512."""
import statistics as st
import sys
import time
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(_root / "vendor"), str(_root)]

import mlx.core as mx  # noqa: E402
from mlx_lm.models.bailing_moe_v3.rows.mla import mla_layer_rows  # noqa: E402
from mlx_lm.models.bailing_moe_v3.rows.norm import rms_norm_rows  # noqa: E402
from mlx_lm.models.cache import BatchKVCache, KVCache  # noqa: E402

from bench._layer_loader import load_layer  # noqa: E402


def bench_queued(fn, reps=30, batch=12, warm=5):
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
    l7, _ = load_layer(7)
    attn = l7.attention
    mx.random.seed(11)

    for S in (82, 512):
        print(f"=== L7 MLA B=1..4 (S={S}) ===")
        for B in (1, 2, 3, 4):
            x = mx.random.normal((B, 4096)).astype(mx.bfloat16) * 0.25
            x3 = x[:, None, :]
            kp = mx.random.normal((B, 1, S, 512)).astype(mx.bfloat16) * 0.1
            vp = mx.random.normal((B, 1, S, 64)).astype(mx.bfloat16) * 0.1
            mx.eval(x, x3, kp, vp)

            if B == 1:
                c_ref = KVCache()
                c_ref.keys, c_ref.values, c_ref.offset = mx.array(kp), mx.array(vp), S
                c_row = KVCache()
                c_row.keys, c_row.values, c_row.offset = mx.array(kp), mx.array(vp), S
            else:
                c_ref = BatchKVCache([0] * B)
                c_ref.keys, c_ref.values = mx.array(kp), mx.array(vp)
                c_ref.offset = mx.array([S] * B)
                c_ref._idx = S
                c_row = BatchKVCache([0] * B)
                c_row.keys, c_row.values = mx.array(kp), mx.array(vp)
                c_row.offset = mx.array([S] * B)
                c_row._idx = S

            buf_k = mx.zeros((B, 1, S + 32, 512), dtype=mx.bfloat16)
            buf_v = mx.zeros((B, 1, S + 32, 64), dtype=mx.bfloat16)
            buf_k[:, :, :S, :] = kp
            buf_v[:, :, :S, :] = vp
            mx.eval(buf_k, buf_v)

            def reset_cache(c):
                c.keys, c.values = buf_k, buf_v
                if B == 1:
                    c.offset = S
                else:
                    c.offset = mx.array([S] * B)
                    c._idx = S

            def run_ref():
                reset_cache(c_ref)
                mask = c_ref.make_mask(N=1, return_array=True, window_size=None)
                return x3 + attn(l7.input_layernorm(x3), mask=mask, cache=c_ref), c_ref.keys

            def run_row():
                reset_cache(c_row)
                hin = rms_norm_rows(x, l7.input_layernorm.weight, l7.input_layernorm.eps)
                return mla_layer_rows(hin, x, attn, c_row), c_row.keys

            t_ref = bench_queued(run_ref)
            t_row = bench_queued(run_row)
            print(f"  B={B}: stock={t_ref:6.1f} us  rows={t_row:6.1f} us  speedup={t_ref/t_row:.2f}x")


if __name__ == "__main__":
    main()
