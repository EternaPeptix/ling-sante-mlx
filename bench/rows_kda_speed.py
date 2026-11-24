"""Queued microbenchmark for fused Wave 3 KDA attention block vs stock BailingKDA on L4."""
import statistics as st
import sys
import time
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(_root / "vendor"), str(_root)]

import mlx.core as mx  # noqa: E402
from mlx_lm.models.bailing_moe_v3.rows.gemv import gemv_rows_multi, gemv_rows_res  # noqa: E402
from mlx_lm.models.bailing_moe_v3.rows.kda import kda_gdn_norm_gate_rows, kda_pre_rows  # noqa: E402
from mlx_lm.models.bailing_moe_v3.rows.norm import rms_norm_rows  # noqa: E402
from mlx_lm.models.cache import ArraysCache  # noqa: E402

from bench._layer_loader import load_layer, load_visits  # noqa: E402


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
    l4, _ = load_layer(4)
    attn = l4.attention
    visits = load_visits(4, limit=8)
    ws = [attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight, attn.f_proj.weight, attn.g_proj.weight, attn.b_proj.weight]

    for B in (1, 2, 3, 4):
        x = mx.concatenate([v["x"][0, 0:1] for v in visits[:B]], axis=0)
        x3 = x[:, None, :]
        cs0 = mx.concatenate([v["cs0"] for v in visits[:B]], axis=0)
        cs1 = mx.concatenate([v["cs1"] for v in visits[:B]], axis=0)
        cs2 = mx.concatenate([v["cs2"] for v in visits[:B]], axis=0)
        cs3 = mx.concatenate([v["cs3"] for v in visits[:B]], axis=0)
        c = ArraysCache(size=4)
        c[0], c[1], c[2], c[3] = mx.array(cs0), mx.array(cs1), mx.array(cs2), mx.array(cs3)
        mx.eval(x, x3, cs0, cs1, cs2, cs3)

        def stock_kda():
            h = l4.input_layernorm(x3)
            out = attn(h, mask=None, cache=c)
            ns = (c[0], c[1], c[2], c[3])
            # restore cache refs for repeated queued invocations
            c[0], c[1], c[2], c[3] = cs0, cs1, cs2, cs3
            c.offset = 0
            return (x3 + out,) + ns

        def rows_kda_fused():
            h = rms_norm_rows(x, l4.input_layernorm.weight, l4.input_layernorm.eps)
            q_r, k_r, v_r, f_r, g_r, b_r = gemv_rows_multi(h, ws)
            q, k, v, g, beta, ns0, ns1, ns2 = kda_pre_rows(
                q_r, k_r, v_r, f_r, b_r, cs0, cs1, cs2,
                attn.q_conv1d.weight, attn.k_conv1d.weight, attn.v_conv1d.weight,
                attn.A_log, attn.dt_bias, lower_bound=attn.lower_bound, eps=1e-6,
            )
            ov, ns3 = kda_gdn_norm_gate_rows(q, k, v, g, beta, cs3, g_r, attn.o_norm.weight, eps=attn.o_norm.eps, fused=True)
            hp = gemv_rows_res(ov, attn.o_proj.weight, x)
            return hp, ns0, ns1, ns2, ns3

        # Also isolate non-GEMV part (pre + gdn) vs stock non-GEMV
        h = rms_norm_rows(x, l4.input_layernorm.weight, l4.input_layernorm.eps)
        q_r, k_r, v_r, f_r, g_r, b_r = gemv_rows_multi(h, ws)
        q, k, v, g, beta, _, _, _ = kda_pre_rows(
            q_r, k_r, v_r, f_r, b_r, cs0, cs1, cs2,
            attn.q_conv1d.weight, attn.k_conv1d.weight, attn.v_conv1d.weight,
            attn.A_log, attn.dt_bias, lower_bound=attn.lower_bound, eps=1e-6,
        )
        mx.eval(h, q_r, k_r, v_r, f_r, g_r, b_r, q, k, v, g, beta)

        t_ref = bench_queued(stock_kda)
        t_row = bench_queued(rows_kda_fused)
        t_pre = bench_queued(lambda: kda_pre_rows(
            q_r, k_r, v_r, f_r, b_r, cs0, cs1, cs2,
            attn.q_conv1d.weight, attn.k_conv1d.weight, attn.v_conv1d.weight,
            attn.A_log, attn.dt_bias, lower_bound=attn.lower_bound, eps=1e-6,
        ))
        t_gdn_fused = bench_queued(lambda: kda_gdn_norm_gate_rows(q, k, v, g, beta, cs3, g_r, attn.o_norm.weight, eps=attn.o_norm.eps, fused=True))
        t_gdn_n2 = bench_queued(lambda: kda_gdn_norm_gate_rows(q, k, v, g, beta, cs3, g_r, attn.o_norm.weight, eps=attn.o_norm.eps, fused=False, nsg=2))
        t_gdn_n4 = bench_queued(lambda: kda_gdn_norm_gate_rows(q, k, v, g, beta, cs3, g_r, attn.o_norm.weight, eps=attn.o_norm.eps, fused=False, nsg=4))
        t_gdn_n8 = bench_queued(lambda: kda_gdn_norm_gate_rows(q, k, v, g, beta, cs3, g_r, attn.o_norm.weight, eps=attn.o_norm.eps, fused=False, nsg=8))
        print(
            f"B={B}: full stock={t_ref:6.1f} us  rows_fused={t_row:6.1f} us ({t_ref/t_row:.2f}x) | "
            f"pre={t_pre:4.1f} us  gdn[fused/n2/n4/n8]={t_gdn_fused:4.1f}/{t_gdn_n2:4.1f}/{t_gdn_n4:4.1f}/{t_gdn_n8:4.1f} us"
        )


if __name__ == "__main__":
    main()
