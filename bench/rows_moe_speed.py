"""Queued microbenchmark for rows/moe.py vs stock SwitchGLU (+ combine + shared) on L4."""
import statistics as st
import sys
import time
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(_root / "vendor"), str(_root)]

import mlx.core as mx  # noqa: E402
from mlx_lm.models.bailing_moe_v3.rows.gemv import gemv_rows, gemv_swiglu_rows  # noqa: E402
from mlx_lm.models.bailing_moe_v3.rows.gather import gather_rows  # noqa: E402
from mlx_lm.models.bailing_moe_v3.rows.moe import (  # noqa: E402
    moe_combine8_sh_res_rows,
    moe_down_combine_rows,
    moe_gate_up_swiglu_rows,
)

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
    sw = l4.mlp.switch_mlp
    se = l4.mlp.shared_experts
    visits = load_visits(4, limit=8)

    for B in (1, 2, 3, 4):
        xm = mx.concatenate([v["mlp_in"][0, 0:1] for v in visits[:B]], axis=0)
        hp = mx.concatenate([v["h_post"][0, 0:1] for v in visits[:B]], axis=0)
        inds = mx.concatenate([v["r_inds"][0, 0:1] for v in visits[:B]], axis=0).astype(mx.uint32)
        wts = mx.concatenate([v["r_wts"][0, 0:1] for v in visits[:B]], axis=0)
        xm3 = xm[:, None, :]
        hp3 = hp[:, None, :]
        inds3 = inds[:, None, :]
        wts3 = wts[:, None, :]
        uniq = len(set(inds.reshape(-1).tolist()))
        mx.eval(xm, hp, inds, wts, xm3, hp3, inds3, wts3)

        def stock_routed_and_sh():
            y = sw(xm3, inds3)
            routed = (y * wts3[..., None]).sum(axis=-2).astype(xm.dtype)
            return hp3 + (routed + se(xm3))

        def rows_routed_and_sh(nsg=8):
            sh = gemv_rows(gemv_swiglu_rows(xm, se.gate_proj.weight, se.up_proj.weight), se.down_proj.weight)
            h_mid = moe_gate_up_swiglu_rows(
                xm, sw.gate_proj.weight, sw.gate_proj.scales, sw.up_proj.weight, sw.up_proj.scales, inds
            )
            return moe_down_combine_rows(
                h_mid, sw.down_proj.weight, sw.down_proj.scales, inds, wts, shared=sh, residual=hp, nsg=nsg
            )

        def rows_gather_combine():
            sh = gemv_rows(gemv_swiglu_rows(xm, se.gate_proj.weight, se.up_proj.weight), se.down_proj.weight)
            h_mid = moe_gate_up_swiglu_rows(
                xm, sw.gate_proj.weight, sw.gate_proj.scales, sw.up_proj.weight, sw.up_proj.scales, inds
            )
            dn = gather_rows(h_mid, sw.down_proj.weight, sw.down_proj.scales, None, inds)
            return moe_combine8_sh_res_rows(dn, wts, shared=sh, residual=hp)

        h_mid_once = moe_gate_up_swiglu_rows(
            xm, sw.gate_proj.weight, sw.gate_proj.scales, sw.up_proj.weight, sw.up_proj.scales, inds
        )
        sh_once = gemv_rows(gemv_swiglu_rows(xm, se.gate_proj.weight, se.up_proj.weight), se.down_proj.weight)
        mx.eval(h_mid_once, sh_once)

        t_ref = bench_queued(stock_routed_and_sh)
        t_gu = bench_queued(lambda: moe_gate_up_swiglu_rows(
            xm, sw.gate_proj.weight, sw.gate_proj.scales, sw.up_proj.weight, sw.up_proj.scales, inds
        ))
        t_dc = {nsg: bench_queued(lambda n=nsg: moe_down_combine_rows(
            h_mid_once, sw.down_proj.weight, sw.down_proj.scales, inds, wts, shared=sh_once, residual=hp, nsg=n
        )) for nsg in (2, 4, 8)}
        t_gc = bench_queued(lambda: moe_combine8_sh_res_rows(
            gather_rows(h_mid_once, sw.down_proj.weight, sw.down_proj.scales, None, inds), wts, shared=sh_once, residual=hp
        ))
        t_nsg = {nsg: bench_queued(lambda n=nsg: rows_routed_and_sh(nsg=n)) for nsg in (2, 4, 8)}
        t_full_gc = bench_queued(rows_gather_combine)
        best_nsg = min(t_nsg, key=t_nsg.get)
        best_t = min(t_nsg[best_nsg], t_full_gc)
        best_lbl = "gath+comb8" if t_full_gc < t_nsg[best_nsg] else f"dc_nsg{best_nsg}"
        print(
            f"B={B} (uniq={uniq:2d}/{B*8:2d}) stock={t_ref:6.1f} us | "
            f"gu_sw={t_gu:5.1f} dc[2/4/8]={t_dc[2]:4.1f}/{t_dc[4]:4.1f}/{t_dc[8]:4.1f} g+c8={t_gc:4.1f} | "
            f"full[dc2/4/8/gc]={t_nsg[2]:5.1f}/{t_nsg[4]:5.1f}/{t_nsg[8]:5.1f}/{t_full_gc:5.1f} "
            f"best={best_lbl} ({t_ref/best_t:.2f}x)"
        )


if __name__ == "__main__":
    main()
