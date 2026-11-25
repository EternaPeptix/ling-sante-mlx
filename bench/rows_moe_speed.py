"""Queued microbenchmark for rows/moe.py vs stock SwitchGLU (+ combine + shared) on L4."""
import argparse
import statistics as st
import sys
import time
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(_root / "vendor"), str(_root)]

import mlx_lm  # noqa: E402,F401
import mlx.core as mx  # noqa: E402
from mlx_lm.models.bailing_moe_v3.rows.gemv_fp8 import gemv_lin, gemv_lin_swiglu  # noqa: E402
from mlx_lm.models.bailing_moe_v3.rows.gather import gather_rows  # noqa: E402
from mlx_lm.models.bailing_moe_v3.rows.moe import (  # noqa: E402
    moe_combine8_sh_res_rows,
    moe_dedup_rows,
    moe_down_combine_rows,
    moe_gate_up_swiglu_rows,
)
from mlx_lm.models.bailing_moe_v3.rows.router import gate_rows  # noqa: E402

from bench._layer_loader import bit_diff_count, load_layer, load_visits  # noqa: E402


def bench_queued(fn, reps=40, batch=16, warm=5):
    for _ in range(warm):
        mx.eval(fn())
    mx.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        outs = [fn() for _ in range(batch)]
        mx.eval(outs)
        ts.append((time.perf_counter() - t0) / batch)
    return 1e6 * st.median(ts)


def bench_pipe(fn, n_lo=4, n_hi=16, reps=4):
    mx.eval(*[fn() for _ in range(2)])
    mx.synchronize()
    dts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(*[fn() for _ in range(n_lo)])
        a = time.perf_counter() - t0
        t1 = time.perf_counter()
        mx.eval(*[fn() for _ in range(n_hi)])
        b = time.perf_counter() - t1
        dts.append((b - a) * 1e6 / (n_hi - n_lo))
    return st.median(dts)


def sweep_large(l4):
    sw = l4.mlp.switch_mlp
    se = l4.mlp.shared_experts
    gate = l4.mlp.gate
    mx.random.seed(41)
    print("\n=== Step 2b MoE B=1..24 Pipelined Sweep & 0-diff Verify ===", flush=True)
    for B in (1, 2, 4, 5, 8, 10, 12, 16, 20, 24):
        # Generate realistic slightly correlated activations (like prompt + speculative draft tokens)
        base = (mx.random.normal((1, 4096)) * 0.25).astype(mx.bfloat16)
        delta = (mx.random.normal((B, 4096)) * 0.12).astype(mx.bfloat16)
        xm = (base + delta).astype(mx.bfloat16)
        hp = (mx.random.normal((B, 4096)) * 0.25).astype(mx.bfloat16)
        inds, wts = gate_rows(
            xm, gate.weight, gate.expert_bias,
            top_k=gate.top_k, n_group=gate.n_group, topk_group=gate.topk_group,
            routed_scaling_factor=gate.routed_scaling_factor,
        )
        sh = gemv_lin(gemv_lin_swiglu(xm, se.gate_proj, se.up_proj), se.down_proj)
        dd = moe_dedup_rows(inds) if B >= 5 else None
        mx.eval(xm, hp, inds, wts, sh)
        if dd is not None:
            mx.eval(*dd)
            uniq = int(dd[4].item())
        else:
            uniq = len(set(inds.reshape(-1).tolist()))

        # Reference bit-exact outputs (hoist_dq=0, baseline rows=4, tmax=1/2)
        ref_gu = moe_gate_up_swiglu_rows(xm, sw.gate_proj.weight, sw.gate_proj.scales, sw.up_proj.weight, sw.up_proj.scales, inds, dedup=dd, rows=4, tmax=2 if B >= 5 else None, hoist_dq=0)
        ref_dc = moe_down_combine_rows(ref_gu, sw.down_proj.weight, sw.down_proj.scales, inds, wts, shared=sh, residual=hp, dedup=dd, tmax=1, hoist_dq=0)
        mx.eval(ref_gu, ref_dc)

        gu_res = []
        gu_tms = (1, 2) if B <= 4 else (2, 4, 8)
        gu_cands = [
            (rw, tm, nsg, hq)
            for rw in ((4,) if B <= 4 else (1, 2, 4))
            for tm in gu_tms
            if rw * tm <= 8
            for nsg in (2, 4)
            for hq in ((0,) if tm == 1 else (0, 1))
        ]
        for rw, tm, nsg, hq in gu_cands:
            g = moe_gate_up_swiglu_rows(xm, sw.gate_proj.weight, sw.gate_proj.scales, sw.up_proj.weight, sw.up_proj.scales, inds, dedup=dd, rows=rw, tmax=tm, nsg=nsg, hoist_dq=hq)
            d_cnt = bit_diff_count(g, ref_gu)
            if d_cnt != 0:
                print(f"    DIFF_WARN gu B={B} r{rw} t{tm} n{nsg} h{hq}: {d_cnt}/{ref_gu.size}", flush=True)
                continue
            t = bench_pipe(lambda rw=rw, tm=tm, nsg=nsg, hq=hq: moe_gate_up_swiglu_rows(xm, sw.gate_proj.weight, sw.gate_proj.scales, sw.up_proj.weight, sw.up_proj.scales, inds, dedup=dd, rows=rw, tmax=tm, nsg=nsg, hoist_dq=hq))
            gu_res.append((t, f"r{rw}t{tm}n{nsg}h{hq}={t:.0f}"))
        gu_res.sort()

        dc_res = []
        dc_tms = (1, 2) if B <= 4 else (1, 2, 4, 8)
        dc_cands = [
            (tm, nsg, hq)
            for tm in dc_tms
            for nsg in (4, 8)
            for hq in ((0,) if tm == 1 else (0, 1, 2))
        ]
        for tm, nsg, hq in dc_cands:
            d = moe_down_combine_rows(ref_gu, sw.down_proj.weight, sw.down_proj.scales, inds, wts, shared=sh, residual=hp, dedup=dd, tmax=tm, nsg=nsg, hoist_dq=hq)
            d_cnt = bit_diff_count(d, ref_dc)
            if d_cnt != 0:
                print(f"    DIFF_WARN dc B={B} t{tm} n{nsg} h{hq}: {d_cnt}/{ref_dc.size}", flush=True)
                continue
            t = bench_pipe(lambda tm=tm, nsg=nsg, hq=hq: moe_down_combine_rows(ref_gu, sw.down_proj.weight, sw.down_proj.scales, inds, wts, shared=sh, residual=hp, dedup=dd, tmax=tm, nsg=nsg, hoist_dq=hq))
            dc_res.append((t, f"t{tm}n{nsg}h{hq}={t:.0f}"))
        dc_res.sort()

        cur_gu = bench_pipe(lambda: moe_gate_up_swiglu_rows(xm, sw.gate_proj.weight, sw.gate_proj.scales, sw.up_proj.weight, sw.up_proj.scales, inds, dedup=dd))
        cur_dc = bench_pipe(lambda: moe_down_combine_rows(ref_gu, sw.down_proj.weight, sw.down_proj.scales, inds, wts, shared=sh, residual=hp, dedup=dd))
        print(
            f"  B={B:2d} (U={uniq:2d}/{B*8:3d}): cur gu={cur_gu:5.0f} ({cur_gu*68/1e3:4.1f}ms) -> best [{', '.join(s for _, s in gu_res[:5])}] | "
            f"cur dc={cur_dc:5.0f} ({cur_dc*68/1e3:4.1f}ms) -> best [{', '.join(s for _, s in dc_res[:5])}]",
            flush=True,
        )

    print("\n=== Step 1 A/B: gu adaptive=0 vs adaptive=1 across B in (4,5,8,10,12,16,20,24) ===", flush=True)
    for B in (4, 5, 8, 10, 12, 16, 20, 24):
        base = (mx.random.normal((1, 4096)) * 0.25).astype(mx.bfloat16)
        xm = (base + (mx.random.normal((B, 4096)) * 0.12)).astype(mx.bfloat16)
        hp = (mx.random.normal((B, 4096)) * 0.25).astype(mx.bfloat16)
        inds, wts = gate_rows(
            xm, gate.weight, gate.expert_bias,
            top_k=gate.top_k, n_group=gate.n_group, topk_group=gate.topk_group,
            routed_scaling_factor=gate.routed_scaling_factor,
        )
        sh = gemv_lin(gemv_lin_swiglu(xm, se.gate_proj, se.up_proj), se.down_proj)
        dd = moe_dedup_rows(inds) if B >= 5 else None
        mx.eval(xm, hp, inds, wts, sh)
        if dd is not None:
            mx.eval(*dd)
            uniq = int(dd[4].item())
        else:
            uniq = len(set(inds.reshape(-1).tolist()))
        g0 = moe_gate_up_swiglu_rows(xm, sw.gate_proj.weight, sw.gate_proj.scales, sw.up_proj.weight, sw.up_proj.scales, inds, dedup=dd, adaptive=0)
        g1 = moe_gate_up_swiglu_rows(xm, sw.gate_proj.weight, sw.gate_proj.scales, sw.up_proj.weight, sw.up_proj.scales, inds, dedup=dd, adaptive=1)
        assert bit_diff_count(g0, g1) == 0, f"gu adapt diff B={B}"
        t_gu0 = bench_pipe(lambda: moe_gate_up_swiglu_rows(xm, sw.gate_proj.weight, sw.gate_proj.scales, sw.up_proj.weight, sw.up_proj.scales, inds, dedup=dd, adaptive=0))
        t_gu1 = bench_pipe(lambda: moe_gate_up_swiglu_rows(xm, sw.gate_proj.weight, sw.gate_proj.scales, sw.up_proj.weight, sw.up_proj.scales, inds, dedup=dd, adaptive=1))
        t_dc = bench_pipe(lambda: moe_down_combine_rows(g0, sw.down_proj.weight, sw.down_proj.scales, inds, wts, shared=sh, residual=hp, dedup=dd))
        print(
            f"  B={B:2d} (U={uniq:2d}): gu ad0={t_gu0:5.0f}us ({t_gu0*68/1e3:4.1f}ms) vs ad1={t_gu1:5.0f}us ({t_gu1*68/1e3:4.1f}ms) [{t_gu0-t_gu1:+.0f}us] | "
            f"dc={t_dc:5.0f}us ({t_dc*68/1e3:4.1f}ms)",
            flush=True,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    a = ap.parse_args()

    l4, _ = load_layer(4)
    if a.sweep:
        sweep_large(l4)
        return
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
            sh = gemv_lin(gemv_lin_swiglu(xm, se.gate_proj, se.up_proj), se.down_proj)
            h_mid = moe_gate_up_swiglu_rows(
                xm, sw.gate_proj.weight, sw.gate_proj.scales, sw.up_proj.weight, sw.up_proj.scales, inds
            )
            return moe_down_combine_rows(
                h_mid, sw.down_proj.weight, sw.down_proj.scales, inds, wts, shared=sh, residual=hp, nsg=nsg
            )

        def rows_gather_combine():
            sh = gemv_lin(gemv_lin_swiglu(xm, se.gate_proj, se.up_proj), se.down_proj)
            h_mid = moe_gate_up_swiglu_rows(
                xm, sw.gate_proj.weight, sw.gate_proj.scales, sw.up_proj.weight, sw.up_proj.scales, inds
            )
            dn = gather_rows(h_mid, sw.down_proj.weight, sw.down_proj.scales, None, inds)
            return moe_combine8_sh_res_rows(dn, wts, shared=sh, residual=hp)

        h_mid_once = moe_gate_up_swiglu_rows(
            xm, sw.gate_proj.weight, sw.gate_proj.scales, sw.up_proj.weight, sw.up_proj.scales, inds
        )
        sh_once = gemv_lin(gemv_lin_swiglu(xm, se.gate_proj, se.up_proj), se.down_proj)
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
