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
    moe_down_combine_packed_rows,
    moe_down_combine_rows,
    moe_gate_up_swiglu_packed_rows,
    moe_gate_up_swiglu_rows,
    pack_moe_dn_weights,
    pack_moe_gu_weights,
)
from mlx_lm.models.bailing_moe_v3.rows.router import gate_dedup_rows, gate_rows  # noqa: E402
from mlx_lm.utils import maybe_set_recommended_wired_limit  # noqa: E402

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


def bench_cold_layers(layers_fn, n_layers: int, reps: int = 5, warm: int = 1) -> float:
    """Zero-sync-bias cold-DRAM per-layer us via (T(3 sweeps) - T(1 sweep)) / (2 * n_layers)."""
    for _ in range(warm):
        mx.eval(layers_fn())
    mx.synchronize()
    dts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(layers_fn())
        a = time.perf_counter() - t0
        t1 = time.perf_counter()
        mx.eval(layers_fn() + layers_fn() + layers_fn())
        b = time.perf_counter() - t1
        dts.append((b - a) * 1e6 / (2 * n_layers))
    return st.median(dts)


def _make_batch_x(B: int, seed: int, mode: str = "c1") -> mx.array:
    """Realistic activations matching real verify_decode U distributions (C1_L1..L8 and C4_L3 at B=12)."""
    mx.random.seed(seed)
    if B == 12 and mode == "c4":
        # Shared topic + 4 stream offsets + 3 correlated draft tokens -> U ~ 38-42 (matches real C4_L3!)
        shared = (mx.random.normal((1, 1, 4096)) * 0.22).astype(mx.bfloat16)
        streams = (mx.random.normal((4, 1, 4096)) * 0.14).astype(mx.bfloat16)
        deltas = (mx.random.normal((4, 3, 4096)) * 0.10).astype(mx.bfloat16)
        return (shared + streams + deltas).reshape(12, 4096).astype(mx.bfloat16)
    base = (mx.random.normal((1, 4096)) * 0.25).astype(mx.bfloat16)
    delta = (mx.random.normal((B, 4096)) * 0.14).astype(mx.bfloat16)
    return (base + delta).astype(mx.bfloat16)


def ab_bw_bench(n_layers: int = 16):
    """Comprehensive 16-layer cold-DRAM ablation + 0-diff verification across B=1,3,5,8,12."""
    maybe_set_recommended_wired_limit()
    layer_ids = [4 + i * (60 // max(1, n_layers - 1)) for i in range(n_layers)]
    print(f"Loading {n_layers} MoE layers {layer_ids} for 16-layer cold-DRAM ablation...", flush=True)
    t0 = time.perf_counter()
    layers = [load_layer(li, use_fp8=True)[0] for li in layer_ids]
    print(f"Loaded {n_layers} layers in {time.perf_counter() - t0:.1f}s", flush=True)

    # Pack T3 weights on the same 16 cold layers (measure pack time once)
    t0 = time.perf_counter()
    pk_gu = [
        pack_moe_gu_weights(
            l.mlp.switch_mlp.gate_proj.weight, l.mlp.switch_mlp.gate_proj.scales,
            l.mlp.switch_mlp.up_proj.weight, l.mlp.switch_mlp.up_proj.scales,
        )
        for l in layers
    ]
    pk_dn = [
        pack_moe_dn_weights(l.mlp.switch_mlp.down_proj.weight, l.mlp.switch_mlp.down_proj.scales)
        for l in layers
    ]
    mx.eval(*pk_gu, *pk_dn)
    mx.synchronize()
    print(f"Packed T3 ({n_layers} layers) in {time.perf_counter() - t0:.2f}s\n", flush=True)

    GU_B = 2 * 1280 * 4096 * 17 // 32
    DC_B = 4096 * 1280 * 17 // 32

    cases = [(1, "c1"), (3, "c1"), (5, "c1"), (8, "c1"), (12, "c4")]
    for B, mode in cases:
        xs, hps, se_mids, shs, indss, wtss, dds, ref_gus, ref_dcs = [], [], [], [], [], [], [], [], []
        uniqs, hots = [], []
        for idx, lay in enumerate(layers):
            xm = _make_batch_x(B, seed=2000 + B * 100 + idx, mode=mode)
            hp = (mx.random.normal((B, 4096)) * 0.25).astype(mx.bfloat16)
            gate = lay.mlp.gate
            se = lay.mlp.shared_experts
            sw = lay.mlp.switch_mlp
            if B >= 2:
                inds, wts, dd = gate_dedup_rows(
                    xm, gate.weight, gate.expert_bias,
                    top_k=gate.top_k, n_group=gate.n_group, topk_group=gate.topk_group,
                    routed_scaling_factor=gate.routed_scaling_factor,
                )
            else:
                inds, wts = gate_rows(
                    xm, gate.weight, gate.expert_bias,
                    top_k=gate.top_k, n_group=gate.n_group, topk_group=gate.topk_group,
                    routed_scaling_factor=gate.routed_scaling_factor,
                )
                dd = None
            sm = gemv_lin_swiglu(xm, se.gate_proj, se.up_proj)
            sh = gemv_lin(sm, se.down_proj)
            rgu = moe_gate_up_swiglu_rows(
                xm, sw.gate_proj.weight, sw.gate_proj.scales, sw.up_proj.weight, sw.up_proj.scales, inds, dedup=dd
            )
            rdc = moe_down_combine_rows(
                rgu, sw.down_proj.weight, sw.down_proj.scales, inds, wts, shared=sh, residual=hp, dedup=dd
            )
            mx.eval(xm, hp, sm, sh, inds, wts, rgu, rdc)
            if dd is not None:
                mx.eval(*dd)
                uniqs.append(int(dd[4].item()))
                hots.append(int(dd[2].item()))
            else:
                uniqs.append(8)
                hots.append(0)
            xs.append(xm); hps.append(hp); se_mids.append(sm); shs.append(sh)
            indss.append(inds); wtss.append(wts); dds.append(dd); ref_gus.append(rgu); ref_dcs.append(rdc)

        u_avg = sum(uniqs) / len(uniqs)
        h_avg = sum(hots) / len(hots)
        print(f"=== B={B:2d} ({mode}, U={u_avg:4.1f}, hot={h_avg:4.1f}, cold={n_layers*u_avg*(GU_B+DC_B)/1e9:.2f} GB) ===", flush=True)

        gu_cands = [
            ("unpacked_prod", lambda i: moe_gate_up_swiglu_rows(
                xs[i], layers[i].mlp.switch_mlp.gate_proj.weight, layers[i].mlp.switch_mlp.gate_proj.scales,
                layers[i].mlp.switch_mlp.up_proj.weight, layers[i].mlp.switch_mlp.up_proj.scales, indss[i], dedup=dds[i])),
            ("T3_pk_2pass", lambda i: moe_gate_up_swiglu_packed_rows(
                xs[i], pk_gu[i], indss[i], dedup=dds[i], single_kb=False)),
            ("T3_pk_1pass", lambda i: moe_gate_up_swiglu_packed_rows(
                xs[i], pk_gu[i], indss[i], dedup=dds[i], single_kb=True)),
        ]
        for name, fn in gu_cands:
            y0 = fn(0)
            mx.eval(y0)
            d0 = bit_diff_count(y0, ref_gus[0])
            if d0 != 0:
                print(f"  GU {name:14s}: DIFF={d0}/{ref_gus[0].size}", flush=True)
                continue
            t_us = bench_cold_layers(lambda fn=fn: [fn(i) for i in range(n_layers)], n_layers, reps=5)
            print(f"  GU {name:14s}: {t_us*68/1e3:6.2f} ms | {(u_avg*GU_B)/(t_us*1e3):6.1f} GB/s", flush=True)

        t_se = bench_cold_layers(lambda: [gemv_lin(se_mids[i], layers[i].mlp.shared_experts.down_proj) for i in range(n_layers)], n_layers, reps=5)
        se_ms = t_se * 68 / 1e3
        dc_cands = [
            ("unpacked_prod", lambda i: moe_down_combine_rows(
                ref_gus[i], layers[i].mlp.switch_mlp.down_proj.weight, layers[i].mlp.switch_mlp.down_proj.scales,
                indss[i], wtss[i], shared=shs[i], residual=hps[i], dedup=dds[i])),
            ("unpacked+T7se", lambda i: moe_down_combine_rows(
                ref_gus[i], layers[i].mlp.switch_mlp.down_proj.weight, layers[i].mlp.switch_mlp.down_proj.scales,
                indss[i], wtss[i], residual=hps[i], dedup=dds[i],
                se_mid=se_mids[i], se_w_fp8=layers[i].mlp.shared_experts.down_proj.weight_fp8,
                se_sc=layers[i].mlp.shared_experts.down_proj.weight_scale_inv)),
            ("T3_dc_packed", lambda i: moe_down_combine_packed_rows(
                ref_gus[i], pk_dn[i], indss[i], wtss[i], shared=shs[i], residual=hps[i], dedup=dds[i])),
            ("T3_dc+T7se", lambda i: moe_down_combine_packed_rows(
                ref_gus[i], pk_dn[i], indss[i], wtss[i], residual=hps[i], dedup=dds[i],
                se_mid=se_mids[i], se_w_fp8=layers[i].mlp.shared_experts.down_proj.weight_fp8,
                se_sc=layers[i].mlp.shared_experts.down_proj.weight_scale_inv)),
        ]
        for name, fn in dc_cands:
            y0 = fn(0)
            mx.eval(y0)
            d0 = bit_diff_count(y0, ref_dcs[0])
            if d0 != 0:
                print(f"  DC {name:14s}: DIFF={d0}/{ref_dcs[0].size}", flush=True)
                continue
            t_us = bench_cold_layers(lambda fn=fn: [fn(i) for i in range(n_layers)], n_layers, reps=5)
            ms = t_us * 68 / 1e3
            extra = f" (fused vs sep {se_ms:.2f})" if "T7se" in name else f" (+se={ms+se_ms:.2f})"
            print(f"  DC {name:14s}: {ms:6.2f} ms | {(u_avg*DC_B)/(t_us*1e3):6.1f} GB/s{extra}", flush=True)


def cold_bw_bench(n_layers: int = 16, bs=(1, 3, 5, 8, 12)):
    """Cold >=16-layer (>1.1-5.4 GB active weights) DRAM bandwidth benchmark for MoE GU & DC."""
    maybe_set_recommended_wired_limit()
    # Spread 16 MoE layers across 4..64 (4, 8, 12, ..., 64)
    layer_ids = [4 + i * (60 // max(1, n_layers - 1)) for i in range(n_layers)]
    print(f"Loading {n_layers} MoE layers {layer_ids} (FP8) for cold-DRAM bandwidth bench...", flush=True)
    t0 = time.perf_counter()
    layers = [load_layer(li, use_fp8=True)[0] for li in layer_ids]
    print(f"Loaded {n_layers} layers in {time.perf_counter() - t0:.1f}s\n", flush=True)

    # Exact unique-expert byte counts for MXFP4 (group_size=32 -> 0.53125 B/param):
    #   GU (gate+up): 2 * 1280 * 4096 * 0.53125 = 5,570,560 B (5.57056 MB)
    #   DC (down):    1 * 4096 * 1280 * 0.53125 = 2,785,280 B (2.78528 MB)
    #   Total/expert: 8,355,840 B (8.35584 MB)
    GU_B = 2 * 1280 * 4096 * 17 // 32
    DC_B = 4096 * 1280 * 17 // 32

    print(f"=== COLD {n_layers}-Layer DRAM Bandwidth (c7f0351 / current rows/moe.py) ===", flush=True)
    print(f"{'B':>3} {'U_avg':>5} {'hot':>4} {'coldGB':>6} | {'GU_ms':>6} {'GU_GB/s':>7} | {'DC_ms':>6} {'DC_GB/s':>7} | {'TOT_ms':>6} {'TOT_GB/s':>8} | {'SE_dn':>5}", flush=True)
    for B in bs:
        xs, hps, se_mids, shs, indss, wtss, dds, h_mids = [], [], [], [], [], [], [], []
        uniqs, hots = [], []
        for idx, lay in enumerate(layers):
            xm = _make_batch_x(B, seed=1000 + B * 100 + idx)
            hp = (mx.random.normal((B, 4096)) * 0.25).astype(mx.bfloat16)
            gate = lay.mlp.gate
            se = lay.mlp.shared_experts
            sw = lay.mlp.switch_mlp
            if B >= 2:
                inds, wts, dd = gate_dedup_rows(
                    xm, gate.weight, gate.expert_bias,
                    top_k=gate.top_k, n_group=gate.n_group, topk_group=gate.topk_group,
                    routed_scaling_factor=gate.routed_scaling_factor,
                )
            else:
                inds, wts = gate_rows(
                    xm, gate.weight, gate.expert_bias,
                    top_k=gate.top_k, n_group=gate.n_group, topk_group=gate.topk_group,
                    routed_scaling_factor=gate.routed_scaling_factor,
                )
                dd = None
            sm = gemv_lin_swiglu(xm, se.gate_proj, se.up_proj)
            sh = gemv_lin(sm, se.down_proj)
            hm = moe_gate_up_swiglu_rows(
                xm, sw.gate_proj.weight, sw.gate_proj.scales, sw.up_proj.weight, sw.up_proj.scales, inds, dedup=dd
            )
            mx.eval(xm, hp, sm, sh, inds, wts, hm)
            if dd is not None:
                mx.eval(*dd)
                uniqs.append(int(dd[4].item()))
                hots.append(int(dd[2].item()))
            else:
                uniqs.append(len(set(inds.reshape(-1).tolist())))
                hots.append(0)
            xs.append(xm); hps.append(hp); se_mids.append(sm); shs.append(sh)
            indss.append(inds); wtss.append(wts); dds.append(dd); h_mids.append(hm)

        u_avg = sum(uniqs) / len(uniqs)
        h_avg = sum(hots) / len(hots)
        cold_gb = n_layers * u_avg * (GU_B + DC_B) / 1e9

        t_gu = bench_cold_layers(lambda: [
            moe_gate_up_swiglu_rows(
                xs[i], layers[i].mlp.switch_mlp.gate_proj.weight, layers[i].mlp.switch_mlp.gate_proj.scales,
                layers[i].mlp.switch_mlp.up_proj.weight, layers[i].mlp.switch_mlp.up_proj.scales,
                indss[i], dedup=dds[i],
            )
            for i in range(n_layers)
        ], n_layers)

        t_dc = bench_cold_layers(lambda: [
            moe_down_combine_rows(
                h_mids[i], layers[i].mlp.switch_mlp.down_proj.weight, layers[i].mlp.switch_mlp.down_proj.scales,
                indss[i], wtss[i], shared=shs[i], residual=hps[i], dedup=dds[i],
            )
            for i in range(n_layers)
        ], n_layers)

        t_se = bench_cold_layers(lambda: [
            gemv_lin(se_mids[i], layers[i].mlp.shared_experts.down_proj)
            for i in range(n_layers)
        ], n_layers)

        gu_ms = t_gu * 68 / 1e3
        dc_ms = t_dc * 68 / 1e3
        se_ms = t_se * 68 / 1e3
        tot_ms = gu_ms + dc_ms
        gu_gbs = (u_avg * GU_B) / (t_gu * 1e3)
        dc_gbs = (u_avg * DC_B) / (t_dc * 1e3)
        tot_gbs = (u_avg * (GU_B + DC_B)) / ((t_gu + t_dc) * 1e3)
        print(
            f"{B:3d} {u_avg:5.1f} {h_avg:4.1f} {cold_gb:6.2f} | "
            f"{gu_ms:6.2f} {gu_gbs:7.1f} | {dc_ms:6.2f} {dc_gbs:7.1f} | "
            f"{tot_ms:6.2f} {tot_gbs:8.1f} | {se_ms:5.2f}",
            flush=True,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--cold-bw", action="store_true")
    ap.add_argument("--ab-bw", action="store_true")
    ap.add_argument("--layers", type=int, default=16)
    a = ap.parse_args()

    if a.ab_bw:
        ab_bw_bench(n_layers=a.layers)
        return
    if a.cold_bw:
        cold_bw_bench(n_layers=a.layers)
        return

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
