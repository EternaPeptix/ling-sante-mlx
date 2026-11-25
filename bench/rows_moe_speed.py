"""Queued & cold multi-layer DRAM microbenchmark for rows/moe.py & rows/router.py (Phase A)."""
import argparse
import statistics as st
import sys
import time
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(_root / "vendor"), str(_root)]

import mlx_lm  # noqa: E402,F401
import mlx.core as mx  # noqa: E402
from mlx_lm.models.bailing_moe_v3.rows.gather import gather_rows  # noqa: E402
from mlx_lm.models.bailing_moe_v3.rows.gemv import router_gemv_rows  # noqa: E402
from mlx_lm.models.bailing_moe_v3.rows.gemv_fp8 import gemv_lin, gemv_lin_swiglu  # noqa: E402
from mlx_lm.models.bailing_moe_v3.rows.moe import (  # noqa: E402
    moe_combine8_sh_res_rows,
    moe_dedup_rows,
    moe_down_combine_packed_rows,
    moe_down_combine_rows,
    moe_gate_up_swiglu_packed_rows,
    moe_gate_up_swiglu_rows,
    moe_gate_up_swiglu_split_rows,
    pack_moe_dn_weights,
    pack_moe_gu_weights,
)
from mlx_lm.models.bailing_moe_v3.rows.router import gate_dedup_rows, gate_rows, group_router_dedup_rows, group_router_rows  # noqa: E402

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


def phase_a_bench(cold_layers=(4, 16, 28, 40, 52, 64)):
    """Comprehensive Phase A A/B across B=(1,3,4,5,6,8,9,12) on BOTH warm L4 and cold 6-layer DRAM."""
    print(f"Loading {len(cold_layers)} MoE layers {cold_layers} (FP8) for cold-DRAM benchmark...", flush=True)
    layers = [load_layer(li, use_fp8=True)[0] for li in cold_layers]
    # Runtime-on-load T3 packed weights for all 6 layers (also time packing!)
    t0 = time.perf_counter()
    packed_gu = [
        pack_moe_gu_weights(l.mlp.switch_mlp.gate_proj.weight, l.mlp.switch_mlp.gate_proj.scales,
                            l.mlp.switch_mlp.up_proj.weight, l.mlp.switch_mlp.up_proj.scales)
        for l in layers
    ]
    packed_dn = [
        pack_moe_dn_weights(l.mlp.switch_mlp.down_proj.weight, l.mlp.switch_mlp.down_proj.scales)
        for l in layers
    ]
    mx.eval(*packed_gu, *packed_dn)
    pack_ms = (time.perf_counter() - t0) * 1e3 / len(layers)
    print(f"T3 pack time: {pack_ms:.2f} ms/layer ({pack_ms*68:.1f} ms for 68 layers)", flush=True)

    # Router microbench
    print("\n=== Router + Fused Router-Dedup (us/layer, x68 ms) ===", flush=True)
    l0 = layers[0]
    g0 = l0.mlp.gate
    mx.random.seed(41)
    for B in (1, 3, 4, 5, 6, 8, 9, 12):
        xm = (mx.random.normal((B, 4096)) * 0.25).astype(mx.bfloat16)
        logits = router_gemv_rows(xm, g0.weight)
        mx.eval(xm, logits)
        t_sel = bench_pipe(lambda: group_router_rows(logits, g0.expert_bias))
        t_sel_dd = bench_pipe(lambda: group_router_dedup_rows(logits, g0.expert_bias)[2])
        inds0, _ = group_router_rows(logits, g0.expert_bias)
        mx.eval(inds0)
        t_sep = bench_pipe(lambda: (group_router_rows(logits, g0.expert_bias)[0], moe_dedup_rows(inds0)[0]))
        print(
            f"  B={B:2d}: select={t_sel:4.1f}us ({t_sel*68/1e3:.2f}ms) | "
            f"sep(sel+dd)={t_sep:4.1f}us -> fused_sel_dd={t_sel_dd:4.1f}us ({t_sep-t_sel_dd:+.1f}us, x68 {(t_sep-t_sel_dd)*68/1e3:+.2f}ms)",
            flush=True,
        )

    print("\n=== COLD 6-Layer DRAM Benchmark (us/layer | x68 ms) across B=(1,3,4,5,6,8,9,12) ===", flush=True)
    for B in (1, 3, 4, 5, 6, 8, 9, 12):
        # Generate realistic slightly correlated activations per layer (different experts per layer!)
        xs, hps, se_mids, shs, indss, wtss, dds, h_mids, h_splits = [], [], [], [], [], [], [], [], []
        uniqs, hots = [], []
        for idx_l, lyr in enumerate(layers):
            mx.random.seed(100 + B * 17 + idx_l)
            base = (mx.random.normal((1, 4096)) * 0.25).astype(mx.bfloat16)
            xm = (base + (mx.random.normal((B, 4096)) * 0.12)).astype(mx.bfloat16)
            hp = (mx.random.normal((B, 4096)) * 0.25).astype(mx.bfloat16)
            se = lyr.mlp.shared_experts
            gt = lyr.mlp.gate
            sw = lyr.mlp.switch_mlp
            sm = gemv_lin_swiglu(xm, se.gate_proj, se.up_proj)
            sh = gemv_lin(sm, se.down_proj)
            inds, wts, dd = gate_dedup_rows(
                xm, gt.weight, gt.expert_bias,
                top_k=gt.top_k, n_group=gt.n_group, topk_group=gt.topk_group,
                routed_scaling_factor=gt.routed_scaling_factor,
            )
            hm = moe_gate_up_swiglu_rows(xm, sw.gate_proj.weight, sw.gate_proj.scales, sw.up_proj.weight, sw.up_proj.scales, inds, dedup=dd)
            hs = moe_gate_up_swiglu_split_rows(xm, sw.gate_proj.weight, sw.gate_proj.scales, sw.up_proj.weight, sw.up_proj.scales, dd)
            mx.eval(xm, hp, sm, sh, inds, wts, *dd, hm, *hs)
            uniqs.append(int(dd[4].item()))
            hots.append(int(dd[2].item()))
            xs.append(xm); hps.append(hp); se_mids.append(sm); shs.append(sh)
            indss.append(inds); wtss.append(wts); dds.append(dd); h_mids.append(hm); h_splits.append(hs)

        u_avg = sum(uniqs) / len(uniqs)
        h_avg = sum(hots) / len(hots)
        nl = len(layers)

        # Gate+Up+SwiGLU variants (cold DRAM, TGL default = dq_mode=1)
        t_gu_2pso = bench_cold_layers(lambda: [
            moe_gate_up_swiglu_split_rows(xs[i], layers[i].mlp.switch_mlp.gate_proj.weight, layers[i].mlp.switch_mlp.gate_proj.scales,
                                          layers[i].mlp.switch_mlp.up_proj.weight, layers[i].mlp.switch_mlp.up_proj.scales, dds[i])
            for i in range(nl)
        ], nl)
        t_gu_lean = bench_cold_layers(lambda: [
            moe_gate_up_swiglu_rows(xs[i], layers[i].mlp.switch_mlp.gate_proj.weight, layers[i].mlp.switch_mlp.gate_proj.scales,
                                    layers[i].mlp.switch_mlp.up_proj.weight, layers[i].mlp.switch_mlp.up_proj.scales,
                                    indss[i], dedup=dds[i], adaptive=1)
            for i in range(nl)
        ], nl)
        t_gu_pk = bench_cold_layers(lambda: [
            moe_gate_up_swiglu_packed_rows(xs[i], packed_gu[i], dds[i])
            for i in range(nl)
        ], nl)

        # Down+Combine variants (cold DRAM, TGL default = dq_mode=1)
        t_dc = bench_cold_layers(lambda: [
            moe_down_combine_rows(h_splits[i], layers[i].mlp.switch_mlp.down_proj.weight, layers[i].mlp.switch_mlp.down_proj.scales,
                                  indss[i], wtss[i], shared=shs[i], residual=hps[i], dedup=dds[i])
            for i in range(nl)
        ], nl)
        t_dc_se = bench_cold_layers(lambda: [
            moe_down_combine_rows(h_splits[i], layers[i].mlp.switch_mlp.down_proj.weight, layers[i].mlp.switch_mlp.down_proj.scales,
                                  indss[i], wtss[i], residual=hps[i], dedup=dds[i],
                                  se_mid=se_mids[i], se_w_fp8=layers[i].mlp.shared_experts.down_proj.weight_fp8,
                                  se_sc=layers[i].mlp.shared_experts.down_proj.weight_scale_inv)
            for i in range(nl)
        ], nl)
        t_se_sep = bench_cold_layers(lambda: [
            gemv_lin(se_mids[i], layers[i].mlp.shared_experts.down_proj)
            for i in range(nl)
        ], nl)
        t_dc_pk = bench_cold_layers(lambda: [
            moe_down_combine_packed_rows(h_mids[i], packed_dn[i], dds[i], wtss[i], shared=shs[i], residual=hps[i])
            for i in range(nl)
        ], nl)
        t_dc_pk_se = bench_cold_layers(lambda: [
            moe_down_combine_packed_rows(h_mids[i], packed_dn[i], dds[i], wtss[i], residual=hps[i],
                                         se_mid=se_mids[i], se_w_fp8=layers[i].mlp.shared_experts.down_proj.weight_fp8,
                                         se_sc=layers[i].mlp.shared_experts.down_proj.weight_scale_inv)
            for i in range(nl)
        ], nl)

        roof_gu = 68 * u_avg * 5.57056 / 680.0
        roof_dc = 68 * u_avg * 2.78528 / 680.0
        print(
            f"  B={B:2d} (U={u_avg:4.1f}, hot={h_avg:4.1f} | floor GU={roof_gu:4.1f} DC={roof_dc:4.1f} tot={roof_gu+roof_dc:4.1f} ms):\n"
            f"    GU (x68 ms): 2PSO={t_gu_2pso*68/1e3:5.2f}  lean2={t_gu_lean*68/1e3:5.2f}  T3_pk={t_gu_pk*68/1e3:5.2f}\n"
            f"    DC (x68 ms): dc={t_dc*68/1e3:5.2f} | sep(se+dc)={(t_se_sep+t_dc)*68/1e3:5.2f} vs T7={t_dc_se*68/1e3:5.2f} | "
            f"T3_pk={t_dc_pk*68/1e3:5.2f} T3_pk+se={t_dc_pk_se*68/1e3:5.2f}\n"
            f"    TOT(GU+DC+se): unpacked={(t_gu_2pso+t_dc_se)*68/1e3:5.2f}ms  T3_packed={(t_gu_pk+t_dc_pk_se)*68/1e3:5.2f}ms",
            flush=True,
        )


def sweep_large(l4):
    sw = l4.mlp.switch_mlp
    se = l4.mlp.shared_experts
    gate = l4.mlp.gate
    mx.random.seed(41)
    print("\n=== Phase A MoE B=1..24 Pipelined Sweep & 0-diff Verify ===", flush=True)
    for B in (1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 16, 20, 24):
        base = (mx.random.normal((1, 4096)) * 0.25).astype(mx.bfloat16)
        delta = (mx.random.normal((B, 4096)) * 0.12).astype(mx.bfloat16)
        xm = (base + delta).astype(mx.bfloat16)
        hp = (mx.random.normal((B, 4096)) * 0.25).astype(mx.bfloat16)
        inds, wts, dd = gate_dedup_rows(
            xm, gate.weight, gate.expert_bias,
            top_k=gate.top_k, n_group=gate.n_group, topk_group=gate.topk_group,
            routed_scaling_factor=gate.routed_scaling_factor,
        )
        sh = gemv_lin(gemv_lin_swiglu(xm, se.gate_proj, se.up_proj), se.down_proj)
        mx.eval(xm, hp, inds, wts, sh, *dd)
        uniq = int(dd[4].item())
        hot = int(dd[2].item())

        t_gu_flat = bench_pipe(lambda: moe_gate_up_swiglu_rows(xm, sw.gate_proj.weight, sw.gate_proj.scales, sw.up_proj.weight, sw.up_proj.scales, inds, dedup=dd, adaptive=0, tmax=2 if B >= 2 else 1))
        t_gu_lean = bench_pipe(lambda: moe_gate_up_swiglu_rows(xm, sw.gate_proj.weight, sw.gate_proj.scales, sw.up_proj.weight, sw.up_proj.scales, inds, dedup=dd, adaptive=1))
        t_gu_sp = bench_pipe(lambda: moe_gate_up_swiglu_split_rows(xm, sw.gate_proj.weight, sw.gate_proj.scales, sw.up_proj.weight, sw.up_proj.scales, dd))
        hm = moe_gate_up_swiglu_rows(xm, sw.gate_proj.weight, sw.gate_proj.scales, sw.up_proj.weight, sw.up_proj.scales, inds, dedup=dd)
        mx.eval(hm)
        t_dc2 = bench_pipe(lambda: moe_down_combine_rows(hm, sw.down_proj.weight, sw.down_proj.scales, inds, wts, shared=sh, residual=hp, dedup=dd, tmax=2 if B >= 2 else 1))
        t_dc4 = bench_pipe(lambda: moe_down_combine_rows(hm, sw.down_proj.weight, sw.down_proj.scales, inds, wts, shared=sh, residual=hp, dedup=dd, tmax=4 if B >= 4 else (2 if B >= 2 else 1)))
        print(
            f"  B={B:2d} (U={uniq:2d}, hot={hot:2d}): "
            f"gu flat={t_gu_flat:5.0f} ({t_gu_flat*68/1e3:4.1f}ms) lean2={t_gu_lean:5.0f} ({t_gu_lean*68/1e3:4.1f}ms) 2pso={t_gu_sp:5.0f} ({t_gu_sp*68/1e3:4.1f}ms) | "
            f"dc tm2={t_dc2:5.0f} ({t_dc2*68/1e3:4.1f}ms) tm4={t_dc4:5.0f} ({t_dc4*68/1e3:4.1f}ms)",
            flush=True,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--phase-a", action="store_true")
    a = ap.parse_args()

    if a.phase_a:
        phase_a_bench()
        return

    l4, _ = load_layer(4, use_fp8=True)
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
        dd = moe_dedup_rows(inds)
        mx.eval(xm, hp, inds, wts, xm3, hp3, inds3, wts3, *dd)
        uniq = int(dd[4].item())

        def stock_routed_and_sh():
            y = sw(xm3, inds3)
            routed = (y * wts3[..., None]).sum(axis=-2).astype(xm.dtype)
            return hp3 + (routed + se(xm3))

        def rows_routed_and_sh(nsg=4):
            sh = gemv_lin(gemv_lin_swiglu(xm, se.gate_proj, se.up_proj), se.down_proj)
            h_mid = moe_gate_up_swiglu_rows(
                xm, sw.gate_proj.weight, sw.gate_proj.scales, sw.up_proj.weight, sw.up_proj.scales, inds, dedup=dd
            )
            return moe_down_combine_rows(
                h_mid, sw.down_proj.weight, sw.down_proj.scales, inds, wts, shared=sh, residual=hp, nsg=nsg, dedup=dd
            )

        def rows_fused_se():
            se_mid = gemv_lin_swiglu(xm, se.gate_proj, se.up_proj)
            h_split = moe_gate_up_swiglu_split_rows(
                xm, sw.gate_proj.weight, sw.gate_proj.scales, sw.up_proj.weight, sw.up_proj.scales, dd
            )
            return moe_down_combine_rows(
                h_split, sw.down_proj.weight, sw.down_proj.scales, inds, wts, residual=hp, dedup=dd,
                se_mid=se_mid, se_w_fp8=se.down_proj.weight_fp8, se_sc=se.down_proj.weight_scale_inv,
            )

        h_mid_once = moe_gate_up_swiglu_rows(
            xm, sw.gate_proj.weight, sw.gate_proj.scales, sw.up_proj.weight, sw.up_proj.scales, inds, dedup=dd
        )
        sh_once = gemv_lin(gemv_lin_swiglu(xm, se.gate_proj, se.up_proj), se.down_proj)
        mx.eval(h_mid_once, sh_once)

        t_ref = bench_queued(stock_routed_and_sh)
        t_gu = bench_queued(lambda: moe_gate_up_swiglu_rows(
            xm, sw.gate_proj.weight, sw.gate_proj.scales, sw.up_proj.weight, sw.up_proj.scales, inds, dedup=dd
        ))
        t_gu_2p = bench_queued(lambda: moe_gate_up_swiglu_split_rows(
            xm, sw.gate_proj.weight, sw.gate_proj.scales, sw.up_proj.weight, sw.up_proj.scales, dd
        ))
        t_dc4 = bench_queued(lambda: moe_down_combine_rows(
            h_mid_once, sw.down_proj.weight, sw.down_proj.scales, inds, wts, shared=sh_once, residual=hp, nsg=4, dedup=dd
        ))
        t_full = bench_queued(lambda: rows_routed_and_sh(nsg=4))
        t_full_se = bench_queued(rows_fused_se)
        print(
            f"B={B} (uniq={uniq:2d}/{B*8:2d}) stock={t_ref:6.1f} us | "
            f"gu={t_gu:5.1f} gu2p={t_gu_2p:5.1f} dc={t_dc4:5.1f} | full={t_full:5.1f} 2pso+se={t_full_se:5.1f} ({t_ref/t_full_se:.2f}x)",
            flush=True,
        )


if __name__ == "__main__":
    main()
