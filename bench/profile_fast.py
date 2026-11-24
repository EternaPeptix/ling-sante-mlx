"""Profile wired LingSante fast decode anatomy & 72-layer component breakdown at C=1 and C=4."""
import argparse
import json
import os
import statistics as st
import sys
import tempfile
import time
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(_root / "vendor"), str(_root)]

import mlx_lm  # noqa: E402,F401
import mlx.core as mx  # noqa: E402
from mlx_lm import load  # noqa: E402
from mlx_lm.models.bailing_moe_v3.rows.gemv import (  # noqa: E402
    gemv_rows,
    gemv_rows_multi,
    gemv_rows_res,
    gemv_swiglu_rows,
)
from mlx_lm.models.bailing_moe_v3.rows.kda import kda_gdn_norm_gate_rows, kda_pre_rows  # noqa: E402
from mlx_lm.models.bailing_moe_v3.rows.mla import (  # noqa: E402
    mla_gate_rows,
    mla_layer_rows,
    mla_prep_kv_rope_rows,
    mla_sdpa_rows,
)
from mlx_lm.models.bailing_moe_v3.rows.moe import moe_down_combine_rows, moe_gate_up_swiglu_rows  # noqa: E402
from mlx_lm.models.bailing_moe_v3.rows.norm import rms_norm_rows  # noqa: E402
from mlx_lm.models.bailing_moe_v3.rows.router import gate_rows  # noqa: E402
from mlx_lm.utils import maybe_set_recommended_wired_limit  # noqa: E402

from bench.batch_dec import merge_caches, prefill_stream  # noqa: E402
from bench.exact import DEFAULT_MODEL, PROMPTS, prompt_ids  # noqa: E402
from bench.profile_decode import bench_step_anatomy  # noqa: E402


def bench_q(fn, reps=30, batch=12, warm=5):
    for _ in range(warm):
        mx.eval(fn())
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        outs = [fn() for _ in range(batch)]
        mx.eval(outs)
        ts.append((time.perf_counter() - t0) / batch)
    return 1e6 * st.median(ts)


def profile_fast_components(model, C: int):
    L0 = model.model.layers[0]
    L4 = model.model.layers[4]
    L7 = model.model.layers[7]

    x = mx.random.normal((C, 4096)).astype(mx.bfloat16)
    x8 = mx.random.normal((C, 8192)).astype(mx.bfloat16)
    cs0 = mx.zeros((C, 3, 8192), dtype=mx.bfloat16)
    cs1 = mx.zeros((C, 3, 8192), dtype=mx.bfloat16)
    cs2 = mx.zeros((C, 3, 8192), dtype=mx.bfloat16)
    cs3 = mx.zeros((C, 64, 128, 128), dtype=mx.float32)

    S = 512
    c7 = model.make_cache()[7]
    buf_k = mx.zeros((C, 1, S + 64, 512), dtype=mx.bfloat16)
    buf_v = mx.zeros((C, 1, S + 64, 64), dtype=mx.bfloat16)
    mx.eval(x, x8, cs0, cs1, cs2, cs3, buf_k, buf_v)

    a4 = L4.attention
    m4 = L4.mlp
    ws4 = [a4.q_proj.weight, a4.k_proj.weight, a4.v_proj.weight, a4.f_proj.weight, a4.g_proj.weight, a4.b_proj.weight]
    q_r, k_r, v_r, f_r, g_r, b_r = gemv_rows_multi(x, ws4)
    q, k, v, g, beta, _, _, _ = kda_pre_rows(
        q_r, k_r, v_r, f_r, b_r, cs0, cs1, cs2,
        a4.q_conv1d.weight, a4.k_conv1d.weight, a4.v_conv1d.weight,
        a4.A_log, a4.dt_bias, lower_bound=a4.lower_bound, eps=1e-6,
    )
    inds, wts = gate_rows(x, m4.gate.weight, m4.gate.expert_bias)
    se = m4.shared_experts
    sw = m4.switch_mlp
    h_mid = moe_gate_up_swiglu_rows(x, sw.gate_proj.weight, sw.gate_proj.scales, sw.up_proj.weight, sw.up_proj.scales, inds)
    sh = gemv_rows(gemv_swiglu_rows(x, se.gate_proj.weight, se.up_proj.weight), se.down_proj.weight)
    a7 = L7.attention
    qa, kva, mg = gemv_rows_multi(x, [a7.q_a_proj.weight, a7.kv_a_proj_with_mqa.weight, a7.g_proj.weight])
    qan = rms_norm_rows(qa, a7.q_a_layernorm.weight, a7.q_a_layernorm.eps)
    qf = gemv_rows(qan, a7.q_b_proj.weight)
    offs = mx.array([S] * C, dtype=mx.int32)
    qn, qr, kl, kr = mla_prep_kv_rope_rows(qf, kva, a7.kv_a_layernorm.weight, offs)
    qne = a7.embed_q(qn)
    c7.keys, c7.values, c7.offset = buf_k, buf_v, S
    kl_all, kr_all = buf_k[:, :, : S + 1, :], buf_v[:, :, : S + 1, :]
    so = mla_sdpa_rows(qne, qr, kl_all, kr_all, a7.scale, [0] * C)
    uo = a7.unembed_out(so)
    mx.eval(q_r, k_r, v_r, f_r, g_r, b_r, q, k, v, g, beta, inds, wts, h_mid, sh, qa, kva, mg, qan, qf, qn, qr, kl, kr, qne, so, uo)

    def mla_full():
        c7.keys, c7.values, c7.offset = buf_k, buf_v, S
        return mla_layer_rows(x, x, a7, c7), c7.keys

    us = {
        "rms_norm_4096": bench_q(lambda: rms_norm_rows(x, L4.input_layernorm.weight, L4.input_layernorm.eps)),
        "kda_qkvfgb": bench_q(lambda: gemv_rows_multi(x, ws4)),
        "kda_pre": bench_q(lambda: kda_pre_rows(q_r, k_r, v_r, f_r, b_r, cs0, cs1, cs2, a4.q_conv1d.weight, a4.k_conv1d.weight, a4.v_conv1d.weight, a4.A_log, a4.dt_bias, lower_bound=a4.lower_bound, eps=1e-6)),
        "kda_gdn": bench_q(lambda: kda_gdn_norm_gate_rows(q, k, v, g, beta, cs3, g_r, a4.o_norm.weight, eps=a4.o_norm.eps, fused=True)),
        "kda_o_proj_res": bench_q(lambda: gemv_rows_res(x8, a4.o_proj.weight, x)),
        "dense_mlp": bench_q(lambda: gemv_rows_res(gemv_swiglu_rows(x, L0.mlp.gate_proj.weight, L0.mlp.up_proj.weight), L0.mlp.down_proj.weight, x)),
        "router": bench_q(lambda: gate_rows(x, m4.gate.weight, m4.gate.expert_bias)),
        "shared_exp": bench_q(lambda: gemv_rows(gemv_swiglu_rows(x, se.gate_proj.weight, se.up_proj.weight), se.down_proj.weight)),
        "moe_gu_sw": bench_q(lambda: moe_gate_up_swiglu_rows(x, sw.gate_proj.weight, sw.gate_proj.scales, sw.up_proj.weight, sw.up_proj.scales, inds)),
        "moe_dn_comb": bench_q(lambda: moe_down_combine_rows(h_mid, sw.down_proj.weight, sw.down_proj.scales, inds, wts, shared=sh, residual=x)),
        "mla_512_full": bench_q(mla_full),
        "mla_qkv_a_g": bench_q(lambda: gemv_rows_multi(x, [a7.q_a_proj.weight, a7.kv_a_proj_with_mqa.weight, a7.g_proj.weight])),
        "mla_q_b": bench_q(lambda: gemv_rows(qan, a7.q_b_proj.weight)),
        "mla_prep": bench_q(lambda: mla_prep_kv_rope_rows(qf, kva, a7.kv_a_layernorm.weight, offs)),
        "mla_embed_q": bench_q(lambda: a7.embed_q(qn)),
        "mla_sdpa_512": bench_q(lambda: mla_sdpa_rows(qne, qr, kl_all, kr_all, a7.scale, [0] * C)),
        "mla_unembed": bench_q(lambda: a7.unembed_out(so)),
        "mla_gate_dense": bench_q(lambda: gemv_rows_res(mla_gate_rows(uo, mg), a7.dense.weight, x)),
        "lm_head": bench_q(lambda: gemv_rows(x, model.lm_head.weight), reps=20, batch=8),
    }

    counts = {
        "rms_norm_4096": 145,
        "kda_qkvfgb": 63,
        "kda_pre": 63,
        "kda_gdn": 63,
        "kda_o_proj_res": 63,
        "dense_mlp": 4,
        "router": 68,
        "shared_exp": 68,
        "moe_gu_sw": 68,
        "moe_dn_comb": 68,
        "mla_512_full": 9,
        "lm_head": 1,
    }
    weighted_ms = {k: round(us[k] * n / 1000.0, 2) for k, n in counts.items()}
    total_ms = round(sum(weighted_ms.values()), 2)
    return {"us": {k: round(v, 1) for k, v in us.items()}, "weighted_ms": weighted_ms, "total_weighted_ms": total_ms}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--C", default="1,4")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    t0 = time.perf_counter()
    maybe_set_recommended_wired_limit()
    model, tok = load(a.model)
    mx.eval(model.parameters())
    print(f"loaded in {time.perf_counter()-t0:.1f}s", flush=True)
    env = {k: v for k, v in os.environ.items() if k.startswith(("MLX_", "LINGSANTE_"))}
    print("env:", env, flush=True)

    Cs = [int(x) for x in a.C.split(",")]
    step_res, comp_res = {}, {}
    for C in Cs:
        per_c, last_t = [], []
        for i in range(C):
            c, lt = prefill_stream(model, prompt_ids(tok, PROMPTS[i][1]))
            per_c.append(c)
            last_t.append(lt)
        caches = per_c[0] if C == 1 else merge_caches(per_c)
        y = mx.array(last_t, dtype=mx.uint32)
        if C == 1:
            n = model(y[:, None], cache=caches)
            with tempfile.NamedTemporaryFile("w+", suffix=".dot") as f:
                mx.export_to_dot(f.name, n)
                f.seek(0)
                nodes = sum(1 for l in f.read().splitlines() if "[label" in l and "->" not in l)
            mx.eval(n)
            print(f"fastdecode C=1 graph nodes={nodes}", flush=True)
        sr = bench_step_anatomy(model, caches, C, y, steps=a.steps)
        step_res[C] = sr
        print(f"step C={C}:", json.dumps(sr), flush=True)
        del caches, per_c
        mx.clear_cache()

        cr = profile_fast_components(model, C)
        comp_res[C] = cr
        print(f"comp C={C} total_weighted={cr['total_weighted_ms']}ms:", json.dumps(cr["weighted_ms"]), flush=True)
        print(f"  mla_sub C={C} (us):", {k: v for k, v in cr["us"].items() if k.startswith("mla_")}, flush=True)

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"env": env, "steps": step_res, "components": comp_res}, indent=1))


if __name__ == "__main__":
    main()
