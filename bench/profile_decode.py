"""Profile LingSante stock decode anatomy at C=1..4 on 512S1.

Measures:
  1. C=1 sync/async step time, build vs encode vs GPU wait, graph node count.
  2. Layer-type breakdown: KDA dense (L0-3), KDA MoE (sample & x59), MLA MoE (sample & x9), norm+lm_head, logsumexp+argmax.
  3. Sub-component breakdown on L4 (KDA MoE) and L7 (MLA MoE) at C=1 and C=4.
"""
import argparse
import json
import os
import resource
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
from mlx_lm.layers.attention import create_attention_mask  # noqa: E402
from mlx_lm.layers.recurrent import create_ssm_mask, gated_delta_update, normalize_qk  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache  # noqa: E402
from mlx_lm.utils import maybe_set_recommended_wired_limit  # noqa: E402

from bench.batch_dec import merge_caches, prefill_stream  # noqa: E402
from bench.exact import DEFAULT_MODEL, PROMPTS, prompt_ids  # noqa: E402


def cpu_time():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def bench_step_anatomy(model, caches, C: int, y_init: mx.array, steps: int = 24):
    def step_fn(toks):
        lg = model(toks[:, None], cache=caches)[:, -1, :]
        lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
        return mx.argmax(lp, axis=-1)

    y = y_init
    for _ in range(4):
        y = step_fn(y)
        mx.eval(y)

    # Sync breakdown
    b_t, e_t = [], []
    c0 = cpu_time()
    w0 = time.perf_counter()
    for _ in range(steps):
        t0 = time.perf_counter()
        n = step_fn(y)
        t1 = time.perf_counter()
        mx.eval(n)
        t2 = time.perf_counter()
        b_t.append(t1 - t0)
        e_t.append(t2 - t1)
        y = n
    sync_ms = 1e3 * (time.perf_counter() - w0) / steps
    sync_cpu_ms = 1e3 * (cpu_time() - c0) / steps

    # Async pipelined breakdown
    B, E, W, T = [], [], [], []
    c0 = cpu_time()
    n = step_fn(y)
    mx.async_eval(n)
    for _ in range(steps):
        t0 = time.perf_counter()
        n2 = step_fn(n)
        t1 = time.perf_counter()
        mx.async_eval(n2)
        t2 = time.perf_counter()
        n.tolist()
        t3 = time.perf_counter()
        n = n2
        B.append(t1 - t0)
        E.append(t2 - t1)
        W.append(t3 - t2)
        T.append(t3 - t0)
    async_cpu_ms = 1e3 * (cpu_time() - c0) / steps
    m = lambda v: round(1e3 * st.median(v), 2)
    step_ms = m(T)
    return {
        "C": C,
        "agg_tok_s": round(C * 1000.0 / step_ms, 2),
        "async_step_ms": step_ms,
        "build_ms": m(B),
        "encode_ms": m(E),
        "wait_ms": m(W),
        "async_cpu_ms": round(async_cpu_ms, 2),
        "sync_step_ms": round(sync_ms, 2),
        "sync_build_ms": m(b_t),
        "sync_eval_ms": m(e_t),
        "sync_cpu_ms": round(sync_cpu_ms, 2),
    }


def timed_op(fn, reps=25, warm=5):
    ts = []
    for i in range(warm + reps):
        t0 = time.perf_counter()
        out = fn()
        mx.eval(out)
        if i >= warm:
            ts.append(time.perf_counter() - t0)
    return round(1e3 * st.median(ts), 3)


def profile_single_layers(model, C: int):
    # Profile isolated L0 (KDA dense), L4 (KDA MoE), L7 (MLA MoE), lm_head on dummy B=C caches
    x = mx.random.normal((C, 1, model.args.hidden_size)).astype(mx.bfloat16)
    mx.eval(x)

    def make_kda_cache():
        c = model.make_cache()[4]
        for j in range(3):
            c[j] = mx.zeros((C, 3, model.args.num_attention_heads * model.args.head_dim), dtype=mx.bfloat16)
        c[3] = mx.zeros((C, model.args.num_attention_heads, model.args.head_dim, model.args.head_dim), dtype=mx.float32)
        mx.eval(c.state)
        return c

    def make_mla_cache(S=512):
        c = model.make_cache()[7]
        c.keys = mx.zeros((C, 1, S, model.args.kv_lora_rank), dtype=mx.bfloat16)
        c.values = mx.zeros((C, 1, S, model.args.qk_rope_head_dim), dtype=mx.bfloat16)
        c.offset = S
        mx.eval(c.state)
        return c

    L0, L4, L7 = model.model.layers[0], model.model.layers[4], model.model.layers[7]
    c0, c4, c7 = make_kda_cache(), make_kda_cache(), make_mla_cache(512)

    res = {}
    res["L0_kda_dense_ms"] = timed_op(lambda: L0(x, None, c0))
    res["L4_kda_moe_ms"] = timed_op(lambda: L4(x, None, c4))
    res["L7_mla_moe_ms"] = timed_op(lambda: L7(x, None, c7))

    # Sub-component breakdown on L4 (KDA MoE)
    attn4 = L4.attention
    mlp4 = L4.mlp
    res["L4_attn_ms"] = timed_op(lambda: attn4(x, None, c4))
    res["L4_qkvfgb_proj_ms"] = timed_op(
        lambda: (attn4.q_proj(x), attn4.k_proj(x), attn4.v_proj(x), attn4.f_proj(x), attn4.g_proj(x), attn4.b_proj(x))
    )
    q = mx.zeros((C, 1, attn4.num_heads, attn4.head_dim), dtype=mx.bfloat16)
    k = mx.zeros((C, 1, attn4.num_heads, attn4.head_dim), dtype=mx.bfloat16)
    v = mx.zeros((C, 1, attn4.num_heads, attn4.head_dim), dtype=mx.bfloat16)
    fg = mx.zeros((C, 1, attn4.num_heads, attn4.head_dim), dtype=mx.bfloat16)
    bb = mx.zeros((C, 1, attn4.num_heads), dtype=mx.float32)
    mx.eval(q, k, v, fg, bb)
    res["L4_gdn_ms"] = timed_op(
        lambda: gated_delta_update(
            q, k, v, fg, bb,
            attn4.A_log.reshape(attn4.num_heads, 1),
            attn4.dt_bias.reshape(attn4.num_heads, attn4.head_dim),
            state=c4[3], use_kernel=True, lower_bound=attn4.lower_bound,
        )
    )
    res["L4_o_proj_ms"] = timed_op(lambda: attn4.o_proj(q.reshape(C, 1, -1)))
    res["L4_moe_ms"] = timed_op(lambda: mlp4(x))
    res["L4_gate_ms"] = timed_op(lambda: mlp4.gate(x))
    inds, _ = mlp4.gate(x)
    mx.eval(inds)
    res["L4_switch_mlp_ms"] = timed_op(lambda: mlp4.switch_mlp(x, inds))
    res["L4_shared_experts_ms"] = timed_op(lambda: mlp4.shared_experts(x))

    # Sub-component breakdown on L7 (MLA MoE)
    attn7 = L7.attention
    res["L7_mla_attn_512_ms"] = timed_op(lambda: attn7(x, None, c7))
    res["lm_head_ms"] = timed_op(lambda: model.lm_head(x))
    return res


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
    print(f"loaded in {time.perf_counter()-t0:.1f}s, active {mx.get_active_memory()/1e9:.1f} GB", flush=True)

    env = {k: v for k, v in os.environ.items() if k.startswith(("MLX_", "LINGSANTE_"))}
    print("env:", env, flush=True)

    Cs = [int(x) for x in a.C.split(",")]
    step_results = {}
    layer_results = {}
    for C in Cs:
        per_c, last_t = [], []
        for i in range(C):
            ids = prompt_ids(tok, PROMPTS[i][1])
            c, lt = prefill_stream(model, ids)
            per_c.append(c)
            last_t.append(lt)
        caches = per_c[0] if C == 1 else merge_caches(per_c)
        y = mx.array(last_t, dtype=mx.uint32)
        if C == 1:
            n = model(y[:, None], cache=caches)
            f = tempfile.NamedTemporaryFile("w+", suffix=".dot")
            mx.export_to_dot(f.name, n)
            f.seek(0)
            txt = f.read()
            nodes = sum(1 for l in txt.splitlines() if "[label" in l and "->" not in l)
            mx.eval(n)
            print(f"C=1 graph nodes={nodes}", flush=True)
        sr = bench_step_anatomy(model, caches, C, y, steps=a.steps)
        step_results[C] = sr
        print("step:", json.dumps(sr), flush=True)
        del caches, per_c
        mx.clear_cache()

        lr = profile_single_layers(model, C)
        layer_results[C] = lr
        print(f"layers C={C}:", json.dumps(lr), flush=True)

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"env": env, "steps": step_results, "layers": layer_results}, indent=1))


if __name__ == "__main__":
    main()
