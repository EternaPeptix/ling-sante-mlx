"""Bit-exactness test for rows/mla.py on L7 & L71 captured visits, heterogeneous BatchKVCache, and long S=512/2048."""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(_root / "vendor"), str(_root)]

import mlx.core as mx  # noqa: E402
from mlx_lm.models.bailing_moe_v3.rows.gemv import gemv_rows, gemv_rows_multi  # noqa: E402
from mlx_lm.models.bailing_moe_v3.rows.mla import (  # noqa: E402
    mla_gate_rows,
    mla_layer_rows,
    mla_prep_kv_rope_rows,
)
from mlx_lm.models.bailing_moe_v3.rows.norm import rms_norm_rows  # noqa: E402
from mlx_lm.models.cache import BatchKVCache, KVCache  # noqa: E402

from bench._layer_loader import bit_diff_count, load_layer, load_visits  # noqa: E402


def check(name, got, ref):
    d = bit_diff_count(got, ref)
    print(f"  {name:46s} {str(got.shape):16s} {'PASS' if d == 0 else f'FAIL ({d}/{ref.size})'}")
    return d == 0


def run_stock_b1(layer, x_raw, k_prevs, v_prevs):
    B = x_raw.shape[0]
    attn = layer.attention
    ref_hposts, ref_ks, ref_vs = [], [], []
    for b in range(B):
        c = KVCache()
        if k_prevs[b].shape[2] > 0:
            c.update_and_fetch(k_prevs[b], v_prevs[b])
        xb = x_raw[b : b + 1, None, :]
        hin = layer.input_layernorm(xb)
        ao = attn(hin, mask=None, cache=c)
        ref_hposts.append((xb + ao)[:, 0, :])
        k_new, v_new = c.keys_and_values()
        ref_ks.append(k_new[:, :, -1:, :])
        ref_vs.append(v_new[:, :, -1:, :])
    return (
        mx.concatenate(ref_hposts, axis=0),
        mx.concatenate(ref_ks, axis=0),
        mx.concatenate(ref_vs, axis=0),
    )


def main():
    ok = True
    for L in (7, 71):
        print(f"=== MLA Layer L{L} ===")
        layer, _ = load_layer(L)
        attn = layer.attention
        visits = load_visits(L, limit=8)

        # 1. Real captured visits B = 1 (KVCache) and B = 2, 3, 4 (heterogeneous BatchKVCache)
        for B in (1, 2, 3, 4):
            x_raw = mx.concatenate([v["x"][0, 0:1] for v in visits[:B]], axis=0)
            kp_list = [v["k_prev"] for v in visits[:B]]
            vp_list = [v["v_prev"] for v in visits[:B]]
            ref_hp, ref_kn, ref_vn = run_stock_b1(layer, x_raw, kp_list, vp_list)

            ref_v_hp = mx.concatenate([v["h_post"][0, 0:1] for v in visits[:B]], axis=0)
            ok &= check(f"L{L} B={B} stock_vs_visit(h_post)", ref_hp, ref_v_hp)

            if B == 1:
                cache = KVCache()
                cache.update_and_fetch(kp_list[0], vp_list[0])
            else:
                cs = []
                for b in range(B):
                    cb = KVCache()
                    cb.update_and_fetch(kp_list[b], vp_list[b])
                    cs.append(cb)
                cache = BatchKVCache.merge(cs)

            hin = rms_norm_rows(x_raw, layer.input_layernorm.weight, layer.input_layernorm.eps)
            got_hp = mla_layer_rows(hin, x_raw, attn, cache)
            k_all, v_all = cache.keys_and_values()
            got_kn = k_all[:, :, -1:, :]
            got_vn = v_all[:, :, -1:, :]
            ok &= check(f"L{L} B={B} mla_h_post", got_hp, ref_hp)
            ok &= check(f"L{L} B={B} mla_k_last", got_kn, ref_kn)
            ok &= check(f"L{L} B={B} mla_v_last", got_vn, ref_vn)

    # 2. Synthetic long context (S=512, S=2048) with equal and heterogeneous left padding on L7
    l7, _ = load_layer(7)
    attn = l7.attention
    mx.random.seed(99)
    for tag, lengths in [("S=512_hetero", [512, 300, 450, 128]), ("S=2048_hetero", [2048, 1500, 2048, 1024])]:
        print(f"=== Long Context {tag} ===")
        B = 4
        x_raw = mx.random.normal((B, 4096)).astype(mx.bfloat16) * 0.25
        kp_list = [mx.random.normal((1, 1, s, 512)).astype(mx.bfloat16) * 0.1 for s in lengths]
        vp_list = [mx.random.normal((1, 1, s, 64)).astype(mx.bfloat16) * 0.1 for s in lengths]
        mx.eval(x_raw, *kp_list, *vp_list)

        ref_hp, ref_kn, ref_vn = run_stock_b1(l7, x_raw, kp_list, vp_list)
        cs = []
        for b in range(B):
            cb = KVCache()
            cb.update_and_fetch(kp_list[b], vp_list[b])
            cs.append(cb)
        bc = BatchKVCache.merge(cs)
        hin = rms_norm_rows(x_raw, l7.input_layernorm.weight, l7.input_layernorm.eps)
        got_hp = mla_layer_rows(hin, x_raw, attn, bc)
        k_all, v_all = bc.keys_and_values()
        ok &= check(f"{tag} B=4 mla_h_post", got_hp, ref_hp)
        ok &= check(f"{tag} B=4 mla_k_last", k_all[:, :, -1:, :], ref_kn)
        ok &= check(f"{tag} B=4 mla_v_last", v_all[:, :, -1:, :], ref_vn)

    print("SUMMARY:", "ALL PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
