"""Bit-exactness test for rows/norm.py and rows/kda.py across L0, L4, L64 captured visits (B = 1..4)."""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(_root / "vendor"), str(_root)]

import mlx.core as mx  # noqa: E402
from mlx_lm.models.bailing_moe_v3.rows.gemv import gemv_rows_multi, gemv_rows_res  # noqa: E402
from mlx_lm.models.bailing_moe_v3.rows.kda import kda_gdn_norm_gate_rows, kda_pre_rows  # noqa: E402
from mlx_lm.models.bailing_moe_v3.rows.norm import add_rms_norm_rows, rms_norm_rows  # noqa: E402
from mlx_lm.models.cache import ArraysCache  # noqa: E402

from bench._layer_loader import bit_diff_count, load_layer, load_visits  # noqa: E402


def check(name, got, ref):
    d = bit_diff_count(got, ref)
    print(f"  {name:44s} {str(got.shape):18s} {'PASS' if d == 0 else f'FAIL ({d}/{ref.size})'}")
    return d == 0


def run_kda_layer(layer, x, cs0, cs1, cs2, cs3, mask=None):
    attn = layer.attention
    ws = [attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight, attn.f_proj.weight, attn.g_proj.weight, attn.b_proj.weight]
    q_r, k_r, v_r, f_r, g_r, b_r = gemv_rows_multi(x, ws)
    q, k, v, g, beta, ns0, ns1, ns2 = kda_pre_rows(
        q_r, k_r, v_r, f_r, b_r,
        cs0, cs1, cs2,
        attn.q_conv1d.weight, attn.k_conv1d.weight, attn.v_conv1d.weight,
        attn.A_log, attn.dt_bias,
        lower_bound=attn.lower_bound, eps=1e-6, mask=mask,
    )
    out_v, ns3 = kda_gdn_norm_gate_rows(q, k, v, g, beta, cs3, g_r, attn.o_norm.weight, eps=attn.o_norm.eps)
    return out_v, ns0, ns1, ns2, ns3


def main():
    ok = True
    for L in (0, 4, 64):
        print(f"=== KDA Layer L{L} ===")
        layer, _ = load_layer(L)
        visits = load_visits(L, limit=8)

        for B in (1, 2, 3, 4):
            x_raw = mx.concatenate([v["x"][0, 0:1] for v in visits[:B]], axis=0)  # (B, 4096)
            cs0 = mx.concatenate([v["cs0"] for v in visits[:B]], axis=0)
            cs1 = mx.concatenate([v["cs1"] for v in visits[:B]], axis=0)
            cs2 = mx.concatenate([v["cs2"] for v in visits[:B]], axis=0)
            cs3 = mx.concatenate([v["cs3"] for v in visits[:B]], axis=0)

            # Norm checks
            ref_hin = mx.concatenate([layer.input_layernorm(x_raw[b : b + 1]) for b in range(B)], axis=0)
            got_hin = rms_norm_rows(x_raw, layer.input_layernorm.weight, layer.input_layernorm.eps)
            ok &= check(f"L{L} B={B} rms_norm_rows(input)", got_hin, ref_hin)

            # Attention B=1 reference
            ref_aout, ref_hpost = [], []
            ref_ns = [[], [], [], []]
            for b in range(B):
                c = ArraysCache(size=4)
                c[0] = mx.array(cs0[b : b + 1])
                c[1] = mx.array(cs1[b : b + 1])
                c[2] = mx.array(cs2[b : b + 1])
                c[3] = mx.array(cs3[b : b + 1])
                ao = layer.attention(ref_hin[b : b + 1, None, :], mask=None, cache=c)
                ref_aout.append(ao[:, 0, :])
                ref_hpost.append(x_raw[b : b + 1] + ao[:, 0, :])
                for j in range(4):
                    ref_ns[j].append(c[j])

            ref_aout = mx.concatenate(ref_aout, axis=0)
            ref_hpost = mx.concatenate(ref_hpost, axis=0)
            ref_ns = [mx.concatenate(ref_ns[j], axis=0) for j in range(4)]

            # Check visit snapshots too
            ref_v_aout = mx.concatenate([v["attn_out"][0, 0:1] for v in visits[:B]], axis=0)
            ok &= check(f"L{L} B={B} stock_vs_visit(attn_out)", ref_aout, ref_v_aout)

            # Batched fused KDA
            from mlx_lm.models.bailing_moe_v3.rows.gemv import gemv_rows
            out_v, ns0, ns1, ns2, ns3 = run_kda_layer(layer, got_hin, cs0, cs1, cs2, cs3)
            got_aout = gemv_rows(out_v, layer.attention.o_proj.weight)
            got_hpost = gemv_rows_res(out_v, layer.attention.o_proj.weight, x_raw)

            ok &= check(f"L{L} B={B} kda_attn_out", got_aout, ref_aout)
            ok &= check(f"L{L} B={B} kda_h_post", got_hpost, ref_hpost)
            ok &= check(f"L{L} B={B} kda_ns0", ns0, ref_ns[0])
            ok &= check(f"L{L} B={B} kda_ns1", ns1, ref_ns[1])
            ok &= check(f"L{L} B={B} kda_ns2", ns2, ref_ns[2])
            ok &= check(f"L{L} B={B} kda_ns3", ns3, ref_ns[3])

            # Test add_rms_norm_rows for post_attention_layernorm
            ref_mlpin = mx.concatenate([layer.post_attention_layernorm(ref_hpost[b : b + 1]) for b in range(B)], axis=0)
            got_hp2, got_mlpin = add_rms_norm_rows(
                x_raw, got_aout, layer.post_attention_layernorm.weight, layer.post_attention_layernorm.eps
            )
            ok &= check(f"L{L} B={B} add_rms_norm(h_post)", got_hp2, ref_hpost)
            ok &= check(f"L{L} B={B} add_rms_norm(mlp_in)", got_mlpin, ref_mlpin)

    # Standalone unfused GDN check vs fused on L4 B=4
    l4, _ = load_layer(4)
    v4 = load_visits(4, limit=4)
    x4 = mx.concatenate([v["h_in"][0, 0:1] for v in v4], axis=0)
    cs = [mx.concatenate([v[f"cs{j}"] for v in v4], axis=0) for j in range(4)]
    a4 = l4.attention
    q_r, k_r, v_r, f_r, g_r, b_r = gemv_rows_multi(
        x4, [a4.q_proj.weight, a4.k_proj.weight, a4.v_proj.weight, a4.f_proj.weight, a4.g_proj.weight, a4.b_proj.weight]
    )
    q, k, v, g, beta, _, _, _ = kda_pre_rows(
        q_r, k_r, v_r, f_r, b_r, cs[0], cs[1], cs[2],
        a4.q_conv1d.weight, a4.k_conv1d.weight, a4.v_conv1d.weight,
        a4.A_log, a4.dt_bias, lower_bound=a4.lower_bound, eps=1e-6,
    )
    out_fused, st_fused = kda_gdn_norm_gate_rows(q, k, v, g, beta, cs[3], g_r, a4.o_norm.weight, eps=a4.o_norm.eps, fused=True)
    out_unfused, st_unfused = kda_gdn_norm_gate_rows(q, k, v, g, beta, cs[3], g_r, a4.o_norm.weight, eps=a4.o_norm.eps, fused=False, nsg=2)
    out_unfused_norm = (rms_norm_rows(out_unfused.reshape(-1, 128), a4.o_norm.weight, a4.o_norm.eps).reshape(4, 8192)
                        * mx.sigmoid(g_r.astype(mx.float32)).astype(out_unfused.dtype))
    ok &= check("L4 B=4 gdn fused vs unfused+norm", out_fused, out_unfused_norm)
    ok &= check("L4 B=4 gdn state fused vs unfused", st_fused, st_unfused)

    print("SUMMARY:", "ALL PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
