"""Bit-exactness test for rows/router.py on L4 & L64 real weights, captured visits, and tie stress."""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(_root / "vendor"), str(_root)]

import mlx.core as mx  # noqa: E402
from mlx_lm.models.bailing_moe_v3.layers import _group_expert_select  # noqa: E402
from mlx_lm.models.bailing_moe_v3.rows.router import gate_rows, group_router_rows  # noqa: E402

from bench._layer_loader import bit_diff_count, load_layer, load_visits  # noqa: E402


def main():
    ok = True
    for L in (4, 64):
        layer, args = load_layer(L)
        gate = layer.mlp.gate
        visits = load_visits(L, limit=8)

        for B in (1, 2, 3, 4):
            xm = mx.concatenate([v["mlp_in"][0, 0:1] for v in visits[:B]], axis=0)
            ref_i_list, ref_w_list = [], []
            for b in range(B):
                ri, rw = gate(xm[b : b + 1])
                ref_i_list.append(ri)
                ref_w_list.append(rw)
            ref_i = mx.concatenate(ref_i_list, axis=0).astype(mx.uint32)
            ref_w = mx.concatenate(ref_w_list, axis=0)

            got_i, got_w = gate_rows(
                xm, gate.weight, gate.expert_bias,
                top_k=gate.top_k, n_group=gate.n_group, topk_group=gate.topk_group,
                routed_scaling_factor=gate.routed_scaling_factor,
            )
            di = bit_diff_count(got_i, ref_i)
            dw = bit_diff_count(got_w, ref_w)
            ok &= (di == 0 and dw == 0)
            print(f"L{L} B={B} real: inds_diff={di} wts_diff={dw} {'PASS' if di==0 and dw==0 else 'FAIL'}")
            if di != 0 or dw != 0:
                print("  ref_i[0]:", ref_i[0].tolist(), "got_i[0]:", got_i[0].tolist())
                print("  ref_w[0]:", ref_w[0].tolist(), "got_w[0]:", got_w[0].tolist())

        # Tie & random stress
        mx.random.seed(42)
        logits_rand = mx.random.normal((4, 512)) * 2.0
        # Inject exact ties in row 0 and group-score ties in row 1
        l_np = logits_rand.tolist()
        for e in range(0, 16):
            l_np[0][e] = 0.5
        for g in range(8):
            l_np[1][g * 64] = 1.0
            l_np[1][g * 64 + 1] = 1.0
        logits_tie = mx.array(l_np, dtype=mx.float32)
        bias_zero = mx.zeros((512,), dtype=mx.float32)
        for name, b_arr in [("real_bias", gate.expert_bias), ("zero_bias", bias_zero)]:
            ri_l, rw_l = [], []
            for b in range(4):
                ri, rw = _group_expert_select(logits_tie[b : b + 1], b_arr, 8, 8, 4, 2.5)
                ri_l.append(ri)
                rw_l.append(rw)
            ref_i = mx.concatenate(ri_l, axis=0).astype(mx.uint32)
            ref_w = mx.concatenate(rw_l, axis=0)
            got_i, got_w = group_router_rows(logits_tie, b_arr)
            di = bit_diff_count(got_i, ref_i)
            dw = bit_diff_count(got_w, ref_w)
            ok &= (di == 0 and dw == 0)
            print(f"L{L} tie ({name}): inds_diff={di} wts_diff={dw} {'PASS' if di==0 and dw==0 else 'FAIL'}")

    print("SUMMARY:", "ALL PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
