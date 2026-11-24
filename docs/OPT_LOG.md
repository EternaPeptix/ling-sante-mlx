# LingSante-3.1-flash-MXFP4 Optimization Log (`512S1`, M3 Ultra)

Hardware: Apple M3 Ultra (`512S1`, 512 GB unified memory), MLX `0.32.4`, model weights ~301 GiB (`323.6 GB` active in `BatchGenerator`).
Invariant: **0 bit diffs** on logits and **100% identical greedy tokens** vs stock `B=1` baseline (`~/science/runs/baseline.json`, 4 prompts, 128 tokens, 32-step logprob SHA-256) for `C in {1, 2, 3, 4}`.

## Stock Baseline (Step 5)

### `profile_decode.py` (`profile_stock.json` vs `profile_gpuonly.json`)

| Mode / Env | C | agg tok/s | async ms/step (build + encode + wait) | sync ms/step | sync cpu ms |
|---|---:|---:|---:|---:|---:|
| Stock default cmdbuf | 1 | 14.33 | 69.77 (3.65 + 66.12 + 0.01) | 72.93 | 37.01 |
| Stock default cmdbuf | 4 | 42.95 | 93.13 (3.73 + 89.40 + 0.00) | 96.84 | 40.37 |
| `MLX_MAX_OPS_PER_BUFFER=100000` | 1 | 15.13 | 66.11 (4.86 + 11.39 + 49.88) | 80.87 | 15.35 |
| `MLX_MAX_OPS_PER_BUFFER=100000` | 4 | 45.56 | 87.80 (5.10 + 12.66 + 70.02) | 102.92 | 16.27 |

Stock isolated layer/op timings (`C=1` / `C=4`, ms, includes `mx.eval` sync):
- `L0_kda_dense`: 1.274 / 1.348 (default), 1.384 / 1.371 (`100000`)
- `L4_kda_moe`: 1.207 / 1.563 (default), 1.378 / 1.660 (`100000`)
- `L4_qkvfgb_proj`: 0.706 / 0.925 (default), 0.710 / 0.726 (`100000`)
- `L4_gdn`: 0.220 / 0.239 (default), 0.219 / 0.222 (`100000`)
- `L4_o_proj`: 0.256 / 0.282 (default), 0.274 / 0.244 (`100000`)
- `L4_gate`: 0.330 / 0.336 (default), 0.342 / 0.344 (`100000`)
- `L4_switch_mlp`: 0.326 / 0.637 (default), 0.313 / 0.597 (`100000`)
- `L4_shared_experts`: 0.242 / 0.238 (default), 0.236 / 0.204 (`100000`)
- `L7_mla_attn_512`: 0.578 / 0.680 (default), 0.572 / 0.683 (`100000`)
- `lm_head`: 2.103 / 2.078 (default), 2.092 / 2.080 (`100000`)

### `BatchGenerator` stock (`batchgen_stock.json`, `MLX_MAX_OPS_PER_BUFFER=100000`, 128 tokens, 3 reps)

| C | agg tok/s | ms/step | peak GB | exact_tok | exact_lp32 | first_tok_diff | first_lp_diff |
|---:|---:|---:|---:|:---:|:---:|---|---|
| 1 | 15.049 | 66.452 | 329.12 | True | True | `{}` | `{}` |
| 2 | 27.768 | 72.027 | 333.60 | False | False | `{0: 52}` | `{0: 0, 1: 0}` |
| 3 | 37.890 | 79.177 | 338.18 | False | False | `{0: 48, 1: 80}` | `{0: 0, 1: 0, 2: 0}` |
| 4 | 46.028 | 86.903 | 358.97 | False | False | `{0: 48, 3: 60}` | `{0: 0, 1: 0, 2: 0, 3: 0}` |

### `LINGSANTE_CMDBUF` sweep (`batch_dec.py --C 1,4 --tokens 64 --reps 2 --mode batchgen`)

| `MLX_MAX_OPS_PER_BUFFER` | C=1 agg tok/s (ms/step) | C=4 agg tok/s (ms/step) | C=4 peak GB |
|---:|---:|---:|---:|
| 200 | 15.07 (66.34 ms) | 45.64 (87.64 ms) | 337.5 |
| **1000 (default)** | **15.20 (65.77 ms)** | **46.38 (86.25 ms)** | **354.3** |
| 100000 | 15.23 (65.66 ms) | 46.37 (86.26 ms) | 359.0 |

## Wave 1: Multi-row bf16 GEMV & Group Router (`rows/gemv.py`, `rows/router.py`, `rows/glue.py`)

Verified **0 bit diffs** (`SUMMARY: ALL PASS` on `rows_gemv_test.py` and `rows_router_test.py`, `B=1..4`).

Queued microbenchmarks (`bench/rows_gemv_speed.py`, `bench/rows_router_speed.py`, us/call):

| Kernel | B=1 (ref -> rows) | B=2 (ref -> rows) | B=3 (ref -> rows) | B=4 (ref -> rows) |
|---|---:|---:|---:|---:|
| KDA `qkvfgb` (6 projs) | 490.2 -> 499.6 (0.98x) | 488.0 -> 505.8 (0.96x) | 508.2 -> 506.4 (1.00x) | 640.5 -> 504.8 (1.27x) |
| KDA `o_proj + res` | 52.8 -> 55.0 (0.96x) | 74.2 -> 52.5 (1.41x) | 76.7 -> 53.8 (1.43x) | 72.2 -> 54.6 (1.32x) |
| Shared expert (`L4`) | 28.9 -> 26.8 (1.08x) | 32.3 -> 27.5 (1.17x) | 34.5 -> 32.4 (1.07x) | 38.4 -> 35.4 (1.08x) |
| Dense MLP (`L0`) | 303.0 -> 323.7 (0.94x) | 359.9 -> 324.5 (1.11x) | 383.2 -> 318.7 (1.20x) | 538.7 -> 318.3 (1.69x) |
| Router (`gate_rows` on `L4`) | 48.7 -> 14.6 (3.33x) | 59.9 -> 14.6 (4.10x) | 60.0 -> 14.7 (4.10x) | 60.2 -> 14.8 (4.06x) |
| `lm_head` | 1861.3 -> 1858.3 (1.00x) | 1882.0 -> 1876.9 (1.00x) | 2370.8 -> 1895.1 (1.25x) | 2664.9 -> 1898.7 (1.40x) |

## Wave 2: Fused MXFP4 MoE (`rows/gather.py`, `rows/moe.py`)

Verified **0 bit diffs** (`SUMMARY: ALL PASS` on `rows_moe_test.py`, 60/60 cases across `L4`, `L64`, `B=1..4`, including heavy duplicate-expert collision stress).
Eliminated FP32 dequant register spill (`wv[ROWS][16]` -> packed `uint32_t rg/ru[ROWS][PPT]` + `TMAX_CAP=1`, `nsg=8 if B==1 else 4`), improving combined MoE + shared + residual from `251–672 us` to `140.7–307.9 us`:

| Config | B=1 (`uniq=8`) | B=2 (`uniq=12`) | B=3 (`uniq=16`) | B=4 (`uniq=17`) |
|---|---:|---:|---:|---:|
| Stock SwitchGLU + combine + shared | 161.1 us | 194.3 us | 243.8 us | 303.3 us |
| `rows/moe.py` (`TMAX_CAP=1`, best `nsg`) | 140.7 us (1.15x, `nsg=8`) | 185.2 us (1.05x, `nsg=8`) | 264.4 us (0.92x, `nsg=4`) | 310.4 us (`nsg=4`; `307.9 us` / 1.01x at `TMAX=2`) |

## Wave 3: Fused KDA & Row Norms (`rows/kda.py`, `rows/norm.py`)

Verified **0 bit diffs** (`SUMMARY: ALL PASS` on `bench/rows_kda_test.py`, all 122 checks on `L0`, `L4`, `L64` across `B=1..4`: `rms_norm_rows`, `add_rms_norm_rows`, `kda_attn_out`, `kda_h_post`, `ns0..ns3`, and fused vs unfused GDN).

Queued `L4` KDA microbenchmarks (`bench/rows_kda_speed.py`, us/call):
- `kda_pre_rows` (3x `Conv1d(K=4)` + state shift + `silu` + `normalize_qk` + `compute_lower_bound_g` + `beta`): `31.0–31.2 us` (`B=1..4`).
- `kda_gdn_norm_gate_rows(fused=True)` (packed vector-gate GDN + `o_norm` + `sigmoid(g_proj)`): `23.7 us (B=1)`, `28.1 us (B=2)`, `34.4 us (B=3)`, `47.8 us (B=4)`.
- Full `L4` KDA attention (`stock` vs `rows_fused`): `B=1: 530.0 -> 579.7 us`, `B=2: 568.4 -> 586.5 us`, `B=3: 583.1 -> 592.5 us`, `B=4: 612.9 -> 609.0 us (1.01x)` (~35 ops collapsed into 4 launches).

## Wave 4: MLA (`rows/mla.py`), `fastdecode.py`, & Gate #1 (`MLX_MAX_OPS_PER_BUFFER=1000`)

Verified **0 bit diffs** (`SUMMARY: ALL PASS` on `bench/rows_mla_test.py`, 38/38 cases on `L7` & `L71` visits across `B=1..4`, heterogeneous `BatchKVCache`, and synthetic `S=512` / `S=2048` hetero contexts).
- Root cause for `BatchGenerator`: `PromptProcessingBatch` (`prefill_batch_size > 1`) pads heterogeneous prompts in `L > 1` prefill (not bit-exact across `B=1` vs `B>1`), and `ArraysCache.advance()` sets negative `left_padding`. Updating `fastdecode.can_fast_decode` to allow `ArraysCache.left_padding <= 0` (cached per `id(lp)`) and using `prefill_batch_size=1` makes `BatchGenerator` **100% bit-exact** at `C=1..4`.

Gate #1 (`bench/batch_dec.py --C 1,2,3,4 --tokens 128 --reps 3 --mode both`, `batch_w4_exact.json`):

| Mode | C=1 tok/s (ms/step) | C=2 tok/s (ms/step) | C=3 tok/s (ms/step) | C=4 tok/s (ms/step) | All exact_tok & exact_lp32 |
|---|---:|---:|---:|---:|:---:|
| `model` | 16.44 (60.84 ms) | 29.11 (68.71 ms) | 38.60 (77.73 ms) | 47.06 (85.00 ms) | True |
| `batchgen` | 17.01 (58.78 ms) | 29.82 (67.07 ms) | 39.58 (75.79 ms) | 47.98 (83.36 ms) | True |

`profile_fast.py` (`C=1, 4`, `profile_fast_w4.json`):
- `C=1`: graph nodes reduced **9,432 -> 1,287** (7.3x); `async_step_ms = 59.42 ms` (`build 5.93 + encode 6.15 + wait 47.38`), `async_cpu_ms = 15.50 ms` (vs `39.37 ms` stock).
- Weighted breakdown `C=1` (`total_weighted = 59.03 ms`): `kda_qkvfgb 32.73 ms`, `kda_o_proj_res 3.84 ms`, `moe_gu_sw 3.49 ms`, `rms_norm_4096 2.80 ms`, `moe_dn_comb 2.61 ms`, `mla_512_full 2.26 ms`, `kda_pre 2.23 ms`, `shared_exp 2.17 ms`, `kda_gdn 1.90 ms`, `lm_head 1.86 ms`, `router 1.80 ms`, `dense_mlp 1.34 ms`.
- Weighted breakdown `C=4` (`total_weighted = 83.08 ms`): `kda_qkvfgb 32.50 ms`, `moe_gu_sw 18.63 ms`, `moe_dn_comb 8.60 ms`, `mla_512_full 4.19 ms`, `kda_o_proj_res 3.70 ms`, `kda_gdn 3.07 ms`, `shared_exp 2.67 ms`, `rms_norm_4096 2.56 ms`, `kda_pre 2.21 ms`, `router 1.82 ms`, `lm_head 1.86 ms`, `dense_mlp 1.27 ms`.

### `LINGSANTE_CMDBUF` re-sweep on wired fast decode (`batch_dec.py --C 1,4 --tokens 64 --reps 2 --mode model`)

| `MLX_MAX_OPS_PER_BUFFER` | C=1 tok/s (ms/step) | C=4 tok/s (ms/step) | C=4 peak GB |
|---:|---:|---:|---:|
| 32 | 16.27 (61.45 ms) | 46.65 (85.74 ms) | 326.2 |
| **64 (final default)** | **16.29 (61.39 ms)** | **46.72 (85.61 ms)** | **326.5** |
| 128 | 16.27 (61.45 ms) | 46.70 (85.65 ms) | 327.1 |
| 256 | 16.21 (61.71 ms) | 46.64 (85.77 ms) | 328.2 |
| 512 | 16.10 (62.10 ms) | 46.44 (86.14 ms) | 328.2 |
| 1000 | 15.88 (62.98 ms) | 46.06 (86.84 ms) | 328.2 |
| 100000 | 15.83 (63.19 ms) | 45.73 (87.48 ms) | 328.2 |

## Wave 5: Final Targeted Optimizations & Gate #2

Changes applied (all verified **0 bit diffs**):
1. `rows/gemv.py`: compile-time `KK` template, `Bf4` loads on `in_vec`, single-call-site pointer selection in `_build_group`, `BM=8` confirmed optimal (`rows_gemv_test.py ALL PASS`).
2. `rows/moe.py`: two-pass single-stream `moe_gate_up_swiglu` (`_GU_TMAX_CAP=4`) eliminating duplicate expert weight reads cleanly without register spills + `_DC_TMAX_CAP=1` with `nsg = 4 if B <= 2 else 2` on `moe_down_combine_rows` (`rows_moe_test.py ALL PASS`):
   - `B=1 (uniq=8/8)`: `162.6 -> 136.8 us` (**1.19x** vs stock)
   - `B=2 (uniq=12/16)`: `192.7 -> 173.8 us` (**1.11x**)
   - `B=3 (uniq=16/24)`: `243.0 -> 248.4 us` (`0.98x`)
   - `B=4 (uniq=17/32)`: `302.2 -> 286.6 us` (**1.05x**)
3. `fastdecode.py` & `rows/mla.py`: switched `rms_norm` to `mx.fast.rms_norm` (`10.8 us` vs `19.3 us`, invariant across `B`), added weights-read-once `mla_multilinear_rows` with fused sigmoid gate onto `unembed_out`, kept per-row unpadded `mla_sdpa_rows` (confirmed batched MLA SDPA introduces 9–111 bit diffs vs `B=1` at S=82..2048) (`rows_mla_test.py ALL PASS` 38/38):
   - `S=82`: `B=1: 204.5 -> 187.5 us` (**1.09x**), `B=2: 258.5 -> 240.2 us` (**1.08x**)
   - `S=512`: `B=1: 215.8 -> 199.7 us` (**1.08x**), `B=2: 288.6 -> 265.9 us` (**1.09x**)
4. Updated `LINGSANTE_CMDBUF` default in `vendor/mlx_lm/__init__.py` to `MLX_MAX_OPS_PER_BUFFER=64`.

### Final Gate #2 (`bench/batch_dec.py --C 1,2,3,4 --tokens 128 --reps 3 --mode both`, `batch_w5_final.json`)

| Mode | C | agg tok/s | per-stream tok/s | ms/step | peak GB | exact_tok | exact_lp32 |
|---|---:|---:|---:|---:|---:|:---:|:---:|
| `model` | 1 | **17.12** | 17.12 | 58.42 | 324.1 | True | True |
| `model` | 2 | **30.23** | 15.11 | 66.16 | 325.1 | True | True |
| `model` | 3 | **39.84** | 13.28 | 75.30 | 325.8 | True | True |
| `model` | 4 | **48.79** | 12.20 | 81.98 | 326.5 | True | True |
| `batchgen` | 1 | **17.44** | 17.44 | 57.34 | 324.7 | True | True |
| `batchgen` | 2 | **30.74** | 15.37 | 65.06 | 324.7 | True | True |
| `batchgen` | 3 | **40.42** | 13.47 | 74.23 | 325.2 | True | True |
| `batchgen` | 4 | **49.30** | 12.32 | 81.14 | 326.3 | True | True |

Final `profile_fast.py` (`profile_fast_w5.json`, `NOPS=64`): `1,260` graph nodes; `C=1`: `17.37 tok/s` (`57.57 ms/step` async, `async_cpu 14.53 ms`); `C=4`: `49.12 tok/s` (`81.44 ms/step` async, `async_cpu 16.40 ms`).

## End-to-End Verification & Serving

- `bench/server_exact_test.py` (`BatchGenerator` dynamic concurrency `1 -> 4 -> 2 -> 3 -> 1`, 5 requests): all `exact_tok=True`, `exact_lp=True`, `SUMMARY: ALL PASS`.
- `bench/ctx_exact_test.py --ctx 2048 --tokens 32`: `C=1` (`15.56 tok/s`, sync step) and `C=4` (`44.92 tok/s`, sync step) vs `LINGSANTE_FAST_DECODE=0` `B=1` -> `exact_tok=True`, `exact_lp=True`, `SUMMARY: ALL PASS`.
- `tools/run_eval.py gen`: all 6 greedy prompts at `17.4 tok/s` (up from `14.3–14.4 tok/s`).
- `tools/run_eval.py ppl`: `chunk 64: mean NLL 0.04473, ppl 1.0458`; `chunk 512: mean NLL 0.04646, ppl 1.0476`; `ppl ratio chunk64/chunk512 = 0.99828` (identical to stock).
- Live HTTP `mlx_lm.server` (`512S1:8080`, `--decode-concurrency 4 --prompt-concurrency 1`): `bench/http_smoke.py` (`max_tokens=128`, `temperature=0.0`): `C=1` end-to-end (incl. prompt processing) `14.28–14.90 tok/s` (`72–128` tok); `C=4` concurrent `423 tok in 11.13s (38.00 agg tok/s, incl. staggered stream completions at 72/95/128 tok)`, `exact_vs_C1=True`.
