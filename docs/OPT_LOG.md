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

## Phase 1–3.5: Multi-Token Verification Kernels (`C*L <= 24`), Native MTP Layer-72 Conversion, & >= 80 tok/s Bit-Exact Speculative Decoding

### Built-in MTP Layer-72 (`model.layers.72.*`) & Multi-Token Verification (`verify_decode.py`, `rows/*`)
1. Converted native MTP Layer 72 weights via `tools/convert_mtp.py` (`512S1:~/science/models/lingsante-mlx-mxfp4-bf16/mtp/model-mtp.safetensors`, `4.2 GiB` on disk / `4.520 GB` weights, SHA-256 `18df4cf9378a21ae3dd44aa1ad9060d2f88e95a8a67002e0f9e48a230e921aec`), paired in `vendor/mlx_lm/models/bailing_moe_v3/mtp.py` with a draft-only 4-bit affine `q4_head` (while target verification keeps full bf16 `lm_head`).
2. Extended custom Metal row kernels in `vendor/mlx_lm/models/bailing_moe_v3/rows/` cleanly up to `B = C * L <= 24` (`B <= 4` path preserved bit-for-bit):
   - `rows/gemv.py`: `MAX_B = 24`, `inter[TM][TN]`-first K-loop, tuned `_cfg` tiling for `5 <= B <= 24`.
   - `rows/moe.py`: chunked >32-lane dedup, single-simdgroup CSR `moe_dedup_rows` + dedup-consuming `_gu_kernel` / `_dc_kernel` (`C*L >= 5`), `EXACT_BLK` unrolled loop on `K=1280`.
   - `rows/kda.py`: fused `kda_pre_seq_rows`, `kda_gdn_verify_seq_rows`, per-stream `kda_gdn_verify_seq_split_rows` (`ArraysCache(3 + C)` so accepted streams require zero state copy), `kda_gdn_advance_split_only`, and `kda_slice_conv_rows`.
   - `rows/mla.py`: `mla_verify_layer_rows` and single-pass `mla_verify_varlen_rows` (`list[KVCache]` unpadded exact MLA verification & O(1) rollback).
   - `verify_decode.py`: `LINGSANTE_VERIFY_FLUSH=4` mid-forward `mx.async_eval(x)` overlapping Python graph construction & Metal buffer submission with GPU execution.
3. Kernel & rollback gates (`SUMMARY: ALL PASS`, **0 bit diffs**):
   - `bench/verify_kernels_test.py`: GEMV `B=1..24`, MoE `B=1..24` + CSR dedup + `EXACT_BLK` + all-duplicate collision stress, split KDA + `adv_split` `M=1..4`, and varlen MLA `Ls=[3,1,2,1]` / `[2,3,1,3]` — `SUMMARY: ALL PASS`.
   - `bench/verify_model_bench.py` (`verify_bench_v4.json`, `exact_ok=True` across adversarial rollback):
     * `C=1 L=1 (B=1)`: `58.14 ms` (`17.14 tok/s`, `commit_full=0.195 ms`, `324.45 GB`).
     * `C=1 L=12 (B=12)`: `106.29 ms` (`ceil 112.69 tok/s`, `commit_full=0.196 ms, rollback_half=1.694 ms`).
     * `C=1 L=16 (B=16)`: `128.13 ms` (`ceil 124.68 tok/s`).
     * `C=4 L=2 (B=8)`: `88.50 ms` (`ceil 90.11 tok/s`, `rollback_half=4.445 ms`).
     * **`C=4 L=3 (B=12)`**: **`105.73 ms`** (`ceil 113.20 tok/s`, `commit_full=0.275 ms`, `rollback_half=4.422 ms`, `328.30 GB`).
     * `C=4 L=4 (B=16)`: `129.38 ms` (`ceil 123.40 tok/s`).

### Native MTP vs External `Ling-3.0-tiny` Drafter (`spec_async_final.json` vs `spec_tiny_phase3.json`, all `td=0, ld=0`)
- Standalone `inclusionAI/Ling-3.0-tiny` 4-bit MLX (`c1=172.34 tok/s`, `c4=433.20 agg tok/s` / `9.23 ms/step`) used as external drafter (`bench/spec_tiny_bench.py`, `spec_tiny_phase3.json`):
  * `C=1 tiny K=4`: `22.75 tok/s` (`E_acc=2.811`, `verify=81.4 ms`, `draft=32.5 ms`); hybrid `MTP+tiny K=4`: `25.99 tok/s` (`E_acc=3.243`).
  * `C=4 tiny K=2`: `43.22 agg tok/s` (`E_acc=1.764`, `verify=122.77 ms`, `draft=39.24 ms`, `338.89 GB`).
- Single-sync GPU-chained **native MTP Layer 72** (`bench/spec_async_bench.py`, `spec_async_final.json`, all `exact=True (td=0, ld=0)`):
  * `C=1 K=2`: `33.52 tok/s` (`E_acc=2.446`, `72.46 ms`); `K=3`: `34.05 tok/s` (`E_acc=2.733`, `78.02 ms`); **`K=4`: `34.06 tok/s`** (`E_acc=2.819`, `81.45 ms`); `256tok K=4`: **`35.17 tok/s`** (`E_acc=2.876`).
  * `C=2 K=2 (B=6)`: `54.81 agg / 56.45 steady tok/s` (`E_acc=2.466`, `85.82 ms`, `332.01 GB`); `K=3 (B=8)`: `54.92 agg / 56.75 steady tok/s` (`E_acc=2.822`, `94.38 ms`).
  * `C=3 K=2 (B=9)`: `69.35 agg / 71.38 steady tok/s` (`E_acc=2.474`, `101.74 ms`, `333.72 GB`).
  * **`C=4 K=2 (B=12)`**: **`80.04 agg / 81.55 steady tok/s`** (`20.01/stream`, `E_acc=2.442`, `alphas=[0.870, 0.674]`, `117.54 ms`, `335.47 GB`); `256tok K=2`: `77.90 agg / 80.80 steady tok/s` (`115.88 ms`, `335.55 GB`).
  * `C=4 K=3 (B=16)`: `73.32 agg / 74.18 steady tok/s` (`E_acc=2.717`, `138.57 ms`, `335.56 GB`).

## Step C: Production Integration (`vendor/mlx_lm`), End-to-End Verification, & Live HTTP Server (`512S1:8080`)

Wired native MTP cleanly into `vendor/mlx_lm` (`LINGSANTE_MTP=1` default-on when `mtp/model-mtp.safetensors` is present; auto-adaptive `K=4` at `C=1` and `K=2` at `2<=C<=4`; exact fallback to `fast_decode_forward` for `temp > 0` or `logits_processors`):
- `utils.py`: `attach_lingsante_mtp` on non-lazy LingSante model loads (wrapped in `_MTPBox` so `nn.Module.parameters()` is unchanged).
- `spec_engine.py`: `LingSanteSpecBatch` (`BatchGenerator`/`GenerationBatch` with zero-copy extend/contraction & `_lingsante_mtp_state` prompt-cache preservation across `LRUPromptCache`) and `mtp_generate_step` (`generate_step`).
- `generate.py`, `sample_utils.py`, `server.py`: tagged greedy/non-greedy samplers, chunked prefill hidden capture (`PromptProcessingBatch`), continuous batching lifecycle (`GenerationBatch`), and mid-burst stop handling.

### Final Production Verification (`512S1`, all `0` bit diffs vs stock `B=1`)
- `bench/batch_dec.py --C 1,2,3,4 --tokens 128 --reps 2 --mode batchgen` (`batchgen_mtp_final.json`):
  * **`C=1`**: **`32.99 agg tok/s`** (`79.97 ms/step`, `329.73 GB`), `exact_tok=True`, `exact_lp32=True` (`{}` diffs)
  * **`C=2`**: **`56.61 agg tok/s`** (`28.31/stream`, `86.52 ms/step`, `330.85 GB`), `exact_tok=True`, `exact_lp32=True`
  * **`C=3`**: **`70.64 agg tok/s`** (`23.55/stream`, `104.22 ms/step`, `331.99 GB`), `exact_tok=True`, `exact_lp32=True`
  * **`C=4`**: **`80.56 agg tok/s`** (`20.14/stream`, `120.54 ms/step`, **`332.89 GB`** vs stock `358.97 GB`), `exact_tok=True`, `exact_lp32=True`
- `LINGSANTE_MTP=0 python bench/batch_dec.py --C 1,4 --tokens 128 --reps 2 --mode batchgen`: `C=1: 17.38 tok/s`, `C=4: 49.33 tok/s`, `exact_tok=True`, `exact_lp32=True` (zero regression).
- `bench/server_exact_test.py` (`LINGSANTE_MTP=1`, dynamic `1 -> 4 -> 2 -> 3 -> 1`, 5 requests): `SUMMARY: ALL PASS` (`exact_tok=True`, `exact_lp=True`).
- Edge-case verification (`EDGE_CASES_PASS`): `max_tokens=1/2`, `copy.deepcopy` Turn-2 `prompt_cache` resume (`M=1` & `M>1`), `remove(return_prompt_caches=True)` mid-flight, mixed `temp=0.7` + greedy exit/re-entry, `stop_sequences`, and `generate_step` early break + continuation.
- `bench/ctx_exact_test.py --ctx 2048 --tokens 32 --mtp`: `SUMMARY: ALL PASS` (`fastdecode C=1, 4` at `15.55 / 44.98 tok/s` AND `mtp generate_step C=1` AND `mtp BatchGenerator C=1, 4` all `exact_tok=True exact_lp=True`).
- `tools/run_eval.py gen`: all 6 medical/chemistry/biology prompts fluent at `32.1–38.8 tok/s`; `tools/run_eval.py ppl`: `chunk 64 mean NLL 0.04473 (ppl 1.0458)`, `chunk 512 mean NLL 0.04646 (ppl 1.0476)`, `ratio 0.99828`.
- Live HTTP `mlx_lm.server` (`512S1:8080`, `LINGSANTE_MTP=1`, `--decode-concurrency 4 --prompt-concurrency 1`):
  * `bench/http_smoke.py --max-tokens 128`: `C=1`: `23.33–33.45 tok/s` (`72–128` tok); `C=4` concurrent: `423 tok in 6.45s` (**`65.58 agg tok/s`** end-to-end incl. HTTP/prefill/staggered finishes, up from `38.00`), **`exact_vs_C1=True`**.
  * `bench/http_smoke.py --max-tokens 256`: `C=1`: `26.58–36.70 tok/s` (`72–217` tok); `C=4` concurrent: `517 tok in 8.73s` (**`59.22 agg tok/s`** incl. natural EOS stagger `72..217` tok), **`exact_vs_C1=True`**.
  * Non-greedy `temperature=0.7` `/v1/chat/completions` HTTP sanity check: HTTP `200`, clean `Prompt Cache` reuse in `server.log`.

---

## Phase 2: M3 Ultra Roofline Analysis, EAGLE-3 Q4-Dense Drafter & Production Integration (`512S1`)

### Empirically Measured M3 Ultra DRAM Roofline & Hardware Limits (`bench/profile_verify_c1.py`)
- **Fixed bytes per verify cycle (`512S1`, 800 GB/s peak / ~650–680 GB/s sustained)**: `16.84 GB` fixed (`63` KDA layers = `12.68 GB FP8 + 0.50 GB` state, `9` MLA layers = `0.95 GB`, `71` shared MLPs = `1.42 GB`, BF16 `lm_head` = `1.29 GB`) + `0.594 GB * mean_uniq_exp`.
- **Real end-to-end `fast_verify_forward` verify latencies (`C=1`)**:
  * `L=1`: `39.09 ms`; `L=3`: `48.94 ms`; `L=5`: `57.23 ms` (`uniq=23.0`: `KDA_gemv=18.77 ms`, `KDA_pre_gdn=2.80 ms`, `MLA=2.83 ms` pipelined / `6.04 ms` sync, `rtr+sh=2.81 ms`, `MoE=24.13 ms`, `head=2.05 ms`); `L=8`: `70.32 ms` (`uniq=29.3`: `KDA_gemv=20.56 ms`, `MoE=30.47 ms`).
  * **Hard Metal `bsg=2` register-spill cliff at `L >= 9`**: `L=9` jumps to `86.31 ms` (`L=10`: `88.23–90.05 ms`, `L=12`: `99.55 ms`, `L=16`: `120.95–121.25 ms`). Same effect confirmed on `rows/gemv_fp8.py` at `B=9` (`bsg=1, tm=2` spills to `76.86 ms` vs `bsg=1, tm=1` `28.79 ms` vs default `29.59 ms` KDA `*63`, `bit_diffs=0`).
  * **Roofline gap to `80 tok/s` at `C=1` (bit-exact)**: KDA, MoE, and `lm_head` already sustain ~92–98% of physical DRAM bandwidth. Reaching `80 tok/s` at `C=1` bit-exact would require `E_acc >= 5.0` at `L=5` or `>= 6.2` at `L=8` (`p1 >= 0.94`), above the target model's native no-drift 1-layer MTP ceiling (`p1=0.862`, `E_acc_nodrift[8]=5.746`).

### Abandoned / Negative Experiments
- **`DFlash2` 5-layer block drafter**: plateaued below `< 50 tok/s` end-to-end.
- **Standalone distilled `Ling-3.0-tiny` drafter**: `p1=0.726`, `E8=4.28`, capped at `<= 60.6 tok/s` even under Q4D estimates.
- **`COOP=1` threadgroup FP8 GEMV**: `-2.3..7.8 ms` slower on `qkvfg` projections than `rows/gemv_fp8.py`.
- **`moe_dc` tile-M sweeps (`tm=4/8`)**: optimal remains `tm=2, nsg=4, hdq=1`.
- **`Beam-W2` verification trees**: `-5.14` to `-5.92 tok/s` slower than linear `DynStop` (`Probe` `e3v2/v3/v4` all negative vs `DynStop`).
- **Blind 4-gram `SuffixMatcher`**: `-1.5 tok/s` net regression due to wrong speculative drafts inflating `mean_uniq_exp`.

### Phase 2B: EAGLE-3 Disjoint Dataset, Training & Held-Out Selection (`e3v4`)
- **600-prompt disjoint rollout corpus (`tools/gen_eagle3_data.py`, `~/science/data/lingsante_eagle3_600/`)**: `med=140, chem=80, bio=80, math=100, code=120, tech=80`, verified `0` prompt and `0` 6-gram overlaps vs `bench/exact.py` and `tools/run_eval.py`; `total_gen_tok=153,552` (`mean_gen=255.9`), `GLOBAL ALIGNMENT SELF-TEST: top1=149509/153552 (97.367%), top4=153529/153552 (99.985%)`; combined with `200` prior sequences (`800` total).
- **Training & Probe Comparison (`bench/probe_mtp_eagle.py`, Q4-Dense drafter + val `DynStop` sweep)**:
  * `e3v3` (`FeedbackBridgeV2`, `29.52M` trainable, `Ep 1 [BEST*]`: `L=1.2585`, `kl1=0.2948`, `p1=0.880`, `E2/4/6/8=2.580/3.500/3.830/3.950`, `dyn=59.18`): Q4D `B=1/2/4 = 1.448/1.550/1.882 ms`, val `DynStop` `58.72 q4d tok/s`, held-out `K=4` `51.99 med` / `dyn_step(0.55)` `51.15 med tok/s`.
  * **`e3v4` (Winner, installed in `~/science/models/lingsante-mlx-mxfp4-bf16/eagle3-mtp/`)**: `FeedbackBridge` (`rank=1024`) + `FeatFuse` on layers `[1,18,35,52,70]`, `3` epochs (`grad_accum=8`, `max_lr=1.0e-4`, `depth_weights=1.5,1.2,1.0,0.9,0.8,0.7`); `Ep 3 [BEST*]`: `L=0.9223`, `kl1=0.2796`, `p1=0.880`, `E2/4/6/8=2.600/3.540/3.940/4.110 (med=3.511/4.156)`, `cos1=0.6113`, `dyn=61.14`. Q4D `B=1/2/4 = 1.401/1.460/1.822 ms`, val `DynStop` (`p1_min=0.00, th=0.55, K_cap=7`): `mean_L=4.35, E_acc=3.610, 60.98 q4d tok/s`.
  * **Held-out bit-exact benchmark (`bench/eagle3_mtp_bench.py`, all `td=0, ld=0`)**: **`e3v4` (`dyn_step, th=0.55`) achieved `53.13 med tok/s` (`52.68 weighted tok/s`, `E_acc=3.060, L=3.84, cyc=58.09 ms`, `alphas=[0.841, 0.769, 0.604, 0.62, 0.478, 0.556, 0.5]`)** (`@256`: `50.72 med / 51.78 weighted tok/s`, `phase_ms={'draft': 4.55, 'verify': 53.59, 'commit': 1.11, 'refresh': 0.50}`) vs `e3v2 @th=0.60` (`50.35 med / 52.55 w`, `@256 49.64 / 51.44`) and stock `K=2` (`46.25 med / 46.29 w`).

### Phase 2C: Production Integration & Final Verification (`512S1`, all `0` bit diffs vs `baseline.json`)
- **Production changes (`vendor/mlx_lm/models/bailing_moe_v3/{mtp.py, spec_engine.py}`, `vendor/mlx_lm/generate.py`)**:
  * Default-on EAGLE-3 Q4-Dense drafter (`FeatFuse` + `FeedbackBridge`/`FeedbackBridgeV2` + `kv_append` / `kv_append_varlen` + `C=1` `dyn_step` policy + `C>=2` `K=2` + `PromptProcessingBatch` `mtp_target_prefill_hidden`).
  * Fixed `mtp_vcache_to_standard()` in `spec_engine.py` for `C == 1` to merge single-stream caches into `BatchKVCache` via `type(per[0][li]).merge(...)` rather than returning bare `KVCache`, preventing `AttributeError: 'KVCache' object has no attribute 'extend'` when a non-greedy request joins a single active greedy stream (`_exit_spec_mode()` + `GenerationBatch.extend()`).
- **Kernel & op parity**: `kv_append` vs `prefill` `PASS (dk=0.0, dv=0.0)`, `kv_append_varlen` vs `prefill` `PASS (dk=0.0, dv=0.0)`, `step_rows(C=4)` vs `4x C=1` `PASS (max_diff=4.69e-02, argmax 4/4 match)`, `bench/rows_fp8_test.py`, `bench/rows_moe_test.py`, and `bench/verify_kernels_test.py`: all `SUMMARY: ALL PASS`.
- **`bench/batch_dec.py --C 1,2,3,4 --tokens 128 --reps 2 --mode batchgen` (`~/science/runs/batchgen_eagle3_final.json`)**:
  * **`C=1`** (`PROMPTS[0]`): **`50.47 tok/s`** (`55.39 ms/step`, `317.7 GB`), `exact_tok=True`, `exact_lp32=True` (vs Phase 1 `32.99`, stock `15.05`)
  * **`C=2`**: **`75.86 agg tok/s`** (`37.93/stream`, `67.92 ms/step`, `317.7 GB`), `exact_tok=True`, `exact_lp32=True`
  * **`C=3`**: **`77.06 agg tok/s`** (`25.69/stream`, `99.39 ms/step`, `318.0 GB`), `exact_tok=True`, `exact_lp32=True`
  * **`C=4`**: **`84.13 agg tok/s`** (`21.03/stream`, `119.15 ms/step`, `319.2 GB`), `exact_tok=True`, `exact_lp32=True`
  * Fallbacks: `LINGSANTE_MTP=0`: `C=1: 25.92 tok/s`, `C=4: 64.43 tok/s`, `exact_tok=True`, `exact_lp32=True`; `LINGSANTE_EAGLE3=0`: `C=1: 42.20 tok/s`, `exact_tok=True`, `exact_lp32=True`.
- **`generate_step` 4-prompt held-out `bench/exact.py` (`128` tok)**: **`median=51.88 / weighted=51.66 tok/s`** (`medical_stemi: 48.77`, `chemistry_kmno4: 59.58`, `biology_lac: 54.99`, `medical_renal_pharm: 45.54`, ALL exact vs `baseline.json`).
- **`bench/ctx_exact_test.py --ctx 2048 --tokens 32 --mtp`**: `SUMMARY: ALL PASS` (`fastdecode C=1/4 = 21.35/56.68 agg tok/s`, `mtp generate_step C=1 = 4.83 e2e`, `mtp BatchGenerator C=1/4 = 4.80/4.78 e2e`).
- **Post-final-edit verification (`spec_engine.py` `C == 1` fix)**:
  * `bench/server_exact_test.py` (`1 -> 4 -> 2 -> 3 -> 1`, 5 requests): `SUMMARY: ALL PASS` (`exact_tok=True`, `exact_lp=True`).
  * Edge-case suite (`EDGE_CASES_PASS`): `[1] max_tokens=1/2`, `[2] generate_step early-break + continuation`, `[3] BatchGenerator deepcopy Turn-2 M=1 (on-path vs baseline & off-path MTP vs non-MTP) & M=3`, `[4] mixed C=1 greedy + temp=0.7 _exit_spec_mode + remove(return_prompt_caches=True) + greedy re-entry + StopSequences`.
  * `tools/run_eval.py gen`: all 6 medical/chemistry/biology prompts fluent at **`47.5–60.0 tok/s`** (`313.4 GB` active mem).
  * `tools/run_eval.py ppl`: `chunk 64 mean NLL 0.04473 (ppl 1.0458)`, `chunk 512 mean NLL 0.04646 (ppl 1.0476)`, `ratio 0.99828`.
- **Live HTTP `mlx_lm.server` (`512S1:8080`, `LINGSANTE_MTP=1`, `--decode-concurrency 4 --prompt-concurrency 1`)**:
  * `bench/http_smoke.py --base-url http://127.0.0.1:8080 --max-tokens 128`: `C=1`: `31.54–45.07 tok/s` (`72–128` tok); `C=4` concurrent: `423 tok in 5.84s` (**`72.46 agg tok/s`** incl. HTTP/prefill/EOS stagger), **`exact_vs_C1=True`**.
  * `bench/http_smoke.py --base-url http://127.0.0.1:8080 --max-tokens 256`: `C=1`: `39.54–52.05 tok/s` (`72–217` tok); `C=4` concurrent: `517 tok in 7.43s` (**`69.60 agg tok/s`**), **`exact_vs_C1=True`**.
  * Non-greedy `temperature=0.7` `/v1/chat/completions` sanity check: HTTP `200`, `24/25` cached prompt tokens, clean `server.log`.

---

## Phase A: Zero-Training Bit-Exact Kernel, Drafter, Scheduling & Serving Optimizations (`512S1`)

All numbers below are strictly **bit-exact greedy (`0` token diffs AND `0` logprob SHA-256 diffs across 32 steps vs `~/science/runs/baseline.json`)** unless explicitly testing exact `temperature > 0` rejection sampling.

### Implemented Changes (5 parallel branches merged & bisected on `512S1`)
1. **FP8 GEMV & `lm_head` (`rows/gemv_fp8.py`, `rows/gemv.py`)**:
   - Fixed P0 barrier UB in `gemv_fp8.py`; replaced LUT with Zero-Mul-256 E4M3 ALU + `bfloat4` vectorized output; eliminated the `B=9..12` `bsg=2` spill cliff (`bsg=1, bm_m=4, tm=2, b_reg=12`, zero `tg_acc` spill across `B=5..12`, `bm_m=8, tm=2` at `B=13..16`); fixed `lm_head` `B=13..16`; fused `b_proj+qkvfg` (`gemv_lin_multi`) and `router+se_swiglu` (`gemv_router_se_swiglu_rows`).
   - Bisection finding: kept `LINGSANTE_PARTSS_KDA_IN=0` (`part_ss` inside KDA `qkvfg` x62 regressed `+2.1..+14.4 ms`) and `LINGSANTE_MERGED_QKVFG=0` (`+1.44..+1.80 ms` slower than `gemv_lin_multi` on cold DRAM).
2. **KDA, MLA & Prefill (`rows/kda.py`, `rows/mla.py`, `rows/glue.py`, `kernels/gated_delta.py`, `fp8.py`, `fastdecode.py`)**:
   - Zero-spill KDA GDN (~50 regs vs 136) with `8x float4` state I/O + `(h, t)`-parallel `kda_pre_seq_rows(emit_hist=False)` + varlen KDA + lazy fused `kda_gdn_commit_and_verify_*` (halves KDA state DRAM writes on rollback).
   - Batched-Steel causal MLA SDPA (`1.88–2.12x` at `S=128..4096`) + Zero-Mul-256 `bsg=1` `mla_multilinear_fp8`.
   - `P1` packed vector-gate KDA prefill (`kernels/gated_delta.py`, `1.87x` on `T=2048` single-layer -> **`+21.9%` e2e 2K prefill, `+21.5%` e2e 32K prefill**).
3. **Router & MoE (`rows/router.py`, `rows/moe.py`)**:
   - Register-only `group_router_rows` (`10.2 -> 3.5 us`, dead `tg_score[512]` removed) + 1-launch fused `group_router_dedup_rows` emitting packed LPT `(cnt DESC, exp ASC)` `u_meta`.
   - Bisection finding: `2-PSO` / `atomic_uint` work-stealing `rows/moe.py` regressed cold 68-layer DRAM by `+4.3..+12.6 ms`; restored lean `9e90dde` `rows/moe.py` with surgical wins: `_DC_TMAX_CAP=2` (`hdq=1` for `B>=2`), `threadgroup bfloat16_t dn_s[NPAIR*4]` (halves TG mem), and packed LPT `u_meta`.
4. **D1 Fused Q4-Dense MTP Drafter (`mtp.py`)**:
   - Fixed P0 `step_varlen_rows` bug; added `eh_norm_concat_rows`, `logits_argmax_prob`, `refresh_and_step0`, Q8 adapters, and dynamic 80K shortlist with 64-token sliding-window OOV fallback (`draft+refresh: 5.05 -> 4.07 ms`).
5. **Speculative Scheduling, Exact `temp > 0` Spec & Serving (`spec_engine.py`, `suffix.py`, `generate.py`, `server.py`, `models/cache.py`)**:
   - `A1`/`A2`: varlen speculative engine (`K_i=0` mixed greedy + non-greedy coexistence, `L_new=1` join piggybacking), Bellman trailing-draft pop, suffix guard, and `B<=12` zero-spill batch-K policy (`C=2 -> K=3 (B=8)`, `C=3 -> K=3 (B=12)`, `C=4 -> K=2 (B=12)`).
   - Exact `temp > 0` speculative rejection sampling (`generate_step` & `BatchGenerator`), eliminating the `2x` non-greedy slowdown.
   - `S1` serving wins: `TimeBudget(0.5)` early-break, zero-copy `ArraysCache` (preserving `lengths`/`left_padding` copy), hoisted fingerprint/SM, in-flight prefix coalescing, `lm_head` skip on non-MTP prefill, `mx.clear_cache()` guard, deferred finish sync, startup PSO warmup, and rolling 4K prefix checkpoints.

### Verified Results (`512S1`, all greedy runs `0` token diffs & `0` logprob SHA-256 diffs vs `baseline.json`)
- **Kernel unit tests**: `bench/rows_fp8_test.py`, `bench/rows_moe_test.py`, `bench/verify_kernels_test.py`: all `SUMMARY: ALL PASS`.
- **`bench/profile_verify_c1.py --e2e-only` (`fast_verify_forward` ms, vs `9e90dde`)**:
  * `C1_L1`: `39.34` (`39.11`), `C1_L3`: **`48.68`** (`49.24`), `C1_L5`: `57.54` (`57.40`), `C1_L8`: **`68.81`** (`71.05`, **`-2.24 ms`**), **`C1_L9`: `77.76`** (`87.44`, **`-9.68 ms`**), **`C1_L10`: `80.74`** (`88.73`, **`-7.99 ms`**), **`C1_L12`: `89.10`** (`99.50`, **`-10.40 ms`**), **`C1_L16`: `114.62`** (`121.30`, **`-6.68 ms`**), **`C4_L3`: `98.01`** (`106.02`, **`-8.01 ms`**).
- **`bench/batch_dec.py --C 1,2,3,4 --tokens 128 --reps 3 --mode batchgen` (`~/science/runs/batchgen_phaseA_final.json`)**:
  * **`C=1`**: **`50.86 tok/s`** (`54.97 ms/step`, `317.9 GB`, vs `50.47`) — `exact_tok=True exact_lp32=True`
  * **`C=2`**: **`76.82 agg tok/s`** (`38.41/stream`, `79.55 ms/step`, `318.9 GB`, vs `75.86`) — `exact_tok=True exact_lp32=True`
  * **`C=3`**: **`84.70 agg tok/s`** (`28.23/stream`, `106.26 ms/step`, `319.3 GB`, vs `77.06`, **`+9.9%`**) — `exact_tok=True exact_lp32=True`
  * **`C=4`**: **`91.65 agg tok/s`** (`22.91/stream`, `109.39 ms/step`, `320.4 GB`, vs `84.13`, **`+8.9%`**) — `exact_tok=True exact_lp32=True`
  * Fallbacks: `LINGSANTE_EAGLE3=0`: `C=1: 42.30 tok/s`, `exact=True`; `LINGSANTE_MTP=0`: direct `fastdecode B=1: 25.38 tok/s (39.41 ms)`, `batchgen C=1/4: 19.78/54.83 tok/s`, `exact=True`.
- **Held-out 4-prompt decode**:
  * `bench/eagle3_mtp_bench.py --dyn-th 0.55`: **`med=53.52 / w=53.49 tok/s`** (`cyc=57.21 ms`, `td=0 ld=0`, vs `53.13/52.68`); `@256`: **`med=51.28 / w=52.54 tok/s`** (`td=0 ld=0`, vs `50.72/51.78`).
  * `bench/exact.py --capture` (`generate_step`, 4 prompts): **`median=53.15 tok/s`** (`medical_stemi: 51.08`, `chemistry_kmno4: 60.26`, `biology_lac: 55.22`, `medical_renal_pharm: 49.03`, vs `51.88` median), **`GATE PASS`**.
- **Verification & Prefill**:
  * `bench/server_exact_test.py`: `SUMMARY: ALL PASS`.
  * `bench/ctx_exact_test.py --ctx 2048 --tokens 32 --mtp`: `SUMMARY: ALL PASS` (`fastdecode C=1/4: 24.53/63.09 agg tok/s`, **`+14.9% / +11.3%`** vs `21.35/56.68`).
  * `EDGE_CASES_PASS`: `max_tokens=1/2`, early-break+continuation, `deepcopy` Turn-2 `M=1`/`M=3`, exact `temp=0.7` spec `C=1` (`33.13 tok/s` incl. prefill), mixed `C=2` greedy+`temp=0.7` coexistence + `remove(return_prompt_caches=True)` (`greedy exact=True`), `StopSequences`.
  * `tools/run_eval.py gen`: `46.9–58.9 tok/s` across all 6 prompts; `tools/run_eval.py ppl`: identical `chunk 64 NLL 0.04473 (ppl 1.0458)`, `chunk 512 NLL 0.04646 (ppl 1.0476)`.
  * `tools/run_eval.py bench --long`: **`prefill 2048: 5.3s (386.9 tok/s)`** (vs `6.5s / 317.4`, **`+21.9%`**); **`prefill 32768: 96.8s (338.6 tok/s)`** (vs `117.5s / 278.8`, **`+21.5%`**).
- **Live HTTP `mlx_lm.server` (`512S1:8080`, `pid=36399`, `--decode-concurrency 4 --prompt-concurrency 1`)**:
  * `bench/http_smoke.py --max-tokens 128`: `C=1`: **`34.67–48.76 tok/s`**; `C=4` concurrent: `423 tok in 5.44s` (**`77.72 agg tok/s`**, vs `72.46`, **`+7.3%`**), **`exact_vs_C1=True`**.
  * `bench/http_smoke.py --max-tokens 256`: `C=1`: **`43.63–54.81 tok/s`**; `C=4` concurrent: `517 tok in 6.45s` (**`80.17 agg tok/s`**, vs `69.60`, **`+15.2%`**), **`exact_vs_C1=True`**.
  * Non-greedy `temperature=0.7, top_p=0.95` `/v1/chat/completions`: HTTP `200`, prompt cache hit (`1/1` delta token on repeat), clean `server.log`.

---

## Phase B: Cold-DRAM Silicon Bandwidth Strike (`512S1`, M3 Ultra `800 GB/s` Peak)

All numbers below are strictly **bit-exact greedy (`0` token diffs AND `0` logprob SHA-256 diffs across 32 steps vs `~/science/runs/baseline.json`)**.

### 1. Ground-Truth Silicon Ceiling (`bench/bw_ledger.py` on `512S1`, Metal `uint4`)
- **Contiguous-per-TG pipelined (`ntg=160` = 2 TGs/core)**: `2 GiB` read-only **`800.0 GB/s`**, `4 GiB` read-only **`789.2 GB/s`**, `1+1 GiB` read+write **`743.7 GB/s`** (strided high-`ntg=320..5120` capped at `685.9 GB/s`).
- **Distinct 16x168 MB FP8-buffer raw `uint4` ceiling**: `711.7 GB/s` pipelined (`659.3 GB/s` median); `lm_head` (`1.29 GB`) hits `717–727 GB/s` pipelined vs `730.7 GB/s` raw `uint4` (~98–99% of silicon cap on that buffer).

### 2. Implemented Kernel-Bandwidth Wins (`bw/kda` + `bw/fp8` + `bw/moe` merged cleanly, 100% additive)
- **1-Launch Mega-Fused KDA + Low-TG MLA + `flush=1` (`rows/kda.py`, `rows/mla.py`, `verify_decode.py`)**:
  - `kda_fused_verify_rows`: fuses `kda_pre_seq` + optional `lazy_commit` + `kda_gdn_verify` + `kda_o_norm_gate` into **1 single Metal launch/layer** (`189 -> 63` launches; cold 63L KDA: `C1_L1 3.03->2.05 ms`, `C1_L5 3.36->2.46 ms`, `C1_L8 3.87->2.59 ms`, `C4_L3 6.74->5.34 ms`).
  - Low-TG contiguous `_ML_FP8_SRC` (`gx = min(nt, 8)`), scalar-int `mla_*add_softmax*` (zero per-token `mx.array` param allocations), multi-token `mla_add_softmax_multi_rows` (`2<=lc<=8`), and `_VERIFY_FLUSH_EVERY = 1` (`C4_L3` `bubble`: `+6.59 -> +2.01 ms`).
- **Unrolled K-Block Prefetch + Zero-Redundant-Shift FP8 GEMVs (`rows/gemv_fp8.py`, `rows/gemv.py`)**:
  - Prefetches `UNROLL` (2/4) K-blocks of `uchar4` weights + `simd_broadcast_first` scales; `WF_MODE=1` (`B<=8`, hoisted `float4 wf[TM]`) and `WF_MODE=2` (`B=9..12`, `tm`-outer single live `float4` avoiding M3 ~32-reg spill); front-of-grid `b_proj`/`router` tails; deduplicated shared `x0` pointer; cold-16L/32L `_fp8_cfg` tuning.
  - Cold 63L `kda_qkvfg_b`: `C1_L5 624.6->644.3 GB/s`, `C1_L8 580.4->632.5 GB/s` (`-1.50 ms`), `C4_L3 491.2->520.9 GB/s` (`-1.23 ms`).
- **Vectorized Zero-Spill MXFP4 MoE + Dependent-Layer Tail-Wave Tiling (`rows/moe.py`, `layers.py`, `utils.py`)**:
  - 128-bit `uint4` activation + `uint2`/`uint32_t` weight loads + native `Dequantize<4>` (`0` TG mem, `0` barriers in `GU`), zero-barrier `B==1` path, fine-grained critical-path tail-wave tiling (`GU NSG=2, ROWS=2(B=1)/4(B>=2)`; `DC ROWS=4, NSG=8(B=1,B>=5)/2(B=2..4), TMAX=1/2`).
  - Optional opt-in `LINGSANTE_PACK_MOE=1` (`T3` contiguous MXFP4 repacking, 0 extra steady-state RAM, lossless prefill unpack) saves another `-0.13 to -0.85 ms/cycle` on `B=5..12` decode (`C=4: 94.93 tok/s`), **defaulted to `"0"`** because live `http_smoke` proved per-request prefill unpacking (`+1.13s`) hurts short-prompt HTTP throughput. `LINGSANTE_T7_SE_MAX_B` defaulted to `"0"` (separate `se_down` beat fused `T7` on 72L `C1_L1/L3` by `-0.4..-0.6 ms`).

### 3. Verified Before (`c7f0351`) -> After (`512S1`, default `LINGSANTE_PACK_MOE=0`, all `0` bit diffs)
- **Full-Model Cold `fast_verify_forward` (`profile_verify_c1 --e2e-only` / `bw_ledger`, vs `c7f0351` & `9e90dde`)**:
  * `C1_L1` (`21.59 GB`): `39.34 -> 37.65 ms` (**`573.5 GB/s`**, `-1.69 ms`)
  * `C1_L3`: `48.68 -> 47.13 ms` (`-1.55 ms`)
  * `C1_L5` (`30.18 GB`): `57.54 -> 54.95 ms` (**`549.2 GB/s`**, `-2.59 ms`; `T3`: `54.66 ms` / `550.5 GB/s`)
  * `C1_L8` (`33.78 GB`): `68.81 -> 66.43 ms` (**`508.5 GB/s`**, `-2.38 ms`; `T3`: `65.73 ms` / `513.9 GB/s`)
  * `C1_L9`: `77.76 -> 73.47 ms` (`-4.29 ms`; `-13.97 ms` vs `9e90dde` `87.44`)
  * `C1_L12`: `89.10 -> 86.36 ms` (`-2.74 ms`; `-13.14 ms` vs `9e90dde` `99.50`)
  * `C1_L16`: `114.62 -> 109.69 ms` (`-4.93 ms`; `-11.61 ms` vs `9e90dde` `121.30`)
  * **`C4_L3` (`45.76 GB`)**: `98.01 -> 92.82 ms` (**`492.7 GB/s`**, **`-5.19 ms`**, `-13.20 ms` vs `9e90dde` `106.02`; `T3`: `91.60 ms` / `499.6 GB/s`)
- **Production `BatchGenerator` (`bench/batch_dec.py --C 1,2,3,4 --tokens 128 --reps 3 --mode batchgen`, `batchgen_bw_final.json`)**:
  * **`C=1`**: **`52.82 tok/s`** (`52.92 ms/step`, `317.9 GB`; vs `c7f0351` `50.86`, `9e90dde` `50.47`, stock `15.05` — **`3.51x`**) — `exact_tok=True exact_lp32=True`
  * **`C=2`**: **`78.31 agg tok/s`** (`39.15/stream`, `78.04 ms/step`; `T3`: `78.82`) — `exact_tok=True exact_lp32=True`
  * **`C=3`**: **`86.81 agg tok/s`** (`28.94/stream`, `103.67 ms/step`; vs `9e90dde` `77.06`, **`+12.7%`**; `T3`: `87.48`) — `exact_tok=True exact_lp32=True`
  * **`C=4`**: **`94.09 agg tok/s`** (`23.52/stream`, `106.55 ms/step`; vs `9e90dde` `84.13`, **`+11.8%`**; `T3`: `94.93`) — `exact_tok=True exact_lp32=True`
- **Held-out 4-prompt decode**:
  * `bench/exact.py --capture` (`generate_step`, 128 tok): **`55.62 median tok/s`** (`medical_stemi: 52.30`, `chemistry_kmno4: 61.95`, `biology_lac: 58.94`, `medical_renal_pharm: 50.64`; vs `9e90dde` `51.88`, **`+7.2%`**), **`GATE PASS`**.
  * `bench/eagle3_mtp_bench.py --dyn-th 0.55`: **`55.46 med / 55.42 weighted tok/s`** (`cyc=55.22 ms`, `td=0 ld=0`; `@256`: `53.16 / 54.39`).
- **Verification suites**: `rows_fp8_test`, `rows_moe_test`, `verify_kernels_test`, `server_exact_test`, `ctx_exact_test` (`fastdecode C=1/4 = 25.13 / 64.55`), `EDGE_CASES_PASS`, `run_eval ppl` (`0.04473 / 0.04646`), `run_eval gen` (`49.0–60.8 tok/s`): **ALL PASS**.
- **Live HTTP `mlx_lm.server` (`512S1:8080`, `pid=50880`, `--decode-concurrency 4 --prompt-concurrency 1`)**:
  * `http_smoke --max-tokens 128`: `C=1`: **`35.38–49.82 tok/s`**; `C=4` (cold PSO): `68.34 agg tok/s`, `exact_vs_C1=True`.
  * `http_smoke --max-tokens 256`: `C=1`: **`45.06–56.69 tok/s`** (vs `9e90dde` `39.54–52.05`); `C=4`: **`517 tok in 6.52s = 79.32 agg tok/s`**, `exact_vs_C1=True`.
  * Non-greedy `temperature=0.7, top_p=0.95`: HTTP `200`, `~45 tok/s` (128 tok in `2.8s`), prompt cache hit (`1/1` delta token), clean `server.log`.
