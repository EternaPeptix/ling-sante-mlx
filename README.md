# ling-sante-mlx

Strictly **bit-exact** MLX and custom Metal inference engine for [`inclusionAI/LingSante-3.1-flash-MXFP4`](https://huggingface.co/inclusionAI/LingSante-3.1-flash-MXFP4) on Apple Silicon (verified on Apple M3 Ultra 512 GB, 800 GB/s peak DRAM).

Every optimization in the default greedy path maintains **`0` token diffs** and **`0` logprob SHA-256 diffs across 32 steps** against the stock single-stream BF16/MXFP4/FP8 MLX reference (`baseline.json`), while reaching up to **`3.51x`** single-stream (`C=1`) decode speedup and **`94+ agg tok/s`** at `C=4`.

---

## Model & Hardware Target

- **Model**: `inclusionAI/LingSante-3.1-flash-MXFP4` (`72` decoder layers + native Layer-72 MTP layer, `hidden_size=4096`, `vocab_size=157,184`):
  - **Layers `0..3`**: `4` dense SwiGLU FP8 layers (`intermediate=10240`)
  - **Layers `4..71`**: `68` MoE layers (`512` routed MXFP4 experts, `top-8`, `moe_intermediate=1280` + FP8 `shared_experts`)
  - **Attention / Sequence Mixing**: `63` KDA (vector-gate Gated DeltaNet, `Dk=Dv=128`) layers alternating `7:1` with `9` MLA attention layers (`7, 15, ..., 71`)
- **Converted Checkpoint Footprint**: `301 GiB` (`73` safetensors shards + `639` native 1-byte FP8 sidecar tensors + `4.2 GiB` native Layer-72 MTP + `eagle3-mtp/` Q4-Dense weights & `FeedbackBridge`/`FeatFuse` adapters)

---

## Verified Performance (`512S1`, M3 Ultra 512 GB, Strictly Bit-Exact)

### `BatchGenerator` Decode (`bench/batch_dec.py --C 1,2,3,4 --tokens 128 --reps 3`)

| Configuration | `C=1` | `C=2` (agg) | `C=3` (agg) | `C=4` (agg) | Peak RAM | Bit-Exact vs `B=1` |
|---|---:|---:|---:|---:|---:|:---:|
| Stock `BatchGenerator` | `15.05 tok/s` | `27.77 tok/s` | `37.89 tok/s` | `46.03 tok/s` | `359.0 GB` | `C=1` only (`C>=2` diverges) |
| Phase 1 Fast Decode (`LINGSANTE_MTP=0`) | `17.44 tok/s` | `30.74 tok/s` | `40.42 tok/s` | `49.30 tok/s` | `326.3 GB` | `0` diffs |
| Phase 1 Native MTP Layer-72 | `32.99 tok/s` | `56.61 tok/s` | `70.64 tok/s` | `80.56 tok/s` | `332.9 GB` | `0` diffs |
| **Phase B `HEAD` (`EAGLE-3 Q4-Dense + Metal BW Strike`)** | **`52.82 tok/s`** | **`78.31 tok/s`** | **`86.81 tok/s`** | **`94.09 tok/s`** | `317.9–320.4 GB` | **`0` diffs** |
| Phase B + `LINGSANTE_PACK_MOE=1` (opt-in `T3`) | — | `78.82 tok/s` | `87.48 tok/s` | **`94.93 tok/s`** | same | **`0` diffs** |

### Held-Out Decode, Prefill & Live Serving (`HEAD` `666c6ab`)

- **`generate_step` 4-prompt held-out (`bench/exact.py`, 128 tok)**: **`55.62 median tok/s`** (`medical_stemi: 52.30`, `chemistry_kmno4: 61.95`, `biology_lac: 58.94`, `medical_renal_pharm: 50.64`, `GATE PASS`).
- **Standalone EAGLE-3 benchmark (`bench/eagle3_mtp_bench.py --dyn-th 0.55`)**: **`55.46 med / 55.42 weighted tok/s`** (`cyc=55.22 ms`, `td=0 ld=0`; `@256` tokens: **`53.16 med / 54.39 weighted tok/s`**).
- **Prefill (`tools/run_eval.py bench --long`)**:
  - `2048` tokens: **`5.3s (386.9 tok/s)`** (up from `6.5s / 317.4`, `+21.9%`)
  - `32768` tokens: **`96.8s (338.6 tok/s)`** (up from `117.5s / 278.8`, `+21.5%`)
- **Perplexity (`tools/run_eval.py ppl`, `passage.txt`, 1239 tokens)**: `chunk 64 mean NLL 0.04473 (ppl 1.0458)`, `chunk 512 mean NLL 0.04646 (ppl 1.0476)` (identical to stock).
- **Live HTTP `mlx_lm.server` (`--decode-concurrency 4 --prompt-concurrency 1`, `bench/http_smoke.py --max-tokens 256`)**:
  - `C=1`: **`45.06–56.69 tok/s`** end-to-end
  - `C=4` concurrent: **`517 tok in 6.52s (79.32 agg tok/s)`**, `exact_vs_C1=True`
  - Non-greedy `temperature=0.7, top_p=0.95`: `HTTP 200`, exact distribution-preserving rejection-sampling speculative decode (`~45 tok/s`, 128 tok in `2.8s`) with prompt-cache reuse.

### Silicon Bandwidth Ledger (`bench/bw_ledger.py`, M3 Ultra `800 GB/s` Peak)

- Raw contiguous `uint4` read ceiling (`ntg=160`): **`800.0 GB/s`** (`2 GiB`), **`789.2 GB/s`** (`4 GiB`), **`743.7 GB/s`** (`1+1 GiB` R+W).
- Distinct multi-buffer ceiling (`16 x 168 MB` separate FP8 buffers): **`711.7 GB/s`** pipelined (`659.3 GB/s` median); single-op BF16 `lm_head` (`1.29 GB`): **`717–727 GB/s`**.
- Full 72-layer cold `fast_verify_forward`:
  - `C1_L1` (`21.59 GB`): **`37.65 ms (573.5 GB/s)`**
  - `C1_L5` (`30.18 GB`): **`54.95 ms (549.2 GB/s)`** (`LINGSANTE_PACK_MOE=1`: `54.66 ms / 550.5 GB/s`)
  - `C1_L8` (`33.78 GB`): **`66.43 ms (508.5 GB/s)`**
  - `C4_L3` (`45.76 GB`): **`92.82 ms (492.7 GB/s)`** (`LINGSANTE_PACK_MOE=1`: `91.60 ms / 499.6 GB/s`)

---

## Key Optimizations & Architecture

- **Native 1-Byte FP8 Sidecar + Zero-Spill Multi-Row GEMVs** (`vendor/mlx_lm/models/bailing_moe_v3/rows/gemv_fp8.py`, `rows/gemv.py`, `tools/extract_fp8_sidecar.py`): Zero-Mul-256 E4M3 ALU dequantization, `bfloat4` vectorized stores, K-unroll prefetching, `b_proj + qkvfg` and `router + se_swiglu` launch fusions, and `B=5..16` register-diet configurations eliminating the `B=9..12` `bsg=2` spill cliff.
- **Deduplicated MXFP4 MoE & Register-Only Routing** (`rows/moe.py`, `rows/router.py`): 1-launch fused `group_router_dedup_rows` emitting packed LPT `(cnt DESC, exp ASC)` metadata (`u_meta`), `TMAX=2` deduplication (`hdq=1` for `B>=2`), `bfloat16_t dn_s` threadgroup staging, plus opt-in contiguous K-major repacking (`LINGSANTE_PACK_MOE=1`).
- **Fused Vector-Gate KDA & Batched-Steel MLA** (`rows/kda.py`, `rows/mla.py`, `kernels/gated_delta.py`): Single-launch zero-spill KDA verify (`~50` regs vs `136`) with `8x float4` state I/O, lazy fused `kda_gdn_commit_and_verify_*` (halving KDA state writes on rollback), `P1` packed TG-staged KDA prefill (`1.87x` single-layer), and batched-Steel causal MLA SDPA (`1.88–2.12x` at `S=128..4096`).
- **EAGLE-3 Q4-Dense MTP Drafter** (`mtp.py`, `tools/gen_eagle3_data.py`, `tools/train_eagle3_mtp.py`): Multi-layer `FeatFuse` + `FeedbackBridge` (`e3v4`), fused `eh_norm_concat_rows` and `logits_argmax_prob` kernels, `refresh_and_step0`, Q8 adapters, and an 80K draft shortlist with a 64-token sliding-window fallback to full 157K vocab on OOV tokens.
- **Varlen Speculative Engine & Serving** (`spec_engine.py`, `server.py`, `generate.py`, `models/cache.py`): Heterogeneous varlen `K_i` scheduling (including `K_i=0` mixed greedy + non-greedy coexistence and `L_new=1` join piggybacking), exact distribution-preserving `temperature > 0` rejection sampling, `TimeBudget` early-break on queued HTTP requests, zero-copy `ArraysCache` slicing/merging, in-flight prefix coalescing, rolling 4K prefix checkpoints, and startup PSO warmup.

---

## Quick Start

### 1. Convert Weights, FP8 Sidecar & MTP

```zsh
# Convert HF inclusionAI/LingSante-3.1-flash-MXFP4 -> MLX BF16/MXFP4
python tools/convert.py --src /path/to/LingSante-3.1-flash-MXFP4 --dst /path/to/lingsante-mlx-mxfp4-bf16

# Verify bit-exact conversion
python tools/verify_convert.py /path/to/LingSante-3.1-flash-MXFP4 /path/to/lingsante-mlx-mxfp4-bf16 24

# Extract native 1-byte FP8 sidecar & Layer-72 MTP weights
python tools/extract_fp8_sidecar.py --src /path/to/LingSante-3.1-flash-MXFP4 --dst /path/to/lingsante-mlx-mxfp4-bf16
python tools/convert_mtp.py --src /path/to/LingSante-3.1-flash-MXFP4 --dst /path/to/lingsante-mlx-mxfp4-bf16
```

### 2. Serve via OpenAI-Compatible `mlx_lm.server`

```zsh
HF_HUB_OFFLINE=1 PYTHONPATH=$PWD/vendor:$PWD python -m mlx_lm server \
  --model /path/to/lingsante-mlx-mxfp4-bf16 \
  --host 0.0.0.0 --port 8080 \
  --decode-concurrency 4 --prompt-concurrency 1
```

### 3. Evaluate & Run Bit-Exact Verification

All scripts require running with `PYTHONPATH=$PWD/vendor:$PWD` (importing `mlx_lm` before `mlx.core` so `LINGSANTE_CMDBUF` settings apply):

```zsh
PYTHONPATH=$PWD/vendor:$PWD python bench/verify_kernels_test.py
PYTHONPATH=$PWD/vendor:$PWD python bench/rows_fp8_test.py
PYTHONPATH=$PWD/vendor:$PWD python bench/rows_moe_test.py
PYTHONPATH=$PWD/vendor:$PWD python bench/batch_dec.py --C 1,2,3,4 --tokens 128 --reps 3 --mode batchgen
PYTHONPATH=$PWD/vendor:$PWD python bench/server_exact_test.py
PYTHONPATH=$PWD/vendor:$PWD python bench/ctx_exact_test.py --ctx 2048 --tokens 32 --mtp
PYTHONPATH=$PWD/vendor:$PWD python bench/http_smoke.py --base-url http://127.0.0.1:8080 --max-tokens 256
PYTHONPATH=$PWD/vendor:$PWD python tools/run_eval.py gen /path/to/lingsante-mlx-mxfp4-bf16
PYTHONPATH=$PWD/vendor:$PWD python tools/run_eval.py ppl /path/to/lingsante-mlx-mxfp4-bf16
```

### Runtime Environment Toggles

| Variable | Default | Effect |
|---|---|---|
| `LINGSANTE_MTP` | `1` | Enable speculative decoding (`0` disables MTP speculation) |
| `LINGSANTE_EAGLE3` | `1` | Use `eagle3-mtp/` adapter if present (`0` falls back to native Layer-72 MTP) |
| `LINGSANTE_FAST_DECODE` | `1` | Custom Metal single-token decode kernels (`fastdecode.py`) |
| `LINGSANTE_FAST_VERIFY` | `1` | Custom Metal multi-token speculative verification kernels (`verify_decode.py`) |
| `LINGSANTE_PACK_MOE` | `0` | Runtime K-major MoE repacking (`1` gives ~+0.8 tok/s at `C=3..4`, adds prefill unpack latency) |

---

## Repository Layout

- [`vendor/mlx_lm/models/bailing_moe_v3/`](vendor/mlx_lm/models/bailing_moe_v3/) — Model, FP8 sidecar, custom Metal kernels (`rows/`), `fastdecode.py`, `verify_decode.py`, `mtp.py`, `spec_engine.py`
- [`vendor/mlx_lm/`](vendor/mlx_lm/) — Patched `generate.py`, `server.py`, `models/cache.py`, `kernels/gated_delta.py`
- [`tools/`](tools/) — HF-to-MLX conversion, FP8/MTP extraction, EAGLE-3 dataset generation/training, and `run_eval.py`
- [`bench/`](bench/) — Bit-exactness regression tests, kernel microbenchmarks, DRAM roofline ledger (`bw_ledger.py`), and HTTP smoke tests
- [`docs/RESULTS.md`](docs/RESULTS.md), [`docs/OPT_LOG.md`](docs/OPT_LOG.md), [`docs/ARCH.md`](docs/ARCH.md) — Verified metrics, optimization & bisection log, and architecture notes
