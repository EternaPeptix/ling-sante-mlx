# Results (status at hand-off)

Verified on a toy checkpoint (tools/make_toy.py, same on-disk format, 8 layers, partial FP8 blocks):
* converter output vs source: 48 sampled experts bit-identical (mx.dequantize mxfp4 vs independent numpy decode), 70 FP8 linears bit-identical to bf16(fp8*scale), 76 copied tensors identical.
* per-layer MLX(fp32 compute, converted weights) vs independent fp32 torch reference with bf16-rounded weights: relL2 4e-7..7e-7 for KDA+dense, KDA+MoE, MLA+MoE, incl. prefill+decode cache path.
* whole model logits fp32: relL2 3.6e-6 (one-shot and cached).
* bf16 compute: relL2 ~7e-3..1e-2 per layer (bf16 noise; router top-k flips on rare near-ties).

NOT yet done (needs the real shards; download at ~10 MB/s, ~8 h): real-layer errors, 6 generations, perplexity, speed/memory.
Run order once ~/science/dl.log has DONE (all via LANE=port scripts/r):
 python tools/convert.py --src ~/science/models/LingSante-3.1-flash-MXFP4 --dst ~/science/models/lingsante-mlx-mxfp4-bf16
 python tools/verify_convert.py <src> <dst> 24
 python tools/test_layers.py <src> <dst> 0,4,7,10 48 [--fp32 --refbf16]
 python tools/run_eval.py gen|ppl|bench <dst> [--long]

## Real-weight layer tests (layers 0,1,4,7; 48 real tokens; converted bf16 dir vs independent fp32 reference)
fp32 compute, bf16-rounded reference weights, full-layer relL2: L0 3.8e-6, L1 1.2e-5, L4 8.3e-5, L7 2.3e-6; MoE block 1.4e-6/1.5e-6; router top-k agreement 1.000; cache path identical.
bf16 compute vs true fp32: L0 4.7e-3, L1 7.5e-3, L4 1.2e-1 (2/48 tokens flip a router expert), L7 2.8e-2.
Experts: 48 sampled bit-identical. Nibble order confirmed only self-consistently; generation sanity pending.

## Full 72-layer conversion, verification, and stock baseline on 512S1 (M3 Ultra, 512 GB)
* Conversion (`tools/convert.py --cpu` on 512S2, transferred via 80 Gbps Thunderbolt 5 to `512S1:~/science/models/lingsante-mlx-mxfp4-bf16`): `301 GiB` (`73` safetensors files, `1798` tensors).
* Verification (`tools/verify_convert.py ... 24 --cpu`): `checked {'expert': 1632, 'fp8': 639, 'copy': 751} mismatches 0`.
* `run_eval.py gen` (stock baseline): all 6 greedy medical, chemistry, and biology prompts fluent and domain-accurate (`14.3–14.4 tok/s`, loaded in `50s` cold / `30s` warm, `323.6 GB` active memory).
* `run_eval.py ppl` (`passage.txt`, `1239` tokens): chunk 64 mean NLL `0.04473` (`ppl 1.0458`), chunk 512 mean NLL `0.04646` (`ppl 1.0476`), `ppl ratio chunk64/chunk512 = 0.99828`.
* `run_eval.py bench --long` (stock):
  * Decode (`256` greedy tokens, prompt `81` tok): `14.36 tok/s`, prompt `42.2 tok/s`, peak `324.3 GB`.
  * Prefill `2048` tokens (`step=2048`): `6.5s = 317.4 tok/s`, peak `331.6 GB`.
  * Prefill `32768` tokens (`step=2048`): `117.5s = 278.8 tok/s`, peak `340.8 GB`.

## Bit-exact `C <= 4` decode optimization & serving (`512S1:8080`)
See [docs/OPT_LOG.md](OPT_LOG.md) for the full optimization progression.
* **Kernels (`vendor/mlx_lm/models/bailing_moe_v3/rows/`, `fastdecode.py`, `LINGSANTE_CMDBUF` in `vendor/mlx_lm/__init__.py`)**:
  * Multi-row bf16 GEMV + fused `swiglu`/`residual` (`rows/gemv.py`), warp-level `512`-expert group router (`rows/router.py`), two-pass deduplicating MXFP4 MoE (`rows/moe.py`, `rows/gather.py`), fused KDA pre-projection + packed vector-gate GDN (`Dk=Dv=128`) + fused `o_norm * sigmoid(g)` (`rows/kda.py`), fused MLA prep/RoPE + `mla_multilinear_rows` + unpadded bit-exact MLA SDPA (`rows/mla.py`), and `MLX_MAX_OPS_PER_BUFFER=64`.
  * Graph nodes reduced from `9,432` to `1,260` (7.5x); async CPU time reduced from `39.37 ms/step` to `14.53 ms/step` at `C=1`.
* **Batch exactness & throughput (`bench/batch_dec.py --C 1,2,3,4 --tokens 128 --reps 3 --mode both`, vs stock `B=1` `~/science/runs/baseline.json`)**:
  * Stock `BatchGenerator`: `C=1`: `15.05 tok/s` (`66.45 ms`, exact); `C=2..4`: `27.77 / 37.89 / 46.03 tok/s`, `359.0 GB` peak, **not bit-exact** (`exact_tok=False, exact_lp32=False`, logprob diff @ step 0).
  * Fast decode (`model`): `C=1`: **`17.12 tok/s`** (`58.42 ms`), `C=2`: **`30.23 tok/s`**, `C=3`: **`39.84 tok/s`**, `C=4`: **`48.79 tok/s`** (`81.98 ms`), `326.5 GB` peak, **all `exact_tok=True exact_lp32=True`**.
  * Fast decode (`batchgen`, `prefill_batch_size=1`): `C=1`: **`17.44 tok/s`** (`57.34 ms`), `C=2`: **`30.74 tok/s`**, `C=3`: **`40.42 tok/s`**, `C=4`: **`49.30 tok/s`** (`81.14 ms`), `326.3 GB` peak (`-32.7 GB` vs stock), **all `exact_tok=True exact_lp32=True`**.
  * `profile_fast.py` (`C=1` / `C=4`): `17.37 tok/s` (`57.57 ms/step`) / `49.12 tok/s` (`81.44 ms/step`).
* **Verification suite**:
  * Dynamic continuous batching (`bench/server_exact_test.py`, `1 -> 4 -> 2 -> 3 -> 1`): `ALL PASS` (`exact_tok=True`, `exact_lp=True`).
  * Long context (`bench/ctx_exact_test.py --ctx 2048 --tokens 32`): `ALL PASS` at `C=1` and `C=4` (`exact_tok=True`, `exact_lp=True`).
  * `run_eval.py gen`: `17.4 tok/s` across all 6 prompts; `run_eval.py ppl`: identical NLL `0.04473` / `0.04646` (`ppl 1.0458` / `1.0476`).
* **Live server (`512S1:8080`, `mlx_lm.server --decode-concurrency 4 --prompt-concurrency 1`)**:
  * HTTP smoke test (`bench/http_smoke.py`, `max_tokens=128`, `temperature=0.0`): `C=1` end-to-end `14.28–14.90 tok/s` (`72–128` tok); `C=4` concurrent `423 tok in 11.13s` (`38.00 agg tok/s`), `exact_vs_C1=True`.

## Bit-exact >= 80 tok/s native MTP Layer-72 speculative decoding & production serving
See [docs/OPT_LOG.md](OPT_LOG.md) for the full multi-token kernel & speculative tuning log.
* **Drafter choice (`512S1`, all `exact=True`, `0` diffs)**:
  * Converted built-in **MTP Layer 72** (`tools/convert_mtp.py` -> `model-mtp.safetensors`, `4.2 GiB` / `4.520 GB`, draft-only 4-bit `q4_head`, `vendor/mlx_lm/models/bailing_moe_v3/mtp.py` & `spec_engine.py`) paired with multi-token target verification (`C*L <= 24`, `rows/*`, `verify_decode.py`, `LINGSANTE_VERIFY_FLUSH=4`).
  * Benchmarked vs external `Ling-3.0-tiny-4bit`: native MTP clearly dominates (`spec_async_final.json`: `C=1 K=4 = 34.06 tok/s` / `256tok 35.17 tok/s`, `C=4 K=2 = 80.04 agg / 81.55 steady tok/s`, `256tok steady = 80.80 tok/s` vs `Ling-3.0-tiny-4bit` `C=1 = 22.75`, hybrid `25.99`, `C=4 = 43.22 agg tok/s`).
* **Production `BatchGenerator` (`bench/batch_dec.py --C 1,2,3,4 --tokens 128 --reps 2 --mode batchgen`, `batchgen_mtp_final.json`, vs `baseline.json`)**:
  * `C=1`: **`32.99 tok/s`** (`79.97 ms/step`, `329.7 GB`), `exact_tok=True`, `exact_lp32=True`
  * `C=2`: **`56.61 tok/s`** (`28.31/stream`, `86.52 ms/step`, `330.9 GB`), `exact_tok=True`, `exact_lp32=True`
  * `C=3`: **`70.64 tok/s`** (`23.55/stream`, `104.22 ms/step`, `332.0 GB`), `exact_tok=True`, `exact_lp32=True`
  * **`C=4`**: **`80.56 agg tok/s`** (`20.14/stream`, `120.54 ms/step`, **`332.9 GB`**), `exact_tok=True`, `exact_lp32=True`
  * `LINGSANTE_MTP=0`: `C=1 = 17.38 tok/s`, `C=4 = 49.33 tok/s`, `exact_tok=True`, `exact_lp32=True` (zero regression).
* **Production verification suite**:
  * `bench/verify_kernels_test.py`: `SUMMARY: ALL PASS`.
  * `bench/server_exact_test.py` (`LINGSANTE_MTP=1`, `1 -> 4 -> 2 -> 3 -> 1`): `SUMMARY: ALL PASS` (`exact_tok=True`, `exact_lp=True`).
  * `bench/ctx_exact_test.py --ctx 2048 --tokens 32 --mtp`: `SUMMARY: ALL PASS` (`exact_tok=True`, `exact_lp=True`).
  * Edge-case checks (`max_tokens=1/2`, multi-turn `deepcopy` cache resume, mid-flight `remove`, mixed `temp=0.7` + greedy, `stop_sequences`, `generate_step` early break): `EDGE_CASES_PASS`.
  * `tools/run_eval.py gen`: `32.1–38.8 tok/s` across all 6 prompts; `tools/run_eval.py ppl`: `chunk 64 mean NLL 0.04473 (ppl 1.0458)`, `chunk 512 mean NLL 0.04646 (ppl 1.0476)`.
* **Live server (`512S1:8080`, `mlx_lm.server --decode-concurrency 4 --prompt-concurrency 1`, `LINGSANTE_MTP=1`)**:
  * `bench/http_smoke.py --max-tokens 128`: `C=1`: `23.33–33.45 tok/s`; `C=4` concurrent: `423 tok in 6.45s` (**`65.58 agg tok/s`**), `exact_vs_C1=True`.
  * `bench/http_smoke.py --max-tokens 256`: `C=1`: `26.58–36.70 tok/s`; `C=4` concurrent: `517 tok in 8.73s` (**`59.22 agg tok/s`**), `exact_vs_C1=True`.

---

## Phase 2: EAGLE-3 Q4-Dense Drafter, Roofline Characterization & Production Integration (`512S1`)

All numbers below are strictly **bit-exact greedy (`0` token diffs AND `0` logprob SHA-256 diffs over 32 steps vs `baseline.json`)**.

* **Held-out bit-exact decode (`bench/eagle3_mtp_bench.py`, winning `e3v4` EAGLE-3 Q4-Dense + `FeatFuse` `[1,18,35,52,70]` + `dyn_step th=0.55`)**:
  * **4-prompt median**: **`53.13 tok/s`** (`52.68 weighted`, `E_acc=3.060`, `L=3.84`, `cyc=58.09 ms`, vs stock `15.05 tok/s` — **`3.53x`** speedup; `@256` tokens: **`50.72 med / 51.78 weighted tok/s`**).
* **Production `BatchGenerator` (`bench/batch_dec.py --C 1,2,3,4 --tokens 128 --reps 2 --mode batchgen`, `batchgen_eagle3_final.json`)**:
  * **`C=1`** (`PROMPTS[0]`): **`50.47 tok/s`** (`55.39 ms/step`, `317.7 GB`), `exact_tok=True`, `exact_lp32=True` (vs Phase 1 `32.99`, stock `15.05`)
  * **`C=2`**: **`75.86 agg tok/s`** (`37.93/stream`, `67.92 ms/step`, `317.7 GB`), `exact_tok=True`, `exact_lp32=True`
  * **`C=3`**: **`77.06 agg tok/s`** (`25.69/stream`, `99.39 ms/step`, `318.0 GB`), `exact_tok=True`, `exact_lp32=True`
  * **`C=4`**: **`84.13 agg tok/s`** (`21.03/stream`, `119.15 ms/step`, `319.2 GB`), `exact_tok=True`, `exact_lp32=True`
  * `LINGSANTE_MTP=0`: `C=1 = 25.92 tok/s`, `C=4 = 64.43 tok/s`, `exact_tok=True`, `exact_lp32=True`; `LINGSANTE_EAGLE3=0`: `C=1 = 42.20 tok/s`, `exact_tok=True`, `exact_lp32=True`.
* **Production `generate_step` (all 4 `bench/exact.py` prompts, `128` tok)**:
  * **`51.88 median / 51.66 weighted tok/s`** (`medical_stemi: 48.77`, `chemistry_kmno4: 59.58`, `biology_lac: 54.99`, `medical_renal_pharm: 45.54`), all exact.
* **M3 Ultra DRAM Roofline (`512S1`, 800 GB/s peak / ~650–680 GB/s sustained, `bench/profile_verify_c1.py`)**:
  * `16.84 GB` fixed/cycle + `0.594 GB * mean_uniq_exp`; real `fast_verify_forward` `L=5` = `57.23 ms`, `L=8` = `70.32 ms`, Metal `bsg=2` spill cliff at `L=9` = `86.31 ms` (`L=16` = `120.95–121.25 ms`). KDA/MoE/head run at ~92–98% of sustained DRAM bandwidth; `80 tok/s` at `C=1` bit-exact would require `E_acc >= 5.0` at `L=5` (`>= 6.2` at `L=8`, `p1 >= 0.94`), exceeding the target model's native no-drift 1-layer MTP ceiling (`p1=0.862`, `E_acc_nodrift[8]=5.746`).
* **Verification suite** (`server_exact_test`, `EDGE_CASES_PASS`, `run_eval`, and `http_smoke` run post-final `spec_engine.py` `C == 1` fix; kernel/context suites run prior to that 3-line `to_standard_cache` fix):
  * `bench/rows_fp8_test.py`, `bench/rows_moe_test.py`, `bench/verify_kernels_test.py`: `SUMMARY: ALL PASS`; `kv_append`/`kv_append_varlen`/`step_rows(C=4)` parity checks `PASS`.
  * `bench/server_exact_test.py` (`1 -> 4 -> 2 -> 3 -> 1`, 5 requests): `SUMMARY: ALL PASS` (`exact_tok=True`, `exact_lp=True`).
  * `bench/ctx_exact_test.py --ctx 2048 --tokens 32 --mtp`: `SUMMARY: ALL PASS` (`exact_tok=True`, `exact_lp=True`).
  * Edge-case suite (`max_tokens=1/2`, `generate_step` early break + continuation, `deepcopy` Turn-2 `M=1`/`M=3`, mixed `C=1` greedy + `temp=0.7` `_exit_spec_mode` `BatchKVCache` fix + `remove(return_prompt_caches=True)` + greedy re-entry, `StopSequences`): `EDGE_CASES_PASS`.
  * `tools/run_eval.py gen`: **`47.5–60.0 tok/s`** across all 6 prompts (`313.4 GB` active mem); `tools/run_eval.py ppl`: `chunk 64 mean NLL 0.04473 (ppl 1.0458)`, `chunk 512 mean NLL 0.04646 (ppl 1.0476)`.
* **Live server (`512S1:8080`, `mlx_lm.server --decode-concurrency 4 --prompt-concurrency 1`, `LINGSANTE_MTP=1`)**:
  * `bench/http_smoke.py --base-url http://127.0.0.1:8080 --max-tokens 128`: `C=1`: `31.54–45.07 tok/s`; `C=4` concurrent: `423 tok in 5.84s` (**`72.46 agg tok/s`**), `exact_vs_C1=True`.
  * `bench/http_smoke.py --base-url http://127.0.0.1:8080 --max-tokens 256`: `C=1`: `39.54–52.05 tok/s`; `C=4` concurrent: `517 tok in 7.43s` (**`69.60 agg tok/s`**), `exact_vs_C1=True`.
  * Non-greedy `temperature=0.7` `/v1/chat/completions` sanity check: HTTP `200`, `24/25` cached prompt tokens, clean `server.log`.
