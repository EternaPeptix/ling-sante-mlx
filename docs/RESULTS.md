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
