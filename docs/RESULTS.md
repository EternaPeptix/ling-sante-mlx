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
* `run_eval.py gen`: all 6 greedy medical, chemistry, and biology prompts fluent and domain-accurate (`14.3–14.4 tok/s`, loaded in `50s` cold / `30s` warm, `323.6 GB` active memory).
* `run_eval.py ppl` (`passage.txt`, `1239` tokens): chunk 64 mean NLL `0.04473` (`ppl 1.0458`), chunk 512 mean NLL `0.04646` (`ppl 1.0476`), `ppl ratio chunk64/chunk512 = 0.99828`.
* `run_eval.py bench --long`:
  * Decode (`256` greedy tokens, prompt `81` tok): `14.36 tok/s`, prompt `42.2 tok/s`, peak `324.3 GB`.
  * Prefill `2048` tokens (`step=2048`): `6.5s = 317.4 tok/s`, peak `331.6 GB`.
  * Prefill `32768` tokens (`step=2048`): `117.5s = 278.8 tok/s`, peak `340.8 GB`.
