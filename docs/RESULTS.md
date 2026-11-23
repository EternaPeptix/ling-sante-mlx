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
