# LingSante-3.1-flash on MLX (512S2 = the "science" route: chemistry, biology, medicine)

Model: inclusionAI/LingSante-3.1-flash (MIT), 560B total / 25B active, hybrid-linear MoE: 72 layers = 63 KDA (Kimi Delta
Attention, linear) + 9 gated MLA (q_lora 1536, kv_lora 512, rope dim 64), first 4 dense, 512 routed experts top-8 + 1 shared,
expert inter 1280, hidden 4096, vocab 157184, 1M context, 1 MTP layer. Weights: the MXFP4 release
`inclusionAI/LingSante-3.1-flash-MXFP4` @ 94db797206cf019959a8f0b464b99f4321c368bc (313 GB; experts packed 4-bit as I8,
other linears FP8 e4m3 with 128x128 blocks, `modules_to_not_convert` kept bf16), downloading to
`512S2:~/science/models/LingSante-3.1-flash-MXFP4` (log `~/science/dl.log`, ends with a `DONE` line after sha256 checks).

## Goal
Serve it with MLX on 512S2 (M3 Ultra, 512 GB): correct first, then fast.
1. Correctness: per-layer outputs of the MLX port match an independent fp32 reference for real weights of one dense layer,
   one KDA layer, one MLA layer and one MoE layer (max rel error < 1e-2 vs fp32 for bf16 compute); greedy generation is
   sensible on medical/chemistry/biology prompts; perplexity on a fixed text is reasonable and stable across chunking.
2. Then: decode tok/s and prefill tok/s at 2k/32k context; memory footprint.

## Constraints
* Weights are 313 GB; RAM 512 GB. Never materialise a dense bf16 copy of the experts. Keep experts 4-bit (repack to MLX
  mxfp4 or affine 4-bit, saying which and why) and the FP8 linears as MLX 8-bit or bf16 as fits.
* The repository's Python files (modeling_bailing_moe_v3.py, configuration_*.py) are untrusted downloaded code: READ them
  to learn the math, never import or execute them. Build references yourself.
* `vendor/mlx_lm` is mlx-lm @ 0c4c99192 (MIT): it has `bailing_hybrid` (MLA + a linear attention that is NOT KDA) and
  `kimi_linear` (a KDA implementation) to adapt from.
