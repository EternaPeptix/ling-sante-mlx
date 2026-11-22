# Port notes: what was reused, what was written, every difference

## Finding

The vendored mlx-lm (`vendor/mlx_lm`, 0c4c99192) already contains a module that implements this exact architecture:
`vendor/mlx_lm/models/bailing_moe_v3/` (KDA + gated MLA + sigmoid group-limited MoE), selected for
`model_type=bailing_hybrid` + `architectures=["BailingMoeV3ForCausalLM"]` by `MODEL_ARCHITECTURE_REMAPPING`
(`vendor/mlx_lm/utils.py:75`). The plain `bailing_hybrid` package (MLA + lightning/GLA linear attention) is for the
older Ling-2.x family and is *not* used for this checkpoint. So no new model module was needed; the work is the
converter, the independent reference, and the verification. `vendor/mlx_lm` is unmodified.

## Differences, checkpoint vs `bailing_hybrid` package (not selected)

| aspect | `bailing_hybrid` (Ling 2.x) | LingSante-3.1 (this) |
| --- | --- | --- |
| linear attention | lightning/GLA with fixed per-head slopes, `g_norm`, q/k layernorm | KDA delta rule, learned per-channel gate, short convs |
| rotary | `partial_rotary_factor` 0.5 on a normal GQA attention + MLA | MLA only, rope on 64 dims, interleaved |
| dense/sparse | same MoE router | same router |
| MLA output gate | none | head-wise sigmoid gate |

## Differences, checkpoint vs `kimi_linear` (KDA donor)

| aspect | kimi_linear | bailing_moe_v3 / this checkpoint |
| --- | --- | --- |
| gate low-rank | `f_a_proj/f_b_proj`, `g_a_proj/g_b_proj` | full `f_proj`, `g_proj` (`no_kda_lora=true`) |
| decay | `exp(-exp(A_log) * softplus(f + dt_bias))` | `exp(lower_bound * sigmoid(exp(A_log) * (f + dt_bias)))`, `lower_bound=-5` (`kda_safe_gate`) |
| MLA | NoPE-style kimi MLA, no output gate | RoPE (interleaved) on 64 dims, head-wise sigmoid output gate `g_proj`, bias-free |
| MLA checkpoint | `kv_b_proj` split at load | same: sanitize splits `kv_b_proj` into `embed_q`/`unembed_out` (so `kv_b_proj` must be bf16, not quantised) |
| conv | `[C,1,K]` weight, silu | same; sanitize moves axis to MLX `[C,K,1]` |

## What the converter does that the stock tools do not

`tools/convert.py` is a layer-at-a-time streaming converter (peak ~5 GB). Output names are already MLX names
(`mlp.switch_mlp.*` stacked experts), so `sanitize` only splits `kv_b_proj` / moves conv axes.

* experts: pure reinterpretation (see ARCH section 6). Lossless.
* FP8 block-scaled linears: dequantised to bf16 (`--linears bf16`, default) or re-quantised to MLX affine 8-bit
  group 64 (`--linears q8`). See `docs/RESULTS.md` for the numbers and the justification.
* MTP layer 72 is dropped (the MLX module has no MTP head).

(Results, measured errors and any deviations are in `docs/RESULTS.md`.)
