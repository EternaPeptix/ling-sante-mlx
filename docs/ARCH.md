# LingSante-3.1-flash (BailingMoeV3) architecture, as read from the release files

Sources (READ only, never executed). On 512S2 under `~/science/models/LingSante-3.1-flash-MXFP4/`:
`cfg` = `config.json`, `mod` = `modeling_bailing_moe_v3.py`, `conf` = `configuration_bailing_moe_v3.py`,
`conv` = `conversion_complete.json`, `README` = `README.md`, `hdr(shard)` = the safetensors header of a shard
(dtype/shape only, read with `tools/st_header.py`). Everything below is cited as `file:line`.
`fla` marks math that lives in the `fla` package (imported at mod:60-70, not installed anywhere we can read);
those formulas are therefore from knowledge of fla, cross-checked against the vendored `kimi_linear` /
`bailing_moe_v3` MLX code, and are listed again under "not verified at source" at the end.

## 1. Layer layout

* 72 decoder layers + 1 MTP layer (index 72): cfg:113, cfg:116 (`num_nextn_predict_layers` 1). hidden 4096 (cfg:92),
  64 heads (cfg:110), vocab 157184 (cfg:386), untied lm_head (cfg:372), `model_type` is `bailing_hybrid` (cfg:387) and the
  architecture string is `BailingMoeV3ForCausalLM` (cfg:2). Final norm `model.norm` then `lm_head`.
* Attention type of layer `i` (mod:1004-1009): softmax attention (gated MLA) iff `(i+1) % layer_group_size == 0`
  or `i >= 72 // 8 * 8 = 72`, with `layer_group_size = 8` (cfg:98); otherwise KDA (linear attention).
  So MLA = layers 7, 15, 23, 31, 39, 47, 55, 63, 71 (9 layers, plus the MTP layer 72), KDA = the other 63
  (the 7:1 pattern). Layers 0..6 are KDA.
* MLP type (mod:1016-1020): layers `< first_k_dense_replace = 4` (cfg:87) are dense SwiGLU MLPs with intermediate 10240
  (cfg:94, names `mlp.{gate,up,down}_proj`); layers 4..71 are MoE (`mlp.gate`, `mlp.experts.N.*`,
  `mlp.shared_experts.*`). The MTP layer is also MoE (`conv`: `extra_expert_layers [72]`). Dense layers 0-3 are all KDA.
* Decoder layer (mod:1058-1108): `h = x + attn(input_layernorm(x)); out = h + mlp(post_attention_layernorm(h))`.
  Norm = RMSNorm, variance in fp32, `weight * (x_normed.to(input_dtype))`, eps 1e-6 (mod:207-221, cfg:286).
  `position_ids` start at 0; one RoPE table is built from `qk_rope_head_dim` (mod:249-282).

## 2. KDA (Kimi Delta Attention) layer (mod:722-924)

Dims: `H = 64` heads, `D = head_dim = 128`, projection size `H*D = 8192` for q, k, v (mod:742-747; num_k_heads == num_heads,
`num_kv_heads_for_linear_attn` 0 is unused). Config switches that matter: `no_kda_lora = true` (cfg:108),
`use_kda_lora = false` (cfg:379; never read by modeling), `kda_safe_gate = true` (cfg:96), `kda_lower_bound = -5.0` (cfg:95),
`short_conv_kernel_size = 4` (cfg:371), `linear_silu = true` (cfg:99, **never read** by modeling: the silu after the conv
is hard-wired, mod:749-763), `group_norm_size = 1` (cfg:89; `BailingMoeV3GroupRMSNorm` mod:224 is never instantiated),
`use_qk_norm` (cfg:382) is also unused by KDA and MLA.

Forward for input `x` (already input-normed), per token `t`:

1. `q,k,v = {q,k,v}_proj(x)` (4096 -> 8192, no bias), each then a *depthwise causal conv1d* over time
   (weight `[8192,1,4]`, **no bias**, left context of 3 tokens, zero-initial) followed by `silu` (mod:832-849).
2. Raw gate `f = f_proj(x)` (4096 -> 8192) because `no_kda_lora` (mod:851-852; the lora pair `f_a_proj/f_b_proj` does not exist
   in this checkpoint). `beta = sigmoid(b_proj(x))` in fp32, shape `[H]` (one scalar per head, mod:856).
3. Reshape to `[H, D]`. `q`, `k` are L2-normalised over `D` inside the kernel (`use_qk_l2norm_in_kernel=True`, mod:873) and
   `q` is multiplied by `D**-0.5` (fla default `scale`) (fla).
4. Gate (`use_gate_in_kernel=True`, mod:874-876): with `safe_gate` and `lower_bound`,
   `g_log[h,d] = lower_bound * sigmoid( exp(A_log[h]) * (f[h,d] + dt_bias[h*D+d]) )` (fla `kda_gate`, lower-bound branch), i.e.
   a per-head, per-channel log-decay in `(-5, 0)`. `A_log` is `[64]` fp32, `dt_bias` is `[8192]` fp32 (mod:765, 773).
   (Without lower_bound fla uses `-exp(A_log) * softplus(f + dt_bias)`; not used here.)
5. Delta-rule recurrence, state `S[h] in R^{D_v x D_k}`, zero-init (cache holds S and the 3 last conv inputs for q, k, v):
   `S <- S * exp(g_log_t)[None,:]` (decay along the key dim); `S <- S + beta_t * (v_t - S k_t) k_t^T`; `o_t = S q_t`.
   Prefill uses `chunk_kda` for `q_len > 64`, otherwise `fused_recurrent_kda` (mod:805); mathematically the same.
6. Output: `o = o_norm(o, gate)` with `gate = g_proj(x)` (4096 -> 8192, mod:912-917) and
   `o_norm = FusedRMSNormGated(128, eps=rms_norm_eps, activation='sigmoid')` (mod:783): per head,
   `o = rmsnorm_128(o) * weight[128] * sigmoid(gate)` (fla). Then `o_proj` 8192 -> 4096 (mod:784, 919-920).

## 3. Gated MLA layer (mod:585-719)

* `q_lora_rank = 1536` (cfg:122): `q = q_b_proj(q_a_layernorm(q_a_proj(x)))`, 1536 -> `64*192`; per head
  `[nope 128 | rope 64]` (mod:659-664). `kv_lora_rank = 512`: `compressed = kv_a_proj_with_mqa(x)` (4096 -> 576) split
  `[512 | 64]`; `kv = kv_b_proj(kv_a_layernorm(compressed[:512]))`, 512 -> `64*(128+128)`, split per head into
  `k_nope 128` and `v 128` (mod:666-670). `k_rope` (64) is shared by all heads (mod:672, 680). No GQA (`num_key_value_heads` 64).
* RoPE on the 64 rope dims only, **interleaved** pairs (`rope_interleave = true`, cfg:287; mod:541-576: the pairs are first
  de-interleaved to `[even | odd]` then `rotate_half`), theta `6_000_000` (cfg:289), `rope_scaling` null (cfg:288 region),
  `partial_rotary_factor` is forced to 1.0 for the 64-dim table (mod:262). Because q and k use the same permutation,
  the score equals interleaved ("traditional") RoPE on the original layout.
* Scores `softmax((q.k) * 192**-0.5)` in fp32 with a causal mask (mod:636, 515-538), values `v` (128-dim per head).
  No yarn mscale (rope_scaling is null).
* Head-wise output gate (`gated_attention_proj_granularity_type = "head_wise"`, cfg:88): `g_proj` 4096 -> **64**,
  `attn_out[h] *= sigmoid(g_proj(x)[h])` (fp32 sigmoid) (mod:623-628, 709-715), then `dense` 8192 -> 4096 (mod:630, 718).
  `use_qkv_bias = false`.
* Cache for MLA is the usual K/V (the reference does not absorb); the MLX port caches the 512-d latent + 64-d rope key.

## 4. MoE block (mod:345-482)

* Router `mlp.gate.weight` `[512, 4096]` (bf16 on disk), `expert_bias` `[512]` fp32 buffer. Computed in fp32 (mod:394):
  `scores = sigmoid(x @ W^T)`; selection score `s' = scores + expert_bias` (mod:396-399).
* Group-limited top-k (mod:368-389): 512 experts in `n_group = 8` groups of 64 (cfg:107); group score = sum of the top-2 `s'`
  in the group; keep `topk_group = 4` groups (cfg:373); mask others to `-inf`; take top-`8` experts (cfg:112) of `s'`.
* Weights = the **unbiased** `scores` gathered at the chosen experts, normalised by `sum + 1e-20` (`norm_topk_prob`, cfg:109),
  times `routed_scaling_factor = 2.5` (cfg:291) (mod:401-404).
* Experts: SwiGLU `down(silu(gate(x)) * up(x))`, 4096 -> 1280 -> 4096 (cfg:102). Shared expert: same SwiGLU with intermediate
  `1280 * 1` (cfg:104, `num_shared_experts` 1), ungated, added to the routed sum (mod:448-449).
* `expert_swiglu_limit_list` / `share_expert_swiglu_limit_list` (cfg:13-86, cfg:297-370) are 0 for layers 0..63 and **10**
  for layers 64..71, but the released modeling file never reads them (mod has no `limit`), so the reference does not clamp.
  See "not verified" below.

## 5. MTP (mod:927-996, 1415-1460)

One extra layer (`model.layers.72.*`): `eh_proj(cat[enorm(embed(next_token)), hnorm(hidden)])` (8192 -> 4096), then
`input_layernorm`, MLA attention, `post_attention_layernorm`, MoE, `final_layernorm`; logits via the shared `lm_head`.
`mtp_loss_scaling_factor` 0 (cfg:105). **Not used** by this port (no speculative decoding); the converter skips layer 72.

## 6. On-disk weight formats (hdr, README, conv)

Layout: 153 files `model-WWW-NNNNN.safetensors` (WWW = conversion worker 0..7, written by `fixed-backend-quant_math`,
conv) + `model.safetensors.index.json` + `worker-WWW.json` (per-worker weight maps and error statistics). A layer's
tensors are scattered over several shards.

| tensor | on-disk dtype | shape (example) | notes |
| --- | --- | --- | --- |
| routed expert `experts.E.{gate,up}_proj.weight` | **I8** | `[1280, 2048]` | `K=4096` fp4 values, 2 per byte (`hdr(model-000-00006)`) |
| routed expert `experts.E.down_proj.weight` | I8 | `[4096, 640]` | `K=1280` |
| expert scale `...weight_scale_inv` | **F8_E8M0** | `[1280, 128]` / `[4096, 40]` | one scale per 32 values along K (`K/32`) |
| FP8 linears (KDA q/k/v/f/g/o_proj, MLA q_a/q_b/kv_a/kv_b/dense, dense MLP, shared experts) | **F8_E4M3** | e.g. `q_a_proj.weight [1536,4096]`, `kv_b_proj.weight [16384,512]` | `quant_method fp8, fmt e4m3, weight_block_size [128,128]` (cfg:127-132) |
| their scale `...weight_scale_inv` | **F32** | `[ceil(N/128), ceil(K/128)]`, e.g. `[12,32]`, `[128,4]`, `[5,32]` for `kv_a [576,4096]` | dequant = `w_e4m3 * scale_inv` per 128x128 block (partial edge blocks) |
| bf16 kept (`modules_to_not_convert`, cfg:134-285): embeddings, `lm_head`, router `mlp.gate.weight`, norms, KDA `b_proj`, conv1d weights (`[8192,1,4]`), `o_norm.weight [128]`, MLA `g_proj [64,4096]`, MTP `eh_proj` | BF16 | | no scale tensors |
| fp32 kept: KDA `A_log [64]`, `dt_bias [8192]`, router `expert_bias [512]` | F32 | | |

`modules_to_not_convert` is the list of modules that stay in bf16 (every `...attention.b_proj`, every `...mlp.gate`, the
MLA `attention.g_proj` of layers 7, 15, ..., `model.word_embeddings`, ...); anything with an FP8 weight also has a
`weight_scale_inv`. README: MXFP4 = E2M1 values, group size 32, shared E8M0 scale, two FP4 per byte, RTN, no calibration,
**routed experts only**; the quantisation error of experts vs the FP8 reconstruction is relative-L2 0.1176
(`conv.routed_error_vs_fp8_reconstruction`). `conv` also records `inference_validated: false`, `accuracy_validated: false`:
the publisher did not validate the release.

MLX compatibility of the expert format: MLX `mode="mxfp4"` is E2M1 (sign bit + 3 bits), `group_size 32`, one `uint8` E8M0
scale per group, 8 values per `uint32` with element 0 in the least-significant nibble. The I8 bytes `[N, K/2]` therefore
reinterpret as `uint32 [N, K/8]` with no change if the source packs the first element of a pair in the low nibble
(checked in `tools/verify_convert.py`), and the F8_E8M0 bytes are used as the `uint8` scales unchanged.

## 7. Differences between this checkpoint and what the vendored MLX code expects

See `docs/PORT_NOTES.md` (written with the implementation).
