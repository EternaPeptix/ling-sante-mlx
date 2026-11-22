"""Write a tiny random BailingMoeV3-shaped checkpoint in the *source* on-disk format (FP8 e4m3 128x128 block-scaled
linears with F32 `weight_scale_inv`, experts as I8-packed E2M1 + F8_E8M0 scales, bf16/f32 elsewhere) so that the
converter, the MLX module and the fp32 reference can be exercised end to end before the real shards arrive.

usage: make_toy.py OUT_DIR
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

torch.manual_seed(0)
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)

cfg = dict(
    architectures=["BailingMoeV3ForCausalLM"], model_type="bailing_hybrid", vocab_size=512, hidden_size=256,
    intermediate_size=384, moe_intermediate_size=128, moe_shared_expert_intermediate_size=128, num_hidden_layers=8,
    num_attention_heads=4, num_key_value_heads=4, num_experts=16, num_experts_per_tok=4, num_shared_experts=1,
    first_k_dense_replace=2, n_group=4, topk_group=2, routed_scaling_factor=2.5, layer_group_size=4, head_dim=128,
    q_lora_rank=192, kv_lora_rank=128, qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128,
    short_conv_kernel_size=4, kda_lower_bound=-5.0, kda_safe_gate=True, no_kda_lora=True, rms_norm_eps=1e-6,
    rope_theta=6000000, rope_scaling=None, max_position_embeddings=4096, rope_interleave=True,
    tie_word_embeddings=False, num_nextn_predict_layers=1, eos_token_id=1, pad_token_id=0,
    gated_attention_proj_granularity_type="head_wise",
    quantization_config=dict(activation_scheme="dynamic", fmt="e4m3", quant_method="fp8", weight_block_size=[128, 128],
                             modules_to_not_convert=[]),
)
(out / "config.json").write_text(json.dumps(cfg, indent=2))

E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
T = {}


def bf16(x):
    return x.to(torch.bfloat16)


def put_bf16(name, x):
    T[name] = ("BF16", list(x.shape), bf16(x).contiguous().view(torch.int16).numpy().tobytes())


def put_f32(name, x):
    T[name] = ("F32", list(x.shape), x.float().contiguous().numpy().tobytes())


def put_fp8(name, w):
    n, k = w.shape
    nb, kb = -(-n // 128), -(-k // 128)
    pad = torch.zeros(nb * 128, kb * 128)
    pad[:n, :k] = w
    blocks = pad.view(nb, 128, kb, 128)
    amax = blocks.abs().amax(dim=(1, 3)).clamp_min(1e-8)
    scale = amax / 448.0
    q = (blocks / scale[:, None, :, None]).to(torch.float8_e4m3fn).view(nb * 128, kb * 128)[:n, :k]
    T[name] = ("F8_E4M3", [n, k], q.contiguous().view(torch.uint8).numpy().tobytes())
    put_f32(name + "_scale_inv", scale)


def put_fp4(name, w):
    n, k = w.shape
    g = w.view(n, k // 32, 32).numpy()
    amax = np.abs(g).max(-1)
    e = np.ceil(np.log2(np.maximum(amax, 1e-30) / 6.0)).astype(np.int32)  # smallest pow2 scale covering 6.0
    s = np.ldexp(1.0, e).astype(np.float32)
    v = g / s[..., None]
    sign = (v < 0).astype(np.uint8)
    mag = np.abs(v)
    idx = np.abs(mag[..., None] - E2M1).argmin(-1).astype(np.uint8)
    code = (sign << 3 | idx).reshape(n, k)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).astype(np.uint8)  # low nibble = first element
    T[name] = ("I8", [n, k // 2], packed.tobytes())
    T[name + "_scale_inv"] = ("F8_E8M0", [n, k // 32], (e + 127).astype(np.uint8).tobytes())


def lin(n, k, std=None):
    return torch.randn(n, k) * (std or k**-0.5)


H, D, nh = cfg["hidden_size"], cfg["head_dim"], cfg["num_attention_heads"]
put_bf16("model.word_embeddings.weight", torch.randn(cfg["vocab_size"], H) * 0.5)
put_bf16("lm_head.weight", lin(cfg["vocab_size"], H))
put_bf16("model.norm.weight", 1 + 0.1 * torch.randn(H))
for li in range(cfg["num_hidden_layers"] + 1):
    p = f"model.layers.{li}"
    mla = (li + 1) % cfg["layer_group_size"] == 0 or li >= cfg["num_hidden_layers"]
    put_bf16(f"{p}.input_layernorm.weight", 1 + 0.1 * torch.randn(H))
    put_bf16(f"{p}.post_attention_layernorm.weight", 1 + 0.1 * torch.randn(H))
    a = f"{p}.attention"
    if mla:
        put_fp8(f"{a}.q_a_proj.weight", lin(cfg["q_lora_rank"], H))
        put_bf16(f"{a}.q_a_layernorm.weight", 1 + 0.1 * torch.randn(cfg["q_lora_rank"]))
        put_fp8(f"{a}.q_b_proj.weight", lin(nh * 192, cfg["q_lora_rank"]))
        put_fp8(f"{a}.kv_a_proj_with_mqa.weight", lin(cfg["kv_lora_rank"] + 64, H))
        put_bf16(f"{a}.kv_a_layernorm.weight", 1 + 0.1 * torch.randn(cfg["kv_lora_rank"]))
        put_fp8(f"{a}.kv_b_proj.weight", lin(nh * 256, cfg["kv_lora_rank"]))
        put_bf16(f"{a}.g_proj.weight", lin(nh, H))
        put_fp8(f"{a}.dense.weight", lin(H, nh * 128))
    else:
        for n in ("q_proj", "k_proj", "v_proj", "f_proj", "g_proj"):
            put_fp8(f"{a}.{n}.weight", lin(nh * D, H))
        put_bf16(f"{a}.b_proj.weight", lin(nh, H))
        for n in ("q_conv1d", "k_conv1d", "v_conv1d"):
            put_bf16(f"{a}.{n}.weight", torch.randn(nh * D, 1, 4) * 0.5)
        put_f32(f"{a}.A_log", torch.log(torch.empty(nh).uniform_(1, 16)))
        put_f32(f"{a}.dt_bias", torch.randn(nh * D) * 0.5)
        put_bf16(f"{a}.o_norm.weight", 1 + 0.1 * torch.randn(D))
        put_fp8(f"{a}.o_proj.weight", lin(H, nh * D))
    m = f"{p}.mlp"
    if li < cfg["first_k_dense_replace"]:
        put_fp8(f"{m}.gate_proj.weight", lin(cfg["intermediate_size"], H))
        put_fp8(f"{m}.up_proj.weight", lin(cfg["intermediate_size"], H))
        put_fp8(f"{m}.down_proj.weight", lin(H, cfg["intermediate_size"]))
    else:
        put_bf16(f"{m}.gate.weight", torch.randn(cfg["num_experts"], H) * 0.3)
        put_f32(f"{m}.gate.expert_bias", torch.randn(cfg["num_experts"]) * 0.05)
        mi = cfg["moe_intermediate_size"]
        for e in range(cfg["num_experts"]):
            put_fp4(f"{m}.experts.{e}.gate_proj.weight", lin(mi, H))
            put_fp4(f"{m}.experts.{e}.up_proj.weight", lin(mi, H))
            put_fp4(f"{m}.experts.{e}.down_proj.weight", lin(H, mi))
        put_fp8(f"{m}.shared_experts.gate_proj.weight", lin(mi, H))
        put_fp8(f"{m}.shared_experts.up_proj.weight", lin(mi, H))
        put_fp8(f"{m}.shared_experts.down_proj.weight", lin(H, mi))


def write(path, tensors):
    hdr, off, blobs = {"__metadata__": {"format": "pt"}}, 0, []
    for k, (dt, shape, b) in tensors.items():
        hdr[k] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + len(b)]}
        off += len(b)
        blobs.append(b)
    h = json.dumps(hdr).encode()
    h += b" " * ((8 - len(h) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)


# scatter over two shards, like the real release
names = sorted(T)
write(out / "model-000-00000.safetensors", {k: T[k] for i, k in enumerate(names) if i % 2 == 0})
write(out / "model-001-00000.safetensors", {k: T[k] for i, k in enumerate(names) if i % 2 == 1})
print("toy checkpoint written:", len(T), "tensors")
