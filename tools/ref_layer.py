"""Independent fp32 (torch CPU) reference for BailingMoeV3 decoder layers, written from docs/ARCH.md.

Reads the *source* MXFP4 checkpoint directly and decodes it with its own code (torch float8 dtypes + a
numpy E2M1 table), sharing nothing with tools/convert.py or the MLX model. Pure functions of (weights, x).
"""
import json
import math
import struct
from pathlib import Path

import numpy as np
import torch

F = torch.float32
E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)


class Src:
    def __init__(self, d):
        self.d = Path(d)
        self.idx = {}
        for f in sorted(self.d.glob("model-*.safetensors")):
            with open(f, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                h = json.loads(fh.read(n))
            for k, v in h.items():
                if k != "__metadata__":
                    a, b = v["data_offsets"]
                    self.idx[k] = (f, v["dtype"], v["shape"], 8 + n + a, 8 + n + b)
        self.cfg = json.loads((self.d / "config.json").read_text())

    def raw(self, k):
        f, dt, shape, a, b = self.idx[k]
        with open(f, "rb") as fh:
            fh.seek(a)
            buf = fh.read(b - a)
        return dt, shape, bytearray(buf)

    def t(self, k):
        """tensor decoded to fp32 (FP8 linears dequantised with their block scales)."""
        dt, shape, buf = self.raw(k)
        if dt == "BF16":
            u = torch.frombuffer(buf, dtype=torch.int16).reshape(shape)
            return (u.to(torch.int32) << 16).view(torch.float32).reshape(shape).clone()
        if dt == "F32":
            return torch.frombuffer(buf, dtype=torch.float32).reshape(shape).clone()
        if dt == "F8_E4M3":
            w = torch.frombuffer(buf, dtype=torch.uint8).reshape(shape).view(torch.float8_e4m3fn).to(F)
            s = self.t(k + "_scale_inv")
            n, kk = shape
            s = s.repeat_interleave(128, 0).repeat_interleave(128, 1)[:n, :kk]
            r = w * s
            return r.to(torch.bfloat16).to(F) if getattr(self, "round_bf16", False) else r
        raise ValueError(dt)

    def fp4(self, k):
        """routed expert weight -> fp32 [N, K] from I8 packed nibbles (low nibble = first element) and E8M0 scales."""
        dt, shape, buf = self.raw(k)
        assert dt == "I8"
        b = np.frombuffer(buf, dtype=np.uint8).reshape(shape)
        lo, hi = b & 0xF, b >> 4
        v = np.stack([E2M1[lo], E2M1[hi]], -1).reshape(shape[0], shape[1] * 2)
        sdt, sshape, sbuf = self.raw(k + "_scale_inv")
        assert sdt == "F8_E8M0"
        e = np.frombuffer(sbuf, dtype=np.uint8).reshape(sshape).astype(np.int32)
        s = np.ldexp(1.0, e - 127).astype(np.float32)
        v = v.reshape(shape[0], -1, 32) * s[:, :, None]
        return torch.from_numpy(v.reshape(shape[0], -1).copy())


def rms(x, w, eps=1e-6):
    return w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))


def silu(x):
    return x * torch.sigmoid(x)


def causal_conv_silu(x, w, state=None):
    """x [T,C]; w [C,1,K] depthwise; returns silu(conv) [T,C] (zero left context)."""
    T, C = x.shape
    K = w.shape[-1]
    xp = torch.cat([torch.zeros(K - 1, C) if state is None else state, x], 0)
    out = torch.zeros(T, C)
    for j in range(K):
        out += xp[j : j + T] * w[:, 0, j]
    return silu(out)


def kda(x, W, cfg):
    """x [T,4096] (already normed) -> [T,4096]. W: dict of fp32 weights keyed by short name."""
    T = x.shape[0]
    H, D = cfg["num_attention_heads"], cfg["head_dim"]
    q = causal_conv_silu(x @ W["q_proj"].T, W["q_conv1d"]).view(T, H, D)
    k = causal_conv_silu(x @ W["k_proj"].T, W["k_conv1d"]).view(T, H, D)
    v = causal_conv_silu(x @ W["v_proj"].T, W["v_conv1d"]).view(T, H, D)
    f = (x @ W["f_proj"].T).view(T, H, D)
    beta = torch.sigmoid(x @ W["b_proj"].T)  # [T,H]
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12) * D**-0.5
    k = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    lb = cfg["kda_lower_bound"]
    a = torch.exp(W["A_log"]).view(1, H, 1)
    glog = lb * torch.sigmoid(a * (f + W["dt_bias"].view(1, H, D)))
    S = torch.zeros(H, D, D)  # [H, Dv, Dk]
    outs = []
    for t in range(T):
        S = S * torch.exp(glog[t])[:, None, :]
        pred = torch.einsum("hvk,hk->hv", S, k[t])
        S = S + beta[t][:, None, None] * torch.einsum("hv,hk->hvk", v[t] - pred, k[t])
        outs.append(torch.einsum("hvk,hk->hv", S, q[t]))
    o = torch.stack(outs)  # [T,H,Dv]
    gate = (x @ W["g_proj"].T).view(T, H, D)
    o = rms(o, W["o_norm"], cfg["rms_norm_eps"]) * torch.sigmoid(gate)
    return o.reshape(T, H * D) @ W["o_proj"].T


def rope_interleaved(x, pos, theta):
    """x [..., T, 64] with consecutive-pair rotation."""
    d = x.shape[-1]
    inv = 1.0 / theta ** (torch.arange(0, d, 2, dtype=F) / d)
    ang = pos[:, None].to(F) * inv[None]
    c, s = torch.cos(ang), torch.sin(ang)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * c - x2 * s
    out[..., 1::2] = x1 * s + x2 * c
    return out


def mla(x, W, cfg):
    T = x.shape[0]
    H = cfg["num_attention_heads"]
    nope, rope, vd = cfg["qk_nope_head_dim"], cfg["qk_rope_head_dim"], cfg["v_head_dim"]
    q = rms(x @ W["q_a_proj"].T, W["q_a_layernorm"]) @ W["q_b_proj"].T
    q = q.view(T, H, nope + rope)
    qn, qr = q[..., :nope], q[..., nope:]
    c = x @ W["kv_a_proj_with_mqa"].T
    ckv, kr = c[:, : cfg["kv_lora_rank"]], c[:, cfg["kv_lora_rank"] :]
    kv = (rms(ckv, W["kv_a_layernorm"]) @ W["kv_b_proj"].T).view(T, H, nope + vd)
    kn, v = kv[..., :nope], kv[..., nope:]
    pos = torch.arange(T)
    qr = rope_interleaved(qr.permute(1, 0, 2), pos, cfg["rope_theta"]).permute(1, 0, 2)  # [T,H,64]
    kr = rope_interleaved(kr, pos, cfg["rope_theta"])  # [T,64]
    qf = torch.cat([qn, qr], -1)
    kf = torch.cat([kn, kr[:, None, :].expand(T, H, rope)], -1)
    sc = torch.einsum("thd,shd->hts", qf, kf) * (nope + rope) ** -0.5
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
    sc = sc.masked_fill(mask, float("-inf"))
    p = torch.softmax(sc, -1)
    o = torch.einsum("hts,shd->thd", p, v)
    o = o * torch.sigmoid(x @ W["g_proj"].T)[:, :, None]
    return o.reshape(T, H * vd) @ W["dense"].T


def swiglu(x, g, u, d):
    return (silu(x @ g.T) * (x @ u.T)) @ d.T


def moe(x, src, pre, cfg):
    """x [T,4096] (normed). Experts decoded from fp4 on demand, one at a time."""
    T = x.shape[0]
    gw = src.t(f"{pre}.gate.weight")
    bias = src.t(f"{pre}.gate.expert_bias")
    sc = torch.sigmoid(x @ gw.T)
    sel = sc + bias
    ng, tg, k = cfg["n_group"], cfg["topk_group"], cfg["num_experts_per_tok"]
    gs = sel.view(T, ng, -1).topk(2, -1)[0].sum(-1)
    gi = gs.topk(tg, -1)[1]
    gm = torch.zeros_like(gs).scatter_(1, gi, 1.0)
    em = gm[:, :, None].expand(T, ng, cfg["num_experts"] // ng).reshape(T, -1)
    top = sel.masked_fill(em == 0, float("-inf")).topk(k, -1)[1]
    w = sc.gather(1, top)
    w = w / (w.sum(-1, keepdim=True) + 1e-20) * cfg["routed_scaling_factor"]
    y = torch.zeros(T, x.shape[1])
    for e in top.unique().tolist():
        rows, slot = (top == e).nonzero(as_tuple=True)
        g = src.fp4(f"{pre}.experts.{e}.gate_proj.weight")
        u = src.fp4(f"{pre}.experts.{e}.up_proj.weight")
        d = src.fp4(f"{pre}.experts.{e}.down_proj.weight")
        o = swiglu(x[rows], g, u, d)
        y.index_add_(0, rows, o * w[rows, slot][:, None])
    sp = f"{pre}.shared_experts"
    y = y + swiglu(x, src.t(f"{sp}.gate_proj.weight"), src.t(f"{sp}.up_proj.weight"), src.t(f"{sp}.down_proj.weight"))
    return y, top


def dense_mlp(x, src, pre):
    return swiglu(x, src.t(f"{pre}.gate_proj.weight"), src.t(f"{pre}.up_proj.weight"), src.t(f"{pre}.down_proj.weight"))


KDA_KEYS = ["q_proj", "k_proj", "v_proj", "f_proj", "g_proj", "b_proj", "o_proj"]
KDA_RAW = ["A_log", "dt_bias", "o_norm"]
CONV = ["q_conv1d", "k_conv1d", "v_conv1d"]
MLA_KEYS = ["q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "g_proj", "dense"]
MLA_RAW = ["q_a_layernorm", "kv_a_layernorm"]


def layer_weights(src, li, kind):
    p = f"model.layers.{li}.attention."
    W = {}
    if kind == "kda":
        for k in KDA_KEYS:
            W[k] = src.t(p + k + ".weight")
        for k in KDA_RAW:
            W[k] = src.t(p + k + ("" if k != "o_norm" else ".weight"))
        for k in CONV:
            W[k] = src.t(p + k + ".weight")
    else:
        for k in MLA_KEYS:
            W[k] = src.t(p + k + ".weight")
        for k in MLA_RAW:
            W[k] = src.t(p + k + ".weight")
    return W


def decoder_layer(x, src, li, return_parts=False):
    cfg = src.cfg
    kind = "mla" if (li + 1) % cfg["layer_group_size"] == 0 else "kda"
    pre = f"model.layers.{li}"
    W = layer_weights(src, li, kind)
    h1 = rms(x, src.t(f"{pre}.input_layernorm.weight"))
    a = (kda if kind == "kda" else mla)(h1, W, cfg)
    h = x + a
    h2 = rms(h, src.t(f"{pre}.post_attention_layernorm.weight"))
    if li >= cfg["first_k_dense_replace"]:
        m, top = moe(h2, src, f"{pre}.mlp", cfg)
    else:
        m, top = dense_mlp(h2, src, f"{pre}.mlp"), None
    out = h + m
    if return_parts:
        return out, dict(attn=a, h=h, mlp=m, top=top, kind=kind)
    return out
