"""Native built-in MTP (layer 72) & EAGLE-3 Q4-Dense drafter for LingSante-3.1-flash-MXFP4."""
from collections import deque
import json
import os
from pathlib import Path
from typing import Any, Sequence

import mlx.core as mx
from mlx import nn

from mlx_lm.layers.attention import create_attention_mask
from mlx_lm.layers.common import swiglu
from mlx_lm.models.bailing_moe_v3.config import ModelArgs
from mlx_lm.models.bailing_moe_v3.layers import BailingMLA, BailingSparseMoE
from mlx_lm.models.cache import KVCache
from mlx_lm.utils import infer_quant_config

from .rows.gemv import gemv_rows, gemv_swiglu_rows
from .rows.glue import GLUE_HEADER
from .rows.mla import (
    _mla_append_causal_slice,
    mla_layer_rows,
    mla_multilinear_lin,
    mla_prep_kv_rope_rows,
    mla_verify_layer_rows,
    mla_verify_varlen_rows,
)
from .rows.moe import moe_dedup_rows, moe_down_combine_rows, moe_gate_up_swiglu_rows
from .rows.router import gate_rows

FEAT_LAYERS = (1, 18, 35, 52, 70)

_MTP_KERNELS: dict[str, Any] = {}

_EH_NORM_CONCAT_SRC = r"""
    constexpr int H = 4096;
    constexpr int TG = 128;
    const uint lid = thread_position_in_threadgroup.x;
    const uint side = threadgroup_position_in_grid.y;  // 0 = embed (enorm), 1 = prev_h (hnorm)
    const uint b = threadgroup_position_in_grid.z;

    const device T* src = (side == 0 ? embeds : prev_h) + size_t(b) * H;
    const device WT* nw = (side == 0 ? ew : hw);
    device T* dst = out + size_t(b) * (2 * H) + (side == 0 ? 0 : H);

    threadgroup float part[32];
    const float inv = glue_rms_inv_tg<H, TG, true>(src, RMS_EPS, part, lid);

    const uint lane = lid & 31;
    const uint sg = lid >> 5;
    for (int g = int(sg); g < 32; g += (TG / 32)) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        int idx = (g * 32 + int(lane)) * 4 + i;
        dst[idx] = glue_rms_apply<T, WT>(static_cast<float>(src[idx]), inv, nw[idx]);
      }
    }
"""

_LOGITS_S1_SRC = r"""
    constexpr int TG = 256;
    constexpr int NSG = TG / 32;  // 8
    constexpr int NCHUNKS = 128;

    const uint lid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint chunk = threadgroup_position_in_grid.x;
    const uint row = threadgroup_position_in_grid.y;

    const int V = VV;
    const int base = (V * int(chunk)) / NCHUNKS;
    const int end  = (V * int(chunk + 1)) / NCHUNKS;
    const device T* x = logits + size_t(row) * size_t(V);

    float t_max = -INFINITY;
    uint t_arg = uint(base);
    for (int i = base + int(lid); i < end; i += TG) {
      float v = static_cast<float>(x[i]);
      if (v > t_max || (v == t_max && uint(i) < t_arg)) {
        t_max = v;
        t_arg = uint(i);
      }
    }

    // Simdgroup argmax reduction (deterministic smallest-index tie break)
    for (ushort off = 16; off >= 1; off >>= 1) {
      float om = simd_shuffle_down(t_max, off);
      uint  oa = simd_shuffle_down(t_arg, off);
      if (om > t_max || (om == t_max && oa < t_arg)) {
        t_max = om;
        t_arg = oa;
      }
    }

    threadgroup float tg_m[NSG];
    threadgroup uint  tg_a[NSG];
    threadgroup float tg_s[NSG];
    if (lane == 0) {
      tg_m[sg] = t_max;
      tg_a[sg] = t_arg;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0) {
      float m = (lane < NSG) ? tg_m[lane] : -INFINITY;
      uint  a = (lane < NSG) ? tg_a[lane] : 0xffffffffu;
      for (ushort off = 4; off >= 1; off >>= 1) {
        float om = simd_shuffle_down(m, off);
        uint  oa = simd_shuffle_down(a, off);
        if (om > m || (om == m && oa < a)) {
          m = om;
          a = oa;
        }
      }
      if (lane == 0) {
        tg_m[0] = m;
        tg_a[0] = a;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float c_max = tg_m[0];
    float t_sum = 0.0f;
    for (int i = base + int(lid); i < end; i += TG) {
      t_sum += metal::precise::exp(static_cast<float>(x[i]) - c_max);
    }
    t_sum = simd_sum(t_sum);
    if (lane == 0) tg_s[sg] = t_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0) {
      float s = simd_sum((lane < NSG) ? tg_s[lane] : 0.0f);
      if (lane == 0) {
        const size_t o = size_t(row) * NCHUNKS + size_t(chunk);
        part_max[o] = c_max;
        part_arg[o] = tg_a[0];
        part_sum[o] = s;
      }
    }
"""

_LOGITS_S2_SRC = r"""
    constexpr int NCHUNKS = 128;
    constexpr int NSG = NCHUNKS / 32;  // 4

    const uint lid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint row = threadgroup_position_in_grid.y;

    const size_t o = size_t(row) * NCHUNKS + size_t(lid);
    const float my_m = part_max[o];
    const uint  my_a = part_arg[o];
    const float my_s = part_sum[o];

    float m = my_m;
    uint  a = my_a;
    for (ushort off = 16; off >= 1; off >>= 1) {
      float om = simd_shuffle_down(m, off);
      uint  oa = simd_shuffle_down(a, off);
      if (om > m || (om == m && oa < a)) {
        m = om;
        a = oa;
      }
    }

    threadgroup float tg_m[NSG];
    threadgroup uint  tg_a[NSG];
    threadgroup float tg_s[NSG];
    if (lane == 0) {
      tg_m[sg] = m;
      tg_a[sg] = a;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0) {
      float gm = (lane < NSG) ? tg_m[lane] : -INFINITY;
      uint  ga = (lane < NSG) ? tg_a[lane] : 0xffffffffu;
      for (ushort off = 2; off >= 1; off >>= 1) {
        float om = simd_shuffle_down(gm, off);
        uint  oa = simd_shuffle_down(ga, off);
        if (om > gm || (om == gm && oa < ga)) {
          gm = om;
          ga = oa;
        }
      }
      if (lane == 0) {
        tg_m[0] = gm;
        tg_a[0] = ga;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float g_max = tg_m[0];
    float sc = my_s * metal::precise::exp(my_m - g_max);
    sc = simd_sum(sc);
    if (lane == 0) tg_s[sg] = sc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0) {
      float tot = simd_sum((lane < NSG) ? tg_s[lane] : 0.0f);
      if (lane == 0) {
        out_arg[row] = tg_a[0];
        out_p1[row]  = 1.0f / tot;
      }
    }
"""


def _eh_norm_concat_kernel(eps: float):
    key = ("eh_nc", float(eps))
    k = _MTP_KERNELS.get(key)
    if k is None:
        from .rows.glue import flit
        k = _MTP_KERNELS[key] = mx.fast.metal_kernel(
            name="bv3_mtp_eh_norm_concat",
            input_names=["embeds", "prev_h", "ew", "hw"],
            output_names=["out"],
            header=GLUE_HEADER + f"\n#define RMS_EPS {flit(eps)}\n",
            source=_EH_NORM_CONCAT_SRC,
            ensure_row_contiguous=True,
        )
    return k


def eh_norm_concat_rows(
    embeds: mx.array, prev_h: mx.array, ew: mx.array, hw: mx.array, eps: float = 1e-6
) -> mx.array:
    """Fused enorm(embeds) || hnorm(prev_h) -> (B, 8192) in ONE Metal launch."""
    B = int(embeds.shape[0])
    (out,) = _eh_norm_concat_kernel(eps)(
        inputs=[embeds, prev_h, ew, hw],
        template=[("T", embeds.dtype), ("WT", ew.dtype)],
        grid=(128, 2, B),
        threadgroup=(128, 1, 1),
        output_shapes=[(B, 8192)],
        output_dtypes=[embeds.dtype],
    )
    return out


def _logits_s1_kernel():
    k = _MTP_KERNELS.get("logits_s1")
    if k is None:
        k = _MTP_KERNELS["logits_s1"] = mx.fast.metal_kernel(
            name="bv3_mtp_logits_s1",
            input_names=["logits"],
            output_names=["part_max", "part_arg", "part_sum"],
            header=GLUE_HEADER,
            source=_LOGITS_S1_SRC,
            ensure_row_contiguous=True,
        )
    return k


def _logits_s2_kernel():
    k = _MTP_KERNELS.get("logits_s2")
    if k is None:
        k = _MTP_KERNELS["logits_s2"] = mx.fast.metal_kernel(
            name="bv3_mtp_logits_s2",
            input_names=["part_max", "part_arg", "part_sum"],
            output_names=["out_arg", "out_p1"],
            header=GLUE_HEADER,
            source=_LOGITS_S2_SRC,
            ensure_row_contiguous=True,
        )
    return k


def logits_argmax_prob(logits: mx.array) -> tuple[mx.array, mx.array]:
    """Fused 2-stage Metal reduction on `(C, V)` BF16/FP32 logits -> `(argmax (C,) u32, p_top1 (C,) f32)`."""
    if logits.ndim == 1:
        logits_2d = logits[None, :]
        squeeze = True
    else:
        logits_2d = logits
        squeeze = False
    C, V = int(logits_2d.shape[0]), int(logits_2d.shape[1])
    pm, pa, ps = _logits_s1_kernel()(
        inputs=[logits_2d],
        template=[("T", logits_2d.dtype), ("VV", V)],
        grid=(128 * 256, C, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(C, 128), (C, 128), (C, 128)],
        output_dtypes=[mx.float32, mx.uint32, mx.float32],
    )
    arg, p1 = _logits_s2_kernel()(
        inputs=[pm, pa, ps],
        grid=(128, C, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(C,), (C,)],
        output_dtypes=[mx.uint32, mx.float32],
    )
    if squeeze:
        return arg[0], p1[0]
    return arg, p1


def _concat_quantized_linear(linears: list[nn.QuantizedLinear]) -> nn.QuantizedLinear:
    """Concatenate row-aligned QuantizedLinear layers sharing input dim & quant config along axis=0."""
    first = linears[0]
    gs = first.group_size
    bits = first.bits
    mode = first.mode
    w = mx.concatenate([m.weight for m in linears], axis=0)
    sc = mx.concatenate([m.scales for m in linears], axis=0)
    has_biases = hasattr(first, "biases") and first.biases is not None
    bi = mx.concatenate([m.biases for m in linears], axis=0) if has_biases else None
    out_dims = int(w.shape[0])
    in_dims = int(sc.shape[1]) * gs
    merged = nn.QuantizedLinear(in_dims, out_dims, bias=False, group_size=gs, bits=bits, mode=mode)
    merged.weight = w
    merged.scales = sc
    if has_biases:
        merged.biases = bi
    return merged


def _build_bpe_non_cjk_shortlist(model_dir: Path, target_size: int, vocab_size: int) -> mx.array | None:
    """Build a zero-leakage structural shortlist from tokenizer.json BPE order (non-CJK/Cyrillic + added)."""
    tok_path = model_dir / "tokenizer.json"
    if not tok_path.exists():
        return None
    tok = json.loads(tok_path.read_text())
    vocab = tok.get("model", {}).get("vocab", {})
    added = [int(t["id"]) for t in tok.get("added_tokens", []) if 0 <= int(t["id"]) < vocab_size]

    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    u2b = {chr(c): b for b, c in zip(bs, cs)}

    non_cjk: list[int] = []
    for s, idx in sorted(vocab.items(), key=lambda kv: kv[1]):
        if idx < 0 or idx >= vocab_size:
            continue
        try:
            d = bytes(u2b[c] for c in s).decode("utf-8", errors="replace")
        except KeyError:
            d = s
        if not any(
            (0x4E00 <= ord(c) <= 0x9FFF)
            or (0x3000 <= ord(c) <= 0x30FF)
            or (0xAC00 <= ord(c) <= 0xD7AF)
            or (0x0400 <= ord(c) <= 0x052F)
            or (0x0600 <= ord(c) <= 0x06FF)
            for c in d
        ):
            non_cjk.append(int(idx))

    lim = max(1, target_size - len(added))
    chosen = sorted(set(non_cjk[:lim]) | set(added))
    # Pad to multiple of 64 for clean Q4 alignment if needed
    if len(chosen) < target_size:
        chosen_set = set(chosen)
        for i in range(vocab_size):
            if i not in chosen_set:
                chosen.append(i)
                chosen_set.add(i)
                if len(chosen) >= target_size:
                    break
        chosen.sort()
    elif len(chosen) > target_size:
        chosen = chosen[:target_size]
    return mx.array(chosen, dtype=mx.uint32)


class _AuxBox:
    """Plain Python container keeping auxiliary fused modules out of nn.Module.parameters()."""

    __slots__ = ("d",)

    def __init__(self):
        self.d: dict[str, Any] = {}


class FeedbackBridge(nn.Module):
    """Residual feedback adapter mapping MTP hidden states onto target h_final manifold."""

    def __init__(self, hidden_size: int = 4096, rank: int = 1024, eps: float = 1e-6):
        super().__init__()
        self._is_v2 = False
        self.norm = nn.RMSNorm(hidden_size, eps=eps)
        self.up = nn.Linear(hidden_size, rank, bias=False)
        self.down = nn.Linear(rank, hidden_size, bias=False)

    def __call__(self, h: mx.array) -> mx.array:
        return h + self.down(nn.silu(self.up(self.norm(h))))


class FeedbackBridgeV2(nn.Module):
    """Dual-input SwiGLU residual bridge combining MTP h_out with lookahead h_in."""

    def __init__(self, hidden_size: int = 4096, rank: int = 1024, eps: float = 1e-6):
        super().__init__()
        self._is_v2 = True
        self.norm_out = nn.RMSNorm(hidden_size, eps=eps)
        self.norm_in = nn.RMSNorm(hidden_size, eps=eps)
        self.gate = nn.Linear(hidden_size * 2, rank, bias=False)
        self.up = nn.Linear(hidden_size * 2, rank, bias=False)
        self.down = nn.Linear(rank, hidden_size, bias=False)

    def __call__(self, h_out: mx.array, h_in: mx.array) -> mx.array:
        xo = h_out.astype(mx.float32)
        xi = h_in.astype(mx.float32)
        cat = mx.concatenate([self.norm_out(xo), self.norm_in(xi)], axis=-1).astype(h_out.dtype)
        return h_out + self.down(nn.silu(self.gate(cat)) * self.up(cat))


class FeatFuse(nn.Module):
    """Residual projection from concatenated multi-layer target features onto hidden_size."""

    def __init__(self, in_dim: int = 20480, rank: int = 256, out_dim: int = 4096, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.RMSNorm(in_dim, eps=eps)
        self.up = nn.Linear(in_dim, rank, bias=False)
        self.down = nn.Linear(rank, out_dim, bias=False)

    def __call__(self, feats: mx.array) -> mx.array:
        return self.down(nn.silu(self.up(self.norm(feats))))


def apply_bridge(bridge, h_out: mx.array, h_in: mx.array) -> mx.array:
    """Apply FeedbackBridge or dual-input FeedbackBridgeV2 uniformly."""
    if bridge is None:
        return h_out
    if getattr(bridge, "_is_v2", False):
        return bridge(h_out, h_in)
    return bridge(h_out)


def load_bridge_and_feat_fuse(fb_path: Path, hidden_size: int, eps: float, n_feat_layers: int = 5):
    """Auto-detect and load FeedbackBridge (v1 or v2) and optional FeatFuse from safetensors."""
    bridge, feat_fuse = None, None
    if not fb_path.exists():
        return bridge, feat_fuse
    fb_w = mx.load(str(fb_path))
    fb_items = [
        (k.removeprefix("feedback_bridge."), v.astype(mx.bfloat16))
        for k, v in fb_w.items()
        if k.startswith("feedback_bridge.")
    ]
    ff_items = [
        (k.removeprefix("feat_fuse."), v.astype(mx.bfloat16))
        for k, v in fb_w.items()
        if k.startswith("feat_fuse.")
    ]
    if fb_items:
        if "feedback_bridge.gate.weight" in fb_w:
            rank = int(fb_w["feedback_bridge.gate.weight"].shape[0])
            bridge = FeedbackBridgeV2(hidden_size, rank=rank, eps=eps)
        else:
            rank = int(fb_w["feedback_bridge.up.weight"].shape[0])
            bridge = FeedbackBridge(hidden_size, rank=rank, eps=eps)
        bridge.load_weights(fb_items, strict=True)
    if ff_items:
        ff_rank = int(fb_w["feat_fuse.up.weight"].shape[0])
        feat_fuse = FeatFuse(hidden_size * n_feat_layers, rank=ff_rank, out_dim=hidden_size, eps=eps)
        feat_fuse.load_weights(ff_items, strict=True)
    return bridge, feat_fuse


class BailingMTPLayer(nn.Module):
    """Layer-72 Multi-Token Prediction / EAGLE-3 drafter sharing embeddings & head with target model."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.rms_eps = args.rms_norm_eps
        self.enorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.hnorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.eh_proj = nn.Linear(args.hidden_size * 2, args.hidden_size, bias=False)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.attention = BailingMLA(args)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.mlp = BailingSparseMoE(args)
        self.final_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        # Attached after construction (not registered in nn.Module weights dict)
        self._embed = None
        self._lm_head = None
        self._q4_head = None
        self._bridge = None
        self._feat_fuse = None
        self._feat_layers = None
        self._is_q4_dense = False
        self._eagle3_cfg = None
        self._last_router_inds = None
        self._aux = _AuxBox()
        self._recent_toks: deque[int] = deque(maxlen=64)
        self._sl_lut: bytearray | None = None
        self._shortlist_active: bool = False

    def reset_vocab_window(self, seed_tokens: Sequence[int] = ()) -> bool:
        """Reset 64-token sliding window for dynamic shortlist fallback (O(1) CPU)."""
        self._recent_toks.clear()
        return self.update_vocab_window(seed_tokens)

    def update_vocab_window(self, token_ids: Sequence[int]) -> bool:
        """Update 64-token sliding window; auto-falls back to full q4_head if any recent token is OOV."""
        lut = self._sl_lut
        if lut is None:
            self._shortlist_active = False
            return False
        if token_ids:
            tail = token_ids[-64:] if len(token_ids) > 64 else token_ids
            for t in tail:
                self._recent_toks.append(int(t))
        n_lut = len(lut)
        self._shortlist_active = all(0 <= t < n_lut and lut[t] for t in self._recent_toks)
        return self._shortlist_active

    def _build_fused_q4_projs(self) -> None:
        """Build merged Q4 projections (se.gate+up, q_a+kv_a) for fewer Metal launches."""
        se = self.mlp.shared_experts
        if isinstance(se.gate_proj, nn.QuantizedLinear) and isinstance(se.up_proj, nn.QuantizedLinear):
            merged_se = _concat_quantized_linear([se.gate_proj, se.up_proj])
            merged_se.eval()
            self._aux.d["se_gate_up"] = merged_se
        attn = self.attention
        if (
            hasattr(attn, "q_a_proj")
            and isinstance(attn.q_a_proj, nn.QuantizedLinear)
            and isinstance(attn.kv_a_proj_with_mqa, nn.QuantizedLinear)
        ):
            merged_qkv = _concat_quantized_linear([attn.q_a_proj, attn.kv_a_proj_with_mqa])
            merged_qkv.eval()
            self._aux.d["qkv_a_proj"] = merged_qkv

    def configure_head_mode(
        self,
        *,
        rescore_top_k: int = 0,
        shortlist_ids: mx.array | None = None,
    ) -> None:
        """Configure optional draft-head acceleration (2-stage Top-K BF16 rescore and/or vocab shortlist)."""
        self._aux.d["rescore_top_k"] = int(rescore_top_k)
        if shortlist_ids is not None and self._q4_head is not None:
            sl_ids = shortlist_ids.astype(mx.uint32)
            q4 = self._q4_head
            sl_head = nn.QuantizedLinear(
                int(q4.scales.shape[1]) * q4.group_size,
                int(sl_ids.shape[0]),
                bias=False,
                group_size=q4.group_size,
                bits=q4.bits,
                mode=q4.mode,
            )
            sl_head.weight = q4.weight[sl_ids]
            sl_head.scales = q4.scales[sl_ids]
            if hasattr(q4, "biases") and q4.biases is not None:
                sl_head.biases = q4.biases[sl_ids]
            sl_head.eval()
            mx.eval(sl_ids, sl_head.parameters())
            self._aux.d["shortlist_ids"] = sl_ids
            self._aux.d["shortlist_head"] = sl_head
            vocab_sz = int(q4.weight.shape[0])
            lut = bytearray(vocab_sz)
            for tid in sl_ids.tolist():
                if 0 <= tid < vocab_sz:
                    lut[tid] = 1
            self._sl_lut = lut
            self.update_vocab_window(())
        else:
            self._aux.d.pop("shortlist_ids", None)
            self._aux.d.pop("shortlist_head", None)
            self._sl_lut = None
            self._shortlist_active = False

    @classmethod
    def _load_base_mtp(cls, target_model, mtp_file: Path, *, with_q4_head: bool) -> "BailingMTPLayer":
        if not mtp_file.exists():
            raise FileNotFoundError(f"MTP weights not found: {mtp_file}")
        mtp = cls(target_model.args)
        w = mx.load(str(mtp_file))

        def pred(p, m):
            if not hasattr(m, "to_quantized") or f"{p}.scales" not in w:
                return False
            return infer_quant_config(p, m, w)

        nn.quantize(mtp, group_size=32, bits=4, mode="mxfp4", class_predicate=pred)
        mtp.load_weights(list(w.items()), strict=True)
        mtp._embed = target_model.model.word_embeddings
        mtp._lm_head = target_model.lm_head
        if with_q4_head:
            mtp._q4_head = nn.QuantizedLinear.from_linear(
                target_model.lm_head, group_size=64, bits=4, mode="affine"
            )
            mtp._q4_head.eval()
            mx.eval(mtp._q4_head.parameters())
        return mtp

    @classmethod
    def from_target(
        cls,
        target_model,
        model_dir: str | Path,
        *,
        with_q4_head: bool = True,
    ) -> "BailingMTPLayer":
        mtp = cls._load_base_mtp(
            target_model, Path(model_dir) / "mtp" / "model-mtp.safetensors", with_q4_head=with_q4_head
        )
        mtp.eval()
        mx.eval(mtp.parameters())
        return mtp

    @classmethod
    def from_eagle3(
        cls,
        target_model,
        eagle3_dir: str | Path,
        *,
        with_q4_head: bool = True,
        q4_head: nn.Module | None = None,
        q4_dense: bool = True,
        adapter_bits: int | None = None,
        rescore_top_k: int | None = None,
        shortlist_size: int | None = None,
        model_dir: str | Path | None = None,
    ) -> "BailingMTPLayer":
        """Load trained EAGLE-3 checkpoint (MTP + FeedbackBridge + FeatFuse) and optional Q4-Dense quantization."""
        edir = Path(eagle3_dir)
        mtp = cls._load_base_mtp(
            target_model,
            edir / "model-mtp.safetensors",
            with_q4_head=(with_q4_head and q4_head is None),
        )
        if q4_head is not None:
            mtp._q4_head = q4_head
        cfg = None
        cfg_path = edir / "eagle3_config.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text())
        feat_layers = tuple(cfg.get("feat_layers", FEAT_LAYERS)) if cfg else FEAT_LAYERS

        args = target_model.args
        bridge, feat_fuse = load_bridge_and_feat_fuse(
            edir / "feedback_bridge.safetensors",
            args.hidden_size,
            args.rms_norm_eps,
            n_feat_layers=len(feat_layers),
        )

        # Keep tiny residual adapters (FeedbackBridge ~8.9MB, FeatFuse ~6.7MB Q8) in Q8 by default (<0.03 ms, preserves E_acc)
        if adapter_bits is None:
            adapter_bits = int(os.environ.get("LINGSANTE_E3_ADAPTER_BITS", "8"))

        if q4_dense:
            def qpred_dense(p, m):
                if not hasattr(m, "to_quantized") or isinstance(m, nn.QuantizedLinear):
                    return False
                if p == "eh_proj" or p.startswith("mlp.shared_experts."):
                    return True
                if p in (
                    "attention.q_a_proj",
                    "attention.q_b_proj",
                    "attention.kv_a_proj_with_mqa",
                    "attention.dense",
                ):
                    return True
                return False

            nn.quantize(mtp, group_size=64, bits=4, mode="affine", class_predicate=qpred_dense)
            if adapter_bits in (4, 8):
                if bridge is not None:
                    nn.quantize(bridge, group_size=64, bits=adapter_bits, mode="affine")
                if feat_fuse is not None:
                    nn.quantize(feat_fuse, group_size=64, bits=adapter_bits, mode="affine")
            mtp._is_q4_dense = True
            mtp._build_fused_q4_projs()

        mtp.eval()
        eval_arrs = [mtp.parameters()]
        if bridge is not None:
            bridge.eval()
            eval_arrs.append(bridge.parameters())
        if feat_fuse is not None:
            feat_fuse.eval()
            eval_arrs.append(feat_fuse.parameters())
        for mod in mtp._aux.d.values():
            if isinstance(mod, nn.Module):
                eval_arrs.append(mod.parameters())
        mx.eval(*eval_arrs)

        mtp._bridge = bridge
        mtp._feat_fuse = feat_fuse
        mtp._feat_layers = feat_layers if feat_fuse is not None else None
        mtp._eagle3_cfg = cfg

        if rescore_top_k is None:
            rescore_top_k = int(os.environ.get("LINGSANTE_E3_RESCORE_K", "0"))
        if shortlist_size is None:
            shortlist_size = int(os.environ.get("LINGSANTE_E3_SHORTLIST", "81920"))
        sl_ids = None
        if shortlist_size > 0:
            mdir = Path(model_dir) if model_dir is not None else edir.parent
            sl_ids = _build_bpe_non_cjk_shortlist(mdir, shortlist_size, args.vocab_size)
        if rescore_top_k > 0 or sl_ids is not None:
            mtp.configure_head_mode(rescore_top_k=rescore_top_k, shortlist_ids=sl_ids)

        return mtp

    def apply_feat_fuse(self, h_final: mx.array, captured: mx.array | None) -> mx.array:
        if self._feat_fuse is None or captured is None:
            return h_final
        return (
            h_final.astype(mx.float32) + self._feat_fuse(captured.astype(mx.bfloat16)).astype(mx.float32)
        ).astype(mx.bfloat16)

    def apply_feedback_bridge(self, h_out: mx.array, h_in: mx.array) -> mx.array:
        return apply_bridge(self._bridge, h_out, h_in)

    def make_cache(self, C: int = 1) -> KVCache | list[KVCache]:
        return KVCache() if C == 1 else [KVCache() for _ in range(C)]

    def _mlp_block(self, residual: mx.array) -> mx.array:
        mlp_in = self.post_attention_layernorm(residual)
        return residual + self.mlp(mlp_in)

    def _mlp_rows(self, h_post: mx.array) -> mx.array:
        mlp_in = mx.fast.rms_norm(
            h_post, self.post_attention_layernorm.weight, self.post_attention_layernorm.eps
        )
        gate = self.mlp.gate
        inds, wts = gate_rows(
            mlp_in,
            gate.weight,
            gate.expert_bias,
            top_k=gate.top_k,
            n_group=gate.n_group,
            topk_group=gate.topk_group,
            routed_scaling_factor=gate.routed_scaling_factor,
        )
        # Expose Layer-72 router expert indices for A2 expert-budget-aware dynamic stopping
        self._last_router_inds = inds
        se = self.mlp.shared_experts
        se_gu = self._aux.d.get("se_gate_up")
        if se_gu is not None:
            gu = se_gu(mlp_in)
            half = gu.shape[-1] // 2
            sh = se.down_proj(swiglu(gu[..., :half], gu[..., half:]))
        elif isinstance(se.gate_proj, nn.QuantizedLinear):
            sh = se.down_proj(swiglu(se.gate_proj(mlp_in), se.up_proj(mlp_in)))
        else:
            sh = gemv_rows(
                gemv_swiglu_rows(mlp_in, se.gate_proj.weight, se.up_proj.weight),
                se.down_proj.weight,
            )
        sw = self.mlp.switch_mlp
        dedup = moe_dedup_rows(inds) if h_post.shape[0] >= 5 else None
        h_mid = moe_gate_up_swiglu_rows(
            mlp_in,
            sw.gate_proj.weight,
            sw.gate_proj.scales,
            sw.up_proj.weight,
            sw.up_proj.scales,
            inds,
            dedup=dedup,
        )
        return moe_down_combine_rows(
            h_mid,
            sw.down_proj.weight,
            sw.down_proj.scales,
            inds,
            wts,
            shared=sh,
            residual=h_post,
            dedup=dedup,
        )

    def _kv_project(self, toks_2d: mx.array, prev_3d: mx.array) -> tuple[mx.array, mx.array]:
        embeds = self._embed(toks_2d)
        e_norm = mx.fast.rms_norm(embeds, self.enorm.weight, self.enorm.eps)
        h_norm = mx.fast.rms_norm(prev_3d, self.hnorm.weight, self.hnorm.eps)
        h = self.eh_proj(mx.concatenate([e_norm, h_norm], axis=-1))
        x_a = mx.fast.rms_norm(h, self.input_layernorm.weight, self.input_layernorm.eps)
        attn = self.attention
        L_tok = int(toks_2d.shape[1])
        compressed = attn.kv_a_proj_with_mqa(x_a)
        compressed, k_rope = mx.split(compressed, [attn.kv_lora_rank], axis=-1)
        kv_latent = attn.kv_a_layernorm(compressed)
        k_rope = k_rope.reshape(1, L_tok, 1, attn.qk_rope_head_dim).transpose(0, 2, 1, 3)
        return kv_latent, k_rope

    def kv_append(self, token_ids: mx.array, prev_hidden: mx.array, cache: KVCache) -> None:
        """Bit-exact KV-only MTP cache update (skips Q/SDPA/dense/MLP/lm_head)."""
        toks_2d = token_ids.reshape(1, -1)
        L_tok = int(toks_2d.shape[1])
        if L_tok <= 0:
            return
        prev_3d = prev_hidden.reshape(1, L_tok, -1)
        kv_latent, k_rope = self._kv_project(toks_2d, prev_3d)
        k_rope = self.attention.rope(k_rope, offset=cache.offset)
        cache.update_and_fetch(mx.expand_dims(kv_latent, axis=1), k_rope)

    def kv_append_varlen(
        self, token_ids_flat: mx.array, prev_hidden_flat: mx.array, caches: list[KVCache], Ls: list[int]
    ) -> None:
        """Single-pass batched KV-only MTP cache update for variable stream lengths."""
        B = sum(Ls)
        if B <= 0:
            return
        assert token_ids_flat.size == B and len(caches) == len(Ls)
        kv_latent, k_rope = self._kv_project(
            token_ids_flat.reshape(1, B), prev_hidden_flat.reshape(1, B, -1)
        )
        attn = self.attention
        pos = 0
        for c_cache, lc in zip(caches, Ls):
            if lc <= 0:
                continue
            kr_s = attn.rope(k_rope[:, :, pos : pos + lc, :], offset=c_cache.offset)
            kl_s = kv_latent[:, pos : pos + lc, :]
            c_cache.update_and_fetch(mx.expand_dims(kl_s, axis=1), kr_s)
            pos += lc

    def prefill(self, tokens_shifted: mx.array, hidden_states: mx.array, cache: KVCache) -> mx.array:
        """Populate `cache` during prompt prefill (`tokens_shifted` is `(1, L)`, `hidden_states` is `(1, L, H)`)."""
        embeds = self._embed(tokens_shifted)
        e_norm = mx.fast.rms_norm(embeds, self.enorm.weight, self.enorm.eps)
        h_norm = mx.fast.rms_norm(hidden_states, self.hnorm.weight, self.hnorm.eps)
        h = self.eh_proj(mx.concatenate([e_norm, h_norm], axis=-1))
        mask = create_attention_mask(h, cache, return_array=True)
        h = h + self.attention(self.input_layernorm(h), mask=mask, cache=cache)
        h = self._mlp_block(h)
        return self.final_layernorm(h)

    def _mla_q4_step(
        self,
        h_in: mx.array,
        res: mx.array,
        cache: Any,
        *,
        C: int,
        q_a: mx.array | None = None,
        kv_a_full: mx.array | None = None,
    ) -> mx.array:
        """Single-token (L=1) fused Q4-Dense MLA step for C=1..4 over KVCache or list[KVCache]."""
        attn = self.attention
        if q_a is None or kv_a_full is None:
            qkv_fused = self._aux.d.get("qkv_a_proj")
            if qkv_fused is not None:
                qkv_a = qkv_fused(h_in)
                q_a = qkv_a[:, : attn.q_lora_rank]
                kv_a_full = qkv_a[:, attn.q_lora_rank :]
            else:
                q_a = attn.q_a_proj(h_in)
                kv_a_full = attn.kv_a_proj_with_mqa(h_in)
        q_a_norm = mx.fast.rms_norm(q_a, attn.q_a_layernorm.weight, attn.q_a_layernorm.eps)
        q_full = attn.q_b_proj(q_a_norm)
        g_raw = gemv_rows(h_in, attn.g_proj.weight)
        caches = cache if isinstance(cache, list) else [cache]
        assert len(caches) == C
        offs = int(caches[0].offset) if C == 1 else mx.array([int(ci.offset) for ci in caches], dtype=mx.int32)
        q_nope, q_rope, kv_latent, k_rope = mla_prep_kv_rope_rows(
            q_full,
            kv_a_full,
            attn.kv_a_layernorm.weight,
            offs,
            kv_eps=attn.kv_a_layernorm.eps,
            base=float(getattr(attn.rope, "base", 6000000.0)),
            scale=float(getattr(attn.rope, "scale", 1.0)),
        )
        q_nope = mla_multilinear_lin(q_nope, attn.embed_q)
        scale = attn.scale
        if C == 1:
            kl_c, kr_c = caches[0].update_and_fetch(kv_latent, k_rope)
            pe = (q_rope * scale) @ kr_c.swapaxes(-1, -2)
            out = mx.fast.scaled_dot_product_attention(q_nope, kl_c, kl_c, scale=scale, mask=pe)
        else:
            outs = []
            for c in range(C):
                kl_c, kr_c = caches[c].update_and_fetch(kv_latent[c : c + 1], k_rope[c : c + 1])
                _mla_append_causal_slice(outs, kl_c, kr_c, q_nope, q_rope, c, 0, kl_c.shape[2] - 1, 1, scale)
            out = mx.concatenate(outs, axis=0)
        gated = mla_multilinear_lin(out, attn.unembed_out, g_raw=g_raw)
        return res + attn.dense(gated)

    def _mtp_embed_and_mla_in(self, token_ids: mx.array, prev_hidden: mx.array, B: int):
        toks_flat = token_ids.reshape(-1)
        prev_flat = prev_hidden.reshape(B, -1)
        embeds = self._embed(toks_flat)
        if prev_flat.shape[-1] == 4096 and embeds.dtype == prev_flat.dtype:
            eh_in = eh_norm_concat_rows(embeds, prev_flat, self.enorm.weight, self.hnorm.weight, self.enorm.eps)
        else:
            e_norm = mx.fast.rms_norm(embeds, self.enorm.weight, self.enorm.eps)
            h_norm = mx.fast.rms_norm(prev_flat, self.hnorm.weight, self.hnorm.eps)
            eh_in = mx.concatenate([e_norm, h_norm], axis=-1)
        h = self.eh_proj(eh_in) if self._is_q4_dense else gemv_rows(eh_in, self.eh_proj.weight)
        h_in = mx.fast.rms_norm(h, self.input_layernorm.weight, self.input_layernorm.eps)
        return h, h_in

    def _mtp_head(self, head_in: mx.array, *, use_q4_head: bool):
        if use_q4_head and self._q4_head is not None:
            return self._q4_head(head_in)
        return gemv_rows(head_in, self._lm_head.weight)

    def head_argmax_prob(
        self, head_in: mx.array, *, use_q4_head: bool = True
    ) -> tuple[mx.array, mx.array]:
        """Compute `(argmax_tok (C,) u32, p_top1 (C,) f32)` using fused Metal reduction (+ dynamic shortlist/rescore)."""
        use_sl = use_q4_head and self._shortlist_active and ("shortlist_head" in self._aux.d)
        sl_head = self._aux.d["shortlist_head"] if use_sl else None
        sl_ids = self._aux.d["shortlist_ids"] if use_sl else None
        rescore_k = int(self._aux.d.get("rescore_top_k", 0)) if use_q4_head else 0

        if sl_head is not None:
            logits = sl_head(head_in)
        else:
            logits = self._mtp_head(head_in, use_q4_head=use_q4_head)

        if rescore_k > 0 and self._lm_head is not None:
            C = int(logits.shape[0])
            idx_k = mx.argpartition(-logits, kth=rescore_k - 1, axis=-1)[..., :rescore_k]
            cand_ids = sl_ids[idx_k] if sl_ids is not None else idx_k.astype(mx.uint32)
            # Gather K BF16 rows of lm_head: (C, K, 4096) -> dot with head_in (C, 4096, 1)
            w_k = self._lm_head.weight[cand_ids]
            lg_k = mx.matmul(w_k, head_in.astype(w_k.dtype)[:, :, None]).squeeze(-1).astype(mx.float32)
            best_i = mx.argmax(lg_k, axis=-1)
            tok = mx.take_along_axis(cand_ids, best_i[:, None], axis=-1).squeeze(-1)
            max_k = mx.max(lg_k, axis=-1)
            p1 = mx.exp(max_k - mx.logsumexp(lg_k, axis=-1))
            return tok, p1

        arg, p1 = logits_argmax_prob(logits)
        if sl_ids is not None:
            arg = sl_ids[arg]
        return arg, p1

    def refresh_and_step0(
        self,
        accepted_toks_flat: mx.array | None,
        accepted_hs_flat: mx.array | None,
        cur_toks: mx.array,
        cur_hs: mx.array,
        caches: list[KVCache] | KVCache,
        Ls_acc: list[int],
        *,
        use_q4_head: bool = True,
        return_logits: bool = True,
        return_argmax_prob: bool = False,
    ):
        """Fused accepted-prefix KV-append + step-0 draft sharing eh_proj & qkv_a_proj reads."""
        cache_list = caches if isinstance(caches, list) else [caches]
        C = len(cache_list)
        assert len(Ls_acc) == C
        B_acc = sum(Ls_acc)
        if B_acc <= 0 or accepted_toks_flat is None or accepted_hs_flat is None:
            return self.step_rows(
                cur_toks,
                cur_hs,
                caches,
                C=C,
                L=1,
                use_q4_head=use_q4_head,
                return_logits=return_logits,
                return_argmax_prob=return_argmax_prob,
            )

        if not self._is_q4_dense:
            nz_caches = [c_i for c_i, lc in zip(cache_list, Ls_acc) if lc > 0]
            nz_ls = [lc for lc in Ls_acc if lc > 0]
            self.kv_append_varlen(accepted_toks_flat, accepted_hs_flat, nz_caches, nz_ls)
            return self.step_rows(
                cur_toks,
                cur_hs,
                caches,
                C=C,
                L=1,
                use_q4_head=use_q4_head,
                return_logits=return_logits,
                return_argmax_prob=return_argmax_prob,
            )

        B_tot = B_acc + C
        cur_t1 = cur_toks.reshape(C)
        cur_h2 = cur_hs.reshape(C, -1)
        all_toks = mx.concatenate([accepted_toks_flat.reshape(B_acc), cur_t1], axis=0)
        all_hs = mx.concatenate([accepted_hs_flat.reshape(B_acc, -1), cur_h2], axis=0)

        h_all, h_in_all = self._mtp_embed_and_mla_in(all_toks, all_hs, B_tot)
        attn = self.attention
        qkv_fused = self._aux.d.get("qkv_a_proj")
        if qkv_fused is not None:
            qkv_all = qkv_fused(h_in_all)
            q_a_cur = qkv_all[B_acc:, : attn.q_lora_rank]
            kv_a_all = qkv_all[:, attn.q_lora_rank :]
        else:
            kv_a_all = attn.kv_a_proj_with_mqa(h_in_all)
            q_a_cur = attn.q_a_proj(h_in_all[B_acc:])

        # Append accepted prefix rows [0 .. B_acc-1] to per-stream KVCaches
        comp_acc, kr_acc = mx.split(kv_a_all[:B_acc], [attn.kv_lora_rank], axis=-1)
        kl_acc = attn.kv_a_layernorm(comp_acc)  # (B_acc, 512)
        kr_acc = kr_acc.reshape(1, B_acc, 1, attn.qk_rope_head_dim).transpose(0, 2, 1, 3)
        pos = 0
        for c_cache, lc in zip(cache_list, Ls_acc):
            if lc <= 0:
                continue
            kr_s = attn.rope(kr_acc[:, :, pos : pos + lc, :], offset=c_cache.offset)
            kl_s = kl_acc[pos : pos + lc, :].reshape(1, 1, lc, attn.kv_lora_rank)
            c_cache.update_and_fetch(kl_s, kr_s)
            pos += lc

        # Step-0 on current anchor rows [B_acc .. B_tot-1]
        h_post = self._mla_q4_step(
            h_in_all[B_acc:],
            h_all[B_acc:],
            caches,
            C=C,
            q_a=q_a_cur,
            kv_a_full=kv_a_all[B_acc:],
        )
        h_out = mx.fast.rms_norm(self._mlp_rows(h_post), self.final_layernorm.weight, self.final_layernorm.eps)
        if return_argmax_prob:
            tok, p1 = self.head_argmax_prob(h_out, use_q4_head=use_q4_head)
            return h_out, tok, p1
        if not return_logits:
            return h_out, None
        logits = self._mtp_head(h_out, use_q4_head=use_q4_head)
        return h_out, logits

    def step_rows(
        self,
        token_ids: mx.array,
        prev_hidden: mx.array,
        cache: Any,
        *,
        C: int = 1,
        L: int = 1,
        use_q4_head: bool = True,
        head_only_last: bool = True,
        return_logits: bool = True,
        return_argmax_prob: bool = False,
    ):
        """Fast row-kernel / Q4-Dense MTP step supporting `(C,)` single-step (`L=1`) and `(C*L,)` refresh (`C*L <= 24`).

        `token_ids`: `(C*L,)` or `(C, L)` uint32/int32.
        `prev_hidden`: `(C*L, 4096)` or `(C, L, 4096)` bf16.
        Returns `(next_hidden_last (C, 4096), logits (C, V) or (C, L, V) or None)`,
        or `(next_hidden_last (C, 4096), argmax (C,) u32, p_top1 (C,) f32)` when `return_argmax_prob=True`.
        """
        B = C * L
        assert 1 <= B <= 24

        if self._is_q4_dense and L > 1:
            toks_2d = token_ids.reshape(C, L)
            prev_3d = prev_hidden.reshape(C, L, -1)
            caches = cache if isinstance(cache, list) else [cache]
            if not return_logits and not return_argmax_prob:
                self.kv_append_varlen(toks_2d.reshape(-1), prev_3d.reshape(B, -1), caches, [L] * C)
                return prev_3d[:, -1, :], None
            if head_only_last or return_argmax_prob:
                return self.refresh_and_step0(
                    toks_2d[:, :-1].reshape(-1),
                    prev_3d[:, :-1, :].reshape(C * (L - 1), -1),
                    toks_2d[:, -1],
                    prev_3d[:, -1, :],
                    cache,
                    [L - 1] * C,
                    use_q4_head=use_q4_head,
                    return_logits=return_logits,
                    return_argmax_prob=return_argmax_prob,
                )
            h_list, l_list = [], []
            for t in range(L):
                ht, lt = self.step_rows(
                    toks_2d[:, t],
                    prev_3d[:, t, :],
                    cache,
                    C=C,
                    L=1,
                    use_q4_head=use_q4_head,
                    return_logits=True,
                )
                h_list.append(ht)
                l_list.append(lt)
            return h_list[-1], mx.stack(l_list, axis=1)

        h, h_in = self._mtp_embed_and_mla_in(token_ids, prev_hidden, B)

        if self._is_q4_dense:
            h_post = self._mla_q4_step(h_in, h, cache, C=C)
        elif L == 1 and not isinstance(cache, list):
            h_post = mla_layer_rows(h_in, h, self.attention, cache)
        else:
            h_post = mla_verify_layer_rows(h_in, h, self.attention, cache, C=C, L=L)

        h_out = mx.fast.rms_norm(self._mlp_rows(h_post), self.final_layernorm.weight, self.final_layernorm.eps)
        last_hidden = h_out.reshape(C, L, -1)[:, -1, :]

        if return_argmax_prob:
            tok, p1 = self.head_argmax_prob(last_hidden, use_q4_head=use_q4_head)
            return last_hidden, tok, p1

        if not return_logits:
            return last_hidden, None

        head_in = last_hidden if (head_only_last or L == 1) else h_out
        logits = self._mtp_head(head_in, use_q4_head=use_q4_head)
        if not head_only_last and L > 1:
            logits = logits.reshape(C, L, -1)
        return last_hidden, logits

    def step_varlen_rows(
        self,
        token_ids_flat: mx.array,
        prev_hidden_flat: mx.array,
        caches: list[KVCache],
        Ls: list[int],
        *,
        use_q4_head: bool = True,
        return_logits: bool = True,
        return_argmax_prob: bool = False,
    ):
        """Single-pass variable-length MTP refresh for heterogeneous stream accept counts (`sum(Ls) <= 24`)."""
        B = sum(Ls)
        assert 1 <= B <= 24 and token_ids_flat.size == B and prev_hidden_flat.reshape(B, -1).shape[0] == B
        assert len(caches) == len(Ls)

        if self._is_q4_dense:
            prev_2d = prev_hidden_flat.reshape(B, -1)
            tok_flat = token_ids_flat.reshape(-1)
            if not return_logits and not return_argmax_prob:
                self.kv_append_varlen(tok_flat, prev_2d, caches, Ls)
                return None, None
            pre_toks, pre_hs, pre_ls = [], [], []
            last_toks, last_hs = [], []
            pos = 0
            for lc in Ls:
                if lc > 1:
                    pre_toks.append(tok_flat[pos : pos + lc - 1])
                    pre_hs.append(prev_2d[pos : pos + lc - 1])
                    pre_ls.append(lc - 1)
                else:
                    pre_ls.append(0)
                last_toks.append(tok_flat[pos + lc - 1])
                last_hs.append(prev_2d[pos + lc - 1])
                pos += lc
            acc_toks = mx.concatenate(pre_toks, axis=0) if pre_toks else None
            acc_hs = mx.concatenate(pre_hs, axis=0) if pre_hs else None
            return self.refresh_and_step0(
                acc_toks,
                acc_hs,
                mx.stack(last_toks),
                mx.stack(last_hs, axis=0),
                caches,
                pre_ls,
                use_q4_head=use_q4_head,
                return_logits=return_logits,
                return_argmax_prob=return_argmax_prob,
            )

        h, h_in = self._mtp_embed_and_mla_in(token_ids_flat, prev_hidden_flat, B)
        h_post = mla_verify_varlen_rows(h_in, h, self.attention, caches, Ls)
        h_out = mx.fast.rms_norm(self._mlp_rows(h_post), self.final_layernorm.weight, self.final_layernorm.eps)

        end_rows = []
        pos = 0
        for lc in Ls:
            pos += lc
            end_rows.append(pos - 1)
        last_hidden = h_out[mx.array(end_rows, dtype=mx.int32)]

        if return_argmax_prob:
            tok, p1 = self.head_argmax_prob(last_hidden, use_q4_head=use_q4_head)
            return last_hidden, tok, p1
        if not return_logits:
            return last_hidden, None
        logits = self._mtp_head(last_hidden, use_q4_head=use_q4_head)
        return last_hidden, logits
