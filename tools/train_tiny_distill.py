"""LoRA logit-distillation feasibility gate and trainer for inclusionAI/Ling-3.0-tiny on 512S1."""
import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

os.environ["LINGSANTE_FAST_DECODE"] = "0"
os.environ["LINGSANTE_FASTDECODE"] = "0"
os.environ["LINGSANTE_FP8"] = "0"

_root = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(_root / "vendor"), str(_root)]

import mlx_lm  # noqa: E402,F401
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import mlx.optimizers as optim  # noqa: E402
from mlx.utils import tree_flatten, tree_map, tree_unflatten  # noqa: E402
from mlx_lm.models.bailing_moe_v3.layers import BailingDecoderLayer  # noqa: E402
from mlx_lm.utils import load_model, maybe_set_recommended_wired_limit, save_model  # noqa: E402

# Verified Step 1 C=1 target verify latencies (ms) from bench/profile_verify_c1.py
_VERIFY_ANCHORS = [
    (1, 39.09),
    (3, 48.94),
    (5, 57.23),
    (8, 70.32),
    (9, 86.31),
    (10, 90.05),
    (12, 99.55),
    (16, 121.25),
]


def verify_cost_ms(L: float) -> float:
    if L <= _VERIFY_ANCHORS[0][0]:
        return _VERIFY_ANCHORS[0][1]
    if L >= _VERIFY_ANCHORS[-1][0]:
        return _VERIFY_ANCHORS[-1][1]
    for (x0, y0), (x1, y1) in zip(_VERIFY_ANCHORS[:-1], _VERIFY_ANCHORS[1:]):
        if x0 <= L <= x1:
            return y0 + ((L - x0) / (x1 - x0)) * (y1 - y0)
    return _VERIFY_ANCHORS[-1][1]


class LoRALinear(nn.Module):
    """Zero-init fp32 LoRA adapter over a frozen bfloat16 Linear weight (Epoch-0 bit-identical)."""

    def __init__(self, linear: nn.Linear, r: int = 32, alpha: float = 32.0):
        super().__init__()
        out_dim, in_dim = linear.weight.shape
        self.weight = linear.weight
        self.bias = linear.bias if hasattr(linear, "bias") and linear.bias is not None else None
        self.r = int(r)
        self.scale = float(alpha) / float(r)
        self.lora_A = mx.random.normal((r, in_dim), dtype=mx.float32) * (1.0 / math.sqrt(in_dim))
        self.lora_B = mx.zeros((out_dim, r), dtype=mx.float32)
        keys = ["weight", "bias"] if self.bias is not None else ["weight"]
        self.freeze(keys=keys, recurse=False)

    def __call__(self, x: mx.array) -> mx.array:
        base = x @ self.weight.T
        if self.bias is not None:
            base = base + self.bias
        xf = x.astype(mx.float32)
        lora = ((xf @ self.lora_A.T) @ self.lora_B.T) * self.scale
        return (base.astype(mx.float32) + lora).astype(x.dtype)

    def to_linear(self) -> nn.Linear:
        out_dim, in_dim = self.weight.shape
        lin = nn.Linear(in_dim, out_dim, bias=(self.bias is not None))
        lin.weight = (self.weight.astype(mx.float32) + self.scale * (self.lora_B @ self.lora_A)).astype(mx.bfloat16)
        if self.bias is not None:
            lin.bias = self.bias
        return lin


def attach_lora_and_freeze(model, r_attn: int = 32, alpha_attn: float = 32.0, r_head: int = 64, alpha_head: float = 64.0):
    """Wrap KDA, MLA, dense/shared MLP, and lm_head in zero-init LoRALinear; freeze all else."""
    for li, layer in enumerate(model.model.layers):
        attn = layer.attention
        if layer.is_linear:
            for attr in ("q_proj", "k_proj", "v_proj", "f_proj", "g_proj", "o_proj"):
                setattr(attn, attr, LoRALinear(getattr(attn, attr), r=r_attn, alpha=alpha_attn))
        else:
            for attr in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "dense"):
                setattr(attn, attr, LoRALinear(getattr(attn, attr), r=r_attn, alpha=alpha_attn))
        mlp = layer.mlp
        mlp_tgt = mlp.shared_experts if hasattr(mlp, "shared_experts") else mlp
        for attr in ("gate_proj", "up_proj", "down_proj"):
            setattr(mlp_tgt, attr, LoRALinear(getattr(mlp_tgt, attr), r=r_attn, alpha=alpha_attn))

    model.lm_head = LoRALinear(model.lm_head, r=r_head, alpha=alpha_head)
    model.freeze(recurse=True)
    for _, mod in model.named_modules():
        if isinstance(mod, LoRALinear):
            mod.unfreeze(keys=["lora_A", "lora_B"], recurse=False)


def enable_train_grad_checkpoint():
    """Wrap BailingDecoderLayer.__call__ with mx.checkpoint only during model.train()."""
    if getattr(BailingDecoderLayer.__call__, "_tiny_ckpt_wrapped", False):
        return
    orig_call = BailingDecoderLayer.__call__

    def ckpt_call(self, x: mx.array, mask: mx.array | None = None, cache=None) -> mx.array:
        if not self.training:
            return orig_call(self, x, mask, cache)

        def inner_fn(params, h_in):
            self.update(params)
            return orig_call(self, h_in, mask, None)

        return mx.checkpoint(inner_fn)(self.trainable_parameters(), x)

    ckpt_call._tiny_ckpt_wrapped = True
    BailingDecoderLayer.__call__ = ckpt_call


def load_distill_dataset(data_dir: Path):
    """Load 200 train / 20 val sequences (ids + top32_ids/top32_logits) with zero bench/exact overlap."""
    idx_data = json.loads((data_dir / "index.json").read_text())
    shard_cache: dict[str, dict] = {}
    train_seqs, val_seqs = [], []
    for entry in idx_data["entries"]:
        sh = entry["shard"]
        if sh not in shard_cache:
            shard_cache[sh] = mx.load(str(data_dir / "shards" / sh))
        st = shard_cache[sh]
        k = entry["key"]
        ids_mx = st[f"{k}.ids"]
        t32_ids = st[f"{k}.top32_ids"]
        t32_lgs = st[f"{k}.top32_logits"]
        mx.eval(ids_mx, t32_ids, t32_lgs)
        rec = {
            "key": k,
            "idx": entry["idx"],
            "domain": entry["domain"],
            "split": entry["split"],
            "prompt_len": int(entry["prompt_len"]),
            "total_len": int(entry["total_len"]),
            "ids": ids_mx.tolist(),
            "ids_mx": ids_mx,
            "top32_ids": t32_ids,
            "top32_logits": t32_lgs,
        }
        if entry["split"] == "val":
            val_seqs.append(rec)
        else:
            train_seqs.append(rec)
    del shard_cache
    mx.clear_cache()
    print(f"Loaded dataset: {len(train_seqs)} train, {len(val_seqs)} val ({mx.get_active_memory()/1e9:.2f} GB active)", flush=True)
    return train_seqs, val_seqs


def loss_on_sequence(model, seq: dict):
    """Exact fp32 0.8*KL(top32) + 0.2*CE on response tokens [plen-1 : T-1] -> [plen : T]."""
    ids_mx = seq["ids_mx"]
    plen = seq["prompt_len"]
    T = seq["total_len"]
    hidden = model.model(ids_mx[None], cache=None)
    resp_h = hidden[0, plen - 1 : T - 1]
    logits = model.lm_head(resp_h).astype(mx.float32)

    lse = mx.logsumexp(logits, axis=-1, keepdims=True)
    lp = logits - lse

    tgt_ids = ids_mx[plen:T]
    ce = -mx.mean(mx.take_along_axis(lp, tgt_ids[:, None], axis=-1).squeeze(-1))

    t32_ids = seq["top32_ids"][plen - 1 : T - 1]
    t32_probs = mx.softmax(seq["top32_logits"][plen - 1 : T - 1].astype(mx.float32), axis=-1)
    stu_t32_lp = mx.take_along_axis(lp, t32_ids, axis=-1)
    log_pt = mx.log(mx.maximum(t32_probs, 1e-12))
    kl = mx.mean(mx.sum(t32_probs * (log_pt - stu_t32_lp), axis=-1))

    tot = 0.8 * kl + 0.2 * ce
    return tot, (kl, ce)


def _sample_anchors(plen: int, T: int, k_max: int = 15, n_blk: int = 5) -> list[int]:
    lo = plen
    hi = T - 1 - k_max
    if hi < lo:
        return []
    span = hi - lo + 1
    out = []
    for m in range(n_blk):
        b0 = lo + int(m * span / n_blk)
        b1 = lo + int((m + 1) * span / n_blk) - 1
        b1 = max(b0, min(hi, b1))
        out.append((b0 + b1) // 2)
    return out


def eval_tiny_val(model, val_seqs: list[dict], *, k_max: int = 10, n_blk: int = 5) -> dict:
    """Evaluate teacher-forced p1/top2/top4 + unrolled greedy alphas[1..10] and projected tok/s."""
    model.eval()
    n_tot, n_med = 0, 0
    h1_tot, h2_tot, h4_tot = 0, 0, 0
    h1_med = 0

    for seq in val_seqs:
        plen = seq["prompt_len"]
        T = seq["total_len"]
        is_med = seq["domain"] == "med"
        logits = model(seq["ids_mx"][None], cache=model.make_cache())[0]
        pred_sl = logits[plen - 1 : T - 1]
        idx4 = mx.argpartition(-pred_sl, kth=3, axis=-1)[:, :4]
        val4 = mx.take_along_axis(pred_sl, idx4, axis=-1)
        ord4 = mx.argsort(-val4, axis=-1)
        top4 = mx.take_along_axis(idx4, ord4, axis=-1)
        tgts = seq["ids_mx"][plen:T]
        m1 = (top4[:, 0] == tgts)
        m2 = mx.any(top4[:, :2] == tgts[:, None], axis=-1)
        m4 = mx.any(top4 == tgts[:, None], axis=-1)
        mx.eval(m1, m2, m4)
        c1, c2, c4 = int(mx.sum(m1).item()), int(mx.sum(m2).item()), int(mx.sum(m4).item())
        N = int(tgts.shape[0])
        n_tot += N
        h1_tot += c1
        h2_tot += c2
        h4_tot += c4
        if is_med:
            n_med += N
            h1_med += c1

    # Real greedy unrolled autoregressive evaluation across 100 val anchors
    recs = []
    for seq in val_seqs:
        anchors = _sample_anchors(seq["prompt_len"], seq["total_len"], k_max=k_max, n_blk=n_blk)
        ids = seq["ids"]
        ids_mx = seq["ids_mx"]
        is_med = seq["domain"] == "med"
        for a in anchors:
            cache = model.make_cache()
            lg = model(ids_mx[: a + 1][None], cache=cache)[:, -1, :]
            drafts, probs = [], []
            for j in range(k_max):
                lg_f = lg.astype(mx.float32)
                tok_mx = mx.argmax(lg_f, axis=-1)
                ml = mx.max(lg_f, axis=-1, keepdims=True)
                p_mx = mx.exp(mx.max(lg_f - ml, axis=-1)) / mx.sum(mx.exp(lg_f - ml), axis=-1)
                mx.eval(tok_mx, p_mx)
                drafts.append(int(tok_mx.item()))
                probs.append(float(p_mx.item()))
                if j + 1 < k_max:
                    lg = model(tok_mx.reshape(1, 1).astype(mx.uint32), cache=cache)[:, 0, :]
            gold = ids[a + 1 : a + 1 + k_max]
            pos_ok = [d == g for d, g in zip(drafts, gold)]
            mlen = 0
            for ok in pos_ok:
                if ok:
                    mlen += 1
                else:
                    break
            recs.append({"is_med": is_med, "probs": probs, "match_len": mlen, "pos_ok": pos_ok})

    n_rec = len(recs)
    alphas = [round(sum(1 for r in recs if r["pos_ok"][j]) / n_rec, 3) for j in range(k_max)]
    e_k = {k: round(sum(min(r["match_len"], k) + 1 for r in recs) / n_rec, 3) for k in (4, 6, 8, 10)}

    proj = {}
    for label, step_ms in (("5p8ms", 5.8), ("2p0ms", 2.0)):
        fixed = {}
        for k in (4, 6, 8, 10):
            ea = e_k[k]
            cyc = verify_cost_ms(k + 1) + k * step_ms
            fixed[str(k)] = round(1e3 * ea / cyc, 2)
        best_dyn_tps, best_th, best_ea, best_ml = 0.0, 0.25, 0.0, 0.0
        for th in (0.15, 0.25, 0.35, 0.45, 0.55):
            accs, cycs, ls = [], [], []
            for r in recs:
                cum, kb = 1.0, 1
                for j, p in enumerate(r["probs"], start=1):
                    cum *= p
                    if cum >= th:
                        kb = j
                    else:
                        break
                accs.append(min(r["match_len"], kb) + 1)
                ls.append(kb + 1)
                cycs.append(verify_cost_ms(kb + 1) + kb * step_ms)
            tps = 1e3 * sum(accs) / sum(cycs)
            if tps > best_dyn_tps:
                best_dyn_tps = tps
                best_th = th
                best_ea = sum(accs) / n_rec
                best_ml = sum(ls) / n_rec
        proj[label] = {
            "fixed_tok_s": fixed,
            "dyn_best_tok_s": round(best_dyn_tps, 2),
            "dyn_th": best_th,
            "dyn_E_acc": round(best_ea, 3),
            "dyn_mean_L": round(best_ml, 2),
        }

    return {
        "p1_all": round(h1_tot / max(1, n_tot), 4),
        "p1_med": round(h1_med / max(1, n_med), 4),
        "top2_all": round(h2_tot / max(1, n_tot), 4),
        "top4_all": round(h4_tot / max(1, n_tot), 4),
        "alphas": alphas,
        "E_acc": e_k,
        "proj": proj,
    }


def run_startup_tests(src_dir: Path, model, train0: dict, val0: dict) -> float:
    """Verify Ep0 identity, train/eval recurrence parity, trainable isolation, and seq fwd+bwd timing gate."""
    # Test 1: Epoch-0 identity vs unwrapped base model
    base_model, _ = load_model(src_dir)
    base_model.eval()
    mx.eval(base_model.parameters())
    sub_ids = val0["ids_mx"][:64][None]
    model.eval()
    lg_base = base_model(sub_ids, cache=None)
    lg_wrap = model(sub_ids, cache=None)
    mx.eval(lg_base, lg_wrap)
    d_id = float(mx.max(mx.abs(lg_base.astype(mx.float32) - lg_wrap.astype(mx.float32))).item())
    assert d_id == 0.0, f"Startup Test 1 Epoch-0 identity diff={d_id}"
    print(f"Startup test 1 (Epoch-0 identity max_diff={d_id:.2e}): PASS", flush=True)
    del base_model
    mx.clear_cache()

    # Test 2a: Single KDA layer exact parity (gated_delta_ops vs gated_delta_kernel)
    kda_layer = model.model.layers[1]
    assert kda_layer.is_linear
    x_kda = model.model.word_embeddings(sub_ids)
    kda_layer.eval()
    y_eval = kda_layer.attention(kda_layer.input_layernorm(x_kda), mask=None, cache=None)
    kda_layer.train()
    y_tr = kda_layer.attention(kda_layer.input_layernorm(x_kda), mask=None, cache=None)
    mx.eval(y_eval, y_tr)
    max_kda_diff = float(mx.max(mx.abs(y_tr.astype(mx.float32) - y_eval.astype(mx.float32))).item())
    assert max_kda_diff < 1e-2, f"Startup Test 2a KDA parity fail: max_kda_diff={max_kda_diff}"
    print(f"Startup test 2a (Single KDA layer train/eval max_diff={max_kda_diff:.3e}): PASS", flush=True)

    # Test 2b: Full 24-layer MoE train vs eval recurrence parity on val0[:64]
    model.train()
    lg_tr = model(sub_ids, cache=None)
    mx.eval(lg_tr)
    diff_te = mx.abs(lg_tr.astype(mx.float32) - lg_wrap.astype(mx.float32))
    mean_te = float(mx.mean(diff_te).item())
    max_te = float(mx.max(diff_te).item())
    agree_te = float(mx.mean((mx.argmax(lg_tr, axis=-1) == mx.argmax(lg_wrap, axis=-1)).astype(mx.float32)).item())
    assert mean_te < 0.15 and agree_te >= 0.90, f"Startup Test 2b parity fail: mean_diff={mean_te}, max_diff={max_te}, agree={agree_te}"
    print(f"Startup test 2b (Full 24L train/eval mean_diff={mean_te:.4f}, max_diff={max_te:.4f}, argmax_agree={agree_te:.4f}): PASS", flush=True)

    # Test 3: Trainable parameter isolation
    trainable = tree_flatten(model.trainable_parameters())
    n_train = sum(v.size for _, v in trainable)
    for tk, _ in trainable:
        assert tk.endswith(".lora_A") or tk.endswith(".lora_B"), f"Unexpected trainable key: {tk}"
    print(f"Startup test 3 (Trainable isolation: {len(trainable)} tensors, {n_train/1e6:.2f}M params): PASS", flush=True)

    # Test 4: Full-sequence value_and_grad timing gate on train_seqs[0]
    enable_train_grad_checkpoint()
    model.train()
    vg = nn.value_and_grad(model, lambda s: loss_on_sequence(model, s))
    t0 = time.perf_counter()
    (lv, (kl, ce)), grads = vg(train0)
    mx.eval(lv, kl, ce, grads)
    seq_fb_s = time.perf_counter() - t0
    lv_f = float(lv.item())
    gnorm = math.sqrt(sum(float(mx.sum(g * g).item()) for _, g in tree_flatten(grads) if g is not None))
    assert math.isfinite(lv_f) and math.isfinite(gnorm) and gnorm > 1e-6, f"Startup Test 4 bad grad/loss: {lv_f}, {gnorm}"
    print(
        f"Startup test 4 (Full-seq T={train0['total_len']} VJP loss={lv_f:.4f}, gnorm={gnorm:.3e}, seq_fb_s={seq_fb_s:.3f}s): PASS",
        flush=True,
    )
    return seq_fb_s


def _cosine_lr(step: int, total_steps: int, max_lr: float, min_lr: float, warmup_frac: float = 0.10) -> float:
    warm = max(1, int(total_steps * warmup_frac))
    if step < warm:
        return max_lr * (step + 1) / warm
    t = (step - warm) / max(1, total_steps - warm)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * min(1.0, t)))


def _scale_clip_grads(acc, scale: float, max_norm: float):
    flat = [(k, g * scale) for k, g in tree_flatten(acc) if g is not None]
    sq = mx.array(0.0, dtype=mx.float32)
    for _, g in flat:
        sq = sq + mx.sum(g * g)
    norm = mx.sqrt(sq)
    coef = mx.minimum(mx.array(1.0, dtype=mx.float32), max_norm / (norm + 1e-6))
    return tree_unflatten([(k, g * coef) for k, g in flat]), norm


def save_merged_model(src_dir: Path, out_dir: Path, model) -> None:
    """Replace all LoRALinear with merged BF16 nn.Linear and save clean MLX checkpoint."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for layer in model.model.layers:
        attn = layer.attention
        attrs = ("q_proj", "k_proj", "v_proj", "f_proj", "g_proj", "o_proj") if layer.is_linear else ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "dense")
        for attr in attrs:
            mod = getattr(attn, attr)
            if isinstance(mod, LoRALinear):
                setattr(attn, attr, mod.to_linear())
        mlp = layer.mlp
        mlp_tgt = mlp.shared_experts if hasattr(mlp, "shared_experts") else mlp
        for attr in ("gate_proj", "up_proj", "down_proj"):
            mod = getattr(mlp_tgt, attr)
            if isinstance(mod, LoRALinear):
                setattr(mlp_tgt, attr, mod.to_linear())
    if isinstance(model.lm_head, LoRALinear):
        model.lm_head = model.lm_head.to_linear()
    mx.eval(model.parameters())

    for fname in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        src_f = src_dir / fname
        if src_f.exists():
            shutil.copy2(src_f, out_dir / fname)
    save_model(out_dir, model, donate_model=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-model", default=os.path.expanduser("~/science/models/ling-3.0-tiny-bf16"))
    ap.add_argument("--data-dir", default=os.path.expanduser("~/science/data/lingsante_dflash_train"))
    ap.add_argument("--out-dir", default=os.path.expanduser("~/science/models/ling-3.0-tiny-distill-bf16"))
    ap.add_argument("--metrics-out", default=os.path.expanduser("~/science/runs/train_tiny_distill.json"))
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-lr", type=float, default=2e-4)
    ap.add_argument("--min-lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--max-seq-fb-s", type=float, default=4.0)
    a = ap.parse_args()

    maybe_set_recommended_wired_limit()
    src_dir = Path(a.src_model).expanduser()
    data_dir = Path(a.data_dir).expanduser()
    out_dir = Path(a.out_dir).expanduser()

    train_seqs, val_seqs = load_distill_dataset(data_dir)
    model, _ = load_model(src_dir)
    attach_lora_and_freeze(model, r_attn=32, alpha_attn=32.0, r_head=64, alpha_head=64.0)
    mx.eval(model.parameters())

    seq_fb_s = run_startup_tests(src_dir, model, train_seqs[0], val_seqs[0])
    if seq_fb_s > a.max_seq_fb_s:
        print(
            f"SPEED GATE EXIT 3: seq_fb_s={seq_fb_s:.3f}s > {a.max_seq_fb_s:.2f}s threshold "
            f"(~{len(train_seqs) * seq_fb_s * a.epochs / 60:.1f} min for {a.epochs} ep); handing back to planner.",
            flush=True,
        )
        m_out = Path(a.metrics_out).expanduser()
        m_out.parent.mkdir(parents=True, exist_ok=True)
        m_out.write_text(json.dumps({"speed_gate_triggered": True, "seq_fb_s": round(seq_fb_s, 3)}, indent=1))
        return 3

    vr0 = eval_tiny_val(model, val_seqs)
    print(
        f"  Val Ep  0: p1={vr0['p1_all']:.4f} (med={vr0['p1_med']:.4f}) top2/4={vr0['top2_all']:.4f}/{vr0['top4_all']:.4f} | "
        f"E4/6/8/10={vr0['E_acc'][4]:.3f}/{vr0['E_acc'][6]:.3f}/{vr0['E_acc'][8]:.3f}/{vr0['E_acc'][10]:.3f} | "
        f"dyn(5.8/2.0)={vr0['proj']['5p8ms']['dyn_best_tok_s']:.2f}/{vr0['proj']['2p0ms']['dyn_best_tok_s']:.2f} "
        f"alphas={vr0['alphas']}",
        flush=True,
    )

    best_score = (vr0["E_acc"][8], vr0["p1_all"])
    best_state = [(k, mx.array(v)) for k, v in tree_flatten(model.trainable_parameters())]
    mx.eval(*[v for _, v in best_state])

    vg = nn.value_and_grad(model, lambda s: loss_on_sequence(model, s))
    opt = optim.AdamW(learning_rate=a.max_lr, weight_decay=a.weight_decay)
    steps_per_ep = math.ceil(len(train_seqs) / a.grad_accum)
    total_opt_steps = a.epochs * steps_per_ep
    opt_step = 0
    rng = random.Random(42)
    history = [{"epoch": 0, **vr0}]

    for ep in range(1, a.epochs + 1):
        t_ep0 = time.perf_counter()
        model.train()
        order = list(range(len(train_seqs)))
        rng.shuffle(order)
        ep_loss, ep_kl, ep_ce, ep_n = 0.0, 0.0, 0.0, 0
        grad_acc = None
        acc_cnt = 0

        for idx_i, s_idx in enumerate(order):
            (loss, (kl_v, ce_v)), grads = vg(train_seqs[s_idx])
            grad_acc = (
                grads
                if grad_acc is None
                else tree_map(lambda a0, b0: a0 + b0 if (a0 is not None and b0 is not None) else a0, grad_acc, grads)
            )
            acc_cnt += 1
            mx.eval(loss, kl_v, ce_v, grad_acc)
            lv = float(loss.item())
            assert math.isfinite(lv), f"NaN/Inf loss at ep={ep} idx={s_idx}"
            ep_loss += lv
            ep_kl += float(kl_v.item())
            ep_ce += float(ce_v.item())
            ep_n += 1

            if acc_cnt >= a.grad_accum or (idx_i == len(order) - 1 and acc_cnt > 0):
                opt.learning_rate = _cosine_lr(opt_step, total_opt_steps, a.max_lr, a.min_lr, warmup_frac=0.10)
                clipped, _ = _scale_clip_grads(grad_acc, 1.0 / acc_cnt, 1.0)
                opt.update(model, clipped)
                mx.eval(model.trainable_parameters(), opt.state)
                grad_acc = None
                acc_cnt = 0
                opt_step += 1

        vr = eval_tiny_val(model, val_seqs)
        score = (vr["E_acc"][8], vr["p1_all"])
        tag = ""
        if score > best_score:
            best_score = score
            best_state = [(k, mx.array(v)) for k, v in tree_flatten(model.trainable_parameters())]
            mx.eval(*[v for _, v in best_state])
            tag = " [BEST*]"
        print(
            f"  Ep {ep:2d}/{a.epochs} ({time.perf_counter()-t_ep0:.1f}s) L={ep_loss/ep_n:.4f} kl={ep_kl/ep_n:.4f} ce={ep_ce/ep_n:.4f} || "
            f"p1={vr['p1_all']:.4f} (med={vr['p1_med']:.4f}) top2/4={vr['top2_all']:.4f}/{vr['top4_all']:.4f} | "
            f"E4/6/8/10={vr['E_acc'][4]:.3f}/{vr['E_acc'][6]:.3f}/{vr['E_acc'][8]:.3f}/{vr['E_acc'][10]:.3f} | "
            f"dyn(5.8/2.0)={vr['proj']['5p8ms']['dyn_best_tok_s']:.2f}/{vr['proj']['2p0ms']['dyn_best_tok_s']:.2f} "
            f"alphas={vr['alphas']}{tag}",
            flush=True,
        )
        history.append({"epoch": ep, "loss": round(ep_loss / ep_n, 4), "kl": round(ep_kl / ep_n, 4), "ce": round(ep_ce / ep_n, 4), **vr})

    model.update(tree_unflatten(best_state))
    save_merged_model(src_dir, out_dir, model)
    m_out = Path(a.metrics_out).expanduser()
    m_out.parent.mkdir(parents=True, exist_ok=True)
    m_out.write_text(json.dumps({"speed_gate_triggered": False, "seq_fb_s": round(seq_fb_s, 3), "best_E_acc_8": best_score[0], "best_p1": best_score[1], "history": history}, indent=1))
    print(f"Saved merged distilled Ling-3.0-tiny to {out_dir} and metrics to {m_out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
