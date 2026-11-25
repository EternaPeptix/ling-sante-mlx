"""End-to-end evaluation of the converted model with the vendored mlx_lm.

  run_eval.py gen   MODEL            6 greedy generations (2 medical, 2 chemistry, 2 biology)
  run_eval.py ppl   MODEL            perplexity of tools/passage.txt with prefill chunk sizes 64 and 512
  run_eval.py bench MODEL [--long]   decode tok/s (256 greedy tokens), prefill tok/s at 2k and (--long) 32k, peak memory
"""
import sys
import time
from pathlib import Path

_here = Path(__file__).resolve().parent
sys.path[:0] = [str(_here), str(_here.parent / "vendor")]

from mlx_lm import load, stream_generate  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache  # noqa: E402
from mlx_lm.sample_utils import make_sampler  # noqa: E402
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import numpy as np  # noqa: E402

PROMPTS = [
    ("medical", "A 58-year-old man presents with crushing substernal chest pain radiating to the left arm, diaphoresis, and ST elevation in leads II, III and aVF. What is the most likely diagnosis, which coronary artery is probably involved, and what are the first three management steps?"),
    ("medical", "Explain the mechanism of action of metformin, its most important adverse effect, and when it must be withheld."),
    ("chemistry", "Balance the reaction of potassium permanganate with hydrochloric acid that produces chlorine gas, and state the oxidation state changes of manganese and chlorine."),
    ("chemistry", "Why is the SN2 reaction of tert-butyl bromide with sodium methoxide in methanol dominated by elimination rather than substitution? Name the major product."),
    ("biology", "Describe how the lac operon of E. coli responds when lactose is present and glucose is absent, naming the roles of the repressor, cAMP-CAP and allolactose."),
    ("biology", "What is the difference between mitosis and meiosis in terms of chromosome number, crossing over, and the genetic identity of daughter cells?"),
]


def chat_ids(tok, text, thinking=False):
    msgs = [{"role": "user", "content": text}]
    s = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=thinking)
    return tok.encode(s, add_special_tokens=False), s


def cmd_gen(model, tok, max_tokens=300):
    for domain, p in PROMPTS:
        ids, _ = chat_ids(tok, p)
        out, last = [], None
        for r in stream_generate(model, tok, ids, max_tokens=max_tokens, sampler=make_sampler(temp=0.0)):
            out.append(r.text)
            last = r
        print(f"\n=== [{domain}] {p}\n--- ({last.generation_tokens} tokens, {last.generation_tps:.1f} tok/s, prompt {last.prompt_tokens} tok)\n{''.join(out)}", flush=True)


def passage_ids(tok):
    return tok.encode((_here / "passage.txt").read_text(), add_special_tokens=False)


def chunked_nll(model, ids, chunk):
    cache = make_prompt_cache(model)
    ids = mx.array(ids)[None]
    n = ids.shape[1]
    nll, cnt = 0.0, 0
    for s in range(0, n - 1, chunk):
        e = min(s + chunk, n - 1)
        logits = model(ids[:, s:e], cache=cache).astype(mx.float32)
        loss = nn.losses.cross_entropy(logits, ids[:, s + 1 : e + 1], reduction="sum")
        mx.eval(loss)
        nll += loss.item()
        cnt += e - s
    return nll / cnt, cnt


def cmd_ppl(model, tok):
    ids = passage_ids(tok)
    print("passage tokens:", len(ids))
    res = {}
    for chunk in (64, 512):
        nll, cnt = chunked_nll(model, ids, chunk)
        res[chunk] = nll
        print(f"chunk {chunk:4d}: tokens scored {cnt}  mean NLL {nll:.5f}  ppl {np.exp(nll):.4f}")
    print(f"ppl ratio chunk64/chunk512 = {np.exp(res[64]-res[512]):.5f}")


def prefill_speed(model, tok, n, step=2048):
    base = passage_ids(tok)
    ids = (base * (n // len(base) + 1))[:n]
    cache = make_prompt_cache(model)
    x = mx.array(ids)[None]
    mx.reset_peak_memory()
    t = time.time()
    for s in range(0, n, step):
        out = model(x[:, s : s + step], cache=cache)
        mx.eval(out[:, -1], [c.state for c in cache])
    dt = time.time() - t
    print(f"prefill {n:6d} tokens: {dt:.1f}s = {n/dt:.1f} tok/s (step {step}), peak mem {mx.get_peak_memory()/1e9:.1f} GB", flush=True)


def cmd_bench(model, tok, long=False):
    ids, _ = chat_ids(tok, PROMPTS[0][1])
    last = None
    mx.reset_peak_memory()
    for r in stream_generate(model, tok, ids, max_tokens=256, sampler=make_sampler(temp=0.0)):
        last = r
    print(f"decode: {last.generation_tokens} tokens, {last.generation_tps:.2f} tok/s; prompt {last.prompt_tokens} tok at {last.prompt_tps:.1f} tok/s; peak mem {last.peak_memory:.1f} GB", flush=True)
    prefill_speed(model, tok, 2048)
    if long:
        prefill_speed(model, tok, 32768)


if __name__ == "__main__":
    cmd, path = sys.argv[1], sys.argv[2]
    t0 = time.time()
    model, tok = load(path)
    print(f"loaded in {time.time()-t0:.0f}s, active mem {mx.get_active_memory()/1e9:.1f} GB", flush=True)
    {"gen": cmd_gen, "ppl": cmd_ppl, "bench": lambda m, t: cmd_bench(m, t, "--long" in sys.argv)}[cmd](model, tok)
