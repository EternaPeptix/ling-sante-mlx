"""HTTP /v1/chat/completions smoke test from 64SM against 512S1:8080 (C=1 vs C=4 deterministic)."""
import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

PROMPTS = [
    "What is the most likely diagnosis for ST elevation in leads II, III, aVF with hypotension and clear lungs? Answer concisely in 2 sentences.",
    "Balance KMnO4 + HCl -> KCl + MnCl2 + Cl2 + H2O and state oxidation state changes concisely.",
    "Briefly explain the lac operon response when lactose is present and glucose is absent.",
    "Why are NSAIDs contraindicated in CKD stage 3b with heart failure taking lisinopril and furosemide? Answer in 2 sentences.",
]


def one_req(url: str, model: str, prompt: str, max_tokens: int):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=180) as resp:
        body = json.loads(resp.read().decode())
    dt = time.perf_counter() - t0
    msg = body["choices"][0]["message"]
    txt = msg.get("content") or msg.get("reasoning") or ""
    ctok = body.get("usage", {}).get("completion_tokens", 0)
    return txt, ctok, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://100.116.244.105:8080")
    ap.add_argument("--model", default="")
    ap.add_argument("--max-tokens", type=int, default=64)
    a = ap.parse_args()

    with urllib.request.urlopen(f"{a.base_url}/v1/models", timeout=15) as r:
        models = json.loads(r.read().decode())
    ids = [m["id"] for m in models["data"]]
    model_id = a.model or next((x for x in ids if "lingsante" in x.lower()), ids[0])
    print("Serving model:", model_id)
    chat_url = f"{a.base_url}/v1/chat/completions"

    # Warmup (triggers lazy model load once)
    w_txt, w_tok, w_dt = one_req(chat_url, model_id, "Say OK.", 8)
    print(f"Warmup: {w_tok} tok in {w_dt:.2f}s | {w_txt[:40]!r}")

    # C=1 sequential
    c1_out = []
    for i, p in enumerate(PROMPTS):
        txt, ctok, dt = one_req(chat_url, model_id, p, a.max_tokens)
        c1_out.append(txt)
        print(f"C=1 req {i}: {ctok} tok in {dt:.2f}s ({ctok/dt:.2f} tok/s) | {txt[:90]!r}")

    # C=4 concurrent
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as ex:
        c4_res = list(ex.map(lambda p: one_req(chat_url, model_id, p, a.max_tokens), PROMPTS))
    wall = time.perf_counter() - t0
    tot_tok = sum(r[1] for r in c4_res)
    exact = all(c4_res[i][0] == c1_out[i] for i in range(4))
    print(f"C=4 concurrent: {tot_tok} tok in {wall:.2f}s ({tot_tok/wall:.2f} agg tok/s) exact_vs_C1={exact}")
    if not exact:
        for i in range(4):
            if c4_res[i][0] != c1_out[i]:
                print(f"  DIFF on {i}:\n    C1: {c1_out[i]}\n    C4: {c4_res[i][0]}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
