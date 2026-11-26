"""Path 3: activation directions as the cheap per-token layer, and the asking cascade (docs/EXPERIMENTS.md).

    uv run python -I experiments/activation_directions.py run      # GPU: snippets, activations, embedding reference
    uv run python -I experiments/activation_directions.py report   # CPU: results/activation_directions.md

Set: the 10 PubMed dev abstracts of the evidence table (declared segment length 30).
- 3a  model-written examples: the model writes 20 short passages per regime from the regime's name
      and description only; per-regime mean activation; difference-of-means directions; scores
      standardised by their spread over the passages' tokens. Layers 16, 20, 24, 28 of 32 (24 = 75%).
- 3b  distilled from asking: directions from the abstracts 1-5 activations labelled by the asking
      readout (full history, 1 ordering, running-prior forward filter: Path 1's most accurate
      variant); evaluated on abstracts 6-10.
- 3c  upper bound: multinomial logistic probe on gold labels of abstracts 1-5; evaluated on 6-10.
- 3d  cascade: the forward filter takes the cheap reading at every token and the asked reading
      instead wherever the filter's posterior entropy after the cheap update exceeds a threshold.
Activations are the residual stream of each token from one pass over the stream alone (no prompt).
Readings use the token's own activation and, as a variant, the mean over the last 16 tokens.
"""
from __future__ import annotations

import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pubmed_rct  # noqa: E402
from common import MODEL, RUNS  # noqa: E402

OUT = RUNS / "activations"
LAYERS = [16, 20, 24, 28]
N_SNIPPETS = 20
SNIPPET_PROMPT = ("Write one short passage of two or three sentences whose content is: {name}: {desc}. "
                  "Output only the passage.")


def stream_set(tok):
    s3 = pubmed_rct.load("dev")
    random.Random(0).shuffle(s3)
    return s3[:10]


def run():
    import torch
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
    from head_of_state.harness import tokens_and_gold
    from head_of_state.readouts import run_embed
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    # 3a: model-written passages per regime
    snippets = {}
    for name, desc in pubmed_rct.STATES.items():
        msg = [{"role": "user", "content": SNIPPET_PROMPT.format(name=name, desc=desc)}]
        text = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        x = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
        gen = model.generate(**x, do_sample=True, temperature=0.9, top_p=0.95, max_new_tokens=90,
                             num_return_sequences=N_SNIPPETS, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        snippets[name] = [tok.decode(g[x["input_ids"].shape[1]:], skip_special_tokens=True).strip() for g in gen]
    (OUT / "snippets.json").write_text(json.dumps(snippets, indent=1))

    @torch.no_grad()
    def acts(ids):
        out = model(input_ids=torch.tensor([ids], device="cuda"), output_hidden_states=True, use_cache=False,
                    logits_to_keep=1)
        return np.stack([out.hidden_states[l][0].float().cpu().numpy().astype(np.float16) for l in LAYERS])

    snip_acts = {name: [acts(tok(s, add_special_tokens=False)["input_ids"]) for s in texts if s]
                 for name, texts in snippets.items()}
    np.savez_compressed(OUT / "snippet_acts.npz",
                        **{f"{name}__{i}": a for name, lst in snip_acts.items() for i, a in enumerate(lst)})

    streams, tsec, ttok = stream_set(tok), 0.0, 0
    meta = []
    arrays = {}
    for s in streams:
        ids, gold = tokens_and_gold(tok, s["text"], s["spans"])
        torch.cuda.synchronize(); t0 = time.time()
        a = acts(ids)
        torch.cuda.synchronize(); tsec += time.time() - t0; ttok += len(ids)
        arrays[s["id"]] = a
        meta.append({"id": s["id"], "gold": gold, "n_tokens": len(ids)})
    np.savez_compressed(OUT / "stream_acts.npz", **arrays)
    (OUT / "streams.json").write_text(json.dumps({"meta": meta, "ms_per_token_whole_stream_pass": 1000 * tsec / ttok}))
    print(f"activations: {ttok} tokens, {1000 * tsec / ttok:.2f} ms/token (one pass per whole abstract)", flush=True)

    # reference: sentence-embedding similarity (bge-m3, last 32 tokens) on the same abstracts
    emb_tok = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    emb = AutoModel.from_pretrained("BAAI/bge-m3", dtype=torch.float16).to("cuda").eval()
    with (OUT / "embed_ref.jsonl").open("w") as f:
        for s in streams:
            rec = run_embed(emb, emb_tok, tok, pubmed_rct.STATES, s, window=32)
            rec["id"] = s["id"]
            f.write(json.dumps(rec) + "\n")
    print("embedding reference done", flush=True)


# ----------------------------------------------------------------------------- offline part
def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def smooth(h, n):
    """Mean of the last n token vectors (causal)."""
    c = np.cumsum(h, 0)
    out = c.copy()
    out[n:] = c[n:] - c[:-n]
    counts = np.minimum(np.arange(1, len(h) + 1), n)[:, None]
    return out / counts


class DiffMeans:
    """Difference-of-means directions: centre c = mean of the class means; direction k = mean_k - c.
    Scores are standardised by their spread over the fitting tokens, then softmaxed."""

    def fit(self, X, y, K):
        means = np.stack([X[y == k].mean(0) for k in range(K)])
        self.c = means.mean(0)
        self.D = means - self.c
        s = (X - self.c) @ self.D.T
        self.scale = s.std() + 1e-6
        return self

    def proba(self, X):
        return softmax(((X - self.c) @ self.D.T) / self.scale)


class Probe:
    """Multinomial logistic regression on standardised features (L2), fitted with L-BFGS."""

    def fit(self, X, y, K, l2=1e-2):
        import torch
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Xt = torch.tensor((X - self.mu) / self.sd, dtype=torch.float32)
        yt = torch.tensor(y)
        W = torch.zeros(X.shape[1], K, requires_grad=True)
        b = torch.zeros(K, requires_grad=True)
        opt = torch.optim.LBFGS([W, b], max_iter=200, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(Xt @ W + b, yt) + l2 * (W ** 2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        self.W, self.b = W.detach().numpy(), b.detach().numpy()
        return self

    def proba(self, X):
        return softmax(((X - self.mu) / self.sd) @ self.W + self.b)


def cascade(cheap: np.ndarray, asked: np.ndarray, seg_len: float, theta: float):
    """Forward filter on cheap readings; where the posterior entropy after the cheap update exceeds
    theta, the asked reading is used for that token instead. Returns posteriors and #asked."""
    T, K = cheap.shape
    stay = 1.0 - 1.0 / seg_len
    A = np.full((K, K), (1 - stay) / (K - 1)); np.fill_diagonal(A, stay)
    b, out, n_ask = np.full(K, 1.0 / K), np.empty((T, K)), 0
    for t in range(T):
        pred = b @ A
        c = pred * np.maximum(cheap[t], 1e-12); c /= c.sum()
        if -(c * np.log(np.maximum(c, 1e-12))).sum() > theta:
            c = pred * np.maximum(asked[t], 1e-12); c /= c.sum()
            n_ask += 1
        b = c
        out[t] = b
    return out, n_ask


def report():
    from head_of_state.decoders import probs
    from head_of_state.metrics import forward_posteriors, pool, stream_counts
    K = len(pubmed_rct.STATES)
    names = list(pubmed_rct.STATES)
    meta = json.loads((OUT / "streams.json").read_text())
    ms_cheap = meta["ms_per_token_whole_stream_pass"]
    acts = np.load(OUT / "stream_acts.npz")
    snip = np.load(OUT / "snippet_acts.npz")
    streams = meta["meta"]
    dev = {r["id"]: r for r in map(json.loads, open(RUNS / "s3_dev_r5.jsonl"))}
    asked = {s["id"]: probs(dev[s["id"]], rotations=1)[0] for s in streams}
    warm, test = streams[:5], streams[5:]

    def score_rows(label, q_by_id, subset, cost, rows):
        for dname in ("raw argmax", "forward filter η=1"):
            counts, tokens = [], 0
            for s in subset:
                q = q_by_id[s["id"]]
                post = q if dname == "raw argmax" else forward_posteriors(q, 1.0, 30.0)
                counts.append(stream_counts(post.argmax(1), post, s["gold"], np.arange(1, len(q) + 1)))
                tokens += s["n_tokens"]
            m = pool(counts, tokens)
            rows.append(f"| {label} | {dname} | {m['accuracy']:.3f} | {m['boundary_f1@8']:.2f} | {m['boundary_f1@16']:.2f} | "
                        f"{m['matched']} | {m['mean_delay']:.1f} | {m['spurious_per_1k']:.1f} | {m['brier']:.3f} | "
                        f"{m['ece']:.3f} | {cost} |")
        return rows

    head = ("| Mechanism | decoder | token acc. | boundary F1 ±8 | ±16 | matched | mean delay | spurious /1k | "
            "Brier | ECE | cost per token |\n|---|---|---|---|---|---|---|---|---|---|---|")
    cheap_cost = f"{ms_cheap:.2f} ms (whole-abstract pass); 0 extra passes when the stream is fed for asking"
    saved = {}

    # 3a: layer and smoothing comparison on dev
    print("## 3a. Directions from model-written passages: layer and smoothing (PubMed dev, all 10 abstracts)\n")
    print(head)
    best = None
    for li, layer in enumerate(LAYERS):
        X = np.concatenate([snip[k][li].astype(np.float32) for k in snip.files])
        y = np.concatenate([[names.index(k.split("__")[0])] * snip[k].shape[1] for k in snip.files])
        dm = DiffMeans().fit(X, y, K)
        for n in (1, 16):
            q = {s["id"]: dm.proba(smooth(acts[s["id"]][li].astype(np.float32), n)) for s in streams}
            rows = score_rows(f"3a, layer {layer}, {'own token' if n == 1 else 'mean of last 16 tokens'}", q, streams,
                              cheap_cost, [])
            print("\n".join(rows))
            acc = float(rows[1].split("|")[3])
            if best is None or acc > best[0]:
                best = (acc, li, n, q)
    _, li_best, n_best, q3a = best
    saved["3a"] = q3a
    layer = LAYERS[li_best]
    print(f"\nChosen on dev for the rows below: layer {layer}, "
          f"{'own token' if n_best == 1 else 'mean of last 16 tokens'} (best filtered token accuracy).\n")

    # 3b / 3c on held-out abstracts 6-10, with 3a and asking on the same abstracts for reference
    Xw = np.concatenate([smooth(acts[s["id"]][li_best].astype(np.float32), n_best) for s in warm])
    asked_labels = np.concatenate([forward_posteriors(asked[s["id"]], 1.0, 30.0, "running").argmax(1) for s in warm])
    gold_w = np.concatenate([s["gold"] for s in warm])
    q3b_model = DiffMeans().fit(Xw, asked_labels, K)
    q3c_model = Probe().fit(Xw, gold_w, K)
    feats = {s["id"]: smooth(acts[s["id"]][li_best].astype(np.float32), n_best) for s in test}
    q3b = {i: q3b_model.proba(x) for i, x in feats.items()}
    q3c = {i: q3c_model.proba(x) for i, x in feats.items()}
    emb = {r["id"]: softmax(np.asarray(r["logp"])) for r in map(json.loads, open(OUT / "embed_ref.jsonl"))}
    print("## 3a–3c on held-out abstracts 6–10 (abstracts 1–5 used for fitting 3b and 3c)\n")
    print(head)
    rows = []
    score_rows("asking, full history, 1 ordering", asked, test, "103 ms; 1 stream + 1 question pass", rows)
    score_rows(f"3a model-written directions (layer {layer})", q3a, test, cheap_cost, rows)
    score_rows(f"3b directions distilled from asking (layer {layer})", q3b, test, cheap_cost, rows)
    score_rows(f"3c probe on gold labels, upper bound (layer {layer})", q3c, test, cheap_cost, rows)
    score_rows("reference: bge-m3 similarity, last 32 tokens", emb, test, "about 1 ms; 1 embedding pass", rows)
    print("\n".join(rows))

    # 3d cascade
    print("\n## 3d. Cascade: cheap reading everywhere, asking where the filter's entropy exceeds a threshold\n")
    print("| Cheap reading | set | threshold (nats) | fraction asked | token acc. | matched | mean delay | "
          "spurious /1k | est. ms/token |\n|---|---|---|---|---|---|---|---|---|")
    curves = {}
    for label, q, subset in (("3a", q3a, streams), ("3b", q3b, test)):
        pts = []
        for theta in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.3, math.log(K) + 1]:
            counts, tokens, n_ask = [], 0, 0
            for s in subset:
                post, na = cascade(q[s["id"]], asked[s["id"]], 30.0, theta)
                n_ask += na
                counts.append(stream_counts(post.argmax(1), post, s["gold"], np.arange(1, len(post) + 1)))
                tokens += s["n_tokens"]
            m = pool(counts, tokens)
            frac = n_ask / tokens
            pts.append((frac, m["accuracy"]))
            print(f"| {label} | {'1–10' if subset is streams else '6–10'} | {theta:.2f} | {frac:.2f} | {m['accuracy']:.3f} | "
                  f"{m['matched']} | {m['mean_delay']:.1f} | {m['spurious_per_1k']:.1f} | {ms_cheap + frac * 103:.0f} |")
        curves[label] = pts
    (OUT / "cascade_curves.json").write_text(json.dumps(curves))
    for key, q in (("3a", q3a), ("3b", q3b), ("3c", q3c), ("embed", emb)):
        with (OUT / f"probs_{key}.jsonl").open("w") as f:
            for i, p in q.items():
                s = next(x for x in streams if x["id"] == i)
                f.write(json.dumps({"id": i, "gold": s["gold"], "n_tokens": s["n_tokens"],
                                    "logp": np.log(np.maximum(p, 1e-12)).tolist()}) + "\n")


if __name__ == "__main__":
    {"run": run, "report": report}[sys.argv[1]]()
