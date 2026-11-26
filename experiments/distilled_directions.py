"""Follow-ups to Path 3b: directions distilled from asking, on more data and where regimes recur.

    uv run python -I experiments/distilled_directions.py run      # GPU
    uv run python -I experiments/distilled_directions.py report   # CPU: results/distilled_directions.md

Settings carried over unchanged from Path 3 (no new tuning): layer 24 of 32, each token's own
activation, difference-of-means directions, warm-up labels = argmax of the running-prior forward filter
over 1-ordering asked readings, forward filter with the declared segment length.

- PubMed: fit on the 10 dev abstracts with saved asked readings; score on dev abstracts 11-40 (in the
  dev run's order). Asking every 8th token on 11-40 is the reference and the input to fusion.
- WikiSection (regimes recur): 7 new validation streams of about 1,200 tokens (seed 1); asked readings
  every 2nd token on all 7; fit on streams 0-2, score on 3-6.
- S1 lag streams (WikiSection test): directions fitted on the WikiSection warm-up streams; delay by
  previous-segment length against full-history asking (saved in experiments/runs/cde/s1_q.jsonl).
- Decoders on the distilled readings: raw, forward filter (eta 1, 0.5, 0.25), persist 8 / 16 tokens,
  fixed-lag Viterbi (lag 8), and fusion with the sparse asked readings (both as evidence in one filter).
"""
from __future__ import annotations

import json
import random
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compare_readouts  # noqa: E402
import pubmed_rct  # noqa: E402
import s1_lag  # noqa: E402
import wikisection  # noqa: E402
from activation_directions import DiffMeans  # noqa: E402
from common import MODEL, RUNS  # noqa: E402

OUT = RUNS / "distilled"
LAYER = 24


def pubmed_streams():
    s3 = pubmed_rct.load("dev")
    random.Random(0).shuffle(s3)
    return s3[:40]


def ws_streams(tok):
    streams, _, _, _ = compare_readouts.e2(tok, n=7, target_tokens=1200, seed=1)
    for i, s in enumerate(streams):
        s["id"] = f"WS7-{i}"
    return streams


def run():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from head_of_state.harness import UnitReader, run_stream, tokens_and_gold
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
    OUT.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def acts(ids):
        out = model(input_ids=torch.tensor([ids], device="cuda"), output_hidden_states=True, use_cache=False,
                    logits_to_keep=1)
        return out.hidden_states[LAYER][0].float().cpu().numpy().astype(np.float16)

    pm, ws, s1 = pubmed_streams(), ws_streams(tok), s1_lag.build(tok)
    arrays, meta = {}, {}
    for s in pm + ws + s1:
        ids, gold = tokens_and_gold(tok, s["text"], s["spans"])
        arrays[s["id"]] = acts(ids)
        meta[s["id"]] = {"gold": gold, "n_tokens": len(ids), "prev_len": s.get("prev_len")}
    np.savez_compressed(OUT / "acts_layer24.npz", **arrays)
    (OUT / "meta.json").write_text(json.dumps(meta))
    print("activations done", flush=True)

    for name, streams, states, unit, stride, f in (
            ("pubmed 11-40", pm[10:], pubmed_rct.STATES, pubmed_rct.UNIT, 8, "asked_pubmed.jsonl"),
            ("wikisection 7", ws, wikisection.STATES, wikisection.UNIT, 2, "asked_ws.jsonl")):
        reader = UnitReader(model, tok, states, rotations=1, stride=stride, unit=unit)
        t0, n = time.time(), 0
        with (OUT / f).open("w") as fh:
            for s in streams:
                rec = run_stream(reader, s, tok)
                n += rec["n_tokens"]
                fh.write(json.dumps(rec) + "\n")
        print(f"asked {name}: {n} tokens, {1000 * (time.time() - t0) / n:.1f} ms/token at stride {stride}", flush=True)


# ----------------------------------------------------------------------------- offline
def expand(labels_at, positions, n):
    """Per-token labels from readings at 1-based positions: each reading labels the tokens since the
    previous one; tokens after the last reading take its label."""
    out, prev = np.empty(n, dtype=int), 0
    for lab, p in zip(labels_at, positions):
        out[prev:p] = lab
        prev = p
    out[prev:] = labels_at[-1]
    return out


def report():
    from head_of_state.decoders import decode_persist, probs
    from head_of_state.metrics import forward_posteriors, pool, stream_counts
    from head_of_state.reader import FixedLagViterbi, sticky_log_transition
    acts = np.load(OUT / "acts_layer24.npz")
    meta = json.loads((OUT / "meta.json").read_text())
    X = lambda i: acts[i].astype(np.float32)

    def warm_labels(recs, L):
        out = {}
        for r in recs:
            p, pos = probs(r, rotations=1)
            post = forward_posteriors(p, 1.0, L / (pos[1] - pos[0] if len(pos) > 1 else 1), "running")
            out[r["id"]] = expand(post.argmax(1), pos, r["n_tokens"])
        return out

    def fit(ids, labels, K):
        return DiffMeans().fit(np.concatenate([X(i) for i in ids]), np.concatenate([labels[i] for i in ids]), K)

    def viterbi(q, L, lag):
        v = FixedLagViterbi(sticky_log_transition(q.shape[1], L), lag)
        lab = []
        for row in np.log(np.maximum(q, 1e-12)):
            lab += v.push(row)
        return np.asarray(lab + v.flush())

    def fused(q, asked_p, asked_pos, L, eta_cheap):
        """One filter over both: cheap evidence (tempered) at every token, asked evidence where available."""
        T, K = q.shape
        stay = 1.0 - 1.0 / L
        A = np.full((K, K), (1 - stay) / (K - 1)); np.fill_diagonal(A, stay)
        at = {int(p) - 1: a for p, a in zip(asked_pos, asked_p)}
        b, out = np.full(K, 1.0 / K), np.empty((T, K))
        for t in range(T):
            b = (b @ A) * np.power(np.maximum(q[t], 1e-12), eta_cheap)
            if t in at:
                b = b * np.maximum(at[t], 1e-12)
            b /= b.sum()
            out[t] = b
        return out

    def table(title, test_ids, q_by_id, L, asked_recs=None, extra=None):
        print(f"\n## {title}\n")
        print("| Readout | decoder | token acc. | boundary F1 ±8 | ±16 | matched | mean delay | spurious /1k | Brier | ECE |\n"
              "|---|---|---|---|---|---|---|---|---|---|")
        asked_by = {r["id"]: probs(r, rotations=1) for r in asked_recs} if asked_recs else {}
        rows = [("distilled directions", "raw argmax", lambda i: (q_by_id[i], q_by_id[i].argmax(1)))]
        for eta in (1.0, 0.5, 0.25):
            rows.append(("distilled directions", f"forward filter η={eta:g}",
                         lambda i, e=eta: (lambda post: (post, post.argmax(1)))(forward_posteriors(q_by_id[i], e, L))))
        for n in (8, 16):
            rows.append(("distilled directions", f"persist {n} tokens",
                         lambda i, n=n: (q_by_id[i], np.asarray(decode_persist(q_by_id[i], n)))))
        rows.append(("distilled directions", "fixed-lag Viterbi, lag 8 (+8 latency)",
                     lambda i: (q_by_id[i], viterbi(q_by_id[i], L, 8))))
        if asked_by:
            stride = int(asked_recs[0]["stride"])
            for eta in (1.0, 0.25):
                rows.append((f"distilled + asked every {stride}th token, fused", f"forward filter, cheap η={eta:g}",
                             lambda i, e=eta: (lambda post: (post, post.argmax(1)))(
                                 fused(q_by_id[i], asked_by[i][0], asked_by[i][1], L, e))))
            rows.append((f"asking alone, every {stride}th token", "forward filter η=1",
                         lambda i: (lambda p, pos: (lambda post: (post, post.argmax(1), pos))(
                             forward_posteriors(p, 1.0, L / stride)))(*asked_by[i])))
        rows += extra or []
        for rname, dname, fn in rows:
            counts, tokens = [], 0
            for i in test_ids:
                res = fn(i)
                post, lab = res[0], res[1]
                pos = res[2] if len(res) > 2 else np.arange(1, len(lab) + 1)
                counts.append(stream_counts(lab, post, meta[i]["gold"], pos))
                tokens += meta[i]["n_tokens"]
            m = pool(counts, tokens)
            print(f"| {rname} | {dname} | {m['accuracy']:.3f} | {m['boundary_f1@8']:.2f} | {m['boundary_f1@16']:.2f} | "
                  f"{m['matched']} | {m['mean_delay']:.1f} | {m['spurious_per_1k']:.1f} | {m['brier']:.3f} | {m['ece']:.3f} |")

    # PubMed: fit on the 10 abstracts with saved asked readings, score on 11-40
    K = len(pubmed_rct.STATES)
    dev = [json.loads(l) for l in open(RUNS / "s3_dev_r5.jsonl")][:10]
    warm = warm_labels(dev, 30.0)
    dm = fit(list(warm), warm, K)
    asked_pm = [json.loads(l) for l in open(OUT / "asked_pubmed.jsonl")]
    test = [r["id"] for r in asked_pm]
    table(f"PubMed dev abstracts 11–40 ({sum(meta[i]['n_tokens'] for i in test)} tokens); directions fitted on "
          "abstracts 1–10 labelled by asking", test, {i: dm.proba(X(i)) for i in test}, 30.0, asked_pm)

    # WikiSection: fit on streams 0-2 labelled by asking, score on 3-6
    K = len(wikisection.STATES)
    asked_ws = [json.loads(l) for l in open(OUT / "asked_ws.jsonl")]
    warm_ids, test_ids = [r["id"] for r in asked_ws[:3]], [r["id"] for r in asked_ws[3:]]
    labels = warm_labels(asked_ws[:3], 190.5)
    dmw = fit(warm_ids, labels, K)
    seen = sorted(set(np.concatenate([labels[i] for i in warm_ids]).tolist()))
    table(f"WikiSection validation streams 3–6 ({sum(meta[i]['n_tokens'] for i in test_ids)} tokens, regimes recur); "
          f"directions fitted on streams 0–2 labelled by asking (regimes seen in warm-up: {len(seen)} of {K})",
          test_ids, {i: dmw.proba(X(i)) for i in test_ids}, 190.5, asked_ws[3:])

    # S1: WikiSection-fitted directions; delay by previous-segment length against full-history asking
    s1_ids = [i for i in meta if i.startswith("S1lag")]
    s1_q = {r["id"]: r for r in map(json.loads, open(RUNS / "cde" / "s1_q.jsonl"))}
    print("\n## S1 lag streams (12): delay by previous-segment length, scored from 16 tokens before the switch\n")
    print("| Readout | decoder | token acc. | matched | mean delay | spurious /1k | delay at 50 / 200 / 800 / 3000 |\n"
          "|---|---|---|---|---|---|---|")
    for rname, fn in (("distilled directions (fitted on WikiSection validation)", lambda i: forward_posteriors(dmw.proba(X(i)), 1.0, 190.5)),
                      ("asking, full history, every 2nd token", None)):
        per_len, all_counts, all_tok = {}, [], 0
        for i in s1_ids:
            start = s1_q[i]["positions"][0]
            if fn is None:
                p, pos = probs(s1_q[i], rotations=1)
                post = forward_posteriors(p, 1.0, 190.5 / 2)
            else:
                post = fn(i)
                pos = np.arange(1, len(post) + 1)
            keep = pos >= start
            c = stream_counts(post[keep].argmax(1), post[keep], meta[i]["gold"], pos[keep])
            all_counts.append(c); all_tok += meta[i]["n_tokens"] - start + 1
            per_len.setdefault(meta[i]["prev_len"], []).append(c)
        m = pool(all_counts, all_tok)
        by = " / ".join(f"{pool(per_len[L], 1)['mean_delay']:.0f}" for L in s1_lag.PREV)
        print(f"| {rname} | forward filter η=1 | {m['accuracy']:.3f} | {m['matched']} | {m['mean_delay']:.1f} | "
              f"{m['spurious_per_1k']:.1f} | {by} |")


if __name__ == "__main__":
    {"run": run, "report": report}[sys.argv[1]]()
