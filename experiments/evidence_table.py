"""One table for the evidence-mechanism comparison (docs/EXPERIMENTS.md section 1), from saved readouts.

    uv run python -I experiments/evidence_table.py > results/evidence_table.md

Every mechanism yields per-token regime probabilities on the same PubMed dev abstracts (the first 10
of the dev run, 36 switches; declared segment length 30) and, where run, on the WikiSection validation
streams (regimes recur; 190.5). Each goes through raw argmax and the sticky forward filter; the asked
readings also go through the double-counting variants of Path 1b. Mechanisms whose runs are missing
are skipped.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import RUNS  # noqa: E402
from head_of_state.decoders import decode_persist, probs  # noqa: E402
from head_of_state.metrics import forward_posteriors, pool, stream_counts  # noqa: E402

KS = (8, 16)
COST = {   # ms per token (measured, see docs/LOG.md) and model passes per token
    "ask1": ("103", "1 stream + 1 question (126 tok)"),
    "ask2": ("150", "1 stream + 2 questions, batched"),
    "lik": ("55", "11-token rebuild of 5+1 copies per token (hop 1)"),
}


def load(f):
    return [json.loads(l) for l in open(f)] if Path(f).exists() else None


def asked(recs, rotations):
    out = []
    for r in recs:
        p, pos = probs(r, rotations=rotations)
        out.append((p, pos, r["gold"], r["n_tokens"]))
    return out


def emission(recs):
    out = []
    for r in recs:
        e = np.asarray(r["logp"], dtype=np.float64)
        p = np.exp(e - e.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        out.append((p, np.arange(1, len(p) + 1), r["gold"], r["n_tokens"]))
    return out


def rows_for(name, data, L, cost, variants):
    tokens = sum(d[3] for d in data)
    batch_prior = np.concatenate([d[0] for d in data]).mean(0)   # unlabelled positions of this set
    lines = []
    for vname, kind, kw in variants:
        counts = []
        for p, pos, gold, _ in data:
            if kind == "raw":
                q, lab = p, p.argmax(1)
            elif kind == "persist":
                q, lab = p, np.asarray(decode_persist(p, kw["n"]))
            else:
                prior = {"none": None, "batch": batch_prior, "running": "running"}[kw.get("prior", "none")]
                q = forward_posteriors(p, kw.get("eta", 1.0), L, prior)
                lab = q.argmax(1)
            counts.append(stream_counts(lab, q, gold, pos, KS))
        m = pool(counts, tokens, KS)
        lines.append(f"| {name} | {vname} | {m['accuracy']:.3f} | {m['boundary_f1@8']:.2f} | {m['boundary_f1@16']:.2f} | "
                     f"{m['matched']} | {m['mean_delay']:.1f} | {m['spurious_per_1k']:.1f} | {m['brier']:.3f} | "
                     f"{m['ece']:.3f} | {cost[0]} | {cost[1]} |")
    return lines


BASIC = [("raw argmax", "raw", {}), ("forward filter η=1", "filter", {"eta": 1.0})]
PATH_1B = [("raw argmax", "raw", {}), ("persist 3", "persist", {"n": 3}),
           ("forward filter η=1", "filter", {"eta": 1.0}),
           ("forward filter η=0.5", "filter", {"eta": 0.5}),
           ("forward filter η=0.25", "filter", {"eta": 0.25}),
           ("filter, ÷ set prior (unlabelled), η=1", "filter", {"eta": 1.0, "prior": "batch"}),
           ("filter, ÷ running prior, η=1", "filter", {"eta": 1.0, "prior": "running"})]


def main():
    head = ("| Mechanism | decoder | token acc. | boundary F1 ±8 | ±16 | switches matched | mean delay | "
            "spurious /1k | Brier | ECE | ms/token | passes per token |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
    dev = load(RUNS / "s3_dev_r5.jsonl")
    print("## PubMed 20k RCT dev, 10 abstracts (4,739 tokens, 36 switches, declared length 30)\n")
    print(head)
    if dev:
        dev = dev[:10]
        for line in rows_for("asking, full history, 1 ordering", asked(dev, 1), 30.0, COST["ask1"], PATH_1B):
            print(line)
        for line in rows_for("asking, full history, 2 orderings (prototype default)", asked(dev, 2), 30.0,
                             COST["ask2"], PATH_1B[:3]):
            print(line)
    lik = load(RUNS / "window" / "S3dev_W2.jsonl")
    if lik:
        for line in rows_for("likelihood, hopped window W=2", emission(lik), 30.0, COST["lik"], BASIC):
            print(line)
    for W in (30, 60, 120):
        recs = load(RUNS / "winask" / f"S3dev_W{W}.jsonl")
        if recs:
            ms = 1000 * sum(r["seconds"] for r in recs) / sum(len(r["positions"]) for r in recs)
            for line in rows_for(f"asking, last {W} tokens only ({W // 30}×), every 2nd token", asked(recs, 1), 15.0,
                                 (f"{ms / 2:.0f}", f"1 pass over opening + {W} tokens + question, per 2 tokens"), BASIC):
                print(line)
    acts = RUNS / "activations"
    meta = json.loads((acts / "streams.json").read_text()) if (acts / "streams.json").exists() else None
    for key, label, cost in (("3a", "3a activation directions from model-written passages", None),
                             ("embed", "reference: bge-m3 similarity to descriptions, last 32 tokens", ("1", "1 embedding pass"))):
        recs = load(acts / f"probs_{key}.jsonl")
        if recs:
            if cost is None:
                cost = (f"{meta['ms_per_token_whole_stream_pass']:.2f}", "0 extra (read from the stream pass)")
            data = []
            for r in recs:
                q = np.exp(np.asarray(r["logp"])); q /= q.sum(1, keepdims=True)
                data.append((q, np.arange(1, len(q) + 1), r["gold"], r["n_tokens"]))
            for line in rows_for(label, data, 30.0, cost, BASIC):
                print(line)
    if load(acts / "probs_3b.jsonl"):
        print("\n3b (distilled from asking) and 3c (probe on gold labels) are fitted on abstracts 1–5 and scored on "
              "6–10: see `results/activation_directions.md`.")
    ws = load(RUNS / "compare" / "E2_qafter.jsonl")
    if ws:
        print("\n## WikiSection en_disease validation, 2 streams (2,645 tokens, 15 switches, regimes recur, declared length 190.5)\n")
        print(head)
        data = [(np.exp(np.asarray(r["logp"])), np.arange(1, r["n_tokens"] + 1), r["gold"], r["n_tokens"]) for r in ws]
        data = [(p / p.sum(1, keepdims=True), pos, g, n) for p, pos, g, n in data]
        for line in rows_for("asking, full history, 2 orderings", data, 190.5, ("248", "1 stream + 2 questions, batched"),
                             PATH_1B):
            print(line)


if __name__ == "__main__":
    main()
