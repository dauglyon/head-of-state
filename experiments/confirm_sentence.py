"""Confirmation on unread documents: placing switches with the current-sentence readout, and allowing switches only at
sentence starts.

    H2O_LIGHTNING_PATH=<checkout> uv run python -I experiments/confirm_sentence.py run --set news|pubmed
    uv run python -I experiments/confirm_sentence.py report          # results/confirm_sentence.md

Declared before any of these documents was read (docs/TEST_SUITE.md, 2026-11-27):
  documents  news: 12 streams of about 1,000 tokens built with seed 4 from AG News test articles in none of the seed-0,
             seed-2 and seed-3 streams; PubMed: test abstracts 101-140 of the seed-0 shuffle (1-100 used before).
  readouts   H2O-Lightning-4B: the question readout with the frozen settings ("the last few sentences"; "the last
             sentence" for PubMed), and the current-sentence readout of joined_documents.py (TargetedH2OReader, the
             current sentence as its own part of the record, neutral option names).
  decodes    (1) Viterbi with the declared length (news 270, PubMed 30) on the question readout; (2) sentence-placed: (1)'s
             switches each moved back to where the current-sentence readout first favoured the new state over the old,
             no further than 128 tokens; (3) sentence starts: Viterbi in which a switch is allowed only into a token
             after ". ! ?" or a line break; (4) both: (3) then (2), the move landing on a sentence start.
  comparisons (a) 2 against 1, (b) 3 against 1; secondary 4 against 1. Token accuracy paired by document and by gold
             segment; switches placed within 16 tokens; false switches per 1,000 tokens.
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import joined_documents as jd  # noqa: E402
import news_confirm  # noqa: E402
import pubmed_rct  # noqa: E402
from common import RUNS  # noqa: E402
from scored_runs import FROZEN  # noqa: E402

OUT = RUNS / "confirm_sentence"
DECLARED = {"news": 270.0, "pubmed": 30.0}
BACK = 128


def documents(name, tok):
    if name == "news":
        used = {d for seed in (0, 2) for s in jd.streams_for("ag_news", tok, seed) for d in s["docs"]}
        used |= {d for s in news_confirm.streams(tok) for d in s["docs"]}
        return jd.LISTS["ag_news"], jd.build("ag_news", tok, seed=4, n_streams=12, stream_tokens=1000, exclude=used)[0]
    s = pubmed_rct.load("test")
    random.Random(0).shuffle(s)
    return pubmed_rct.STATES, s[100:140]


def run(name):
    import torch
    from transformers import AutoModelForCausalLM
    from head_of_state.h2o import H2OReader, TargetedH2OReader
    from head_of_state.harness import run_stream
    tok = jd.tokenizer()
    model = AutoModelForCausalLM.from_pretrained(os.environ.get("H2O_LIGHTNING_PATH", "h2oai/h2o-lightning-4b"),
                                                 dtype=torch.bfloat16, device_map="cuda")
    states, docs = documents(name, tok)
    q = H2OReader(model, tok, states, unit=pubmed_rct.UNIT if name == "pubmed" else jd.UNIT, **FROZEN)
    s = TargetedH2OReader(model, tok, states, span_kind="sentence", **FROZEN,
                          question_text="Which option describes the current sentence?",
                          option_names=[f"s{i + 1}" for i in range(len(states))])
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / f"{name}_question.jsonl").open("w") as fq, (OUT / f"{name}_sentence.jsonl").open("w") as fs:
        for d in docs:
            fq.write(json.dumps(dict(run_stream(q, d, tok), set=name)) + "\n"); fq.flush()
            fs.write(json.dumps(dict(run_stream(s, d, tok), set=name)) + "\n"); fs.flush()
        print(f"{name}: {len(docs)} documents", flush=True)


def sentence_starts(tok, ids):
    allow = np.zeros(len(ids), bool); allow[0] = True
    for t in range(1, len(ids)):
        prev = tok.decode([ids[t - 1]]).rstrip(" ")
        allow[t] = prev.endswith((".", "!", "?", '."', '?"', ".'", "\n")) or "\n" in tok.decode([ids[t]])
    return allow


def viterbi_gated(lp, L, allow):
    T, K = lp.shape; stay = np.log(1 - 1 / L); move = np.log(1 / L / (K - 1))
    d = lp[0] - np.log(K); bp = np.zeros((T, K), int)
    for t in range(1, T):
        if allow[t]:
            s = d[:, None] + np.where(np.eye(K, dtype=bool), stay, move); bp[t] = s.argmax(0); d = s.max(0) + lp[t]
        else:
            bp[t] = np.arange(K); d = d + stay + lp[t]
    path = [int(d.argmax())]
    for t in range(T - 1, 0, -1):
        path.append(int(bp[t][path[-1]]))
    return np.array(path[::-1])


def place(lab, q, allow=None, back=BACK):
    new = lab.copy()
    for s in range(1, len(lab)):
        if lab[s] != lab[s - 1]:
            old, nxt = lab[s - 1], lab[s]; t = s
            while t - 1 >= max(0, s - back) and q[t - 1, nxt] > q[t - 1, old] and new[t - 1] == old:
                t -= 1
            if allow is not None:
                while t < s and not allow[t]:
                    t += 1
            new[t:s] = nxt
    return new


def decodes(tok, qrec, srec, doc, L):
    """{1, 2, 3, 4: labels} for one document."""
    from head_of_state.decoders import probs
    from head_of_state.harness import tokens_and_gold
    from head_of_state.settle import settled_labels
    p, pos = probs(qrec, rotations=1); q, _ = probs(srec, rotations=1)
    ids, gold = tokens_and_gold(tok, doc["text"], doc["spans"])
    assert gold == qrec["gold"]
    allow = sentence_starts(tok, ids); lp = np.log(np.maximum(p, 1e-12))
    l1 = settled_labels(p, pos, L); l3 = viterbi_gated(lp, L, allow)
    return {1: l1, 2: place(l1, q), 3: l3, 4: place(l3, q, allow)}, p, pos, gold


def report():
    from head_of_state.metrics import pool, stream_counts
    tok = jd.tokenizer()
    print("Unread documents; decodes and comparisons declared beforehand. Cells: token accuracy · switches placed within "
          "16 tokens · false switches per 1,000 tokens.\n")
    print("| Set | documents | (1) question readout | (2) sentence-placed | (3) sentence starts | (4) both |\n|---|---|---|---|---|---|")
    for name, L in DECLARED.items():
        fq, fs = OUT / f"{name}_question.jsonl", OUT / f"{name}_sentence.jsonl"
        if not (fq.exists() and fs.exists()):
            continue
        Q = {r["id"]: r for r in map(json.loads, fq.open())}
        S = {r["id"]: r for r in map(json.loads, fs.open())}
        docs = {d["id"]: d for d in documents(name, tok)[1]}
        ids = [i for i in Q if i in S]
        if not ids:
            continue
        cs = {k: [] for k in (1, 2, 3, 4)}; tokens = 0
        for i in ids:
            labs, p, pos, gold = decodes(tok, Q[i], S[i], docs[i], L)
            for k in cs:
                cs[k].append(stream_counts(labs[k], p, gold, pos))
            tokens += len(gold)
        cell = lambda c: (f"{pool(c, tokens)['accuracy']:.3f} · {sum(x['match'][16] for x in c)}/{sum(x['true_b'] for x in c)}"
                          f" · {1000 * sum(x['pred_b'] - x['match'][16] for x in c) / tokens:.1f}")
        print(f"| {name} | {len(ids)} | " + " | ".join(cell(cs[k]) for k in (1, 2, 3, 4)) + " |")


if __name__ == "__main__":
    a = sys.argv
    if a[1] == "run":
        run(a[a.index("--set") + 1])
    else:
        report()
