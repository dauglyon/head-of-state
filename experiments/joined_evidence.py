"""Settled labels from evidence recovered from the readings (docs/TEST_SUITE.md), offline.

    uv run python -I experiments/joined_evidence.py > results/joined_evidence.md

A reading is the model's answer given all the text so far: a running estimate of the state, which is what an
HMM's forward pass produces, not the evidence of the newest tokens. Decoding the readings as if they were
evidence counts the history again at every token. Here the evidence is recovered from consecutive readings
(head_of_state.settle.reading_evidence, with the readings' own switch rate 1/20 per token, fixed on the
seed-0 streams) and decoded by Viterbi with the declared segment length. Sets not used to choose it: the
seed-2 joined-document streams, the held-out PubMed abstracts 31-60 and the seed-1 null streams.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import joined_documents as jd  # noqa: E402
from common import RUNS  # noqa: E402



def main():
    from head_of_state.decoders import probs
    from head_of_state.metrics import pool, stream_counts
    from head_of_state.settle import READING_RATE_LEN, reading_evidence, settled_labels
    sets = [(f"{n} (seed 2)", jd.out_file(n, 2), jd.DECLARED[n]) for n in jd.LISTS] + \
           [("PubMed abstracts 31-60 (held out)", RUNS / "heldout" / "S3.jsonl", 30.0),
            ("single-state streams (seed 1)", RUNS / "heldout" / "S2.jsonl", 190.5)]
    print("A reading at every token. Settled labels; most common state = accuracy of always answering it.\n")
    print("| Set | decoded | token acc. | always the most common state | placed within 16 | false /1k |\n|---|---|---|---|---|---|")
    for label, f, L in sets:
        recs = [json.loads(l) for l in f.open()]
        tokens = sum(r["n_tokens"] for r in recs)
        gold = np.concatenate([np.asarray(r["gold"]) for r in recs])
        base = np.bincount(gold[gold >= 0]).max() / (gold >= 0).sum()
        for how in ("readings (current)", "recovered evidence"):
            c = []
            for r in recs:
                p, pos = probs(r, rotations=1)
                q = p if how.startswith("readings") else reading_evidence(p, pos, READING_RATE_LEN)
                c.append(stream_counts(settled_labels(q, pos, L), p, r["gold"], pos))
            m = pool(c, tokens)
            true = sum(x["true_b"] for x in c)
            placed = f"{sum(x['match'][16] for x in c)}/{true}" if true else "—"
            print(f"| {label} | {how} | {m['accuracy']:.3f} | {base:.3f} | {placed} | "
                  f"{1000 * sum(x['pred_b'] - x['match'][16] for x in c) / tokens:.1f} |")


if __name__ == "__main__":
    main()
