"""Emissions derived from the readings, with a transition table estimated from them (docs/TEST_SUITE.md).

    uv run python -I experiments/estimated_transitions.py > results/estimated_transitions.md

The model's readings are running estimates P(state | text so far), not emissions. Here the transition table
is estimated from the readings alone (no labels: head_of_state.settle.estimate_transition), each token's
evidence is derived as reading_t / (reading_{t-1} A), and forward-backward gives each token's settled
probabilities (head_of_state.settle.smoothed_estimated). Compared with the current settled labels (Viterbi
over the readings with the declared segment length) on sets not used while building it.
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

SETS = [(f"{n} (seed 2)", jd.out_file(n, 2), jd.DECLARED[n]) for n in jd.LISTS] + \
       [("PubMed abstracts 31-60 (held out)", RUNS / "heldout" / "S3.jsonl", 30.0),
        ("single-state streams (seed 1)", RUNS / "heldout" / "S2.jsonl", 190.5)]


def main():
    from head_of_state import settle
    from head_of_state.decoders import probs
    from head_of_state.metrics import pool, stream_counts
    print("A reading at every token. Settled labels; most common state = accuracy of always answering it. Stay at "
          "start = the fixed-point iteration's starting value.\n")
    print("| Set | settled labels | token acc. | always the most common state | placed within 16 | false /1k | "
          "estimated stay (mean) |\n|---|---|---|---|---|---|---|")
    for label, f, L in SETS:
        recs = [json.loads(l) for l in f.open()]
        tokens = sum(r["n_tokens"] for r in recs)
        gold = np.concatenate([np.asarray(r["gold"]) for r in recs])
        base = np.bincount(gold).max() / len(gold)
        arms = [("current: Viterbi, declared length", None)] + [(f"estimated table, start {s:g}", s) for s in (0.9, 0.5, 0.99)]
        for how, start in arms:
            c, stay = [], []
            for r in recs:
                p, pos = probs(r, rotations=1)
                if start is None:
                    c.append(stream_counts(settle.settled_labels(p, pos, L), p, r["gold"], pos))
                    continue
                orig = settle.estimate_transition.__defaults__
                settle.estimate_transition.__defaults__ = (orig[0], start, orig[2])
                try:
                    post, A = settle.smoothed_estimated(p)
                finally:
                    settle.estimate_transition.__defaults__ = orig
                stay.append(np.diag(A).mean())
                c.append(stream_counts(post.argmax(1), post, r["gold"], pos))
            m = pool(c, tokens)
            true = sum(x["true_b"] for x in c)
            placed = f"{sum(x['match'][16] for x in c)}/{true}" if true else "—"
            print(f"| {label} | {how} | {m['accuracy']:.3f} | {base:.3f} | {placed} | "
                  f"{1000 * sum(x['pred_b'] - x['match'][16] for x in c) / tokens:.1f} | "
                  f"{np.mean(stay):.3f} |" if stay else
                  f"| {label} | {how} | {m['accuracy']:.3f} | {base:.3f} | {placed} | "
                  f"{1000 * sum(x['pred_b'] - x['match'][16] for x in c) / tokens:.1f} | — |")


if __name__ == "__main__":
    main()
