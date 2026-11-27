"""Next-token emissions from the question readout by the FUDGE identity (offline, saved readings).

    uv run python -I experiments/readout_ratio.py        # results/readout_ratio.md

Bayes' rule gives P(x_t | x_<t, k) = P(k | x_<=t) P(x_t | x_<t) / P(k | x_<t) (Yang & Klein 2021, FUDGE, sec. 3).
P(x_t | x_<t) is the same for every state, so between states the designed emission of token t is the ratio of
two consecutive readings of P(state | text so far): e_t(k) proportional to r_t(k) / r_{t-1}(k), with r_0 uniform.
No constant is set. The question readout (Jev-style, joined_documents.py) asks about "the last few sentences",
not the whole text, so its readings are not exactly that posterior; this measures how far the identity carries.
Readings: every token, the joined-document streams (seed 0 and seed 2), PubMed abstracts 31-60 (S3), single-state
streams (S2). Settled labels: Viterbi with the declared length, weight 1, against Viterbi on the readings
themselves (the earlier settled labels).
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

SETS = {f"{n}, seed {s}": (jd.out_file(n, s), jd.DECLARED[n]) for s in (0, 2) for n in ("ag_news", "wikisection")}
SETS["PubMed abstracts 31-60"] = (RUNS / "heldout" / "S3.jsonl", 30.0)
SETS["single-state streams"] = (RUNS / "heldout" / "S2.jsonl", 190.5)


def main():
    from head_of_state.decoders import probs
    from head_of_state.metrics import pool, stream_counts
    from head_of_state.settle import settled_labels
    print("| Set | readings as emissions (earlier settled labels) | ratio of consecutive readings |\n|---|---|---|")
    for name, (f, L) in SETS.items():
        recs = [json.loads(l) for l in f.open()]
        tokens = sum(r["n_tokens"] for r in recs)
        a, b = [], []
        for r in recs:
            p, pos = probs(r, rotations=1)
            lp = np.log(np.maximum(p, 1e-12))
            ratio = lp - np.vstack([np.full((1, p.shape[1]), -np.log(p.shape[1])), lp[:-1]])
            q = np.exp(ratio - ratio.max(1, keepdims=True)); q /= q.sum(1, keepdims=True)
            a.append(stream_counts(settled_labels(p, pos, L), p, r["gold"], pos))
            b.append(stream_counts(settled_labels(q, pos, L), q, r["gold"], pos))
        cell = lambda cs: (f"{pool(cs, tokens)['accuracy']:.3f} · {sum(c['match'][16] for c in cs)}/{sum(c['true_b'] for c in cs)}"
                           f" · {1000 * sum(c['pred_b'] - c['match'][16] for c in cs) / tokens:.1f}")
        print(f"| {name} | {cell(a)} | {cell(b)} |", flush=True)


if __name__ == "__main__":
    main()
