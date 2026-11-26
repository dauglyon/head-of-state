"""Reading weight eta on the joined-document confirmation streams (docs/TEST_SUITE.md), offline.

    uv run python -I experiments/joined_eta.py > results/joined_eta.md

Each reading's probabilities are raised to eta and renormalised before Viterbi (settled labels) and the
online decision (time to notice). Readings every token are strongly correlated, and the decoder treats each
as independent evidence; eta < 1 discounts them. eta = 1/8 was chosen on the seed-0 streams
(docs/LOG.md); this scores the saved seed-2 readings with eta = 1 and 1/8.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import joined_documents as jd  # noqa: E402

ETAS = (1.0, 0.125)


def temper(p, eta):
    lp = eta * np.log(np.maximum(p, 1e-12))
    q = np.exp(lp - lp.max(1, keepdims=True))
    return q / q.sum(1, keepdims=True)


def main(seed=2):
    from head_of_state.decoders import probs
    from head_of_state.metrics import pool, stream_counts
    from head_of_state.settle import online_labels, settled_labels
    print(f"Seed-{seed} streams, every token. Accuracy, placed and false switches from settled labels; time to notice "
          "from the online decision, averaged over the switches it noticed.\n")
    print("| State list | eta | token acc. | always the most common state | placed within 16 | false /1k | "
          "switches noticed | time to notice |\n|---|---|---|---|---|---|---|---|")
    for name in jd.LISTS:
        recs = [json.loads(l) for l in jd.out_file(name, seed).open()]
        tokens = sum(r["n_tokens"] for r in recs)
        gold = np.concatenate([r["gold"] for r in recs])
        base = np.bincount(gold).max() / len(gold)
        for eta in ETAS:
            st, on = [], []
            for r in recs:
                p, pos = probs(r, rotations=1)
                q = temper(p, eta)
                st.append(stream_counts(settled_labels(q, pos, jd.DECLARED[name]), q, r["gold"], pos))
                on.append(stream_counts(online_labels(q, pos, jd.DECLARED[name]), q, r["gold"], pos))
            m, o = pool(st, tokens), pool(on, tokens)
            print(f"| {name} | {eta:g} | {m['accuracy']:.3f} | {base:.3f} | {sum(c['match'][16] for c in st)}/"
                  f"{sum(c['true_b'] for c in st)} | {1000 * sum(c['pred_b'] - c['match'][16] for c in st) / tokens:.1f} | "
                  f"{o['matched']} | {o['mean_delay']:.1f} |")


if __name__ == "__main__":
    main()
