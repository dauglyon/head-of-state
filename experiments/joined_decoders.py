"""HANDOFF open question 2 on the joined-document streams: decoders compared on the same readings, offline.

    uv run python -I experiments/joined_decoders.py > results/joined_decoders.md

The saved every-token readings of the seed-2 (confirmation) streams, decoded four ways: raw argmax, the
persist rule (3 readings), the sticky forward filter (live, no lookahead) and Viterbi over the whole stream
(settled labels). Accuracy, switches placed within 16 tokens and false switches per 1,000 tokens.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import joined_documents as jd  # noqa: E402


def main(seed=2):
    from head_of_state.decoders import decode_filter, decode_persist, decode_raw, probs
    from head_of_state.metrics import pool, stream_counts
    from head_of_state.settle import settled_labels
    print(f"Seed-{seed} streams, a reading at every token.\n")
    print("| State list | decoder | token acc. | placed within 16 | false /1k |\n|---|---|---|---|---|")
    for name in jd.LISTS:
        L = jd.DECLARED[name]
        recs = [json.loads(l) for l in jd.out_file(name, seed).open()]
        tokens = sum(r["n_tokens"] for r in recs)
        for label, fn in (("raw argmax", lambda p, pos: decode_raw(p)),
                          ("persist 3", lambda p, pos: decode_persist(p, 3)),
                          ("forward filter (live)", lambda p, pos: decode_filter(p, 1.0, L)),
                          ("Viterbi (settled)", lambda p, pos: settled_labels(p, pos, L))):
            c = []
            for r in recs:
                p, pos = probs(r, rotations=1)
                c.append(stream_counts(np.asarray(fn(p, pos)), p, r["gold"], pos))
            m = pool(c, tokens)
            print(f"| {name} | {label} | {m['accuracy']:.3f} | {sum(x['match'][16] for x in c)}/{sum(x['true_b'] for x in c)} | "
                  f"{1000 * sum(x['pred_b'] - x['match'][16] for x in c) / tokens:.1f} |")


if __name__ == "__main__":
    main()
