"""Upper bound for restarting the reader at a switch, from the inertia runs (joined_inertia.py).

    uv run python -I experiments/restart_bound.py [--seed N]      # results/restart_bound[_seedN].md

For each switch, the in-stream readings of the tokens re-read by joined_inertia.py (from the switch, or d
tokens after / before it, through 128 tokens after the switch) are replaced by the re-readings; all other
readings stay. Settled labels as joined_documents.py (Viterbi, declared length). The restart points are the
true switches moved by d, so each row is an upper bound for a reader that restarts d tokens off.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import joined_documents as jd  # noqa: E402
from joined_inertia import out_file  # noqa: E402

OFFSETS = (0, 16, 32, -32, -64)


def main(seed):
    from head_of_state.decoders import probs
    from head_of_state.metrics import pool, stream_counts
    from head_of_state.settle import settled_labels
    print(f"Seed {seed}. Cells: token accuracy · switches placed within 16 tokens · false switches per 1,000 tokens.\n")
    print("| State list | in stream | " + " | ".join(f"restart {'at' if d == 0 else (f'{d} after' if d > 0 else f'{-d} before')}"
                                         " the switch" for d in OFFSETS) + " |\n|" + "---|" * (2 + len(OFFSETS)))
    alone = {}
    for d in OFFSETS:
        f = out_file(seed, d)
        if f.exists():
            for l in f.open():
                r = json.loads(l)
                alone.setdefault(d, {}).setdefault(r["id"], []).append(r)
    for name in jd.LISTS:
        recs = [json.loads(l) for l in jd.out_file(name, seed).open()]
        tokens = sum(r["n_tokens"] for r in recs)
        cells = []
        for d in (None,) + OFFSETS:
            if d is not None and d not in alone:
                cells.append("—"); continue
            cs = []
            for r in recs:
                p, pos = probs(r, rotations=1)
                if d is not None:
                    p = p.copy()
                    idx = {int(x): i for i, x in enumerate(pos)}
                    for a in alone[d].get(r["id"], []):
                        pa, posa = probs(a, rotations=1)
                        for k, x in enumerate(posa):
                            p[idx[int(x)]] = pa[k]
                cs.append(stream_counts(settled_labels(p, pos, jd.DECLARED[name]), p, r["gold"], pos))
            m = pool(cs, tokens)
            cells.append(f"{m['accuracy']:.3f} · {sum(c['match'][16] for c in cs)}/{sum(c['true_b'] for c in cs)} · "
                         f"{1000 * sum(c['pred_b'] - c['match'][16] for c in cs) / tokens:.1f}")
        print(f"| {name} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    a = sys.argv
    main(int(a[a.index("--seed") + 1]) if "--seed" in a else 0)
