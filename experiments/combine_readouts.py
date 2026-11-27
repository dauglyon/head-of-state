"""Combining the readouts on the same tokens (screening and tie-break documents), offline.

    uv run python -I experiments/combine_readouts.py          # results/combine_readouts.md

Readouts (files of experiments/runs/wording/):
  jev     the question readout, P(state | recent text) at every token (jev_screen.py)
  reply   the reply slot, ref-instruction on H2O-Lightning
  cache   next-token emissions from one cache per state (per-state-cache, H2O-Lightning)
  think   next-token emissions after a thinking block with the last tokens repeated (think-write-repeat, H2O-Lightning)
The question readout and the reply slot restate the state evidence at every token; consecutive readings are nearly
the same, so counting each as a fresh observation overcounts. "Fractional" counts each reading as 1/tau of a fresh one
with the weight eta = (on - off) / (s2 tau), tau = (1 + phi) / (1 - phi) from the stream's own lag-1 autocorrelation
(tied_target_hmm.fit; no labels). The next-token emissions count each token once. Combinations add the fractional
question readout or reply slot to a next-token form in log space: the readout says which state, the next-token
evidence, concentrated just after switches, where. Every row: Viterbi with the declared length, settled labels.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import wording_screen as ws  # noqa: E402

DECLARED = {"news": 270.0, "articles": 184.0, "pubmed": 30.0}
K = {"news": 4, "articles": 10, "pubmed": 5}
FILES = {"jev": "jev__h2o-lightning-4b", "reply": "ref-instruction__h2o-lightning-4b",
         "cache": "per-state-cache__h2o-lightning-4b", "think": "think-write-repeat__h2o-lightning-4b"}


def load(stem):
    out = {}
    for suffix in ("", "__tiebreak"):
        f = ws.OUT / f"{stem}{suffix}.jsonl"
        if f.exists():
            for r in map(json.loads, f.open()):
                out[(r["set"], r["id"])] = r
    return out


def rows(kind, r, k):
    """[T, K] log scores aligned to tokens 0..T-1."""
    from head_of_state.decoders import probs
    if kind == "jev":
        p, _ = probs(r, rotations=1)
        return np.log(np.maximum(p, 1e-12))
    lp = np.asarray(r["state_logp"])
    return np.vstack([np.zeros((1, k)), lp])


def fractional(lp, L):
    from tied_target_hmm import fit
    p = np.exp(lp - lp.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    _, _, prm = fit(p, L)
    y = np.log(np.maximum(p, 1e-6)); y -= y.mean(1, keepdims=True)
    return prm["eta"] * y


def main():
    from head_of_state.metrics import pool, stream_counts
    from head_of_state.settle import settled_labels
    data = {k: load(v) for k, v in FILES.items()}
    keys = set.intersection(*(set(d) for d in data.values()))
    print(f"Documents with all four readouts: {len(keys)} (screening and tie-break). Cells: token accuracy · switches "
          "placed within 16 tokens · false switches per 1,000 tokens.\n")
    combos = {
        "question readout": lambda R, L: R["jev"],
        "question readout, fractional": lambda R, L: fractional(R["jev"], L),
        "reply slot": lambda R, L: R["reply"],
        "reply slot, fractional": lambda R, L: fractional(R["reply"], L),
        "next-token, per-state caches": lambda R, L: R["cache"],
        "next-token, thinking block": lambda R, L: R["think"],
        "question fractional + per-state caches": lambda R, L: fractional(R["jev"], L) + R["cache"],
        "question fractional + thinking block": lambda R, L: fractional(R["jev"], L) + R["think"],
        "reply fractional + per-state caches": lambda R, L: fractional(R["reply"], L) + R["cache"],
        "reply fractional + thinking block": lambda R, L: fractional(R["reply"], L) + R["think"],
    }
    sets = [s for s in DECLARED if any(k[0] == s for k in keys)]
    print("| Emission | " + " | ".join(sets) + " |\n|---|" + "---|" * len(sets))
    for name, fn in combos.items():
        cells = []
        for s in sets:
            L = DECLARED[s]; cs = []; tokens = 0
            for key in sorted(k for k in keys if k[0] == s):
                R = {kind: rows(kind, data[kind][key], K[s]) for kind in data}
                g = data["cache"][key]["gold"]
                lp = fn(R, L)
                p = np.exp(lp - lp.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
                pos = np.arange(1, len(g) + 1)
                cs.append(stream_counts(settled_labels(p, pos, L), p, g, pos)); tokens += len(g)
            m = pool(cs, tokens)
            cells.append(f"{m['accuracy']:.3f} · {sum(c['match'][16] for c in cs)}/{sum(c['true_b'] for c in cs)} · "
                         f"{1000 * sum(c['pred_b'] - c['match'][16] for c in cs) / tokens:.1f}")
        print(f"| {name} | " + " | ".join(cells) + " |", flush=True)


if __name__ == "__main__":
    main()
