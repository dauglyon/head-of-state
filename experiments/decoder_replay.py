"""Fixed-lag Viterbi against raw argmax, persist 3 and the forward filter, replayed on saved readings.

    uv run python -I experiments/decoder_replay.py

Placement error: distance from a true switch to where the decoder's final labels put the switch into
the new state (signed: negative = placed early). Decision latency: tokens after the true switch at
which the decoder first emits the new state = placed position + lag (lag 0 for argmax, persist and
the forward filter). A true switch is matched by the nearest decoder switch into the same new state
between the previous and the next true switch (so early placements count); every other decoder
switch is spurious. Short segments: decoded segments under 5 tokens per 1k tokens (the geometric-
duration check).
"""
from __future__ import annotations

import json
import math
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import RUNS  # noqa: E402
from head_of_state.decoders import decode_filter, decode_persist, decode_raw, probs  # noqa: E402
from head_of_state.reader import FixedLagViterbi, sticky_log_transition  # noqa: E402

LAGS = [0, 4, 8, 16, 32, None]
ETAS = [0.1, 0.25, 0.5, 1.0, 2.0]


def viterbi_labels(e: np.ndarray, seg_len: float, lag) -> list[int]:
    v = FixedLagViterbi(sticky_log_transition(e.shape[1], seg_len), lag)
    out = []
    for row in e:
        out += v.push(row)
    return out + v.flush()


def running_prior_scaled(p: np.ndarray) -> np.ndarray:
    """log p_t - log prior_t, prior_t = mean of p_1..p_{t-1} (uniform at t=1), as the reader's prior-power."""
    K = p.shape[1]
    prior, out = np.full(K, 1.0 / K), []
    for t, pt in enumerate(p):
        out.append(np.log(np.maximum(pt, 1e-12)) - np.log(prior))
        prior = prior + (pt - prior) / (t + 1)
    return np.asarray(out)


def score_placement(labels, gold, pos, lag_tokens):
    labels, gold, pos = list(labels), list(gold), list(pos)
    true = [i for i in range(1, len(gold)) if gold[i] != gold[i - 1]]
    dec = [j for j in range(1, len(labels)) if labels[j] != labels[j - 1]]
    used, place = set(), []
    for n, i in enumerate(true):
        lo = true[n - 1] if n else 0
        hi = true[n + 1] if n + 1 < len(true) else len(gold)
        cands = [j for j in dec if lo < j < hi and labels[j] == gold[i] and j not in used]
        if cands:
            j = min(cands, key=lambda j: abs(pos[j] - pos[i]))
            used.add(j)
            place.append(pos[j] - pos[i])
    seg_starts = [0] + dec + [len(labels)]
    short = sum(1 for a, b in zip(seg_starts, seg_starts[1:]) if pos[b - 1] - pos[a] + 1 < 5)
    return {"ok": sum(l == g for l, g in zip(labels, gold)), "n": len(gold), "true": len(true),
            "matched": len(place), "spurious": len(dec) - len(place), "place": place,
            "latency": [x + lag_tokens for x in place], "short": short}


def pooled(results, tokens):
    place = [x for r in results for x in r["place"]]
    lat = [x for r in results for x in r["latency"]]
    return {"acc": sum(r["ok"] for r in results) / sum(r["n"] for r in results),
            "matched": sum(r["matched"] for r in results), "true": sum(r["true"] for r in results),
            "spur_1k": 1000 * sum(r["spurious"] for r in results) / tokens,
            "place_abs": float(np.mean(np.abs(place))) if place else math.nan,
            "place_signed": float(np.mean(place)) if place else math.nan,
            "latency": float(np.mean(lat)) if lat else math.nan,
            "short_1k": 1000 * sum(r["short"] for r in results) / tokens}


def run_set(name, recs, seg_len, kind):
    tokens = sum(r["n_tokens"] for r in recs)
    data = []
    for r in recs:
        p, pos = probs(r, rotations=5 if "shift_logp" in r else 1)
        gold = [r["gold"][x - 1] for x in pos]
        data.append((p, pos, gold))
    rows = []

    def add(dname, param, fn, lag_tokens):
        res = [score_placement(fn(p), g, pos, lag_tokens) for p, pos, g in data]
        rows.append((dname, param, pooled(res, tokens)))

    if kind == "posterior":
        add("raw argmax", "", decode_raw, 0)
        add("persist 3", "", partial(decode_persist, persist=3), 0)
    for eta in ETAS if kind == "posterior" else [1.0]:
        add("forward filter", f"η={eta:g}", partial(decode_filter, eta=eta, seg_len_readings=seg_len), 0)
    for lag in LAGS:
        lt = 0 if lag is None else lag
        for eta in ETAS if kind == "posterior" else [1.0]:
            add(f"Viterbi L={'full' if lag is None else lag}", f"η={eta:g}",
                lambda p, eta=eta, lag=lag: viterbi_labels(eta * np.log(np.maximum(p, 1e-12)), seg_len, lag), lt)
            if kind == "posterior":
                add(f"Viterbi L={'full' if lag is None else lag}", f"η={eta:g} prior-scaled",
                    lambda p, eta=eta, lag=lag: viterbi_labels(eta * running_prior_scaled(p), seg_len, lag), lt)
    print(f"\n## {name}: {len(recs)} streams, {tokens} tokens, declared segment length {seg_len:g}\n")
    print("| Decoder | setting | token acc. | matched | spurious/1k | placement |err| (signed) | "
          "decision latency | segments <5 tokens /1k |\n|---|---|---|---|---|---|---|---|")
    for dname, param, m in rows:
        print(f"| {dname} | {param} | {m['acc']:.3f} | {m['matched']}/{m['true']} | {m['spur_1k']:.1f} | "
              f"{m['place_abs']:.1f} ({m['place_signed']:+.1f}) | {m['latency']:.1f} | {m['short_1k']:.1f} |")
    return rows


def main():
    load = lambda f: [json.loads(l) for l in open(f)]
    sets = [("PubMed dev, question readings (5 orderings averaged)", load(RUNS / "s3_dev_r5.jsonl")[:10],
             30.0, "posterior")]
    demo = RUNS / "demo_readings.json"
    if demo.exists():
        sets.append(("Demo stream, question readings (2 orderings averaged; declared length 50, set by hand)",
                     [json.load(open(demo))], 50.0, "posterior"))
    ws = RUNS / "compare" / "E2_qafter.jsonl"
    if ws.exists():
        sets.append(("WikiSection validation streams (states recur in random order), question readings "
                     "(2 orderings averaged)", load(ws), 190.5, "posterior"))
    lik = RUNS / "window" / "S3dev_W2.jsonl"
    if lik.exists():
        sets.append(("PubMed dev, windowed likelihood emissions W=2", load(lik), 30.0, "emission"))
    for name, recs, L, kind in sets:
        run_set(name, recs, L, kind)


if __name__ == "__main__":
    main()
