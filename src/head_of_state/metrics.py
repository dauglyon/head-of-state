"""Scoring any per-token regime evidence the same way: decoder output plus the metrics of
docs/EXPERIMENTS.md (token accuracy, boundary F1 within ±k, mean switch delay, spurious switches per
1k tokens, Brier score, ECE).

A mechanism supplies, per stream, probabilities q_t(k) at reading positions. Decoders turn them into
labels and (for the forward filter) posterior probabilities; Brier and ECE are computed on whatever
probabilities the decoder reports.
"""
from __future__ import annotations

import numpy as np

from .reader import Reading, score


def forward_posteriors(p: np.ndarray, eta: float, seg_len: float, prior=None) -> np.ndarray:
    """Sticky forward filter, b_t ∝ (p_t / prior)^η ⊙ (b_{t-1} A); A_kk = 1 - 1/L.
    prior: None, a fixed vector (e.g. the mean of p over unlabelled positions), or "running" (the mean
    of p over the stream's earlier positions, uniform at the start)."""
    T, K = p.shape
    stay = 1.0 - 1.0 / max(seg_len, 1.0 + 1e-9)
    A = np.full((K, K), (1.0 - stay) / (K - 1))
    np.fill_diagonal(A, stay)
    b, run, out = np.full(K, 1.0 / K), np.full(K, 1.0 / K), np.empty((T, K))
    for t in range(T):
        pt = np.maximum(p[t], 1e-12)
        if prior is None:
            e = pt
        elif isinstance(prior, str):
            e = pt / run
            run = run + (p[t] - run) / (t + 1)
        else:
            e = pt / np.maximum(prior, 1e-12)
        b = (b @ A) * np.power(e / e.max(), eta)
        b /= b.sum()
        out[t] = b
    return out


def boundary_matches(pred: list[int], true: list[int], k: int) -> int:
    """One-to-one matches between predicted and true boundary positions within ±k (nearest first)."""
    pairs = sorted((abs(a - b), i, j) for i, a in enumerate(pred) for j, b in enumerate(true) if abs(a - b) <= k)
    used_p, used_t, n = set(), set(), 0
    for _, i, j in pairs:
        if i not in used_p and j not in used_t:
            used_p.add(i); used_t.add(j); n += 1
    return n


def stream_counts(labels, probs: np.ndarray, gold_all: list[int], positions, ks=(8, 16), bins=10) -> dict:
    """Per-stream sums for pooling. gold_all is the gold regime of every token (1-based positions)."""
    pos = list(positions)
    g = [gold_all[x - 1] for x in pos]
    lab = list(labels)
    pb = [pos[i] for i in range(1, len(lab)) if lab[i] != lab[i - 1]]
    tb = [pos[i] for i in range(1, len(g)) if g[i] != g[i - 1]]
    rs = [Reading(x, [], s, s, i > 0 and s != lab[i - 1]) for i, (x, s) in enumerate(zip(pos, lab))]
    m = score(rs, gold_all)
    onehot = np.eye(probs.shape[1])[g]
    conf, hit = probs.max(1), (probs.argmax(1) == np.asarray(g))
    which = np.minimum((conf * bins).astype(int), bins - 1)
    return {"n": len(g), "ok": sum(a == b for a, b in zip(lab, g)), "pred_b": len(pb), "true_b": len(tb),
            "match": {k: boundary_matches(pb, tb, k) for k in ks},
            "found": m["switches_found"], "true_sw": m["true_switches"], "spurious": m["spurious_switches"],
            "delay_sum": (m["mean_delay_tokens"] or 0) * m["switches_found"],
            "brier": float(((probs - onehot) ** 2).sum(1).sum()),
            "bin_n": np.bincount(which, minlength=bins), "bin_conf": np.bincount(which, conf, bins),
            "bin_hit": np.bincount(which, hit.astype(float), bins)}


def pool(counts: list[dict], tokens: int, ks=(8, 16)) -> dict:
    s = lambda key: sum(c[key] for c in counts)
    n = s("n")
    out = {"accuracy": s("ok") / n, "matched": f"{s('found')}/{s('true_sw')}",
           "mean_delay": s("delay_sum") / s("found") if s("found") else float("nan"),
           "spurious_per_1k": 1000 * s("spurious") / tokens, "brier": s("brier") / n}
    for k in ks:
        tp = sum(c["match"][k] for c in counts)
        prec, rec = tp / max(s("pred_b"), 1), tp / max(s("true_b"), 1)
        out[f"boundary_f1@{k}"] = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    bn, bc, bh = (sum(c[key] for c in counts) for key in ("bin_n", "bin_conf", "bin_hit"))
    out["ece"] = float(sum(abs(bh[i] - bc[i]) for i in range(len(bn))) / n)
    return out
