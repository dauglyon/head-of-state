"""Scoring any per-token regime evidence the same way: decoder output plus the metrics of
docs/EXPERIMENTS.md (token accuracy, boundary F1 within ±k, mean switch delay, spurious switches per
1k tokens, Brier score, ECE).

A mechanism supplies, per stream, probabilities q_t(k) at reading positions. Decoders turn them into
labels and (for the forward filter) posterior probabilities; Brier and ECE are computed on whatever
probabilities the decoder reports.
"""
from __future__ import annotations

import numpy as np

from .reader import Reading, score, sticky_transition


def forward_posteriors(p: np.ndarray, eta: float, seg_len: float, prior=None, gap=1) -> np.ndarray:
    """Sticky forward filter, b_t ∝ (p_t / prior)^η ⊙ (b_{t-1} A_t). seg_len is the declared segment
    length in tokens; gap is the number of tokens between readings (one number, or one per reading for
    irregular readings) and A_t is the per-token chain raised to that gap (reader.sticky_transition).
    prior: None, a fixed vector (e.g. the mean of p over unlabelled positions), or "running" (the mean
    of p over the stream's earlier positions, uniform at the start)."""
    T, K = p.shape
    gaps = np.broadcast_to(np.asarray(gap, dtype=int), (T,))
    cache = {}
    b, run, out = np.full(K, 1.0 / K), np.full(K, 1.0 / K), np.empty((T, K))
    for t in range(T):
        g = int(gaps[t])
        if g not in cache:
            cache[g] = sticky_transition(K, seg_len, g)
        pt = np.maximum(p[t], 1e-12)
        if prior is None:
            e = pt
        elif isinstance(prior, str):
            e = pt / run
            run = run + (p[t] - run) / (t + 1)
        else:
            e = pt / np.maximum(prior, 1e-12)
        b = (b @ cache[g]) * np.power(e / e.max(), eta)
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


def stream_counts(labels, probs: np.ndarray, gold_all: list[int], positions, ks=(8, 16), bins=10, decided=None,
                  start_token: int = 1) -> dict:
    """Per-stream sums for pooling. gold_all is the gold state of every token (1-based positions).

    decided: the position at which each reading was available (defaults to its position).
    start_token: score only from this 1-based token on (readings before it must already be dropped);
    the first reading then labels the tokens from start_token to its position.
    Each reading labels the tokens since the previous reading, so accuracy, boundaries, Brier and ECE
    are computed per token after expanding the readings over the tokens they label (a predicted
    boundary is the first token of the span whose label differs). Switch matching, delay and spurious
    switches stay on the readings: delay is the time at which the decoder first shows the new state.
    """
    off = start_token - 1
    if off:
        gold_all = gold_all[off:]
        positions = [int(x) - off for x in positions]
        decided = None if decided is None else [int(x) - off for x in decided]
    pos = [int(x) for x in positions]
    lab = [int(x) for x in labels]
    probs = np.asarray(probs)
    span = np.diff([0] + pos)                                  # tokens labelled by each reading
    tok_lab = np.repeat(lab, span)
    tok_p = np.repeat(probs, span, axis=0)
    g_tok = np.asarray(gold_all[: pos[-1]])
    pb = [i for i in range(1, len(tok_lab)) if tok_lab[i] != tok_lab[i - 1]]
    tb = [i for i in range(1, len(g_tok)) if g_tok[i] != g_tok[i - 1]]
    g = [gold_all[x - 1] for x in pos]
    dec = pos if decided is None else [int(x) for x in decided]
    rs = [Reading(x, [], s, s, i > 0 and s != lab[i - 1], decided=d) for i, (x, s, d) in enumerate(zip(pos, lab, dec))]
    m = score(rs, gold_all)
    onehot = np.eye(probs.shape[1])[g_tok]
    conf, hit = tok_p.max(1), (tok_p.argmax(1) == g_tok)
    which = np.minimum((conf * bins).astype(int), bins - 1)
    return {"n": len(g_tok), "ok": int((tok_lab == g_tok).sum()), "pred_b": len(pb), "true_b": len(tb),
            "match": {k: boundary_matches(pb, tb, k) for k in ks},
            "found": m["switches_found"], "true_sw": m["true_switches"], "spurious": m["spurious_switches"],
            "delay_sum": (m["mean_delay_tokens"] or 0) * m["switches_found"],
            "brier": float(((tok_p - onehot) ** 2).sum(1).sum()),
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
