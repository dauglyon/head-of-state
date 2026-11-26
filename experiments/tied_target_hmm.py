"""Readings as observations of an HMM-like model whose emission scale is estimated from the stream itself.

    uv run python -I experiments/tied_target_hmm.py      # results/tied_target_hmm.md

The current settled labels treat each reading's log-probabilities as one independent emission per token.
Consecutive readings are nearly the same vector (lag-1 autocorrelation of the residuals 0.92-0.97), so that
counts the same evidence many times. Hybrid speech recognisers face the same problem with network posteriors
and scale the log-posteriors by a constant chosen on development data; KL-HMMs (Aradilla et al., IDIAP-RR
06-60) instead model the posterior vector itself as the observation, with a target vector per state. This
model is a tied KL-HMM-like model whose scale is estimated, not tuned:
- observation y_t: the reading's centred log-probabilities, log r_t - mean_k log r_t;
- state k's target: `on` in coordinate k and `off` in the others, both shared by all states; Gaussian
  noise with variance s2 per coordinate, inflated by tau = (1 + phi) / (1 - phi), where phi is the lag-1
  autocorrelation of the residuals (n AR(1) observations carry the information of n / tau independent ones);
- with these tied targets the log-emission is eta * y_t(k) + const, eta = (on - off) / (s2 * tau): the
  current decoding with every reading's weight set by the stream's own spread and autocorrelation;
- transitions: one stay probability, the declared 1 - 1/L (variant A) or re-estimated by EM from 0.999 with
  no declared length (variant C);
- on, off, s2, phi (and the stay probability in C) by EM, 30 iterations from the readings as the initial
  state posteriors; labels by posterior decoding (each token's likeliest state under forward-backward).
No setting is fitted to true labels. Variants A and C were fixed on the seed-0 streams before the other sets
were scored (a sub-agent's exploration; this script re-runs them).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import RUNS  # noqa: E402

SETS = {   # name: (file, declared segment length)
    "News topics, seed 0": (RUNS / "joined" / "ag_news.jsonl", 270.0),
    "Disease sections, seed 0": (RUNS / "joined" / "wikisection.jsonl", 441.0),
    "Reviews, seed 0": (RUNS / "joined" / "imdb.jsonl", 274.0),
    "News topics, seed 2": (RUNS / "joined" / "ag_news_seed2.jsonl", 270.0),
    "Disease sections, seed 2": (RUNS / "joined" / "wikisection_seed2.jsonl", 441.0),
    "Reviews, seed 2": (RUNS / "joined" / "imdb_seed2.jsonl", 274.0),
    "PubMed abstracts 31-60": (RUNS / "heldout" / "S3.jsonl", 30.0),
    "Single-state streams": (RUNS / "heldout" / "S2.jsonl", 190.5),
}


def sticky(K, stay):
    A = np.full((K, K), (1 - stay) / (K - 1))
    np.fill_diagonal(A, stay)
    return A


def forward_backward(logb, A):
    """State posteriors and the expected number of stays (sum over t of P(s_t = s_{t+1}))."""
    T, K = logb.shape
    b = np.exp(logb - logb.max(1, keepdims=True))
    al, sc = np.empty((T, K)), np.empty(T)
    a = b[0] / K; sc[0] = a.sum(); al[0] = a / sc[0]
    for t in range(1, T):
        a = b[t] * (al[t - 1] @ A); sc[t] = a.sum(); al[t] = a / sc[t]
    be = np.ones((T, K))
    for t in range(T - 2, -1, -1):
        be[t] = (A @ (b[t + 1] * be[t + 1])) / sc[t + 1]
    g = al * be
    g /= g.sum(1, keepdims=True)
    stays = (al[:-1] * np.diag(A)[None] * b[1:] * be[1:] / sc[1:, None]).sum()
    return g, stays


def lag1(e):
    e = e - e.mean(0)
    return float((e[1:] * e[:-1]).sum() / max((e * e).sum(), 1e-12))


def fit(p, L, em_stay=False, stay0=None, iters=30):
    """Returns (labels, posteriors, fitted parameters)."""
    T, K = p.shape
    lp = np.log(np.maximum(p, 1e-6))
    y = lp - lp.mean(1, keepdims=True)
    stay = stay0 if stay0 is not None else 1 - 1 / L
    g = p / p.sum(1, keepdims=True)
    for _ in range(iters):
        on = float((g * y).sum() / T)
        off = float(((1 - g) * y).sum() / (T * (K - 1)))
        d2 = ((y - off) ** 2).sum(1, keepdims=True) - (y - off) ** 2 + (y - on) ** 2   # |y - target_k|^2
        s2 = float((g * d2).sum() / (T * K))
        phi = min(max(lag1(y - (off + (on - off) * g)), 0.0), 0.995)
        tau = (1 + phi) / (1 - phi)
        eta = (on - off) / (s2 * tau)
        g, stays = forward_backward(eta * y, sticky(K, stay))
        if em_stay:
            stay = min(max(stays / (T - 1), 0.5), 1 - 1e-6)
    return g.argmax(1), g, dict(eta=eta, phi=phi, seg_len=1 / (1 - stay))


def main():
    from head_of_state.decoders import probs
    from head_of_state.metrics import pool, stream_counts
    from head_of_state.settle import settled_labels
    print("Settled labels from the same saved readings (a reading at every token). Cells: token accuracy, switches "
          "placed within 16 tokens, false switches per 1,000 tokens.\n")
    print("| Set | current: Viterbi, declared length | A: estimated scale, declared stay | C: estimated scale and stay | "
          "eta (A, median) | phi | segment length (C, median) / true mean |\n|---|---|---|---|---|---|---|")
    for name, (f, L) in SETS.items():
        recs = [json.loads(l) for l in f.open()]
        tokens = sum(r["n_tokens"] for r in recs)
        counts = {k: [] for k in "0AC"}
        fa, fc, true_len = [], [], []
        for r in recs:
            p, pos = probs(r, rotations=1)
            counts["0"].append(stream_counts(settled_labels(p, pos, L), p, r["gold"], pos))
            lab, q, prm = fit(p, L)
            counts["A"].append(stream_counts(lab, q, r["gold"], pos)); fa.append(prm)
            lab, q, prm = fit(p, L, em_stay=True, stay0=0.999)
            counts["C"].append(stream_counts(lab, q, r["gold"], pos)); fc.append(prm)
            g = np.asarray(r["gold"])
            true_len.append(len(g) / (1 + (np.diff(g) != 0).sum()))

        def cell(cs):
            m = pool(cs, tokens)
            return (f"{m['accuracy']:.3f} · {sum(c['match'][16] for c in cs)}/{sum(c['true_b'] for c in cs)} · "
                    f"{1000 * sum(c['pred_b'] - c['match'][16] for c in cs) / tokens:.1f}")
        med = lambda xs: float(np.median(xs))
        print(f"| {name} | {cell(counts['0'])} | {cell(counts['A'])} | {cell(counts['C'])} | "
              f"{med([x['eta'] for x in fa]):.2f} | {med([x['phi'] for x in fa]):.2f} | "
              f"{med([x['seg_len'] for x in fc]):.0f} / {med(true_len):.0f} |", flush=True)


if __name__ == "__main__":
    main()
