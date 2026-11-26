"""Autoregressive HMM over the readings (docs/TEST_SUITE.md), offline on saved every-token readings.

    uv run python -I experiments/ar_hmm.py > results/ar_hmm.md

Observation: the reading's centred log-ratios, o_t(j) = log r_t(j) - mean_i log r_t(i) (Aitchison's
coordinates for probability vectors). Consecutive readings carry over, so the emission is conditioned on the
previous observation as in a Markov-switching autoregression (Hamilton 1989), with a switching intercept:
    o_t = c_k + phi * o_{t-1} + noise,  noise ~ N(0, diag(var))
when the state at t is k. phi and var are per coordinate and shared by the states; c_k is the state's own.
EM over each stream (no labels) learns c, phi, var and the transition table A. Settled labels: the
likeliest state per token from forward-backward. Starting values: c_k = (1 - phi) times the log-ratios of a
reading of 0.9 on option k (so state k keeps option k's name), A from the declared segment length.
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

ITERS = 50


def clr(p):
    lp = np.log(np.maximum(p, 1e-6))
    return lp - lp.mean(1, keepdims=True)


def forward_backward(logb, A):
    T, K = logb.shape
    b = np.exp(logb - logb.max(1, keepdims=True))
    alpha, scale = np.empty((T, K)), np.empty(T)
    a = b[0] / K
    scale[0] = a.sum(); alpha[0] = a / scale[0]
    for t in range(1, T):
        a = b[t] * (alpha[t - 1] @ A)
        scale[t] = a.sum(); alpha[t] = a / scale[t]
    beta = np.ones((T, K))
    for t in range(T - 2, -1, -1):
        x = A @ (b[t + 1] * beta[t + 1])
        beta[t] = x / scale[t + 1]
    gamma = alpha * beta
    gamma /= gamma.sum(1, keepdims=True)
    xi = np.zeros((K, K))
    for t in range(1, T):
        x = alpha[t - 1][:, None] * A * (b[t] * beta[t])[None, :]
        xi += x / x.sum()
    return gamma, xi


def fit(p, seg_len, iters=ITERS):
    o = clr(p)
    y, x = o[1:], o[:-1]                     # emission of o_t given o_{t-1}; the first reading is the start
    T, K = y.shape
    phi = np.full(K, 0.9)
    anchor = clr(np.where(np.eye(K, dtype=bool), 0.9, 0.1 / (K - 1)))
    c = (1 - phi) * anchor                   # row k: state k's intercept
    var = np.maximum(((y - x * phi) - (y - x * phi).mean(0)).var(0), 1e-4)
    stay = 1 - 1 / seg_len
    A = np.full((K, K), (1 - stay) / (K - 1)); np.fill_diagonal(A, stay)
    for _ in range(iters):
        mu = c[None, :, :] + (x * phi)[:, None, :]                     # [T, state, coord]
        logb = -0.5 * (((y[:, None, :] - mu) ** 2) / var + np.log(2 * np.pi * var)).sum(2)
        gamma, xi = forward_backward(logb, A)
        A = xi / xi.sum(1, keepdims=True)
        S = gamma.sum(0)[:, None] + 1e-9                               # [state, 1]
        Y, X = gamma.T @ y, gamma.T @ x                                # [state, coord]
        XX, XY = (x * x).sum(0), (x * y).sum(0)
        phi = np.clip((XY - (Y * X / S).sum(0)) / np.maximum(XX - (X * X / S).sum(0), 1e-9), 0.0, 0.999)
        c = (Y - phi * X) / S
        res = y[:, None, :] - c[None] - (x * phi)[:, None, :]
        var = np.maximum((gamma[:, :, None] * res ** 2).sum((0, 1)) / T, 1e-6)
    gamma, _ = forward_backward(logb, A)
    labels = np.concatenate([[gamma[0].argmax()], gamma.argmax(1)])    # first reading takes the next one's label
    means = c / (1 - phi)                                              # each state's long-run log-ratios
    return labels, gamma, A, phi, means


SETS = [(f"{n} (seed {s})", jd.out_file(n, s), jd.DECLARED[n]) for s in (0, 2) for n in jd.LISTS] + \
       [("PubMed abstracts 31-60 (held out)", RUNS / "heldout" / "S3.jsonl", 30.0),
        ("single-state streams (seed 1)", RUNS / "heldout" / "S2.jsonl", 190.5)]


def main():
    from head_of_state.decoders import probs
    from head_of_state.metrics import pool, stream_counts
    from head_of_state.settle import settled_labels
    print("A reading at every token. AR-HMM fitted per stream with no labels. Kept names: states whose learned "
          "long-run reading still favours their own option.\n")
    print("| Set | settled labels | token acc. | always the most common state | placed within 16 | false /1k | "
          "stay per token | carry-over phi | kept names |\n|---|---|---|---|---|---|---|---|---|")
    for label, f, L in SETS:
        recs = [json.loads(l) for l in f.open()]
        tokens = sum(r["n_tokens"] for r in recs)
        gold = np.concatenate([np.asarray(r["gold"]) for r in recs])
        base = np.bincount(gold).max() / len(gold)
        cur, ar, stays, phis, kept = [], [], [], [], []
        for r in recs:
            p, pos = probs(r, rotations=1)
            cur.append(stream_counts(settled_labels(p, pos, L), p, r["gold"], pos))
            lab, gamma, A, phi, means = fit(p, L)
            ar.append(stream_counts(lab, p, r["gold"], pos))
            stays.append(np.diag(A).mean()); phis.append(phi.mean())
            kept.append(f"{int((means.argmax(1) == np.arange(len(means))).sum())}/{len(means)}")
        for how, c, extra in (("current: Viterbi, declared length", cur, "| — | — | — |"),
                              ("AR-HMM", ar, f"| {np.mean(stays):.3f} | {np.mean(phis):.2f} | {', '.join(kept)} |")):
            m = pool(c, tokens)
            true = sum(x["true_b"] for x in c)
            placed = f"{sum(x['match'][16] for x in c)}/{true}" if true else "—"
            print(f"| {label} | {how} | {m['accuracy']:.3f} | {base:.3f} | {placed} | "
                  f"{1000 * sum(x['pred_b'] - x['match'][16] for x in c) / tokens:.1f} {extra}")


if __name__ == "__main__":
    main()
