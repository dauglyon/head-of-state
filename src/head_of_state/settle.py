"""Settled labels and time to notice (docs/TEST_SUITE.md, "Judging").

A stream read left to right can only notice a switch once enough of the new text has arrived, so the
output is judged on its settled labels: the best path from Viterbi over all readings, where each
reading is treated as evidence for the state `shift` tokens earlier (a reading describes the last few
sentences, so it runs behind the text). Time to notice is measured separately, from the online
decision: the likeliest state of a max-product forward pass at each reading, with no lookahead.
"""
from __future__ import annotations

import numpy as np

from .reader import FixedLagViterbi, sticky_log_transition, sticky_transition


def settled_labels(p: np.ndarray, positions, seg_len: float, shift: int = 0) -> np.ndarray:
    """Best-path label for each reading. Readings are moved `shift` tokens earlier (in readings at the
    median spacing); the last readings are repeated to fill the end."""
    positions = np.asarray(positions)
    gap = int(round(float(np.median(np.diff(positions))))) if len(positions) > 1 else 1
    k = int(round(shift / max(gap, 1)))
    ps = np.concatenate([p[k:], np.repeat(p[-1:], k, axis=0)]) if k > 0 else p
    v = FixedLagViterbi(sticky_log_transition(p.shape[1], seg_len, gap), None)
    for row in np.log(np.maximum(ps, 1e-12)):
        v.push(row)
    return np.asarray(v.flush())


def online_labels(p: np.ndarray, positions, seg_len: float) -> np.ndarray:
    """The decision available at each reading with no lookahead: argmax of the max-product forward pass."""
    positions = np.asarray(positions)
    gap = int(round(float(np.median(np.diff(positions))))) if len(positions) > 1 else 1
    v = FixedLagViterbi(sticky_log_transition(p.shape[1], seg_len, gap), 0)
    out = []
    for row in np.log(np.maximum(p, 1e-12)):
        out += v.push(row)
    return np.asarray(out)


def reading_evidence(p: np.ndarray, positions, seg_len: float, floor: float = 1e-6) -> np.ndarray:
    """Per-reading evidence implied by readings that are already running estimates of the state.

    A reading is the model's answer given all the text so far, i.e. a filtered estimate P(state | text so
    far), not evidence from the newest tokens alone. The forward pass of an HMM satisfies
        estimate_t  ∝  evidence_t * (estimate_{t-1} @ A)
    so the evidence of the tokens since the previous reading is recovered as
        evidence_t  ∝  reading_t / (reading_{t-1} @ A)
    with A the sticky transition between the two readings (uniform before the first reading). This assumes the
    readings behave like a forward filter with transition A, which has not been shown; the seg_len used for A
    in experiments/joined_evidence.py was tuned against true labels. Not used by the tool (docs/LOG.md)."""
    positions = np.asarray(positions)
    gap = int(round(float(np.median(np.diff(positions))))) if len(positions) > 1 else 1
    A = sticky_transition(p.shape[1], seg_len, gap)
    prev = np.full(p.shape[1], 1.0 / p.shape[1])
    out = np.empty_like(p, dtype=np.float64)
    for t, row in enumerate(np.maximum(p, floor)):
        e = row / np.maximum(prev @ A, floor)
        out[t] = e / e.sum()
        prev = row / row.sum()
    return out


def estimate_transition(p: np.ndarray, iters: int = 200, init_stay: float = 0.9, floor: float = 1e-6) -> np.ndarray:
    """Per-reading transition table estimated from the readings alone (no labels).

    Readings are running estimates r_t = P(state_t | text so far). With a table A, the probability that
    the state was j at the previous reading and is k now, given the text so far, is
        xi_t(j, k) = r_{t-1}(j) A_jk r_t(k) / (r_{t-1} A)_k
    (the filtered form of the pair term in Baum-Welch's transition update). The update
    A_jk <- sum_t xi_t(j, k) / sum_t sum_k xi_t(j, k) is repeated to a fixed point."""
    q = np.maximum(p, floor)
    q = q / q.sum(1, keepdims=True)
    prev, cur = q[:-1], q[1:]
    K = q.shape[1]
    A = np.full((K, K), (1 - init_stay) / (K - 1))
    np.fill_diagonal(A, init_stay)
    for _ in range(iters):
        w = cur / np.maximum(prev @ A, floor)
        num = A * (prev.T @ w)
        new = num / num.sum(1, keepdims=True)
        if np.abs(new - A).max() < 1e-9:
            A = new
            break
        A = new
    return A


def evidence_from_readings(p: np.ndarray, A: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    """Evidence of the tokens since the previous reading: e_t ∝ r_t / (r_{t-1} A), from the forward-pass
    identity r_t ∝ e_t * (r_{t-1} A); the first reading's evidence is the reading itself."""
    q = np.maximum(p, floor)
    q = q / q.sum(1, keepdims=True)
    out = np.empty_like(q)
    out[0] = q[0]
    e = q[1:] / np.maximum(q[:-1] @ A, floor)
    out[1:] = e / e.sum(1, keepdims=True)
    return out


def settled_labels_estimated(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Settled labels with the transition table estimated from the readings: Viterbi over the evidence
    derived from the readings, using that same table. Returns (labels, table)."""
    import torch
    A = estimate_transition(p)
    e = evidence_from_readings(p, A)
    v = FixedLagViterbi(torch.tensor(np.log(np.maximum(A, 1e-300)), dtype=torch.float64), None)
    for row in np.log(np.maximum(e, 1e-300)):
        v.push(row)
    return np.asarray(v.flush()), A


def smoothed_estimated(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Settled state probabilities from the readings alone: the transition table is estimated from the
    readings (estimate_transition), the evidence derived from them (evidence_from_readings), and the
    forward-backward pass combines both. Since the forward pass of (evidence, table) reproduces the readings,
    the readings are the forward half and only the backward half is computed. Returns (probabilities, table);
    the likeliest state per token maximises expected token accuracy (posterior decoding)."""
    A = estimate_transition(p)
    e = evidence_from_readings(p, A)
    q = np.maximum(p, 1e-6)
    alpha = q / q.sum(1, keepdims=True)
    beta = np.ones_like(alpha)
    for t in range(len(alpha) - 2, -1, -1):
        b = A @ (e[t + 1] * beta[t + 1])
        beta[t] = b / b.sum()
    post = alpha * beta
    return post / post.sum(1, keepdims=True), A
