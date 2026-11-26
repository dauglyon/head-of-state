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
