"""Offline knobs and decoders over saved readings (rule H6 of docs/TEST_SUITE.md).

From one run that kept every ordering's letter log-probabilities, this reproduces what an online
StreamStateReader would have output for any rotations (a subset of the orderings run), stride,
prior-power and persist, and adds the sticky-transition forward filter of HANDOFF open question 2.
"""
from __future__ import annotations

import math

import numpy as np

from .reader import Reading, score


def shifts_for(K: int, rotations: int) -> list[int]:
    """The orderings StreamStateReader uses for `rotations` (same formula as __post_init__)."""
    r = max(1, min(rotations, K))
    return sorted({(i * K) // r for i in range(r)})


def probs(rec: dict, rotations: int, stride: int = 1, prior_power: float = 0.0, calibrate: bool = False):
    """Per-reading state probabilities and reading positions, as the online reader computes them.
    Positions are 1-based token counts; readings exist at multiples of the run's stride."""
    if "logp" in rec:   # head_of_state.readouts records: one row of K scores per token, no orderings to choose
        mean = np.asarray(rec["logp"], dtype=np.float64)
        K = mean.shape[1]
        pos = np.asarray(rec["positions"]) if "positions" in rec else np.arange(1, len(mean) + 1)
    else:
        lp = np.asarray(rec["shift_logp"], dtype=np.float64)          # [readings, shifts_run, K]
        K = lp.shape[2]
        if calibrate:   # subtract the content-free bias of each ordering (state order, log space)
            lp = lp - np.asarray(rec["content_free"], dtype=np.float64)[None]
        cols = [rec["shifts"].index(s) for s in shifts_for(K, rotations)]   # ValueError if not run
        mean = lp[:, cols, :].mean(1)
        pos = (np.asarray(rec["positions"]) if "positions" in rec
               else np.arange(1, len(lp) + 1) * rec["stride"])
    keep = pos % stride == 0
    mean, pos = mean[keep], pos[keep]
    out, prior, n = [], np.full(K, 1.0 / K), 0
    for m in mean:
        logp = m - prior_power * np.log(np.maximum(prior, 1e-6)) if prior_power > 0 else m
        p = np.exp(logp - logp.max()); p /= p.sum()
        if prior_power > 0:
            raw = np.exp(m - m.max()); raw /= raw.sum()
            n += 1
            prior += (raw - prior) / n
        out.append(p)
    return np.asarray(out), pos


def decode_raw(p: np.ndarray) -> list[int]:
    return [int(x) for x in p.argmax(1)]


def decode_persist(p: np.ndarray, persist: int) -> list[int]:
    """Same rule as StreamStateReader._commit."""
    committed, cand, run, out = None, None, 0, []
    for raw in p.argmax(1):
        raw = int(raw)
        if committed is None or raw == committed:
            committed, cand, run = raw, None, 0
        else:
            cand, run = (raw, run + 1) if raw == cand else (raw, 1)
            if run >= persist:
                committed, cand, run = raw, None, 0
        out.append(committed)
    return out


def decode_filter(p: np.ndarray, eta: float, seg_len_readings: float) -> list[int]:
    """b_t(k) ∝ p_t(k)^η · Σ_j b_{t-1}(j)·A_jk, with A_kk = 1 - 1/L and the rest spread evenly
    (D1: L is the stream's declared typical segment length, in readings)."""
    K = p.shape[1]
    stay = 1.0 - 1.0 / max(seg_len_readings, 1.0 + 1e-9)
    A = np.full((K, K), (1.0 - stay) / (K - 1))
    np.fill_diagonal(A, stay)
    b, out = np.full(K, 1.0 / K), []
    for pt in p:
        b = (b @ A) * np.power(np.maximum(pt, 1e-12), eta)
        b /= b.sum()
        out.append(int(b.argmax()))
    return out


def _transition(K: int, seg_len_readings: float) -> np.ndarray:
    stay = 1.0 - 1.0 / max(seg_len_readings, 1.0 + 1e-9)
    A = np.full((K, K), (1.0 - stay) / (K - 1))
    np.fill_diagonal(A, stay)
    return A


def decode_fixedlag(p: np.ndarray, eta: float, seg_len_readings: float, lag: int) -> list[int]:
    """Fixed-lag smoothing: the state at reading t is decided at reading t+lag, from the forward
    belief at t and the evidence of readings t+1..t+lag (same transition and η as decode_filter).
    score() then measures delay in positions; the decision latency is that delay plus `lag`."""
    K = p.shape[1]
    A = _transition(K, seg_len_readings)
    e = np.power(np.maximum(p, 1e-12), eta)
    alpha, a = [], np.full(K, 1.0 / K)
    for et in e:
        a = (a @ A) * et
        a /= a.sum()
        alpha.append(a)
    out, T = [], len(e)
    for t in range(T):
        b = np.ones(K)
        for s in range(min(t + lag, T - 1), t, -1):
            b = A @ (e[s] * b)
            b /= b.sum()
        out.append(int(np.argmax(alpha[t] * b)))
    return out


def surprisal_z(neutral_logp, window: int = 64) -> np.ndarray:
    """Causal z-score of each token's surprisal (-log p under the neutral opening) against the
    previous `window` tokens; 0 until 8 tokens of history exist."""
    s = -np.asarray(neutral_logp, dtype=np.float64)
    z = np.zeros_like(s)
    for t in range(len(s)):
        h = s[max(0, t - window): t]
        if len(h) >= 8:
            z[t] = (s[t] - h.mean()) / (h.std() + 1e-6)
    return z


def decode_filter_hazard(p: np.ndarray, rec: dict, eta: float, seg_len_readings: float, gamma: float) -> list[int]:
    """decode_filter with a per-token switch probability: 1/L times exp(gamma * z_t), z_t the
    surprisal z-score of token t (EM-LLM-style boundary signal), capped at 0.5."""
    K = p.shape[1]
    z = surprisal_z(rec["neutral"])
    b, out = np.full(K, 1.0 / K), []
    for t, pt in enumerate(p):
        h = min(0.5, np.exp(gamma * max(z[t], 0.0)) / max(seg_len_readings, 1.0 + 1e-9))
        b = (1 - h) * b + h * (1 - b) / (K - 1)
        b = b * np.power(np.maximum(pt, 1e-12), eta)
        b /= b.sum()
        out.append(int(b.argmax()))
    return out


def as_readings(p: np.ndarray, pos: np.ndarray, states: list[int]) -> list[Reading]:
    raw = p.argmax(1)
    return [Reading(int(n), [], int(r), s, i > 0 and s != states[i - 1])
            for i, (n, r, s) in enumerate(zip(pos, raw, states))]


def evaluate(records: list[dict], decoder, rotations: int, stride: int = 1, prior_power: float = 0.0,
             calibrate: bool = False):
    """Pool score() over streams. decoder(p) -> committed state per reading."""
    tot = dict(readings=0, raw_ok=0, com_ok=0, true=0, found=0, spurious=0, delay_sum=0.0, tokens=0)
    for rec in records:
        p, pos = probs(rec, rotations, stride, prior_power, calibrate)
        if len(p) == 0:
            continue
        states = decoder(p, rec) if getattr(decoder, "uses_record", False) else decoder(p)
        rs = as_readings(p, pos, states)
        start = rec.get("score_from", 1)   # decode the whole stream, score only from this position on
        if start > 1:
            rs = [r for r in rs if r.pos >= start]
            if rs:
                rs[0].switched = False
        m = score(rs, rec["gold"])
        n = m["readings"]
        tot["readings"] += n
        tot["raw_ok"] += round(m["accuracy_raw"] * n)
        tot["com_ok"] += round(m["accuracy_committed"] * n)
        tot["true"] += m["true_switches"]
        tot["found"] += m["switches_found"]
        tot["spurious"] += m["spurious_switches"]
        if m["switches_found"]:
            tot["delay_sum"] += m["mean_delay_tokens"] * m["switches_found"]
        tot["tokens"] += rec["n_tokens"] - rec.get("score_from", 1) + 1
    r = max(tot["readings"], 1)
    return {
        "accuracy_raw": tot["raw_ok"] / r,
        "accuracy_committed": tot["com_ok"] / r,
        "switches_matched": tot["found"],
        "true_switches": tot["true"],
        "mean_delay_tokens": tot["delay_sum"] / tot["found"] if tot["found"] else math.nan,
        "spurious_per_1k_tokens": 1000 * tot["spurious"] / max(tot["tokens"], 1),
    }
