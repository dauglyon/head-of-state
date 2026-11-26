"""Summarise design-comparison runs.

    uv run python -I experiments/compare_readouts_report.py > results/compare_readouts.md

For every set and design: ms per token; best committed accuracy over all decoder settings; the
shortest mean delay among settings with at most SPUR_CAP spurious switches per 1k tokens and at
least MATCH_MIN of true switches matched; the decoder setting that achieved each. For next-token
(emission) designs, also the mean per-token evidence gap (log p under the true state minus the mean
over the other states) by distance from the description and by tokens since the last true switch.
"""
from __future__ import annotations

import json
import math
import sys
from functools import partial
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import RUNS as ALL_RUNS  # noqa: E402
from head_of_state.decoders import (decode_filter, decode_filter_hazard, decode_fixedlag, decode_persist,  # noqa: E402
                    decode_raw, evaluate)

RUNS = ALL_RUNS / "compare"
DEV_RUN = ALL_RUNS / "s3_dev_r5.jsonl"   # question-after baseline for E1 (5 orderings saved)
SPUR_CAP, MATCH_MIN = 2.0, 0.8
ETAS = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
LAGS = [8, 16, 32]
SEG = {"E1": 30.0, "E2": 190.5}
MS_PER_TOKEN_QAFTER_E1 = 149   # measured, 2 orderings batched, S3 prompt (eval run used 5 orderings)


def decoders(kind, seg, with_neutral=False):
    out = []
    if kind == "posterior":
        out.append(("raw", decode_raw, 0))
        out += [(f"persist {n}", partial(decode_persist, persist=n), 0) for n in (2, 3, 5, 8)]
    for f in (0.25, 1.0, 4.0):
        out += [(f"filter L×{f:g} η={e:g}", partial(decode_filter, eta=e, seg_len=seg * f), 0)
                for e in ETAS]
    out += [(f"fixed-lag {lag} L×1 η={e:g}", partial(decode_fixedlag, eta=e, seg_len=seg, lag=lag), lag)
            for lag in LAGS for e in ETAS]
    if with_neutral:
        for g in (0.5, 1.0, 2.0):
            for e in ETAS:
                f = partial(decode_filter_hazard, eta=e, seg_len=seg, gamma=g)
                f.uses_record = True
                out.append((f"surprisal-hazard γ={g:g} η={e:g}", f, 0))
    return out


def gaps(records):
    by_pos, by_since = {}, {}
    for rec in records:
        e, g = np.asarray(rec["logp"]), rec["gold"]
        since = 0
        for t, (row, y) in enumerate(zip(e, g)):
            since = 0 if t and g[t - 1] != y else since + 1
            gap = row[y] - (row.sum() - row[y]) / (len(row) - 1)
            pb = min(t // 256, 8) * 256
            sb = next(b for b in (8, 16, 32, 64, 128, 256, 10**9) if since < b)
            by_pos.setdefault(pb, []).append(gap)
            by_since.setdefault(sb, []).append(gap)
    return ({k: float(np.mean(v)) for k, v in sorted(by_pos.items())},
            {k: float(np.mean(v)) for k, v in sorted(by_since.items())})


def main():
    lines = ["# Design comparison", "",
             f"Delay figure: shortest mean delay (positions) among decoder settings with ≤{SPUR_CAP:g} spurious "
             f"switches per 1k tokens and ≥{MATCH_MIN:.0%} of true switches matched; for fixed-lag decoders the "
             "decision latency is that delay plus the lag.", ""]
    for set_name in ("E1", "E2"):
        lines += [f"## {set_name}", "", "| Design | ms/token | Best committed acc. (decoder) | "
                  "Shortest delay at the cap (decoder; spurious/1k, matched) |", "|---|---|---|---|"]
        files = sorted(RUNS.glob(f"{set_name}_*.jsonl"))
        sources = [(f.stem.split("_", 1)[1], f) for f in files]
        if set_name == "E1" and DEV_RUN.exists():
            sources.insert(0, ("qafter", DEV_RUN))
        for design, f in sources:
            recs = [json.loads(l) for l in f.open()]
            if design == "qafter" and set_name == "E1":
                recs = recs[:6]   # same abstracts as compare_readouts.E1_N
                ms = MS_PER_TOKEN_QAFTER_E1
            else:
                ms = 1000 * sum(r["seconds"] for r in recs) / sum(r["n_tokens"] for r in recs)
            kind = recs[0].get("kind", "posterior")
            best_acc, best_delay = (0, ""), (math.inf, "", 0, 0)
            for name, dec, lag in decoders(kind, SEG[set_name], "neutral" in recs[0]):
                m = evaluate(recs, dec, rotations=2)
                if m["accuracy_committed"] > best_acc[0]:
                    best_acc = (m["accuracy_committed"], name)
                frac = m["switches_matched"] / max(m["true_switches"], 1)
                if m["spurious_per_1k_tokens"] <= SPUR_CAP and frac >= MATCH_MIN and m["mean_delay_tokens"] < best_delay[0]:
                    best_delay = (m["mean_delay_tokens"], name, m["spurious_per_1k_tokens"], frac)
            d = (f"{best_delay[0]:.1f} ({best_delay[1]}; {best_delay[2]:.2f}, {best_delay[3]:.0%})"
                 if best_delay[1] else "none meets the cap")
            lines.append(f"| {design} ({len(recs)} streams, {sum(r['n_tokens'] for r in recs)} tokens) | {ms:.0f} | "
                         f"{best_acc[0]:.3f} ({best_acc[1]}) | {d} |")
            if kind == "emission":
                bp, bs = gaps(recs)
                lines.append(f"| ↳ {design} evidence gap by position | | " +
                             ", ".join(f"{k}: {v:+.3f}" for k, v in bp.items()) + " | |")
                lines.append(f"| ↳ {design} evidence gap by tokens since switch (<) | | " +
                             ", ".join(f"{k if k < 10**9 else '∞'}: {v:+.3f}" for k, v in bs.items()) + " | |")
        lines.append("")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
