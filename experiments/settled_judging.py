"""Re-judge saved every-token runs by the revised judging: settled labels, time to notice alongside.

    uv run python -I experiments/settled_judging.py > results/settled_judging.md

Sets: the held-out samples (experiments/runs/heldout, H2O-Lightning-4B, every token) and the earlier
every-token runs on the first test samples (experiments/runs/every_token; not held out for the model
and settings). Settled labels: head_of_state.settle.settled_labels with shift = round(0.2 x declared
length). Time to notice: head_of_state.settle.online_labels scored with reader.score.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import aci_bench  # noqa: E402
from common import RUNS  # noqa: E402
from head_of_state.decoders import probs  # noqa: E402
from head_of_state.metrics import pool, stream_counts  # noqa: E402
from head_of_state.settle import online_labels, settled_labels  # noqa: E402

SETS = [("Held-out: PubMed 20k RCT test, abstracts 31-60", RUNS / "heldout" / "S3.jsonl", 30.0),
        ("Held-out: ACI-Bench test, encounters 11-20", RUNS / "heldout" / "ACI.jsonl", aci_bench.SEG_LEN),
        ("Held-out: null streams, seed 1", RUNS / "heldout" / "S2.jsonl", 190.5),
        ("First test sample (not held out): PubMed abstracts 1-30", RUNS / "every_token" / "S3test.jsonl", 30.0),
        ("First test sample (not held out): ACI-Bench encounters 1-10", RUNS / "every_token" / "ACI.jsonl", aci_bench.SEG_LEN),
        ("First test sample (not held out): null streams, seed 0", RUNS / "every_token" / "S2null.jsonl", 190.5)]


def main():
    print("| Set | token acc. (settled) | boundary F1 ±8 | ±16 | switches found (±16) | spurious /1k | "
          "time to notice (online, tokens) | ms/token |\n|---|---|---|---|---|---|---|---|")
    for label, f, L in SETS:
        if not f.exists():
            continue
        recs = [json.loads(l) for l in f.open()]
        shift = int(round(0.2 * L))
        settled, online, tokens = [], [], 0
        for r in recs:
            p, pos = probs(r, rotations=1)
            settled.append(stream_counts(settled_labels(p, pos, L, shift), p, r["gold"], pos))
            online.append(stream_counts(online_labels(p, pos, L), p, r["gold"], pos))
            tokens += r["n_tokens"]
        m, o = pool(settled, tokens), pool(online, tokens)
        # settled labels may place a switch slightly before or after the true one: a switch is found when a
        # settled switch lies within 16 tokens of it, and every other settled switch is spurious
        found = sum(c["match"][16] for c in settled)
        true = sum(c["true_b"] for c in settled)
        spurious = sum(c["pred_b"] - c["match"][16] for c in settled)
        ms = 1000 * sum(r["seconds"] for r in recs) / tokens
        notice = f"{o['mean_delay']:.1f}" if o["mean_delay"] == o["mean_delay"] else "—"
        print(f"| {label} ({len(recs)} streams, {tokens} tokens) | {m['accuracy']:.3f} | {m['boundary_f1@8']:.2f} | "
              f"{m['boundary_f1@16']:.2f} | {found}/{true} | {1000 * spurious / tokens:.1f} | {notice} | {ms:.0f} |")


if __name__ == "__main__":
    main()
