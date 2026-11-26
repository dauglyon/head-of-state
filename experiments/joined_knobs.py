"""HANDOFF task 6 on the joined-document streams: time to notice against false switches, offline.

    uv run python -I experiments/joined_knobs.py              # results/joined_knobs.md, results/joined_knobs.csv
    uv run python -I experiments/plot_tradeoff.py results/joined_knobs.csv results/joined_knobs.html \
        --title "Joined documents (tuning streams)" --rotations 1 --stride swept --prior-power 0 --series joined --keep-order \
        --subtitle "Each line runs from a reading at every token to one every 16th token (hover a point for its interval). Delay is averaged over the switches the live output noticed; lower left is better."

Knobs replayed on the saved every-token readings of the seed-0 streams (joined_documents.py):
  - the decoder's segment length, 0.25x to 4x the declared length (stickier transitions at larger values);
  - the reading interval, 1 to 16 tokens. A reading at token N sees the same cache whether or not the tokens
    before it were read, so keeping every Nth saved reading reproduces a run with --stride N.
Time to notice comes from the online decision, false switches and accuracy from the settled labels.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import joined_documents as jd  # noqa: E402
from common import RESULTS  # noqa: E402

SCALES = (0.25, 0.5, 1, 2, 4)
STRIDES = (1, 2, 4, 8, 16)


def main():
    from head_of_state.decoders import probs
    from head_of_state.metrics import pool, stream_counts
    from head_of_state.settle import online_labels, settled_labels
    rows = []
    for name in jd.LISTS:
        recs = [json.loads(l) for l in jd.out_file(name, 0).open()]
        tokens = sum(r["n_tokens"] for r in recs)
        for stride in STRIDES:
            for scale in SCALES:
                L = jd.DECLARED[name] * scale
                settled, online = [], []
                for r in recs:
                    p, pos = probs(r, rotations=1, stride=stride)
                    settled.append(stream_counts(settled_labels(p, pos, L), p, r["gold"], pos))
                    online.append(stream_counts(online_labels(p, pos, L), p, r["gold"], pos))
                m, o = pool(settled, tokens), pool(online, tokens)
                rows.append({"decoder": name, "param": f"L×{scale:g}", "stride": stride, "rotations": 1,
                             "prior_power": 0.0, "mean_delay_tokens": round(o["mean_delay"], 2),
                             "spurious_per_1k_tokens": round(1000 * sum(c["pred_b"] - c["match"][16] for c in settled) / tokens, 2),
                             "accuracy_committed": round(m["accuracy"], 4),
                             "switches_matched": sum(c["match"][16] for c in settled),
                             "true_switches": sum(c["true_b"] for c in settled)})
    # The plotted tradeoff: the reading interval at the declared length (the length barely moves anything).
    plot = [dict(r, param=f"{r['stride']} tokens", stride="swept") for r in rows if r["param"] == "L×1"]
    with (RESULTS / "joined_knobs.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(plot[0]))
        w.writeheader(); w.writerows(plot)
    print("Seed-0 (tuning) streams, settled labels. Time to notice: tokens from a true switch to the online "
          "decision showing the new state. False switches: settled switches not within 16 tokens of a true one.\n")
    print("| State list | reading interval | decoder length | token acc. | placed within 16 | false /1k | time to notice |\n"
          "|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['decoder']} | {r['stride']} | {r['param']} | {r['accuracy_committed']:.3f} | "
              f"{r['switches_matched']}/{r['true_switches']} | {r['spurious_per_1k_tokens']:.1f} | {r['mean_delay_tokens']:.1f} |")


if __name__ == "__main__":
    main()
