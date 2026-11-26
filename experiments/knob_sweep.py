"""Offline knob and decoder sweep over a saved run (HANDOFF task 6 and open question 2).

    uv run python -I experiments/knob_sweep.py experiments/runs/s3_dev_r5.jsonl --seg-len 30 --out results/s3_dev_sweep.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from head_of_state.decoders import decode_filter, decode_persist, decode_raw, evaluate  # noqa: E402

ROTATIONS = [1, 2, 3, 5]
STRIDES = [1, 4, 8]
PRIOR_POWERS = [0.0, 0.75]
PERSISTS = [1, 2, 3, 5, 8]
ETAS = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
SEG_LEN_FACTORS = [0.25, 1.0, 4.0]   # report: forward filter at declared length x0.25 and x4 (D1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--seg-len", type=float, required=True, help="declared segment length, tokens (D1)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    records = [json.loads(line) for line in a.run.open()]
    K = len(records[0]["shift_logp"][0][0])
    rows = []
    for rot in [r for r in ROTATIONS if r <= K]:
        for stride in STRIDES:
            for pp in PRIOR_POWERS:
                decoders = [("raw", "", decode_raw)]
                decoders += [("persist", n, partial(decode_persist, persist=n)) for n in PERSISTS]
                decoders += [(f"filter_L×{f:g}", eta, partial(decode_filter, eta=eta,
                              seg_len_readings=a.seg_len * f / stride))
                             for f in SEG_LEN_FACTORS for eta in ETAS]
                for name, param, dec in decoders:
                    m = evaluate(records, dec, rot, stride, pp)
                    rows.append({"rotations": rot, "stride": stride, "prior_power": pp,
                                 "decoder": name, "param": param, **{k: round(v, 4) if isinstance(v, float) else v
                                                                     for k, v in m.items()}})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} configurations, {len(records)} streams, "
          f"{sum(r['n_tokens'] for r in records)} tokens -> {a.out}")


if __name__ == "__main__":
    main()
