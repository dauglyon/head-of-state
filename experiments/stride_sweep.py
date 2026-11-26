"""Option orderings x reading stride for the question readout, offline from the saved PubMed dev run.

    uv run python -I experiments/stride_sweep.py

Uses the first 6 abstracts of experiments/runs/s3_dev_r5.jsonl (all 5 orderings saved at every token).
Decision latency = mean delay + lag x stride for fixed-lag smoothing, = mean delay otherwise; the cap
is at most 2 spurious switches per 1k tokens with at least 80% of true switches matched. ms/token
assumes chunked feeding: one 31 ms stream pass plus the measured reading cost (1 ordering 69 ms,
2 orderings 118 ms, S3 prompt), together divided by the stride.
"""
import json
import math
import sys
from functools import partial
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import RUNS  # noqa: E402
from head_of_state.decoders import decode_filter, decode_fixedlag, decode_persist, decode_raw, evaluate  # noqa: E402

SEG, FEED, READ = 30.0, 31.0, {1: 69.0, 2: 118.0}


def main():
    recs = [json.loads(l) for l in (RUNS / "s3_dev_r5.jsonl").open()][:6]
    print("| orderings | stride | ms/token (est.) | best causal acc. (decoder) | latency at cap, causal | "
          "latency at cap, fixed-lag |\n|---|---|---|---|---|---|")
    for rot in (1, 2):
        for stride in (1, 2, 4, 8, 16):
            causal = [("raw", decode_raw, 0)] + [(f"persist {n}", partial(decode_persist, persist=n), 0) for n in (2, 3, 5)]
            causal += [(f"filter η={e:g}", partial(decode_filter, eta=e, seg_len_readings=SEG / stride), 0)
                       for e in (0.25, 0.5, 1, 2, 4)]
            lagged = [(f"lag {l} η={e:g}", partial(decode_fixedlag, eta=e, seg_len_readings=SEG / stride, lag=l), l)
                      for l in (1, 2, 4) for e in (0.5, 1, 2)]
            out = []
            for decs in (causal, lagged):
                best, lat = (0, ""), (math.inf, "", 0.0)
                for n, d, lag in decs:
                    m = evaluate(recs, d, rotations=rot, stride=stride)
                    if m["accuracy_committed"] > best[0]:
                        best = (m["accuracy_committed"], n)
                    frac = m["switches_matched"] / m["true_switches"]
                    latency = m["mean_delay_tokens"] + lag * stride
                    if m["spurious_per_1k_tokens"] <= 2 and frac >= 0.8 and latency < lat[0]:
                        lat = (latency, n, m["spurious_per_1k_tokens"])
                out.append((best, lat))
            (cb, cl), (_, ll) = out
            f = lambda l: f"{l[0]:.1f} ({l[1]}; {l[2]:.2f}/1k)" if l[1] else "none"
            print(f"| {rot} | {stride} | {(FEED + READ[rot]) / stride:.0f} | {cb[0]:.3f} ({cb[1]}) | {f(cl)} | {f(ll)} |")


if __name__ == "__main__":
    main()
