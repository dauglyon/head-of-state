"""Readout-quality check at the cheap setting: content-free calibration and 1 vs 2 option orderings.

    uv run python -I experiments/calibration_check.py run      # GPU: experiments/runs/calib/{E1,E2}.jsonl
    uv run python -I experiments/calibration_check.py report   # CPU

Readings every 8th token with 2 orderings saved per reading (1 ordering is a subset) and the
content-free bias (the question asked before any stream token) saved per stream. Sets: E1 and E2 of
compare_readouts.py.
"""
import json
import math
import sys
from functools import partial
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import RUNS  # noqa: E402
import compare_readouts  # noqa: E402
from head_of_state.decoders import decode_filter, decode_persist, decode_raw, evaluate  # noqa: E402

OUT = RUNS / "calib"


def run():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from head_of_state.harness import UnitReader, run_stream
    tok = AutoTokenizer.from_pretrained(compare_readouts.MODEL)
    model = AutoModelForCausalLM.from_pretrained(compare_readouts.MODEL, dtype=torch.bfloat16, device_map="cuda")
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (streams, states, unit, seg) in (("E1", compare_readouts.e1()), ("E2", compare_readouts.e2(tok))):
        reader = UnitReader(model, tok, states, rotations=2, stride=8, unit=unit)
        with (OUT / f"{name}.jsonl").open("w") as f:
            for s in streams:
                rec = run_stream(reader, s, tok)
                rec["seg_len"] = seg
                f.write(json.dumps(rec) + "\n")
                print(name, s["id"], f"{1000 * rec['seconds'] / rec['n_tokens']:.1f} ms/token", flush=True)


def report():
    print("| Set | orderings | calibrated | raw acc. | best committed acc. (decoder) | "
          "shortest delay at ≤4 spurious/1k and ≥70% matched |\n|---|---|---|---|---|---|")
    for name in ("E1", "E2"):
        recs = [json.loads(l) for l in (OUT / f"{name}.jsonl").open()]
        seg = recs[0]["seg_len"]
        for rot in (1, 2):
            for cal in (False, True):
                decs = [("raw", decode_raw)] + [(f"persist {n}", partial(decode_persist, persist=n)) for n in (2, 3)]
                decs += [(f"filter η={e:g}", partial(decode_filter, eta=e, seg_len=seg, gap=8)) for e in (0.5, 1, 2, 4)]
                best, delay, raw_acc = (0, ""), (math.inf, ""), None
                for n, d in decs:
                    m = evaluate(recs, d, rotations=rot, calibrate=cal)
                    raw_acc = m["accuracy_raw"]
                    if m["accuracy_committed"] > best[0]:
                        best = (m["accuracy_committed"], n)
                    frac = m["switches_matched"] / m["true_switches"]
                    if m["spurious_per_1k_tokens"] <= 4 and frac >= 0.7 and m["mean_delay_tokens"] < delay[0]:
                        delay = (m["mean_delay_tokens"], n)
                d = f"{delay[0]:.1f} ({delay[1]})" if delay[1] else "none"
                print(f"| {name} | {rot} | {cal} | {raw_acc:.3f} | {best[0]:.3f} ({best[1]}) | {d} |")


if __name__ == "__main__":
    {"run": run, "report": report}[sys.argv[1]]()
