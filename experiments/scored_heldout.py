"""Held-out confirmation with the revised frozen settings (docs/TEST_SUITE.md, "Held-out samples").

    H2O_LIGHTNING_PATH=<checkout> uv run python -I experiments/scored_heldout.py run S3|ACI|S2
    uv run python -I experiments/scored_heldout.py report     # results/scored_heldout.md

The model (H2O-Lightning-4B) and the settings (a reading at every token, 1 option ordering, forward
filter with eta 1 as the headline decoder, persist 3 alongside) were chosen after the first test
samples had been seen, so these samples are new: PubMed 20k RCT test abstracts 31-60 (seed-0 order),
the next 10 two-speaker ACI-Bench clinicalnlp_taskB_test1 encounters, and 2 null streams built with
seed 1. The report adds the forward filter at 0.25x and 4x the declared length (the pre-registered
sensitivity check).
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import aci_bench  # noqa: E402
import pubmed_rct  # noqa: E402
import wikisection  # noqa: E402
from common import RUNS  # noqa: E402
from scored_runs import FROZEN, s2_null  # noqa: E402

OUT = RUNS / "heldout"
SETS = {"S3": ("PubMed 20k RCT test, abstracts 31-60", 30.0), "ACI": ("ACI-Bench test, encounters 11-20", aci_bench.SEG_LEN),
        "S2": ("Null streams, seed 1", 190.5)}


def streams(name, tok):
    if name == "S3":
        s = pubmed_rct.load("test")
        random.Random(0).shuffle(s)
        return s[30:60], pubmed_rct.STATES, pubmed_rct.UNIT
    if name == "ACI":
        return aci_bench.load(n=20)[10:], aci_bench.STATES, aci_bench.UNIT
    return s2_null(tok, seed=1)[:2], wikisection.STATES, wikisection.UNIT


def run(name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from head_of_state.h2o import H2OReader
    from head_of_state.harness import run_stream
    path = os.environ.get("H2O_LIGHTNING_PATH", "h2oai/h2o-lightning-4b")
    tok_path = os.path.join(path, "tokenizer") if os.path.isdir(os.path.join(path, "tokenizer")) else path
    tok = AutoTokenizer.from_pretrained(tok_path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda")
    ss, states, unit = streams(name, tok)
    OUT.mkdir(parents=True, exist_ok=True)
    reader = H2OReader(model, tok, states, unit=unit, **FROZEN)
    n = sec = 0
    with (OUT / f"{name}.jsonl").open("w") as f:
        for s in ss:
            rec = run_stream(reader, s, tok)
            n += rec["n_tokens"]; sec += rec["seconds"]
            f.write(json.dumps(rec) + "\n")
    print(f"{name}: {len(ss)} streams, {n} tokens, {1000 * sec / n:.1f} ms/token", flush=True)


def report():
    from head_of_state.decoders import decode_persist, probs
    from head_of_state.metrics import forward_posteriors, pool, stream_counts
    print("| Set | decoder | token acc. | boundary F1 ±8 | ±16 | switches found | mean delay | spurious /1k | "
          "Brier | ECE | ms/token |\n|---|---|---|---|---|---|---|---|---|---|---|")
    for name, (label, L) in SETS.items():
        f = OUT / f"{name}.jsonl"
        if not f.exists():
            continue
        recs = [json.loads(l) for l in f.open()]
        tokens = sum(r["n_tokens"] for r in recs)
        ms = 1000 * sum(r["seconds"] for r in recs) / tokens
        for dname, fn in (("forward filter η=1 (headline)", lambda p: forward_posteriors(p, 1.0, L)),
                          ("persist 3", None),
                          ("forward filter, length ×0.25", lambda p: forward_posteriors(p, 1.0, L * 0.25)),
                          ("forward filter, length ×4", lambda p: forward_posteriors(p, 1.0, L * 4))):
            counts = []
            for r in recs:
                p, pos = probs(r, rotations=1)
                if fn is None:
                    q, lab = p, np.asarray(decode_persist(p, 3))
                else:
                    q = fn(p)
                    lab = q.argmax(1)
                counts.append(stream_counts(lab, q, r["gold"], pos))
            m = pool(counts, tokens)
            print(f"| {label} ({len(recs)} streams, {tokens} tokens) | {dname} | {m['accuracy']:.3f} | "
                  f"{m['boundary_f1@8']:.2f} | {m['boundary_f1@16']:.2f} | {m['matched']} | {m['mean_delay']:.1f} | "
                  f"{m['spurious_per_1k']:.1f} | {m['brier']:.3f} | {m['ece']:.3f} | {ms:.0f} |")


if __name__ == "__main__":
    run(sys.argv[2]) if sys.argv[1] == "run" else report()
