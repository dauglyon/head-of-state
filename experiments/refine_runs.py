"""Bisection refinement between disagreeing readings, on the scored PubMed test and ACI-Bench sets.

    H2O_LIGHTNING_PATH=<checkout> uv run python -I experiments/refine_runs.py run
    uv run python -I experiments/refine_runs.py report     # results/refine_runs.md

H2O-Lightning-4B with the frozen settings (a reading every 8th token, 1 ordering, persist 1), plus
`refine`: where two consecutive readings disagree on the likeliest state, three bisection readings
(at +4, then +-2, then +-1 tokens) place the switch at a single token. The same streams without refine
are in experiments/runs/h2o/S3test.jsonl and experiments/runs/aci/frozen.jsonl.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import aci_bench  # noqa: E402
import pubmed_rct  # noqa: E402
from common import RUNS  # noqa: E402
from scored_runs import FROZEN, s3_test  # noqa: E402

OUT = RUNS / "refine"
SETS = {"S3test": (30.0, RUNS / "h2o" / "S3test.jsonl"), "ACI": (aci_bench.SEG_LEN, RUNS / "aci" / "frozen.jsonl")}


def run():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from head_of_state.h2o import H2OReader
    from head_of_state.harness import run_stream
    path = os.environ.get("H2O_LIGHTNING_PATH", "h2oai/h2o-lightning-4b")
    tok_path = os.path.join(path, "tokenizer") if os.path.isdir(os.path.join(path, "tokenizer")) else path
    tok = AutoTokenizer.from_pretrained(tok_path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda")
    OUT.mkdir(parents=True, exist_ok=True)
    for name, streams, states, unit in (("S3test", s3_test(), pubmed_rct.STATES, pubmed_rct.UNIT),
                                        ("ACI", aci_bench.load(), aci_bench.STATES, aci_bench.UNIT)):
        reader = H2OReader(model, tok, states, unit=unit, refine=True, **FROZEN)
        n = sec = reads = 0
        with (OUT / f"{name}.jsonl").open("w") as f:
            for s in streams:
                rec = run_stream(reader, s, tok)
                n += rec["n_tokens"]; sec += rec["seconds"]; reads += len(rec["positions"])
                f.write(json.dumps(rec) + "\n")
        print(f"{name}: {n} tokens, {1000 * sec / n:.1f} ms/token, {1000 * reads / n:.0f} readings per 1k tokens", flush=True)


def viterbi(p, seg_len_tokens, positions, lag):
    """Fixed-lag Viterbi over the readings (emissions log p, eta 1); the sticky transition uses the
    declared length in readings at the median reading spacing. lag in readings; None = offline."""
    from head_of_state.reader import FixedLagViterbi, sticky_log_transition
    gap = float(np.median(np.diff(positions))) if len(positions) > 1 else 1.0
    v = FixedLagViterbi(sticky_log_transition(p.shape[1], max(seg_len_tokens / gap, 1.01)), lag)
    out = []
    for row in np.log(np.maximum(p, 1e-12)):
        out += v.push(row)
    return np.asarray(out + v.flush())


def report():
    from head_of_state.decoders import decode_persist, probs
    from head_of_state.metrics import pool, stream_counts
    decoders = (("persist 1", lambda p, pos, L: np.asarray(decode_persist(p, 1)), 0),
                ("Viterbi, lag 1 reading", lambda p, pos, L: viterbi(p, L, pos, 1), 8),
                ("Viterbi, offline", lambda p, pos, L: viterbi(p, L, pos, None), 0))
    print("| Set | reading, decoder | token acc. | boundary F1 ±2 | ±8 | ±16 | matched | mean delay | spurious /1k | "
          "readings /1k tokens | ms/token |\n|---|---|---|---|---|---|---|---|---|---|---|")
    for name, (L, plain) in SETS.items():
        for label, f in (("every 8th token", plain), ("every 8th token + bisection", OUT / f"{name}.jsonl")):
            recs = [json.loads(l) for l in open(f)]
            for dname, dec, lag_tokens in decoders:
                counts, tokens, reads = [], 0, 0
                for r in recs:
                    p, pos = probs(r, rotations=1)
                    if "positions" not in r:      # older records: readings at multiples of the stride
                        pos = np.arange(1, len(p) + 1) * r["stride"]
                    counts.append(stream_counts(dec(p, pos, L), p, r["gold"], pos, ks=(2, 8, 16)))
                    tokens += r["n_tokens"]
                    reads += len(p)
                m = pool(counts, tokens, ks=(2, 8, 16))
                ms = 1000 * sum(r["seconds"] for r in recs) / tokens
                delay = f"{m['mean_delay']:.1f}" + (f" (+{lag_tokens} decision)" if lag_tokens else "")
                print(f"| {name} ({len(recs)} streams, {tokens} tokens) | {label}, {dname} | {m['accuracy']:.3f} | "
                      f"{m['boundary_f1@2']:.2f} | {m['boundary_f1@8']:.2f} | {m['boundary_f1@16']:.2f} | "
                      f"{m['matched']} | {delay} | {m['spurious_per_1k']:.1f} | {1000 * reads / tokens:.0f} | {ms:.0f} |")

if __name__ == "__main__":
    {"run": run, "report": report}[sys.argv[1]]()
