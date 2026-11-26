"""Scored runs with the revised frozen settings: a reading at every token (docs/TEST_SUITE.md, H3).

    H2O_LIGHTNING_PATH=<checkout> uv run python -I experiments/scored_every_token.py run S3test|ACI|S2null
    uv run python -I experiments/scored_every_token.py report     # results/scored_every_token.md

H2O-Lightning-4B, 1 option ordering, a reading at every token, persist 3 and the forward filter
(eta 1). Streams: the 30 PubMed 20k RCT test abstracts and 4 null streams of scored_runs.py, and the 10
ACI-Bench encounters of aci_bench.py. The same streams read every 8th token are in
experiments/runs/h2o (PubMed, null) and experiments/runs/aci (ACI-Bench).
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
import wikisection  # noqa: E402
from common import RUNS  # noqa: E402
from scored_runs import FROZEN, s2_null, s3_test  # noqa: E402

OUT = RUNS / "every_token"
SETS = {"S3test": (30.0, RUNS / "h2o" / "S3test.jsonl"), "ACI": (aci_bench.SEG_LEN, RUNS / "aci" / "frozen.jsonl"),
        "S2null": (190.5, RUNS / "h2o" / "S2null.jsonl")}


def run(name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from head_of_state.h2o import H2OReader
    from head_of_state.harness import run_stream
    path = os.environ.get("H2O_LIGHTNING_PATH", "h2oai/h2o-lightning-4b")
    tok_path = os.path.join(path, "tokenizer") if os.path.isdir(os.path.join(path, "tokenizer")) else path
    tok = AutoTokenizer.from_pretrained(tok_path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda")
    streams, states, unit = {"S3test": (s3_test(), pubmed_rct.STATES, pubmed_rct.UNIT),
                             "ACI": (aci_bench.load(), aci_bench.STATES, aci_bench.UNIT),
                             "S2null": (s2_null(tok), wikisection.STATES, wikisection.UNIT)}[name]
    OUT.mkdir(parents=True, exist_ok=True)
    reader = H2OReader(model, tok, states, unit=unit, **FROZEN)
    n = sec = 0
    with (OUT / f"{name}.jsonl").open("w") as f:
        for s in streams:
            rec = run_stream(reader, s, tok)
            n += rec["n_tokens"]; sec += rec["seconds"]
            f.write(json.dumps(rec) + "\n")
    print(f"{name}: {len(streams)} streams, {n} tokens, {1000 * sec / n:.1f} ms/token", flush=True)


def report():
    from head_of_state.decoders import decode_persist, probs
    from head_of_state.metrics import forward_posteriors, pool, stream_counts
    print("| Set | reading | decoder | token acc. | boundary F1 ±8 | ±16 | matched | mean delay | spurious /1k | "
          "Brier | ECE | ms/token |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, (L, stride8) in SETS.items():
        for label, f, stride, persist in (("every token (frozen)", OUT / f"{name}.jsonl", 1, 3),
                                          ("every 8th token (earlier)", stride8, 8, 1)):
            if not f.exists():
                continue
            recs = [json.loads(l) for l in f.open()]
            ms = 1000 * sum(r["seconds"] for r in recs) / sum(r["n_tokens"] for r in recs)
            for dname in (f"persist {persist}", "forward filter η=1"):
                counts, tokens = [], 0
                for r in recs:
                    p, pos = probs(r, rotations=1)
                    if "positions" not in r:
                        pos = np.arange(1, len(p) + 1) * r["stride"]
                    if dname.startswith("persist"):
                        q, lab = p, np.asarray(decode_persist(p, persist))
                    else:
                        q = forward_posteriors(p, 1.0, L, None, stride)
                        lab = q.argmax(1)
                    counts.append(stream_counts(lab, q, r["gold"], pos))
                    tokens += r["n_tokens"]
                m = pool(counts, tokens)
                print(f"| {name} ({len(recs)} streams, {tokens} tokens) | {label} | {dname} | {m['accuracy']:.3f} | "
                      f"{m['boundary_f1@8']:.2f} | {m['boundary_f1@16']:.2f} | {m['matched']} | {m['mean_delay']:.1f} | "
                      f"{m['spurious_per_1k']:.1f} | {m['brier']:.3f} | {m['ece']:.3f} | {ms:.0f} |")


if __name__ == "__main__":
    run(sys.argv[2]) if sys.argv[1] == "run" else report()
