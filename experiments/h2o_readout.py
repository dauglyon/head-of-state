"""H2O-Lightning-4B as the readout model, on the scored sets of scored_runs.py.

    H2O_LIGHTNING_PATH=<local checkout or hf id> uv run python -I experiments/h2o_readout.py run
    uv run python -I experiments/h2o_readout.py report     # results/h2o_readout.md

H2O-Lightning-4B (h2oai/h2o-lightning-4b, Apache-2.0) is a decision model fine-tuned from Qwen3.5-4B to
answer choice questions about a record. head_of_state.h2o.H2OReader builds its prompt format around the
cached stream (checked token for token against the model's own serving shim). Everything else is the
frozen setting of scored_runs.py: 1 option ordering, a reading every 8th token, persist 1, forward
filter with eta 1; the same 30 PubMed 20k RCT test abstracts and 4 null streams. The base model's
results on the same streams come from experiments/runs/scored.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pubmed_rct  # noqa: E402
import wikisection  # noqa: E402
from common import RUNS  # noqa: E402
from scored_runs import STRIDE8, s2_null, s3_test  # noqa: E402

OUT = RUNS / "h2o"


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
                                        ("S2null", s2_null(tok), wikisection.STATES, wikisection.UNIT)):
        reader = H2OReader(model, tok, states, unit=unit, **STRIDE8)
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
    print("| Set | model | decoder | token acc. | boundary F1 ±8 | ±16 | matched | mean delay | spurious /1k | "
          "Brier | ECE | ms/token |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, L in (("S3test", 30.0), ("S2null", 190.5)):
        for model, d in (("Qwen3.5-4B (base)", RUNS / "scored"), ("H2O-Lightning-4B", OUT)):
            recs = [json.loads(l) for l in (d / f"{name}.jsonl").open()]
            ms = 1000 * sum(r["seconds"] for r in recs) / sum(r["n_tokens"] for r in recs)
            for dname in ("persist 1", "forward filter η=1"):
                counts, tokens = [], 0
                for r in recs:
                    p, pos = probs(r, rotations=1)
                    if dname == "persist 1":
                        q, lab = p, np.asarray(decode_persist(p, 1))
                    else:
                        q = forward_posteriors(p, 1.0, L, None, STRIDE8["stride"])
                        lab = q.argmax(1)
                    counts.append(stream_counts(lab, q, r["gold"], pos))
                    tokens += r["n_tokens"]
                m = pool(counts, tokens)
                print(f"| {name} | {model} | {dname} | {m['accuracy']:.3f} | {m['boundary_f1@8']:.2f} | "
                      f"{m['boundary_f1@16']:.2f} | {m['matched']} | {m['mean_delay']:.1f} | {m['spurious_per_1k']:.1f} | "
                      f"{m['brier']:.3f} | {m['ece']:.3f} | {ms:.0f} |")


if __name__ == "__main__":
    {"run": run, "report": report}[sys.argv[1]]()
