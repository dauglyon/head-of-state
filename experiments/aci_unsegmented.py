"""Conversations without line breaks: does the tool find speaker changes from the words alone?

    H2O_LIGHTNING_PATH=<checkout> uv run python -I experiments/aci_unsegmented.py run
    uv run python -I experiments/aci_unsegmented.py report     # results/aci_unsegmented.md

In the ACI-Bench streams of aci_bench.py each turn is one line, so every switch falls on a line break.
Here the same two-speaker test encounters (31-38, not used before) are joined into continuous text with
one space between turns, and compared with the line-break version. H2O-Lightning-4B, the current
question, a reading every 4th token, judged on settled labels with time to notice.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import aci_bench  # noqa: E402
from common import RUNS  # noqa: E402

OUT = RUNS / "aci_unsegmented"
STRIDE = 4
VERSIONS = {"no line breaks": " ", "line breaks": "\n"}


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
    reader = H2OReader(model, tok, aci_bench.STATES, unit=aci_bench.UNIT, rotations=1, stride=STRIDE, persist=1)
    for name, joiner in VERSIONS.items():
        streams = aci_bench.load(n=38, joiner=joiner)[30:]
        with (OUT / f"{name.replace(' ', '_')}.jsonl").open("w") as f:
            for s in streams:
                rec = run_stream(reader, s, tok)
                f.write(json.dumps(rec) + "\n")
        print(f"{name}: {len(streams)} encounters done", flush=True)


def report():
    from head_of_state.decoders import probs
    from head_of_state.metrics import pool, stream_counts
    from head_of_state.settle import online_labels, settled_labels
    L = aci_bench.SEG_LEN
    print("| Text | token acc. (settled) | boundary F1 ±8 | ±16 | switches within 16 | spurious /1k | time to notice |\n"
          "|---|---|---|---|---|---|---|")
    for name in VERSIONS:
        f = OUT / f"{name.replace(' ', '_')}.jsonl"
        if not f.exists():
            continue
        recs = [json.loads(l) for l in f.open()]
        settled, online, tokens = [], [], 0
        for r in recs:
            p, pos = probs(r, rotations=1)
            settled.append(stream_counts(settled_labels(p, pos, L, int(round(0.2 * L))), p, r["gold"], pos))
            online.append(stream_counts(online_labels(p, pos, L), p, r["gold"], pos))
            tokens += r["n_tokens"]
        m, o = pool(settled, tokens), pool(online, tokens)
        found = sum(c["match"][16] for c in settled)
        true = sum(c["true_b"] for c in settled)
        spur = sum(c["pred_b"] - c["match"][16] for c in settled)
        print(f"| {name} ({len(recs)} encounters, {tokens} tokens) | {m['accuracy']:.3f} | {m['boundary_f1@8']:.2f} | "
              f"{m['boundary_f1@16']:.2f} | {found}/{true} | {1000 * spur / tokens:.1f} | {o['mean_delay']:.1f} |")


if __name__ == "__main__":
    {"run": run, "report": report}[sys.argv[1]]()
