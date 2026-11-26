"""S5: speaker roles in doctor-patient conversations (ACI-Bench, CC BY 4.0), docs/TEST_SUITE.md.

    H2O_LIGHTNING_PATH=<checkout> uv run python -I experiments/aci_bench.py run
    uv run python -I experiments/aci_bench.py report     # results/aci_bench.md

Data: data/aci-bench/*.csv from github.com/wyim/aci-bench (data/challenge_data). Speaker tags are removed
and turns are joined with one newline (H2); the tags are the gold states; adjacent turns by the same
speaker are fused. Encounters with a third speaker (patient_guest) are left out. Declared segment
length: 12 tokens (median turn length, train split, H2O-Lightning tokenizer). Unit: "the last turn".
Model: H2O-Lightning-4B (the default). Runs: the frozen settings (every 8th token, 1 ordering, persist
1) on the first 10 two-speaker encounters of clinicalnlp_taskB_test1; and, as a labelled non-frozen
comparison, a reading at every token on the first 3.
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
from common import RUNS  # noqa: E402
from scored_runs import STRIDE8  # noqa: E402

DATA = common.DATA / "aci-bench"
OUT = RUNS / "aci"
SEG_LEN = 12.0
UNIT = "the last turn"
# Descriptions written for this evaluation (not taken from the dataset).
STATES = {
    "doctor": "the clinician is speaking: asking questions, examining, explaining findings or the plan",
    "patient": "the patient is speaking: describing symptoms, history or concerns, or answering questions",
}


def load(split="clinicalnlp_taskB_test1", n=10, joiner="\n"):
    csv.field_size_limit(10**9)
    names = list(STATES)
    out = []
    for row in csv.DictReader(open(DATA / f"{split}.csv")):
        turns = [re.match(r"^\[([a-z_]+)\]\s*(.*)$", l) for l in row["dialogue"].split("\n")]
        turns = [(m.group(1), m.group(2).strip()) for m in turns if m]
        if any(t not in names for t, _ in turns):
            continue
        text, spans = "", []
        for who, said in turns:
            if not said:
                continue
            start = len(text) + (len(joiner) if text else 0)
            text = (text + joiner + said) if text else said
            g = names.index(who)
            if spans and spans[-1][2] == g:
                spans[-1] = (spans[-1][0], len(text), g)
            else:
                spans.append((start, len(text), g))
        out.append({"id": row["encounter_id"], "text": text, "spans": spans})
        if len(out) == n:
            break
    return out


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
    streams = load()
    for name, subset, settings in (("frozen", streams, STRIDE8),
                                   ("every_token", streams[:3], dict(STRIDE8, stride=1, persist=3))):
        reader = H2OReader(model, tok, STATES, unit=UNIT, **settings)
        n = sec = 0
        with (OUT / f"{name}.jsonl").open("w") as f:
            for s in subset:
                rec = run_stream(reader, s, tok)
                n += rec["n_tokens"]; sec += rec["seconds"]
                f.write(json.dumps(rec) + "\n")
        print(f"{name}: {len(subset)} encounters, {n} tokens, {1000 * sec / n:.1f} ms/token", flush=True)


def report():
    from head_of_state.decoders import decode_persist, probs
    from head_of_state.metrics import forward_posteriors, pool, stream_counts
    print("| Run | encounters (tokens) | decoder | token acc. | boundary F1 ±8 | ±16 | matched | mean delay | "
          "spurious /1k | Brier | ECE | ms/token |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, label, stride, persist in (("frozen", "frozen: every 8th token", 8, 1),
                                         ("every_token", "every token (not frozen)", 1, 3)):
        recs = [json.loads(l) for l in (OUT / f"{name}.jsonl").open()]
        first3 = {r["id"] for r in recs[:3]}
        for subset_name, subset in (("all", recs), ("first 3", [r for r in recs if r["id"] in first3])):
            if name == "every_token" and subset_name == "all":
                continue
            tokens = sum(r["n_tokens"] for r in subset)
            ms = 1000 * sum(r["seconds"] for r in subset) / tokens
            for dname in (f"persist {persist}", "forward filter η=1"):
                counts = []
                for r in subset:
                    p, pos = probs(r, rotations=1)
                    if dname.startswith("persist"):
                        q, lab = p, np.asarray(decode_persist(p, persist))
                    else:
                        q = forward_posteriors(p, 1.0, SEG_LEN, None, stride)
                        lab = q.argmax(1)
                    counts.append(stream_counts(lab, q, r["gold"], pos))
                m = pool(counts, tokens)
                print(f"| {label} | {len(subset)} ({tokens}) | {dname} | {m['accuracy']:.3f} | {m['boundary_f1@8']:.2f} | "
                      f"{m['boundary_f1@16']:.2f} | {m['matched']} | {m['mean_delay']:.1f} | {m['spurious_per_1k']:.1f} | "
                      f"{m['brier']:.3f} | {m['ece']:.3f} | {ms:.0f} |")


if __name__ == "__main__":
    {"run": run, "report": report}[sys.argv[1]]()
