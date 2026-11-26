"""S3: PubMed 20k RCT abstracts, one stream per abstract (docs/TEST_SUITE.md).

    uv run python -I experiments/pubmed_rct.py measure                 # D1 declared length from dev
    uv run python -I experiments/pubmed_rct.py run --split dev --n 40 --rotations 5 --out experiments/runs/s3_dev_r5.jsonl

Data: data/pubmed-rct/{train,dev,test}.txt from github.com/Franck-Dernoncourt/pubmed-rct
(PubMed_20k_RCT, numbers not replaced). Each sentence is a span; sentences are joined with one space.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
from head_of_state.harness import UnitReader, run_all  # noqa: E402

DATA = common.DATA / "pubmed-rct"
MODEL = "Qwen/Qwen3.5-4B"
LABELS = ["BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS"]
# Descriptions written for this evaluation (not taken from the dataset).
STATES = {
    "background": "what is already known about the problem",
    "objective": "the aim or question the study sets out to answer",
    "methods": "how the study was done: participants, design, interventions and measurements",
    "results": "what the study found, with numbers and statistical comparisons",
    "conclusions": "what the authors conclude from the findings and what it implies",
}
UNIT = "the last sentence"   # H4 / pre-registration table


def load(split: str) -> list[dict]:
    streams, cur = [], None
    for line in (DATA / f"{split}.txt").open(encoding="utf-8"):
        line = line.rstrip("\n")
        if line.startswith("###"):
            cur = {"id": line[3:], "sents": []}
            streams.append(cur)
        elif line.strip():
            label, sent = line.split("\t", 1)
            cur["sents"].append((LABELS.index(label), sent.strip()))
    out = []
    for s in streams:
        text, spans = "", []
        for k, sent in s["sents"]:
            if text:
                text += " "
            spans.append((len(text), len(text) + len(sent), k))
            text += sent
        out.append({"id": s["id"], "text": text, "spans": spans})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["measure", "run"])
    ap.add_argument("--split", default="dev")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rotations", type=int, default=5)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)

    if a.cmd == "measure":
        lens = [len(tok(" " + s["text"][st:en], add_special_tokens=False)["input_ids"])
                for s in load("dev") for st, en, _ in s["spans"]]
        print(json.dumps({"set": "S3", "declared_segment_length_tokens": statistics.median(lens),
                          "source": "median sentence length, PubMed 20k RCT dev, Qwen3.5-4B tokenizer",
                          "n_sentences": len(lens)}))
        return

    import torch
    from transformers import AutoModelForCausalLM
    streams = load(a.split)
    random.Random(a.seed).shuffle(streams)
    streams = streams[: a.n]
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
    reader = UnitReader(model, tok, STATES, rotations=a.rotations, unit=UNIT)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    run_all(reader, streams, tok, a.out)


if __name__ == "__main__":
    main()
