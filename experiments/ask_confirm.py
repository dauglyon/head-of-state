"""Second, larger tuning check of the targeted question (ask_variants.py, docs/LOG.md).

    H2O_LIGHTNING_PATH=<checkout> uv run python -I experiments/ask_confirm.py run
    uv run python -I experiments/ask_confirm.py report     # results/ask_confirm.md

Tuning data not used in the first comparison: PubMed 20k RCT dev abstracts 4-10 (seed-0 order) and
two-speaker ACI-Bench train encounters 2-4, comparing the current question (V0) with R2 (the current
sentence or line as its own part of the record, a direct question, neutral option names). For
WikiSection, whose states are multi-sentence sections, both validation streams of compare_readouts.e2,
comparing V0 with R2 aimed at the current passage (the last 3 sentences). A reading every 4th token;
judged on settled labels with time to notice, as in ask_variants.py.
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import aci_bench  # noqa: E402
import compare_readouts  # noqa: E402
import pubmed_rct  # noqa: E402
import wikisection  # noqa: E402
from common import RUNS  # noqa: E402

OUT = RUNS / "ask_confirm"
STRIDE = 4


def sets(tok):
    s3 = pubmed_rct.load("dev")
    random.Random(0).shuffle(s3)
    ws, _, _, _ = compare_readouts.e2(tok)
    return {"pubmed": (s3[3:10], pubmed_rct.STATES, pubmed_rct.UNIT, 30.0, dict(span_kind="sentence")),
            "aci": (aci_bench.load(split="train", n=4)[1:], aci_bench.STATES, aci_bench.UNIT, aci_bench.SEG_LEN,
                    dict(span_kind="line")),
            "wikisection": (ws, wikisection.STATES, wikisection.UNIT, 190.5, dict(span_kind="passage", span_sentences=3))}


def run():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from head_of_state.h2o import H2OReader, TargetedH2OReader
    from head_of_state.harness import run_stream
    path = os.environ.get("H2O_LIGHTNING_PATH", "h2oai/h2o-lightning-4b")
    tok_path = os.path.join(path, "tokenizer") if os.path.isdir(os.path.join(path, "tokenizer")) else path
    tok = AutoTokenizer.from_pretrained(tok_path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda")
    OUT.mkdir(parents=True, exist_ok=True)
    for key, (streams, states, unit, L, span) in sets(tok).items():
        word = {"sentence": "sentence", "line": "line", "passage": "passage"}[span["span_kind"]]
        neutral = [f"s{i + 1}" for i in range(len(states))]
        readers = {"V0": H2OReader(model, tok, states, unit=unit, rotations=1, stride=STRIDE, persist=1),
                   "R2": TargetedH2OReader(model, tok, states, rotations=1, stride=STRIDE, persist=1, option_names=neutral,
                                           question_text=f"Which option describes the current {word}?", **span)}
        for v, reader in readers.items():
            with (OUT / f"{key}_{v}.jsonl").open("w") as f:
                for s in streams:
                    rec = run_stream(reader, s, tok)
                    rec["seg_len"] = L
                    f.write(json.dumps(rec) + "\n")
            print(f"{key} {v} done", flush=True)


def report():
    from head_of_state.decoders import probs
    from head_of_state.metrics import pool, stream_counts
    from head_of_state.settle import online_labels, settled_labels
    print("| Set | question | token acc. (settled) | boundary F1 ±8 | ±16 | switches within 16 | spurious /1k | "
          "time to notice |\n|---|---|---|---|---|---|---|---|")
    label = {"V0": "current question", "R2": "R2: current sentence/line/passage as its own part, neutral names"}
    for key in ("pubmed", "aci", "wikisection"):
        for v in ("V0", "R2"):
            f = OUT / f"{key}_{v}.jsonl"
            if not f.exists():
                continue
            recs = [json.loads(l) for l in f.open()]
            settled, online, tokens = [], [], 0
            for r in recs:
                p, pos = probs(r, rotations=1)
                L = r["seg_len"]
                settled.append(stream_counts(settled_labels(p, pos, L, int(round(0.2 * L))), p, r["gold"], pos))
                online.append(stream_counts(online_labels(p, pos, L), p, r["gold"], pos))
                tokens += r["n_tokens"]
            m, o = pool(settled, tokens), pool(online, tokens)
            found = sum(c["match"][16] for c in settled)
            true = sum(c["true_b"] for c in settled)
            spur = sum(c["pred_b"] - c["match"][16] for c in settled)
            notice = f"{o['mean_delay']:.1f}" if o["mean_delay"] == o["mean_delay"] else "—"
            print(f"| {key} ({len(recs)} streams, {tokens} tokens) | {label[v]} | {m['accuracy']:.3f} | "
                  f"{m['boundary_f1@8']:.2f} | {m['boundary_f1@16']:.2f} | {found}/{true} | {1000 * spur / tokens:.1f} | {notice} |")


if __name__ == "__main__":
    {"run": run, "report": report}[sys.argv[1]]()
