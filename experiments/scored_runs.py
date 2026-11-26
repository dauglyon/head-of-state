"""First scored (non-tuning) runs, made with the earlier frozen setting (a reading every 8th token).

    uv run python -I experiments/scored_runs.py run      # GPU
    uv run python -I experiments/scored_runs.py report   # CPU: results/scored_runs.md

Frozen settings (fixed 2026-11-25 from the dev and validation runs in docs/LOG.md): 1 option ordering,
a reading every 8th token, persist 1, prior-power 0, max_ctx 8192, keep_ctx 2048; decoders reported:
persist 1 and the forward filter with eta = 1. Question frame unchanged.
- S3: 30 PubMed 20k RCT test abstracts (seed 0); declared length 30; unit "the last sentence".
- S2: 4 null streams of about 3,000 tokens, each one merged state (symptom, cause, diagnosis,
  treatment) from different WikiSection en_disease test articles; declared length 190.5; false
  switches per 1k tokens.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pubmed_rct  # noqa: E402
import wikisection  # noqa: E402
from common import MODEL, RUNS  # noqa: E402

OUT = RUNS / "scored"
FROZEN = dict(rotations=1, stride=1, persist=3, prior_power=0.0, max_ctx=8192, keep_ctx=2048)
STRIDE8 = dict(FROZEN, stride=8, persist=1)   # the earlier frozen setting, kept as an experiment
NULL_STATES = ["symptom", "cause", "diagnosis", "treatment"]


def s3_test():
    s = pubmed_rct.load("test")
    random.Random(0).shuffle(s)
    return s[:30]


def s2_null(tok, target=3000, seed=0):
    docs = wikisection.load("test")
    rng, streams = random.Random(seed), []
    for name in NULL_STATES:
        g = wikisection.NAMES.index(name)
        secs = [d["text"][a:b].strip() for d in docs for a, b, gg in d["spans"] if gg == g]
        rng.shuffle(secs)
        text, n = "", 0
        for sec in secs:
            if not sec:
                continue
            text = (text + "\n" + sec) if text else sec
            n += len(tok(sec, add_special_tokens=False)["input_ids"])
            if n >= target:
                break
        streams.append({"id": f"S2-{name}", "text": text, "spans": [(0, len(text), g)]})
    return streams


def run():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from head_of_state.harness import UnitReader, run_stream
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
    OUT.mkdir(parents=True, exist_ok=True)
    for name, streams, states, unit in (("S3test", s3_test(), pubmed_rct.STATES, pubmed_rct.UNIT),
                                        ("S2null", s2_null(tok), wikisection.STATES, wikisection.UNIT)):
        reader = UnitReader(model, tok, states, unit=unit, **STRIDE8)
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
    print("| Set | decoder | token acc. | boundary F1 ±8 | ±16 | matched | mean delay | spurious /1k | Brier | ECE | ms/token |\n"
          "|---|---|---|---|---|---|---|---|---|---|---|")
    for name, L in (("S3test", 30.0), ("S2null", 190.5)):
        recs = [json.loads(l) for l in (OUT / f"{name}.jsonl").open()]
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
            print(f"| {name} ({len(recs)} streams, {tokens} tokens) | {dname} | {m['accuracy']:.3f} | "
                  f"{m['boundary_f1@8']:.2f} | {m['boundary_f1@16']:.2f} | {m['matched']} | {m['mean_delay']:.1f} | "
                  f"{m['spurious_per_1k']:.1f} | {m['brier']:.3f} | {m['ece']:.3f} | {ms:.0f} |")


if __name__ == "__main__":
    {"run": run, "report": report}[sys.argv[1]]()
