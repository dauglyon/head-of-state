"""Does the question ("the ask") limit accuracy? Four ask variants on tuning splits only.

    H2O_LIGHTNING_PATH=<checkout> uv run python -I experiments/ask_variants.py run
    uv run python -I experiments/ask_variants.py report     # results/ask_variants.md

Variants (H2O-Lightning-4B, 1 option ordering, a reading every 2nd token; judged on settled labels with
time to notice, as in docs/TEST_SUITE.md):
  V0  current descriptions, current unit wording
  V1  contrastive descriptions (each says what separates it from its neighbours), current unit
  V2  current descriptions, unit "the current sentence" (conversations: "the current line")
  V3  contrastive descriptions and the narrower unit
Sets (tuning data, never held out), small enough to rank arms in minutes: the first 3 PubMed 20k RCT dev
abstracts (seed-0 order); the first WikiSection en_disease validation stream of compare_readouts.e2; the
first two-speaker ACI-Bench train encounter. Readings every 4th token. The contrastive descriptions were written for this experiment.
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

OUT = RUNS / "ask_small"
STRIDE = 4

CONTRASTIVE = {
    "pubmed": {
        "background": "context before the study: what is known about the disease or problem and why it matters; no details of this study yet",
        "objective": "the aim of this study: what it set out to test, assess or determine",
        "methods": "how this study was done: design, participants, randomisation, interventions, outcomes measured, statistics; no results yet",
        "results": "what this study found: numbers, percentages, p-values, confidence intervals, comparisons between groups",
        "conclusions": "the authors' interpretation of the findings and their implications, after the results",
    },
    "wikisection": {
        "symptom": "what a patient experiences or shows: signs and symptoms, how the illness feels or presents",
        "cause": "why the disease happens: causes, risk factors, genes, infection and how it spreads",
        "diagnosis": "how doctors identify or classify it: tests, imaging, criteria, types and stages",
        "prevention": "how to avoid getting it: vaccines, screening, hygiene, lifestyle measures",
        "treatment": "how it is treated or managed: drugs, surgery, therapy, care",
        "mechanism": "what happens inside the body: biology and pathology at tissue or cell level",
        "epidemiology": "how many people get it and who: rates, ages, sexes, regions, populations",
        "history": "its past: discovery, historical cases, and where the name comes from",
        "prognosis": "what happens to patients over time: outcomes, survival, long-term complications",
        "society and research": "the disease in culture, society and the economy, and current research on it",
    },
    "aci": {
        "doctor": "the clinician speaking: asks questions, gives instructions, explains findings, the diagnosis or the plan",
        "patient": "the patient speaking: answers questions, describes their own symptoms, feelings, history and daily life, "
                   "often in first person or with short replies such as 'yes' or 'okay'",
    },
}
NARROW_UNIT = {"pubmed": "the current sentence", "wikisection": "the current sentence", "aci": "the current line"}

# Research-based variants (docs/LOG.md, "How Jev-class models should be asked"): the text to judge as its own
# part of the record with a direct question (R1); plus neutral option names (R2).
SPAN = {"pubmed": "sentence", "wikisection": "sentence", "aci": "line"}



DOMAIN_QUESTION = {   # R7: the exact condition, in the domain's own terms
    "pubmed": "Which section of a randomized controlled trial abstract is the current sentence from?",
    "wikisection": "Which section of an encyclopedia article about a disease is the current sentence from?",
    "aci": "Who is speaking in the current line of this doctor-patient conversation?",
}


def examples(key, tok, n=2, seed=0):
    """R9: two real examples per state from the train split (never from the tuning streams): sentences of
    8-30 words for PubMed and WikiSection, lines of 2-25 words from ACI-Bench train encounters 11 and up."""
    import re
    rng = random.Random(seed)
    pool = {}
    if key == "pubmed":
        for s in pubmed_rct.load("train")[:2000]:
            for a, b, g in s["spans"]:
                pool.setdefault(list(pubmed_rct.STATES)[g], []).append(s["text"][a:b])
    elif key == "wikisection":
        for d in wikisection.load("train")[:800]:
            for a, b, g in d["spans"]:
                first = re.split(r"(?<=[.!?])\s", d["text"][a:b].strip())[0]
                pool.setdefault(wikisection.NAMES[g], []).append(first)
    else:
        for s in aci_bench.load(split="train", n=40)[10:]:
            for a, b, g in s["spans"]:
                pool.setdefault(list(aci_bench.STATES)[g], []).extend(l for l in s["text"][a:b].split("\n") if l.strip())
    lo, hi = (2, 25) if key == "aci" else (8, 30)
    out = {}
    for name, items in pool.items():
        ok = [x.strip() for x in items if lo <= len(x.split()) <= hi]
        out[name] = rng.sample(ok, n)
    return out


def research_variants(key, states, tok=None, seg_len=None):
    q = f"Which option describes the current {SPAN[key]}?"
    neutral = [f"s{i + 1}" for i in range(len(states))]
    base = dict(span_kind=SPAN[key], question_text=q)
    arms = {"R1": (states, base), "R2": (states, dict(base, option_names=neutral))}
    if tok is not None:
        ex = examples(key, tok)
        arms.update({
            "R4": (states, dict(base, rotations=2)),
            "R5": (states, dict(base, record_format="json")),
            "R6": (states, dict(base, decompose=True)),
            "R7": (states, dict(base, question_text=DOMAIN_QUESTION[key])),
            "R8": (states, dict(base, keep_ctx=int(round(2 * seg_len)))),
            "R9": ({k: {"description": v, "examples": ex[k]} for k, v in states.items()}, base),
        })
    return arms


def sets(tok):
    s3 = pubmed_rct.load("dev")
    random.Random(0).shuffle(s3)
    ws, _, _, _ = compare_readouts.e2(tok)
    return {"pubmed": (s3[:3], pubmed_rct.STATES, pubmed_rct.UNIT, 30.0),
            "wikisection": (ws[:1], wikisection.STATES, wikisection.UNIT, 190.5),
            "aci": (aci_bench.load(split="train", n=1), aci_bench.STATES, aci_bench.UNIT, aci_bench.SEG_LEN)}


def variants(key, states, unit):
    return {"V0": (states, unit), "V1": (CONTRASTIVE[key], unit),
            "V2": (states, NARROW_UNIT[key]), "V3": (CONTRASTIVE[key], NARROW_UNIT[key])}


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
    for key, (streams, states, unit, L) in sets(tok).items():
        for v, (st, un) in variants(key, states, unit).items():
            assert list(st) == list(states)
            reader = H2OReader(model, tok, st, unit=un, rotations=1, stride=STRIDE, persist=1)
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
    print("| Set | variant | token acc. (settled) | boundary F1 ±8 | ±16 | switches within 16 | spurious /1k | "
          "time to notice |\n|---|---|---|---|---|---|---|---|")
    names = {"V0": "V0 current", "V1": "V1 contrastive descriptions", "V2": "V2 narrower unit", "V3": "V3 both",
             "R1": "R1 current sentence/line as its own field, direct question", "R2": "R2 = R1 + neutral option names",
             "R4": "R4 = R1 + 2 option orders", "R5": "R5 = R1 as a JSON record", "R6": "R6 = R1 as one yes/no per state",
             "R7": "R7 = R1 + domain question", "R8": "R8 = R1 + recent text only (2x length)",
             "R9": "R9 = R1 + 2 real examples per state (few-shot reference)"}
    for key in ("pubmed", "wikisection", "aci"):
        for v in ("V0", "R1", "R2", "R4", "R5", "R6", "R7", "R8", "R9"):
            f = OUT / f"{key}_{v}.jsonl"
            if not f.exists():
                continue
            recs = [json.loads(l) for l in f.open()]
            settled, online, tokens = [], [], 0
            for r in recs:
                p, pos = probs(r, rotations=len(r.get("shifts", [0])))   # all saved option orders
                L = r["seg_len"]
                settled.append(stream_counts(settled_labels(p, pos, L, int(round(0.2 * L))), p, r["gold"], pos))
                online.append(stream_counts(online_labels(p, pos, L), p, r["gold"], pos))
                tokens += r["n_tokens"]
            m, o = pool(settled, tokens), pool(online, tokens)
            found = sum(c["match"][16] for c in settled)
            true = sum(c["true_b"] for c in settled)
            spur = sum(c["pred_b"] - c["match"][16] for c in settled)
            notice = f"{o['mean_delay']:.1f}" if o["mean_delay"] == o["mean_delay"] else "—"
            print(f"| {key} ({len(recs)} streams, {tokens} tokens) | {names[v]} | {m['accuracy']:.3f} | "
                  f"{m['boundary_f1@8']:.2f} | {m['boundary_f1@16']:.2f} | {found}/{true} | {1000 * spur / tokens:.1f} | {notice} |")


def run_research():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from head_of_state.h2o import TargetedH2OReader
    from head_of_state.harness import run_stream
    path = os.environ.get("H2O_LIGHTNING_PATH", "h2oai/h2o-lightning-4b")
    tok_path = os.path.join(path, "tokenizer") if os.path.isdir(os.path.join(path, "tokenizer")) else path
    tok = AutoTokenizer.from_pretrained(tok_path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda")
    OUT.mkdir(parents=True, exist_ok=True)
    only = sys.argv[2:]                        # optional arm names
    for key, (streams, states, unit, L) in sets(tok).items():
        arms = {"V0": None, **research_variants(key, states, tok, L)}
        for v, arm in arms.items():
            if only and v not in only:
                continue
            if arm is None:                          # baseline: the current question, same samples and stride
                from head_of_state.h2o import H2OReader
                reader = H2OReader(model, tok, states, unit=unit, rotations=1, stride=STRIDE, persist=1)
                kw = {}
            else:
                st, kw = arm
                assert list(st) == list(states)
                kw = dict(kw)
                reader = TargetedH2OReader(model, tok, st, stride=STRIDE, persist=1, **{"rotations": 1, **kw})
            if "keep_ctx" in kw:                     # R8: re-anchor so the record holds only the recent text
                reader.max_ctx = len(reader.head_ids) + kw["keep_ctx"] + 600 + STRIDE
            with (OUT / f"{key}_{v}.jsonl").open("w") as f:
                for s in streams:
                    rec = run_stream(reader, s, tok)
                    rec["seg_len"] = L
                    f.write(json.dumps(rec) + "\n")
            print(f"{key} {v} done", flush=True)


if __name__ == "__main__":
    {"run": run, "run_research": run_research, "report": report}[sys.argv[1]]()
