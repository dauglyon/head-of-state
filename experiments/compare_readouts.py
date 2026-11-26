"""Design comparison runs (see head_of_state.readouts).

    uv run python -I experiments/compare_readouts.py E1:qbefore E1:gen E2:qafter ...

E1: the first 6 PubMed 20k RCT dev abstracts in the dev run's order (seed 0), S3 states and unit.
E2: 2 streams of about 1,200 tokens, each a sequence of merged sections drawn from different
    WikiSection en_disease validation (tuning) articles, no two neighbours in the same state.
Each job writes experiments/runs/compare/<set>_<design>.jsonl, one record per stream; finished streams are
skipped, so a stopped run resumes.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import RUNS  # noqa: E402
import pubmed_rct  # noqa: E402
import wikisection  # noqa: E402
from head_of_state.readouts import run_embed, run_gen, run_qafter, run_qbefore  # noqa: E402

OUT = RUNS / "compare"
MODEL = "Qwen/Qwen3.5-4B"
EMBED_MODEL = "BAAI/bge-m3"
# Sized to rank designs whose differences should be large; each job runs in minutes.
E1_N = 6
DESIGNS = {
    "qafter": lambda m, t, st, s, unit, seg: run_qafter(m, t, st, s, unit=unit, rotations=2),
    "qbefore": lambda m, t, st, s, unit, seg: run_qbefore(m, t, st, s, unit=unit, rotations=2),
    "gen": lambda m, t, st, s, unit, seg: run_gen(m, t, st, s),
    "genrefresh": lambda m, t, st, s, unit, seg: run_gen(m, t, st, s, every=64, keep=64),
}


def e1():
    streams = pubmed_rct.load("dev")
    random.Random(0).shuffle(streams)
    return streams[:E1_N], pubmed_rct.STATES, pubmed_rct.UNIT, 30.0


def e2(tok, n=2, target_tokens=1200, seed=0):
    docs = wikisection.load(wikisection.TUNING_SPLIT)
    pool = [(d["text"][st:en].strip(), g, d["id"]) for d in docs for st, en, g in d["spans"]]
    pool = [x for x in pool if x[0] and len(x[0]) <= 2500]   # about 600 tokens at most
    rng = random.Random(seed)
    streams = []
    for k in range(n):
        text, spans, used, last, ntok = "", [], set(), None, 0
        while ntok < target_tokens:
            sec, g, doc = rng.choice(pool)
            if g == last or doc in used:
                continue
            if text:
                text += "\n"
            spans.append((len(text), len(text) + len(sec), g))
            text += sec
            used.add(doc)
            last = g
            ntok += len(tok(sec, add_special_tokens=False)["input_ids"])
        streams.append({"id": f"E2-{k}", "text": text, "spans": spans})
    return streams, wikisection.STATES, wikisection.UNIT, 190.5


def main():
    jobs = [j.split(":") for j in sys.argv[1:]]
    import torch
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    sets = {"E1": e1(), "E2": e2(tok)}
    model = emb = None
    OUT.mkdir(parents=True, exist_ok=True)
    for set_name, design in jobs:
        streams, states, unit, seg = sets[set_name]
        out = OUT / f"{set_name}_{design}.jsonl"
        done = {json.loads(l)["id"] for l in out.open()} if out.exists() else set()
        for s in streams:
            if s["id"] in done:
                continue
            if design == "embed":
                if emb is None:
                    emb_tok = AutoTokenizer.from_pretrained(EMBED_MODEL)
                    emb = AutoModel.from_pretrained(EMBED_MODEL, dtype=torch.float16).to("cuda").eval()
                window = 32 if set_name == "E1" else 192     # about one declared segment length
                rec = run_embed(emb, emb_tok, tok, states, s, window=window)
            else:
                if model is None:
                    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
                rec = DESIGNS[design](model, tok, states, s, unit, seg)
            rec.update(id=s["id"], design=design, set=set_name, seg_len=seg)
            with out.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"{set_name}:{design} {s['id']}: {rec['n_tokens']} tokens, "
                  f"{1000 * rec['seconds'] / rec['n_tokens']:.0f} ms/token", flush=True)


if __name__ == "__main__":
    main()
