"""The Jev-style question readout on the screening and tie-break documents (for combining readouts on the same tokens).

    H2O_LIGHTNING_PATH=<checkout> uv run python -I experiments/jev_screen.py run [--tiebreak]

The question readout with the frozen settings of joined_documents.py (H2OReader, every token, one ordering), unit "the
last few sentences" ("the last sentence" for PubMed, as in pubmed_rct.py). The news documents are the seed-0 joined
streams already read by joined_documents.py; this reads the article and the PubMed abstracts and writes all records,
news included, to experiments/runs/wording/jev__h2o-lightning-4b[__tiebreak].jsonl with a "set" field.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import joined_documents as jd  # noqa: E402
import pubmed_rct  # noqa: E402
import wording_screen as ws  # noqa: E402
from scored_runs import FROZEN  # noqa: E402


def run(which):
    import torch
    from transformers import AutoModelForCausalLM
    from head_of_state.h2o import H2OReader
    from head_of_state.harness import run_stream
    tok = jd.tokenizer()
    model = AutoModelForCausalLM.from_pretrained(os.environ.get("H2O_LIGHTNING_PATH", "h2oai/h2o-lightning-4b"),
                                                 dtype=torch.bfloat16, device_map="cuda")
    saved = {json.loads(l)["id"]: json.loads(l) for l in jd.out_file("ag_news", 0).open()}
    out = ws.OUT / ("jev__h2o-lightning-4b.jsonl" if which == "screen" else "jev__h2o-lightning-4b__tiebreak.jsonl")
    with out.open("w") as f:
        for s, (states, docs) in ws.screen_docs(tok, which).items():
            reader = H2OReader(model, tok, states, unit=pubmed_rct.UNIT if s == "pubmed" else jd.UNIT, **FROZEN)
            for d in docs:
                rec = saved[d["id"]] if s == "news" else run_stream(reader, d, tok)
                f.write(json.dumps(dict(rec, set=s)) + "\n"); f.flush()
            print(f"{s} done", flush=True)


if __name__ == "__main__":
    run("tiebreak" if "--tiebreak" in sys.argv else "screen")
