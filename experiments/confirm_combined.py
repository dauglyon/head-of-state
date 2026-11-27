"""Confirmation on unread documents: the question readout counted in full, counted fractionally, and fractional plus
next-token evidence from one cache per state.

    H2O_LIGHTNING_PATH=<checkout> uv run python -I experiments/confirm_combined.py run --set news|pubmed
    uv run python -I experiments/confirm_combined.py report            # results/confirm_combined.md

Declared before any of these documents was read (docs/LOG.md, 2026-11-27):
  documents  news: the 12 seed-3 streams of news_confirm.py (AG News test articles in neither the seed-0 nor the seed-2
             streams); PubMed: test abstracts 61-100 of the seed-0 shuffle of scored_runs.s3_test (1-60 used before).
  readouts   H2O-Lightning-4B; the question readout with the frozen settings (jev_screen.py: unit "the last few
             sentences", "the last sentence" for PubMed); one cache per state (head_of_state.next_token.PerStateCacheReader,
             the owner's opening).
  decodes    Viterbi with the declared length (news 270, PubMed 30) on: (1) the question readout's log readings, (2) the
             same counted fractionally (tied_target_hmm.fit, weight from the stream's own autocorrelation), (3) (2) plus
             the cache log-likelihoods. combine_readouts.py's functions.
  comparisons (a) 2 against 1, (b) 3 against 2: token accuracy paired by document and by gold segment; false switches per
             1,000 tokens; switches placed within 16 tokens.
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import joined_documents as jd  # noqa: E402
import news_confirm  # noqa: E402
import pubmed_rct  # noqa: E402
from common import RUNS  # noqa: E402
from scored_runs import FROZEN  # noqa: E402

OUT = RUNS / "confirm_combined"
DECLARED = {"news": 270.0, "pubmed": 30.0}


def documents(name, tok):
    if name == "news":
        return jd.LISTS["ag_news"], news_confirm.streams(tok)
    s = pubmed_rct.load("test")
    random.Random(0).shuffle(s)
    return pubmed_rct.STATES, s[60:100]


def run(name):
    import time
    import torch
    from transformers import AutoModelForCausalLM
    from head_of_state.h2o import H2OReader
    from head_of_state.harness import run_stream, tokens_and_gold
    from head_of_state.next_token import PerStateCacheReader
    tok = jd.tokenizer()
    model = AutoModelForCausalLM.from_pretrained(os.environ.get("H2O_LIGHTNING_PATH", "h2oai/h2o-lightning-4b"),
                                                 dtype=torch.bfloat16, device_map="cuda")
    states, docs = documents(name, tok)
    jev = H2OReader(model, tok, states, unit=pubmed_rct.UNIT if name == "pubmed" else jd.UNIT, **FROZEN)
    cache = PerStateCacheReader(model, tok, states)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / f"{name}_jev.jsonl").open("w") as fj, (OUT / f"{name}_cache.jsonl").open("w") as fc:
        for d in docs:
            fj.write(json.dumps(dict(run_stream(jev, d, tok), set=name)) + "\n"); fj.flush()
            ids, gold = tokens_and_gold(tok, d["text"], d["spans"])
            t0 = time.time()
            lp, plain = cache.emissions(ids)
            fc.write(json.dumps({"set": name, "id": d["id"], "n_tokens": len(ids), "gold": gold, "state_logp": lp,
                                 "plain_logp": plain, "seconds": time.time() - t0}) + "\n"); fc.flush()
        print(f"{name}: {len(docs)} documents", flush=True)


def report():
    from combine_readouts import fractional, rows
    from head_of_state.metrics import pool, stream_counts
    from head_of_state.settle import settled_labels
    print("Unread documents; decodes and comparisons declared beforehand. Cells: token accuracy · switches placed within "
          "16 tokens · false switches per 1,000 tokens.\n")
    print("| Set | documents | (1) question readout | (2) fractional | (3) fractional + per-state caches |\n|---|---|---|---|---|")
    for name, L in DECLARED.items():
        fj, fc = OUT / f"{name}_jev.jsonl", OUT / f"{name}_cache.jsonl"
        if not (fj.exists() and fc.exists()):
            continue
        J = {r["id"]: r for r in map(json.loads, fj.open())}
        C = {r["id"]: r for r in map(json.loads, fc.open())}
        ids = [i for i in J if i in C]
        if not ids:
            continue
        K = len(jd.LISTS["ag_news"]) if name == "news" else len(pubmed_rct.STATES)
        cs = {1: [], 2: [], 3: []}; tokens = 0
        for i in ids:
            g = C[i]["gold"]; pos = np.arange(1, len(g) + 1); tokens += len(g)
            q = rows("jev", J[i], K); c = rows("cache", C[i], K)
            for k, lp in ((1, q), (2, fractional(q, L)), (3, fractional(q, L) + c)):
                p = np.exp(lp - lp.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
                cs[k].append(stream_counts(settled_labels(p, pos, L), p, g, pos))
        cell = lambda c: (f"{pool(c, tokens)['accuracy']:.3f} · {sum(x['match'][16] for x in c)}/{sum(x['true_b'] for x in c)}"
                          f" · {1000 * sum(x['pred_b'] - x['match'][16] for x in c) / tokens:.1f}")
        print(f"| {name} | {len(ids)} | {cell(cs[1])} | {cell(cs[2])} | {cell(cs[3])} |")


if __name__ == "__main__":
    a = sys.argv
    if a[1] == "run":
        run(a[a.index("--set") + 1])
    else:
        report()
