"""Confirmation of the news-topic result on streams nothing has read: next-token emissions against the question.

    H2O_LIGHTNING_PATH=<checkout> uv run python -I experiments/news_confirm.py run
    uv run python -I experiments/news_confirm.py report [--instruction]   # results/news_confirm.md

On the three seed-0 tuning streams the state-conditioned next-token emissions (head_of_state.next_token) beat the
Jev-style readout on news topics by 6.4 points of token accuracy (docs/LOG.md, 2026-11-26). Three readouts on the
same streams: the question readout with its frozen settings (joined_documents.py), the next-token instruction
form of stated_next_token.py, and the catalogue form (all states listed once at the start, each branch naming one
state). Declared before any confirmation data: primary comparison catalogue against the question readout;
secondary, instruction against the question readout and catalogue against instruction. Frozen otherwise:
H2O-Lightning-4B, weight 1, Viterbi with the declared length (270), the AG News descriptions. New streams: 12 of
about 1,000 tokens, built as joined_documents.py does with seed 3 from AG News test documents used in neither
the seed-0 nor the seed-2 streams. Both readouts read the same streams; the comparison is paired by stream and
by gold segment.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import joined_documents as jd  # noqa: E402
from scored_runs import FROZEN  # noqa: E402
from stated_next_token import emission_probs  # noqa: E402

NAME, SEED, N, TOKENS = "ag_news", 3, 12, 1000
OUT = jd.OUT / "news_confirm"


def streams(tok):
    used = {d for seed in (0, 2) for s in jd.streams_for(NAME, tok, seed) for d in s["docs"]}
    return jd.build(NAME, tok, seed=SEED, n_streams=N, stream_tokens=TOKENS, exclude=used)[0]


def run():
    import torch
    from transformers import AutoModelForCausalLM
    from head_of_state.h2o import H2OReader
    from head_of_state.harness import run_stream, tokens_and_gold
    from head_of_state.next_token import StatedNextTokenReader
    tok = jd.tokenizer()
    path = os.environ.get("H2O_LIGHTNING_PATH", "h2oai/h2o-lightning-4b")
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda")
    OUT.mkdir(parents=True, exist_ok=True)
    ss = streams(tok)
    states = jd.LISTS[NAME]
    question = H2OReader(model, tok, states, unit=jd.UNIT, **FROZEN)
    described = {k: "about " + v for k, v in states.items()}
    stated = {"stated": StatedNextTokenReader(model, tok, described),
              "catalogue": StatedNextTokenReader(model, tok, described, form="catalogue")}
    import time
    files = {k: (OUT / f"{k}.jsonl").open("w") for k in ("question", "stated", "catalogue")}
    for s in ss:
        files["question"].write(json.dumps(run_stream(question, s, tok)) + "\n"); files["question"].flush()
        ids, gold = tokens_and_gold(tok, s["text"], s["spans"])
        for k, reader in stated.items():
            t0 = time.time()
            lp, plain = reader.emissions(ids)
            files[k].write(json.dumps({"list": NAME, "id": s["id"], "n_tokens": len(ids), "gold": gold, "state_logp": lp,
                                       "plain_logp": plain, "seconds": time.time() - t0}) + "\n"); files[k].flush()
        print(f"{s['id']}: {len(ids)} tokens", flush=True)


def report():
    from head_of_state.decoders import probs
    from head_of_state.metrics import pool, stream_counts
    from head_of_state.settle import online_labels, settled_labels
    L, K = jd.DECLARED[NAME], len(jd.LISTS[NAME])
    q = {r["id"]: r for r in map(json.loads, (OUT / "question.jsonl").open())}
    which = "stated" if "--instruction" in sys.argv else "catalogue"     # primary: catalogue against the question
    s = {r["id"]: r for r in map(json.loads, (OUT / f"{which}.jsonl").open())}
    ids = [i for i in s if i in q]
    rows, seg = {"question": [], "stated": []}, []          # seg: (tokens, correct question, correct stated)
    per_stream = []
    for i in ids:
        gold = np.asarray(s[i]["gold"])
        pq, posq = probs(q[i], rotations=1)
        ps, poss = emission_probs(s[i], K)
        lq, ls = settled_labels(pq, posq, L), settled_labels(ps, poss, L)
        rows["question"].append((stream_counts(lq, pq, gold, posq), stream_counts(online_labels(pq, posq, L), pq, gold, posq)))
        rows["stated"].append((stream_counts(ls, ps, gold, poss), stream_counts(online_labels(ps, poss, L), ps, gold, poss)))
        per_stream.append(((lq == gold).mean(), (ls == gold).mean(), len(gold)))
        start = 0
        for t in range(1, len(gold) + 1):
            if t == len(gold) or gold[t] != gold[start]:
                seg.append((t - start, (lq[start:t] == gold[start:t]).sum(), (ls[start:t] == gold[start:t]).sum()))
                start = t
    tokens = sum(n for *_, n in per_stream)
    print(f"{len(ids)} new streams ({tokens} tokens), news topics. Settled labels: Viterbi, weight 1, declared length.\n")
    print("| Readout | token acc. | placed within 16 | false /1k | time to notice | ms/token |\n|---|---|---|---|---|---|")
    for k, label, recs in (("question", "Jev-style question", q), ("stated", "state-conditioned next token", s)):
        m = pool([c for c, _ in rows[k]], tokens); o = pool([c for _, c in rows[k]], tokens)
        placed = sum(c["match"][16] for c, _ in rows[k]); true = sum(c["true_b"] for c, _ in rows[k])
        false = sum(c["pred_b"] - c["match"][16] for c, _ in rows[k])
        ms = 1000 * sum(recs[i]["seconds"] for i in ids) / tokens
        print(f"| {label} | {m['accuracy']:.3f} | {placed}/{true} | {1000 * false / tokens:.1f} | {o['mean_delay']:.0f} | {ms:.0f} |")
    seg = np.asarray(seg, float)
    st = np.asarray(per_stream, float)
    rng = np.random.default_rng(0)
    diff = lambda a: (a[:, 2].sum() - a[:, 1].sum()) / a[:, 0].sum()
    bs = [diff(seg[rng.integers(0, len(seg), len(seg))]) for _ in range(10000)]
    bt = [((st[b, 1] - st[b, 0]) * st[b, 2]).sum() / st[b, 2].sum()
          for b in (rng.integers(0, len(st), len(st)) for _ in range(10000))]
    d = (st[:, 1] - st[:, 0]) * st[:, 2]
    flips = np.array([(d * rng.choice([-1, 1], len(d))).sum() for _ in range(20000)])
    print(f"\nDifference in token accuracy (next token minus question): {diff(seg):+.3f}; 95% interval resampling "
          f"{len(seg)} segments [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}], resampling {len(st)} "
          f"streams [{np.percentile(bt, 2.5):+.3f}, {np.percentile(bt, 97.5):+.3f}]; streams better / worse "
          f"{(d > 0).sum()} / {(d < 0).sum()}; sign-flip p over streams {np.mean(np.abs(flips) >= abs(d.sum())):.3f}.")


if __name__ == "__main__":
    {"run": run, "report": report}[sys.argv[1]]()
