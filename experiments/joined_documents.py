"""HANDOFF task 5: streams of labelled documents joined end to end, three unrelated state lists.

    uv run python -I experiments/joined_documents.py measure          # declared segment lengths (seed 1)
    H2O_LIGHTNING_PATH=<checkout> uv run python -I experiments/joined_documents.py run [list ...]
    uv run python -I experiments/joined_documents.py report           # results/joined_documents.md
    ... run --seed 2, report --seed 2: confirmation streams of documents not in the seed-0 streams

Each stream is a run of segments; a segment is one or more documents with the same label, joined with
single spaces, so every token has an exact true state and nothing in the formatting marks a switch.
Segment sizes are drawn from 64/128/256/512 tokens (whole documents, so a segment reaches at least its
size), and consecutive segments always differ in state, so time to notice can be read against the
length of the segment before the switch (HANDOFF open question 1).

State lists (descriptions written for this evaluation):
  ag_news      AG News test split: world / sports / business / science and technology
  wikisection  WikiSection en_disease train split, sections with the merged labels of wikisection.py
  imdb         IMDB test split: positive / negative reviews of at most 300 tokens
Evaluated streams use seed 0; the declared segment length of each list is the median segment length of
streams built with seed 1 (docs/TEST_SUITE.md, D1). Frozen settings: a reading at every token, 1 option
ordering, H2O-Lightning-4B; judged on settled labels, with time to notice from the online decision.
"""
from __future__ import annotations

import json
import os
import random
import re
import statistics
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import wikisection  # noqa: E402
from common import DATA, RUNS  # noqa: E402
from scored_runs import FROZEN  # noqa: E402

OUT = RUNS / "joined"
SIZES = (64, 128, 256, 512)
N_STREAMS, STREAM_TOKENS = 3, 2000
UNIT = "the last few sentences"
DECLARED = {"ag_news": 270.0, "wikisection": 441.0, "imdb": 274.0}   # `measure` (seed 1), fixed before any run

LISTS = {
    "ag_news": {
        "world": "international news: politics, conflicts, diplomacy and events in countries around the world",
        "sports": "sports news: games, teams, athletes, results and competitions",
        "business": "business news: companies, markets, the economy, earnings and deals",
        "science and technology": "science and technology news: computers, the internet, software, space and research",
    },
    "wikisection": wikisection.STATES,
    "imdb": {
        "positive": "a movie review in which the writer likes the film",
        "negative": "a movie review in which the writer dislikes the film",
    },
}


def documents(name, tok):
    """{state index: [document text]} for one list."""
    by = {}
    if name == "ag_news":
        for line in open(DATA / "ag-news" / "test.jsonl"):
            r = json.loads(line)
            by.setdefault(r["label"], []).append(re.sub(r"\s+", " ", r["text"].replace("\\", " ")).strip())
    elif name == "imdb":
        for line in open(DATA / "imdb" / "test.jsonl"):
            r = json.loads(line)
            t = re.sub(r"\s+", " ", re.sub(r"<br\s*/?>", " ", r["text"])).strip()
            if len(tok(t, add_special_tokens=False)["input_ids"]) <= 300:
                by.setdefault(1 - r["label"], []).append(t)        # dataset: 0 negative, 1 positive
    else:
        for d in wikisection.load("train"):
            for a, b, g in d["spans"]:
                t = re.sub(r"\s+", " ", d["text"][a:b]).strip()
                if t:
                    by.setdefault(g, []).append(t)
    return by


def build(name, tok, seed, n_streams=N_STREAMS, stream_tokens=STREAM_TOKENS, exclude=frozenset()):
    """Streams for one list. Documents in `exclude` are not used; every stream's `docs` lists its own."""
    rng = random.Random(seed)
    pools = {g: [d for d in rng.sample(v, len(v)) if d not in exclude] for g, v in documents(name, tok).items()}
    streams, seg_lens = [], []
    for i in range(n_streams):
        text, spans, n, prev, used = "", [], 0, None, []
        while n < stream_tokens:
            g = rng.choice([k for k in pools if k != prev and pools[k]])
            size, parts, m = rng.choice(SIZES), [], 0
            while m < size and pools[g]:
                doc = pools[g].pop()
                parts.append(doc); used.append(doc)
                m += len(tok(doc, add_special_tokens=False)["input_ids"])
            start = len(text) + (1 if text else 0)
            text = (text + " " + " ".join(parts)) if text else " ".join(parts)
            spans.append((start, len(text), g))
            seg_lens.append(m); n += m; prev = g
        streams.append({"id": f"{name}-{seed}-{i}", "text": text, "spans": spans, "docs": used})
    return streams, seg_lens


def tokenizer():
    from transformers import AutoTokenizer
    path = os.environ.get("H2O_LIGHTNING_PATH", "h2oai/h2o-lightning-4b")
    return AutoTokenizer.from_pretrained(os.path.join(path, "tokenizer") if os.path.isdir(os.path.join(path, "tokenizer")) else path)


def measure():
    tok = tokenizer()
    for name in LISTS:
        _, lens = build(name, tok, seed=1, n_streams=10)
        print(f"{name}: median segment length {statistics.median(lens)} tokens ({len(lens)} segments, seed 1)")


def streams_for(name, tok, seed):
    """Seed 0: the first streams. Any other seed: streams of documents not used in the seed-0 streams."""
    if seed == 0:
        return build(name, tok, seed=0)[0]
    used = {d for s in build(name, tok, seed=0)[0] for d in s["docs"]}
    return build(name, tok, seed=seed, exclude=used)[0]


def out_file(name, seed):
    return OUT / (f"{name}.jsonl" if seed == 0 else f"{name}_seed{seed}.jsonl")


def run(names, seed=0):
    import torch
    from transformers import AutoModelForCausalLM
    from head_of_state.h2o import H2OReader
    from head_of_state.harness import run_stream
    assert all(DECLARED[n] for n in names), "declare segment lengths first (measure, docs/TEST_SUITE.md)"
    tok = tokenizer()
    path = os.environ.get("H2O_LIGHTNING_PATH", "h2oai/h2o-lightning-4b")
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda")
    OUT.mkdir(parents=True, exist_ok=True)
    for name in names:
        streams = streams_for(name, tok, seed)
        reader = H2OReader(model, tok, LISTS[name], unit=UNIT, **FROZEN)
        n = sec = 0
        with out_file(name, seed).open("w") as f:
            for s in streams:
                rec = run_stream(reader, s, tok)
                n += rec["n_tokens"]; sec += rec["seconds"]
                f.write(json.dumps(rec) + "\n")
        print(f"{name} (seed {seed}): {len(streams)} streams, {n} tokens, {1000 * sec / n:.0f} ms/token", flush=True)


def notice_by_previous_length(online, positions, gold):
    """(length of the segment before the switch, tokens from the switch to the first reading whose online
    decision shows the new state, or None if it never does before the next switch)."""
    gold = np.asarray(gold)
    sw = [i for i in range(1, len(gold)) if gold[i] != gold[i - 1]]
    bounds = [0] + sw + [len(gold)]
    pos = np.asarray(positions)
    out = []
    for j, s in enumerate(sw):
        nxt = bounds[j + 2]
        hit = [p for p, lab in zip(pos, online) if s < p <= nxt and lab == gold[s]]
        out.append((s - bounds[j], (hit[0] - (s + 1)) if hit else None))
    return out


def report(seed=0):
    from head_of_state.decoders import probs
    from head_of_state.metrics import pool, stream_counts
    from head_of_state.settle import online_labels, settled_labels
    print("Settled labels: `0.2 L` moves each reading 0.2 x the declared length earlier (the pre-registered rule); `none` "
          "uses the readings where they were taken.\n\n| State list | streams (tokens) | reading shift | token acc. (settled) | "
          "always the most common state | boundary F1 ±16 | switches placed within 16 | spurious /1k | time to notice | "
          "ms/token |\n|---|---|---|---|---|---|---|---|---|---|")
    lag = {}
    for name in LISTS:
        f = out_file(name, seed)
        if not f.exists():
            continue
        L = DECLARED[name]
        recs = [json.loads(l) for l in f.open()]
        if not recs:
            continue
        settled, unshifted, online, tokens = [], [], [], 0
        for r in recs:
            p, pos = probs(r, rotations=1)
            on = online_labels(p, pos, L)
            settled.append(stream_counts(settled_labels(p, pos, L, int(round(0.2 * L))), p, r["gold"], pos))
            unshifted.append(stream_counts(settled_labels(p, pos, L, 0), p, r["gold"], pos))
            online.append(stream_counts(on, p, r["gold"], pos))
            tokens += r["n_tokens"]
            for prev_len, d in notice_by_previous_length(on, pos, r["gold"]):
                lag.setdefault(name, []).append((prev_len, d))
        o = pool(online, tokens)
        gold = np.concatenate([r["gold"] for r in recs])
        base = np.bincount(gold).max() / len(gold)
        ms = 1000 * sum(r["seconds"] for r in recs) / tokens
        for label, counts in ((f"0.2 L = {int(round(0.2 * L))}", settled), ("none", unshifted)):
            m = pool(counts, tokens)
            found = sum(c["match"][16] for c in counts)
            true = sum(c["true_b"] for c in counts)
            spur = sum(c["pred_b"] - c["match"][16] for c in counts)
            print(f"| {name} ({len(LISTS[name])} states) | {len(recs)} ({tokens}) | {label} | {m['accuracy']:.3f} | {base:.3f} | "
                  f"{m['boundary_f1@16']:.2f} | {found}/{true} | {1000 * spur / tokens:.1f} | {o['mean_delay']:.1f} | {ms:.0f} |")
    print("\nTime to notice by the length of the segment before the switch (median tokens; never = not shown "
          "before the next switch)\n\n| State list | previous segment | switches | median time to notice | never |\n|---|---|---|---|---|")
    for name, rows in lag.items():
        for lo, hi in ((0, 128), (128, 256), (256, 512), (512, 10**9)):
            sel = [d for pl, d in rows if lo <= pl < hi]
            if not sel:
                continue
            seen = [d for d in sel if d is not None]
            med = f"{statistics.median(seen):.0f}" if seen else "—"
            print(f"| {name} | {lo}–{hi if hi < 10**9 else ''} tokens | {len(sel)} | {med} | {len(sel) - len(seen)} |")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "measure":
        measure()
    elif cmd == "run":
        args = sys.argv[2:]
        seed = int(args[args.index("--seed") + 1]) if "--seed" in args else 0
        names = [a for i, a in enumerate(args) if a in LISTS]
        run(names or list(LISTS), seed)
    else:
        report(int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 0)
