"""S1 lag streams (docs/TEST_SUITE.md S1, HANDOFF Q1) and the question-method runner used on them.

    uv run python -I experiments/s1_lag.py question [W]     # GPU: question method; W = hopped window or none
    uv run python -I experiments/s1_lag.py likelihood W     # GPU: windowed likelihood emissions

Streams: previous segment (symptom) of 50 / 200 / 800 / 3000 tokens, then a treatment segment of
about 300 tokens; 3 streams per length. The previous segment joins symptom sections of different
WikiSection en_disease test-split articles (the same join pattern at every length) and is cut at
the first sentence end at or after the target length; the following segment is one treatment
section cut the same way. States: all 10 merged groups. Not tuning data.
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import RUNS  # noqa: E402
import wikisection  # noqa: E402
from head_of_state.harness import UnitReader, tokens_and_gold  # noqa: E402

OUT = RUNS / "s1lag"
PREV = [50, 200, 800, 3000]
FOLLOW = 300
PAIR = ("symptom", "treatment")
SEG_LEN = 190.5
READ_BEFORE = 16


def _cut(tok, text, n_tokens):
    """Shortest prefix of `text` with at least n_tokens tokens that ends at a sentence end."""
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    if len(enc["input_ids"]) <= n_tokens:
        return text
    start = enc["offset_mapping"][n_tokens - 1][1]
    m = re.search(r"[.!?](\s|$)", text[start:])
    return text[: start + m.end()].rstrip() if m else text


def build(tok, n=3, seed=0):
    docs = wikisection.load("test")
    a, b = (wikisection.NAMES.index(x) for x in PAIR)
    secs = lambda g: [(d["id"], d["text"][s:e].strip()) for d in docs for s, e, gg in d["spans"] if gg == g]
    pool_a, pool_b = secs(a), secs(b)
    rng = random.Random(seed)
    streams = []
    for L in PREV:
        for i in range(n):
            rng.shuffle(pool_a)
            prev, used = "", set()
            for doc, sec in pool_a:
                if doc in used or not sec:
                    continue
                prev = (prev + "\n" + sec) if prev else sec
                used.add(doc)
                if len(tok(prev, add_special_tokens=False)["input_ids"]) >= L:
                    break
            prev = _cut(tok, prev, L)
            follow = next(sec for doc, sec in rng.sample(pool_b, len(pool_b))
                          if doc not in used and len(tok(sec, add_special_tokens=False)["input_ids"]) >= FOLLOW)
            follow = _cut(tok, follow, FOLLOW)
            text = prev + "\n" + follow
            spans = [(0, len(prev), a), (len(prev) + 1, len(text), b)]
            streams.append({"id": f"S1lag-{L}-{i}", "prev_len": L, "text": text, "spans": spans})
    return streams


def question_run(reader: UnitReader, ids, read_from: int, W: int | None):
    """Readings at every token from read_from on; earlier tokens fed in chunks without readings.
    W: hopped window — every W/2 tokens the cache is rebuilt from the opening text + last W tokens."""
    reader.reset()
    hop = max(1, W // 2) if W else None
    rows, pos = [], []
    for i, t in enumerate(ids):
        n = i + 1
        reader.stream.append(t)
        reader._pending.append(t)
        if hop and n % hop == 0:
            reader._pending = []
            reader.keep_ctx = W
            reader._reanchor()
        if n >= read_from:
            reader.flush()
            reader._read()
            rows.append(reader.last_shift_logp)
            pos.append(n)
    return rows, pos


def main():
    import time

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mode = sys.argv[1]
    W = None if len(sys.argv) < 3 or sys.argv[2] == "none" else int(sys.argv[2])
    tok = AutoTokenizer.from_pretrained(wikisection.MODEL)
    model = AutoModelForCausalLM.from_pretrained(wikisection.MODEL, dtype=torch.bfloat16, device_map="cuda")
    streams = build(tok)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{mode}_W{W or 'none'}.jsonl"
    reader = UnitReader(model, tok, wikisection.STATES, rotations=1, unit=wikisection.UNIT)
    with out.open("w") as f:
        for s in streams:
            ids, gold = tokens_and_gold(tok, s["text"], s["spans"])
            switch = gold.index(gold[-1]) + 1            # 1-based position of the first following token
            torch.cuda.synchronize(); t0 = time.time()
            if mode == "question":
                rows, pos = question_run(reader, ids, max(1, switch - READ_BEFORE), W)
                rec = {"shift_logp": rows, "positions": pos, "shifts": reader.shifts, "stride": 1}
            else:
                from head_of_state.window_likelihood import run_window
                rec = run_window(model, tok, wikisection.STATES, s, W)
                rec["kind"] = "emission"
            torch.cuda.synchronize()
            rec.update(id=s["id"], prev_len=s["prev_len"], gold=gold, n_tokens=len(ids), switch=switch,
                       seconds=time.time() - t0, seg_len=SEG_LEN, W=W)
            f.write(json.dumps(rec) + "\n")
            print(s["id"], len(ids), "tokens", f"{rec['seconds']:.1f} s", flush=True)


if __name__ == "__main__":
    main()
