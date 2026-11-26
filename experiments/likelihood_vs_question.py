"""Experiments C (head-to-head), D (S1 delay by previous-segment length) and E (question method
restricted to the same hopped window), for windowed likelihood vs the question method.

    uv run python -I experiments/likelihood_vs_question.py run <job>...   # jobs: s3_qwin s1_lik s1_q s1_qwin
    uv run python -I experiments/likelihood_vs_question.py report

Question readings use 1 option ordering. On S1 lag streams they are taken every 2nd token from 16
tokens before the switch to the end (earlier tokens are fed without readings); on S3 dev the windowed
question reads every 4th token and is compared with the full-history question at the same stride.
Windowed question: at each reading the cache is rebuilt from the opening text + the tokens the same
hopped window would hold (hop W/2; window start = hop start - W), so it sees what the likelihood
copies see. W = 0.0625 x declared length (pre-registered): S3 2, S1 12.
"""
from __future__ import annotations

import json
import math
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import RUNS  # noqa: E402
import window_likelihood_sweep  # noqa: E402
import s1_lag  # noqa: E402
import pubmed_rct  # noqa: E402
import wikisection  # noqa: E402
from head_of_state.decoders import decode_filter, decode_persist, evaluate  # noqa: E402
from head_of_state.harness import UnitReader, tokens_and_gold  # noqa: E402

OUT = RUNS / "cde"
W_S3, W_S1 = 2, 12


def question_reads(reader: UnitReader, ids, positions, W=None):
    """Per-ordering letter log-probs at the given 1-based positions. W=None: full history (tokens fed
    in chunks between readings). W: cache rebuilt per reading from the hopped window."""
    reader.reset()
    rows, fed = [], 0
    hop = max(1, W // 2) if W else None
    for n in positions:
        if W is None:
            reader._feed(ids[fed:n])
            fed = n
        else:
            t = n - 1
            s = (t // hop) * hop
            reader.cache, reader.n_cached = None, 0
            reader._feed(reader.head_ids + ids[max(0, s - W): t + 1])
        reader._read()
        rows.append(reader.last_shift_logp)
    return rows


def load_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(wikisection.MODEL)
    model = AutoModelForCausalLM.from_pretrained(wikisection.MODEL, dtype=torch.bfloat16, device_map="cuda")
    return tok, model


def run(jobs):
    import torch
    tok, model = load_model()
    OUT.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        out = OUT / f"{job}.jsonl"
        if job == "s3_qwin":
            streams, _, _ = window_likelihood_sweep.sets(tok)["S3dev"]
            states, unit, W, L = pubmed_rct.STATES, pubmed_rct.UNIT, W_S3, 30.0
        else:
            streams = s1_lag.build(tok)
            states, unit, W, L = wikisection.STATES, wikisection.UNIT, W_S1, s1_lag.SEG_LEN
        reader = UnitReader(model, tok, states, rotations=1, unit=unit)
        with out.open("w") as f:
            for s in streams:
                ids, gold = tokens_and_gold(tok, s["text"], s["spans"])
                torch.cuda.synchronize(); t0 = time.time()
                if job == "s1_lik":
                    from head_of_state.window_likelihood import run_window
                    rec = run_window(model, tok, states, s, W)
                else:
                    if job == "s3_qwin":
                        positions = list(range(4, len(ids) + 1, 4))
                    else:
                        sw = gold.index(gold[-1]) + 1
                        positions = list(range(max(1, sw - s1_lag.READ_BEFORE), len(ids) + 1, 2))
                    rows = question_reads(reader, ids, positions, None if job == "s1_q" else W)
                    rec = {"shift_logp": rows, "positions": positions, "shifts": reader.shifts, "stride": 1}
                torch.cuda.synchronize()
                rec.update(id=s["id"], gold=gold, n_tokens=len(ids), seconds=time.time() - t0, seg_len=L, W=W,
                           prev_len=s.get("prev_len"), n_readings=len(rec.get("positions", ids)))
                f.write(json.dumps(rec) + "\n")
                print(job, s["id"], f"{rec['seconds']:.1f} s", flush=True)


def summarize(recs, dec):
    m = evaluate(recs, dec, rotations=1)
    return m


def fmt(m, cost):
    d = m["mean_delay_tokens"]
    return (f"{m['accuracy_committed']:.3f} | {m['switches_matched']}/{m['true_switches']} | "
            f"{'—' if math.isnan(d) else f'{d:.1f}'} | {m['spurious_per_1k_tokens']:.1f} | {cost}")


def report():
    load = lambda f: [json.loads(l) for l in open(f)]
    hdr = "| Method | decoder | token acc. | matched | mean delay | spurious/1k | ms/token |\n|---|---|---|---|---|---|---|"
    filt = lambda L, stride=1: partial(decode_filter, eta=1.0, seg_len_readings=L / stride)
    p3 = partial(decode_persist, persist=3)

    print("## C. S3 dev, 10 abstracts (declared length 30)\n" + hdr)
    lik = load(window_likelihood_sweep.OUT / f"S3dev_W{W_S3}.jsonl")
    q = load(RUNS / "s3_dev_r5.jsonl")[:10]
    ms_lik = 1000 * sum(r["seconds"] for r in lik) / sum(r["n_tokens"] for r in lik)
    print(f"| likelihood, window W={W_S3} | filter η=1 | {fmt(evaluate(lik, filt(30), 1), f'{ms_lik:.0f} (hop-batched)')} |")
    for rot, ms in ((1, 103), (2, 150)):
        for name, dec in (("persist 3", p3), ("filter η=1", filt(30))):
            print(f"| question, full history, {rot} ordering(s), every token | {name} | "
                  f"{fmt(evaluate(q, dec, rotations=rot), f'{ms} (measured separately)')} |")
    qw = load(OUT / "s3_qwin.jsonl")
    for name, dec in (("persist 3", p3), ("filter η=1", filt(30, 4))):
        print(f"| question, full history, every 4th token | {name} | {fmt(evaluate(q, dec, rotations=1, stride=4), '—')} |")
        print(f"| question, window W={W_S3}, every 4th token (E) | {name} | {fmt(evaluate(qw, dec, rotations=1), '—')} |")

    print("\n## C/D/E. S1 lag streams, 12 (declared length 190.5); scored from 16 tokens before the switch to the end; question read every 2nd token there, likelihood every token\n" + hdr)
    runs = {"likelihood, window W=12": (load(OUT / "s1_lik.jsonl"), filt(190.5)),
            "question, full history": (load(OUT / "s1_q.jsonl"), filt(190.5, 2)),
            "question, window W=12 (E)": (load(OUT / "s1_qwin.jsonl"), filt(190.5, 2))}
    # Score every method from the question method's first reading (16 tokens before the switch) on;
    # the likelihood path is still decoded over the whole stream.
    first = {r["id"]: r["positions"][0] for r in runs["question, full history"][0]}
    for recs, _ in runs.values():
        for r in recs:
            r["score_from"] = first[r["id"]]
    for name, (recs, dec) in runs.items():
        decs = [("filter η=1", dec)] + ([("persist 3", p3)] if "question" in name else [])
        for dn, d in decs:
            print(f"| {name} | {dn} | {fmt(evaluate(recs, d, rotations=1), '—')} |")
    print("\n## D. Mean delay after the switch by previous-segment length (tokens; matched switches only; "
          "n = 3 streams each)\n| Method | decoder | " + " | ".join(str(L) for L in s1_lag.PREV) + " |\n|---|---|" +
          "---|" * len(s1_lag.PREV))
    for name, (recs, dec) in runs.items():
        decs = [("filter η=1", dec)] + ([("persist 3", p3)] if "question" in name else [])
        for dn, d in decs:
            cells = []
            for L in s1_lag.PREV:
                sub = [r for r in recs if r["prev_len"] == L]
                m = evaluate(sub, d, rotations=1)
                cells.append(f"{m['mean_delay_tokens']:.0f} ({m['switches_matched']}/3)"
                             if m["switches_matched"] else "— (0/3)")
            print(f"| {name} | {dn} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2:])
    else:
        report()
