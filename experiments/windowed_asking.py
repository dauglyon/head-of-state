"""Path 1a / 1c: asking with the question seeing only the last W stream tokens (docs/EXPERIMENTS.md).

    uv run python -I experiments/windowed_asking.py run          # GPU: experiments/runs/winask/*.jsonl
    uv run python -I experiments/windowed_asking.py report       # CPU: results/windowed_asking.md

Each reading is one pass over [opening text + the last W tokens + question] (1 option ordering); many
readings go into one left-padded batch, with no cache. Declared windows (fractions of the declared
segment length, fixed before these runs): PubMed dev 1x, 2x, 4x (30, 60, 120 tokens), read every 2nd
token over the 10 dev abstracts; S1 lag streams 0.25x, 0.5x, 1x (48, 95, 190), read every 2nd token
from 16 tokens before the switch. Full-history asking at the same positions comes from the saved runs.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pubmed_rct  # noqa: E402
import s1_lag  # noqa: E402
import window_likelihood_sweep  # noqa: E402
import wikisection  # noqa: E402
from common import MODEL, RUNS  # noqa: E402
from head_of_state.harness import UnitReader, tokens_and_gold  # noqa: E402

OUT = RUNS / "winask"
W_S3 = [30, 60, 120]
W_S1 = [48, 95, 190]


@torch.no_grad()
def window_reads(model, tok, reader: UnitReader, ids, positions, W, batch_tokens=6000):
    """Letter log-probabilities (state order, ordering 0) after [head + ids[n-W:n] + question] for
    each 1-based position n."""
    dev = next(model.parameters()).device
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    seqs = [reader.head_ids + ids[max(0, n - W): n] + reader.branches[0] for n in positions]
    per = max(1, batch_tokens // max(len(s) for s in seqs))
    out = []
    for b in range(0, len(seqs), per):
        chunk = seqs[b: b + per]
        L = max(len(s) for s in chunk)
        x = torch.tensor([[pad] * (L - len(s)) + s for s in chunk], device=dev)
        mask = torch.tensor([[0] * (L - len(s)) + [1] * len(s) for s in chunk], device=dev)
        pos = (mask.cumsum(1) - 1).clamp_min(0)
        logits = model(input_ids=x, attention_mask=mask, position_ids=pos, use_cache=False, logits_to_keep=1).logits
        lp = torch.log_softmax(logits[:, -1, reader.letters].float(), -1).cpu()
        shift = reader.shifts[0]
        out += [[float(row[(k - shift) % reader.K]) for k in range(reader.K)] for row in lp]
    return out


def run():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
    OUT.mkdir(parents=True, exist_ok=True)
    s3, _, _ = window_likelihood_sweep.sets(tok)["S3dev"]
    jobs = [("S3dev", s3, pubmed_rct.STATES, pubmed_rct.UNIT, 30.0, W, None) for W in W_S3]
    s1 = s1_lag.build(tok)
    jobs += [("S1lag", s1, wikisection.STATES, wikisection.UNIT, s1_lag.SEG_LEN, W, s1_lag.READ_BEFORE) for W in W_S1]
    for name, streams, states, unit, L, W, before in jobs:
        reader = UnitReader(model, tok, states, rotations=1, unit=unit)
        with (OUT / f"{name}_W{W}.jsonl").open("w") as f:
            tot_t = tot_n = 0
            for s in streams:
                ids, gold = tokens_and_gold(tok, s["text"], s["spans"])
                if before is None:
                    positions = list(range(2, len(ids) + 1, 2))
                    score_from = 1
                else:
                    sw = gold.index(gold[-1]) + 1
                    positions = list(range(max(1, sw - before), len(ids) + 1, 2))
                    score_from = positions[0]
                torch.cuda.synchronize(); t0 = time.time()
                rows = window_reads(model, tok, reader, ids, positions, W)
                torch.cuda.synchronize(); sec = time.time() - t0
                tot_t += sec; tot_n += len(positions)
                f.write(json.dumps({"id": s["id"], "gold": gold, "n_tokens": len(ids), "shift_logp": [[r] for r in rows],
                                    "positions": positions, "shifts": [0], "stride": 1, "W": W, "seg_len": L,
                                    "prev_len": s.get("prev_len"), "score_from": score_from, "seconds": sec}) + "\n")
            print(f"{name} W={W}: {len(streams)} streams, {1000 * tot_t / tot_n:.0f} ms per reading", flush=True)


def report():
    from head_of_state.decoders import decode_persist, probs
    from head_of_state.metrics import forward_posteriors, pool, stream_counts
    load = lambda f: [json.loads(l) for l in open(f)]

    def metrics(recs, L, kind, stride=None):
        counts, tokens = [], 0
        for r in recs:
            p, pos = probs(r, rotations=1, stride=stride or 1)
            start = r.get("score_from", 1)
            keep = pos >= start
            p, pos = p[keep], pos[keep]
            if kind == "raw":
                q, lab = p, p.argmax(1)
            elif kind == "persist":
                q, lab = p, np.asarray(decode_persist(p, 3))
            else:
                q = forward_posteriors(p, 1.0, L / 2)          # readings every 2nd token
                lab = q.argmax(1)
            counts.append(stream_counts(lab, q, r["gold"], pos))
            tokens += r["n_tokens"] - start + 1
        return pool(counts, tokens), counts

    print("## 1a. Delay against window, PubMed dev (10 abstracts, 36 switches; readings every 2nd token, 1 ordering)\n")
    print("| Question sees | decoder | token acc. | boundary F1 ±8 | ±16 | matched | mean delay | spurious /1k | Brier | ECE |\n"
          "|---|---|---|---|---|---|---|---|---|---|")
    full = load(RUNS / "s3_dev_r5.jsonl")[:10]
    variants = [("last 30 tokens (1×)", OUT / "S3dev_W30.jsonl"), ("last 60 tokens (2×)", OUT / "S3dev_W60.jsonl"),
                ("last 120 tokens (4×)", OUT / "S3dev_W120.jsonl"), ("full history", None)]
    for label, f in variants:
        recs = full if f is None else load(f)
        for kind in ("raw", "persist", "filter"):
            m, _ = metrics(recs, 30.0, kind, stride=2 if f is None else None)
            print(f"| {label} | {kind if kind != 'filter' else 'forward filter η=1'} | {m['accuracy']:.3f} | "
                  f"{m['boundary_f1@8']:.2f} | {m['boundary_f1@16']:.2f} | {m['matched']} | {m['mean_delay']:.1f} | "
                  f"{m['spurious_per_1k']:.1f} | {m['brier']:.3f} | {m['ece']:.3f} |")

    print("\n## 1c. S1 lag streams (12; scored from 16 tokens before the switch; readings every 2nd token, 1 ordering)\n")
    print("| Question sees | decoder | token acc. | matched | mean delay | spurious /1k | delay by previous segment "
          "50 / 200 / 800 / 3000 |\n|---|---|---|---|---|---|---|")
    s1_full = load(RUNS / "cde" / "s1_q.jsonl")
    for r in s1_full:
        r["score_from"] = r["positions"][0]
    variants = [(f"last {W} tokens ({W / s1_lag.SEG_LEN:.2g}×)", OUT / f"S1lag_W{W}.jsonl") for W in W_S1] + \
               [("full history", None)]
    for label, f in variants:
        recs = s1_full if f is None else load(f)
        for kind in ("persist", "filter"):
            m, _ = metrics(recs, s1_lag.SEG_LEN, kind)
            by = []
            for L in s1_lag.PREV:
                mm, _ = metrics([r for r in recs if r["prev_len"] == L], s1_lag.SEG_LEN, kind)
                by.append(f"{mm['mean_delay']:.0f}")
            print(f"| {label} | {kind if kind != 'filter' else 'forward filter η=1'} | {m['accuracy']:.3f} | {m['matched']} | "
                  f"{m['mean_delay']:.1f} | {m['spurious_per_1k']:.1f} | {' / '.join(by)} |")


if __name__ == "__main__":
    {"run": run, "report": report}[sys.argv[1]]()
