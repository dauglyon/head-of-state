"""Experiments A (fade curve) and B (W sweep) for windowed likelihood emissions.

    uv run python -I experiments/window_likelihood_sweep.py run        # GPU: writes experiments/runs/window/<set>_W<w>.jsonl
    uv run python -I experiments/window_likelihood_sweep.py report     # CPU: fade curves and the W sweep

Sets (tuning data): S3dev = first 10 PubMed 20k RCT dev abstracts in the dev run's order (the
abstracts the question-method baseline covers); WSval = the 2 WikiSection validation streams of
compare_readouts.e2. W = None (no window) and W = 0.25, 0.5, 1, 2 x the set's declared segment length.
Decoder: sticky forward filter, eta = 1, switch probability 1/L from the declared length L.
"""
from __future__ import annotations

import json
import math
import random
import sys
from functools import partial
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import RUNS  # noqa: E402
import compare_readouts  # noqa: E402
import pubmed_rct  # noqa: E402
from head_of_state.decoders import decode_filter, evaluate  # noqa: E402

OUT = RUNS / "window"
FRACTIONS = [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0]
DIST_BINS = [0, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 10**9]


def sets(tok):
    s3 = pubmed_rct.load("dev")
    random.Random(0).shuffle(s3)
    ws, wstates, _, wL = compare_readouts.e2(tok)
    return {"S3dev": (s3[:10], pubmed_rct.STATES, 30.0), "WSval": (ws, wstates, wL)}


def widths(L):
    return [None] + [max(2, round(f * L)) for f in FRACTIONS]


def run():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from head_of_state.window_likelihood import run_window
    tok = AutoTokenizer.from_pretrained(compare_readouts.MODEL)
    model = AutoModelForCausalLM.from_pretrained(compare_readouts.MODEL, dtype=torch.bfloat16, device_map="cuda")
    OUT.mkdir(parents=True, exist_ok=True)
    only = sys.argv[2:]                                   # optional set names to (re)run
    for name, (streams, states, L) in sets(tok).items():
        if only and name not in only:
            continue
        for W in widths(L):
            f = OUT / f"{name}_W{W or 'none'}.jsonl"
            with f.open("w") as fh:
                for s in streams:
                    rec = run_window(model, tok, states, s, W)
                    rec["seg_len"] = L
                    fh.write(json.dumps(rec) + "\n")
            recs = [json.loads(l) for l in f.open()]
            ms = 1000 * sum(r["seconds"] for r in recs) / sum(r["n_tokens"] for r in recs)
            print(f"{name} W={W}: {ms:.1f} ms/token (hop-batched)", flush=True)


def distance(t, W):
    """Tokens between the end of the description and token t (t's own position counts as 1)."""
    if W is None:
        return t + 1
    hop = max(1, W // 2)
    s = (t // hop) * hop
    return t - max(0, s - W) + 1


def fade(recs, W):
    """Mean gap (true-state log p minus mean over the other states) per distance bin, with SE."""
    bins = {}
    for r in recs:
        e = np.asarray(r["logp"])
        for t, y in enumerate(r["gold"]):
            g = e[t, y] - (e[t].sum() - e[t, y]) / (e.shape[1] - 1)
            d = distance(t, W)
            lo = max(b for b in DIST_BINS if b < d) if d > 0 else 0
            bins.setdefault(lo, []).append(g)
    return {k: (float(np.mean(v)), float(np.std(v) / math.sqrt(len(v))), len(v)) for k, v in sorted(bins.items())}


def report():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(compare_readouts.MODEL)
    for name, (streams, states, L) in sets(tok).items():
        print(f"\n## {name} (declared segment length {L:g} tokens, K={len(states)})")
        for W in widths(L):
            f = OUT / f"{name}_W{W or 'none'}.jsonl"
            recs = [json.loads(l) for l in f.open()]
            ms = 1000 * sum(r["seconds"] for r in recs) / sum(r["n_tokens"] for r in recs)
            m = evaluate(recs, partial(decode_filter, eta=1.0, seg_len_readings=L), rotations=1)
            fd = fade(recs, W)
            print(f"W={str(W):>4s}: acc {m['accuracy_committed']:.3f} | matched {m['switches_matched']}/{m['true_switches']}"
                  f" | delay {m['mean_delay_tokens']:.1f} | spurious/1k {m['spurious_per_1k_tokens']:.2f} | {ms:.1f} ms/token")
            print("        gap by distance from description (bin start: mean ± SE, n): " +
                  "; ".join(f"{k}: {v[0]:+.3f}±{v[1]:.3f} ({v[2]})" for k, v in fd.items()))


if __name__ == "__main__":
    {"run": run, "report": report}[sys.argv[1]]()
