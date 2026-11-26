"""S1 seam control (docs/TEST_SUITE.md H5): do labels jump where the cache is re-anchored?

    uv run python -I experiments/seam_runs.py run      # GPU
    uv run python -I experiments/seam_runs.py report   # CPU: results/seam_runs.md

One stream of whole WikiSection en_disease test articles (about 15,000 tokens, natural section order),
read with the frozen settings; the cache re-anchors (opening text + last 2,048 tokens) whenever it would
pass 8,192 tokens, and the stream positions of those re-anchors are logged. Run A reads the stream from
the start; run B starts at the first section boundary after token 1,000, which moves every switch about
1,000 tokens relative to the seams. Each true switch is then scored by its delay in both runs and
grouped by whether it lies within 500 tokens after a seam.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import wikisection  # noqa: E402
from common import MODEL, RUNS  # noqa: E402
from scored_runs import STRIDE8  # noqa: E402

OUT = RUNS / "seam"
TARGET, SHIFT, NEAR = 15000, 1000, 500


def build(tok, seed=0):
    docs = wikisection.load("test")
    random.Random(seed).shuffle(docs)
    text, spans, n = "", [], 0
    for d in docs:
        base = len(text) + (1 if text else 0)
        text = (text + "\n" + d["text"]) if text else d["text"]
        spans += [(base + a, base + b, g) for a, b, g in d["spans"]]
        n += len(tok(d["text"], add_special_tokens=False)["input_ids"])
        if n >= TARGET:
            break
    return {"id": "seam-A", "text": text, "spans": spans}


def shifted(tok, s):
    from head_of_state.harness import tokens_and_gold
    enc = tok(s["text"], add_special_tokens=False, return_offsets_mapping=True)
    cut_char = enc["offset_mapping"][SHIFT][0]
    start = next(a for a, b, g in s["spans"] if a >= cut_char)
    spans = [(a - start, b - start, g) for a, b, g in s["spans"] if a >= start]
    return {"id": "seam-B", "text": s["text"][start:], "spans": spans, "offset_chars": start}


def run():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from head_of_state.harness import UnitReader, run_stream

    class LoggingReader(UnitReader):
        def _reanchor(self):
            self.seams.append(len(self.stream))
            super()._reanchor()

        def reset(self):
            self.seams = []
            super().reset()

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
    OUT.mkdir(parents=True, exist_ok=True)
    a = build(tok)
    b = shifted(tok, a)
    reader = LoggingReader(model, tok, wikisection.STATES, unit=wikisection.UNIT, **STRIDE8)
    with (OUT / "seam.jsonl").open("w") as f:
        for s in (a, b):
            rec = run_stream(reader, s, tok)
            rec["seams"] = list(reader.seams)
            rec["offset_chars"] = s.get("offset_chars", 0)
            f.write(json.dumps(rec) + "\n")
            print(f"{s['id']}: {rec['n_tokens']} tokens, seams at {rec['seams']}, "
                  f"{1000 * rec['seconds'] / rec['n_tokens']:.1f} ms/token", flush=True)


def report():
    from transformers import AutoTokenizer
    from head_of_state.decoders import decode_persist, probs
    from head_of_state.metrics import forward_posteriors
    tok = AutoTokenizer.from_pretrained(MODEL)
    recs = [json.loads(l) for l in (OUT / "seam.jsonl").open()]
    a_text = build(tok)["text"]
    rows = {}
    for r in recs:
        p, pos = probs(r, rotations=1)
        lab = np.asarray(decode_persist(p, STRIDE8["persist"]))
        gold = r["gold"]
        g = [gold[x - 1] for x in pos]
        # character offset of every token start, to identify the same switch in both runs
        enc = tok(a_text[r["offset_chars"]:], add_special_tokens=False, return_offsets_mapping=True)
        starts = [o[0] + r["offset_chars"] for o in enc["offset_mapping"]]
        for i in range(1, len(g)):
            if g[i] == g[i - 1]:
                continue
            nxt = next((j for j in range(i + 1, len(g)) if g[j] != g[i]), len(g))
            hit = next((j for j in range(i, nxt) if lab[j] == g[i] and lab[j - 1] != g[i]), None)
            t = int(pos[i])                       # the token where the true switch happens
            while t > int(pos[i - 1]) + 1 and gold[t - 2] == g[i]:
                t -= 1
            delay = None if hit is None else int(pos[hit] - t)
            tpos = int(pos[i])
            near = any(0 <= tpos - s_ < NEAR for s_ in r["seams"])
            rows.setdefault(starts[tpos - 1] // 50, {})[r["id"]] = (delay, near, tpos)
    both = {k: v for k, v in rows.items() if len(v) == 2}
    print(f"Seams: run A at {recs[0]['seams']}, run B at {recs[1]['seams']} (stream positions); "
          f"{len(both)} switches present in both runs.\n")
    print("| Switch condition in the run | switches | found | mean delay (tokens) |\n|---|---|---|---|")
    groups = {"within 500 tokens after a seam": [], "elsewhere": []}
    for v in both.values():
        for run_id, (delay, near, _) in v.items():
            groups["within 500 tokens after a seam" if near else "elsewhere"].append(delay)
    for k, ds in groups.items():
        found = [d for d in ds if d is not None]
        print(f"| {k} | {len(ds)} | {len(found)} | {np.mean(found):.1f} |" if found else f"| {k} | {len(ds)} | 0 | — |")
    print("\nPaired: switches near a seam in one run and away from seams in the other\n")
    print("| switch at (run A position) | delay near seam | delay away from seam |\n|---|---|---|")
    for v in both.values():
        (d1, n1, p1), (d2, n2, p2) = v.get("seam-A"), v.get("seam-B")
        if n1 != n2:
            near_d, far_d = (d1, d2) if n1 else (d2, d1)
            print(f"| {p1} | {near_d if near_d is not None else 'missed'} | {far_d if far_d is not None else 'missed'} |")


if __name__ == "__main__":
    {"run": run, "report": report}[sys.argv[1]]()
