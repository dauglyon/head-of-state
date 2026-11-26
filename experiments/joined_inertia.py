"""Does the earlier stream hold the readings back after a switch? (HANDOFF open question 1)

    H2O_LIGHTNING_PATH=<checkout> uv run python -I experiments/joined_inertia.py run
    uv run python -I experiments/joined_inertia.py report     # results/joined_inertia.md

For every switch in the seed-0 joined-document streams (joined_documents.py), the first 128 tokens of the
new segment (fewer if the segment is shorter) are read again on their own, with nothing before them.
Comparing those readings with the in-stream readings at the same tokens separates two causes of late
noticing: the new text not yet showing its state (both late) and the earlier stream outweighing it (only
the in-stream readings late). H2O-Lightning-4B, a reading at every token, as in the stream runs.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import joined_documents as jd  # noqa: E402
from common import RUNS  # noqa: E402
from scored_runs import FROZEN  # noqa: E402

OUT = RUNS / "joined" / "inertia.jsonl"
WINDOW = 128
CHECK = (8, 16, 32, 64, 128)


def switches(gold):
    return [i for i in range(1, len(gold)) if gold[i] != gold[i - 1]]


def run():
    import torch
    from transformers import AutoModelForCausalLM
    from head_of_state.h2o import H2OReader
    from head_of_state.harness import tokens_and_gold
    tok = jd.tokenizer()
    path = os.environ.get("H2O_LIGHTNING_PATH", "h2oai/h2o-lightning-4b")
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda")
    with OUT.open("w") as f:
        for name in jd.LISTS:
            reader = H2OReader(model, tok, jd.LISTS[name], unit=jd.UNIT, **FROZEN)
            saved = {json.loads(l)["id"]: json.loads(l) for l in jd.out_file(name, 0).open()}
            for s in jd.streams_for(name, tok, 0):
                ids, gold = tokens_and_gold(tok, s["text"], s["spans"])
                assert gold == saved[s["id"]]["gold"]
                sw = switches(gold)
                for j, start in enumerate(sw):
                    end = min(start + WINDOW, sw[j + 1] if j + 1 < len(sw) else len(ids))
                    reader.reset()
                    rows, positions = [], []
                    for t in ids[start:end]:
                        for r in reader.push_tokens([t]):
                            rows.append(r.detail); positions.append(start + r.pos)
                    for r in reader.finish():
                        rows.append(r.detail); positions.append(start + r.pos)
                    f.write(json.dumps({"list": name, "id": s["id"], "switch": start, "new": gold[start],
                                        "prev_len": start - (sw[j - 1] if j else 0), "shifts": reader.shifts,
                                        "shift_logp": rows, "positions": positions}) + "\n")
            print(f"{name} done", flush=True)


def first_correct(p, new):
    hit = np.flatnonzero(p.argmax(1) == new)
    return int(hit[0]) + 1 if len(hit) else None


def report():
    from head_of_state.decoders import probs
    rows = [json.loads(l) for l in OUT.open()]
    streams = {}
    for name in jd.LISTS:
        for l in jd.out_file(name, 0).open():
            r = json.loads(l)
            streams[r["id"]] = probs(r, rotations=1)
    print("Readings of the new segment's first tokens: in the stream (after the earlier text) and alone (nothing "
          "before). P(new state) is the mean probability the reading gives the new segment's state after k tokens.\n")
    print("| State list | switches | first reading naming the new state, in stream (median tokens) | alone | "
          + " | ".join(f"P(new) at {k}: stream / alone" for k in CHECK) + " |\n|" + "---|" * (4 + len(CHECK)))
    for name in jd.LISTS:
        sel = [r for r in rows if r["list"] == name]
        if not sel:
            continue
        fs, fa, ps, pa = [], [], {k: [] for k in CHECK}, {k: [] for k in CHECK}
        for r in sel:
            p_alone, _ = probs(r, rotations=1)
            p_all, pos_all = streams[r["id"]]
            n = len(p_alone)
            idx = {int(x): i for i, x in enumerate(pos_all)}
            p_stream = np.stack([p_all[idx[r["switch"] + k]] for k in range(1, n + 1)])
            fs.append(first_correct(p_stream, r["new"])); fa.append(first_correct(p_alone, r["new"]))
            for k in CHECK:
                if k <= n:
                    ps[k].append(p_stream[k - 1, r["new"]]); pa[k].append(p_alone[k - 1, r["new"]])
        med = lambda xs: f"{statistics.median([x for x in xs if x is not None]):.0f}" + (
            f" ({sum(x is None for x in xs)} never)" if any(x is None for x in xs) else "")
        cells = " | ".join(f"{np.mean(ps[k]):.2f} / {np.mean(pa[k]):.2f} (n={len(ps[k])})" if ps[k] else "—" for k in CHECK)
        print(f"| {name} | {len(sel)} | {med(fs)} | {med(fa)} | {cells} |")


if __name__ == "__main__":
    {"run": run, "report": report}[sys.argv[1]]()
