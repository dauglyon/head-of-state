"""Upper bound for next-token emissions whose context restarts at the segment start (true boundaries given).

    H2O_LIGHTNING_PATH=<checkout> uv run python -I experiments/restart_upper_bound.py run --model <path>
    uv run python -I experiments/restart_upper_bound.py report      # results/restart_upper_bound.md

History-aware segmenters get their evidence from what text is in context: a segment scored on its own words
(Eisenstein & Barzilay 2008, sec. 3.2-3.3), a likelihood whose context restarts at the boundary (Wagner et al.
2022, App. A). With one cache per state holding the whole stream, the stated state barely moved the prediction once
the text revealed it (docs/LOG.md, 2026-11-26). Here each state's context restarts at every true segment start: for
token t in a segment that began at r, the emission for state k is P(x_t | opening_k, x_r..x_{t-1}), with the opening
of head_of_state.next_token.PerStateCacheReader (the owner's wording) and a plain cache (no state line) beside it.
The true boundaries are given, so this bounds what a decoder over hypothesised boundaries could get from restarted
contexts. Screening documents of wording_screen.py; files in wording_screen's record format.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import joined_documents as jd  # noqa: E402
import wording_screen as ws  # noqa: E402
from common import RUNS  # noqa: E402

OUT = RUNS / "restart_upper"


def run(model_path):
    import re
    import time
    import torch
    from transformers import AutoModelForCausalLM
    from head_of_state.harness import tokens_and_gold
    from head_of_state.next_token import PerStateCacheReader
    tok = jd.tokenizer()
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="cuda").eval()
    m = re.search(r"models--([^/]+)", model_path)
    tag = m.group(1) if m else Path(model_path.rstrip("/")).name
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / f"{tag}.jsonl").open("w") as f:
        for s, (states, docs) in ws.screen_docs(tok).items():
            rd = PerStateCacheReader(model, tok, states)
            for d in docs:
                ids, gold = tokens_and_gold(tok, d["text"], d["spans"])
                t0 = time.time()
                state_lp, plain_lp, since = [], [], []
                start = 0
                for t in range(1, len(ids) + 1):
                    if t == len(ids) or gold[t] != gold[start]:
                        seg = ids[start:t]
                        # emissions for tokens start+1 .. t-1 (the segment's first token has no restarted context)
                        lp, pl = rd.emissions(seg) if len(seg) > 1 else ([], [])
                        if start > 0:   # the segment's first token: predicted from the opening alone
                            first = []
                            for h in rd.heads:
                                with torch.no_grad():
                                    out = model(input_ids=torch.tensor([h], device="cuda")).logits[0, -1].float()
                                first.append(float(torch.log_softmax(out, -1)[seg[0]]))
                            state_lp.append(first[:-1]); plain_lp.append(first[-1]); since.append(0)
                        state_lp += lp; plain_lp += pl; since += list(range(1, len(seg)))
                        start = t
                f.write(json.dumps({"set": s, "id": d["id"], "n_tokens": len(ids), "gold": gold, "state_logp": state_lp,
                                    "plain_logp": plain_lp, "since_start": since, "seconds": time.time() - t0,
                                    "model": model_path}) + "\n")
                f.flush()
            print(f"{s} done", flush=True)


def report():
    import numpy as np
    sets = {"news": len(jd.LISTS["ag_news"]), "articles": 10, "pubmed": 5}
    print("True segment starts given; each state's context restarts there. Evidence for the true state per token "
          "(own minus mean of the other occurring states), by tokens since the segment start.\n")
    print("| Model | set | all | 0-8 | 8-32 | 32-128 | 128+ | positive | state predictions minus plain |\n|---|---|---|---|---|---|---|---|---|")
    for f in sorted(OUT.glob("*.jsonl")):
        recs = list(map(json.loads, f.open()))
        for s, K in sets.items():
            rs = [r for r in recs if r["set"] == s]
            if not rs:
                continue
            lp = np.vstack([np.asarray(r["state_logp"]) for r in rs])
            g = np.concatenate([np.asarray(r["gold"][1:]) for r in rs])
            since = np.concatenate([np.asarray(r["since_start"]) for r in rs])
            plain = np.concatenate([np.asarray(r["plain_logp"]) for r in rs])
            occ = [k for k in range(K) if (g == k).sum() >= 50]
            keep = np.isin(g, occ)
            ev = ws.evidence(lp, g, occ)
            cell = lambda m: f"{ev[keep & m].mean():+.3f}" if (keep & m).any() else "—"
            bins = [(0, 8), (8, 32), (32, 128), (128, 10**9)]
            print(f"| {f.stem} | {s} | {cell(np.ones_like(keep))} | " + " | ".join(cell((since >= a) & (since < b)) for a, b in bins)
                  + f" | {(ev[keep] > 0).mean():.0%} | {(lp.mean(1) - plain).mean():+.2f} |")


if __name__ == "__main__":
    a = sys.argv
    {"run": lambda: run(a[a.index("--model") + 1]), "report": report}[a[1]]()
