"""Path 2a, remaining check: how concentrated the likelihood evidence is on a few tokens.

    uv run python -I experiments/likelihood_concentration.py > results/likelihood_concentration.md

Per token, the log-likelihood ratio (LLR) of the right regime's description against the mean of the
others, from the saved emissions of window_likelihood_sweep.py (PubMed dev, 10 abstracts; WikiSection
validation, 2 streams), with the description only at the start (no window) and with the declared
hopped window. Reports the share of the total positive LLR carried by the top 1% / 5% / 10% of
tokens, the share of tokens with LLR > 0, and the tokens that carry the most evidence.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import window_likelihood_sweep  # noqa: E402
from common import MODEL, RUNS  # noqa: E402
from head_of_state.harness import tokens_and_gold  # noqa: E402


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    sets = window_likelihood_sweep.sets(tok)
    print("| Set | window | tokens | LLR > 0 | top 1% share | top 5% share | top 10% share | mean LLR |\n"
          "|---|---|---|---|---|---|---|---|")
    tops = {}
    for name, W in (("S3dev", None), ("S3dev", 2), ("WSval", None), ("WSval", 12)):
        streams, _, _ = sets[name]
        recs = {r["id"]: r for r in map(json.loads, open(RUNS / "window" / f"{name}_W{W or 'none'}.jsonl"))}
        llr, words = [], []
        for s in streams:
            r = recs[s["id"]]
            ids, gold = tokens_and_gold(tok, s["text"], s["spans"])
            e = np.asarray(r["logp"])
            K = e.shape[1]
            for t, y in enumerate(gold):
                llr.append(e[t, y] - (e[t].sum() - e[t, y]) / (K - 1))
                words.append(tok.decode([ids[t]]).strip())
        llr = np.asarray(llr)
        pos = np.sort(np.clip(llr, 0, None))[::-1]
        share = lambda f: pos[: max(1, int(f * len(pos)))].sum() / pos.sum()
        print(f"| {name} | {W or 'none'} | {len(llr)} | {np.mean(llr > 0):.2f} | {share(0.01):.2f} | {share(0.05):.2f} | "
              f"{share(0.10):.2f} | {llr.mean():+.3f} |")
        agg = collections.defaultdict(list)
        for w, v in zip(words, llr):
            agg[w].append(v)
        tops[(name, W)] = sorted(((sum(v), w, len(v)) for w, v in agg.items()), reverse=True)[:12]
    print("\nTokens carrying the most total LLR (token: total, count):\n")
    for (name, W), items in tops.items():
        print(f"- {name}, window {W or 'none'}: " + ", ".join(f"`{w}` {s:+.1f} ({n})" for s, w, n in items))


if __name__ == "__main__":
    main()
