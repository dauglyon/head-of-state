"""Per-state word distributions from the descriptions, used as history-free emissions (trials).

    H2O_LIGHTNING_PATH=<checkout> uv run python -I experiments/state_vocab.py build
    uv run python -I experiments/state_vocab.py report          # results/state_vocab.md

Classic HMM topic segmenters score each word against a per-state unigram model (Dragon's system in the TDT pilot
report, sec. 2.2; Blei & Moreno 2001): the emission ignores the text before it, so every word is evidence about the
state. Here the per-state distribution comes from the frozen model and the state's description, no training:
  sampled   Qwen3.5-4B-Base writes 8 passages of 96 tokens from an opening that names the state (seed 0, temperature
            1); P_k is the model's next-token distribution averaged over every position of those passages, i.e. its
            own estimate of which tokens state-k text contains. Two openings: "own" ("The following text is about
            <description>.", the project's wording) and "genz" (Gen-Z's topic template, arXiv 2311.07115 Table 5,
            "The topic of this text revolves around <description>: ").
  reply     H2O-Lightning's reply-slot distribution of ref-instruction (experiments/wording_screen.py) with no stream
            text, computed once per state.
Each P_k sums to 1 over the vocabulary, so no state can raise the probability of every token. A token's emission
for state k is P_k(token); a background shared by all states cancels between them. Scored on the screening
documents of wording_screen.py with its measures, and decoded at weight 1 with the declared lengths.
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
import wording_screen as ws  # noqa: E402
from common import RUNS  # noqa: E402

OUT = RUNS / "state_vocab"
OPENINGS = {"own": "The following text is about {description}.\n\n",
            "genz": "The topic of this text revolves around {description}: "}
N_PASSAGES, LENGTH = 8, 96
DECLARED = {"news": 270.0, "articles": 184.0, "pubmed": 30.0}


def base_path():
    return next(Path.home().glob(".cache/huggingface/hub/models--Qwen--Qwen3.5-4B-Base/snapshots/*/")).as_posix()


def build():
    import torch
    from transformers import AutoModelForCausalLM
    from head_of_state.next_token import StatedNextTokenReader
    tok = jd.tokenizer()
    OUT.mkdir(parents=True, exist_ok=True)
    sets = {s: st for s, (st, _) in ws.screen_docs(tok).items()}
    sep = tok.convert_tokens_to_ids("<|endoftext|>")
    model = AutoModelForCausalLM.from_pretrained(base_path(), dtype=torch.bfloat16, device_map="cuda").eval()
    torch.manual_seed(0)
    for opening, template in OPENINGS.items():
        for s, states in sets.items():
            dists = []
            for name, desc in states.items():
                prompt = [sep] + tok(template.format(description=desc), add_special_tokens=False)["input_ids"]
                x = torch.tensor([prompt] * N_PASSAGES, device="cuda")
                with torch.no_grad():
                    gen = model.generate(x, do_sample=True, temperature=1.0, top_p=1.0, max_new_tokens=LENGTH,
                                         min_new_tokens=LENGTH, pad_token_id=sep)
                    lp = torch.log_softmax(model(input_ids=gen).logits[:, len(prompt) - 1:-1].float(), -1)
                    dists.append(torch.logsumexp(lp.reshape(-1, lp.shape[-1]), 0).cpu().numpy() - np.log(lp.shape[0] * lp.shape[1]))
            np.save(OUT / f"sampled_{opening}_{s}.npy", np.stack(dists).astype(np.float32))
            print(f"sampled {opening} {s}: {len(states)} states", flush=True)
    del model
    torch.cuda.empty_cache()
    h2o = AutoModelForCausalLM.from_pretrained(os.environ.get("H2O_LIGHTNING_PATH", "h2oai/h2o-lightning-4b"),
                                               dtype=torch.bfloat16, device_map="cuda").eval()
    for s, states in sets.items():
        rd = StatedNextTokenReader(h2o, tok, states, spec=ws.SPECS["ref-instruction"])
        rows = torch.log_softmax(torch.stack(rd._branch_all()), -1).numpy()   # cache holds the instruction only
        np.save(OUT / f"reply_once_{s}.npy", rows.astype(np.float32))
        print(f"reply once {s}", flush=True)


def records(kind, tok):
    """Screen-document records whose state_logp rows are log P_k(next token), in wording_screen's format."""
    from head_of_state.harness import tokens_and_gold
    out = []
    for s, (states, docs) in ws.screen_docs(tok).items():
        P = np.load(OUT / f"{kind}_{s}.npy")                     # [K, vocab] log-probabilities
        bg = np.logaddexp.reduce(P, 0) - np.log(len(P))          # the states' mixture
        for d in docs:
            ids, gold = tokens_and_gold(tok, d["text"], d["spans"])
            nxt = np.asarray(ids[1:])
            out.append({"set": s, "id": d["id"], "n_tokens": len(ids), "gold": gold, "seconds": 0.0,
                        "state_logp": P[:, nxt].T.tolist(), "plain_logp": bg[nxt].tolist()})
    return out


def report():
    from head_of_state.settle import settled_labels
    from stated_next_token import emission_probs
    tok = jd.tokenizer()
    sets = {"news": len(jd.LISTS["ag_news"]), "articles": 10, "pubmed": 5}
    print("Screening documents. Per set: D (nats per token) · share of tokens favouring the true state · mean/SD · "
          "pairs below zero · lift range · token accuracy of settled labels (weight 1, declared length).\n")
    print("| Emission | " + " | ".join(sets) + " |\n|---|" + "---|" * len(sets))
    rows = {f"sampled, {o} opening (Qwen3.5-4B-Base)": f"sampled_{o}" for o in OPENINGS}
    rows["reply slot once per state (H2O-Lightning)"] = "reply_once"
    for label, kind in rows.items():
        recs = records(kind, tok)
        cells = []
        for s, K in sets.items():
            rs = [r for r in recs if r["set"] == s]
            m = ws.measures(rs, K)
            ok = n = 0
            for r in rs:
                p, pos = emission_probs(r, K)
                lab = settled_labels(p, pos, DECLARED[s]); g = np.asarray(r["gold"])
                ok += (lab == g).sum(); n += len(g)
            cells.append(f"{m['D']:.2f} · {m['pos']:.0%} · {m['snr']:.2f} · {m['wrong']} · {m['lift']:.2f} · {ok / n:.3f}")
        print(f"| {label} | " + " | ".join(cells) + " |", flush=True)


if __name__ == "__main__":
    {"build": build, "report": report}[sys.argv[1]]()
