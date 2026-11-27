"""Screen wordings (and models) for the state-conditioned next-token emissions, before any decoding.

    H2O_LIGHTNING_PATH=<checkout> uv run python -I experiments/wording_screen.py run --spec <name> [--model <path>] [--tiebreak]
    uv run python -I experiments/wording_screen.py report            # results/wording_screen.md

A wording is a full prompt spec for head_of_state.next_token.StatedNextTokenReader (chat or plain text, system
line, text before the stream, the temporary branch per state, prefill). Every wording is read on the same
tuning documents of three sets whose states change what is written, so no wording is fitted to one list:
  news       the first seed-0 joined news stream (4 topics, 2,101 tokens)
  articles   one WikiSection disease article from the tuning sample (Paroxysmal sympathetic hyperactivity,
             7 sections, 1,961 tokens)
  pubmed     8 PubMed 20k RCT dev abstracts (5 section states), seed 0
Measured per set, from the emissions alone: the mean pairwise evidence D(i, j) = mean over tokens of state i of
log e(i) - log e(j), among states with at least 50 tokens; the share of those pairs at or below zero; the range
of the per-state lift (how much a state's branch raises the probability of other states' text); and the drop of
the mean state emission below the plain next-token probability. A wording is better when D is larger and fewer
pairs are wrong at a lift range no larger. The winner is frozen and confirmed once on unread data.
SPECS holds the reference wordings tried so far and the candidates. The published guidance fixes a direction
(direct next-token reading, plain text, a base model, descriptions that name the kind of text); how a state is
written partway through a text has no published form, so every candidate's mid-text form is the project's own and
is labelled so.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import joined_documents as jd  # noqa: E402
import pubmed_rct  # noqa: E402
import stated_sets  # noqa: E402
import wikisection  # noqa: E402
from common import RUNS  # noqa: E402

OUT = RUNS / "wording"
CATALOGUE_HEAD = "The text below is made of parts of these kinds:\n{kinds}\n\nText:\n"
SPECS = {
    # reference: the two wordings run so far (written without a survey of the options)
    "ref-instruction": {"chat": True, "head": "Text:\n", "prefill": "Next word:",
                        "branch": "\n\nThis part of the text is about {description}. What word comes next?"},
    "ref-catalogue": {"chat": True, "head": CATALOGUE_HEAD, "prefill": "Next word:",
                      "branch": "\n\nThis part of the text is {name}. What word comes next?"},
    # Candidates. The sources (a second session's reader, 2026-11-26) support a direction, not a form: read the
    # next-token probability directly rather than asking about the next word (Hu & Levy 2023, arXiv 2305.13264,
    # Table 2), plain text rather than a chat template (Releasing the Inner Jev, arXiv 2610.17205, App. I,
    # Table 23), a base model rather than a tuned one (Kuribayashi et al. 2024, arXiv 2311.07484, Table 1; Gen-Z,
    # arXiv 2311.07115, fn. 4), descriptions that say what kind of text it is (Gen-Z, sec. 3-4). Every published
    # method states its condition once, before the whole text; none states it partway through, which the design
    # needs at every token. So how the state is written mid-text is this project's own choice in every
    # candidate below, and each is one draw from a space where format alone moves results a lot (Sclar et al.
    # 2023, arXiv 2310.11324).
    # Own form: a bracketed note " [name: description]" before the next token (first used 2026-11-26).
    "cont-bracket": {"chat": False, "head": "", "branch": " [{name}: {description}]", "prefill": ""},
    # Own form: ref-instruction's question without chat markers.
    "plain-instruction": {"chat": False, "head": "Text:\n", "prefill": "\nNext word:", "spacing": True,
                          "branch": "\n\nThis part of the text is about {description}. What word comes next?"},
    # Own form: a lead-in sentence in the words of Gen-Z's topic template ("The topic of this [DOMAIN] revolves
    # around [TOPIC]: ", Table 5) placed mid-text before the next token, [DOMAIN] = "text". Gen-Z itself puts the
    # sentence once before the whole input; the placement and the domain word are not from Gen-Z.
    # The owner's design and wording (2026-11-26): one key/value cache per state, opening with a sentence that the text
    # is written in stretches each in one of the listed states, the list, and "The end of the following text is in
    # state: <name>", then the stream as plain text (head_of_state.next_token.PER_STATE_HEAD); the head at the
    # stream's last token of each cache is
    # that state's prediction of the next token (head_of_state.next_token.PerStateCacheReader).
    "per-state-cache": {"kind": "per_state_cache"},
    # The same with two model-written passages per state in the opening (experiments/state_examples.py).
    "per-state-examples": {"kind": "per_state_cache", "examples": True},
    # The owner's suggestion (2026-11-26): the stream is the assistant's own reply, and before the next token a
    # temporary thinking block states the state; a reasoning model writes text that follows its thinking, so after
    # "</think>" the next token should continue the reply under the stated state.
    "think-state": {"chat": True, "stream_in": "assistant", "answer": False, "spacing": False, "prefill": "",
                    "head": "The text moves between these states:\n{kinds}",
                    "branch": "<think>\nThe text is now in state: {name} ({description}).\n</think>\n\n"},
    # The same with the user turn asking for the text, so the reply is the text being written; with and without the
    # line break that the template puts after "</think>" before a reply begins.
    "think-write": {"chat": True, "stream_in": "assistant", "answer": False, "spacing": False, "prefill": "",
                    "head": "Write a text that moves between these states:\n{kinds}",
                    "branch": "<think>\nThe text is now in state: {name} ({description}).\n</think>\n\n"},
    # The thinking block, then the stream's last 8 tokens again: the next token resumes text the model can see.
    "think-write-repeat": {"chat": True, "stream_in": "assistant", "answer": False, "spacing": False, "prefill": "",
                           "repeat": 8, "head": "Write a text that moves between these states:\n{kinds}",
                           "branch": "<think>\nThe text is now in state: {name} ({description}).\n</think>\n\n"},
    "think-write-inline": {"chat": True, "stream_in": "assistant", "answer": False, "spacing": False, "prefill": "",
                           "head": "Write a text that moves between these states:\n{kinds}",
                           "branch": "<think>\nThe text is now in state: {name} ({description}).\n</think>"},
    "cont-genz": {"chat": False, "head": "", "prefill": "", "spacing": True,
                  "branch": "\n\nThe topic of this text revolves around {description}: "},
}


# Paired comparisons: each candidate against the reference it differs from in one thing, on the same model.
PAIR_WITH = {"ref-catalogue": "ref-instruction", "plain-instruction": "ref-instruction",
             "cont-bracket": "ref-instruction", "cont-genz": "cont-bracket"}
# Cells read as a mis-fit, fixed before the runs: a topic sentence does not describe parts of an abstract, so
# cont-genz on PubMed says nothing about lead-in sentences.
MISFIT = {("cont-genz", "pubmed")}


def screen_docs(tok, which="screen"):
    """{set: (states, [stream])}: fixed tuning documents. "tiebreak": a second document per set for wordings whose
    leads on the screen are under 0.1 nats per token (seed-0 news streams 1 and 2; a WikiSection validation
    article of 1,000-2,500 tokens outside the tuning sample; 8 other PubMed dev abstracts), read once."""
    if which == "tiebreak":
        news = jd.streams_for("ag_news", tok, 0)[1:]
        used = {d["id"] for d in stated_sets.samples("wiki_articles", tok)["tuning"]}
        n_tok = lambda d: len(tok(d["text"], add_special_tokens=False)["input_ids"])
        cand = [dict(d, id=f"wiki-{d['id']}") for d in wikisection.load("validation") if len(d["spans"]) >= 5]
        art = random.Random(1).sample([d for d in cand if d["id"] not in used and 1000 <= n_tok(d) <= 2500], 1)
        pub = random.Random(0).sample(pubmed_rct.load("dev"), 16)[8:]
        return {"news": (jd.LISTS["ag_news"], news), "articles": (wikisection.STATES, art),
                "pubmed": (pubmed_rct.STATES, pub)}
    news = jd.streams_for("ag_news", tok, 0)[:1]
    art = [d for d in stated_sets.samples("wiki_articles", tok)["tuning"] if "Paroxysmal" in d["id"]]
    pub = random.Random(0).sample(pubmed_rct.load("dev"), 8)
    return {"news": (jd.LISTS["ag_news"], news), "articles": (wikisection.STATES, art),
            "pubmed": (pubmed_rct.STATES, pub)}


def out_file(spec, model_tag, which="screen"):
    return OUT / (f"{spec}__{model_tag}.jsonl" if which == "screen" else f"{spec}__{model_tag}__{which}.jsonl")


def run(spec, model_path, which="screen"):
    import torch
    from transformers import AutoModelForCausalLM
    from head_of_state.harness import tokens_and_gold
    from head_of_state.next_token import PerStateCacheReader, StatedNextTokenReader
    tok = jd.tokenizer()
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="cuda")
    import re
    m = re.search(r"models--([^/]+)", model_path)      # a Hugging Face cache path: name the repository
    tag = m.group(1) if m else Path(model_path.rstrip("/")).name
    OUT.mkdir(parents=True, exist_ok=True)
    with out_file(spec, tag, which).open("w") as f:
        for set_name, (states, docs) in screen_docs(tok, which).items():
            if SPECS[spec].get("kind") == "per_state_cache":
                ex = json.loads((RUNS / "state_vocab" / "examples.json").read_text())[set_name] if SPECS[spec].get("examples") else None
                reader = PerStateCacheReader(model, tok, states, examples=ex)
            else:
                reader = StatedNextTokenReader(model, tok, states, spec=SPECS[spec])
            for s in docs:
                ids, gold = tokens_and_gold(tok, s["text"], s["spans"])
                torch.cuda.synchronize(); t0 = time.time()
                lp, plain = reader.emissions(ids)
                torch.cuda.synchronize()
                f.write(json.dumps({"set": set_name, "id": s["id"], "n_tokens": len(ids), "gold": gold,
                                    "state_logp": lp, "plain_logp": plain, "seconds": time.time() - t0,
                                    "model": model_path, "spec": SPECS[spec]}) + "\n")
                f.flush()
        print(f"{spec} on {tag}: done", flush=True)


BLOCK, BOOT = 16, 2000


def _block_ci(v, rng):
    """95% interval of the mean of v by resampling 16-token blocks (consecutive evidence is correlated)."""
    n = len(v) // BLOCK
    if n < 2:
        return -np.inf, np.inf
    b = v[: n * BLOCK].reshape(n, BLOCK).mean(1)
    m = b[rng.integers(0, n, (BOOT, n))].mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def _arrays(recs):
    lp = np.vstack([np.asarray(r["state_logp"]) for r in recs])
    plain = np.concatenate([np.asarray(r["plain_logp"]) for r in recs])
    g = np.concatenate([np.asarray(r["gold"][1:]) for r in recs])
    return lp, plain, g


def evidence(lp, g, occ):
    """Per token: log e(true) minus the mean over the other occurring states."""
    own = lp[np.arange(len(g)), g]
    others = (lp[:, occ].sum(1) - own) / (len(occ) - 1)
    return own - others


def measures(recs, K, ref=None):
    """ref: records of the reference wording on the same documents, for the paired difference."""
    rng = np.random.default_rng(0)
    lp, plain, g = _arrays(recs)
    occ = [k for k in range(K) if (g == k).sum() >= 50]
    D, wrong = [], 0
    for i in occ:
        for j in occ:
            if i != j:
                d = lp[g == i, i] - lp[g == i, j]
                D.append(d.mean())
                wrong += _block_ci(d, rng)[1] < 0
    centred = lp - lp[:, occ].mean(1, keepdims=True)
    lift = [centred[(g != k) & np.isin(g, occ), k].mean() for k in occ]
    ms = 1000 * sum(r["seconds"] for r in recs) / sum(r["n_tokens"] for r in recs)
    ev = evidence(lp, g, occ)[np.isin(g, occ)]
    out = {"D": float(np.mean(D)), "wrong": f"{wrong}/{len(D)}", "lift": float(np.ptp(lift)),
           "shared": float((lp.mean(1) - plain).mean()), "ms": ms, "paired": None,
           # scale-free: a sharper model gives larger log-ratios for the same separation
           "pos": float((ev > 0).mean()), "snr": float(ev.mean() / ev.std())}
    if ref is not None:
        rlp, _, rg = _arrays(ref)
        keep = np.isin(g, occ)
        diff = (evidence(lp, g, occ) - evidence(rlp, rg, occ))[keep]
        out["paired"] = (float(diff.mean()), *_block_ci(diff, rng))
    return out


def report():
    sets = {"news": len(jd.LISTS["ag_news"]), "articles": len(wikisection.STATES), "pubmed": len(pubmed_rct.STATES)}
    print("Tuning documents only. Per set: D = mean pairwise evidence between occurring states (nats per token) · "
          "share of tokens whose evidence (true state against the mean of the others) is positive · that evidence's "
          "mean over its SD (scale-free across models) · "
          "pairs whose 95% interval is below zero · lift range (nats) · drop below the plain prediction (nats) · "
          "ms per token · paired difference in per-token evidence from the wording in PAIR_WITH, same model, same "
          "tokens, 95% interval by 16-token blocks. Leads under 0.1 nats per token count as ties (best-of-N "
          "selection). Cells marked mis-fit are not read (MISFIT).\n")
    print("| Wording (paired with) | model | " + " | ".join(sets) + " |\n|---|---|" + "---|" * len(sets))
    for f in sorted(OUT.glob("*__*.jsonl")):
        if f.stem.count("__") != 1:
            continue                       # tie-break files are reported separately
        spec, tag = f.stem.split("__")
        recs = list(map(json.loads, f.open()))
        ref_file = OUT / f"{PAIR_WITH.get(spec, '')}__{tag}.jsonl"
        ref = list(map(json.loads, ref_file.open())) if spec in PAIR_WITH and ref_file.exists() else []
        cells = []
        for s, K in sets.items():
            rs = [r for r in recs if r["set"] == s]
            rr = [r for r in ref if r["set"] == s and r["id"] in {x["id"] for x in rs}]
            if not rs:
                cells.append("—"); continue
            m = measures(rs, K, rr if rr else None)
            if (spec, s) in MISFIT:
                m["paired"] = None
            p = "" if m["paired"] is None else f" · {m['paired'][0]:+.2f} [{m['paired'][1]:+.2f}, {m['paired'][2]:+.2f}]"
            cells.append(f"{m['D']:.2f} · {m['pos']:.0%} · {m['snr']:.2f} · {m['wrong']} · {m['lift']:.2f} · "
                         f"{m['shared']:+.1f} · {m['ms']:.0f}{p}")
            if (spec, s) in MISFIT:
                cells[-1] = "mis-fit: " + cells[-1]
        print(f"| {spec} ({PAIR_WITH.get(spec, '—')}) | {tag} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    a = sys.argv
    if a[1] == "run":
        run(a[a.index("--spec") + 1],
            a[a.index("--model") + 1] if "--model" in a else os.environ.get("H2O_LIGHTNING_PATH", "h2oai/h2o-lightning-4b"),
            "tiebreak" if "--tiebreak" in a else "screen")
    else:
        report()
