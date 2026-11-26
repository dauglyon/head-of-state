"""Switchboard dialogue acts (SwDA, github.com/cgpotts/swda, GPL-2.0) as streams with one state per utterance.

    uv run python -I experiments/swda.py measure      # act list and declared length from the training conversations

Data: data/swda/swda/sw*utt/*.utt.csv (experiments/data/download.sh). Published reference: Stolcke et al. 2000
(Computational Linguistics 26(3)), 42 acts, an HMM over utterances with act n-gram transitions and per-act word
likelihoods, 71% utterance accuracy on transcripts (chance 35%, human 84%).
- Acts: the SWBD-DAMSL tag collapsed as the corpus code does (swda.py, Utterance.damsl_act_tag); "+" (an
  utterance continued after the other speaker's interruption) takes the same speaker's previous act. Acts with
  at least 2% of the training utterances are states; the rest are "other".
- Text: transcription markup removed ({D ...} and similar braces keep their words; [ ] + / -- # dropped;
  <laughter> and other sounds kept as words in parentheses). A new line and the speaker's letter start each
  change of speaker; utterances within a turn are joined by a space, so act changes inside a turn carry no
  formatting cue.
- Split: conversations sorted by number; TUNING are 2 and CONFIRM 12 drawn with seed 0 from the last 100;
  all others are the training conversations for the act list and the declared length.
"""
from __future__ import annotations

import csv
import json
import random
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402

DATA = common.DATA / "swda" / "swda"
MIN_SHARE = 0.02


def collapse(tag: str) -> str:
    """Utterance.damsl_act_tag of the corpus code."""
    out = []
    for t in re.split(r"\s*[,;]\s*", tag):
        if t in ("qy^d", "qw^d", "b^m"):
            pass
        elif t == "nn^e":
            t = "ng"
        elif t == "ny^e":
            t = "na"
        else:
            t = re.sub(r"(.)\^.*", r"\1", t)
            t = re.sub(r"[\(\)@*]", "", t)
            t = {"qr": "qy", "fe": "ba", "oo": "oo_co_cc", "co": "oo_co_cc", "cc": "oo_co_cc", "fx": "sv",
                 "aap": "aap_am", "am": "aap_am", "arp": "arp_nd", "nd": "arp_nd", "fo": "fo_o_fw_by_bc",
                 "o": "fo_o_fw_by_bc", "fw": "fo_o_fw_by_bc", '"': "fo_o_fw_by_bc", "by": "fo_o_fw_by_bc",
                 "bc": "fo_o_fw_by_bc"}.get(t, t)
        out.append(t)
    return out[0]


def clean(text: str) -> str:
    t = re.sub(r"<<?([^<>]*)>>?", lambda m: f"({m.group(1).strip().lower()})", text)
    t = re.sub(r"\{[A-Z]\s*", "", t).replace("}", "")
    t = re.sub(r"[\[\]+#/]|--+", " ", t)
    t = re.sub(r"\s+([,.?!])", r"\1", re.sub(r"\s+", " ", t)).strip()
    return t


def conversations():
    """{conversation number: [(caller, act, text)]} with "+" resolved."""
    out = {}
    for f in sorted(DATA.glob("sw*utt/*.utt.csv")):
        rows, last = [], {}
        for r in csv.DictReader(f.open()):
            act = collapse(r["act_tag"])
            if act == "+":
                act = last.get(r["caller"], "%")
            last[r["caller"]] = act
            text = clean(r["text"])
            if text:
                rows.append((r["caller"], act, text))
        out[int(f.name.split("_")[2].split(".")[0])] = rows
    return out


def split(convs):
    nums = sorted(convs)
    rng = random.Random(0)
    held = rng.sample(nums[-100:], 14)
    return [n for n in nums if n not in held], held[:2], held[2:]


def acts(convs, train):
    counts = {}
    for n in train:
        for _, a, _ in convs[n]:
            counts[a] = counts.get(a, 0) + 1
    total = sum(counts.values())
    return sorted((a for a, c in counts.items() if c / total >= MIN_SHARE), key=lambda a: -counts[a]), counts, total


def stream(num, rows, kept):
    """{id, text, spans} with one span per utterance; acts outside `kept` are "other" (index len(kept))."""
    text, spans, prev = "", [], None
    for caller, act, said in rows:
        g = kept.index(act) if act in kept else len(kept)
        if caller != prev:
            piece = ("\n" if text else "") + f"{caller}: {said}"
        else:
            piece = " " + said
        start = len(text)
        text += piece
        spans.append((start, len(text), g))
        prev = caller
    return {"id": f"swda-{num}", "text": text, "spans": spans}


def measure():
    import joined_documents as jd
    tok = jd.tokenizer()
    convs = conversations()
    train, tune, conf = split(convs)
    kept, counts, total = acts(convs, train)
    from head_of_state.harness import tokens_and_gold
    lens = []
    for n in train[:200]:
        s = stream(n, convs[n], kept)
        _, gold = tokens_and_gold(tok, s["text"], s["spans"])
        run = 1
        for a, b in zip(gold, gold[1:]):
            if a == b:
                run += 1
            else:
                lens.append(run); run = 1
        lens.append(run)
    print(json.dumps({"states": kept + ["other"], "shares": {a: round(counts[a] / total, 3) for a in kept},
                      "other_share": round(1 - sum(counts[a] for a in kept) / total, 3),
                      "declared_length_tokens": statistics.median(lens), "train_conversations": len(train),
                      "tuning": tune, "confirmation": conf}, indent=1))


if __name__ == "__main__":
    {"measure": measure}[sys.argv[1]]()
