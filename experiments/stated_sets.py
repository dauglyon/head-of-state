"""State-conditioned next-token emissions on test sets from prior work (head_of_state.next_token).

    uv run python -I experiments/stated_sets.py measure                       # declared lengths (training data)
    H2O_LIGHTNING_PATH=<checkout> uv run python -I experiments/stated_sets.py run --set wiki_articles --sample tuning
    uv run python -I experiments/stated_sets.py report --sample tuning        # results/stated_sets_tuning.md

Sets (chosen by a second session the owner delegated the choice to; protocol: tuning sample for the go / no-go
checks only, everything frozen before a confirmation sample of 12 documents nothing has read):
  wiki_articles  WikiSection en_disease whole articles in their real section order (CC BY-SA 3.0), the 10
                 merged section groups of wikisection.py. Tuning: 3 validation articles of 1,000-2,500 tokens
                 (seed 0); confirmation: 12 test articles (seed 0). Declared length: median merged-section length
                 of the training articles. Published reference: SECTOR Pk 26.3, C99 37.4 (Arnold et al. 2019).
  swda           Switchboard dialogue acts (swda.py): 7 acts with at least 2% of training utterances, plus
                 other. Tuning: 2 conversations; confirmation: 12. Declared length: median run of one act in the
                 training conversations. Published reference: 71% utterance accuracy (Stolcke et al. 2000).
  tripod         TRIPOD movie synopses (tripod.py): the six story stages between the five turning points. Tuning:
                 3 training movies; confirmation: 12 test movies (seed 0). Declared length: median stage of the
                 other training movies. Published reference (turning points, not stages): total agreement 41.33,
                 human 64 (Papalampidi et al. 2019, Table 4).
  newsdiscourse  NewsDiscourse (newsdiscourse.py): the role of each sentence relative to the main event, 9 roles.
                 Tuning: 3 validation articles; confirmation: 12 test articles (seed 0). Declared length: median
                 run of one role in the training articles. Published reference: macro / micro F1 38.3 / 45.7
                 for word-frequency features, 54.4 / 60.9 for a document-level LSTM (Choubey et al. 2020, Table 4).
H2O-Lightning-4B, the owner's wording ("This part of the text is <description>. What word comes next?"),
every token, weight 1, the declared length. Descriptions written for this evaluation.
"""
from __future__ import annotations

import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import joined_documents as jd  # noqa: E402
import newsdiscourse  # noqa: E402
import swda  # noqa: E402
import tripod  # noqa: E402
import wikisection  # noqa: E402
from common import RUNS  # noqa: E402
from stated_next_token import checks, emission_probs  # noqa: E402

OUT = RUNS / "stated_sets"
DECLARED = {"wiki_articles": 184.0, "swda": 9.0, "tripod": 124.0, "newsdiscourse": 42.0}     # `measure` (training data); fixed before any reading

SWDA_STATES = {
    "statement": "a statement of fact or a description: the speaker tells what happened or how things are",
    "backchannel": "a short backchannel such as uh-huh, yeah or right, said while the other person is talking",
    "opinion": "a statement of opinion: the speaker says what they think or feel about something",
    "abandoned": "an abandoned, interrupted or unintelligible fragment of speech",
    "agreement": "agreement with or acceptance of what the other person just said, such as that's right or exactly",
    "yes-no question": "a yes-or-no question",
    "appreciation": "an appreciation of what the other person said, such as oh that's great or wow",
    "other": "another kind of utterance: a question asking what, why or how, an answer, a greeting or goodbye, "
             "an acknowledgement, or a sound such as laughter",
}
# Stage names from Papalampidi, Keller & Lapata 2019 (sec. 1); each stage is described by the turning points
# that open and close it (their Table 1 definitions).
TRIPOD_STATES = {
    "setup": "the setup of a story: the setting and the background of the main characters are presented",
    "new situation": "the new situation of a story: an introductory event has occurred after the setup, and the "
                     "main goal of the story is not yet defined",
    "progress": "the progress of a story: the main goal of the story has been defined and the action begins to "
                "increase, before the main characters fully commit to their goal",
    "complications": "the complications and higher stakes of a story: the main characters are fully committed to "
                     "their goal, before everything falls apart",
    "final push": "the final push of a story: everything has fallen apart, and the main characters push on toward "
                  "the final event of the main story",
    "aftermath": "the aftermath of a story: after its final event and moment of resolution",
}
# Roles from the NewsDiscourse annotation guidelines (Choubey et al. 2020).
NEWS_ROLES = {
    "main event": "about the main event of a news story: the most recent event that gave rise to the report, or a "
                  "restatement of it by those directly involved",
    "consequence": "about real consequences of a news story's main event, which happened right after it and because of it",
    "previous event": "about a specific, recent previous event that caused a news story's main event or was a "
                      "precondition for it",
    "current context": "about the general circumstances or the ongoing situation in which a news story's main event "
                       "took place",
    "historical event": "about events from well before a news story's main event, at least two months earlier or in "
                        "previous years",
    "anecdotal event": "about an anecdotal event with specific participants that cannot be verified or may happen "
                       "in the future",
    "evaluation": "an opinion or comment on the events of a news story",
    "expectation": "about expectations of how the events of a news story will be resolved or what their "
                   "consequences will be",
    "not applicable": "not part of the news story itself, such as a photo caption, an editor's credit or a link",
}
STATES = {"wiki_articles": {k: "about " + v for k, v in wikisection.STATES.items()}, "swda": SWDA_STATES,
          "tripod": TRIPOD_STATES, "newsdiscourse": NEWS_ROLES}


def samples(name, tok):
    """{"tuning": [stream], "confirmation": [stream], "train": [stream]} for one set."""
    if name == "wiki_articles":
        n_tok = lambda d: len(tok(d["text"], add_special_tokens=False)["input_ids"])
        val = [d for d in wikisection.load("validation") if len(d["spans"]) >= 3 and 1000 <= n_tok(d) <= 2500]
        test = [d for d in wikisection.load("test") if len(d["spans"]) >= 3 and 1000 <= n_tok(d) <= 2500]
        pick = lambda ds, k: [dict(d, id=f"wiki-{d['id']}") for d in random.Random(0).sample(ds, k)]
        return {"tuning": pick(val, 3), "confirmation": pick(test, 12), "train": wikisection.load("train")}
    if name == "tripod":
        train = tripod.load("train")
        tune = random.Random(0).sample(train, 3)
        return {"tuning": tune, "confirmation": random.Random(0).sample(tripod.load("test"), 12),
                "train": [d for d in train if d not in tune]}
    if name == "newsdiscourse":
        return {"tuning": random.Random(0).sample(newsdiscourse.load("validation"), 3),
                "confirmation": random.Random(0).sample(newsdiscourse.load("test"), 12),
                "train": newsdiscourse.load("train")}
    convs = swda.conversations()
    train, tune, conf = swda.split(convs)
    kept, _, _ = swda.acts(convs, train)
    assert [a for a in kept] == ["sd", "b", "sv", "%", "aa", "qy", "ba"], kept   # SWDA_STATES order
    mk = lambda ns: [swda.stream(n, convs[n], kept) for n in ns]
    return {"tuning": mk(tune), "confirmation": mk(conf), "train": mk(train[:200])}


def measure():
    from head_of_state.harness import tokens_and_gold
    tok = jd.tokenizer()
    for name in STATES:
        lens = []
        for s in samples(name, tok)["train"]:
            _, gold = tokens_and_gold(tok, s["text"], s["spans"])
            run = 1
            for a, b in zip(gold, gold[1:]):
                if a == b:
                    run += 1
                else:
                    lens.append(run); run = 1
            lens.append(run)
        print(f"{name}: median segment {statistics.median(lens)} tokens ({len(lens)} segments)")


def out_file(name, sample, form="instruction"):
    return OUT / (f"{name}_{sample}.jsonl" if form == "instruction" else f"{name}_{sample}_{form}.jsonl")


UNITS = {"swda": "the last utterance"}     # question readout only; the others use joined_documents.UNIT


def run(name, sample, form="instruction"):
    """form: a next-token form of head_of_state.next_token, or "question" for the Jev-style readout with the
    frozen settings of joined_documents.py (the comparison on the same documents)."""
    import torch
    from transformers import AutoModelForCausalLM
    from head_of_state.harness import run_stream, tokens_and_gold
    from head_of_state.h2o import H2OReader
    from head_of_state.next_token import StatedNextTokenReader
    from scored_runs import FROZEN
    tok = jd.tokenizer()
    path = os.environ.get("H2O_LIGHTNING_PATH", "h2oai/h2o-lightning-4b")
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda")
    if form == "question":
        reader = H2OReader(model, tok, STATES[name], unit=UNITS.get(name, jd.UNIT), **FROZEN)
    else:
        reader = StatedNextTokenReader(model, tok, STATES[name], form=form)
    f = out_file(name, sample, form)
    f.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["id"] for l in f.open()} if f.exists() else set()
    with f.open("a") as out:
        for s in samples(name, tok)[sample]:
            if s["id"] in done:
                continue
            if form == "question":
                rec = dict(run_stream(reader, s, tok), set=name)
                out.write(json.dumps(rec) + "\n"); out.flush()
                print(f"{s['id']}: {rec['n_tokens']} tokens (question)", flush=True)
                continue
            ids, gold = tokens_and_gold(tok, s["text"], s["spans"])
            torch.cuda.synchronize(); t0 = time.time()
            lp, plain = reader.emissions(ids)
            torch.cuda.synchronize()
            out.write(json.dumps({"set": name, "id": s["id"], "n_tokens": len(ids), "gold": gold, "state_logp": lp,
                                  "plain_logp": plain, **reader.extra, "seconds": time.time() - t0}) + "\n")
            out.flush()
            print(f"{s['id']}: {len(ids)} tokens, {1000 * (time.time() - t0) / len(ids):.0f} ms/token", flush=True)


def unit_accuracy(labels, gold):
    """Share of gold segments (runs of one state) whose majority settled label is their state."""
    gold = np.asarray(gold); ok = n = 0
    start = 0
    for t in range(1, len(gold) + 1):
        if t == len(gold) or gold[t] != gold[start]:
            seg = np.asarray(labels[start:t])
            ok += np.bincount(seg, minlength=gold.max() + 1).argmax() == gold[start]; n += 1
            start = t
    return ok, n


def report(sample):
    from head_of_state.metrics import pool, stream_counts
    from head_of_state.settle import online_labels, settled_labels
    print(f"{sample.capitalize()} sample. Emission: P(actual next token | text so far, state) with the owner's wording; "
          "weight 1, declared length.\n")
    print("| Set | document | tokens | D (nats/token) | weakest pair | SD | lag-1 | calibration slope | state offsets | "
          "shared distortion | token acc. | segments right | placed within 16 | false /1k | time to notice | ms/token |\n|"
          + "---|" * 16)
    from head_of_state.decoders import probs
    for name, form in [(n, fm) for n in STATES for fm in ("instruction", "catalogue", "question")]:
        f = out_file(name, sample, form)
        if not f.exists():
            continue
        names = list(STATES[name])
        for r in map(json.loads, f.open()):
            if form == "question":
                p, pos = probs(r, rotations=1)
                c = {"D": float("nan"), "weakest": "—", "sd": float("nan"), "lag1": float("nan"),
                     "slope": float("nan"), "offsets": "—", "shared": float("nan")}
            else:
                c = checks(r, names)
                p, pos = emission_probs(r, len(names))
            lab = settled_labels(p, pos, DECLARED[name])
            sc = stream_counts(lab, p, r["gold"], pos)
            oc = stream_counts(online_labels(p, pos, DECLARED[name]), p, r["gold"], pos)
            ok, n = unit_accuracy(lab, r["gold"])
            print(f"| {name}{'' if form == 'instruction' else ', ' + form} | {r['id']} | {r['n_tokens']} | {c['D']:.3f} | {c['weakest']} | {c['sd']:.2f} | {c['lag1']:.2f} | "
                  f"{c['slope']:.2f} | {c['offsets']} | {c['shared']:+.2f} | {pool([sc], r['n_tokens'])['accuracy']:.3f} | "
                  f"{ok}/{n} | {sc['match'][16]}/{sc['true_b']} | {1000 * (sc['pred_b'] - sc['match'][16]) / r['n_tokens']:.1f} | "
                  f"{pool([oc], r['n_tokens'])['mean_delay']:.0f} | {1000 * r['seconds'] / r['n_tokens']:.0f} |")


if __name__ == "__main__":
    a = sys.argv
    if a[1] == "measure":
        measure()
    elif a[1] == "run":
        run(a[a.index("--set") + 1], a[a.index("--sample") + 1], a[a.index("--form") + 1] if "--form" in a else "instruction")
    else:
        report(a[a.index("--sample") + 1] if "--sample" in a else "tuning")
