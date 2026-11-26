"""State-conditioned next-token emissions on the joined-document streams (head_of_state.next_token).

    H2O_LIGHTNING_PATH=<checkout> uv run python -I experiments/stated_next_token.py run [--model h2o|qwen] [--form instruction|bracket] [--streams first|rest] [list ...]
    uv run python -I experiments/stated_next_token.py report           # results/stated_next_token.md

Tuning streams (seed 0): the first stream of each state list (--streams first) or the other two (rest); every token. Emission of token t:
e_t(k) = P(x_t | x_<t, state k), the state given on a temporary branch at t-1 (head_of_state.next_token: the
instruction form "This part of the text is about <description>. What word comes next?" with the answer slot read, or the
bracket form for comparison).
Measured before any decoding (the go / no-go checks):
  - information rate D(i, j): mean over tokens whose true state is i of log e(i) - log e(j), with its SD and
    the lag-1 autocorrelation of the per-token values;
  - calibration: the slope b of a logistic fit of "i is the true state" on the pairwise log-ratio (b = 1:
    the ratios are calibrated and weight 1 is right), and each state's offset: how much its statement
    raises the probability of text in other states, relative to the other statements;
  - shared distortion: mean over k of log e(k) minus the plain log P(x_t | x_<t) with no branch.
Then the HMM-like decoder with these emissions at weight 1: Viterbi with the declared length (settled labels)
and the max-product forward pass (time to notice), against the Jev-style readout on the same stream.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import joined_documents as jd  # noqa: E402

MODELS = {"h2o": os.environ.get("H2O_LIGHTNING_PATH", "h2oai/h2o-lightning-4b"), "qwen": "Qwen/Qwen3.5-4B"}


FORMS = ("instruction", "bracket")


PARTS = {"first": [0], "rest": [1, 2]}   # which seed-0 streams of each list


def out_file(model, form="instruction", part="first"):
    stem = f"{model}_seed0_{part}" if form == "bracket" else f"{model}_{form}_seed0_{part}"
    return jd.OUT / "stated" / f"{stem}.jsonl"


def run(which, form, part, lists=None):
    import torch
    from transformers import AutoModelForCausalLM
    from head_of_state.harness import tokens_and_gold
    from head_of_state.next_token import StatedNextTokenReader
    tok = jd.tokenizer()
    model = AutoModelForCausalLM.from_pretrained(MODELS[which], dtype=torch.bfloat16, device_map="cuda")
    f = out_file(which, form, part)
    f.parent.mkdir(parents=True, exist_ok=True)
    with f.open("w") as out:
        for name, states in jd.LISTS.items():
            if lists and name not in lists:
                continue
            # the frame is "This part of the text is {description}."; these descriptions were written for
            # "...is about {description}", so "about " keeps the wording these runs used
            reader = StatedNextTokenReader(model, tok, {k: "about " + v for k, v in states.items()}, form=form)
            streams = jd.streams_for(name, tok, 0)
            for i in PARTS[part]:
                s = streams[i]
                ids, gold = tokens_and_gold(tok, s["text"], s["spans"])
                torch.cuda.synchronize(); t0 = time.time()
                lp, plain = reader.emissions(ids)
                torch.cuda.synchronize()
                out.write(json.dumps({"list": name, "id": s["id"], "n_tokens": len(ids), "gold": gold, "state_logp": lp,
                                      "plain_logp": plain, "seconds": time.time() - t0}) + "\n")
                out.flush()
                print(f"{s['id']}: {len(ids)} tokens, {1000 * (time.time() - t0) / len(ids):.0f} ms/token", flush=True)


def logistic_slope(x, y, iters=50):
    """b in P(y=1) = sigmoid(b x), by Newton's method (no intercept: the pairs are symmetric)."""
    b = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-b * x))
        g, h = ((y - p) * x).sum(), (p * (1 - p) * x * x).sum()
        if h <= 0:
            break
        b += g / h
    return b


def lag1(v):
    v = v - v.mean()
    return float((v[1:] * v[:-1]).sum() / max((v * v).sum(), 1e-12))


def checks(r, names):
    """The go / no-go checks for one record of K emissions per token."""
    K = len(names)
    lp = np.asarray(r["state_logp"]); plain = np.asarray(r["plain_logp"])
    g = np.asarray(r["gold"][1:])                       # emission row t is for token t + 1
    D, sds, ac, xs, ys = {}, [], [], [], []
    for i in range(K):
        m = g == i
        if m.sum() < 20:
            continue
        for j in range(K):
            if j != i:
                d = lp[m, i] - lp[m, j]
                D[(i, j)] = d.mean(); sds.append(d.std()); ac.append(lag1(d))
                xs += [d, -d]; ys += [np.ones(len(d)), np.zeros(len(d))]
    centred = lp - lp.mean(1, keepdims=True)
    off = [centred[g != k, k].mean() for k in range(K) if (g != k).any()]
    w = min(D, key=D.get)
    return {"D": float(np.mean(list(D.values()))), "weakest": f"{names[w[0]]} vs {names[w[1]]}: {D[w]:.3f}",
            "sd": float(np.mean(sds)), "lag1": float(np.mean(ac)),
            "slope": logistic_slope(np.concatenate(xs), np.concatenate(ys)),
            "offsets": f"{min(off):+.2f} to {max(off):+.2f}", "shared": float((lp.mean(1) - plain).mean())}


def emission_probs(r, K):
    """Emissions normalised over states per token, for the decoders (weight 1); the first token, which has no
    emission, gets a flat row. Positions are 1-based token counts."""
    lp = np.asarray(r["state_logp"])
    e = np.exp(np.vstack([np.zeros((1, K)), lp - lp.max(1, keepdims=True)]))
    return e / e.sum(1, keepdims=True), np.arange(1, r["n_tokens"] + 1)


def report():
    from head_of_state.metrics import pool, stream_counts
    from head_of_state.settle import online_labels, settled_labels
    from head_of_state.decoders import probs
    print("Tuning streams (seed 0), first stream per list, every token. Emission: P(actual next token | text so far, "
          "state stated on a temporary branch).\n")
    print("| Model, branch form | State list | tokens | D, mean over state pairs (nats/token) | weakest pair | SD per token | "
          "lag-1 autocorr. | calibration slope b | state offsets, range | shared distortion | ms/token |\n"
          "|---|---|---|---|---|---|---|---|---|---|---|")
    decoded = []
    for which, form, part in [(m, fm, pt) for m in MODELS for fm in FORMS for pt in PARTS]:
        f = out_file(which, form, part)
        if not f.exists():
            continue
        for r in map(json.loads, f.open()):
            name = r["list"]
            names = list(jd.LISTS[name])
            c = checks(r, names)
            print(f"| {which}, {form} | {r['id']} | {r['n_tokens']} | {c['D']:.3f} | {c['weakest']} | {c['sd']:.2f} | "
                  f"{c['lag1']:.2f} | {c['slope']:.2f} | {c['offsets']} | {c['shared']:+.2f} | "
                  f"{1000 * r['seconds'] / r['n_tokens']:.0f} |")
            p, pos = emission_probs(r, len(names))
            L = jd.DECLARED[name]
            sc = stream_counts(settled_labels(p, pos, L), p, r["gold"], pos)
            oc = stream_counts(online_labels(p, pos, L), p, r["gold"], pos)
            jev = next(x for x in map(json.loads, jd.out_file(name, 0).open()) if x["id"] == r["id"])
            q, qpos = probs(jev, rotations=1)
            jsc = stream_counts(settled_labels(q, qpos, L), q, jev["gold"], qpos)
            joc = stream_counts(online_labels(q, qpos, L), q, jev["gold"], qpos)
            decoded.append((f"{which}, {form}", r["id"], r["n_tokens"], sc, oc, jsc, joc))
    print("\nDecoded, weight 1, declared length. Cells: token accuracy · switches placed within 16 tokens · false "
          "switches per 1,000 tokens · time to notice (tokens).\n")
    print("| Model, branch form | State list | state-conditioned next token | Jev-style readout, same stream |\n|---|---|---|---|")
    for which, name, n, sc, oc, jsc, joc in decoded:
        cell = lambda s, o: (f"{pool([s], n)['accuracy']:.3f} · {s['match'][16]}/{s['true_b']} · "
                             f"{1000 * (s['pred_b'] - s['match'][16]) / n:.1f} · {pool([o], n)['mean_delay']:.0f}")
        print(f"| {which} | {name} | {cell(sc, oc)} | {cell(jsc, joc)} |")


if __name__ == "__main__":
    a = sys.argv
    if a[1] == "run":
        run(a[a.index("--model") + 1] if "--model" in a else "h2o", a[a.index("--form") + 1] if "--form" in a else "instruction",
            a[a.index("--streams") + 1] if "--streams" in a else "first", [x for x in a if x in jd.LISTS])
    else:
        report()
