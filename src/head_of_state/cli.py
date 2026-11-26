"""Head of State: label every token of a text stream with hidden states defined in plain English.

At every token the model's output head predicts the next token once per state, and the probability it gives
the token that actually comes next is that token's emission for the state (head_of_state.next_token). An
HMM-like decoder with sticky transitions turns the emissions into a probability for each state at every token
(forward pass, live), the likeliest state, and settled segments (Viterbi over the whole stream).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from .reader import DEMO_SEGMENTS, DEMO_STATES, build_stream

DEFAULT_MODEL = "h2oai/h2o-lightning-4b"


def emission_rows(state_logp: list[list[float]], n_tokens: int) -> np.ndarray:
    """Per-token emissions normalised over states, [n_tokens, K]; the first token has no emission (nothing
    precedes it) and gets a flat row."""
    lp = np.asarray(state_logp, dtype=np.float64).reshape(-1, len(state_logp[0]) if state_logp else 1)
    K = lp.shape[1]
    e = np.exp(np.vstack([np.zeros((1, K)), lp - lp.max(1, keepdims=True)]))[:n_tokens]
    return e / e.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="Hugging Face id or local path of a causal language model with a chat template\n"
                         "(default: H2O-Lightning-4B)")
    ap.add_argument("--demo", action="store_true", help="run the built-in support-call stream")
    ap.add_argument("--text", help="path to a text file to label instead of the demo")
    ap.add_argument("--states", help="path to a JSON file {name: description}")
    ap.add_argument("--segment-length", type=float, default=30.0,
                    help="declared typical segment length in tokens (default 30): the transition model")
    ap.add_argument("--save", help="write the per-token emissions, live and settled labels to this JSON file")
    a = ap.parse_args()
    if not (a.demo or a.text):
        ap.error("give --demo or --text")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .metrics import forward_posteriors
    from .next_token import StatedNextTokenReader
    from .settle import online_labels, settled_labels
    sub = os.path.join(a.model, "tokenizer")
    if os.path.isdir(sub):
        tok = AutoTokenizer.from_pretrained(sub)          # H2O-Lightning keeps its tokenizer in a subfolder
    elif "h2o-lightning" in a.model.lower() and not os.path.isdir(a.model):
        tok = AutoTokenizer.from_pretrained(a.model, subfolder="tokenizer")
    else:
        tok = AutoTokenizer.from_pretrained(a.model)
    if not getattr(tok, "chat_template", None):
        ap.error("the model's tokenizer has no chat template; the state questions need one")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype, device_map="auto")

    states = json.load(open(a.states)) if a.states else DEMO_STATES
    names = list(states)
    if a.text:
        ids, gold = tok(open(a.text).read(), add_special_tokens=False)["input_ids"], None
    else:
        ids, gold = build_stream(tok, DEMO_SEGMENTS, names)

    reader = StatedNextTokenReader(model, tok, states, form="catalogue")
    state_lp, _ = reader.emissions(ids)
    e = emission_rows(state_lp, len(ids))
    pos = np.arange(1, len(ids) + 1)
    live = online_labels(e, pos, a.segment_length)
    prob = forward_posteriors(e, 1.0, a.segment_length)
    settled = settled_labels(e, pos, a.segment_length)

    print(f"{'token':>7} {'text':>16}  {'live state':<16} p")
    for t in range(len(ids)):
        mark = "  <-- switch" if t and live[t] != live[t - 1] else ""
        word = tok.decode([ids[t]]).replace("\n", "\\n")
        print(f"{t + 1:>7} {word!r:>16}  {names[live[t]]:<16} {prob[t].max():.2f}{mark}")
    print("settled segments:")
    start = 0
    for t in range(len(ids)):
        if t == len(ids) - 1 or settled[t + 1] != settled[t]:
            print(f"  tokens {start + 1:>5d}-{t + 1:<5d} {names[settled[t]]}")
            start = t + 1
    if gold:
        g = np.asarray(gold)
        print(f"token accuracy: live {np.mean(live == g):.3f}, settled {np.mean(settled == g):.3f}")
    if a.save:
        with open(a.save, "w") as f:
            json.dump({"states": names, "n_tokens": len(ids), "segment_length": a.segment_length,
                       "state_logp": state_lp, "probabilities": prob.tolist(), "live": live.tolist(),
                       "settled": settled.tolist(), "gold": gold}, f)


if __name__ == "__main__":
    main()
