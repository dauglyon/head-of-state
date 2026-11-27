"""Mechanics test for head_of_state.next_token on a tiny random-weight model (plumbing only).

  1. each state's emission, read through its temporary branch on the cache, equals the same probability
     recomputed from scratch over the instruction, the stream so far and the branch
  2. the cache holds only the instruction and the stream after every token (no branch is left behind)
  3. the decoder runs on the emissions and gives one label per token
"""
import string

import numpy as np
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from head_of_state.cli import emission_rows
from head_of_state.next_token import StatedNextTokenReader
from head_of_state.reader import DEMO_SEGMENTS, DEMO_STATES, build_stream
from head_of_state.settle import online_labels, settled_labels

torch.manual_seed(0)
chars = sorted(set(string.printable) | {"␞"})
vocab = {c: i for i, c in enumerate(["<unk>", "<bos>"] + chars)}
t = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
t.pre_tokenizer = pre_tokenizers.Split("", "isolated")
tok = PreTrainedTokenizerFast(tokenizer_object=t, unk_token="<unk>", bos_token="<bos>")
tok.chat_template = ("{% for m in messages %}<{{ m['role'] }}>{{ m['content'] }}{% endfor %}"
                     "{% if add_generation_prompt %}<assistant>{% endif %}")
model = GPT2LMHeadModel(GPT2Config(vocab_size=len(vocab), n_positions=4096, n_embd=64, n_layer=3, n_head=4,
                                   bos_token_id=1, eos_token_id=1)).eval()

names = list(DEMO_STATES)
ids, gold = build_stream(tok, DEMO_SEGMENTS[:2], names)
ids = ids[:60]
for form in ("catalogue", "instruction"):
    rd = StatedNextTokenReader(model, tok, DEMO_STATES, form=form, max_ctx=4096)
    lp, plain = rd.emissions(ids)
    assert rd.n_cached == len(rd.head_ids) + len(ids) - 1, "cache holds more than the instruction and stream"
    worst = 0.0
    with torch.no_grad():
        for t_ in (10, 35, 57):
            for k, b in enumerate(rd.branches):
                x = torch.tensor([rd.head_ids + ids[:t_ + 1] + b])
                ref = float(torch.log_softmax(model(input_ids=x).logits[0, -1].float(), -1)[ids[t_ + 1]])
                worst = max(worst, abs(ref - lp[t_][k]))
            x = torch.tensor([rd.head_ids + ids[:t_ + 1]])
            worst = max(worst, abs(float(torch.log_softmax(model(input_ids=x).logits[0, -1], -1)[ids[t_ + 1]]) - plain[t_]))
    assert worst < 1e-4, f"{form}: branch emission differs from full recompute by {worst}"
    print(f"{form}: branch emissions equal a full recompute (max diff {worst:.1e})")

e = emission_rows(lp, len(ids))
pos = np.arange(1, len(ids) + 1)
assert e.shape == (len(ids), len(names)) and np.allclose(e.sum(1), 1)
assert len(online_labels(e, pos, 30.0)) == len(ids) and len(settled_labels(e, pos, 30.0)) == len(ids)
print("decoder gives one live and one settled label per token")
print("all next-token mechanics tests passed")

# Per-state caches: each state's emission equals a full recompute of its head plus the stream so far.
from head_of_state.next_token import PerStateCacheReader
tok.add_special_tokens({"additional_special_tokens": []})
ps = PerStateCacheReader(model, tok, DEMO_STATES)
lp, plain = ps.emissions(ids)
worst = 0.0
with torch.no_grad():
    for t_ in (10, 35, 57):
        for k, h in enumerate(ps.heads):
            x = torch.tensor([h + ids[:t_ + 1]])
            ref = float(torch.log_softmax(model(input_ids=x).logits[0, -1].float(), -1)[ids[t_ + 1]])
            got = lp[t_][k] if k < len(lp[t_]) else plain[t_]
            worst = max(worst, abs(ref - got))
assert worst < 1e-4, f"per-state cache emission differs from full recompute by {worst}"
print(f"per-state caches: emissions equal a full recompute (max diff {worst:.1e})")
