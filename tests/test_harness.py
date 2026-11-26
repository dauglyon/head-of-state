"""The offline sweep is only valid if it reproduces the online reader. Checks, on a tiny model:
  1. UnitReader with the default unit gives the same probabilities as StreamStateReader
  2. probs() on a run with all K orderings equals an online run with fewer orderings, a stride and
     prior-power, and decode_persist equals the online committed states
  3. H1 gold: a token starting on the separator between two spans belongs to the next span
"""
import string

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from head_of_state.decoders import decode_persist, probs
from head_of_state.harness import UnitReader, run_stream, tokens_and_gold
from head_of_state.reader import DEMO_STATES, StreamStateReader

torch.manual_seed(0)
chars = sorted(set(string.printable) | {"␞"})
vocab = {c: i for i, c in enumerate(["<unk>", "<bos>"] + chars)}
t = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
t.pre_tokenizer = pre_tokenizers.Split("", "isolated")
tok = PreTrainedTokenizerFast(tokenizer_object=t, unk_token="<unk>", bos_token="<bos>")
model = GPT2LMHeadModel(GPT2Config(vocab_size=len(vocab), n_positions=2048, n_embd=64, n_layer=2,
                                   n_head=4, bos_token_id=1, eos_token_id=1))

text = "Hello there. My router keeps dropping. Unplug it and wait."
spans = [(0, 12, 0), (13, 38, 1), (39, len(text), 2)]
stream = {"id": "t", "text": text, "spans": spans}

# 1
a = StreamStateReader(model, tok, DEMO_STATES, rotations=2)
b = UnitReader(model, tok, DEMO_STATES, rotations=2)
assert a.branches == b.branches
ids, _ = tokens_and_gold(tok, text, spans)
pa = [r.probs for r in a.push_tokens(ids)]
pb = [r.probs for r in b.push_tokens(ids)]
assert max(abs(x - y) for ra, rb in zip(pa, pb) for x, y in zip(ra, rb)) < 1e-5
c = UnitReader(model, tok, DEMO_STATES, unit="the last sentence")
assert "Judge only the last sentence, not" in c._question(0)
print("1 ok  default unit reproduces the reader; custom unit lands in the question")

# 2
rec = run_stream(UnitReader(model, tok, DEMO_STATES, rotations=5), stream, tok)
online = StreamStateReader(model, tok, DEMO_STATES, rotations=2, stride=3, prior_power=0.75, persist=2)
ref = online.push_tokens(ids)
p, pos = probs(rec, rotations=2, stride=3, prior_power=0.75)
assert list(pos) == [r.pos for r in ref]
diff = max(abs(float(x) - y) for row, r in zip(p, ref) for x, y in zip(row, r.probs))
assert diff < 1e-5, diff
assert decode_persist(p, 2) == [r.state for r in ref]
print(f"2 ok  offline rotations/stride/prior-power/persist match the online reader (max diff {diff:.1e})")

# 3
_, gold = tokens_and_gold(tok, text, spans)
assert gold[11] == 0 and gold[12] == 1 and gold[13] == 1 and gold[-1] == 2
print("3 ok  separator tokens go to the following span")
print("all harness tests passed")
