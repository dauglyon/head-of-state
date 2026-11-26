"""Mechanics of the batched stream copies in head_of_state.readouts, on tiny random models:
  1. rows with openings of different length (left padding) give the same per-token log-probabilities
     as each opening run alone
  2. a temporary suffix branch leaves the cache unchanged: feeding afterwards matches never branching
  3. run_gen with a refresh interval longer than the stream equals run_gen without refresh
"""
import string

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast, Qwen3_5ForCausalLM,
                          Qwen3_5TextConfig)

from head_of_state.readouts import Copies, run_gen

torch.manual_seed(0)
chars = sorted(set(string.printable) | {"␞"})
vocab = {c: i for i, c in enumerate(["<unk>", "<bos>"] + chars)}
t = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
t.pre_tokenizer = pre_tokenizers.Split("", "isolated")
tok = PreTrainedTokenizerFast(tokenizer_object=t, unk_token="<unk>", bos_token="<bos>")
dev = "cuda" if torch.cuda.is_available() else "cpu"
gpt2 = GPT2LMHeadModel(GPT2Config(vocab_size=len(vocab), n_positions=1024, n_embd=64, n_layer=2, n_head=4,
                                  bos_token_id=1, eos_token_id=1)).eval()
hyb = Qwen3_5ForCausalLM(Qwen3_5TextConfig(
    vocab_size=len(vocab), hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
    num_key_value_heads=2, head_dim=16, linear_key_head_dim=16, linear_value_head_dim=16,
    linear_num_key_heads=2, linear_num_value_heads=4,
    layer_types=["linear_attention"] * 3 + ["full_attention"], tie_word_embeddings=True)).float().to(dev).eval()

enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
heads = [enc("Short head. "), enc("A considerably longer opening text here. "), enc("Mid-size head. ")]
stream = enc("The router keeps dropping the connection every evening.")

for label, m in (("gpt2", gpt2), ("hybrid", hyb)):
    batched = Copies(m, tok, heads)
    rows = torch.stack([batched.feed(x) for x in stream])                     # [T, R]
    alone = torch.stack([torch.stack([c.feed(x) for x in stream]) for c in
                         [Copies(m, tok, [h]) for h in heads]], 1)[..., 0]     # [T, R]
    d = (rows - alone).abs().max().item()
    assert d < 1e-4, f"{label}: padded rows differ from rows run alone by {d}"

    a, b = Copies(m, tok, heads), Copies(m, tok, heads)
    out_a, out_b = [], []
    for i, x in enumerate(stream):
        out_a.append(a.feed(x))
        out_b.append(b.feed(x))
        if i % 5 == 4:
            b.branch(enc(" Which state? Answer:"))
    d2 = (torch.stack(out_a) - torch.stack(out_b)).abs().max().item()
    assert d2 < 1e-5, f"{label}: branching changed later readings by {d2}"
    print(f"{label}: 1 ok padded rows vs alone {d:.1e}; 2 ok branch leaves cache unchanged ({d2:.1e})")

states = {"a": "about routers", "b": "about billing"}
s = {"id": "x", "text": "The router keeps dropping. I was charged twice.", "spans": [(0, 26, 0), (27, 47, 1)]}
g1 = torch.tensor(run_gen(hyb, tok, states, s)["logp"])
g2 = torch.tensor(run_gen(hyb, tok, states, s, every=10_000, keep=8)["logp"])
assert (g1 - g2).abs().max().item() < 1e-6
g3 = torch.tensor(run_gen(hyb, tok, states, s, every=10, keep=6)["logp"])
assert g3.shape == g1.shape and not torch.allclose(g1[15:], g3[15:])
print("3 ok  refresh longer than the stream changes nothing; a short refresh changes later scores")
print("all readout tests passed")

# 4: window_emissions with no window equals run_gen (copies fed one token at a time); with a window,
#    the emission of a token equals a direct pass over [description + its window context + token].
from head_of_state.readouts import GEN_NEUTRAL, _gen_heads  # noqa: E402
from head_of_state.window_likelihood import window_emissions  # noqa: E402
ids_s = enc(s["text"])
hs = _gen_heads(tok, states) + [enc(GEN_NEUTRAL)]
full = window_emissions(hyb, tok, hs, ids_s, None)[:, :-1]
assert (full - g1).abs().max().item() < 1e-4, (full - g1).abs().max().item()
W = 8
win = window_emissions(hyb, tok, hs, ids_s, W)
t_ = 21                                   # hop = 4: token 21 is in the hop starting at 20, context ids[12:20]
seq = hs[1] + ids_s[12:21]   # ends with token 20; its logits predict token 21
with torch.no_grad():
    ref = torch.log_softmax(hyb(torch.tensor([seq], device=dev)).logits[0, -1].float(), -1)[ids_s[21]].item()
assert abs(win[t_, 1].item() - ref) < 1e-4, (win[t_, 1].item(), ref)
print("4 ok  window emissions: no window equals token-by-token copies; windowed token matches a direct pass")
print("all readout tests passed")
