"""Mechanics test for head_of_state.reader on a tiny random-weight model.

Checks plumbing only (a random model cannot classify anything):
  1. a reading taken through the cached branch equals a reading recomputed from scratch
  2. the cache is the same length after a reading as before it
  3. option-order rotation maps letters back to the right states
  4. re-anchoring keeps the stream running past the context limit
  5. the commit rule and the scoring function behave on a hand-made case
"""
import string

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from head_of_state.reader import (DEMO_SEGMENTS, DEMO_STATES, Reading, StreamStateReader,
                           build_stream, score)

torch.manual_seed(0)


def tiny():
    chars = sorted(set(string.printable) | {"␞"})
    vocab = {c: i for i, c in enumerate(["<unk>", "<bos>"] + chars)}
    t = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    t.pre_tokenizer = pre_tokenizers.Split("", "isolated")
    tok = PreTrainedTokenizerFast(tokenizer_object=t, unk_token="<unk>", bos_token="<bos>")
    cfg = GPT2Config(vocab_size=len(vocab), n_positions=4096, n_embd=64, n_layer=3, n_head=4,
                     bos_token_id=1, eos_token_id=1)
    return GPT2LMHeadModel(cfg), tok


model, tok = tiny()
names = list(DEMO_STATES)
ids, gold = build_stream(tok, DEMO_SEGMENTS[:2], names)
ids, gold = ids[:120], gold[:120]

# 1 + 2: cached branch == full recompute, cache length unchanged
rd = StreamStateReader(model, tok, DEMO_STATES, stride=40, rotations=5, persist=2, max_ctx=4096)
n0 = rd.n_cached
out = rd.push_tokens(ids)
assert rd.n_cached == n0 + len(ids), "cache length drifted after readings"
assert len(out) == 3 and [r.pos for r in out] == [40, 80, 120]
with torch.no_grad():
    acc = torch.zeros(rd.K)
    for shift, br in zip(rd.shifts, rd.branches):
        full = torch.tensor([rd.head_ids + ids + br])
        lp = torch.log_softmax(model(full).logits[0, -1][rd.letters], -1)
        for slot in range(rd.K):
            acc[(slot + shift) % rd.K] += lp[slot]
    ref = torch.softmax(acc / len(rd.shifts), -1)
diff = (ref - torch.tensor(out[-1].probs)).abs().max().item()
assert diff < 1e-4, f"cached reading differs from full recompute by {diff}"
print(f"1-2 ok  cached vs recomputed max diff = {diff:.2e}; cache length stable")

# 3: every rotation lists every state exactly once and shifts are distinct
assert len(rd.shifts) == 5 and len(set(rd.shifts)) == 5
for s in rd.shifts:
    q = rd._question(s)
    assert all(q.count(f". {n}:") == 1 for n in names)
print("3   ok  rotations cover each state once per ordering")

# 4: re-anchoring
small = StreamStateReader(model, tok, DEMO_STATES, stride=50, rotations=1, max_ctx=900, keep_ctx=100)
long_ids = (ids * 6)[:600]
res = small.push_tokens(long_ids)
assert len(res) == 12 and small.n_cached <= 900
assert all(abs(sum(r.probs) - 1) < 1e-4 for r in res)
print(f"4   ok  ran {len(long_ids)} tokens through a 900-token limit; cache now {small.n_cached}")

# 5: commit rule + scoring on a hand-made sequence (persist=2)
rd2 = StreamStateReader(model, tok, DEMO_STATES, persist=2)
raws = [0, 0, 1, 0, 1, 1, 1, 0, 1, 1]         # one flicker at index 2, a real switch at index 4
got = [rd2._commit(r) for r in raws]
assert [s for s, _ in got] == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1], got
assert [sw for _, sw in got].count(True) == 1 and got[5][1]
fake = [Reading(i + 1, [0.0] * 5, r, s, sw) for i, (r, (s, sw)) in enumerate(zip(raws, got))]
m = score(fake, [0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
assert m["true_switches"] == 1 and m["switches_found"] == 1 and m["mean_delay_tokens"] == 1.0
assert m["spurious_switches"] == 0 and m["accuracy_committed"] == 0.9 and m["accuracy_raw"] == 0.8
print("5   ok  flicker ignored, switch committed after 2 readings;", m)


# 5b: switches are found only by commit events (gold A-B-A; the stream returns to A)
def committed(seq):
    return [Reading(i + 1, [], s, s, i > 0 and s != seq[i - 1]) for i, s in enumerate(seq)]


A, B, C = 0, 1, 2
aba = [A] * 10 + [B] * 10 + [A] * 10
cases = {   # predicted committed states -> (switches_found, spurious_switches, mean_delay_tokens)
    "stays on A": ([A] * 30, (0, 0, None)),
    "stays on A, one false switch": ([A] * 25 + [C] * 5, (0, 1, None)),
    "false A->C->A in segment 1": ([A] * 3 + [C] * 3 + [A] * 24, (0, 2, None)),
    "follows both switches late": ([A] * 12 + [B] * 11 + [A] * 7, (2, 0, 2.5)),
}
for name, (pred, want) in cases.items():
    m = score(committed(pred), aba)
    got = (m["switches_found"], m["spurious_switches"], m["mean_delay_tokens"])
    assert got == want, f"{name}: got {got}, want {want}"
print("5b  ok  switches found only by commit events; staying put earns no credit")

# 6: hybrid linear-attention model (Qwen3.5 layout: 3 Gated DeltaNet layers per attention layer).
#    After readings, every cached state must equal that of a reader that took no readings, and
#    must NOT equal it when the restore is skipped (proves the restore is what keeps them equal).
#    States are compared directly: a random model's letter probabilities are near uniform, so
#    comparing readings alone cannot tell a corrupted cache from a correct one.
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

qcfg = Qwen3_5TextConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                         linear_key_head_dim=16, linear_value_head_dim=16, linear_num_key_heads=2,
                         linear_num_value_heads=4, layer_types=["linear_attention"] * 3 + ["full_attention"],
                         tie_word_embeddings=True)
# On the GPU when there is one: with flash-linear-attention installed, Qwen3.5 layers run its
# GPU-only kernels, the same path the real model takes.
qdev = "cuda" if torch.cuda.is_available() else "cpu"
qmodel = Qwen3_5ForCausalLM(qcfg).float().to(qdev)


def recompute(r, stream):
    with torch.no_grad():
        acc = torch.zeros(r.K)
        for shift, br in zip(r.shifts, r.branches):
            lp = torch.log_softmax(qmodel(torch.tensor([r.head_ids + stream + br], device=qdev)).logits[0, -1].cpu()[r.letters], -1)
            for slot in range(r.K):
                acc[(slot + shift) % r.K] += lp[slot]
        return torch.softmax(acc / len(r.shifts), -1)


hy = StreamStateReader(qmodel, tok, DEMO_STATES, stride=40, rotations=2, max_ctx=4096)
outq = hy.push_tokens(ids)
diffs = [(recompute(hy, ids[: r.pos]) - torch.tensor(r.probs)).abs().max().item() for r in outq]
assert max(diffs) < 1e-4, f"hybrid cached reading differs from recompute by {max(diffs)}"

clean = StreamStateReader(qmodel, tok, DEMO_STATES, stride=10_000, max_ctx=4096)   # never reads
clean.push_tokens(ids)
clean.flush()
broken = StreamStateReader(qmodel, tok, DEMO_STATES, stride=40, rotations=2, max_ctx=4096)
broken._linear_state = lambda layer: None if not hasattr(layer, "recurrent_states") else []
broken._branch_all = lambda: [broken._branch(b) for b in broken.branches]   # one-at-a-time path
broken.push_tokens(ids)


def state_gap(a, b):
    gap = 0.0
    for la, lb in zip(a.cache.layers, b.cache.layers):
        ta = StreamStateReader._linear_state(la) or [la.keys, la.values]
        tb = StreamStateReader._linear_state(lb) or [lb.keys, lb.values]
        gap = max([gap] + [(x - y).abs().max().item() / (y.abs().max().item() or 1) for x, y in zip(ta, tb)])
    return gap


kept, lost = state_gap(hy, clean), state_gap(broken, clean)
assert kept < 1e-5, f"states after readings differ from a reader that took none by {kept}"
assert lost > 1e-2, f"skipping the restore changed the states by only {lost}; test has no power"
print(f"6   ok  hybrid model: readings vs recompute {max(diffs):.1e}; cached states vs no-reading "
      f"reader {kept:.1e}; without restore {lost:.1e}")

# 7: batched branches (one pass for all orderings) equal running each branch alone, on both cache
#    types, including branches of unequal length (right padding), and leave the cache unchanged.
for label, mdl in (("gpt2", model), ("hybrid", qmodel)):
    rb = StreamStateReader(mdl, tok, DEMO_STATES, stride=10_000, rotations=3, max_ctx=4096)
    rb.push_tokens(ids[:60])
    rb.flush()
    rb.branches[1] = rb.branches[1][:-7]           # unequal lengths
    before = StreamStateReader(mdl, tok, DEMO_STATES, stride=10_000, max_ctx=4096)
    before.push_tokens(ids[:60])
    before.flush()
    batched = rb._branch_all()
    single = [rb._branch(b) for b in rb.branches]
    d = max((x - y).abs().max().item() / y.abs().max().item() for x, y in zip(batched, single))
    assert d < 1e-4, f"{label}: batched branches differ from single branches by {d}"
    assert state_gap(rb, before) < 1e-6, f"{label}: batched branches changed the cache"
    print(f"7   ok  {label}: batched vs one-at-a-time branches, max relative diff {d:.1e}; cache unchanged")

# 8: fixed-lag Viterbi. Full lag equals a brute-force best path; lag 0 emits argmax of delta_t;
#    every lag emits exactly one label per position; a large lag equals the offline path.
import itertools  # noqa: E402
from head_of_state.reader import FixedLagViterbi, sticky_log_transition  # noqa: E402

g = torch.Generator().manual_seed(1)
e = torch.log_softmax(torch.randn(7, 3, generator=g, dtype=torch.float64) * 2, -1)
logA = sticky_log_transition(3, 4.0)
best = max(itertools.product(range(3), repeat=7),
           key=lambda path: float(e[0, path[0]] + sum(logA[path[i - 1], path[i]] + e[i, path[i]] for i in range(1, 7))))
v = FixedLagViterbi(logA, None)
full = sum((v.push(r) for r in e), []) + v.flush()
assert full == list(best), (full, best)
for lag in (0, 2, 5, 100):
    v = FixedLagViterbi(logA, lag)
    out = sum((v.push(r) for r in e), []) + v.flush()
    assert len(out) == 7, (lag, out)
    if lag >= 7:
        assert out == list(best)
print("8   ok  fixed-lag Viterbi: offline path is the brute-force best path; one label per position at every lag")

# 9: a reading interval (stride) still labels every token: each reading covers the tokens since the
#    previous one, finish() reads the tail, and token accuracy counts every covered token.
rs = StreamStateReader(model, tok, DEMO_STATES, stride=7, rotations=1, max_ctx=4096)
got = rs.push_tokens(ids[:60]) + rs.finish()
assert [r.pos for r in got] == [7, 14, 21, 28, 35, 42, 49, 56, 60], [r.pos for r in got]
assert [r.start for r in got] == [1, 8, 15, 22, 29, 36, 43, 50, 57]
assert rs.finish() == [] and len(StreamStateReader.token_states(got)) == 60
labels = StreamStateReader.token_states(got)
m = score(got, gold[:60])
assert abs(m["accuracy_committed"] - sum(a == b for a, b in zip(labels, gold[:60])) / 60) < 1e-9
print("9   ok  stride 7: every token labelled by the reading that follows it; token accuracy over all 60 tokens")

# 10: delay counts from the token where the true switch happens: with readings every 4th token, a
#     switch at token 6 committed by the reading at token 8 has delay 2 (not 0 from the reading).
rs10 = [Reading(4 * (i + 1), [], s, s, i > 0 and s != [0, 1, 1][i - 1]) for i, s in enumerate([0, 1, 1])]
m10 = score(rs10, [0] * 5 + [1] * 7)
assert m10["switches_found"] == 1 and m10["mean_delay_tokens"] == 2.0, m10
print("10  ok  delay counted from the switch token (switch at token 6, committed at token 8: 2 tokens)")

# 11: refine. Every probe and every regular reading equals an every-token reader's reading at the same
#     position (both see the full history), the regular readings equal those of the same reader without
#     refine (the main cache is untouched), and probes happen only where two readings disagree.
for label, mdl in (("gpt2", model), ("hybrid", qmodel)):
    every = {r.pos: r.probs for r in StreamStateReader(mdl, tok, DEMO_STATES, rotations=1, max_ctx=4096).push_tokens(ids)}
    plain = StreamStateReader(mdl, tok, DEMO_STATES, rotations=1, stride=8, max_ctx=4096)
    plain_out = plain.push_tokens(ids) + plain.finish()
    ref = StreamStateReader(mdl, tok, DEMO_STATES, rotations=1, stride=8, refine=True, max_ctx=4096)
    out = ref.push_tokens(ids) + ref.finish()
    d = max(abs(a - b) for r in out for a, b in zip(r.probs, every[r.pos]))
    assert d < 1e-4, f"{label}: refined readings differ from every-token readings by {d}"
    regular = {r.pos: r.probs for r in out if r.pos % 8 == 0 or r.pos == len(ids)}
    d2 = max(abs(a - b) for r in plain_out for a, b in zip(r.probs, regular[r.pos]))
    assert d2 < 1e-5, f"{label}: refine changed the regular readings by {d2}"
    assert [r.start for r in out][1:] == [r.pos + 1 for r in out][:-1]       # contiguous spans
    n_probe = len(out) - len(plain_out)
    n_change = sum(1 for a, b in zip(plain_out, plain_out[1:]) if a.raw != b.raw)
    assert n_change == 0 or n_probe > 0
    print(f"11  ok  {label}: {n_probe} probes at {n_change} disagreements; probes match every-token readings ({d:.1e})")
# 11b: the tiny hybrid model gives one answer everywhere, so force a disagreement before every reading
#      (mark the previous reading as a different state): every reading then bisects, exercising the
#      Gated DeltaNet snapshot and restore; probes must still equal every-token readings.
every = {r.pos: r.probs for r in StreamStateReader(qmodel, tok, DEMO_STATES, rotations=1, max_ctx=4096).push_tokens(ids)}
forced = StreamStateReader(qmodel, tok, DEMO_STATES, rotations=1, stride=8, refine=True, max_ctx=4096)
out = []
for i in range(0, len(ids), 8):
    if forced.readings:
        forced.readings[-1].raw = (forced.readings[-1].raw + 1) % forced.K
    out += forced.push_tokens(ids[i: i + 8])
n_probe = sum(1 for r in out if r.pos % 8)
d = max(abs(a - b) for r in out for a, b in zip(r.probs, every[r.pos]))
assert n_probe >= 3 * (len(ids) // 8 - 1) and d < 1e-4, (n_probe, d)
print(f"11b ok  hybrid with forced disagreements: {n_probe} probes, all equal every-token readings ({d:.1e})")
print("all mechanics tests passed")
