"""Candidate readouts for the design comparison. Each turns a token stream into one row of K scores per
token, so the decoders and scorer in decode.py apply to all of them unchanged.

  qafter   the current design: question appended after the stream at every token (StreamStateReader)
  qbefore  question and options in the cached opening text; at every token only a short answer
           prompt is appended. One stream copy per option ordering, fed together as one batch.
  gen      next-token likelihood: one stream copy per state, each opening with that state's
           description; the score of state k at token t is log p(x_t | description_k, x_<t),
           read from the pass that feeds x_{t-1}. No question at all.
  genrefresh  gen, but every `every` tokens each copy is rebuilt as description + the last `keep`
           stream tokens, so the description never sits more than every+keep tokens back.
  embed    cosine similarity between an embedding of the last `window` tokens and an embedding of
           each state description (frozen sentence-embedding model).

Record format (JSON line): {id, n_tokens, gold, kind, logp: [T][K], seconds, design, params}.
kind "posterior": logp are per-token state log-probabilities; kind "emission": logp are per-token
log-likelihoods of the observed token under each state (decode with the forward filter).
"""
from __future__ import annotations

import string
import time

import torch

from .harness import DEFAULT_UNIT, UnitReader, tokens_and_gold

LETTERS = string.ascii_uppercase


# ----------------------------------------------------------------------------- batched stream copies
class Copies:
    """R copies of one stream, each with its own opening text, kept in one batched cache.

    Openings of unequal length are left-padded (attention mask 0); position ids count only real
    tokens, so each row matches that row run alone. `feed` appends one stream token to every row in
    one pass and keeps each row's next-token log-probabilities. `branch` runs a temporary suffix on
    every row and restores the cache by putting each layer's attributes back (linear-attention
    states are cloned first because the pass updates them in place).
    """

    def __init__(self, model, tok, heads: list[list[int]]):
        self.model, self.tok = model, tok
        self.dev = next(model.parameters()).device
        self.pad = tok.pad_token_id if tok.pad_token_id is not None else 0
        self.start(heads)

    @torch.no_grad()
    def start(self, heads: list[list[int]]):
        R, L = len(heads), max(len(h) for h in heads)
        x = torch.tensor([[self.pad] * (L - len(h)) + h for h in heads], device=self.dev)
        self.mask = torch.tensor([[0] * (L - len(h)) + [1] * len(h) for h in heads], device=self.dev)
        pos = (self.mask.cumsum(1) - 1).clamp_min(0)
        out = self.model(input_ids=x, attention_mask=self.mask, position_ids=pos, use_cache=True,
                         logits_to_keep=1)
        self.cache = out["cache"] if "cache" in out.keys() else out.past_key_values
        self.n_real = self.mask.sum(1)
        self.next_logp = torch.log_softmax(out.logits[:, -1].float(), -1)   # [R, V]

    @torch.no_grad()
    def feed(self, token: int) -> torch.Tensor:
        """Log-probability each row gave `token` before seeing it; then append it to every row."""
        lp = self.next_logp[:, token].cpu()
        R = self.mask.shape[0]
        self.mask = torch.cat([self.mask, torch.ones(R, 1, dtype=self.mask.dtype, device=self.dev)], 1)
        out = self.model(input_ids=torch.full((R, 1), token, device=self.dev), attention_mask=self.mask,
                         position_ids=self.n_real[:, None], use_cache=True, cache=self.cache, logits_to_keep=1)
        self.n_real = self.n_real + 1
        self.next_logp = torch.log_softmax(out.logits[:, -1].float(), -1)
        return lp

    @torch.no_grad()
    def branch(self, suffix: list[int]) -> torch.Tensor:
        """Next-token logits of every row after a temporary suffix; the cache is left unchanged."""
        layers = self.cache.layers
        snaps = [{k: (dict(v) if isinstance(v, dict) else v) for k, v in vars(l).items()} for l in layers]
        for layer in layers:
            if hasattr(layer, "recurrent_states"):
                for name in ("conv_states", "recurrent_states"):
                    setattr(layer, name, {i: (None if t is None else t.clone())
                                          for i, t in getattr(layer, name).items()})
        R, S = self.mask.shape[0], len(suffix)
        mask = torch.cat([self.mask, torch.ones(R, S, dtype=self.mask.dtype, device=self.dev)], 1)
        pos = self.n_real[:, None] + torch.arange(S, device=self.dev)[None]
        try:
            out = self.model(input_ids=torch.tensor([suffix] * R, device=self.dev), attention_mask=mask,
                             position_ids=pos, use_cache=True, cache=self.cache, logits_to_keep=1)
        finally:
            for layer, snap in zip(layers, snaps):
                vars(layer).clear()
                vars(layer).update(snap)
        return out.logits[:, -1].float().cpu()


def _timed(fn):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    out = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return out, time.time() - t0


# ----------------------------------------------------------------------------- readouts
def run_qafter(model, tok, states, stream, unit=DEFAULT_UNIT, rotations=2):
    ids, gold = tokens_and_gold(tok, stream["text"], stream["spans"])
    r = UnitReader(model, tok, states, rotations=rotations, unit=unit)

    def go():
        rows = []
        for t in ids:
            r.push_tokens([t])
            rows.append(torch.tensor(r.last_shift_logp).mean(0).tolist())
        return rows
    rows, sec = _timed(go)
    return {"kind": "posterior", "logp": rows, "gold": gold, "n_tokens": len(ids), "seconds": sec}


QB_INSTRUCTION = (
    "You will be shown a text stream that is still arriving. The stream can move between the "
    "states listed below over time. When the stream pauses you will be asked which state "
    "{unit} of the stream is in.\n\nStates:\n{options}\n\nStream so far:\n")
QB_QUESTION = "\n\n[stream paused here]\n\nWhich state is {unit} of the stream in? Reply with the letter only."


def _qbefore_parts(tok, states, unit, shift):
    names = list(states)
    K = len(names)
    opts = "\n".join(f"{LETTERS[s]}. {names[(s + shift) % K]}: {states[names[(s + shift) % K]]}" for s in range(K))
    split = "␞STREAM␞"
    msg = [{"role": "user", "content": QB_INSTRUCTION.format(unit=unit, options=opts) + split +
            QB_QUESTION.format(unit=unit)}]
    text = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    head, tail = text.split(split)
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    return enc(head), enc(tail + "Answer:")


def run_qbefore(model, tok, states, stream, unit=DEFAULT_UNIT, rotations=2):
    ids, gold = tokens_and_gold(tok, stream["text"], stream["spans"])
    K = len(states)
    shifts = sorted({(i * K) // rotations for i in range(rotations)})
    parts = [_qbefore_parts(tok, states, unit, s) for s in shifts]
    suffix = parts[0][1]
    assert all(p[1] == suffix for p in parts)
    letters = [tok(" " + L, add_special_tokens=False)["input_ids"] for L in LETTERS[:K]]
    assert all(len(x) == 1 for x in letters), "option letters are not single tokens"
    letters = [x[0] for x in letters]

    def go():
        c = Copies(model, tok, [p[0] for p in parts])
        rows = []
        for t in ids:
            c.feed(t)
            lg = c.branch(suffix)
            acc = torch.zeros(K)
            for row, s in enumerate(shifts):
                lp = torch.log_softmax(lg[row, letters], -1)
                for slot in range(K):
                    acc[(slot + s) % K] += lp[slot]
            rows.append((acc / len(shifts)).tolist())
        return rows
    rows, sec = _timed(go)
    return {"kind": "posterior", "logp": rows, "gold": gold, "n_tokens": len(ids), "seconds": sec}


GEN_PREFIX = "The following text is {desc}.\n\n"
GEN_NEUTRAL = "The following text is a passage.\n\n"


def _gen_heads(tok, states):
    return [tok(GEN_PREFIX.format(desc=d), add_special_tokens=False)["input_ids"] for d in states.values()]


def run_gen(model, tok, states, stream, every=None, keep=None):
    """every/keep None: description only at the start. Otherwise rebuild each copy every `every`
    tokens as description + last `keep` stream tokens."""
    ids, gold = tokens_and_gold(tok, stream["text"], stream["spans"])
    # One extra copy opens with a neutral sentence: its log-probabilities are the stream's plain
    # surprisal (a possible switch signal), recorded beside the state scores at one more batch row.
    heads = _gen_heads(tok, states) + [tok(GEN_NEUTRAL, add_special_tokens=False)["input_ids"]]

    def go():
        c = Copies(model, tok, heads)
        rows = []
        for i, t in enumerate(ids):
            rows.append(c.feed(t).tolist())
            if every and (i + 1) % every == 0:
                c.start([h + ids[max(0, i + 1 - keep): i + 1] for h in heads])
        return rows
    rows, sec = _timed(go)
    return {"kind": "emission", "logp": [r[:-1] for r in rows], "neutral": [r[-1] for r in rows],
            "gold": gold, "n_tokens": len(ids), "seconds": sec}


def run_embed(emb_model, emb_tok, llm_tok, states, stream, window=64, tau=0.05, batch=64):
    """Embedding of the decoded last `window` LLM tokens (CLS pooling, normalised; bge-m3 style)."""
    ids, gold = tokens_and_gold(llm_tok, stream["text"], stream["spans"])
    dev = next(emb_model.parameters()).device

    @torch.no_grad()
    def emb(texts):
        x = emb_tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(dev)
        h = emb_model(**x).last_hidden_state[:, 0]
        return torch.nn.functional.normalize(h.float(), dim=-1)

    def go():
        d = emb([f"{n}: {s}" for n, s in states.items()])
        rows = []
        texts = [llm_tok.decode(ids[max(0, i + 1 - window): i + 1]) for i in range(len(ids))]
        for b in range(0, len(texts), batch):
            sims = emb(texts[b: b + batch]) @ d.T
            rows += torch.log_softmax(sims / tau, -1).cpu().tolist()
        return rows
    rows, sec = _timed(go)
    return {"kind": "posterior", "logp": rows, "gold": gold, "n_tokens": len(ids), "seconds": sec}
