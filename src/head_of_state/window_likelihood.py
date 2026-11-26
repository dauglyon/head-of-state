"""Likelihood emissions with hopped windows (HANDOFF open question 3, extended).

One copy of the stream per state, opened by that state's description. Every `hop` (= W/2) tokens each
copy is rebuilt as [description + last W stream tokens]; the emission of token t under state k is
log p(x_t | description_k, the W to W+hop stream tokens before t). W=None means no windowing: the
description stays at the start and every emission sees the whole history.

Emissions are computed hop by hop in batched passes over [description + context + hop tokens],
reading the logits at the hop's positions. Online, these are the same numbers a copy fed one token
at a time would give; computing a hop at once only delays them to the hop's end.
"""
from __future__ import annotations

import time

import torch

from .harness import tokens_and_gold
from .readouts import GEN_NEUTRAL, GEN_PREFIX

MAX_LOGIT_POSITIONS = 1200   # positions x vocabulary held as float32 logits per batch (~1.2 GB)


def heads_for(tok, states) -> list[list[int]]:
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    return [enc(GEN_PREFIX.format(desc=d)) for d in states.values()] + [enc(GEN_NEUTRAL)]


@torch.no_grad()
def window_emissions(model, tok, heads: list[list[int]], ids: list[int], W: int | None) -> torch.Tensor:
    """[T, len(heads)] log-probabilities of each stream token under each head."""
    dev = next(model.parameters()).device
    T = len(ids)
    hop = T if W is None else max(1, W // 2)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    rows = []   # (head index, hop start, sequence, chunk length)
    for s in range(0, T, hop):
        ctx = ids[:s] if W is None else ids[max(0, s - W): s]
        chunk = ids[s: s + hop]
        for k, h in enumerate(heads):
            rows.append((k, s, h + ctx + chunk, len(chunk)))
    out = torch.empty(T, len(heads))
    keep = hop + 1
    per_batch = max(1, MAX_LOGIT_POSITIONS // keep)
    for b in range(0, len(rows), per_batch):
        batch = rows[b: b + per_batch]
        L = max(len(r[2]) for r in batch)
        x = torch.tensor([[pad] * (L - len(r[2])) + r[2] for r in batch], device=dev)
        mask = torch.tensor([[0] * (L - len(r[2])) + [1] * len(r[2]) for r in batch], device=dev)
        pos = (mask.cumsum(1) - 1).clamp_min(0)
        logits = model(input_ids=x, attention_mask=mask, position_ids=pos, use_cache=False,
                       logits_to_keep=keep).logits
        for i, (k, s, seq, c) in enumerate(batch):
            rows_logits = logits[i, -(c + 1):-1]                                # predicts chunk tokens
            for a in range(0, c, 256):                                          # bounded float32 slices
                lp = torch.log_softmax(rows_logits[a: a + 256].float(), -1)
                tgt = torch.tensor(ids[s + a: s + min(c, a + 256)], device=dev)
                out[s + a: s + a + len(tgt), k] = lp.gather(1, tgt[:, None])[:, 0].cpu()
        del logits
    return out


def run_window(model, tok, states, stream, W: int | None) -> dict:
    ids, gold = tokens_and_gold(tok, stream["text"], stream["spans"])
    heads = heads_for(tok, states)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    e = window_emissions(model, tok, heads, ids, W)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return {"kind": "emission", "logp": e[:, :-1].tolist(), "neutral": e[:, -1].tolist(), "gold": gold,
            "n_tokens": len(ids), "seconds": time.time() - t0, "W": W, "id": stream["id"]}
