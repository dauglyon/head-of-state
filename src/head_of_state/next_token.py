"""State-conditioned next-token emissions.

At every position t the stream x_1..x_t sits in the key/value cache. For each declared state k a
temporary branch states "the text is in state k here" and the head's next-token distribution is read at the
branch's end; the probability it gives the token that actually comes next is that token's emission:

    e_{t+1}(k) = P(x_{t+1} | x_1..x_t, state k)

This is the regime density of a Markov-switching model (Hamilton, "Regime switching models", eq. 7): it
depends on the current state and the past text only, so the usual forward filter, smoother and Viterbi
apply, and the product of emissions over tokens is the likelihood of the text under a state path. Nothing
that names a state stays in the cache: every branch is removed after its reading (StreamStateReader._branch_all).

Two forms of the branch:
- "instruction" (the owner's form): the stream sits in the user turn of the model's chat format, and the
  branch closes that turn with a question in the words a reader would be asked, "This part of the text is
  about <description>. What word comes next?", and opens the answer with the prefill "Next word:", where the
  Jev-style question used to go. The answer slot's distribution is read at the actual next token (the next
  word's first token). A chat answer writes a word with or without its leading
  space, so the emission is the probability of the token plus that of its other spacing (when that is a
  single token); the same mapping is used for every state.
- "bracket": plain text after the document separator, branch " [name: description]" (kept for comparison).
The plain next-token probability (no branch) is kept beside the K emissions.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .reader import StreamStateReader

STATEMENT = " [{name}: {description}]"
INSTRUCTION = "\n\nThis part of the text is about {description}. What word comes next?"
PREFILL = "Next word:"
_SPLIT = "\u241eSTREAM\u241e"


@dataclass
class StatedNextTokenReader(StreamStateReader):
    form: str = "instruction"
    statement: str = STATEMENT

    def __post_init__(self):
        import inspect
        self.model.eval()
        self.dev = next(self.model.parameters()).device
        params = inspect.signature(self.model.forward).parameters
        self._cache_kw = "cache" if "cache" in params else "past_key_values"
        self._keep_kw = "logits_to_keep" in params
        self.names = list(self.states)
        self.K = len(self.names)
        self.rotations, self.shifts = 1, [0]
        if self.form == "instruction":
            text = self.tok.apply_chat_template([{"role": "user", "content": "Text:\n" + _SPLIT}], tokenize=False,
                                                add_generation_prompt=True, enable_thinking=False)
            head, rest = text.split(_SPLIT)
            self.head_ids = self._enc(head)
            self.branches = [self._enc(INSTRUCTION.format(name=n, description=d) + rest + PREFILL)
                             for n, d in self.states.items()]
        else:
            sep = self.tok.convert_tokens_to_ids("<|endoftext|>")
            self.head_ids = [sep] if isinstance(sep, int) and sep >= 0 else self._enc("\n")
            self.branches = [self._enc(self.statement.format(name=n, description=d)) for n, d in self.states.items()]
        self._other_spacing = {}
        self.reset()

    def _targets(self, tid: int) -> list[int]:
        """The token, plus the same text with the other leading-space choice when that is one token."""
        if self.form != "instruction":
            return [tid]
        if tid not in self._other_spacing:
            t = self.tok.decode([tid])
            alt = t[1:] if t.startswith(" ") else " " + t
            ids = self.tok(alt, add_special_tokens=False)["input_ids"] if alt.strip() else []
            self._other_spacing[tid] = [tid] + ([ids[0]] if len(ids) == 1 and ids[0] != tid else [])
        return self._other_spacing[tid]

    @torch.no_grad()
    def _feed(self, ids):
        self.last_logits, self.cache = self._run(ids, self.cache)
        self.n_cached += len(ids)

    @torch.no_grad()
    def emissions(self, ids: list[int]):
        """Per token t >= 1 (0-based): log P(ids[t] | ids[:t], state k) for each k, and the plain
        log P(ids[t] | ids[:t]). Returns (state_logp [T-1, K], plain_logp [T-1])."""
        self.reset()
        state_lp, plain_lp = [], []
        for t in range(len(ids) - 1):
            self._feed([ids[t]])
            self.stream.append(ids[t])
            nxt = ids[t + 1]
            plain_lp.append(float(torch.log_softmax(self.last_logits, -1)[nxt]))
            rows = self._branch_all()
            tg = self._targets(nxt)
            state_lp.append([float(torch.logsumexp(torch.log_softmax(r, -1)[tg], 0)) for r in rows])
            if self.n_cached + max(len(b) for b in self.branches) + 1 > self.max_ctx:
                raise RuntimeError("stream longer than max_ctx; re-anchoring is not implemented here")
        return state_lp, plain_lp
