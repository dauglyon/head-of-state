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
  <description>. What word comes next?" (a description carries its own phrasing: "about ...", "in English",
  "a yes-or-no question"), and opens the answer with the prefill "Next word:", where the
  Jev-style question used to go. The answer slot's distribution is read at the actual next token (the next
  word's first token). A chat answer writes a word with or without its leading
  space, so the emission is the probability of the token plus that of its other spacing (when that is a
  single token); the same mapping is used for every state.
- "catalogue": the instruction form with every state's name and description listed once at the start of the
  cache ("The text below is made of parts of these kinds: ..."), and each branch naming only its state ("This
  part of the text is <name>. What word comes next?"). A description tells the model what kind of text this is as
  well as which state, and that differs between descriptions, so it does not cancel between states; listing all
  of them once gives every branch the same information, and the branches differ only in the state they name.
  The catalogue names no particular state, so the emission still depends only on the current state and the text.
- "bracket": plain text after the document separator, branch " [name: description]" (kept for comparison).
The plain next-token probability (no branch) is kept beside the K emissions.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .reader import StreamStateReader

STATEMENT = " [{name}: {description}]"
INSTRUCTION = "\n\nThis part of the text is {description}. What word comes next?"
PREFILL = "Next word:"
CATALOGUE_HEAD = "The text below is made of parts of these kinds:\n{kinds}\n\nText:\n"
CATALOGUE_BRANCH = "\n\nThis part of the text is {name}. What word comes next?"
_SPLIT = "\u241eSTREAM\u241e"


@dataclass
class StatedNextTokenReader(StreamStateReader):
    form: str = "instruction"
    statement: str = STATEMENT
    # A wording given in full, overriding `form` (used to screen wordings, experiments/wording_screen.py):
    #   chat      True: the model's chat format, stream in the user turn; False: plain text
    #   system    system message (chat only) or None
    #   head      text before the stream; may contain {kinds} (one "- name: description" line per state)
    #   branch    the temporary text after the stream for one state; {name} and {description}
    #   answer    chat only: close the user turn and open the assistant turn after the branch (default True)
    #   repeat    after the branch, repeat the stream's last n tokens, so the next token continues visible text
    #   stream_in "assistant": the stream is the assistant's reply, `head` is the user turn, the branch follows the
    #             stream inside the reply (e.g. a thinking block) and the head is read at its end
    #   prefill   text after that (chat) or at the end of the branch (plain)
    #   spacing   also count the actual token with its other leading-space choice (default: same as answer)
    spec: dict | None = None

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
        if self.spec is not None:
            self._from_spec(self.spec)
        elif self.form in ("instruction", "catalogue"):
            opening = "Text:\n" if self.form == "instruction" else CATALOGUE_HEAD.format(
                kinds="\n".join(f"- {n}: {d}" for n, d in self.states.items()))
            text = self.tok.apply_chat_template([{"role": "user", "content": opening + _SPLIT}], tokenize=False,
                                                add_generation_prompt=True, enable_thinking=False)
            head, rest = text.split(_SPLIT)
            self.head_ids = self._enc(head)
            ask = INSTRUCTION if self.form == "instruction" else CATALOGUE_BRANCH
            self.branches = [self._enc(ask.format(name=n, description=d) + rest + PREFILL)
                             for n, d in self.states.items()]
        else:
            sep = self.tok.convert_tokens_to_ids("<|endoftext|>")
            self.head_ids = [sep] if isinstance(sep, int) and sep >= 0 else self._enc("\n")
            self.branches = [self._enc(self.statement.format(name=n, description=d)) for n, d in self.states.items()]
        self._other_spacing = {}
        self.reset()

    def _from_spec(self, sp: dict):
        kinds = "\n".join(f"- {n}: {d}" for n, d in self.states.items())
        head = sp.get("head", "").format(kinds=kinds)
        prefill = sp.get("prefill", "")
        if sp.get("chat", True) and sp.get("stream_in") == "assistant":
            # the stream is the assistant's own reply (after its opening empty thinking block); `head` is the user turn
            msgs = ([{"role": "system", "content": sp["system"]}] if sp.get("system") else []) + \
                   [{"role": "user", "content": head}]
            text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            self.head_ids = self._enc(text)
            tail = prefill
        elif sp.get("chat", True):
            msgs = ([{"role": "system", "content": sp["system"]}] if sp.get("system") else []) + \
                   [{"role": "user", "content": head + _SPLIT}]
            text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            before, rest = text.split(_SPLIT)
            self.head_ids = self._enc(before)
            tail = (rest + prefill) if sp.get("answer", True) else prefill
        else:
            sep = self.tok.convert_tokens_to_ids("<|endoftext|>")
            self.head_ids = ([sep] if isinstance(sep, int) and sep >= 0 else []) + self._enc(head)
            tail = prefill
        self.branches = [self._enc(sp["branch"].format(name=n, description=d) + tail) for n, d in self.states.items()]
        self._spacing = sp.get("spacing", sp.get("chat", True) and sp.get("answer", True))

    def _targets(self, tid: int) -> list[int]:
        """The token, plus the same text with the other leading-space choice when that is one token."""
        if self.spec is not None and not self._spacing:
            return [tid]
        if self.spec is None and self.form == "bracket":
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
        log P(ids[t] | ids[:t]). Returns (state_logp [T-1, K], plain_logp [T-1]). Also sets self.extra:
        state_explogp and state_varlogp [T-1, K], the mean and variance of log P(x | ids[:t], state k) for x drawn
        from the plain prediction (the reference of the Fast-DetectGPT statistic)."""
        self.reset()
        state_lp, plain_lp = [], []
        self.extra = {"state_explogp": [], "state_varlogp": []}
        for t in range(len(ids) - 1):
            self._feed([ids[t]])
            self.stream.append(ids[t])
            nxt = ids[t + 1]
            plain = torch.log_softmax(self.last_logits, -1)
            plain_lp.append(float(plain[nxt]))
            n_rep = (self.spec or {}).get("repeat", 0)
            if n_rep:      # the branch ends with the stream's last n_rep tokens again, so the prediction resumes the text
                fixed = self.branches
                self.branches = [b + self.stream[-n_rep:] for b in fixed]
                rows = torch.log_softmax(torch.stack(self._branch_all()), -1).clamp_min(-1e4)
                self.branches = fixed
            else:
                rows = torch.log_softmax(torch.stack(self._branch_all()), -1).clamp_min(-1e4)   # [K, vocab]
            tg = self._targets(nxt)
            state_lp.append(torch.logsumexp(rows[:, tg], 1).tolist())
            w = plain.exp()
            m = rows @ w
            self.extra["state_explogp"].append(m.tolist())
            self.extra["state_varlogp"].append(((rows ** 2) @ w - m ** 2).tolist())
            if self.n_cached + max(len(b) for b in self.branches) + 1 > self.max_ctx:
                raise RuntimeError("stream longer than max_ctx; re-anchoring is not implemented here")
        return state_lp, plain_lp


# The states listed, then the state of the end of the text (where the next token is predicted), directly before
# the stream.
PER_STATE_HEAD = ("The following text is written in stretches, and each stretch is in one of these states. The state "
                  "can change from one stretch to the next.\n\nStates:\n{kinds}\n\n"
                  "The end of the following text is in state: {name}\n\n")


@dataclass
class PerStateCacheReader:
    """One key/value cache per state. Cache k opens with a sentence saying the text is written in stretches, each in
    one of the listed states, the list ("States:", one "- name: description" line each) and "The end of the
    following text is in state: <name>", then holds the stream as plain text; each stream token is fed once into every cache. The head's
    distribution at the stream's last token of cache k is P(next token | text so far, state k): nothing follows the
    stream, so the head predicts the text itself. A further cache opens with the list only and gives the plain
    prediction. Same interface as StatedNextTokenReader.emissions."""
    model: torch.nn.Module
    tok: object
    states: dict[str, str]
    head: str = PER_STATE_HEAD
    max_ctx: int = 8192
    examples: dict | None = None      # optional {state name: text}, placed after the list as "Text in state <name>:"

    def __post_init__(self):
        import inspect
        self.model.eval()
        self.dev = next(self.model.parameters()).device
        self._cache_kw = "cache" if "cache" in inspect.signature(self.model.forward).parameters else "past_key_values"
        self.names = list(self.states)
        self.K = len(self.names)
        kinds = "\n".join(f"- {n}: {d}" for n, d in self.states.items())
        enc = lambda s: self.tok(s, add_special_tokens=False)["input_ids"]
        sep = self.tok.convert_tokens_to_ids("<|endoftext|>")
        start = [sep] if isinstance(sep, int) and sep >= 0 else []
        def head_for(n, d):
            h = self.head.format(kinds=kinds, name=n, description=d)
            if self.examples:     # examples of the state's text go before the line naming the end's state
                cut = h.index("The end of")
                h = h[:cut] + f"Text in state {n}:\n{self.examples[n].strip()}\n\n" + h[cut:]
            return h
        self.heads = [start + enc(head_for(n, d)) for n, d in self.states.items()]
        self.heads.append(start + enc(self.head.split("The end of")[0].format(kinds=kinds)))   # plain: list only

    @torch.no_grad()
    def _step(self, ids, cache):
        x = torch.tensor([ids], device=self.dev)
        out = self.model(input_ids=x, use_cache=True, **{self._cache_kw: cache})
        new = out[self._cache_kw] if self._cache_kw in out.keys() else out.past_key_values
        return torch.log_softmax(out.logits[0, -1].float(), -1), new

    @torch.no_grad()
    def emissions(self, ids: list[int]):
        """Per token t >= 1 (0-based): log P(ids[t] | ids[:t], state k) for each k, and the plain log P(ids[t] |
        ids[:t]). Returns (state_logp [T-1, K], plain_logp [T-1])."""
        if max(len(h) for h in self.heads) + len(ids) > self.max_ctx:
            raise RuntimeError("stream longer than max_ctx")
        caches = []
        for h in self.heads:
            _, c = self._step(h, None)
            caches.append(c)
        state_lp, plain_lp = [], []
        for t in range(len(ids) - 1):
            row = []
            for i in range(len(caches)):
                lp, caches[i] = self._step([ids[t]], caches[i])
                row.append(float(lp[ids[t + 1]]))
            state_lp.append(row[:-1]); plain_lp.append(row[-1])
        self.extra = {}
        return state_lp, plain_lp
