"""head_of_state.reader (Head of State)

Zero-shot regime tracking from an open-weights LLM's output head.
Hidden-state inference from plain-English descriptions.

    Input : a token stream + a list of named states (given at run time)
    Output: for every position N, a probability for each state and the likeliest one,
            plus the positions where the state switches.

How it reads the state at position N
    1. The stream so far lives in the model's key/value cache (read once, never re-read).
    2. A short "which state?" question is appended as a temporary branch.
    3. The model's own output layer gives a probability for each option letter.
    4. The branch is discarded; the cache is exactly as it was.
    5. The reading at N is compared with the reading at N-1 to mark switches.

No weights are changed and nothing is fitted per problem.

Usage
    uv run head-of-state --demo
"""
from __future__ import annotations

import argparse
import copy
import inspect
import math
import string
from dataclasses import dataclass, field

import numpy as np
import torch

INSTRUCTION = (
    "You will be shown a text stream that is still arriving. "
    "The stream can move between different states over time. "
    "You will then be asked which state the MOST RECENT part of the stream is in.\n\n"
    "Stream so far:\n"
)
QUESTION_HEAD = (
    "\n\n[stream paused here]\n\n"
    "Which state is the most recent part of the stream in? "
    "Judge only the last few sentences, not the stream as a whole.\n"
)
QUESTION_TAIL = "Reply with the letter only."
ANSWER_PREFILL = "Answer:"
_SPLIT = "␞STREAM␞"


@dataclass
class Reading:
    pos: int                    # number of stream tokens read so far (N)
    probs: list[float]          # one probability per state, in the order given
    raw: int                    # likeliest state at N
    state: int                  # committed state at N (after the persistence rule)
    switched: bool              # True if the committed state changed at this reading
    start: int | None = None    # first token position this reading labels (None: only `pos`); with
                                # stride N a reading labels the tokens since the previous reading
    detail: list | None = None  # per-ordering letter log-probabilities, when the reader keeps them
    decided: int | None = None  # stream position at which this reading was available (None: `pos`);
                                # bisection probes are placed earlier than they are decided


@dataclass
class StreamStateReader:
    model: torch.nn.Module
    tok: object
    states: dict[str, str]              # name -> one-line description
    stride: int = 1                     # read every `stride` tokens (1 = every token)
    rotations: int = 2                  # option orderings averaged per reading (>=1)
    persist: int = 3                    # readings a new state must hold before it is committed
    prior_power: float = 0.0            # 0 = off; 0.75 divides out a running label prior
    max_ctx: int = 8192                 # re-anchor when the cache passes this many tokens
    keep_ctx: int = 2048                # stream tokens kept when re-anchoring
    refine: bool = False                # with stride > 1: bisect between two disagreeing readings
    readings: list[Reading] = field(default_factory=list)

    # ------------------------------------------------------------------ setup
    def __post_init__(self):
        self.model.eval()
        self.dev = next(self.model.parameters()).device
        params = inspect.signature(self.model.forward).parameters
        self._cache_kw = "cache" if "cache" in params else "past_key_values"
        self._keep_kw = "logits_to_keep" in params
        self.names = list(self.states)
        self.K = len(self.names)
        if not 2 <= self.K <= 26:
            raise ValueError("need between 2 and 26 states")
        self.rotations = max(1, min(self.rotations, self.K))
        head, tail = self._template()
        self.head_ids = self._enc(head, bos=True)
        self.letters = self._letter_ids(tail)
        # one question branch per option ordering; shifts are spread around the cycle
        self.shifts = sorted({(r * self.K) // self.rotations for r in range(self.rotations)})
        self.branches = [self._enc(self._question(s) + tail) for s in self.shifts]
        self.reset()

    def _template(self) -> tuple[str, str]:
        """Text before the stream, and text after the question (ends with the prefill)."""
        if getattr(self.tok, "chat_template", None):
            msg = [{"role": "user", "content": INSTRUCTION + _SPLIT + QUESTION_TAIL}]
            # enable_thinking=False: Qwen3/3.5 templates otherwise open a <think> block, and the
            # prefill would land inside the model's reasoning. Templates without the flag ignore it.
            text = self.tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True,
                                                enable_thinking=False)
            head, rest = text.split(_SPLIT)
            return head, rest + ANSWER_PREFILL
        return INSTRUCTION, QUESTION_TAIL + "\n" + ANSWER_PREFILL

    def _question(self, shift: int) -> str:
        lines = []
        for slot in range(self.K):
            k = (slot + shift) % self.K
            lines.append(f"{string.ascii_uppercase[slot]}. {self.names[k]}: {self.states[self.names[k]]}")
        return QUESTION_HEAD + "\n".join(lines) + "\n"

    def _enc(self, text: str, bos: bool = False) -> list[int]:
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        if bos and self.tok.bos_token_id is not None and (not ids or ids[0] != self.tok.bos_token_id):
            if not getattr(self.tok, "chat_template", None):
                ids = [self.tok.bos_token_id] + ids
        return ids

    def _letter_ids(self, tail: str) -> list[int]:
        """Token id the model would emit for each option letter right after the prefill."""
        base = self._enc(tail)
        out = []
        for sep in (" ", ""):
            out = []
            for L in string.ascii_uppercase[: self.K]:
                full = self._enc(tail + sep + L)
                if len(full) == len(base) + 1 and full[: len(base)] == base:
                    out.append(full[-1])
            if len(out) == self.K and len(set(out)) == self.K:
                return out
        raise ValueError("this tokenizer does not give single-token option letters")

    # -------------------------------------------------------------- streaming
    def reset(self):
        self.cache = None
        self.n_cached = 0
        self.stream: list[int] = []
        self._pending: list[int] = []
        self.readings = []
        self._last_read = 0
        self._prior = torch.full((self.K,), 1.0 / self.K)
        self._n_prior = 0
        self._committed = None
        self._cand, self._cand_run = None, 0
        self._feed(self.head_ids)

    def _run(self, ids: list[int], cache):
        """One forward pass that extends `cache`. Works with both cache argument names."""
        x = torch.tensor([ids], device=self.dev)
        out = self.model(input_ids=x, use_cache=True, **{self._cache_kw: cache})
        new = out[self._cache_kw] if self._cache_kw in out.keys() else out.past_key_values
        return out.logits[0, -1].float().cpu(), new

    @torch.no_grad()
    def _feed(self, ids: list[int]):
        _, self.cache = self._run(ids, self.cache)
        self.n_cached += len(ids)

    @torch.no_grad()
    def _branch(self, ids: list[int]) -> torch.Tensor:
        """Run a temporary question branch and return next-token logits; cache is restored."""
        layers = getattr(self.cache, "layers", None)
        if layers is not None:
            # Linear-attention layers (e.g. Qwen3.5's Gated DeltaNet) hold one running state that
            # the question overwrites in place, and `crop` cannot undo that. Save those states and
            # copy them back afterwards; per-token key/value layers are cropped.
            saved = [None if (s := self._linear_state(layer)) is None else [t.clone() for t in s]
                     for layer in layers]
            logits, self.cache = self._run(ids, self.cache)
            for layer, kept in zip(self.cache.layers, saved):
                if kept is None:
                    layer.crop(-len(ids))       # negative = remove that many tokens
                else:
                    for live, old in zip(self._linear_state(layer), kept):
                        live.copy_(old)
            assert self.cache.get_seq_length() == self.n_cached, "cache not restored"
        elif hasattr(self.cache, "crop"):
            logits, self.cache = self._run(ids, self.cache)
            self.cache.crop(-len(ids))
        else:  # tuple-style caches: branch from a copy
            logits, _ = self._run(ids, copy.deepcopy(self.cache))
        return logits

    @torch.no_grad()
    def _branch_all(self) -> list[torch.Tensor]:
        """Next-token logits after every question branch; the cache is left as it was.

        All branches run as one batch, so the weights are read once per reading instead of once
        per ordering. The cache is widened to one row per branch for that pass: key/value tensors
        are broadcast views (the pass concatenates onto them, so the originals are never written)
        and linear-attention states are copied (the pass updates them in place). Afterwards each
        layer's attributes are put back as they were, which also undoes any counters the pass bumped.
        Branches of unequal length are right-padded; padding sits after each branch's last real
        token, so causal attention keeps it from affecting that token's logits.
        """
        layers = getattr(self.cache, "layers", None)
        if layers is None or len(self.branches) == 1:
            return [self._branch(ids) for ids in self.branches]
        R, lens = len(self.branches), [len(b) for b in self.branches]
        L = max(lens)
        snaps = [{k: (dict(v) if isinstance(v, dict) else v) for k, v in vars(layer).items()}
                 for layer in layers]
        for layer in layers:
            if self._linear_state(layer) is not None:
                for name in ("conv_states", "recurrent_states"):
                    setattr(layer, name, {i: (None if t is None else t.repeat(R, *[1] * (t.dim() - 1)))
                                          for i, t in getattr(layer, name).items()})
            elif getattr(layer, "keys", None) is not None and layer.keys.numel():
                layer.keys = layer.keys.expand(R, *layer.keys.shape[1:])
                layer.values = layer.values.expand(R, *layer.values.shape[1:])
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        x = torch.tensor([b + [pad] * (L - len(b)) for b in self.branches], device=self.dev)
        keep = L - min(lens) + 1
        kw = {"logits_to_keep": keep} if self._keep_kw else {}
        try:
            out = self.model(input_ids=x, use_cache=True, **{self._cache_kw: self.cache}, **kw)
        finally:
            for layer, snap in zip(layers, snaps):
                vars(layer).clear()
                vars(layer).update(snap)
        logits = out.logits[:, -keep:] if self._keep_kw else out.logits
        off = logits.shape[1] - L
        rows = torch.stack([logits[r, off + n - 1] for r, n in enumerate(lens)]).float().cpu()
        assert self.cache.get_seq_length() == self.n_cached, "cache not restored"
        return list(rows)

    @staticmethod
    def _linear_state(layer) -> list[torch.Tensor] | None:
        """A linear-attention layer's live conv and recurrent state tensors; None for key/value layers."""
        if not hasattr(layer, "recurrent_states"):
            return None
        if getattr(layer, "keys", None) is not None:
            raise NotImplementedError("layers mixing linear attention and key/value caching are not handled")
        return [t for d in (layer.conv_states, layer.recurrent_states) for t in d.values() if t is not None]

    def _read(self) -> torch.Tensor:
        """Probability of each state now, averaged over option orderings in log space."""
        acc = torch.zeros(self.K)
        for shift, logits in zip(self.shifts, self._branch_all()):
            lp = torch.log_softmax(logits[self.letters], dim=-1)   # per slot
            for slot in range(self.K):
                acc[(slot + shift) % self.K] += lp[slot]
        logp = acc / len(self.shifts)
        if self.prior_power > 0:
            logp = logp - self.prior_power * torch.log(self._prior.clamp_min(1e-6))
        p = torch.softmax(logp, dim=-1)
        if self.prior_power > 0:
            raw = torch.softmax(acc / len(self.shifts), dim=-1)
            self._n_prior += 1
            self._prior += (raw - self._prior) / self._n_prior
        return p

    def _commit(self, raw: int) -> tuple[int, bool]:
        """A new state is committed only after it has been likeliest `persist` readings in a row."""
        if self._committed is None:
            self._committed = raw
            return raw, False
        if raw == self._committed:
            self._cand, self._cand_run = None, 0
            return raw, False
        if raw == self._cand:
            self._cand_run += 1
        else:
            self._cand, self._cand_run = raw, 1
        if self._cand_run >= self.persist:
            self._committed, self._cand, self._cand_run = raw, None, 0
            return raw, True
        return self._committed, False

    def _reanchor(self):
        """Context is full: rebuild the cache from the instruction plus the most recent tokens."""
        self.cache, self.n_cached = None, 0
        self._feed(self.head_ids + self.stream[-self.keep_ctx:])

    def flush(self):
        """Feed tokens received since the last reading into the cache (one pass for all of them)."""
        if not self._pending:
            return
        longest = max(len(b) for b in self.branches)
        if self.n_cached + len(self._pending) + longest > self.max_ctx:
            self._reanchor()
        else:
            self._feed(self._pending)
        self._pending = []

    def push_tokens(self, ids: list[int]) -> list[Reading]:
        """Tokens are fed in one pass per reading: with stride N, the N tokens since the last reading
        go into the cache together just before the next reading. Tokens after the last reading wait
        in a buffer until the next reading or flush()."""
        new = []
        for t in ids:
            self.stream.append(t)
            self._pending.append(t)
            if len(self.stream) % self.stride == 0:
                new += self._reading()
        return new

    def finish(self) -> list[Reading]:
        """End of stream: take one more reading if tokens arrived after the last one, so that every
        token is labelled."""
        return self._reading() if len(self.stream) > self._last_read else []

    def _reading(self) -> list[Reading]:
        """One reading at the current position; with `refine`, plus the bisection probes placed between
        the previous reading and this one when the two disagree on the likeliest state."""
        prev = self.readings[-1] if self.readings else None
        pending = list(self._pending)
        snap = self._take() if self.refine and prev is not None and len(pending) > 1 else None
        base_n = self.n_cached
        self.flush()
        p = self._read()
        detail = getattr(self, "last_shift_logp", None)
        probes = []
        if snap is not None and self.n_cached == base_n + len(pending) and int(p.argmax()) != prev.raw:
            probes = self._bisect(snap, base_n, pending, int(p.argmax()))
        out = []
        for offset, pp, dd in probes + [(len(pending), p, detail)]:
            raw = int(pp.argmax())
            state, sw = self._commit(raw)
            pos = self._last_read + offset
            r = Reading(pos, pp.tolist(), raw, state, sw, start=(out[-1].pos if out else self._last_read) + 1, detail=dd,
                        decided=len(self.stream))
            out.append(r)
        self._last_read = len(self.stream)
        self.readings += out
        return out

    def _bisect(self, snap, base_n, pending, new_raw):
        """Readings inside (previous reading, current position) by bisection: each probe rebuilds the
        cache from the snapshot taken at the previous reading plus the tokens up to the probe, so the
        main cache is untouched. Returns (offset, probabilities, detail) sorted by offset; the first
        offset whose reading shows `new_raw` is the placed switch."""
        if self.prior_power > 0:
            raise NotImplementedError("refine does not support prior_power > 0")
        main, main_n = self._take(), self.n_cached
        lo, hi, probes = 0, len(pending), []
        while hi - lo > 1:
            mid = (lo + hi) // 2
            self._put(snap)
            self.n_cached = base_n
            self._feed(pending[:mid])
            pm = self._read()
            probes.append((mid, pm, getattr(self, "last_shift_logp", None)))
            if int(pm.argmax()) == new_raw:
                hi = mid
            else:
                lo = mid
        self._put(main)
        self.n_cached = main_n
        return sorted(probes, key=lambda x: x[0])

    def _take(self):
        """The cache's per-layer attributes; linear-attention states cloned (the model updates them in
        place), key/value tensors referenced (the model replaces them rather than writing into them)."""
        snaps = []
        for layer in self.cache.layers:
            d = {k: (dict(v) if isinstance(v, dict) else v) for k, v in vars(layer).items()}
            if hasattr(layer, "recurrent_states"):
                for name in ("conv_states", "recurrent_states"):
                    d[name] = {i: (None if t is None else t.clone()) for i, t in getattr(layer, name).items()}
            snaps.append(d)
        return snaps

    def _put(self, snaps):
        """Install a snapshot from _take; its linear-attention states are cloned again so it can be reused."""
        for layer, d in zip(self.cache.layers, snaps):
            vars(layer).clear()
            vars(layer).update({k: (dict(v) if isinstance(v, dict) else v) for k, v in d.items()})
            if hasattr(layer, "recurrent_states"):
                for name in ("conv_states", "recurrent_states"):
                    setattr(layer, name, {i: (None if t is None else t.clone()) for i, t in d[name].items()})

    @staticmethod
    def token_states(readings: list[Reading]) -> list[int]:
        """Committed state of every token, expanded from readings that each label a span of tokens."""
        out = []
        for r in readings:
            out += [r.state] * (r.pos - (r.start or r.pos) + 1)
        return out

    def push_text(self, text: str) -> list[Reading]:
        return self.push_tokens(self._enc(text))


# ------------------------------------------------------------------ evaluation
def build_stream(tok, segments: list[tuple[str, str]], names: list[str]):
    """Concatenate (state_name, text) segments; return token ids and a gold state per token."""
    ids, gold = [], []
    for i, (name, text) in enumerate(segments):
        piece = tok((" " if i else "") + text.strip(), add_special_tokens=False)["input_ids"]
        ids += piece
        gold += [names.index(name)] * len(piece)
    return ids, gold


def score(readings: list[Reading], gold: list[int]) -> dict:
    """Token accuracy, delay after each true switch, and switches that should not have happened.

    True switches come from the token-level gold up to the last reading, so a segment that no reading
    lands in still counts (as missed). A true switch is found by a commit event into its new state from
    a reading whose position falls inside the new segment; matching on the committed state alone would
    credit a decoder that never left the state the stream returns to (gold A-B-A, decoder stays on A).
    Delay runs from the switch token to the position at which the committing reading was available
    (`decided`), so waiting for the next reading and decisions made after a bisection both count.
    """
    if not readings:
        return {}
    last = readings[-1].pos
    gtok = gold[:last]
    # Token accuracy counts every token a reading labels (all tokens since the previous reading).
    spans = [range((r.start or r.pos) - 1, r.pos) for r in readings]
    n_tok = sum(len(sp) for sp in spans)
    acc_raw = sum(gold[t] == r.raw for r, sp in zip(readings, spans) for t in sp) / n_tok
    acc_com = sum(gold[t] == r.state for r, sp in zip(readings, spans) for t in sp) / n_tok
    starts = [t for t in range(1, len(gtok)) if gtok[t] != gtok[t - 1]]      # 0-based first token of each segment
    commits = [r for r in readings if r.switched]
    delays = []
    for j, t in enumerate(starts):
        end = starts[j + 1] if j + 1 < len(starts) else len(gtok)
        hit = next((r for r in commits if t + 1 <= r.pos <= end and r.state == gtok[t]), None)
        delays.append(math.nan if hit is None else (hit.decided or hit.pos) - (t + 1))
    found = [d for d in delays if not math.isnan(d)]
    return {
        "readings": len(readings),
        "accuracy_raw": round(acc_raw, 4),
        "accuracy_committed": round(acc_com, 4),
        "true_switches": len(starts),
        "switches_found": len(found),
        "mean_delay_tokens": round(sum(found) / len(found), 1) if found else None,
        "spurious_switches": len(commits) - len(found),
    }


# ------------------------------------------------------------------ fixed-lag Viterbi
def sticky_transition(K: int, seg_len: float, gap: int = 1) -> np.ndarray:
    """Transition between two readings `gap` tokens apart, for a chain that stays with probability
    1 - 1/L per token (L = declared segment length in tokens) and spreads the rest evenly: the per-token
    matrix raised to the power `gap`. Raising the per-token chain, rather than putting 1/(L/gap) into
    one step, keeps it sticky when segments are shorter than a few readings."""
    stay = 1.0 - 1.0 / max(seg_len, 1.0 + 1e-9)
    A = np.full((K, K), (1.0 - stay) / (K - 1))
    np.fill_diagonal(A, stay)
    return np.linalg.matrix_power(A, max(int(gap), 1))


def sticky_log_transition(K: int, seg_len: float, gap: int = 1) -> torch.Tensor:
    """log of sticky_transition, as used by FixedLagViterbi."""
    return torch.tensor(np.log(sticky_transition(K, seg_len, gap)), dtype=torch.float64)


class FixedLagViterbi:
    """Online max-product decoding with a fixed lag.

    push(e_t) takes one row of per-state log-scores (delta_t(k) = e_t(k) + max_j delta_{t-1}(j) + log A_jk)
    and returns the label of position t - lag, read off the best path ending at t; flush() returns the
    labels still owed at the end of the stream. lag=None decodes the whole stream offline at flush().
    The labels are emitted once and never revised, so a later, longer traceback may disagree with them.
    """

    def __init__(self, log_A: torch.Tensor, lag: int | None):
        self.log_A, self.lag = log_A, lag
        self.delta, self.back, self.emitted = None, [], 0

    def push(self, e) -> list[int]:
        e = torch.as_tensor(e, dtype=torch.float64)
        if self.delta is None:
            self.delta = e - torch.log(torch.tensor(float(len(e)), dtype=torch.float64))
            self.back.append(None)
        else:
            scores = self.delta[:, None] + self.log_A           # [from j, to k]
            best, arg = scores.max(0)
            self.delta = best + e
            self.back.append(arg)
        t = len(self.back) - 1
        if self.lag is None or t < self.lag:
            return []
        k = int(self.delta.argmax())
        for s in range(t, t - self.lag, -1):
            k = int(self.back[s][k])
        self.emitted += 1
        return [k]

    def flush(self) -> list[int]:
        if self.delta is None:
            return []
        path = [int(self.delta.argmax())]
        for s in range(len(self.back) - 1, 0, -1):
            path.append(int(self.back[s][path[-1]]))
        path.reverse()
        out = path[self.emitted:]
        self.emitted = len(path)
        return out


DEFAULT_MODEL = "h2oai/h2o-lightning-4b"

DEMO_STATES = {
    "greeting": "opening pleasantries and identifying the caller",
    "problem": "the customer describes what is wrong",
    "troubleshooting": "the agent gives steps to try and the customer tries them",
    "billing": "discussion of charges, refunds or payment",
    "closing": "wrapping up and saying goodbye",
}
DEMO_SEGMENTS = [
    ("greeting", "Hello, thanks for calling Northwind support, my name is Dana. Can I get your name "
                 "and account number please? Sure, it's Priya Raman, account four four one nine. "
                 "Thanks Priya, nice to meet you."),
    ("problem", "So what's going on today? My router keeps dropping the connection every evening. "
                "It started about a week ago. The light goes orange, everything stalls for a few "
                "minutes, and then it comes back. It is making video calls impossible."),
    ("troubleshooting", "Okay, let's try a few things. Please unplug the router, wait thirty "
                        "seconds, and plug it back in. Done. Now open the admin page and tell me "
                        "the firmware version. It says two point one. Right, press update and "
                        "wait for the restart. It's restarting now. The light is green again."),
    ("billing", "While I have you, I noticed I was charged twice last month. Let me look. Yes, I "
                "see two payments of thirty nine dollars on the fourth. I'll refund one of them to "
                "your card, it should arrive within five business days."),
    ("closing", "Is there anything else I can help with? No, that's everything. Great, thanks for "
                "calling Northwind, have a good evening. You too, goodbye."),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="Hugging Face id or local path (default: H2O-Lightning-4B, a decision-tuned\n"
                         "Qwen3.5-4B); any causal LM of 4B parameters or more, e.g. Qwen/Qwen3.5-4B")
    ap.add_argument("--prompt", choices=["auto", "h2o", "plain"], default="auto",
                    help="question format: h2o = H2O-Lightning's record/question/options format, plain = the\n"
                         "original instruction format; auto picks h2o for H2O-Lightning models")
    ap.add_argument("--stride", type=int, default=1,
                    help="read the regime every Nth token (default 1: every token); the tokens since the\n"
                         "previous reading are fed in one pass and take the next reading's label, so the\n"
                         "cost per token falls about N-fold. 4 to 8 kept accuracy in tests (docs/LOG.md)")
    ap.add_argument("--rotations", type=int, default=2,
                    help="option orderings averaged per reading (default 2); 1 matched 2 in tests")
    ap.add_argument("--refine", action="store_true",
                    help="with --stride N: where two readings disagree, bisect between them (about log2 N extra\n"
                         "readings per change) so a switch is placed at a single token")
    ap.add_argument("--persist", type=int, default=3,
                    help="readings a new regime must lead before it is committed (default 3). Counts\n"
                         "readings, not tokens: with --stride 8 use --persist 1")
    ap.add_argument("--prior-power", type=float, default=0.0)
    ap.add_argument("--demo", action="store_true", help="run the built-in support-call stream")
    ap.add_argument("--text", help="path to a text file to label instead of the demo")
    ap.add_argument("--states", help="path to a JSON file {name: description}")
    ap.add_argument("--segment-length", type=float, default=30.0,
                    help="declared typical segment length in tokens (default 30); sets the transition model\n"
                         "of the settled labels and how far their readings are moved back (0.2x)")
    ap.add_argument("--save-readings", help="write every reading (state log-probabilities, orderings "
                    "averaged), its position and the gold states to this JSON file, for offline decoding")
    a = ap.parse_args()

    import os

    from transformers import AutoModelForCausalLM, AutoTokenizer
    h2o = a.prompt == "h2o" or (a.prompt == "auto" and "h2o-lightning" in a.model.lower())
    if h2o and os.path.isdir(os.path.join(a.model, "tokenizer")):
        tok = AutoTokenizer.from_pretrained(os.path.join(a.model, "tokenizer"))   # the model's own tokenizer
    elif h2o and not os.path.isdir(a.model):
        tok = AutoTokenizer.from_pretrained(a.model, subfolder="tokenizer")
    else:
        tok = AutoTokenizer.from_pretrained(a.model)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype, device_map="auto")
    except TypeError:  # older transformers
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dtype, device_map="auto")

    states = DEMO_STATES
    if a.states:
        import json
        states = json.load(open(a.states))
    if h2o:
        from .h2o import H2OReader
        reader = H2OReader(model, tok, states, stride=a.stride, rotations=a.rotations, refine=a.refine,
                           persist=a.persist, prior_power=a.prior_power)
    else:
        reader = StreamStateReader(model, tok, states, stride=a.stride, rotations=a.rotations, refine=a.refine,
                                   persist=a.persist, prior_power=a.prior_power)
    names = list(states)
    if a.text:
        ids, gold = tok(open(a.text).read(), add_special_tokens=False)["input_ids"], None
    else:
        ids, gold = build_stream(tok, DEMO_SEGMENTS, names)

    for r in reader.push_tokens(ids) + reader.finish():
        mark = "  <-- switch" if r.switched else ""
        word = tok.decode([ids[r.pos - 1]]).replace("\n", "\\n")
        span = f"{r.start}-{r.pos}" if r.start != r.pos else f"{r.pos}"
        print(f"{span:>9} {word!r:>16}  {names[r.state]:<16} p={max(r.probs):.2f}{mark}")
    if gold:
        print("live:", score(reader.readings, gold))
    # Settled labels: the best path over all readings, each reading moved back 0.2x the segment length
    from .settle import settled_labels
    rp = np.asarray([r.probs for r in reader.readings])
    rpos = [r.pos for r in reader.readings]
    lab = settled_labels(rp, rpos, a.segment_length, int(round(0.2 * a.segment_length)))
    print("settled segments:")
    start = 1
    for i in range(len(lab)):
        if i == len(lab) - 1 or lab[i + 1] != lab[i]:
            print(f"  tokens {start:>5d}-{rpos[i]:<5d} {names[lab[i]]}")
            start = rpos[i] + 1
    if gold:
        settled = [Reading(x, [], int(l), int(l), i > 0 and l != lab[i - 1]) for i, (x, l) in enumerate(zip(rpos, lab))]
        print("settled token accuracy:", round(score(settled, gold)["accuracy_committed"], 4))
    if a.save_readings:
        import json
        rec = {"id": a.text or "demo", "kind": "posterior", "states": names, "n_tokens": len(ids),
               "stride": a.stride, "rotations": reader.rotations, "prior_power": a.prior_power,
               "positions": [r.pos for r in reader.readings],
               "logp": [[math.log(max(p, 1e-30)) for p in r.probs] for r in reader.readings],
               "gold": gold}
        with open(a.save_readings, "w") as f:
            json.dump(rec, f)


if __name__ == "__main__":
    main()
