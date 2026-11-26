"""Readout through H2O-Lightning-4B, an open-weights Jev-class decision model built on Qwen3.5-4B.

The model was fine-tuned to answer choice questions about a record in one fixed prompt format; this
reader builds that format around the cached stream (one-way reading, the stock causal pattern):

    system:    the decision-engine instruction of the model's serve_config.json
    user:      "record: " + the stream so far + "\\nquestion: ..." + "\\noptions:\\nA) name: description ..."
    assistant: thinking off, prefilled "Answer:"; the answer is one option letter

The stream sits in the cache after "record: "; each reading appends the question and options as the
temporary branch, exactly as StreamStateReader does with its own wording. Letter probabilities use the
model's choice temperature (0.75 in its serve_config.json).
"""
from __future__ import annotations

import json
import string
from dataclasses import dataclass

import torch

from .harness import DEFAULT_UNIT, UnitReader

H2O_SYSTEM = ("You are a decision engine. You read a record and answer one question about it by choosing "
              "exactly one option. Reply with a single letter.")
H2O_QUESTION = ("Which state is the most recent part of the record in? Judge only {unit}, "
                "not the record as a whole.")
H2O_CHOICE_TEMPERATURE = 0.75
_SPLIT = "␞STREAM␞"


@dataclass
class H2OReader(UnitReader):
    temperature: float = H2O_CHOICE_TEMPERATURE

    def _template(self) -> tuple[str, str]:
        msgs = [{"role": "system", "content": H2O_SYSTEM}, {"role": "user", "content": "record: " + _SPLIT}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        head, rest = text.split(_SPLIT)
        return head, rest + "Answer:"

    def _question(self, shift: int) -> str:
        lines = []
        for slot in range(self.K):
            k = (slot + shift) % self.K
            lines.append(f"{string.ascii_uppercase[slot]}) {self.names[k]}: {self.states[self.names[k]]}")
        return "\nquestion: " + H2O_QUESTION.format(unit=self.unit) + "\noptions:\n" + "\n".join(lines)





@dataclass
class TargetedH2OReader(H2OReader):
    """H2O format with the text to judge as its own part of the record (Jev documentation: state the exact
    condition and mark the target). The record holds the stream so far and, at each reading, the current
    sentence or line (the tokens since the last sentence or line end), followed by a question about it.

    Options: `option_names` replaces the state names shown in the options (e.g. neutral names); a state's
    description may be a string or an object (rendered as JSON, as the serving shim does).
    `record_format="json"` sends the record as {"text_so_far": ..., "current_<span>": ...} with every stream
    token JSON-escaped as it is fed. `decompose=True` asks one yes/no question per state ("the current
    sentence matches this description") and normalises the K yes-probabilities, instead of one choice.
    A recent-text-only record is obtained with `keep_ctx` (tokens kept) and a small `max_ctx`.
    """
    span_kind: str = "sentence"            # "sentence", "line", or "passage" (the last `span_sentences` sentences)
    span_sentences: int = 3
    question_text: str = "Which option describes the current sentence?"
    option_names: list | None = None
    max_span: int = 64
    record_format: str = "text"            # "text" or "json"
    decompose: bool = False
    yes_no_temperature: float = 0.8        # the model's yes/no temperature in its serve_config.json

    def _template(self) -> tuple[str, str]:
        opening = 'record: {"text_so_far":"' if self.record_format == "json" else "record: text so far:\n"
        msgs = [{"role": "system", "content": H2O_SYSTEM}, {"role": "user", "content": opening + _SPLIT}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        head, rest = text.split(_SPLIT)
        self._tail_text = rest + "Answer:"
        return head, self._tail_text

    @staticmethod
    def _desc(v) -> str:
        return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)

    def _options(self, shift: int) -> str:
        shown = self.option_names or self.names
        lines = []
        for slot in range(self.K):
            k = (slot + shift) % self.K
            lines.append(f"{string.ascii_uppercase[slot]}) {shown[k]}: {self._desc(self.states[self.names[k]])}")
        return "\nquestion: " + self.question_text + "\noptions:\n" + "\n".join(lines)

    def _span_open(self) -> str:
        if self.record_format == "json":
            return f'","current_{self.span_kind}":"'
        return f"\n\ncurrent {self.span_kind}:\n"

    def _span_close(self) -> str:
        return '"}' if self.record_format == "json" else ""

    def _question(self, shift: int) -> str:       # used once at start-up for sizing; readings rebuild it
        return self._span_open() + self._span_close() + self._options(shift)

    # -- stream tokens as they appear in the record
    def _escape(self, ids: list[int]) -> list[int]:
        if self.record_format != "json":
            return list(ids)
        cache = self.__dict__.setdefault("_escape_cache", {})
        out = []
        for t in ids:
            if t not in cache:
                s = self.tok.decode([t])
                cache[t] = [t] if not any(c in s for c in '"\\\n\r\t') else self._enc(json.dumps(s)[1:-1])
            out += cache[t]
        return out

    def flush(self):
        if not self._pending:
            return
        ids = self._escape(self._pending)
        longest = max(len(b) for b in self.branches)
        if self.n_cached + len(ids) + longest > self.max_ctx:
            self._reanchor()
        else:
            self._feed(ids)
        self._pending = []

    def _reanchor(self):
        self.cache, self.n_cached = None, 0
        self._feed(self.head_ids + self._escape(self.stream[-self.keep_ctx:]))

    def _is_boundary(self, token_id: int) -> bool:
        cache = self.__dict__.setdefault("_boundary_cache", {})
        if token_id not in cache:
            s = self.tok.decode([token_id])
            cache[token_id] = "\n" in s or (self.span_kind in ("sentence", "passage") and s.rstrip().endswith((".", "?", "!")))
        return cache[token_id]

    def _span(self) -> list[int]:
        """Tokens of the current sentence or line: back to the last boundary token. If the newest token is
        itself a boundary, the span is the sentence or line it just closed."""
        st = self.stream
        end = len(st)
        i = end - 1
        if i >= 0 and self._is_boundary(st[i]):
            i -= 1
        need = self.span_sentences if self.span_kind == "passage" else 1
        limit = self.max_span * need
        while i >= 0 and end - i <= limit:
            if self._is_boundary(st[i]):
                need -= 1
                if need == 0:
                    break
            i -= 1
        return st[i + 1: end]

    def _read(self):
        if self.refine:
            raise NotImplementedError("refine bisection probes see the full stream; not supported with targeted spans")
        span = self._enc(self._span_open()) + self._escape(self._span()) + self._enc(self._span_close())
        if not self.decompose:
            self.branches = [span + self._enc(self._options(s) + self._tail_text) for s in self.shifts]
            return super()._read()
        # one yes/no question per state, read in one batched pass; P(yes) normalised over states
        self.branches = []
        for k in range(self.K):
            q = (f"\nquestion: The current {self.span_kind} matches this description: "
                 f"{self._desc(self.states[self.names[k]])}\noptions:\nA) true: it matches\nB) false: it does not")
            self.branches.append(span + self._enc(q + self._tail_text))
        logits = self._branch_all()
        yes = torch.stack([torch.log_softmax(lg[self.letters[:2]] / self.yes_no_temperature, -1)[0] for lg in logits])
        logp = torch.log_softmax(yes, -1)
        self.last_shift_logp = [logp.tolist()]
        return logp.exp()


__all__ = ["H2OReader", "TargetedH2OReader", "DEFAULT_UNIT"]
