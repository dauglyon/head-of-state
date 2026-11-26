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

import string
from dataclasses import dataclass

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


__all__ = ["H2OReader", "DEFAULT_UNIT"]
