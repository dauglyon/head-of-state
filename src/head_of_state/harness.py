"""Evaluation harness for the reader (rules H1, H4, H6, H8 of docs/TEST_SUITE.md).

A run stores, for every reading, the letter log-probabilities of every option ordering
(mapped back to state order). Every knob except the ordering set itself (stride, rotations as a
subset of the orderings run, persist, prior-power, decoder) is then applied offline in decode.py,
so one run with all K orderings covers the whole sweep.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from .reader import StreamStateReader

DEFAULT_UNIT = "the last few sentences"   # the wording in reader.QUESTION_HEAD


@dataclass
class UnitReader(StreamStateReader):
    """H4: the unit to judge is supplied per stream; the rest of the question frame is fixed."""
    unit: str = DEFAULT_UNIT
    temperature: float = 1.0   # letter logits are divided by this before the softmax (a model's own calibration)

    def _question(self, shift: int) -> str:
        q = super()._question(shift)
        old = f"Judge only {DEFAULT_UNIT},"
        assert old in q, "QUESTION_HEAD changed; update DEFAULT_UNIT"
        return q.replace(old, f"Judge only {self.unit},", 1)

    def _read(self) -> torch.Tensor:
        """Parent's reading, plus the per-ordering log-probabilities kept in self.last_shift_logp."""
        per_shift = []
        for shift, logits in zip(self.shifts, self._branch_all()):
            lp = torch.log_softmax(logits[self.letters] / self.temperature, dim=-1)
            per_shift.append([float(lp[(k - shift) % self.K]) for k in range(self.K)])  # state order
        self.last_shift_logp = per_shift
        acc = torch.tensor(per_shift).sum(0)
        logp = acc / len(self.shifts)
        if self.prior_power > 0:
            logp = logp - self.prior_power * torch.log(self._prior.clamp_min(1e-6))
        p = torch.softmax(logp, dim=-1)
        if self.prior_power > 0:
            raw = torch.softmax(acc / len(self.shifts), dim=-1)
            self._n_prior += 1
            self._prior += (raw - self._prior) / self._n_prior
        return p


def tokens_and_gold(tok, text: str, spans: list[tuple[int, int, int]]):
    """H1: tokenise the whole stream once; a token's gold state is that of the first span ending
    after the token's first character (a token that starts on the separator between two spans
    belongs to the following span)."""
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, gold, j = enc["input_ids"], [], 0
    for start, _ in enc["offset_mapping"]:
        while j < len(spans) - 1 and spans[j][1] <= start:
            j += 1
        gold.append(spans[j][2])
    return ids, gold


def run_stream(reader: UnitReader, stream: dict, tok) -> dict:
    """stream: {id, text, spans [(start, end, state_index)]}. Returns a JSON-ready record."""
    ids, gold = tokens_and_gold(tok, stream["text"], stream["spans"])
    reader.reset()
    # Content-free reading (contextual calibration, Zhao et al. 2021): the question asked with the
    # stream still empty; its per-ordering letter log-probabilities are the model's bias.
    reader._read()
    content_free = reader.last_shift_logp
    reader.reset()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    rows, positions = [], []
    for t in ids:
        for r in reader.push_tokens([t]):
            rows.append(r.detail)
            positions.append(r.pos)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return {"id": stream["id"], "n_tokens": len(ids), "gold": gold, "shifts": reader.shifts,
            "content_free": content_free, "positions": positions,
            "shift_logp": rows, "seconds": time.time() - t0, "stride": reader.stride}


def run_all(reader: UnitReader, streams: list[dict], tok, out: Path, log=lambda s: print(s, flush=True)):
    """Append one JSON line per stream; skips streams already in `out` (resumable)."""
    done = set()
    if out.exists():
        done = {json.loads(line)["id"] for line in out.open()}
    with out.open("a") as f:
        for i, s in enumerate(streams):
            if s["id"] in done:
                continue
            rec = run_stream(reader, s, tok)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            log(f"[{i + 1}/{len(streams)}] {s['id']}: {rec['n_tokens']} tokens, "
                f"{rec['n_tokens'] / rec['seconds']:.2f} tokens/s")   # H8
