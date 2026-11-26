"""TRIPOD movie synopses (github.com/ppapalampidi/TRIPOD; no licence stated, data not committed): the stage of
the story at every token.

Data: data/tripod/TRIPOD_synopses_{train,test}.csv (experiments/data/download.sh). Each synopsis is split into
sentences ([STR_SENT] ... [END_SENT]) and tp1..tp5 give the 0-based sentence of each turning point
(Papalampidi, Keller & Lapata 2019, Table 1). The six stages between them are the states. A turning-point
sentence opens the stage after it (the paper defines each turning point as the event that starts what follows,
and places turning points "at the junctions of acts"; it does not say which side they belong to). Movies
annotated twice keep their first annotation, and train and test are separate movies.
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402

DATA = common.DATA / "tripod"


def load(split: str) -> list[dict]:
    csv.field_size_limit(10**9)
    out, seen = [], set()
    for r in csv.DictReader((DATA / f"TRIPOD_synopses_{split}.csv").open(encoding="utf-8")):
        m = re.match(r"^(.*?)(?:_(\d+))?$", r["movie_name"].strip())
        movie, k = m.group(1), int(m.group(2) or 0)
        if k != 0 or movie in seen:
            continue
        seen.add(movie)
        sents = [s.strip() for s in re.findall(r"\[STR_SENT\](.*?)\[END_SENT\]", r["synopsis_segmented"], re.S)]
        tps = [int(float(r[f"tp{i}"])) for i in range(1, 6)]
        text, spans = "", []
        for i, s in enumerate(sents):
            stage = sum(tp <= i for tp in tps)
            start = len(text) + (1 if text else 0)
            text = f"{text} {s}" if text else s
            spans.append((start, len(text), stage))
        out.append({"id": f"tripod-{movie}", "text": text, "spans": spans})
    return out
