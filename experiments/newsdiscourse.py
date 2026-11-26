"""NewsDiscourse (Choubey et al. 2020, ACL; aclanthology.org/attachments/2020.acl-main.478.Dataset.zip; no
licence stated, data not committed): the discourse role of each news sentence relative to the main event.

Data: data/newsdiscourse/raw/{train,validation,test}/*.txt and .ann (experiments/data/download.sh). Each article
line "S<n> <sentence>" is a sentence; the brat .ann file gives each sentence one role (the separate "Speech"
tag marks quoted speech and is not a role). The URL, date and headline lines are left out; sentences are joined
with single spaces. A sentence without a role is NA.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402

DATA = common.DATA / "newsdiscourse" / "raw"
ROLES = ["Main", "Main_Consequence", "Cause_Specific", "Cause_General", "Distant_Historical", "Distant_Anecdotal",
         "Distant_Evaluation", "Distant_Expectations_Consequences", "NA"]


def load(split: str) -> list[dict]:
    out = []
    for f in sorted((DATA / split).rglob("*.txt")):     # train and test are in topic subfolders
        role = {}
        for line in f.with_suffix(".ann").open(encoding="utf-8"):
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 3 and parts[0].startswith("T"):
                lab, sid = parts[1].split()[0], parts[2].strip()
                if lab in ROLES:
                    role[sid] = lab
        text, spans = "", []
        for line in f.open(encoding="utf-8"):
            m = re.match(r"^(S\d+) (.*\S)", line)
            if not m:
                continue
            g = ROLES.index(role.get(m.group(1), "NA"))
            start = len(text) + (1 if text else 0)
            text = f"{text} {m.group(2)}" if text else m.group(2)
            spans.append((start, len(text), g))
        if spans:
            out.append({"id": f"nd-{f.parent.name}-{f.stem}", "text": text, "spans": spans})
    return out
