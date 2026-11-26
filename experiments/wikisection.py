"""WikiSection en_disease with the merged labels of docs/TEST_SUITE.md (S1, S2, S4).

    uv run python -I experiments/wikisection.py measure     # D1 declared length from the tuning articles

Data: data/wikisection/wikisection_en_disease_{train,validation,test}.json
(github.com/sebastianarnold/WikiSection, CC BY-SA 3.0). The validation split is the tuning set
(D1, η sweep); reported S1/S2/S4 streams are built from the other splits.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402

DATA = common.DATA / "wikisection"
MODEL = "Qwen/Qwen3.5-4B"
TUNING_SPLIT = "validation"

GROUPS = {   # merged state -> WikiSection labels (disease.<label>); fauna and other are dropped
    "symptom": ["symptom"],
    "cause": ["cause", "risk", "genetics", "infection"],
    "diagnosis": ["diagnosis", "tomography", "classification"],
    "prevention": ["prevention", "screening"],
    "treatment": ["treatment", "management", "medication", "surgery"],
    "mechanism": ["mechanism", "pathology", "pathophysiology"],
    "epidemiology": ["epidemiology", "geography"],
    "history": ["history", "etymology"],
    "prognosis": ["prognosis", "complication"],
    "society and research": ["culture", "research"],
}
DROP = {"fauna", "other"}
# Descriptions written for this evaluation (not taken from the dataset).
STATES = {
    "symptom": "the signs and symptoms a patient experiences",
    "cause": "what causes the disease, including risk factors, genetics and how infection spreads",
    "diagnosis": "how the disease is diagnosed or classified, including tests and imaging",
    "prevention": "how the disease can be prevented or screened for",
    "treatment": "how the disease is treated or managed: medication, surgery and other therapy",
    "mechanism": "the biological mechanism and pathology of the disease",
    "epidemiology": "how common the disease is, and where and in whom it occurs",
    "history": "the history of the disease and the origin of its name",
    "prognosis": "the likely outcome of the disease and its complications",
    "society and research": "the disease in society and culture, and current research on it",
}
UNIT = "the last few sentences"   # pre-registration table, S1/S2/S4
LABEL_TO_GROUP = {lab: g for g, labs in GROUPS.items() for lab in labs}
NAMES = list(GROUPS)
assert list(STATES) == NAMES


def load(split: str) -> list[dict]:
    """Documents as {id, title, text, spans [(start, end, group_index)]}, adjacent same-group
    sections fused (gold fusion rule). Documents with a dropped label are skipped."""
    out = []
    for doc in json.load((DATA / f"wikisection_en_disease_{split}.json").open(encoding="utf-8")):
        labels = [a["sectionLabel"].split(".", 1)[1] for a in doc["annotations"]]
        if DROP & set(labels):
            continue
        spans = []
        for a, lab in zip(doc["annotations"], labels):
            g = NAMES.index(LABEL_TO_GROUP[lab])
            st, en = a["begin"], a["begin"] + a["length"]
            if spans and spans[-1][2] == g:
                spans[-1] = (spans[-1][0], en, g)
            else:
                spans.append((st, en, g))
        out.append({"id": doc["id"], "title": doc["title"], "text": doc["text"], "spans": spans})
    return out


def main():
    if sys.argv[1:] != ["measure"]:
        sys.exit(__doc__)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    docs = load(TUNING_SPLIT)
    lens = [len(tok(d["text"][st:en], add_special_tokens=False)["input_ids"])
            for d in docs for st, en, _ in d["spans"]]
    print(json.dumps({"set": "S1/S2/S4", "declared_segment_length_tokens": statistics.median(lens),
                      "source": f"median merged-section length, WikiSection en_disease {TUNING_SPLIT}, "
                                "Qwen3.5-4B tokenizer", "n_documents": len(docs), "n_segments": len(lens)}))


if __name__ == "__main__":
    main()
