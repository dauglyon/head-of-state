"""Paths and the model shared by the experiment scripts."""
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"                       # downloaded datasets (not in git; see experiments/data/download.sh)
RUNS = REPO / "experiments" / "runs"       # raw per-run outputs (not in git)
RESULTS = REPO / "results"                 # small summary tables (in git)
MODEL = "Qwen/Qwen3.5-4B"
