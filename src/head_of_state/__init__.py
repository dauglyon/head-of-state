"""Head of State.

Zero-shot regime tracking from an open-weights LLM's output head.
Hidden-state inference from plain-English descriptions.
"""
from .reader import FixedLagViterbi, Reading, StreamStateReader, score, sticky_log_transition

__all__ = ["FixedLagViterbi", "Reading", "StreamStateReader", "score", "sticky_log_transition"]
