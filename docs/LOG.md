# Experiment log

Newest last. Model: Qwen3.5-4B (bf16) on one RTX 3060 (12 GB) unless stated. Raw outputs go to
`experiments/runs/` (not in git); summary tables to `results/`. Entries dated 2026-11-25 before the
repository existed were run from an earlier unversioned copy of the same scripts; their commands are
given in this repository's layout and reproduce from the same code.

---

**2026-11-25 · Mechanics tests on tiny random models**
- Command: `uv run python tests/test_reader.py`
- Result: a reading through the cached branch equals a full recompute (max diff 3e-8); the cache is
  restored after every reading; re-anchoring keeps a stream running past the context limit; the
  persist rule and the scorer behave on hand-made cases.
- Conclusion: the plumbing is correct; nothing here measures accuracy.

**2026-11-25 · Model choice**
- Checked: weight sizes on Hugging Face; layer layouts in each model's config; transformers 5.24.1
  cache code.
- Result: Qwen2.5-7B needs about 15 GB at bf16 and Gemma 4 E4B ships a 16.0 GB file, so neither fits
  12 GB. Qwen3.5-4B (9.32 GB with its vision encoder; 8.4 GB loaded as text-only) fits. 24 of its 32
  layers are Gated DeltaNet layers, which keep a running state that transformers' `crop` cannot undo.
- Conclusion: use Qwen3.5-4B. Save and restore those states around each question branch (tested in
  `tests/test_reader.py`, case 6).

**2026-11-25 · Demo stream**
- Command: `uv run head-of-state --demo` (2 orderings, a reading at every token, persist 3)
- Result: token accuracy 0.869 raw, 0.837 committed. 4 of 4 switches found, at 3, 9, 11 and 17 tokens
  after the true switch (mean 10). No spurious switches. The prompt shows an empty `<think>` block
  before the `Answer:` prefill, and the option letters are single tokens.
- Conclusion: the method runs end to end on a real model. One 245-token stream is a smoke test, not
  an accuracy estimate.

**2026-11-25 · Scorer fix (TEST_SUITE H7)**
- Change: `score()` now counts a true switch as found only through a commit event into the new
  regime. Before the fix, a decoder that stayed in a regime the stream returned to was credited.
- Command: `uv run python tests/test_reader.py` (case 5b)
- Conclusion: the demo numbers are unchanged, because no regime recurs in the demo.

**2026-11-25 · Declared segment lengths (TEST_SUITE D1)**
- Commands: `uv run python -I experiments/pubmed_rct.py measure`;
  `uv run python -I experiments/wikisection.py measure`
- Result: PubMed 20k RCT dev median sentence is 30 tokens (30,212 sentences). WikiSection en_disease
  validation median merged section is 190.5 tokens (260 documents).
- Conclusion: recorded in the pre-registration table of `docs/TEST_SUITE.md` before any scored run.

**2026-11-25 · Cost profile and speed-ups**
- Measured with the S3 prompt:
  - One stream token: 31 ms.
  - One question pass: 66 ms at 113 tokens, 69 ms at 128 tokens, 96 ms at 129–130 tokens.
  - A reading at every token: 233 ms with 2 orderings, 539 ms with 5.
- Changes:
  - Ran all orderings as one batched pass.
  - Shortened the S3 background description so the question is 126 tokens.
  - With stride N, fed the N tokens since the last reading in one pass.
  - Installed flash-linear-attention.
- Result: 150 ms per token with 2 orderings at every token. With 1 ordering:
  - every token: 103 ms
  - every 4th token: 27 ms
  - every 8th token: 14 ms
  - every 16th token: 7 ms

  On the 10-regime WikiSection prompt the corresponding figures are 248, 153, 40, 20 and 10 ms. The
  demo numbers were identical before and after.
- Conclusion: the cost is now about (stream pass + reading) / stride. The remaining lever is the
  length of the question.

**2026-11-25 · PubMed dev question run, all 5 orderings saved**
- Command: `uv run python -I experiments/pubmed_rct.py run --split dev --n 40 --rotations 5 --out experiments/runs/s3_dev_r5.jsonl`
- Result: stopped by hand at 10 of 40 abstracts. The full sweep was too large for an exploration
  stage. These 10 abstracts (36 switches) are the question-method baseline used below.
- Conclusion: offline replay of saved readings replaces most reruns (`src/head_of_state/decoders.py`;
  checked against the online reader in `tests/test_harness.py`).

**2026-11-25 · Option orderings × stride, offline**
- Command: `uv run python -I experiments/stride_sweep.py` → `results/stride_sweep.md`
- Result (6 abstracts, 21 switches):
  - 1, 2 or 5 orderings give the same accuracy (0.81–0.82).
  - With a reading at every token, no decoder keeps spurious switches at or under 2 per 1k tokens.
  - Reading every 2nd to 8th token meets that cap with a decision latency of 26–32 tokens.
- Conclusion: 1 ordering, reading every 4th–8th token, keeps accuracy at about a tenth of the cost.
  This conflicts with the brief's "one reading per token" and needs the owner's decision.

**2026-11-25 · Readout comparison (question after the stream vs four alternatives)**
- Commands: `uv run python -I experiments/compare_readouts.py E1:gen E1:embed E1:qbefore E1:genrefresh`,
  then `uv run python -I experiments/compare_readouts.py E2:gen E2:embed E2:qafter`. Report:
  `uv run python -I experiments/compare_readouts_report.py` → `results/compare_readouts.md`
- Sets: E1 is 6 PubMed dev abstracts (2,830 tokens). E2 is 2 WikiSection validation streams
  (2,645 tokens, 10 regimes).
- Result (best committed token accuracy E1 / E2):

  | Readout | E1 | E2 |
  |---|---|---|
  | question after the stream | 0.835 | 0.578 |
  | question before the stream | 0.505 | — |
  | next-token likelihood, description at the start | 0.293 | 0.218 |
  | same, refreshed every 64 tokens | 0.289 | — |
  | embedding similarity (bge-m3) | 0.554 | 0.258 |

  The per-token log-likelihood advantage of the right description averaged about 0.01 nats on both
  sets.
- Conclusion: the question readout is the only one with a strong signal on these sets.

**2026-11-25 · Content-free calibration**
- Commands: `uv run python -I experiments/calibration_check.py run`, then `... report` →
  `results/calibration_check.md`
- Result: calibration lowered accuracy from 0.811 to 0.768 (E1) and from 0.578 to 0.389 (E2) with 1
  ordering.
- Conclusion: drop calibration.

**2026-11-25 · Windowed likelihood emissions: fade curve (A) and window sweep (B)**
- Commands: `uv run python -I experiments/window_likelihood_sweep.py run`, then `... report` →
  `results/window_likelihood_sweep.md`
- Sets: 10 PubMed dev abstracts; 2 WikiSection validation streams. Decoder: forward filter, η = 1.
- Fade curve (right minus mean wrong log-probability per token, by distance from the description):
  - PubMed: +0.32 at 0–4 tokens, about +0.08 at 4–32, about 0 beyond 32.
  - WikiSection: +0.43, +0.15, +0.13 and +0.06 up to 32 tokens; about 0 beyond 128.
- Token accuracy by W (fraction of the declared length):

  | W | 0.0625× | 0.125× | 0.25× | 0.5× | 1× | 2× | no window |
  |---|---|---|---|---|---|---|---|
  | PubMed | 0.587 | 0.625 | 0.463 | 0.403 | 0.340 | 0.272 | 0.294 |
  | WikiSection | 0.389 | 0.311 | 0.340 | 0.222 | 0.175 | 0.188 | 0.200 |

  Spurious switches stayed at 58–120 per 1k tokens at every W.
- Conclusion: W = 0.0625 × declared length (2 tokens for S3, 12 for S1), by the pre-stated rule (best
  mean accuracy over the two sets). Recorded in `docs/TEST_SUITE.md` before the scored runs below.

**2026-11-25 · Likelihood vs question, head to head (C), delay by history length (D), windowed question (E)**
- Commands: `uv run python -I experiments/likelihood_vs_question.py run s1_lik s3_qwin s1_q s1_qwin`, then
  `... report` → `results/likelihood_vs_question.md`
- C, PubMed dev (10 abstracts, 36 switches), each method read at every token:
  - windowed likelihood (W = 2): 0.587 accuracy, 26/36 matched, 10.6-token delay, 108 spurious per 1k
  - question, 2 orderings, persist 3: 0.784 accuracy, 34/36 matched, 22.2-token delay, 5.0 spurious per 1k
- C, S1 lag streams (WikiSection test; 12 streams, a symptom→treatment switch each), scored from 16
  tokens before the switch:

  | Method | accuracy | delay (tokens) | spurious per 1k |
  |---|---|---|---|
  | question, full history | 0.879 | 23.7 | 2.9 |
  | question, window W = 12 | 0.581 | 14.8 | 40.0 |
  | windowed likelihood | 0.519 | 54.8 | 51.1 |

- D, mean delay for previous segments of 50 / 200 / 800 / 3000 tokens:
  - question, full history: 21 / 7 / 54 / 13
  - question, window W = 12: 15 / 6 / 29 / 10
  - windowed likelihood: 52 / 74 / 43 / 50
- E on PubMed (W = 2): the question sees 2–3 tokens; accuracy 0.08.
- Conclusion:
  - The question readout stays well ahead of windowed likelihood.
  - With 3 streams per length, D shows no growth of the question method's delay with history length.
  - Windowing the question shortens delay a little but costs about 30 points of accuracy and raises
    spurious switches more than tenfold.

**2026-11-25 · Fixed-lag Viterbi decoder, replayed on saved readings**
- Command: `uv run head-of-state --demo --save-readings experiments/runs/demo_readings.json`; then
  `uv run python -I experiments/decoder_replay.py` → `results/decoder_replay.md`
- Result, against persist 3 (placement error is distance from the true switch to where the final
  labels put it; decision latency adds the lag):
  - Demo: Viterbi L ≥ 8 places every switch 8 tokens after the true switch (persist 3: 10), with 0
    spurious and accuracy 0.869 (persist 3: 0.837).
  - PubMed dev (10 abstracts), L ≥ 8: spurious per 1k drops from 5.8 to 1.0–2.1. Placement 21–25
    tokens (persist 3: 22.7; raw readings 16.6). Accuracy 0.78–0.81 (persist 3: 0.79).
  - WikiSection (2 streams, regimes recur), L ≥ 8: spurious per 1k drops from 17 to 4.5–6.8.
    Placement 15–22 tokens (persist 3: 22.5). Accuracy 0.55–0.58 (persist 3: 0.56).
  - Scaling the emissions by a running regime prior raised PubMed accuracy to 0.88 with 8–9 tokens
    placement. The gain did not hold where regimes recur (WikiSection: placement 7–64, more spurious
    switches). PubMed abstracts list each section once, in a fixed order, which is likely what the
    scaling exploits.
  - Decoded segments under 5 tokens: about 0 per 1k at L ≥ 8.
- Conclusion:
  - Fixed-lag Viterbi mainly removes spurious switches.
  - Placement moves toward the readings' own lag on the demo but not on PubMed.
  - The geometric-duration assumption does not visibly hurt (no surviving short segments), so an
    explicit-duration decoder is not needed yet.

**2026-11-25 · Reading interval adopted as a parameter**
- Decision (owner): a reading every 4th–8th token is acceptable, kept as the `stride` parameter
  (`--stride`). The default stays at every token.
- Change: each reading now labels every token since the previous reading, and `finish()` reads the
  tail, so every token gets a label. Token accuracy in `score()` counts every labelled token. Tested in
  `tests/test_reader.py`, case 9.
- Commands: `uv run head-of-state --demo --stride 8 --rotations 1 --persist 1` (and `--persist 2`, `3`);
  PubMed dev persist × stride replayed offline from `experiments/runs/s3_dev_r5.jsonl`.
- Result:
  - Demo, every 8th token, 1 ordering, persist 1: committed accuracy 0.894, 4 of 4 switches, mean delay
    8 tokens, 0 spurious. With persist 3 at the same stride: 0.645, 3 of 4, 18.7 tokens.
  - PubMed dev (10 abstracts), 1 ordering, every 8th token, persist 1: 0.795 accuracy, 19.3-token
    delay, 7.1 spurious per 1k. Every token, persist 3: 0.779, 22.5, 7.7.
- Conclusion: `persist` counts readings, so it has to shrink with the stride. Every 8th token with
  persist 1 matches every token with persist 3 at about a seventh of the cost.
