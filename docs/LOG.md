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

**2026-11-25 · Evidence mechanism, Path 1b: double counting (offline)**
- Command: `uv run python -I experiments/evidence_table.py` → `results/evidence_table.md`
- Data: the saved asked readings (PubMed dev, 10 abstracts, 1 and 2 option orderings, every token);
  WikiSection validation (2 streams, regimes recur).
- New scorer: `src/head_of_state/metrics.py`. It adds boundary F1 within ±8 and ±16 tokens, and Brier
  score and ECE on the probabilities each decoder reports.
- Result, PubMed, 1 ordering:
  - Raw argmax: accuracy 0.794, ECE 0.037, 41 spurious per 1k.
  - Forward filter: accuracy 0.774, ECE 0.205, 3.5 spurious per 1k. Tempering (η = 0.5, 0.25) leaves
    ECE at 0.20 and lowers accuracy.
  - Dividing by the mean prediction over unlabelled positions: 0.758, ECE 0.22.
  - Dividing by a running prior: accuracy 0.834 and delay 13.2 tokens (filter: 25.9), but 12.5
    spurious per 1k.
- Result, WikiSection: the running prior gives accuracy 0.563 (raw 0.564), delay 12.0 tokens (filter:
  17.3) and 17 spurious per 1k.
- Conclusion: filtering the asked probabilities makes them overconfident (ECE about 0.04 → 0.20),
  which is the double-counting signature; tempering does not repair it. The running prior trades lower
  delay for more spurious switches, and its accuracy gain holds only on PubMed.

**2026-11-25 · Evidence mechanism, Path 2a: how concentrated the likelihood evidence is (offline)**
- Command: `uv run python -I experiments/likelihood_concentration.py` → `results/likelihood_concentration.md`
- Result:
  - Description only at the start: the top 10% of tokens carry 74% (PubMed) and 69% (WikiSection) of
    the positive log-likelihood ratio, and the mean ratio is about 0.
  - With the declared hopped window: 48% and 61%, and the mean ratio is positive.
  - On PubMed the evidence sits on number and statistics tokens (`0`, `=`, `P`, `%`, `vs`, `placebo`).
- Conclusion: the evidence decays within about 32 tokens and sits on a few content tokens. That meets
  the brief's stop condition, so Path 2 is closed as no-go and the reset-on-switch variant is not run.

**2026-11-25 · Supervised ceiling, checked**
- Source: Dernoncourt & Lee 2017, Table 3 (https://ar5iv.labs.arxiv.org/html/1710.06071).
- Result: bi-ANN (bi-LSTM token and sentence layers with a CRF) scores F1 90.0 on the PubMed 20k RCT
  test set. The paper does not state the averaging. It is a sentence-level score on the test split,
  so it is not directly comparable with token accuracy on dev.

**2026-11-25 · Evidence mechanism, Paths 1a and 1c: asking with the question seeing only the last W tokens**
- Commands: `uv run python -I experiments/windowed_asking.py run`, then `... report` →
  `results/windowed_asking.md`
- Setup: windows declared before the runs as fractions of the declared segment length: PubMed
  1×, 2× and 4× (30 / 60 / 120 tokens); S1 0.25×, 0.5× and 1× (48 / 95 / 190 tokens). 1 option
  ordering, a reading every 2nd token, one batched pass per reading over opening + window + question
  (79–171 ms per reading).
- Result, PubMed (10 abstracts), delay of the raw readings:
  - full history: 16.4 tokens
  - last 30 tokens: 17.9
  - last 60 tokens: 15.2
  - last 120 tokens: 21.9

  Accuracy 0.756 / 0.796 / 0.761 against 0.795 with full history.
- Result, S1 (12 streams), forward filter:
  - delay 18.8 / 22.5 / 20.0 tokens against 23.7 with full history
  - accuracy 0.767 / 0.826 / 0.857 against 0.879
  - only the 800-token history condition gets shorter (about 37 against 54 tokens)
- Conclusion: restricting the question to recent tokens does not remove the lag, so earlier history is
  not its cause. A new section stays ambiguous to the question until enough of it has arrived. Small
  windows also cost accuracy on S1.

**2026-11-25 · Evidence mechanism, Path 3: activation directions and the asking cascade**
- Commands: `uv run python -I experiments/activation_directions.py run`, then `... report` →
  `results/activation_directions.md`; the consolidated table is `results/evidence_table.md`.
- Setup:
  - Residual-stream activations at layers 16 / 20 / 24 / 28 of 32, from one pass over each abstract
    (0.51 ms per token).
  - 3a: 20 passages per regime written by the model from the regime's name and description only.
  - 3b: directions fitted on abstracts 1–5 labelled by asking (full history, running-prior filter);
    no gold labels.
  - 3c: logistic probe on gold labels of abstracts 1–5.
  - 3b and 3c are scored on abstracts 6–10 (18 switches). The layer, 24, was chosen with 3a on all
    10 dev abstracts.
- Result on abstracts 6–10, forward filter:

  | Mechanism | accuracy | delay (tokens) | spurious per 1k | ECE |
  |---|---|---|---|---|
  | asking | 0.760 | 27.3 | 2.1 | 0.22 |
  | 3a (model-written passages) | 0.743 | 57.7 | 18.3 | 0.17 |
  | 3b (distilled from asking) | 0.821 | 7.9 | 19.1 | 0.10 |
  | 3c (gold-label probe) | 0.847 | 24.6 | 28.7 | 0.12 |
  | bge-m3 similarity | 0.599 | — | — | — |

  3a misses 8 of 18 switches.
- Result, 3d cascade (the cheap reading everywhere, the asked reading wherever the filter's entropy
  is above a threshold):
  - With 3a, asking at 9% of tokens gives 0.810 accuracy, 19.6-token delay and 2.9 spurious per 1k,
    at an estimated 9 ms per token. That is better than either part alone.
  - With 3b, asking more never helps: 3b alone is already ahead of asking.
- Conclusion: directions distilled from asking are the most promising per-token sensor so far. They
  are 200 times cheaper than asking and switch 3.5 times sooner, but have more spurious switches.
  The sample is small (5 held-out abstracts), the directions are fitted per regime set without gold
  labels (a departure from the original brief), and they have not yet been tested where regimes
  recur or on S1.

**2026-11-25 · Distilled activation directions on more data and where states recur**
- Commands: `uv run python -I experiments/distilled_directions.py run`, then `... report` →
  `results/distilled_directions.md`
- Setup: the Path 3 settings, unchanged (layer 24, each token's own activation, difference of means,
  warm-up labels from the running-prior filter over asked readings).
- Result, PubMed dev abstracts 11–40 (97 switches), directions fitted on asked labels of abstracts 1–10:
  - Distilled directions: 0.734–0.756 accuracy, 17–20-token delay, 11–23 spurious per 1k.
  - Asking every 8th token alone: 0.694 accuracy, 34.5-token delay.
  - Both fused in one filter: 0.790 accuracy, 13.5-token delay, 29 spurious per 1k, at about 15 ms
    per token.
- Result, WikiSection validation streams 3–6 (10 states, recurring), directions fitted on asked labels
  of streams 0–2:
  - Distilled directions: 0.28–0.33 accuracy.
  - Asking alone (every 2nd token): 0.599.
  - Fused: 0.556–0.582.
- Result, S1 lag streams (directions fitted on WikiSection validation):
  - Distilled directions: 0.702 accuracy, 13.3-token delay, 30 spurious per 1k.
  - Full-history asking: 0.879, 23.7 tokens, 2.9 per 1k.
  - The distilled delay does not grow with history (19 / 5 / 23 / 6 tokens).
- Conclusion: the earlier 0.82 on 5 held-out abstracts did not hold up. Distilled directions are fast
  and react sooner, but they generalise poorly, and with 10 similar states they fail. Asking stays the
  per-token sensor. Directions are worth keeping only as a supplement fused with sparse asking, and
  only on state sets where they carry signal (here PubMed).

**2026-11-25 · First scored runs with frozen settings (PubMed test, null streams)**
- Frozen settings, recorded in `docs/TEST_SUITE.md` before the runs: 1 option ordering, a reading
  every 8th token, persist 1, prior-power 0, max_ctx 8192, keep_ctx 2048, forward filter η = 1
  (persist 1 reported alongside).
- Commands: `uv run python -I experiments/scored_runs.py run`, then `... report` →
  `results/scored_runs.md`
- Result, S3 (30 PubMed 20k RCT test abstracts, 98 switches, 13.7 ms per token):
  - persist 1: token accuracy 0.701, 83/98 switches, mean delay 26.8 tokens, 8.2 spurious per 1k
  - forward filter: 0.647, 77/98, 39.0 tokens, 3.0 spurious per 1k
  - The same setting scored 0.795 on 10 dev abstracts.
- Result, S2 (4 single-state WikiSection test streams, 15,039 tokens, 21.1 ms per token): false
  switches per 1k, 24.5 with persist 1 and 5.3 with the filter. Share of tokens on the true state:
  - treatment 95%, diagnosis 88%
  - symptom 48%, often read as epidemiology or mechanism
  - cause 49%, often read as mechanism or epidemiology
- Conclusion: on held-out data the cheap frozen setting scores about 0.70, well below the 0.80 seen on
  the small dev sample. Some states are rarely mistaken and others are often confused with related
  states, so the descriptions of those states are the next lever.

**2026-11-25 · S1 seam control (re-anchoring the cache)**
- Commands: `uv run python -I experiments/seam_runs.py run`, then `... report` → `results/seam_runs.md`
- Setup: one stream of whole WikiSection en_disease test articles (15,440 tokens), frozen settings.
  The cache re-anchors to opening + last 2,048 tokens when it would pass 8,192, at stream positions
  7,936 and 13,824. Run B starts about 1,000 tokens later, so each switch sits at a different distance
  from the seams.
- Result:
  - Switches within 500 tokens after a seam: 7 of 7 found, mean delay 20.6 tokens.
  - Elsewhere: 57 of 69 found, 34.0 tokens.
  - Matched pairs (the same switch near a seam in one run and away from it in the other) were no
    slower near the seam.
  - Committed switches in the 24 tokens right after a seam: 4 across the 4 seams, against about 2
    expected from the background rate. Inconclusive.
  - The background rate on these whole-article streams is 19–21 spurious switches per 1k (persist 1).
- Conclusion: re-anchoring does not delay or lose switches. A small label jump right at the seam is
  possible but not established with 4 seams.

**2026-11-25 · H2O-Lightning-4B as the readout model**
- Why: H2O-Lightning-4B (h2oai/h2o-lightning-4b, Apache-2.0) is an open-weights Jev-class decision
  model fine-tuned from Qwen3.5-4B to answer choice questions about a record with calibrated option
  probabilities. It was not considered when the base model was chosen.
- Setup: `head_of_state.h2o.H2OReader` builds the model's own prompt format around the cached stream
  (system line, `record:`, `question:`, lettered `options:`, `Answer:` prefill, choice temperature
  0.75). The prompt text was checked identical to the model's serving shim. Frozen settings and the
  scored streams of the previous entry.
- Commands: `H2O_LIGHTNING_PATH=<checkout> uv run python -I experiments/h2o_readout.py run`, then
  `... report` → `results/h2o_readout.md`
- Result, PubMed test (30 abstracts), persist 1, base model → H2O-Lightning:
  - token accuracy 0.701 → 0.757
  - boundary F1 ±8 0.26 → 0.43
  - mean delay 26.8 → 18.8 tokens
  - spurious per 1k 8.2 → 7.2
  - ECE 0.142 → 0.059
  - cost about 13–14 ms per token for both
- Result, null streams: the true state held for 0.705 of tokens (base 0.622), with 14.4 false
  switches per 1k under persist 1 (base 24.5) and 4.3 under the filter (base 5.3).
- Conclusion: the decision-tuned model is better on every measure at the same cost, including about 8
  tokens less delay and much better calibrated probabilities.

**2026-11-25 · H2O-Lightning-4B made the default model**
- Decision (owner): H2O-Lightning-4B becomes the default model of the `head-of-state` command and the
  demo. `--prompt auto` uses its prompt format for H2O-Lightning models and the original format for any
  other model (`--model Qwen/Qwen3.5-4B`).
- Commands: `uv run head-of-state --demo --model <local checkout>`, with and without
  `--stride 8 --rotations 1 --persist 1`; and `uv run head-of-state --demo --model Qwen/Qwen3.5-4B --stride 8 --rotations 1 --persist 1`
- Result, demo:
  - H2O-Lightning, defaults: committed accuracy 0.841, 4 of 4 switches, 9.8-token mean delay,
    0 spurious (base model: 0.837, 4/4, 10.0, 0).
  - H2O-Lightning, every 8th token with persist 1: 0.861, 4 of 4 switches, 2 spurious. "Thanks Priya,
    nice to meet you" is read as closing for one reading.
  - The base model at the same setting still gives 0.894 with no spurious switches.
- Conclusion: on the 245-token demo the two models are on a par. The 30-abstract test comparison is
  the evidence for the default.

**2026-11-25 · S5: speaker roles in doctor–patient conversations (ACI-Bench)**
- Data: ACI-Bench (CC BY 4.0) `clinicalnlp_taskB_test1`, the first 10 two-speaker encounters (11,921
  tokens). Speaker tags removed, one turn per line, the tags as gold. Declared length 12 tokens (median
  turn, train split; recorded in `docs/TEST_SUITE.md` before the run). Unit "the last turn".
  H2O-Lightning-4B.
- Commands: `H2O_LIGHTNING_PATH=<checkout> uv run python -I experiments/aci_bench.py run`, then `... report`
  → `results/aci_bench.md`
- Result, frozen settings (every 8th token, 12.8 ms per token):
  - token accuracy 0.855 (persist 1) and 0.862 (filter)
  - switches matched 181–199 of 378, caught within about 3 tokens
  - 2.6–4.8 spurious per 1k, ECE 0.03–0.06
- Result, every token on 3 encounters (not frozen, 95.7 ms per token): 0.835–0.846 accuracy, 76–77 of
  194 switches matched. The frozen setting on the same 3 encounters scored 0.875–0.882.
- Conclusion: who is speaking is read quickly and accurately, so detection lag depends on how clearly
  the text signals a change. About half the turn changes are missed because many turns are shorter
  than the reading interval or the commit rule allows; reading every token does not recover them.

**2026-11-25 · Correction: delay is now counted from the switch token**
- Change: `score()` counted a switch's delay from the first reading after the true switch. With a
  reading every N tokens that left out the wait for that reading (0 to N−1 tokens), so delays of
  sparse-reading runs were understated. Delay now runs from the token where the true switch happens.
  Tested in `tests/test_reader.py`, case 10. Every results table was regenerated from the saved runs.
- Corrected delays (old → new), runs with sparse readings:
  - PubMed test, every 8th token: base model, persist 1, 26.8 → 30.7 tokens (filter 39.0 → 42.8);
    H2O-Lightning, persist 1, 18.8 → 22.8 (filter 27.0 → 31.0).
  - ACI-Bench, every 8th token: 2.9–3.8 → 6.4–7.2 tokens. A breakdown of the caught turn changes:
    3.4 tokens of wait until the next reading, then 86% labelled correctly at that first reading.
  - Seam control: 20.6 → 23.4 near seams, 34.0 → 36.9 elsewhere.
  - Asking every 8th token on PubMed dev 11–40: 34.5 → 38.3. Windowed asking and S1 runs read every
    2nd token and changed by about 1 token (`results/windowed_asking.md`,
    `results/likelihood_vs_question.md`).
  - Stride sweep: see `results/stride_sweep.md`.
- Accuracy, switch counts and spurious rates are unchanged. Runs that read every token are unchanged.

**2026-11-25 · Bisection between disagreeing readings; per-token scoring; Viterbi on the scored sets**
- Change 1, `--refine` (`StreamStateReader.refine`): with a reading every N tokens, where two
  consecutive readings disagree on the likeliest state, the reader bisects the gap (+4, then ±2, then
  ±1 tokens for N = 8). Each probe rebuilds the cache from a snapshot taken at the previous reading, so
  the main cache is untouched. Tested in `tests/test_reader.py` (cases 11 and 11b): probes equal the
  readings of an every-token reader, and regular readings are unchanged.
- Change 2, scoring: `metrics.stream_counts` averaged accuracy, Brier and ECE over readings and placed
  a predicted boundary at the reading. It now expands each reading over the tokens it labels and scores
  per token, with a boundary at the first token of the span whose label changes. Runs that read every
  token are unchanged. With a fixed interval the change is small (PubMed test, base model 0.701 →
  0.712; H2O-Lightning 0.757 → 0.763; ACI-Bench 0.855 → 0.842). With bisection it is large, because
  probes concentrate at ambiguous boundaries. All tables using these metrics were regenerated.
- Commands: `H2O_LIGHTNING_PATH=<checkout> uv run python -I experiments/refine_runs.py run`, then
  `... report` → `results/refine_runs.md`
- Result, H2O-Lightning, frozen settings, persist 1 (every 8th token → plus bisection):

  | Set | token accuracy | boundary F1 ±2 | delay (tokens) | spurious per 1k | ms per token |
  |---|---|---|---|---|---|
  | ACI-Bench | 0.842 → 0.855 | 0.24 → 0.39 | 6.4 → 3.6 | 2.6 → 1.7 | 13 → 18 |
  | PubMed test | 0.763 → 0.755 | 0.22 → 0.20 | 22.8 → 20.1 | 7.2 → 8.5 | 13 → 18 |

- Result, Viterbi on the same readings:
  - Offline Viterbi on PubMed: boundary F1 ±16 0.52 → 0.61, spurious per 1k 7.2 → 2.1.
  - On ACI-Bench it adds spurious switches (17–49 per 1k). With 12-token turns and a reading every
    8 tokens, the declared length is 1.5 readings, so the transition model is no longer sticky.
- Conclusion: bisection resolves switches below the reading interval where the change is sharp
  (speaker turns), at about 40% more cost. It cannot place a switch before the readings turn, so it
  does not help where they drift (PubMed sections). Viterbi suits sets whose segments span several
  readings.

**2026-11-25 · Audit of the evaluation code, and fixes**
- An independent review of the scoring and evaluation code, plus my own review, found these problems.
  All are fixed in code, and every results table was regenerated from the saved runs.
  1. **Bisection delay.** A switch placed by a bisection probe was scored from the probe's position,
     but it was only decided at the next regular reading. Readings now carry `decided`, and delay runs
     to it. Bisection does not shorten delay (ACI-Bench 5.9 → 6.0 tokens, PubMed 22.8 → 23.4). It
     sharpens placement only (ACI-Bench boundary F1 ±2 0.24 → 0.39). The earlier "6.4 → 3.6" was wrong.
  2. **Switch counts.** True switches were counted only between readings, so short segments vanished
     at stride 8. They now come from the token-level gold, and an unread segment counts as missed.
     ACI-Bench: 190 of 545 turn changes caught (about a third), not "about half".
  3. **Transition model.** It was built from "declared length ÷ readings". With 12-token turns read
     every 8 tokens, that favoured switching. It is now the per-token chain raised to the gap between
     readings (`reader.sticky_transition`). The ACI-Bench filter has 1.5 spurious per 1k (was 4.8),
     and Viterbi on ACI-Bench is no longer degenerate.
  4. **Two accuracy definitions.** `decoders.evaluate` scored one token per reading. It now goes
     through the same per-token scoring as everything else.
  5. **Lookahead.** At a reading every 8th token, a token's label comes from a reading up to 7 tokens
     later. That is why the every-8th-token accuracy looked higher than every-token accuracy. The
     every-token readings rebuilt into the every-8th-token scheme give the same accuracy (PubMed test
     0.759 against 0.763), and with 4–8 tokens of lookahead every-token labels reach 0.762–0.764.
  6. **Mixed comparisons and selection.**
     - Some comparisons changed the decoder together with the reading interval.
     - The README picked a different decoder per row.
     - H2O-Lightning and the revised settings were chosen after seeing the first test samples.
     - Headline numbers now come from new held-out samples with one pre-registered decoder (next
       entry).
     - The earlier claim that H2O-Lightning was "better on every measure" was wrong: under the
       forward filter it had more spurious switches (3.8 against 3.0 per 1k).
  7. **Unequal starting points in S1.** In the S1 comparison, the likelihood path was filtered over
     the whole stream but the question method only near the switch. Both are now decoded and scored
     from the same token.
  8. **Unread tails.** Tokens after the last regular reading were never read. `run_stream` now calls
     `finish()`; runs made before this miss up to the last stride − 1 tokens per stream.
- Tests: `tests/test_reader.py`, case 12 (unread segments counted as missed, decision-time delay, a
  sticky transition across 8-token gaps).

**2026-11-25 · Revised judging (settled labels) and the held-out confirmation**
- Judging, revised before the held-out samples were scored and recorded in `docs/TEST_SUITE.md`:
  - The output is judged on its settled labels: Viterbi over all readings, each reading treated as
    evidence for the state round(0.2 × declared length) tokens earlier. The code is
    `head_of_state.settle`.
  - Time to notice (the online decision, no lookahead) is reported beside it and not counted as error,
    because a stream read left to right cannot know what comes next.
  - Reason: the earlier scoring compared a live labeller with hindsight gold. At "Okay, let's…" the
    readings correctly answer "problem" about the previous sentences, so the first tokens of every new
    segment were counted as errors that no left-to-right reader can avoid.
- Commands: `H2O_LIGHTNING_PATH=<checkout> uv run python -I experiments/scored_heldout.py run S3|ACI|S2`;
  `uv run python -I experiments/settled_judging.py` → `results/settled_judging.md`;
  `uv run python -I experiments/scored_heldout.py report` → `results/scored_heldout.md` (live decoders)
- Result, held-out samples (H2O-Lightning-4B, every token, 1 ordering; about 100 ms per token):

  | Sample | settled accuracy | boundary F1 ±8 / ±16 | switches placed within 16 tokens | spurious per 1k | time to notice |
  |---|---|---|---|---|---|
  | PubMed test abstracts 31–60 | 0.788 | 0.51 / 0.57 | 63 of 95 | 4.5 | 22.8 tokens |
  | ACI-Bench encounters 11–20 | 0.868 | 0.56 / 0.58 | 199 of 476 | 0.7 | 5.1 tokens |
  | 2 null streams (symptom, cause) | 0.468 | — | — | 7.0 | — |

  Live labels with the forward filter, for comparison: 0.750 (PubMed) and 0.841 (ACI-Bench).
- Result, demo with settled labels: switches at tokens 51 / 95 / 167 / 214 against true 49 / 97 / 163 /
  213; settled token accuracy 0.963 (live 0.841).
- Conclusion: once corrected labels are judged, most of the apparent lag disappears. What remains is
  the time to notice, set by how soon the text reveals the change. Short conversational turns are still
  mostly missed, and related states (symptom, cause) are still often confused.

**2026-11-26 · How to ask a Jev-class model: documented question formats, small comparison**
- Research: a literature and documentation survey of how Jev-class decision models should be asked.
  Sources: TypeSafe's Jev docs, the H2O-Lightning card, Sun et al., AnyJev, LLM-as-Jev, Large Decision
  Models, Releasing the Inner Jev. Documented recommendations:
  - state the exact condition and mark the target in the record;
  - use neutral option names, with the meaning carried by the descriptions;
  - average over option orders;
  - prefer a JSON record with named parts;
  - send only what the decision needs;
  - give examples for options the model confuses;
  - one joint choice beats one yes/no question per option, by 2–5 points in the papers.
- Command: `H2O_LIGHTNING_PATH=<checkout> uv run python -I experiments/ask_variants.py run_research`,
  then `... report` → `results/ask_variants.md`
- Setup: tuning data only, small (3 PubMed dev abstracts, 1 WikiSection validation stream, 1 ACI-Bench
  train encounter), a reading every 4th token, H2O-Lightning-4B. Judged on settled labels.
- Result (settled token accuracy; current question → R1 → R2):
  - PubMed: 0.878 → 0.893 → 0.927, time to notice 18.3 → 10.7 → 7.1 tokens
  - ACI-Bench: 0.847 → 0.923 → 0.924, turn changes placed within 16 tokens 16 → 37 of 54
  - WikiSection: 0.675 → 0.547 → 0.523
  - R1 is the current sentence (or line) as its own part of the record with a direct question; R2 adds
    neutral option names.
- Other arms against R1:
  - two option orders, a JSON record, a domain-worded question, recent text only and two real examples
    per state did not help;
  - one yes/no question per state was much worse on every set (PubMed 0.452).
- Conclusion: marking the current sentence or line as the target, with neutral names, helps where the
  states change at that scale (sections of an abstract, speaker turns). It hurts for multi-sentence
  topic sections (WikiSection), where one sentence alone is often ambiguous. These are small samples;
  a larger check and a passage-sized target for WikiSection are next.

**2026-11-26 · Targeted question, larger tuning check**
- Command: `H2O_LIGHTNING_PATH=<checkout> uv run python -I experiments/ask_confirm.py run`, then `... report`
  → `results/ask_confirm.md`
- Setup: tuning samples not used in the first comparison; a reading every 4th token; settled labels.
- Result (current question → R2):
  - ACI-Bench train encounters 2–4 (230 turn changes): accuracy 0.871 → 0.905, turn changes placed
    within 16 tokens 63 → 159, boundary F1 ±8 0.42 → 0.81, time to notice 5.3 → 2.9 tokens.
  - PubMed dev abstracts 4–10: accuracy 0.894 → 0.872, time to notice 18.3 → 10.6 tokens.
  - WikiSection validation, with the last 3 sentences as the target: 0.533 → 0.507.
- Conclusion: marking the current line as the target is a large gain for conversations. For abstracts
  it trades a little accuracy for faster noticing, and the earlier 0.927 came from 3 abstracts. For
  multi-sentence topic sections it does not help.

**2026-11-26 · Targeted question on conversations, held-out check**
- Declared in `docs/TEST_SUITE.md` before the run: ACI-Bench test encounters 21–30, the current
  question against R2 (the current line as its own part of the record, a direct question, neutral
  option names), both read every 4th token by H2O-Lightning-4B, settled labels.
- Commands: `H2O_LIGHTNING_PATH=<checkout> uv run python -I experiments/ask_heldout.py run`, then
  `... report` → `results/ask_heldout.md`
- Result (507 turn changes, current → targeted):
  - token accuracy 0.793 → 0.831
  - turn changes placed within 16 tokens 167 → 341 (33% → 67%)
  - boundary F1 ±8 0.47 → 0.79
  - time to notice 7.7 → 3.7 tokens
  - spurious per 1k 0.7 → 0.6
- Conclusion: confirmed on held-out conversations. Available as `--target line` (off by default).

**2026-11-26 · Conversations without line breaks (stopped; speaker work parked)**
- Command: `H2O_LIGHTNING_PATH=<checkout> uv run python -I experiments/aci_unsegmented.py run` (stopped
  after the first version), then `... report`
- Setup: ACI-Bench test encounters 31–38 joined with spaces, so speaker changes are not marked by line
  breaks (the text has no punctuation either); the current question, a reading every 4th token, settled labels.
- Result (8 encounters, 10,502 tokens, 355 speaker changes): token accuracy 0.793; labelling every token
  "doctor" scores 0.734. 67 of 355 changes placed within 16 tokens; time to notice 11 tokens. The
  comparison with line breaks was not run.
- Majority-state accuracy of earlier conversation samples: 0.729 (held-out encounters 11–20, reported
  0.87) and 0.67 (encounters 21–30).
- Conclusion: in all earlier conversation runs each turn was one line, so the boundaries were given and
  only the speaker was inferred. Without that, the gain over always answering "doctor" is 6 points.
  Speaker turns (median 12 tokens) are not the state changes this tool is for; this line of work stops
  here, and evaluation returns to HANDOFF task 5 (labelled documents joined into streams, several
  unrelated state lists).

**2026-11-26 · Joined labelled documents, three unrelated state lists (HANDOFF task 5)**
- Declared in `docs/TEST_SUITE.md` before the run. Commands: `uv run python -I experiments/joined_documents.py
  measure`; `H2O_LIGHTNING_PATH=<checkout> uv run python -I experiments/joined_documents.py run`, then `... report`
  → `results/joined_documents.md`
- Setup: segments of one or more same-label documents (sizes 64–512 tokens), joined with single spaces, so
  nothing in the formatting marks a switch; 3 streams of about 2,000 tokens per list, seed 0; a reading at
  every token, H2O-Lightning-4B. Declared lengths from seed-1 streams: AG News 270, WikiSection 441, IMDB 274.
- Result, settled labels with the pre-registered reading shift (0.2 × declared length) / without it:
  - AG News, 4 topics: token accuracy 0.779 / 0.865 (always the most common state 0.338); switches placed
    within 16 tokens 6 / 15 of 21; time to notice 33 tokens.
  - WikiSection, 10 section types: 0.616 / 0.683 (0.271); 3 / 4 of 8; time to notice 90 tokens.
  - IMDB, positive / negative: 0.890 / 0.889 (0.507); 3 / 7 of 19; time to notice 49 tokens.
- The shift moves every reading 54–88 tokens earlier here, and placed AG News switches a median 48 tokens
  early. Without it, Viterbi puts switches after the true point (AG News median +6 tokens, IMDB +28: a
  review's opening plot summary carries no sentiment).
- Time to notice did not grow with the length of the segment before the switch (HANDOFF question 1), on
  4–12 switches per length bin.
- Conclusion: on streams whose switches are not marked by formatting, the readings plus Viterbi beat the
  most-common-state baseline by 38–53 points. The 0.2 × length shift is wrong for long segments: how far a
  reading runs behind the text depends on how much new text the model needs, not on the segment length. The
  unshifted rule was chosen after seeing these streams, so it is confirmed on new streams (seed 2) next.

**2026-11-26 · Joined labelled documents, confirmation on new streams; judging revised**
- Declared in `docs/TEST_SUITE.md` before the run. Commands: `H2O_LIGHTNING_PATH=<checkout> uv run python -I
  experiments/joined_documents.py run --seed 2`, then `... report --seed 2` → `results/joined_documents_seed2.md`
- Setup: as the seed-0 run; 3 new streams per list built with seed 2 from documents not used in the seed-0
  streams.
- Result, settled labels without the reading shift (with the 0.2 × length shift in brackets):
  - AG News: token accuracy 0.874 (0.832), always the most common state 0.339; switches placed within 16
    tokens 10 of 18 (3); time to notice 30 tokens.
  - WikiSection: 0.655 (0.642), baseline 0.296; 5 of 11 (2); time to notice 92 tokens.
  - IMDB: 0.809 (0.811), baseline 0.665; 5 of 15 (1); time to notice 55 tokens.
- The held-out PubMed and null streams re-judged without the shift: PubMed 0.776 (0.788 with its 6-token
  shift), 63 of 95 placed either way; null streams 0.468 either way.
- Conclusion: confirmed. Settled labels are now the Viterbi path over the readings where they were taken
  (TEST_SUITE judging revised; the CLI's settled output changed to match). Viterbi still places most switches
  after the true point, because the readings name the new state only once its text shows it.

**2026-11-26 · Does the earlier stream hold the readings back? (HANDOFF question 1, joined documents)**
- Commands: `H2O_LIGHTNING_PATH=<checkout> uv run python -I experiments/joined_inertia.py run`, then
  `... report` → `results/joined_inertia.md`
- Setup: for each of the 48 switches in the seed-0 joined-document streams, the first 128 tokens of the new
  segment read again alone (nothing before them), against the in-stream readings at the same tokens.
- Result, mean probability of the new state 8 / 16 / 64 tokens into the new segment, in stream → alone:
  - AG News: 0.46 → 0.76, 0.56 → 0.82, 0.67 → 0.79
  - IMDB: 0.34 → 0.65, 0.46 → 0.74, 0.71 → 0.89
  - WikiSection: 0.35 → 0.33, 0.42 → 0.55, 0.61 → 0.70
- Conclusion: for news topics and review sentiment, the text before the switch holds the readings back:
  the new text alone is named within a few tokens, in the stream only after tens of tokens. For disease
  article sections the new text itself is ambiguous at first, as the earlier windowed-question runs found
  on WikiSection. Part of the hold-back is the question itself, which asks about "the last few sentences"
  and so still covers the previous segment for the first sentence or two.

**2026-11-26 · Joined documents: asking about the current sentence (tuning streams)**
- Declared in `docs/TEST_SUITE.md` before the run. Commands: `H2O_LIGHTNING_PATH=<checkout> uv run python -I
  experiments/joined_documents.py run --target sentence`, then `... report --target sentence` →
  `results/joined_documents_target_sentence.md`
- Setup: the seed-0 streams; the current sentence as its own part of the record, a direct question,
  neutral option names; every token; settled labels without shift.
- Result, current question → current-sentence question:
  - AG News: accuracy 0.865 → 0.881, switches placed within 16 tokens 15 → 18 of 21, time to notice
    33 → 13 tokens, spurious per 1k 3.1 → 4.9
  - WikiSection: 0.683 → 0.628, 4 → 8 of 8, 90 → 70 tokens, spurious 4.8 → 14.0
  - IMDB: 0.889 → 0.881, 7 → 7 of 19, 49 → 46 tokens, spurious 2.1 → 2.3
- Offline, both readings combined as two emission streams in Viterbi (product of the probabilities, each
  raised to 0.5): AG News 0.903, WikiSection 0.675, IMDB 0.881, at twice the reading cost.
- Conclusion: not better on every list, so no confirmation run and no change of default. Asking about one
  sentence notices news-topic changes much sooner but gives noisier readings where a state spans many
  sentences (disease-article sections), as on WikiSection before.

**2026-11-26 · Joined documents: reading interval and decoder length (HANDOFF task 6, offline)**
- Commands: `uv run python -I experiments/joined_knobs.py > results/joined_knobs.md`, then the
  `plot_tradeoff.py` line in its docstring → `results/joined_knobs.html`
- Setup: the saved every-token readings of the seed-0 streams, replayed. Keeping every Nth reading
  reproduces `--stride N`, since a reading at token N sees the same cache either way. Decoder length 0.25×
  to 4× the declared length; reading interval 1 to 16 tokens.
- Result:
  - The decoder length changes almost nothing at one reading per token (AG News accuracy 0.865 at every
    length): each reading is counted as independent evidence, and with one per token they outweigh the
    transition probabilities.
  - Reading less often lowers false switches and keeps or raises settled accuracy; time to notice grows.
    Every token → every 8th token: AG News 0.865 → 0.886, false switches 3.1 → 1.9 per 1k, time to notice
    33 → 50 tokens; WikiSection 0.683 → 0.706, 4.8 → 1.6; IMDB 0.889 → 0.891, 2.1 → 1.5, 49 → 64.
  - Keeping every reading but raising each to the power η (η = 1/8) gives the same trade: AG News 0.898,
    WikiSection 0.712, IMDB 0.885.
  - Mean time to notice is averaged over the switches the live output noticed, so it moves when one slow
    switch drops in or out (WikiSection: 90 tokens at every token, 26 at every 2nd).
- Conclusion: consecutive readings are not independent evidence, and the decoder treats them as if they
  were. Reading every token buys earlier noticing only; settled labels are as good or better from a reading
  every 4th–8th token at a quarter to an eighth of the cost.

**2026-11-26 · Reading weight η = 1/8, checked on the confirmation streams (offline)**
- Declared in `docs/TEST_SUITE.md` before scoring. Command: `uv run python -I experiments/joined_eta.py >
  results/joined_eta.md`
- Result, η = 1 → 1/8 on the saved seed-2 readings (every token):
  - AG News: accuracy 0.874 → 0.879, placed within 16 tokens 10 → 8 of 18, false switches 2.4 → 1.6 per
    1k, time to notice 30 → 56 tokens
  - WikiSection: 0.655 → 0.669, 5 → 5 of 11, 5.3 → 2.1, 92 → 197
  - IMDB: 0.809 → 0.833, 5 → 4 of 15, 2.4 → 1.7, 55 → 90
- Conclusion: a trade, not a gain: slightly higher accuracy and 30–60% fewer false switches for much later
  noticing and a few switches placed further off. The default stays η = 1; the trade is the same one a
  longer reading interval gives at lower cost.

**2026-11-26 · Decoders on the same readings, joined documents (HANDOFF question 2, offline)**
- Command: `uv run python -I experiments/joined_decoders.py > results/joined_decoders.md`
- Result, seed-2 streams, every token (raw argmax / persist 3 / forward filter / Viterbi):
  - AG News: accuracy 0.858 / 0.858 / 0.859 / 0.874; false switches per 1k 27.9 / 6.6 / 3.4 / 2.4
  - WikiSection: 0.630 / 0.634 / 0.628 / 0.655; 91.1 / 24.1 / 10.5 / 5.3
  - IMDB: 0.800 / 0.799 / 0.803 / 0.809; 33.2 / 8.2 / 4.8 / 2.4
- Conclusion: as the literature in HANDOFF expects, decoding changes stability far more than accuracy:
  Viterbi removes about 90% of the raw readings' false switches and adds 1–2.5 points of accuracy. Raw
  argmax places the most switches within 16 tokens only because it switches so often.

**2026-11-26 · Settled labels from the evidence each reading adds**
- Declared in `docs/TEST_SUITE.md` before scoring the sets below. Command: `uv run python -I
  experiments/joined_evidence.py > results/joined_evidence.md`
- Reason: a reading is the model's answer given all text so far, a running estimate of the state, which is
  what an HMM's forward pass produces. Decoding readings as evidence counts the history again at every token
  (why the decoder length barely mattered, and why consecutive readings acted as repeated evidence). The
  forward identity estimate_t ∝ evidence_t × (estimate_{t-1} A) gives evidence_t ∝ reading_t / (reading_{t-1}
  A); Viterbi decodes that with the declared segment length (`settle.reading_evidence`).
- The A in that identity is how fast the readings themselves change, a property of the model and question.
  Chosen once on the seed-0 streams from 10 / 20 / 50 tokens: 20.
- Result, current settled labels → recovered evidence (sets not used to choose it; every token):
  - AG News (seed 2): accuracy 0.874 → 0.850, placed within 16 tokens 10 → 8 of 18, false switches 2.4 → 0.9
    per 1k
  - WikiSection (seed 2): 0.655 → 0.705, 5 → 5 of 11, 5.3 → 1.3
  - IMDB (seed 2): 0.809 → 0.838, 5 → 5 of 15, 2.4 → 1.5
  - PubMed abstracts 31–60: 0.776 → 0.783, 63 → 53 of 95, 4.9 → 2.2
  - single-state streams (seed 1): 0.468 → 0.574, false switches 7.2 → 2.1
- Conclusion: false switches fall by 40–75% on every set and accuracy rises on four of five (AG News loses
  2.4 points); a few switches are placed further off. Adopted as the tool's settled labels (CLI); the demo
  settles at 0.894 token accuracy. README results updated.

**2026-11-26 · Correction: recovered-evidence decoding withdrawn from the tool**
- The previous entry's second transition table (switch probability 1/20 per token, used to turn readings
  into evidence) was not estimated from anything: 20 was picked from 10 / 20 / 50 by accuracy against true
  labels, and its description as "how fast the readings themselves change" was never measured. The forward
  identity it relies on assumes the model's readings behave like an HMM filter with that table, which has not
  been shown.
- The tool's settled labels are back to Viterbi over the readings with the declared segment length (the
  constant-hazard transition of HANDOFF question 2). README results restored. The held-out numbers in the
  previous entry stand as an experiment, not as a method.

**2026-11-26 · Emissions derived from the readings, transitions estimated from them**
- Declared in `docs/TEST_SUITE.md` before scoring. Command: `uv run python -I
  experiments/estimated_transitions.py > results/estimated_transitions.md`
- Method (`settle.estimate_transition`, `evidence_from_readings`, `smoothed_estimated`): the readings are
  running estimates P(state | text so far), so they are the forward half of the HMM, not its emissions. The
  transition table is estimated from the readings alone, with no labels, by the filtered form of the
  Baum-Welch transition update repeated to a fixed point. Each token's evidence is reading_t / (reading_{t-1} A).
  The backward pass with the same table completes forward-backward; each token takes its likeliest state.
  Posterior decoding replaced Viterbi here: on one tuning stream Viterbi over the derived evidence scored 0.655
  against 0.934, because Viterbi picks one whole path and the per-token evidence is weak.
- Result, current settled labels → estimated table (start value 0.5 / 0.9 / 0.99 changes nothing):
  - AG News (seed 2): accuracy 0.874 → 0.899, placed within 16 tokens 10 → 12 of 18, false switches 2.4 →
    1.8 per 1k; estimated stay probability per token 0.990
  - IMDB (seed 2): 0.809 → 0.828, 5 → 7 of 15, 2.4 → 2.4; stay 0.992
  - PubMed abstracts 31–60: 0.776 → 0.804, 63 → 68 of 95, 4.9 → 5.4; stay 0.952 (about 21 tokens per
    segment; the true median sentence is 30)
  - WikiSection (seed 2, 10 states): 0.655 → 0.627, 5 → 7 of 11, 5.3 → 9.1; stay 0.947
  - single-state streams (seed 1): 0.468 → 0.376, false switches 7.2 → 10.7; stay 0.911
- Conclusion: with no setting tuned against labels, it beats the current decoding on news topics, review
  sentiment and abstracts. It loses where the table has many entries to estimate from few switches (10
  states) and where readings wander with no real switch (single-state streams): the estimated table follows
  the readings' own movement, so noisy readings yield a table that switches often. Not yet the tool's default.

**2026-11-26 · Autoregressive HMM over the readings (offline)**
- Declared in `docs/TEST_SUITE.md` before scoring. Command: `uv run python -I experiments/ar_hmm.py >
  results/ar_hmm.md`
- Setup: observation = the reading's centred log-ratios; emission o_t = c_k + φ·o_{t−1} + Gaussian noise
  (switching intercept, order 1); c, φ, noise and the transition table learned per stream by EM, no labels.
- Result: worse on every set. False switches rise from 2–7 to 75–250 per 1k tokens; accuracy falls 1–6
  points (AG News seed 2 0.874 → 0.861, PubMed 0.776 → 0.752, single-state streams 0.468 → 0.357). EM learns
  a carry-over φ of 0.8–0.9 and a stay probability of 0.38–0.91 per token.
- Conclusion: with φ near 0.9, the state enters each observation only through the small intercept, so the
  fitted states explain the token-to-token noise of the readings rather than the text's regimes, and they
  switch every few tokens. The switching-intercept form makes the states describe changes, not levels.

**2026-11-26 · State-conditioned next-token emissions: the intended method, first run (tuning streams)**
- Commands: `H2O_LIGHTNING_PATH=<checkout> uv run python -I experiments/stated_next_token.py run --model h2o
  --form instruction` (and `--model qwen`; `--form bracket` for comparison), then `... report` →
  `results/stated_next_token.md`
- Method (`head_of_state.next_token.StatedNextTokenReader`): the owner's design, which the earlier runs did not
  build. The stream sits in the user turn of the model's chat format. At every position, one temporary branch
  per state asks "This part of the text is about <description>. What word comes next?" with the answer opened
  by "Next word:"; the head's probability for the token that actually comes next (with or without its leading
  space) is that token's emission for the state, P(next token | text so far, state). Branches run as one batch
  and are removed, so nothing naming a state stays in the cache. The HMM-like decoder uses these emissions at
  weight 1 with the declared length. The Jev-style readout gives P(state | text so far) instead; the
  2026-11-25 likelihood runs put one state sentence above the text rather than asking at each position.
- Setup: the first seed-0 stream of each list (2,100–2,300 tokens, 3–8 switches), every token.
- Result, H2O-Lightning, token accuracy · switches placed within 16 tokens · false switches per 1k (Jev-style
  readout on the same stream in brackets):
  - News topics: 0.916 · 4/7 · 1.4 (0.887 · 4/7 · 2.9); evidence for the true state over the others 0.75 nats
    per token
  - Disease sections: 0.785 · 2/3 · 2.2 (0.596 · 2/3 · 5.6); 0.85 nats per token
  - Reviews: 0.591 · 2/8 · 1.9 (0.827 · 3/8 · 2.4); 0.11 nats per token, and each state's question shifts the
    probability of all text by its own amount, as large as the signal
- Qwen3.5-4B with the same wording: 0.858 / 0.520 / 0.523, per-token evidence about twice as noisy. The
  bracket wording " [name: description]" on plain text (H2O-Lightning): 0.922 / 0.169 / 0.713, with the wrong
  states winning on disease sections.
- Weight 1 is not exact (statistics by a second session): a single token's evidence is 2–2.5 times too weak on
  news and disease sections, and sums over 16 or more tokens are 2–4 times too strong, because the model's
  errors carry over between tokens (lag-1 autocorrelation 0.25–0.36).
- Cost: 104–279 ms per token (H2O-Lightning; ten states with long descriptions cost the most).
- Conclusion: on one stream per list, the intended method beats the question readout on news topics and disease
  sections and fails on review sentiment, which barely changes the next word. Too few switches to establish a
  difference; the remaining tuning streams are next.

**2026-11-26 · State-conditioned next-token emissions on all three tuning streams (news topics, disease sections)**
- Command: `H2O_LIGHTNING_PATH=<checkout> uv run python -I experiments/stated_next_token.py run --model h2o
  --form instruction --streams rest ag_news wikisection`, then `... report` → `results/stated_next_token.md`
- Setup: as the previous entry, the other two seed-0 streams of each list; reviews left out (the owner: sentiment
  is not a fair hidden-state test). The frame is now "This part of the text is {description}. What word comes
  next?", with "about " in these descriptions, so the wording of these runs is unchanged.
- Result, token accuracy of settled labels pooled over the three streams (Jev-style readout in brackets):
  - News topics: 0.929 (0.865); per stream 0.916, 0.916, 0.955 (0.887, 0.835, 0.873); switches placed within 16
    tokens 18 of 21 (15).
  - Disease sections: 0.571 (0.683); per stream 0.785, 0.305, 0.596 (0.596, 0.816, 0.647). In the 0.305 stream
    every symptom token is labelled mechanism, a prevention section splits between cause, prevention and
    treatment, and half of a prognosis section is labelled mechanism.
- Conclusion: better than the question readout on news topics on every stream; worse on disease sections, where
  the first stream's gain did not hold. Related section types change the next word in overlapping ways.

**2026-11-26 · Decoder with an estimated reading weight, re-run; restart-at-the-switch upper bound (question readout)**
- Commands: `uv run python -I experiments/tied_target_hmm.py > results/tied_target_hmm.md`; `uv run python -I
  experiments/joined_inertia.py run --seed 2` (GPU), then `uv run python -I experiments/restart_bound.py [--seed 2]`
  → `results/restart_bound[_seed2].md`
- Both work on the Jev-style readout, before the owner restated the method (next entries).
- The tied-target decoder (readings as observations, reading weight from the stream's own spread and
  autocorrelation, η 0.07–0.41) reproduces a sub-agent's numbers exactly. Paired by segment (statistics by a
  second session): false switches fall by 1.6 per 1,000 tokens [1.0, 2.2]; token accuracy +1.9 points [−0.7, +4.2]
  over four confirmation sets, not established. It is the current decoder at a higher switch cost chosen per
  stream, and lies on the current decoder's own trade-off curve.
- Restarting the reading at the true switch (the first 128 tokens of each new segment read with nothing before
  them), an upper bound: seed 0 news 0.865 → 0.920, disease sections 0.683 → 0.695, reviews 0.889 → 0.978; seed 2
  0.874 → 0.934, 0.655 → 0.688, 0.809 → 0.850. A reader restarted by the live decision was started and stopped
  after 3 of 9 streams when the method was restated; not scored.

**2026-11-26 · Test sets from prior work: Switchboard acts, WikiSection articles, NewsDiscourse, TRIPOD**
- Chosen by a second session the owner delegated the choice to: states published as holding over a stretch and
  switching, from named classes; access requirements reported, not used to exclude. LinCE (word-level language)
  waits for the owner to obtain it: its Hugging Face copy is a loader that fetches the data through another
  account's access link, which is not used.
- Loaders: `experiments/swda.py` (Switchboard dialogue acts, GPL-2.0; acts collapsed as the corpus code does; 7
  acts with at least 2% of training utterances plus other; speaker letter only at a change of speaker),
  `experiments/tripod.py` (TRIPOD synopses; six stages between the five turning points; a turning-point sentence
  opens the stage after it, decided before any run), `experiments/newsdiscourse.py` (9 sentence roles), and
  WikiSection whole articles with the 10 merged groups. Downloads in `experiments/data/download.sh`.
- Declared lengths, measured on training data before any reading (`experiments/stated_sets.py measure`): articles
  184 tokens, acts 9, story stages 124, news roles 42. Protocol: a tuning sample for the go / no-go checks, then
  12 unread documents per set for confirmation.

**2026-11-26 · Next-token emissions on the four tuning samples; the catalogue form**
- Commands: `H2O_LIGHTNING_PATH=<checkout> uv run python -I experiments/stated_sets.py run --set <set> --sample
  tuning [--form catalogue]`, then `... report --sample tuning` → `results/stated_sets_tuning.md`
- Instruction form (each branch "This part of the text is <description>. What word comes next?"), token accuracy
  of settled labels pooled over the tuning documents (always the most common state in brackets):
  - WikiSection articles (3): 0.587 (0.228); evidence for the true state 0.68 nats per token. Go.
  - Switchboard acts (2 conversations): 0.119 (0.639); opinion takes 83% of labels. No-go.
  - NewsDiscourse (3 articles): 0.059 (0.478); main event takes 84% of labels. No-go.
  - TRIPOD (3 synopses): 0.215 (0.231); final push takes 89% of labels. No-go.
  - Evidence per token splits by kind of state: topics and sections 0.62–0.74 nats; sentiment, dialogue acts,
    sentence roles and story stages 0.03–0.11.
- Cause of the takeovers: a description tells the model what kind of text this is as well as which state, so one
  state's description raises the probability of all text, and that differs between states. Catalogue form
  (`head_of_state.next_token`, form "catalogue"): every state's name and description listed once at the start of
  the cache, each branch naming only its state; the list names no state, so emissions still depend only on the
  current state and the text. Branches are 26–28 tokens, cost 170–213 ms per token against 279.
  - Switchboard: 0.119 → 0.587, +46.8 points [+32.2, +58.9] by act run; still below always-statement (0.639).
  - WikiSection articles: 0.587 → 0.674 [+8.7 points, −2.6 to +19.1], not established on 3 articles; symptom's
    share of labels 44% → 26% (true 18%).
  - NewsDiscourse: 0.059 → 0.241; no state takes over, still far below 0.478.
  - TRIPOD: first synopsis only so far, 0.175; final push still takes 90%.
- A normalisation by each state's expected log-probability under the plain prediction (the Fast-DetectGPT
  statistic) removed the lift but more of the signal; accuracy fell on both sets tried. Withdrawn.
- The owner: "Catalogue is a clear improvement yup".
- Conclusion: the catalogue form removes most of the takeover by one state. On states about what the text is
  doing, the per-token evidence stays small. The question readout on the same tuning documents (queued) will show
  whether that is the next-token form or what the model knows.

**2026-11-26 · News confirmation declared (not yet run)**
- `experiments/news_confirm.py`: 12 new news streams of about 1,000 tokens (13,088 tokens, 36 switches), seed 3,
  from AG News test documents in neither the seed-0 nor the seed-2 streams. Three readouts on the same streams:
  the question readout, the next-token instruction form and the catalogue form. Declared before any data: primary
  comparison catalogue against the question readout; secondary, instruction against the question readout and
  catalogue against instruction; paired over the 12 streams, exact sign-flip test, interval by resampling streams.
  The catalogue form is checked on the three seed-0 news streams first.
