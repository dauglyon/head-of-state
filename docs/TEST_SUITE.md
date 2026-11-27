# Test suite for zero-shot stream state labelling — v4 (final)

Tool: this repository (docs/HANDOFF.md, src/head_of_state/reader.py). Model: Qwen3.5-4B, max_ctx 8192, keep_ctx 2048.
Reviewed in three rounds before adoption; round 3 verdict: ready to build.
Owner decisions: D1, typical segment length declared per stream; D2, unit-to-judge wording supplied per stream.

## Build order
0. HANDOFF tasks 1-4: the tool has never run on a real model.
1. S3 PubMed RCT: no offsets, no long streams; exposes prompt and letter-token faults in minutes.
2. S1 + S2 WikiSection controlled and null streams.
3. S8 notebooks, S5 ACI-Bench, S6 Supreme Court, then the rest.

## Harness (all sets)
- H1 Offset-based gold. Tokenise each whole stream once with `return_offsets_mapping`; assign gold from spans. WikiSection offsets are characters; PDNC and `ast` offsets are bytes. Any text edit (S11 quote stripping, H2 tag removal) is applied before spans are computed. `reader.build_stream` is not used: it strips segments and joins with one space.
- H2 Dialogue formatting. Speaker tags removed; turns separated by a single newline in every dialogue set.
- H3 Settings.
  - Frozen across all sets: stride, rotations, persist, prior-power, max_ctx, keep_ctx, η, and the fixed question frame.
  - Per-stream inputs, and only these: state descriptions, declared segment length (D1), unit wording (D2).
  - Declared length: one value per set, fixed in the table below before any run, from dataset documentation or a non-evaluated split; never from the evaluated stream's own gold.
  - η is swept on tuning data only: PubMed 20k dev, and an S1 slice built from articles not used in reported S1 streams.
- H4 Unit wording (D2). Supplied per stream; must not name or imply any state; must be no longer than the set's typical shortest segment. Implemented by overriding `_question` in a subclass; the default reproduces the current `reader.QUESTION_HEAD`, so `tests/test_reader.py` still passes.
- H5 Seam control. Log actual re-anchor positions (subclass wrapping `_reanchor`). Run each S1 seam stream twice, the second time with the first N tokens of the opening segment dropped, so the same switch is scored near and far from a seam.
- H6 Save every reading's probabilities; run raw argmax, persist, and the sticky forward filter offline on the same readings (HANDOFF Q2).
- H7 Scorer fix. `score()` counts a true switch as found when the committed state equals the new gold state anywhere in the segment, with no commit event. On gold A-B-A a tool that never leaves A scores one switch found with delay 0, and that credit cancels a real false switch (verified with a scratch script). Fix: match each commit event to a true switch (same target state, inside that segment); unmatched commits are spurious; delay only from matched commits. Applied 2026-11-25; regression cases in `tests/test_reader.py` (5b).
- H8 Measure tokens/s on the first stream; size every set from it.

## Pre-registration (fixed before any run)
Lengths marked "measure" are taken from the named non-evaluated split before the first scored run and written here; they are not changed afterwards.

| Set | Declared segment length | Unit wording |
|---|---|---|
| S1, S2, S4 | 190.5 tokens (median merged-section length, WikiSection en_disease validation split as the S1 tuning articles, 260 documents, 1,238 segments, Qwen3.5-4B tokenizer; `eval/wikisection.py measure`, 2026-11-25) (one value for all S1 length conditions and S2) | "the last few sentences" |
| S3 | 30 tokens (median sentence length, PubMed 20k RCT dev, 30,212 sentences, Qwen3.5-4B tokenizer; `eval/s3_pubmed.py measure`, 2026-11-25) | "the last sentence" |
| S5 | 12 tokens (median turn length, ACI-Bench train, 3,735 turns, H2O-Lightning-4B tokenizer; 2026-11-25) | "the last turn" |
| S6 | measure: median turn length, one held-out year | "the last turn" |
| S7 | measure: median message length, oasst1 validation | "the last message" |
| S8 | measure: median cell length, held-out notebooks | "the last few lines" |
| S9 | measure: median span length, held-out files | "the last few lines" |
| S10 | measure: median solution length, held-out tasks | "the last few lines" |
| S11 | measure: median narration/quote span, one held-out novel | "the last sentence" |
| S12 | measure: median human-prefix length, SemEval 8C train | "the last few sentences" |

Frozen settings (H3), revised 2026-11-25: a reading at every token, 1 option ordering, persist 3, prior-power 0, max_ctx 8192, keep_ctx 2048, forward filter with eta = 1 (persist 3 reported alongside); model H2O-Lightning-4B. Question frame unchanged. Reading every 8th token (stride 8, persist 1) was the earlier frozen setting; it is kept as an optional experiment (`--stride`, `--refine`), and its scored results are in docs/LOG.md.

Held-out samples (designated 2026-11-25, before their runs): the default model (H2O-Lightning-4B) and the revised frozen settings were chosen after the first test samples (PubMed test abstracts 1-30, the first 10 two-speaker ACI-Bench encounters, the seed-0 null streams) had been seen, so those samples are no longer held out for them. Confirmation uses new samples: PubMed 20k RCT test abstracts 31-60 (seed-0 order), ACI-Bench clinicalnlp_taskB_test1 two-speaker encounters 11-20, and 2 null streams built with seed 1 (`experiments/scored_heldout.py`). Headline decoder: forward filter with eta = 1 at the declared length; persist 3 and the filter at 0.25x and 4x the declared length reported alongside.

Judging (revised 2026-11-25, before the held-out samples are scored): the output is judged on its settled labels, the best path from Viterbi over all readings of the stream, with each reading treated as evidence for the state d tokens earlier, d = round(0.2 x declared segment length) (PubMed 6, ACI-Bench 2, null streams 38); token accuracy, boundary F1 within 8 and 16 tokens and spurious switches come from those labels. Time to notice, the tokens from a true switch to the first reading at which the online decision (max-product forward pass, no lookahead) shows the new state, is reported alongside and is not counted as error: a stream read left to right cannot know what comes next.

Judging revised 2026-11-26 (after the joined-document runs): settled labels are the Viterbi path over the readings where they were taken, with no shift. Moving readings 0.2 x the declared length earlier placed switches about 48 tokens early on segments of about 270 tokens, because how far a reading runs behind the text depends on how much new text the model needs, not on the segment length. Confirmed on new streams (seed 2) before adoption; results judged under the earlier rule are kept as they were reported.

Targeted question, held-out check (declared 2026-11-26, before the run): ACI-Bench clinicalnlp_taskB_test1 two-speaker encounters 21-30, the current question against R2 (the current line as its own part of the record, a direct question, neutral option names), both read every 4th token by H2O-Lightning-4B and judged on settled labels (`experiments/ask_heldout.py`).

Conversations without line breaks (declared 2026-11-26, before the run): in the ACI-Bench streams each turn is one line, so every switch falls on a line break and the task reduces to classifying lines. To test segmentation, ACI-Bench clinicalnlp_taskB_test1 two-speaker encounters 31-38 are joined into continuous text (turns separated by one space) and read with the current question; the same encounters with line breaks are the comparison. H2O-Lightning-4B, a reading every 4th token, settled labels (`experiments/aci_unsegmented.py`).

Joined labelled documents, HANDOFF task 5 (declared 2026-11-26, before the run): three unrelated state lists, AG News test (world / sports / business / science and technology), WikiSection en_disease train sections (the merged labels) and IMDB test reviews of at most 300 tokens (positive / negative). Segments of one or more same-label documents, sizes drawn from 64/128/256/512 tokens, consecutive segments in different states, joined with single spaces; 3 streams of about 2,000 tokens per list, seed 0. Declared segment length: the median segment length of 10 streams built with seed 1 (AG News 270, WikiSection 441, IMDB 274 tokens). Unit: "the last few sentences". Frozen settings (every token, 1 ordering), settled labels with time to notice, reported beside the accuracy of always answering the most common state (`experiments/joined_documents.py`).

Joined labelled documents, confirmation (declared 2026-11-26, before the run): the same three lists and construction, 3 new streams per list built with seed 2 (documents not used in the seed-0 streams), judged on settled labels with no reading shift (readings used where they were taken), the rule chosen after the seed-0 streams; the pre-registered 0.2 x length shift is reported alongside (`experiments/joined_documents.py run --seed 2`).

Joined labelled documents, targeted question (declared 2026-11-26, before the run): tuning comparison on the seed-0 streams, the current question against the current sentence as its own part of the record with a direct question and neutral option names, every token, settled labels without shift. If it is better on the tuning streams, it is confirmed on 3 new streams per list (seed 3, documents not in the seed-0 streams) before any change of default.

Reading weight eta (declared 2026-11-26, before scoring the confirmation streams with it): each reading's probabilities raised to eta = 1/8 before decoding (the per-frame scale used for correlated frame posteriors in hybrid neural-network/HMM speech recognisers), chosen on the seed-0 joined-document streams from 1, 1/2, 1/4, 1/8, 1/16. Checked on the saved seed-2 readings, every token, settled labels and time to notice, against eta = 1 (`experiments/joined_eta.py`).

Evidence recovered from readings (declared 2026-11-26, before scoring the confirmation streams with it): each reading is a running estimate given all text so far, so the evidence of the tokens since the previous reading is recovered as reading_t / (reading_{t-1} A_m), where A_m is a sticky chain with switch probability 1/20 per token (the readings' own rate of change, one value for the model, chosen from 10 / 20 / 50 on the seed-0 joined-document streams); Viterbi then decodes that evidence with the declared segment length (`head_of_state.settle.reading_evidence`). Scored on the saved seed-2 joined-document readings and the held-out PubMed abstracts 31-60 and null streams (seed 1), against the current settled labels (`experiments/joined_evidence.py`).

Emissions derived from the readings, transitions estimated from them (declared 2026-11-26, before scoring the sets below with it): readings are running estimates P(state | text so far); the transition table is estimated from the readings alone by the filtered form of the Baum-Welch transition update, started at stay 0.9; the evidence of each token is reading_t / (reading_{t-1} A); forward-backward gives each token's probabilities and the likeliest state is the label (`head_of_state.settle.smoothed_estimated`). No setting chosen against labels. Scored on the seed-2 joined-document streams, held-out PubMed abstracts 31-60 and the seed-1 single-state streams, against the current settled labels; the start value 0.5 / 0.9 / 0.99 reported alongside (`experiments/estimated_transitions.py`).

Autoregressive HMM over the readings (declared 2026-11-26, before scoring): observation o_t = centred log-ratios of the reading (log r_t minus its mean over options); emission o_t = c_k + phi * o_{t-1} + Gaussian noise (switching intercept, autoregression of order 1, phi and the noise variance per coordinate, shared by states); transition table, c, phi and noise learned per stream by EM with no labels; c_k started at the log-ratios of a reading of 0.9 on option k, the transition table started from the declared segment length; settled labels = likeliest state per token from forward-backward. Scored on the seed-0 and seed-2 joined-document streams, held-out PubMed abstracts 31-60 and the seed-1 single-state streams, against the current settled labels (`experiments/ar_hmm.py`).

Windowed likelihood emissions (HANDOFF Q3 extended, added 2026-11-25): W = 0.0625 x declared segment length (S3: 2 tokens, S1: 12 tokens), hop W/2, sticky forward filter with eta = 1 and switch probability 1/L. Chosen before scored runs by mean committed accuracy over the tuning sets (first 10 PubMed 20k RCT dev abstracts; 2 WikiSection en_disease validation streams), from 0.0625 / 0.125 / 0.25 / 0.5 / 1 / 2 x and no window (`eval/exp_window.py`). The same W applies to the windowed question method (experiment E).

## Sets

### Lag, seam, null rate (HANDOFF Q1, Q3; task 6)
S1 WikiSection en_disease controlled concatenations (CC BY-SA 3.0, 3,590 articles; `text` excludes headings).
- Labels merged into 10 groups: symptom; cause (cause, risk, genetics, infection); diagnosis (diagnosis, tomography, classification); prevention (prevention, screening); treatment (treatment, management, medication, surgery); mechanism (mechanism, pathology, pathophysiology); epidemiology (epidemiology, geography); history (history, etymology); prognosis (prognosis, complication); society and research (culture, research). Documents with fauna or other are dropped. Count class frequencies before freezing the groups.
- Adjacent sections with the same merged label are fused in the gold.
- Previous-segment length 50 / 200 / 800 / 3000 tokens, topic pair fixed. 3000-token segments join same-label sections from several articles (a disease change inside a segment); same join pattern across conditions. After the first re-anchor the model sees at most 2048 stream tokens, so the 3000 condition exists only before the first seam.
- Seam streams 15k+ tokens, switches from 500 tokens before to 500 after a re-anchor point (with H5).
- Also used for: HANDOFF Q3 (log-likelihood gap vs distance from prefix); knob sweep (task 6, with S3 dev); K sweep on a fixed stream with unused in-domain labels as distractors (3, 6, 10), plus 26 reported separately as "with out-of-domain distractors" (en_city labels).
S2 Null streams: long single-label streams with S1's declared length; false switches per 1k tokens.

### Decoder comparison on natural structure (HANDOFF Q2)
S3 PubMed 20k RCT, 5 labels, short ordered segments. Dev split is tuning data. Underlying abstracts carry no stated licence; research use. Knob sweep also runs on the dev split.
S4 WikiSection en_disease natural documents, S1's label mapping.

### Speakers / roles
S5 ACI-Bench doctor/patient (CC BY 4.0): virtassist, `*_virtscribe_humantrans`, `*_aci_asrcorr`, up to 207 encounters (about 52-58 turns and about 1,300 tokens each; never reaches the seam). Spot-check `asrcorr` for swapped tags.
S6 Supreme Court oral arguments (ConvoKit `supreme-YYYY`, source Oyez): justice / petitioner's advocate / respondent's advocate. Corpus licence not stated; confirm before publishing. Token-count one transcript to confirm seam crossing.
S7 OpenAssistant oasst1 (Apache-2.0): user vs assistant, many short streams. Last in order.

### Code
S8 codeparrot/github-jupyter-parsed (59k notebooks, 295 MB, per-row repo licence): explanatory prose vs Python code by cell. Check one row for cell order.
S9 Python files, spans from `ast` + `tokenize`: docstring/comment / imports+signature / body / test assertions. Syntactic ceiling.
S10 Rosetta Code (GFDL): one task in several languages concatenated; states = languages. Sanity case.

### Dominant state
S11 PDNC novels (CC BY 4.0): narration vs quoted speech. Quote marks stripped before spans are computed. Crosses the seam many times.

### Negative control
S12 SemEval-2024 8C small sample: expected near chance. Licence unverified (task repo returned HTTP 451).

## Out of scope
HANDOFF Q4 (detector directions) needs Llama-3.1-8B neuron descriptions.
Dropped after review: AMI roles, Friends, Code4ML, DistilKaggle, legal rhetorical roles, MultiWOZ, LinCE.

## Report per set
- Accuracy raw and committed; matched switches / true switches; mean delay over matched switches; spurious commits per 1k tokens; tokens/s.
- The set's declared length and unit wording.
- Forward-filter results at the declared length ×0.25 and ×4 (sensitivity to D1).
- S1: delay broken out by previous-segment length.
- Optional: one common neutral wording on a small subsample of each set, to separate wording from dataset.

Combined readouts, confirmation (declared 2026-11-27, before any of these documents was read): the 12 seed-3 news
streams of `experiments/news_confirm.py` and PubMed 20k RCT test abstracts 61-100 (seed-0 order; 1-60 used before).
H2O-Lightning-4B; the question readout with the frozen settings (unit "the last few sentences", "the last sentence" for
PubMed) and one cache per state (the owner's opening). Decodes, Viterbi with the declared length (news 270, PubMed 30):
(1) the question readout's log readings; (2) the same counted fractionally, the weight from the stream's own
autocorrelation (`tied_target_hmm.fit`); (3) (2) plus the cache log-likelihoods. Comparisons: (a) 2 against 1, (b) 3
against 2; token accuracy paired by document and by gold segment, false switches per 1,000 tokens, switches placed
within 16 tokens (`experiments/confirm_combined.py`; a second session fixes its analysis script before the run).
