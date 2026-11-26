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

Frozen settings (H3), fixed 2026-11-25 before the first scored runs, from the PubMed dev and WikiSection validation runs in docs/LOG.md: 1 option ordering, a reading every 8th token, persist 1, prior-power 0, max_ctx 8192, keep_ctx 2048, forward filter with eta = 1 (persist 1 reported alongside). Question frame unchanged.

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
