# Head of State

Zero-shot regime tracking from an open-weights LLM's output head.
Hidden-state inference from plain-English descriptions.

Head of State labels a text stream with hidden states (regimes) defined in plain English at run time.
For every token it reports which state the text is in, and it marks where the state changes. Nothing is
trained: the language model's weights are not changed and no classifier is fitted.

A state is a name and a one-line description. The built-in demo, a support call, uses five:

```
greeting         opening pleasantries and identifying the caller
problem          the customer describes what is wrong
troubleshooting  the agent gives steps to try and the customer tries them
billing          discussion of charges, refunds or payment
closing          wrapping up and saying goodbye
```

The text passes once through an open-weights language model and stays in its key/value cache. At each
reading point it takes a Jev-style readout: a short multiple-choice question about the most recent text
is appended, the probabilities the model's output head gives the answer options are the reading, and the
question is then removed from the cache. An HMM-like decoder turns the readings into segments: once a
switch has been noticed, Viterbi decoding corrects the labels of the tokens before it, so the settled
segments start where the new state began rather than where it was noticed. The default model is
H2O-Lightning-4B, an open-weights Qwen3.5-4B fine-tuned to answer choice questions about a record; other
open-weights causal language models work through `--model`.

**Status: early prototype, with no settled use case yet.** Descriptions of problems that resemble this
one are welcome in the issue tracker.

## Results so far

Held-out samples, read at every token by H2O-Lightning-4B on one 12 GB GPU (about 100 ms per token). The
model, settings and judging rule were fixed before these samples were run (`docs/TEST_SUITE.md`). Labels are
judged after the decoder has corrected them (settled labels). Time to notice is how many tokens after a true
switch the live output first shows the new state; a stream read left to right cannot know sooner, so it is
reported and not counted as error.

| Data (held-out sample) | Token accuracy | Switches placed within 16 tokens | False switches per 1,000 tokens | Time to notice |
|---|---|---|---|---|
| PubMed 20k RCT abstracts, test (30 abstracts, 5 states) | 0.79 | 63 of 95 | 4.5 | 23 tokens |
| Doctor–patient conversations, ACI-Bench test (10 encounters, 2 speaker states) | 0.87 | 199 of 476 | 0.7 | 5 tokens |
| Single-state streams, WikiSection disease articles (2 streams, 6,550 tokens) | 0.47 | — | 7.0 | — |

Known limits:
- **Short segments are missed.** Most conversational turns are short (median 12 tokens), and the turn often
  ends before the model can tell it apart; 58% of turn changes are missed. Asking about the current line as its
  own part of the record (`--target line`) raised the share placed within 16 tokens from 33% to 67% on 10 fresh
  conversations, read every 4th token.
- **Related states are confused.** In the single-state streams, "symptom" and "cause" sections are often read
  as epidemiology or mechanism, so the true state holds for under half of the tokens.
- **Time to notice depends on the text.** Speaker changes show within about 5 tokens; a new section of an
  abstract takes about 23 tokens, most of a sentence.
- **Cost.** About 100 ms per token when reading every token. Reading every 8th token (`--stride 8 --persist 1`)
  is about 7 times cheaper; its labels are assigned to the 8 tokens before each reading.

Earlier exploratory, tuning and comparison runs (other models, readouts, decoders and settings) are in
`docs/LOG.md`, with an audit of the scoring and its corrections.

## Run the demo

```
uv sync
uv run head-of-state --demo                                       # downloads H2O-Lightning-4B (9.3 GB) on first use
uv run head-of-state --demo --stride 8 --rotations 1 --persist 1   # read every 8th token: about 7x cheaper
uv run python tests/test_reader.py        # mechanics tests on tiny random models
```

A run prints the live readings, then the settled segments. Own states and text: `--states states.json`
(`{"name": "description", ...}`), `--text file.txt` and `--segment-length <tokens>`. Another
model or a local checkout: `--model <id or path>`, e.g. `--model Qwen/Qwen3.5-4B`.

Method and decoder notes: `docs/method.md`. Experiment log: `docs/LOG.md`; planned experiments:
`docs/EXPERIMENTS.md`; evaluation plan: `docs/TEST_SUITE.md`; prior work: `docs/research/prior-art.md`;
original brief: `docs/HANDOFF.md`. Datasets for the experiments: `experiments/data/download.sh`.

Licensed under the Apache License 2.0 (`LICENSE`).

<p align="center"><img src="docs/assets/logo/head-of-state-logo.svg" width="96" alt="Head of State logo"></p>
