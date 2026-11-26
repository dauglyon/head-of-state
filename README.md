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

A Jev-style reading gives the probability of each state given the text so far. The method's intended
emission is a different quantity: the probability the head gives the token that actually comes next, once
for each state. At every token, one temporary branch per state asks "This part of the text is <state>. What
word comes next?", with every state's name and description listed once at the start of the text; the K
probabilities of the actual next token are that token's emissions for the decoder. This is implemented in
`head_of_state.next_token` and is under evaluation; the command-line tool still uses the Jev-style readout.

**Status: early prototype, with no settled use case yet.** Descriptions of problems that resemble this
one are welcome in the issue tracker.

## Results so far

Streams built by joining labelled documents end to end with single spaces (the evaluation in
`docs/HANDOFF.md`), so every token has a known state and nothing in the formatting marks a switch. Three
unrelated state lists, 3 streams of about 2,000 tokens each, built from documents not used while the
judging rule was chosen. Read at every token by H2O-Lightning-4B on one 12 GB GPU. Labels are judged after
Viterbi has corrected them (settled labels). Time to notice is how many tokens after a true switch the live
output first shows the new state; a stream read left to right cannot know sooner, so it is reported and
not counted as error.

| State list (documents) | Token accuracy | Always the most common state | Switches placed within 16 tokens | False switches per 1,000 tokens | Time to notice |
|---|---|---|---|---|---|
| News topics, 4 states (AG News articles) | 0.87 | 0.34 | 10 of 18 | 2.4 | 30 tokens |
| Section types, 10 states (WikiSection disease article sections) | 0.66 | 0.30 | 5 of 11 | 5.3 | 92 tokens |
| Sentiment, 2 states (IMDB reviews) | 0.81 | 0.67 | 5 of 15 | 2.4 | 55 tokens |

On natural documents, 30 held-out PubMed abstracts with 5 section states, token accuracy is 0.78 against
0.41 for always the most common state. On streams that stay in one state the whole time, 7 false switches
per 1,000 tokens.

Known limits:
- **Switches are placed late.** Viterbi moves a switch back only as far as the readings allow, and the
  readings name the new state only once its text shows it: a median 17 tokens late for news topics, 11 for
  reviews and 79 for disease-article sections.
- **The earlier text holds the readings back.** In the streams used to choose the judging rule, the first
  8 tokens of a new news article, read alone, give its topic 0.76 probability; read after the previous
  article, 0.46. Reviews: 0.65 against 0.34.
- **Related states are confused.** Disease-article section types are the weakest list (0.66): a third of
  the tokens in symptom sections are labelled prognosis.
- **Cost.** About 100 ms per token when reading every token (150 ms with 10 states). Reading every 8th
  token (`--stride 8 --persist 1`) is about 7 times cheaper; its labels are assigned to the 8 tokens before
  each reading.

Next-token emissions, tuning stage (not yet confirmed on unread data):
- **News topics:** 0.929 token accuracy against 0.865 for the Jev-style readout on the same 3 tuning
  streams; 12 unread streams are next, with the comparison declared beforehand.
- **Section types in whole WikiSection articles** (tuning documents only): 0.67 against 0.23 for always the
  most common state.
- Dialogue acts, news sentence roles and story stages were tried and dropped: they label how finished text
  functions, the text barely depends on them, and they are not hidden states in this sense (`docs/LOG.md`).

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
