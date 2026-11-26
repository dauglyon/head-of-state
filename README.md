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
question is then removed from the cache. An HMM-like filter (persist rule, sticky forward filter or
fixed-lag Viterbi) turns the readings into segments. The default model is H2O-Lightning-4B, an
open-weights Qwen3.5-4B fine-tuned to answer choice questions about a record; other open-weights causal
language models work through `--model`.

**Status: early prototype, with no settled use case yet.** Descriptions of problems that resemble this
one are welcome in the issue tracker.

## Results so far

Preliminary, from small samples, on one 12 GB GPU, with Qwen3.5-4B unless the row names H2O-Lightning-4B. The first three rows are scored runs with
settings fixed beforehand; the others are tuning or exploration runs. Every run is in `docs/LOG.md`.

| Data (sample) | Result |
|---|---|
| PubMed 20k RCT abstracts, test (30 abstracts, 98 switches, 5 states), frozen settings: every 8th token | token accuracy 0.70; 83 of 98 switches found, 27 tokens late on average; 8.2 spurious switches per 1,000 tokens; 14 ms per token |
| Same PubMed test abstracts, read through H2O-Lightning-4B (a decision-tuned Qwen3.5-4B) | token accuracy 0.76; 85 of 98 switches found, 19 tokens late on average; 7.2 spurious switches per 1,000 tokens; 13 ms per token |
| Single-state streams, WikiSection disease articles, test (4 streams, 15,039 tokens) | 5.3 false switches per 1,000 tokens with the forward filter; the true state held for 48–95% of tokens, depending on the state |
| Support-call demo (245 tokens, 4 switches), either model | token accuracy 0.84; 4 of 4 switches found, 10 tokens late on average; no spurious switches |
| PubMed 20k RCT abstracts, dev, tuning data (10 abstracts, 36 switches, 5 states) | token accuracy 0.79–0.80; switches committed about 23 tokens late (a median sentence is 30 tokens); 5.8 spurious switches per 1,000 tokens, 1–2 with fixed-lag Viterbi |
| WikiSection disease articles, test (12 streams, one switch each, 10 states offered) | token accuracy 0.88 around the switch; switches about 24 tokens late, not growing over 50 to 3,000 tokens of earlier text |
| WikiSection disease articles, validation (2 streams, 15 switches, 10 states) | token accuracy 0.56–0.58; more than 2 spurious switches per 1,000 tokens with every decoder tried |
| Cost per token | 150 ms reading every token; 14–20 ms reading every 8th token, which matched the every-token accuracy on the demo and PubMed dev |

Known limits: switches are detected late. On PubMed abstracts the readings turn about 16 tokens after
the true switch, and showing the question only the last 30–120 tokens does not shorten this. Many similar
states (ten disease-article section types) are hard. Readouts built on next-token likelihood or on
activation directions have been weaker than the question so far.

## Run the demo

```
uv sync
uv run head-of-state --demo                                       # downloads H2O-Lightning-4B (9.3 GB) on first use
uv run head-of-state --demo --stride 8 --rotations 1 --persist 1   # read every 8th token: about 7x cheaper
uv run python tests/test_reader.py        # mechanics tests on tiny random models
```

Own states and text: `--states states.json` (`{"name": "description", ...}`) and `--text file.txt`. Another
model or a local checkout: `--model <id or path>`, e.g. `--model Qwen/Qwen3.5-4B`.

Method and decoder notes: `docs/method.md`. Experiment log: `docs/LOG.md`; planned experiments:
`docs/EXPERIMENTS.md`; evaluation plan: `docs/TEST_SUITE.md`; prior work: `docs/research/prior-art.md`;
original brief: `docs/HANDOFF.md`. Datasets for the experiments: `experiments/data/download.sh`.

Licensed under the Apache License 2.0 (`LICENSE`).

<p align="center"><img src="docs/assets/logo/head-of-state-logo.svg" width="96" alt="Head of State logo"></p>
