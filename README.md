# Head of State

Zero-shot regime tracking from an open-weights LLM's output head.
Hidden-state inference from plain-English descriptions.

Head of State labels a text stream with hidden states (regimes) defined in plain English at run time.
For every token it reports a probability for each state and the likeliest state, and it marks where the
state changes. Nothing is trained: the language model's weights are not changed and no classifier is fitted.

A state is a name and a one-line description. The built-in demo, a support call, uses five:

```
greeting         opening pleasantries and identifying the caller
problem          the customer describes what is wrong
troubleshooting  the agent gives steps to try and the customer tries them
billing          discussion of charges, refunds or payment
closing          wrapping up and saying goodbye
```

A hidden state here is the condition the text is written under: the topic being discussed, the section of a
document, the phase of a call. The words come out differently because of it, and that difference is what
the method reads.

## How it works

The text passes once through an open-weights language model and stays in its key/value cache. The design
calls for each token's emission to be P(next token | text so far, state k): the model's prediction of the
text's own continuation, once for each state. An HMM-like decoder combines the K emissions with sticky
transitions set from a declared typical segment length:
- the live output is the likeliest state of a forward pass, available as each token arrives;
- the settled labels are the Viterbi path over all tokens, so a switch is placed where the new state's
  tokens begin to favour it rather than where the live output noticed it.

If each emission is the model's prediction of the text given the state, the product of emissions over
tokens is the likelihood of the text under a sequence of states, as in a Markov-switching model (Hamilton,
"Regime switching models", eq. 7).

**What the tool reads today.** All states' names and descriptions are listed at the start of a chat turn
that holds the stream. At every token, one temporary branch per state closes that turn with "This part of the
text is <state>. What word comes next?", and the head is read at the start of the reply. That reading is the
start of an answer, not the text's continuation: the actual next token's median rank there is 229 before a
word and 54,680 inside one, against 16 and 1 for the model's plain prediction. Its differences between states
come from how strongly each token goes with each named state. It is the strongest readout found so far.

**Forms that do predict the text** (`docs/LOG.md`): one cache per state with the state stated before the
stream, and a thinking block stating the state before the next token with the last tokens repeated. Both
predict the text, but the stated state changes the prediction by only 0.002–0.12 nats per token: once the
text so far shows what it is about, being told the state adds little. The emission the design calls for has
not been obtained yet.

The default model is H2O-Lightning-4B, an open-weights Qwen3.5-4B fine-tuned to answer choice questions
about a record.

**Status: early prototype, with no settled use case yet.** Descriptions of problems that resemble this one
are welcome in the issue tracker.

## Results so far

Streams of labelled documents joined end to end with single spaces (the evaluation in `docs/HANDOFF.md`),
so every token has a known state and nothing in the formatting marks a switch; and whole documents whose
sections carry labels. Every token is read, by H2O-Lightning-4B on one 12 GB GPU. Labels are judged after
Viterbi has settled them. Time to notice is how many tokens after a true switch the live output first shows
the new state; a stream read left to right cannot know sooner, so it is reported and not counted as error.

All of the following is tuning data, read with the reply-slot readout described above: on the news and
joined-section streams with each state's description in its own question, on the whole articles with the
descriptions listed at the start. A confirmation on 12 unread news streams is on hold until an emission is
chosen.

| States (documents) | Token accuracy | Always the most common state | Switches placed within 16 tokens |
|---|---|---|---|
| News topics, 4 states (AG News articles, 3 streams) | 0.93 | 0.34 | 18 of 21 |
| Section types, 10 states (3 whole WikiSection disease articles) | 0.67 | 0.23 | 9 of 13 |
| Section types, 10 states (WikiSection disease sections joined, 3 streams) | 0.57 | 0.27 | 6 of 8 |

On news topics, settled switches land a median 3 tokens after the true switch, and the live output shows a
new topic 18–43 tokens after it, by stream.

Known limits:
- **Related states are confused.** Section types that change the words in overlapping ways are read as one
  another, a whole section at a time: in one stream 98% of the symptom tokens are labelled mechanism.
- **Labels the text does not depend on are not recovered.** Dialogue acts, news sentence roles and story
  stages were tried: they describe how finished text functions, a state's prediction barely changes the next
  word, and labels stayed at or below always the most common state. They are not hidden states in the sense
  above.
- **The decoder's weight is not calibrated.** A single token's evidence is 1.8 to 2.8 times too weak, while
  sums over 16 tokens are 2 to 3 times too strong and over 64 tokens 6 to 10 times, because the model's
  errors carry over from token to token.
- **Cost.** One branch per state at every token: about 135 ms per token with 4 states and 210 ms with 10, on
  an RTX 3060.

Every run's settings and results, including the readouts tried before this one, are in `docs/LOG.md`, with
an audit of the scoring and its corrections.

## Run the demo

```
uv sync
uv run head-of-state --demo                  # downloads H2O-Lightning-4B (9.3 GB) on first use
uv run python tests/test_next_token.py       # mechanics tests on a tiny random model
```

A run prints each token's live state and its probability, then the settled segments. Own states and text:
`--states states.json` (`{"name": "description", ...}`), `--text file.txt` and `--segment-length <tokens>`;
`--save out.json` writes the emissions and labels. Another model or a local checkout: `--model <id or path>`
(it needs a chat template). The evaluation runs: `experiments/stated_next_token.py` and
`experiments/stated_sets.py` (commands in their headers).

Method and decoder notes: `docs/method.md`. Experiment log: `docs/LOG.md`; planned experiments:
`docs/EXPERIMENTS.md`; evaluation plan: `docs/TEST_SUITE.md`; prior work: `docs/research/prior-art.md`;
original brief: `docs/HANDOFF.md`. Datasets for the experiments: `experiments/data/download.sh`.

Licensed under the Apache License 2.0 (`LICENSE`).

<p align="center"><img src="docs/assets/logo/head-of-state-logo.svg" width="96" alt="Head of State logo"></p>
