# Method and decoder notes

Terms: a *hidden state* (*regime*) is one of the states described at run time (name plus one-line
description), the condition the text is written under; a *switch* is a change from one hidden state to
another; the *head* is the model's existing output layer. The method is HMM-like: the head supplies the
emissions, a sticky transition model supplies the dynamics, and nothing is trained.

## Emissions: the next token, predicted once per state

`src/head_of_state/next_token.py`, `StatedNextTokenReader` (form `catalogue`, used by the command-line tool).

1. The stream sits in the user turn of the model's chat format, after a list of every state's name and
   description ("The text below is made of parts of these kinds: - name: description ..."). Each stream
   token is fed once and stays in the key/value cache.
2. At every token t, one temporary branch per state closes the user turn with "This part of the text is
   <name>. What word comes next?" and opens the answer with "Next word:". The head's distribution at the end
   of branch k is P(next token | x_1..x_t, state k).
3. The emission of token t+1 for state k is that distribution's probability for the token that actually
   comes next. A chat answer writes a word with or without its leading space, so the probability of the
   token and of its other spacing (when that is one token) are added; the same mapping is used for every
   state. Token 1 has nothing before it and gets a flat emission.
4. The K branches run as one batched pass over a cache widened to one row per branch, and are removed
   afterwards. Attention layers drop the branch's keys and values; Gated DeltaNet layers (Qwen3.5) keep one
   running state that a branch overwrites in place, so those states are copied before and restored after.
   `tests/test_next_token.py` checks that a branch's emission equals a full recompute and that the cache
   holds only the instruction and the stream afterwards.

The emission depends on the current state and the text before the token only. Nothing that names a
particular state stays in the cache (the list of states names all of them), so this is the regime density of
a Markov-switching model (Hamilton, "Regime switching models", eq. 7, econweb.ucsd.edu/~jhamilton/palgrav1.pdf):
the forward filter, smoother and Viterbi apply as printed, and the product of emissions over tokens is the
likelihood of the text under a state path, provided each emission is the model's prediction of the text's own
continuation.

**What this form reads.** The stream sits in a chat turn and the head is read at the start of the reply after
"Next word:", so it predicts the first token of an answer, not the text's continuation (actual next token's
median rank 229 before a word and 54,680 inside one, against 16 and 1 for the plain prediction). Its differences
between states come from how strongly each token goes with each named state. Forms that predict the text (one
cache per state; a thinking block before the next token with the last tokens repeated) give 0.002–0.12 nats per
token of state evidence (`docs/LOG.md`, 2026-11-26).

Why the list of states goes at the start: a description tells the model what kind of text this is as well as
which state. When each branch carried its own description ("This part of the text is <description>. What word
comes next?", form `instruction`), one description raised the probability of all text, and that state took
most of the labels (opinion on Switchboard, symptom on whole WikiSection articles). Listing all descriptions
once gives every branch the same information, so the branches differ only in the state they name; branches are
also shorter (26–28 tokens). A third form, a bare " [name: description]" note on plain text (form `bracket`),
gave less evidence per token than either.

## Decoder

`src/head_of_state/settle.py`, `src/head_of_state/metrics.py`.

- **Transitions**: stay probability 1 − 1/L per token, the rest spread evenly over the other states, where L is
  the declared typical segment length (geometric segment lengths; `reader.sticky_transition`).
- **Live output**: the forward filter's probability for each state at each token (`metrics.forward_posteriors`)
  and the likeliest state of the max-product forward pass (`settle.online_labels`), with no lookahead.
- **Settled labels**: the Viterbi path over all tokens (`settle.settled_labels`). A switch is placed where the
  new state's own tokens begin to favour it.
- **Weight**: emissions enter at weight 1, as the likelihood factorisation implies. Measured on the tuning
  streams (a second session's statistics, `docs/LOG.md`), it is not calibrated: one token's pairwise
  log-ratio is 1.8–2.8 times too weak, and sums over 16 or more tokens are 2–10 times too strong, because the
  model's errors are correlated from token to token (lag-1 autocorrelation of the per-token evidence
  0.2–0.36). No other weight has been chosen; any would need its own justification.

## Judging

Settled labels are scored against the true state of every token: token accuracy, switches placed within 16
tokens, false switches per 1,000 tokens, against always the most common state. Time to notice, the tokens from
a true switch to the first live output showing the new state, is reported separately and not counted as error:
a stream read left to right cannot know sooner. Tuning documents (the ones settings were chosen on) and
confirmation documents (read once, with the comparison declared beforehand) are kept apart (`docs/TEST_SUITE.md`).

## What the emissions can and cannot carry

By Bayes' rule, P(x_t | x_<t, k) = P(k | x_<=t) P(x_t | x_<t) / P(k | x_<t) (Yang & Klein 2021, sec. 3). For an
emission that is a true next-token prediction given the state, the log-ratio between two states summed over a
segment therefore telescopes to the change in the model's own belief about the state across the segment: a few nats
at most (an even split to 0.99 is about 4-5). A switch costs about log(L (K - 1)) under the sticky transitions, 6.7
nats for news segments of 270 tokens. So such emissions give a decoder little reason to switch, and their per-token
evidence is small by construction once the text so far has shown what it is about. Every text-predicting form run so
far behaves this way (0.002-0.12 nats per token), and so does the ratio of consecutive question readings (decoded into
a single state on every set; `docs/LOG.md`, 2026-11-27).

The readouts that decode well (the reply slot, the question readout) are not such conditionals: each restates the
evidence for the state at every token, and the decoder counts it again at each one. Labels the text does not depend
on (dialogue acts, news sentence roles, story stages) were not recovered by any form and are not used as tests.
Related section types whose words overlap (symptom, mechanism, prognosis) are confused a whole section at a time.

## Cost

On an RTX 3060 (12 GB) with H2O-Lightning-4B, every token read: about 111 ms per token with 4 states and
210 ms with 10 in the catalogue form (one stream token pass plus one batched pass of K short branches).

## Readouts tried before

All with numbers and commands in `docs/LOG.md` and code in `src/head_of_state/` and `experiments/`:
a multiple-choice question about the most recent text whose option-letter probabilities give P(state | text
so far) (`reader.StreamStateReader`, `h2o.H2OReader`); one stream copy per state opened by that state's
description, scored by next-token likelihood, whose effect faded within about 32 tokens (`readouts.py`,
`window_likelihood.py`); embedding similarity; activation directions. Decoders tried on the question readings
(persist rule, fixed-lag Viterbi, estimated weights and transitions) are there too.
