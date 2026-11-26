# Method and decoder notes

Terms: a *regime* is one of the states described at run time (name plus one-line description); a
*regime switch* is a change from one regime to another; the *head* is the model's existing output
layer. The method is HMM-like: per-token regime scores are combined with a sticky transition model,
but the scores are read from a frozen language model rather than learned emission tables.

## Reading a regime from the head

`src/head_of_state/reader.py`, `StreamStateReader`.

1. The opening instruction and the stream so far are held in the model's key/value cache; each
   stream token is processed once.
2. At a reading point, a question listing the regimes as lettered options is appended as a temporary
   branch, ending with an `Answer:` prefill. The head's probabilities for the option letters, softmaxed
   over the K letters, are the reading.
3. Option orderings: the question can be asked with the options rotated, and the log-probabilities
   averaged per regime, to cancel position bias (`rotations`). On the samples measured so far a
   second ordering did not improve accuracy.
4. The branch is removed from the cache. Attention layers drop the branch's keys and values. Gated
   DeltaNet layers (Qwen3.5) keep one running state that the branch overwrites in place; transformers
   cannot crop it, so those states are copied before the branch and restored after it.
5. All orderings run as one batched pass over a cache widened to one row per ordering; each layer's
   attributes are restored afterwards.
6. With `stride` N, the N stream tokens since the last reading are fed in one pass just before the
   next reading. A one-token pass and an N-token pass cost about the same (both are dominated by
   reading the weights), so the cost per token falls almost N-fold. Every token is still labelled:
   a reading labels the tokens since the previous reading (`Reading.start` to `Reading.pos`), and
   `finish()` takes a last reading for tokens left at the end of the stream. A token's label therefore
   arrives up to N − 1 tokens after the token. `persist` counts readings, so it shrinks with the
   stride: every 8th token with `persist` 1 matched every token with `persist` 3 in tests.

Cost on an RTX 3060 (12 GB) with Qwen3.5-4B: one stream token 31 ms; one question pass about 66 ms at
113 tokens, 69 ms at 128 tokens, 96 ms at 129–130 tokens (a matrix-tiling step: keep questions at or
under 128 tokens). A reading at every token with 2 orderings costs 150 ms per token; a reading every
8th token with 1 ordering costs 14 ms (5 regimes) to 20 ms (10 regimes) per token.

## Decoders

`src/head_of_state/reader.py` (online) and `src/head_of_state/decoders.py` (offline replay of saved readings).

- **Raw argmax**: the likeliest regime at each reading.
- **Persist rule**: a new regime is committed after it has been likeliest `persist` readings in a row.
- **Sticky forward filter**: b_t(k) ∝ p_t(k)^η · Σ_j b_{t−1}(j)·A_jk, with A_kk = 1 − 1/L and the
  remaining mass spread evenly; L is the declared typical segment length.
- **Fixed-lag Viterbi** (`FixedLagViterbi`): max-product decoding with the same A and emissions
  η·log p_t(k); at token t the best path is traced back and the label for token t − L is emitted
  (L = full gives offline Viterbi). It returns a path, not probabilities; the forward filter's
  per-token probabilities remain the probability output.
- Fixed-lag forward-backward smoothing and a surprisal-modulated switch probability were also built in
  `decoders.py`; neither improved on the decoders above in the tests so far.

## How switches are scored

A true regime switch is found only by a decoder switch event into the new regime; staying in a regime
the stream returns to earns no credit, and every unmatched decoder switch is spurious. Two timing
measures are kept apart:
- **placement error**: distance from the true switch to where the final labels put the switch;
- **decision latency**: tokens after the true switch at which the decoder first emits the new regime
  (placement plus the decoder's lag).
For the persist rule and the forward filter the two are equal; fixed-lag decoders trade latency for
placement.

## Known limits

- **Detection lag.** On PubMed abstracts the raw readings flip about 16 tokens after the true switch,
  and the persist rule commits about 23 tokens after it. Showing the question only the last 30–120
  tokens does not shorten the lag, so it does not come from earlier history: a new section stays
  ambiguous to the question until enough of it has arrived. Decoders cannot place a switch before
  the readings turn; activation directions distilled from the asked readings turned about 8 tokens
  after the switch in a small test.
- **Many similar regimes.** With 10 disease-article section types, token accuracy is 0.56–0.58 and
  every decoder setting tried leaves more than 2 spurious switches per 1,000 tokens.
- **Sample sizes.** All results so far come from 2 to 12 streams and 15 to 36 switches per set;
  differences of a few points are within noise.

## Alternative readouts tested

Details and numbers are in `docs/LOG.md`; code in `src/head_of_state/readouts.py` and
`src/head_of_state/window_likelihood.py`.

- **Question before the stream** (options cached once, a short answer prompt per reading): less
  accurate (0.51 vs 0.84 on PubMed) and not much cheaper.
- **Next-token likelihood** (one stream copy per regime, opened by that regime's description; the score
  is the log-probability of each actual token): the description's effect on the next token fades
  within about 32 tokens; with the description at the start only, accuracy is near chance.
- **Hopped-window likelihood** (each copy rebuilt every W/2 tokens from description + last W tokens):
  raises the likelihood readout (to 0.59 on PubMed at W = 2) but stays below the question readout.
- **Embedding similarity** (bge-m3, window vs regime description): 1–6 ms per token, accuracy 0.55
  (PubMed) and 0.26 (WikiSection).
- **Content-free calibration** of the question readout: lowered accuracy by 4 to 17 points.
- **Windowed asking** (the question sees only the last W tokens): no shorter lag, lower accuracy.
- **Activation directions** (residual stream at layer 24 of 32, difference of class means, read from
  the stream's own pass at 0.5 ms per token): from model-written passages, 0.74 token accuracy with a
  long delay; distilled from asked readings on other streams, 0.82 accuracy with an 8-token delay
  (5 held-out abstracts). The distilled directions are fitted per regime set, without gold labels.
