# Zero-shot hidden-state labelling of a token stream (handoff)

## What this is

An HMM-style classifier in which a frozen LLM stands in for trained emission and transition tables.

- **Input:** a token stream, and a list of named hidden states with one-line descriptions, supplied at run time.
- **Output:** for every position N, a probability for each state and the likeliest state, plus the positions where the state switches.
- **Constraints:** no weight changes, nothing fitted per problem, one reading per token (not per chunk).

## Mechanism

1. The stream so far is held in the model's key/value cache. It is read once and never re-read.
2. At position N, a short "which state is the most recent part of the stream in?" question is appended as a temporary branch.
3. The model's own output layer gives a probability for each option letter. Log-probabilities are averaged over several option orderings to cancel position bias.
4. The branch is cropped off; the cache is exactly as it was.
5. The reading at N is compared with the reading at N-1. A new state is committed only after it has been likeliest `persist` readings in a row.

## Files

- `stream_states.py`: the implementation (`StreamStateReader`), a scoring function, and a built-in demo (a five-state support call with known switch points).
- `test_stream_states.py`: mechanics tests on a tiny random-weight model.

## Status

Verified on a tiny random-weight GPT-2 (plumbing only):

- a reading through the cached branch equals a full recompute to within 3e-8
- the cache is restored exactly after every reading
- the stream keeps running past the context limit (re-anchoring)
- the commit rule and the scoring function give the expected results on a hand-made case

**Not yet done: it has never been run on a real language model.** There are no accuracy numbers. That is the first job.

## Your tasks

1. **Run the tests.** `pip install torch transformers tokenizers`, then `python test_stream_states.py`. It should print `all mechanics tests passed`.
2. **Run the demo on a real local model.**
   `python stream_states.py --model <hf-id-or-local-path> --demo`
   Use an instruction-tuned model of 4B parameters or more. The code needs Hugging Face `transformers` with direct access to the KV cache. A model served only through an API that hides the cache (Ollama, an OpenAI-compatible endpoint) will not work without a new backend.
3. **Check the prompt on that model.** Print `tok.decode(reader.head_ids)` and `tok.decode(reader.branches[0])` and confirm the chat template and the `Answer:` prefill look right. If the constructor raises "does not give single-token option letters", adjust `_letter_ids` for that tokenizer.
4. **Report the three scores** the demo prints: token accuracy (raw and committed), mean delay in tokens after each true switch, and spurious switches.
5. **Build a real evaluation.** The demo is one short hand-written stream and is only a smoke test. Build synthetic streams by concatenating documents with known labels (for example AG News for topic), so every token inherits its segment's label and the switch points are exact. Use at least three unrelated state lists with the same settings.
6. **Sweep the knobs** (below) and plot delay against spurious switches.
7. **Run the comparisons** listed under "Open questions".

## Knobs

| Flag | Default | Meaning |
|---|---|---|
| `--stride` | 1 | Read every Nth token. 1 is every token. Each reading costs one question (about 100 tokens), so 8 cuts cost eightfold. |
| `--rotations` | 2 | Option orderings averaged per reading. More reduces position bias and costs proportionally more. |
| `--persist` | 3 | Readings a new state must hold before the switch is committed. |
| `--prior-power` | 0.0 | 0.75 divides out a running label prior. Off by default because it hurt in published tests when one label dominates, which is normal inside a stream segment. |
| `max_ctx`, `keep_ctx` | 8192, 2048 | When the cache fills, it is rebuilt from the instruction plus the last `keep_ctx` stream tokens. Check that labels do not jump at that seam. |

Speed-ups worth doing if the demo works: batch the rotations in one forward pass, and pack several positions' branches into one pass with a block attention mask ("tree batching").

## Open questions to test

1. **Does the reading lag after a switch?** The model has the whole earlier stream in context. Published work reports that earlier content keeps influencing LLM output for roughly 100 to 200 tokens after a topic switch. Measure delay as a function of how long the previous segment was. If lag is large, try a smaller `keep_ctx` or rewording the question.
2. **Is the commit rule enough?** Compare three decoders on the same readings: raw argmax, the `persist` rule, and a sticky-transition forward filter `b_t(k) ∝ p_t(k)^η · Σ_j b_{t-1}(j)·A_jk` with `A` set from an expected segment length. No published source gives a value for η; sweep it once and freeze it across state lists.
3. **Generative-likelihood emissions as an alternative.** Run one copy of the stream per state, each prefixed with that state's description, and use the model's probability of each actual token as the emission. This is the textbook HMM form and needs no question branch. Untested risk: the prefix's influence may fade as the stream grows. First experiment: plot the per-token log-likelihood gap between the right and wrong prefixes against distance from the prefix.
4. **Detector directions as a cheap alternative.** Per-token cosine between the residual stream and the output vectors of neurons chosen by matching their stored descriptions to the state description. Needs a neuron-description database for the model (public for Llama-3.1-8B). Published document-level AUROC is about 0.90 against 0.97 for asking the model.

## Approaches already ruled out (do not rebuild these)

- **A small trained head on a frozen model, shared across problems.** It lost to the plain output-layer reading on unseen tasks in four independent tests (AnyJev report, Large Decision Models, Releasing the Inner Jev, and a public hobby project).
- **Reading the answer from a middle layer.** The answer forms late: in the AnyJev report six of eight models could skip no layers at a 98% agreement target.
- **A probe trained per task.** Not zero-shot.
- **Classifying chunks instead of tokens.** Not wanted; the requirement is a reading at every position.
- **Fine-tuning or LoRA.** It is what matches the commercial Jev model's accuracy, but it breaks the "weights untouched" requirement.

## Expectations from the literature

- A frozen 4B to 8B model's zero-shot option readout scores roughly 75 to 80% on unseen multi-way tasks, and order-averaging adds about 5 points (AnyJev: 0.747 to 0.803 on a 20-way task).
- Calibration is poor below about 4B parameters (reported ECE 0.278 at 0.6B against 0.057 at 4B).
- Temporal decoding mainly improves stability and switch timing, not per-position accuracy.

## Sources

- AnyJev technical report: https://arxiv.org/html/2610.00831v1
- LLM-as-Jev: https://arxiv.org/pdf/2610.02076
- Large Decision Models: https://arxiv.org/pdf/2610.15501
- Releasing the Inner Jev: https://arxiv.org/pdf/2610.17205
- Jev's Architecture Unmasked: https://archerhume.com/posts/jevs-architecture-unmasked/
- LLM plus Bayesian filter over named regimes ("Belief at Risk"): https://arxiv.org/html/2606.15473
- Two-state HMM filter over per-step scores (SBBT): https://arxiv.org/html/2605.27712v1
- Gen-Z generative zero-shot classification: https://arxiv.org/html/2311.07115v1
- Neurons as zero-shot detectors: https://alignment.anthropic.com/2026/neuron-basis-applications/
- Semantic inertia after topic switches: https://ijcai.org/proceedings/2026/88
- Streaming zero-shot classifiers with trained backbones: https://urchade.github.io/GLiNER/usage.html and https://arxiv.org/html/2609.02292v1
