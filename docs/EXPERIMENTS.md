# Needed experiments

Planned work that has not been run yet. Finished runs are in `docs/LOG.md`; the evaluation plan and
its pre-registration rules are in `docs/TEST_SUITE.md`.

## 1. Choosing the per-token evidence mechanism (owner's brief, 2026-11-25)

**Status (2026-11-25): run on PubMed dev and S1; results in `results/evidence_table.md`,
`results/windowed_asking.md`, `results/activation_directions.md` and `docs/LOG.md`.** Follow-ups the
results call for:
- Test the distilled directions (3b) where regimes recur (WikiSection validation streams) and on S1,
  and on more than 5 held-out abstracts.
- Reduce 3b's spurious switches (19 per 1k) with the decoder: persist, fixed-lag Viterbi, or combining
  3b with occasional asked readings as evidence rather than replacing one with the other.
- Decide whether fitting directions per regime set, without gold labels, is acceptable as zero-shot.

Question: which mechanism supplies the per-token regime evidence, and what role the others keep. Three
paths, in order: fix the question readout ("asking"), test likelihood emissions as a go/no-go, test
activation directions as a cheap layer. The brief replaces the earlier likelihood-only brief; this
version is updated against the runs already logged.

**Common setup**
- Streams: the 10 PubMed 20k RCT dev abstracts already saved with every ordering at every token
  (36 switches); resume the stopped dev run only if a comparison is too close to call. Then the 12 S1
  lag streams.
- Decoders: sticky forward filter with the declared segment length, and raw argmax.
- Metrics: token accuracy; boundary F1 within ±k tokens (k = 8 and 16); mean switch delay; spurious
  switches per 1k tokens; Brier score and ECE; cost per token in ms and in model passes. Computed
  offline for every saved run.
- Windows are declared as fractions of the declared segment length before scored runs (H3):
  W = 1×, 2×, 4× on PubMed (30, 60, 120 tokens) and 0.25×, 0.5×, 1× on S1 (48, 95, 190 tokens). The
  earlier W = 0.0625× declaration belongs to the likelihood path only.

**Path 1: fix asking**
- 1a. Windowed asking: the question sees only the last W tokens. Compare switch delay with
  full-history asking. Earlier runs at W = 2 (PubMed) and W = 12 (S1) lost 30 or more points of
  accuracy; the new windows hold one or more typical segments.
- 1b. Double counting (offline): the same asked probabilities through the forward filter raw, divided
  by a label prior estimated on unlabelled positions, and tempered with η in {0.25, 0.5, 1}. Scored
  on PubMed and on the WikiSection validation streams, where regimes recur (a prior correction helped
  on PubMed in the Viterbi replay and failed where regimes recur).
- 1c. S1 delay by previous-segment length, full history vs the 1a windows. Full-history delay already
  showed no growth with history (21 / 7 / 54 / 13 tokens for 50 / 200 / 800 / 3000).

**Path 2: likelihood emissions (gate)**
- Already run: the fade curve (advantage about zero beyond 32 tokens) and the hopped-window version
  (0.59 vs 0.78 accuracy on PubMed, 0.52 vs 0.88 on S1, 50–108 spurious per 1k).
- Remaining: how concentrated the per-token ratio is on a few tokens (offline). Then close the path
  as no-go unless that changes the picture; the reset-on-switch variant is not run.

**Path 3: activation directions as the cheap layer**
- 3a. Directions from model-written examples: about 20 snippets per regime written by the model from
  the description; mean activation per regime; difference-of-means directions; layer near 75% depth,
  with a few layers compared on dev only.
- 3b. Directions distilled from asking: activations labelled by the best Path 1 asking variant on a
  warm-up set of streams.
- 3c. Upper bound: a linear probe trained on real labels.
- 3d. Cascade: the cheap reading everywhere, asking only where the filter's posterior entropy exceeds a
  threshold; accuracy against the fraction of positions asked (simulated offline from the saved asked
  readings).
- Departure from `docs/HANDOFF.md`: 3a and 3b fit directions for each set of regimes (without gold
  labels), and the handoff excluded fitting per problem and per-task probes; the owner's brief
  overrides this for these experiments. 3c is a reference, not a candidate.

**Reference points**
- Sentence-embedding similarity to the regime descriptions.
- A supervised sentence classifier on PubMed 20k as a ceiling (figure to be checked against the
  published paper).
- The current prototype: option orderings and label-prior correction.

**Report**: one table of every mechanism and variant with all metrics on PubMed dev; the
delay-vs-window curve (1a); the fade curve (2a); the cascade curve (3d); a recommendation of which
mechanism is the per-token sensor and what role the others keep.

## 2. Earlier planned work not yet run

- Scored runs of `docs/TEST_SUITE.md`: PubMed test split; S1 re-anchor seam streams and the K sweep;
  S2 null streams; S4–S12. All runs so far used tuning splits or small exploratory samples.
- Frozen settings: option orderings, persist and η are still to be fixed (the reading interval is a
  parameter, decided 2026-11-25).
- Detector directions from published neuron descriptions (HANDOFF open question 4): out of scope; the
  public description database covers Llama-3.1-8B, not the model in use.
- Explicit-duration (semi-Markov) decoding: not needed so far; the Viterbi replay left no short
  spurious segments.
