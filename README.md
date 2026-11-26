<p align="center"><img src="docs/assets/logo/head-of-state-logo.svg" width="120" alt="Head of State logo"></p>

# Head of State

Zero-shot regime tracking from an open-weights LLM's output head.
Hidden-state inference from plain-English descriptions.

Head of State labels every position of a token stream with one of several regimes, each described in one
line of plain English at run time, and marks the regime switches. A regime and a hidden state are the
same thing here. The stream sits in an open-weights causal language model's key/value cache; at each
reading point a short multiple-choice question is appended, the model's existing output head gives a
probability for each regime, and the question is removed from the cache again. An HMM-like decoder
(persist rule, sticky forward filter or fixed-lag Viterbi) turns the readings into regime labels. No
weights change and nothing is fitted per problem.

**Status: early prototype.** All numbers are preliminary, from small samples, with Qwen3.5-4B on one 12 GB GPU.

| Data (sample) | Result |
|---|---|
| Built-in support-call demo (245 tokens, 4 switches) | token accuracy 0.84; 4 of 4 switches found, 10 tokens late on average; no spurious switches |
| PubMed 20k RCT dev (10 abstracts, 36 switches, 5 regimes) | token accuracy 0.79–0.80; switches detected about 23 tokens late with the persist rule (a median sentence is 30 tokens); 5.8 spurious switches per 1,000 tokens, 1–2 with fixed-lag Viterbi |
| WikiSection en_disease test (12 streams, one symptom→treatment switch each, 10 regimes offered) | token accuracy 0.88 around the switch; switches detected about 24 tokens late, with no growth over 50 to 3,000 tokens of earlier text (3 streams per length) |
| WikiSection en_disease validation (2 streams, 15 switches, 10 regimes) | token accuracy 0.56–0.58; no decoder setting kept spurious switches under 2 per 1,000 tokens |
| Cost per token | 150 ms reading every token with 2 option orderings (default); 14–20 ms reading every 8th token with 1 ordering, which on the demo and PubMed sets matched the default's accuracy |

Known limits: switches are detected late (raw readings turn about 16 tokens after a true switch on
PubMed abstracts); showing the question only the last 30–120 tokens does not shorten this, so the lag
is not caused by earlier history. Readouts built on next-token likelihood have been weaker throughout.
A cheaper sensor, activation directions distilled from the asked readings, scored 0.82 token accuracy
with an 8-token delay at 0.5 ms per token on 5 held-out PubMed abstracts (preliminary; see `docs/LOG.md`).

## Run the demo

```
uv sync
uv run head-of-state --demo
uv run head-of-state --demo --stride 8 --rotations 1 --persist 1   # read every 8th token: about 7x cheaper
uv run python tests/test_reader.py        # mechanics tests on tiny random models
```

Datasets for the experiments: `experiments/data/download.sh`. Method and decoder notes: `docs/method.md`;
the original brief: `docs/HANDOFF.md`; evaluation plan: `docs/TEST_SUITE.md`; planned experiments: `docs/EXPERIMENTS.md`; prior work:
`docs/research/prior-art.md`; every run, its command and its result: `docs/LOG.md`.
