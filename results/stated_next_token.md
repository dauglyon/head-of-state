Tuning streams (seed 0), first stream per list, every token. Emission: P(actual next token | text so far, state stated on a temporary branch).

| Model, branch form | State list | tokens | D, mean over state pairs (nats/token) | weakest pair | SD per token | lag-1 autocorr. | calibration slope b | state offsets, range | shared distortion | ms/token |
|---|---|---|---|---|---|---|---|---|---|---|
| h2o, instruction | ag_news | 2101 | 0.751 | science and technology vs world: 0.201 | 0.89 | 0.28 | 2.56 | -0.54 to +0.05 | -5.51 | 135 |
| h2o, instruction | wikisection | 2302 | 0.849 | mechanism vs prognosis: 0.138 | 0.98 | 0.36 | 2.29 | -0.58 to +0.49 | -5.28 | 279 |
| h2o, instruction | imdb | 2102 | 0.109 | positive vs negative: -0.012 | 0.48 | 0.25 | 0.68 | -0.11 to +0.01 | -3.99 | 104 |
| h2o, bracket | ag_news | 2101 | 0.449 | business vs world: 0.147 | 0.87 | 0.23 | 1.30 | -0.24 to -0.03 | -3.66 | 103 |
| h2o, bracket | wikisection | 2302 | 0.176 | history vs society and research: -0.198 | 1.00 | 0.21 | 0.32 | -0.28 to +0.16 | -3.26 | 175 |
| h2o, bracket | imdb | 2102 | 0.045 | negative vs positive: 0.040 | 0.54 | 0.21 | 0.32 | -0.03 to -0.02 | -2.75 | 78 |
| qwen, instruction | ag_news | 2101 | 0.629 | business vs world: -0.363 | 1.66 | 0.05 | 0.77 | -0.43 to +0.13 | -6.38 | 135 |
| qwen, instruction | wikisection | 2302 | 0.991 | history vs society and research: -0.218 | 1.91 | 0.18 | 0.59 | -0.82 to +0.48 | -6.32 | 279 |
| qwen, instruction | imdb | 2102 | 0.223 | positive vs negative: -0.191 | 1.00 | 0.27 | 0.22 | -0.32 to +0.10 | -5.28 | 104 |

Decoded, weight 1, declared length. Cells: token accuracy · switches placed within 16 tokens · false switches per 1,000 tokens · time to notice (tokens).

| Model, branch form | State list | state-conditioned next token | Jev-style readout, same stream |
|---|---|---|---|
| h2o, instruction | ag_news | 0.916 · 4/7 · 1.4 · 43 | 0.887 · 4/7 · 2.9 · 32 |
| h2o, instruction | wikisection | 0.785 · 2/3 · 2.2 · 39 | 0.596 · 2/3 · 5.6 · 25 |
| h2o, instruction | imdb | 0.591 · 2/8 · 1.9 · 163 | 0.827 · 3/8 · 2.4 · 58 |
| h2o, bracket | ag_news | 0.922 · 4/7 · 2.4 · 40 | 0.887 · 4/7 · 2.9 · 32 |
| h2o, bracket | wikisection | 0.169 · 1/3 · 7.8 · 150 | 0.596 · 2/3 · 5.6 · 25 |
| h2o, bracket | imdb | 0.713 · 4/8 · 0.5 · 50 | 0.827 · 3/8 · 2.4 · 58 |
| qwen, instruction | ag_news | 0.858 · 4/7 · 1.4 · 79 | 0.887 · 4/7 · 2.9 · 32 |
| qwen, instruction | wikisection | 0.520 · 1/3 · 7.8 · 58 | 0.596 · 2/3 · 5.6 · 25 |
| qwen, instruction | imdb | 0.523 · 1/8 · 1.4 · 113 | 0.827 · 3/8 · 2.4 · 58 |
