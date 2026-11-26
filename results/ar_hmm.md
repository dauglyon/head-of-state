A reading at every token. AR-HMM fitted per stream with no labels. Kept names: states whose learned long-run reading still favours their own option.

| Set | settled labels | token acc. | always the most common state | placed within 16 | false /1k | stay per token | carry-over phi | kept names |
|---|---|---|---|---|---|---|---|---|
| ag_news (seed 0) | current: Viterbi, declared length | 0.865 | 0.338 | 15/21 | 3.1 | — | — | — |
| ag_news (seed 0) | AR-HMM | 0.839 | 0.338 | 21/21 | 92.3 | 0.868 | 0.89 | 4/4, 4/4, 4/4 |
| wikisection (seed 0) | current: Viterbi, declared length | 0.683 | 0.271 | 4/8 | 4.8 | — | — | — |
| wikisection (seed 0) | AR-HMM | 0.625 | 0.271 | 8/8 | 247.7 | 0.384 | 0.90 | 9/10, 10/10, 10/10 |
| imdb (seed 0) | current: Viterbi, declared length | 0.889 | 0.507 | 7/19 | 2.1 | — | — | — |
| imdb (seed 0) | AR-HMM | 0.836 | 0.507 | 17/19 | 85.5 | 0.905 | 0.84 | 2/2, 2/2, 2/2 |
| ag_news (seed 2) | current: Viterbi, declared length | 0.874 | 0.339 | 10/18 | 2.4 | — | — | — |
| ag_news (seed 2) | AR-HMM | 0.861 | 0.339 | 18/18 | 75.0 | 0.815 | 0.91 | 4/4, 4/4, 4/4 |
| wikisection (seed 2) | current: Viterbi, declared length | 0.655 | 0.296 | 5/11 | 5.3 | — | — | — |
| wikisection (seed 2) | AR-HMM | 0.626 | 0.296 | 11/11 | 245.1 | 0.465 | 0.90 | 9/10, 8/10, 8/10 |
| imdb (seed 2) | current: Viterbi, declared length | 0.809 | 0.665 | 5/15 | 2.4 | — | — | — |
| imdb (seed 2) | AR-HMM | 0.757 | 0.665 | 12/15 | 124.0 | 0.848 | 0.85 | 2/2, 2/2, 2/2 |
| PubMed abstracts 31-60 (held out) | current: Viterbi, declared length | 0.776 | 0.411 | 63/95 | 4.9 | — | — | — |
| PubMed abstracts 31-60 (held out) | AR-HMM | 0.752 | 0.411 | 92/95 | 128.2 | 0.706 | 0.79 | 5/5, 5/5, 4/5, 5/5, 5/5, 4/5, 4/5, 4/5, 5/5, 4/5, 5/5, 5/5, 4/5, 4/5, 5/5, 5/5, 5/5, 5/5, 5/5, 5/5, 3/5, 4/5, 5/5, 5/5, 5/5, 5/5, 4/5, 5/5, 5/5, 5/5 |
| single-state streams (seed 1) | current: Viterbi, declared length | 0.468 | 0.537 | — | 7.2 | — | — | — |
| single-state streams (seed 1) | AR-HMM | 0.357 | 0.537 | — | 158.8 | 0.389 | 0.92 | 7/10, 8/10 |
