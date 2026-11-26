Settled labels: `0.2 L` moves each reading 0.2 x the declared length earlier (the pre-registered rule); `none` uses the readings where they were taken.

| State list | streams (tokens) | reading shift | token acc. (settled) | always the most common state | boundary F1 ±16 | switches placed within 16 | spurious /1k | time to notice | ms/token |
|---|---|---|---|---|---|---|---|---|---|
| ag_news (4 states) | 3 (6184) | 0.2 L = 54 | 0.779 | 0.338 | 0.21 | 6/21 | 4.7 | 32.7 | 104 |
| ag_news (4 states) | 3 (6184) | none | 0.865 | 0.338 | 0.55 | 15/21 | 3.1 | 32.7 | 104 |
| wikisection (10 states) | 3 (6411) | 0.2 L = 88 | 0.616 | 0.271 | 0.15 | 3/8 | 4.7 | 90.4 | 156 |
| wikisection (10 states) | 3 (6411) | none | 0.683 | 0.271 | 0.19 | 4/8 | 4.8 | 90.4 | 156 |
| imdb (2 states) | 3 (6565) | 0.2 L = 55 | 0.890 | 0.507 | 0.15 | 3/19 | 2.7 | 48.5 | 103 |
| imdb (2 states) | 3 (6565) | none | 0.889 | 0.507 | 0.35 | 7/19 | 2.1 | 48.5 | 103 |

Time to notice by the length of the segment before the switch (median tokens; never = not shown before the next switch)

| State list | previous segment | switches | median time to notice | never |
|---|---|---|---|---|
| ag_news | 0–128 tokens | 7 | 15 | 0 |
| ag_news | 128–256 tokens | 5 | 4 | 0 |
| ag_news | 256–512 tokens | 4 | 38 | 0 |
| ag_news | 512– tokens | 5 | 12 | 0 |
| wikisection | 256–512 tokens | 4 | 22 | 0 |
| wikisection | 512– tokens | 4 | 10 | 1 |
| imdb | 128–256 tokens | 12 | 33 | 0 |
| imdb | 256–512 tokens | 3 | 25 | 0 |
| imdb | 512– tokens | 4 | 52 | 0 |
