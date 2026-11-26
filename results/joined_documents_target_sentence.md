Settled labels: `0.2 L` moves each reading 0.2 x the declared length earlier (the pre-registered rule); `none` uses the readings where they were taken.

| State list | streams (tokens) | reading shift | token acc. (settled) | always the most common state | boundary F1 ±16 | switches placed within 16 | spurious /1k | time to notice | ms/token |
|---|---|---|---|---|---|---|---|---|---|
| ag_news (4 states) | 3 (6184) | 0.2 L = 54 | 0.755 | 0.338 | 0.18 | 6/21 | 6.6 | 12.8 | 118 |
| ag_news (4 states) | 3 (6184) | none | 0.881 | 0.338 | 0.52 | 18/21 | 4.9 | 12.8 | 118 |
| wikisection (10 states) | 3 (6411) | 0.2 L = 88 | 0.539 | 0.271 | 0.10 | 5/8 | 14.2 | 69.8 | 158 |
| wikisection (10 states) | 3 (6411) | none | 0.628 | 0.271 | 0.15 | 8/8 | 14.0 | 69.8 | 158 |
| imdb (2 states) | 3 (6565) | 0.2 L = 55 | 0.890 | 0.507 | 0.19 | 4/19 | 2.9 | 45.9 | 99 |
| imdb (2 states) | 3 (6565) | none | 0.881 | 0.507 | 0.34 | 7/19 | 2.3 | 45.9 | 99 |

Time to notice by the length of the segment before the switch (median tokens; never = not shown before the next switch)

| State list | previous segment | switches | median time to notice | never |
|---|---|---|---|---|
| ag_news | 0–128 tokens | 7 | 7 | 0 |
| ag_news | 128–256 tokens | 5 | 2 | 0 |
| ag_news | 256–512 tokens | 4 | 20 | 0 |
| ag_news | 512– tokens | 5 | 4 | 0 |
| wikisection | 256–512 tokens | 4 | 10 | 0 |
| wikisection | 512– tokens | 4 | 16 | 0 |
| imdb | 128–256 tokens | 12 | 37 | 0 |
| imdb | 256–512 tokens | 3 | 46 | 0 |
| imdb | 512– tokens | 4 | 24 | 0 |
