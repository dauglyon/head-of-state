Settled labels: `0.2 L` moves each reading 0.2 x the declared length earlier (the pre-registered rule); `none` uses the readings where they were taken.

| State list | streams (tokens) | reading shift | token acc. (settled) | always the most common state | boundary F1 ±16 | switches placed within 16 | spurious /1k | time to notice | ms/token |
|---|---|---|---|---|---|---|---|---|---|
| ag_news (4 states) | 3 (6776) | 0.2 L = 54 | 0.832 | 0.339 | 0.14 | 3/18 | 3.2 | 30.3 | 104 |
| ag_news (4 states) | 3 (6776) | none | 0.874 | 0.339 | 0.45 | 10/18 | 2.4 | 30.3 | 104 |
| wikisection (10 states) | 3 (8646) | 0.2 L = 88 | 0.642 | 0.296 | 0.07 | 2/11 | 5.6 | 92.0 | 157 |
| wikisection (10 states) | 3 (8646) | none | 0.655 | 0.296 | 0.16 | 5/11 | 5.3 | 92.0 | 157 |
| imdb (2 states) | 3 (6599) | 0.2 L = 55 | 0.811 | 0.665 | 0.06 | 1/15 | 3.0 | 54.5 | 103 |
| imdb (2 states) | 3 (6599) | none | 0.809 | 0.665 | 0.28 | 5/15 | 2.4 | 54.5 | 103 |

Time to notice by the length of the segment before the switch (median tokens; never = not shown before the next switch)

| State list | previous segment | switches | median time to notice | never |
|---|---|---|---|---|
| ag_news | 0–128 tokens | 2 | 45 | 0 |
| ag_news | 128–256 tokens | 8 | 24 | 1 |
| ag_news | 256–512 tokens | 2 | 8 | 0 |
| ag_news | 512– tokens | 6 | 6 | 0 |
| wikisection | 0–128 tokens | 2 | 112 | 0 |
| wikisection | 256–512 tokens | 4 | 43 | 1 |
| wikisection | 512– tokens | 5 | 99 | 0 |
| imdb | 0–128 tokens | 1 | 33 | 0 |
| imdb | 128–256 tokens | 7 | 25 | 0 |
| imdb | 256–512 tokens | 2 | 0 | 1 |
| imdb | 512– tokens | 5 | 18 | 1 |
