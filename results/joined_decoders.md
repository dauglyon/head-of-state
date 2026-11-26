Seed-2 streams, a reading at every token.

| State list | decoder | token acc. | placed within 16 | false /1k |
|---|---|---|---|---|
| ag_news | raw argmax | 0.858 | 14/18 | 27.9 |
| ag_news | persist 3 | 0.858 | 11/18 | 6.6 |
| ag_news | forward filter (live) | 0.859 | 10/18 | 3.4 |
| ag_news | Viterbi (settled) | 0.874 | 10/18 | 2.4 |
| wikisection | raw argmax | 0.630 | 8/11 | 91.1 |
| wikisection | persist 3 | 0.634 | 7/11 | 24.1 |
| wikisection | forward filter (live) | 0.628 | 5/11 | 10.5 |
| wikisection | Viterbi (settled) | 0.655 | 5/11 | 5.3 |
| imdb | raw argmax | 0.800 | 7/15 | 33.2 |
| imdb | persist 3 | 0.799 | 5/15 | 8.2 |
| imdb | forward filter (live) | 0.803 | 4/15 | 4.8 |
| imdb | Viterbi (settled) | 0.809 | 5/15 | 2.4 |
