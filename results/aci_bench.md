| Run | encounters (tokens) | decoder | token acc. | boundary F1 ±8 | ±16 | matched | mean delay | spurious /1k | Brier | ECE | ms/token |
|---|---|---|---|---|---|---|---|---|---|---|---|
| frozen: every 8th token | 10 (11921) | persist 1 | 0.842 | 0.51 | 0.54 | 181/378 | 6.4 | 2.6 | 0.225 | 0.041 | 13 |
| frozen: every 8th token | 10 (11921) | forward filter η=1 | 0.843 | 0.57 | 0.60 | 199/378 | 6.4 | 4.8 | 0.237 | 0.045 | 13 |
| frozen: every 8th token | 3 (4700) | persist 1 | 0.862 | 0.46 | 0.50 | 58/136 | 6.4 | 1.9 | 0.205 | 0.032 | 13 |
| frozen: every 8th token | 3 (4700) | forward filter η=1 | 0.862 | 0.57 | 0.60 | 71/136 | 7.2 | 4.3 | 0.219 | 0.060 | 13 |
| every token (not frozen) | 3 (4700) | persist 3 | 0.835 | 0.48 | 0.56 | 76/194 | 5.3 | 1.5 | 0.187 | 0.045 | 96 |
| every token (not frozen) | 3 (4700) | forward filter η=1 | 0.846 | 0.51 | 0.57 | 77/194 | 4.8 | 1.3 | 0.248 | 0.108 | 96 |
