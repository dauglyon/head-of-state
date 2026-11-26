| Run | encounters (tokens) | decoder | token acc. | boundary F1 ±8 | ±16 | matched | mean delay | spurious /1k | Brier | ECE | ms/token |
|---|---|---|---|---|---|---|---|---|---|---|---|
| frozen: every 8th token | 10 (11921) | persist 1 | 0.842 | 0.51 | 0.54 | 190/545 | 5.9 | 1.8 | 0.225 | 0.041 | 13 |
| frozen: every 8th token | 10 (11921) | forward filter η=1 | 0.839 | 0.50 | 0.52 | 188/545 | 6.5 | 1.5 | 0.227 | 0.041 | 13 |
| frozen: every 8th token | 3 (4700) | persist 1 | 0.862 | 0.46 | 0.50 | 61/194 | 5.4 | 1.3 | 0.205 | 0.032 | 13 |
| frozen: every 8th token | 3 (4700) | forward filter η=1 | 0.860 | 0.46 | 0.49 | 60/194 | 6.4 | 1.1 | 0.205 | 0.034 | 13 |
| every token (not frozen) | 3 (4700) | persist 3 | 0.835 | 0.48 | 0.56 | 76/194 | 5.3 | 1.5 | 0.187 | 0.045 | 96 |
| every token (not frozen) | 3 (4700) | forward filter η=1 | 0.846 | 0.51 | 0.57 | 77/194 | 4.8 | 1.3 | 0.248 | 0.108 | 96 |
