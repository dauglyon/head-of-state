| Run | encounters (tokens) | decoder | token acc. | boundary F1 ±8 | ±16 | matched | mean delay | spurious /1k | Brier | ECE | ms/token |
|---|---|---|---|---|---|---|---|---|---|---|---|
| frozen: every 8th token | 10 (11921) | persist 1 | 0.855 | 0.59 | 0.62 | 181/378 | 2.9 | 2.6 | 0.212 | 0.029 | 13 |
| frozen: every 8th token | 10 (11921) | forward filter η=1 | 0.862 | 0.64 | 0.67 | 199/378 | 3.0 | 4.8 | 0.207 | 0.062 | 13 |
| frozen: every 8th token | 3 (4700) | persist 1 | 0.875 | 0.55 | 0.56 | 58/136 | 2.9 | 1.9 | 0.191 | 0.039 | 13 |
| frozen: every 8th token | 3 (4700) | forward filter η=1 | 0.882 | 0.63 | 0.64 | 71/136 | 3.8 | 4.3 | 0.189 | 0.073 | 13 |
| every token (not frozen) | 3 (4700) | persist 3 | 0.835 | 0.48 | 0.56 | 76/194 | 5.3 | 1.5 | 0.187 | 0.045 | 96 |
| every token (not frozen) | 3 (4700) | forward filter η=1 | 0.846 | 0.51 | 0.57 | 77/194 | 4.8 | 1.3 | 0.248 | 0.108 | 96 |
