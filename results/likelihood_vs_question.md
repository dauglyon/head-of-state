## C. S3 dev, 10 abstracts (declared length 30)
| Method | decoder | token acc. | matched | mean delay | spurious/1k | ms/token |
|---|---|---|---|---|---|---|
| likelihood, window W=2 | filter η=1 | 0.587 | 26/36 | 10.6 | 108.4 | 55 (hop-batched) |
| question, full history, 1 ordering(s), every token | persist 3 | 0.779 | 32/36 | 22.5 | 7.7 | 103 (measured separately) |
| question, full history, 1 ordering(s), every token | filter η=1 | 0.774 | 32/36 | 25.9 | 3.5 | 103 (measured separately) |
| question, full history, 2 ordering(s), every token | persist 3 | 0.784 | 34/36 | 22.2 | 5.0 | 150 (measured separately) |
| question, full history, 2 ordering(s), every token | filter η=1 | 0.775 | 33/36 | 25.5 | 4.8 | 150 (measured separately) |
| question, full history, every 4th token | persist 3 | 0.726 | 32/36 | 35.2 | 1.5 | — |
| question, window W=2, every 4th token (E) | persist 3 | 0.084 | 0/36 | — | 0.2 | — |
| question, full history, every 4th token | filter η=1 | 0.746 | 31/36 | 30.9 | 1.9 | — |
| question, window W=2, every 4th token (E) | filter η=1 | 0.086 | 3/36 | 65.7 | 1.2 | — |

## C/D/E. S1 lag streams, 12 (declared length 190.5); scored from 16 tokens before the switch to the end; question read every 2nd token there, likelihood every token
| Method | decoder | token acc. | matched | mean delay | spurious/1k | ms/token |
|---|---|---|---|---|---|---|
| likelihood, window W=12 | filter η=1 | 0.519 | 12/12 | 54.8 | 51.1 | — |
| question, full history | filter η=1 | 0.879 | 12/12 | 23.7 | 2.9 | — |
| question, full history | persist 3 | 0.867 | 12/12 | 22.7 | 5.9 | — |
| question, window W=12 (E) | filter η=1 | 0.581 | 12/12 | 14.8 | 40.0 | — |
| question, window W=12 (E) | persist 3 | 0.546 | 12/12 | 17.5 | 36.3 | — |

## D. Mean delay after the switch by previous-segment length (tokens; matched switches only; n = 3 streams each)
| Method | decoder | 50 | 200 | 800 | 3000 |
|---|---|---|---|---|---|
| likelihood, window W=12 | filter η=1 | 52 (3/3) | 74 (3/3) | 43 (3/3) | 50 (3/3) |
| question, full history | filter η=1 | 21 (3/3) | 7 (3/3) | 54 (3/3) | 13 (3/3) |
| question, full history | persist 3 | 21 (3/3) | 7 (3/3) | 53 (3/3) | 10 (3/3) |
| question, window W=12 (E) | filter η=1 | 15 (3/3) | 6 (3/3) | 29 (3/3) | 10 (3/3) |
| question, window W=12 (E) | persist 3 | 17 (3/3) | 7 (3/3) | 37 (3/3) | 9 (3/3) |
