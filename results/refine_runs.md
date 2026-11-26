| Set | reading, decoder | token acc. | boundary F1 ±2 | ±8 | ±16 | matched | mean delay | spurious /1k | readings /1k tokens | ms/token |
|---|---|---|---|---|---|---|---|---|---|---|
| S3test (30 streams, 13327 tokens) | every 8th token, persist 1 | 0.763 | 0.22 | 0.42 | 0.52 | 85/98 | 22.8 | 7.2 | 124 | 13 |
| S3test (30 streams, 13327 tokens) | every 8th token, Viterbi, lag 1 reading | 0.750 | 0.16 | 0.39 | 0.51 | 82/98 | 28.0 (+8 decision) | 3.2 | 124 | 13 |
| S3test (30 streams, 13327 tokens) | every 8th token, Viterbi, offline | 0.770 | 0.22 | 0.48 | 0.61 | 79/98 | 24.8 | 2.1 | 124 | 13 |
| S3test (30 streams, 13327 tokens) | every 8th token + bisection, persist 1 | 0.755 | 0.20 | 0.35 | 0.44 | 86/98 | 23.4 | 8.5 | 165 | 18 |
| S3test (30 streams, 13327 tokens) | every 8th token + bisection, Viterbi, lag 1 reading | 0.744 | 0.18 | 0.38 | 0.48 | 83/98 | 26.0 (+8 decision) | 4.4 | 165 | 18 |
| S3test (30 streams, 13327 tokens) | every 8th token + bisection, Viterbi, offline | 0.759 | 0.24 | 0.44 | 0.55 | 84/98 | 23.9 | 2.9 | 165 | 18 |
| ACI (10 streams, 11921 tokens) | every 8th token, persist 1 | 0.842 | 0.24 | 0.51 | 0.54 | 190/545 | 5.9 | 1.8 | 125 | 13 |
| ACI (10 streams, 11921 tokens) | every 8th token, Viterbi, lag 1 reading | 0.843 | 0.19 | 0.42 | 0.44 | 146/545 | 6.2 (+8 decision) | 0.9 | 125 | 13 |
| ACI (10 streams, 11921 tokens) | every 8th token, Viterbi, offline | 0.841 | 0.17 | 0.42 | 0.43 | 146/545 | 6.1 | 0.8 | 125 | 13 |
| ACI (10 streams, 11921 tokens) | every 8th token + bisection, persist 1 | 0.855 | 0.39 | 0.50 | 0.54 | 192/545 | 6.0 | 1.7 | 178 | 18 |
| ACI (10 streams, 11921 tokens) | every 8th token + bisection, Viterbi, lag 1 reading | 0.854 | 0.38 | 0.49 | 0.53 | 188/545 | 6.0 (+8 decision) | 1.5 | 178 | 18 |
| ACI (10 streams, 11921 tokens) | every 8th token + bisection, Viterbi, offline | 0.854 | 0.37 | 0.48 | 0.51 | 183/545 | 6.0 | 1.3 | 178 | 18 |
