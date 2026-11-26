| Set | reading, decoder | token acc. | boundary F1 ±2 | ±8 | ±16 | matched | mean delay | spurious /1k | readings /1k tokens | ms/token |
|---|---|---|---|---|---|---|---|---|---|---|
| S3test (30 streams, 13327 tokens) | every 8th token, persist 1 | 0.763 | 0.22 | 0.42 | 0.52 | 85/98 | 22.8 | 7.2 | 124 | 13 |
| S3test (30 streams, 13327 tokens) | every 8th token, Viterbi, lag 1 reading | 0.750 | 0.17 | 0.39 | 0.51 | 82/98 | 27.6 (+8 decision) | 3.4 | 124 | 13 |
| S3test (30 streams, 13327 tokens) | every 8th token, Viterbi, offline | 0.769 | 0.22 | 0.48 | 0.61 | 79/98 | 24.9 | 2.1 | 124 | 13 |
| S3test (30 streams, 13327 tokens) | every 8th token + bisection, persist 1 | 0.755 | 0.20 | 0.35 | 0.44 | 86/98 | 20.1 | 8.5 | 165 | 18 |
| S3test (30 streams, 13327 tokens) | every 8th token + bisection, Viterbi, lag 1 reading | 0.746 | 0.19 | 0.38 | 0.49 | 83/98 | 23.1 (+8 decision) | 4.7 | 165 | 18 |
| S3test (30 streams, 13327 tokens) | every 8th token + bisection, Viterbi, offline | 0.759 | 0.24 | 0.44 | 0.55 | 84/98 | 20.5 | 2.9 | 165 | 18 |
| ACI (10 streams, 11921 tokens) | every 8th token, persist 1 | 0.842 | 0.24 | 0.51 | 0.54 | 181/378 | 6.4 | 2.6 | 125 | 13 |
| ACI (10 streams, 11921 tokens) | every 8th token, Viterbi, lag 1 reading | 0.803 | 0.35 | 0.66 | 0.71 | 258/378 | 7.8 (+8 decision) | 17.4 | 125 | 13 |
| ACI (10 streams, 11921 tokens) | every 8th token, Viterbi, offline | 0.805 | 0.36 | 0.66 | 0.71 | 259/378 | 7.3 | 20.6 | 125 | 13 |
| ACI (10 streams, 11921 tokens) | every 8th token + bisection, persist 1 | 0.855 | 0.39 | 0.50 | 0.54 | 192/408 | 3.6 | 1.7 | 178 | 18 |
| ACI (10 streams, 11921 tokens) | every 8th token + bisection, Viterbi, lag 1 reading | 0.807 | 0.38 | 0.53 | 0.57 | 275/408 | 5.6 (+8 decision) | 39.1 | 178 | 18 |
| ACI (10 streams, 11921 tokens) | every 8th token + bisection, Viterbi, offline | 0.814 | 0.38 | 0.50 | 0.54 | 283/408 | 5.0 | 48.5 | 178 | 18 |
