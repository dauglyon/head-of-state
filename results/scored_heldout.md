| Set | decoder | token acc. | boundary F1 ±8 | ±16 | switches found | mean delay | spurious /1k | Brier | ECE | ms/token |
|---|---|---|---|---|---|---|---|---|---|---|
| PubMed 20k RCT test, abstracts 31-60 (30 streams, 13882 tokens) | forward filter η=1 (headline) | 0.750 | 0.28 | 0.40 | 77/95 | 22.5 | 10.1 | 0.466 | 0.225 | 101 |
| PubMed 20k RCT test, abstracts 31-60 (30 streams, 13882 tokens) | persist 3 | 0.753 | 0.31 | 0.39 | 81/95 | 20.8 | 11.8 | 0.374 | 0.041 | 101 |
| PubMed 20k RCT test, abstracts 31-60 (30 streams, 13882 tokens) | forward filter, length ×0.25 | 0.755 | 0.30 | 0.39 | 79/95 | 21.0 | 11.6 | 0.442 | 0.205 | 101 |
| PubMed 20k RCT test, abstracts 31-60 (30 streams, 13882 tokens) | forward filter, length ×4 | 0.746 | 0.27 | 0.40 | 79/95 | 22.9 | 8.9 | 0.481 | 0.234 | 101 |
| ACI-Bench test, encounters 11-20 (10 streams, 13534 tokens) | forward filter η=1 (headline) | 0.841 | 0.59 | 0.64 | 233/476 | 4.1 | 3.8 | 0.261 | 0.106 | 96 |
| ACI-Bench test, encounters 11-20 (10 streams, 13534 tokens) | persist 3 | 0.830 | 0.60 | 0.66 | 235/476 | 4.6 | 3.1 | 0.200 | 0.037 | 96 |
| ACI-Bench test, encounters 11-20 (10 streams, 13534 tokens) | forward filter, length ×0.25 | 0.861 | 0.61 | 0.64 | 268/476 | 2.6 | 11.7 | 0.205 | 0.026 | 96 |
| ACI-Bench test, encounters 11-20 (10 streams, 13534 tokens) | forward filter, length ×4 | 0.819 | 0.53 | 0.60 | 214/476 | 5.5 | 1.6 | 0.315 | 0.149 | 96 |
| Null streams, seed 1 (2 streams, 6550 tokens) | forward filter η=1 (headline) | 0.464 | 0.00 | 0.00 | 0/0 | nan | 11.9 | 1.038 | 0.514 | 158 |
| Null streams, seed 1 (2 streams, 6550 tokens) | persist 3 | 0.476 | 0.00 | 0.00 | 0/0 | nan | 18.9 | 0.799 | 0.251 | 158 |
| Null streams, seed 1 (2 streams, 6550 tokens) | forward filter, length ×0.25 | 0.466 | 0.00 | 0.00 | 0/0 | nan | 13.3 | 1.028 | 0.506 | 158 |
| Null streams, seed 1 (2 streams, 6550 tokens) | forward filter, length ×4 | 0.466 | 0.00 | 0.00 | 0/0 | nan | 10.2 | 1.045 | 0.518 | 158 |
